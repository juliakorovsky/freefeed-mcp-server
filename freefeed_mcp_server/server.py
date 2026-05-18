"""FreeFeed MCP Server - provides FreeFeed API access via MCP protocol."""

import asyncio
import base64
import contextlib
import json
import logging
import os
import signal
import time
from mimetypes import guess_type
from typing import Any
from urllib.parse import urlparse

import httpx
import mcp.server.stdio
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import ImageContent, TextContent, Tool

from .client import FreeFeedAPIError, FreeFeedAuthError, FreeFeedClient
from .filters import slim_response

# Load environment variables
load_dotenv()


def _resolve_log_level() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


# Setup logging
logging.basicConfig(
    level=_resolve_log_level(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OPT_OUT_TAGS = ["#noai", "#opt-out-ai", "#no-bots", "#ai-free"]
FILTER_REASON = "User opted out of AI interactions"
DEFAULT_IMAGE_MAX_BYTES = 2_000_000
MCP_TOOL_SUCCESS_LOG = "MCP tool success: %s duration_ms=%.1f"
LIMIT_DESCRIPTION = "Number of posts to return"
OFFSET_DESCRIPTION = "Offset for pagination"


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _resolve_image_max_bytes() -> int:
    raw = os.getenv("FREEFEED_MCP_IMAGE_MAX_BYTES", str(DEFAULT_IMAGE_MAX_BYTES))
    try:
        value = int(raw)
        return max(256_000, value)
    except ValueError:
        return DEFAULT_IMAGE_MAX_BYTES


def _load_config_from_file(config: dict) -> None:
    """Load configuration from file if specified."""
    config_path = os.getenv("FREEFEED_OPTOUT_CONFIG")
    if not config_path:
        return

    try:
        with open(config_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return

        if isinstance(data.get("enabled"), bool):
            config["enabled"] = data["enabled"]
        if isinstance(data.get("manual_opt_out"), list):
            config["users"] = {
                str(u).strip() for u in data["manual_opt_out"] if str(u).strip()
            }
        if isinstance(data.get("tags"), list):
            config["tags"] = [str(t).strip() for t in data["tags"] if str(t).strip()]
        if isinstance(data.get("respect_private"), bool):
            config["respect_private"] = data["respect_private"]
        if isinstance(data.get("respect_paused"), bool):
            config["respect_paused"] = data["respect_paused"]
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read opt-out config %s: %s", config_path, exc)


def _load_config_from_env(config: dict) -> None:
    """Load configuration from environment variables."""
    enabled_env = _parse_bool(os.getenv("FREEFEED_OPTOUT_ENABLED"))
    if enabled_env is not None:
        config["enabled"] = enabled_env

    users_env = os.getenv("FREEFEED_OPTOUT_USERS")
    if users_env is not None:
        config["users"] = {u.strip() for u in users_env.split(",") if u.strip()}

    tags_env = os.getenv("FREEFEED_OPTOUT_TAGS")
    if tags_env is not None:
        config["tags"] = [t.strip() for t in tags_env.split(",") if t.strip()]

    respect_private_env = _parse_bool(os.getenv("FREEFEED_OPTOUT_RESPECT_PRIVATE"))
    if respect_private_env is not None:
        config["respect_private"] = respect_private_env

    respect_paused_env = _parse_bool(os.getenv("FREEFEED_OPTOUT_RESPECT_PAUSED"))
    if respect_paused_env is not None:
        config["respect_paused"] = respect_paused_env


def _load_opt_out_config() -> dict:
    config = {
        "enabled": False,
        "users": set(),
        "tags": list(DEFAULT_OPT_OUT_TAGS),
        "respect_private": True,
        "respect_paused": True,
    }

    _load_config_from_file(config)
    _load_config_from_env(config)

    return config


def _configure_server_logger() -> None:
    """Ensure server logs are also written to a file."""
    default_path = os.path.join(".", "logs", "freefeed_server.log")
    log_path = os.getenv("FREEFEED_SERVER_LOG_PATH", default_path).strip()
    if not log_path:
        return

    log_path = os.path.abspath(os.path.expanduser(log_path))
    log_dir = os.path.dirname(log_path)
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create log directory %s: %s", log_dir, exc)
            return

    for handler in logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and os.path.abspath(handler.baseFilename) == log_path
        ):
            return

    try:
        file_handler = logging.FileHandler(log_path)
    except OSError as exc:
        logger.warning("Could not open log file %s: %s", log_path, exc)
        return

    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(_resolve_log_level())
    logger.addHandler(file_handler)
    logger.setLevel(_resolve_log_level())


_configure_server_logger()

# Initialize MCP server
app = Server("freefeed-mcp-server")

# Global client instance
freefeed_client: FreeFeedClient | None = None


async def get_client() -> FreeFeedClient:
    """Get or create FreeFeed client instance."""
    global freefeed_client

    if freefeed_client is None:
        base_url = os.getenv("FREEFEED_BASE_URL", "https://freefeed.net")
        api_version_raw = os.getenv("FREEFEED_API_VERSION")
        app_token = os.getenv("FREEFEED_APP_TOKEN")
        username = os.getenv("FREEFEED_USERNAME")
        password = os.getenv("FREEFEED_PASSWORD")

        api_version: int | None = None
        if api_version_raw:
            try:
                api_version = int(api_version_raw)
            except ValueError:
                logger.warning(
                    "Invalid FREEFEED_API_VERSION=%s; using default", api_version_raw
                )

        if not app_token and (not username or not password):
            raise FreeFeedAuthError(
                "Set FREEFEED_APP_TOKEN or FREEFEED_USERNAME and FREEFEED_PASSWORD"
            )

        freefeed_client = FreeFeedClient(
            base_url=base_url,
            username=username,
            password=password,
            auth_token=app_token,
            api_version=api_version,
        )

        if app_token:
            logger.info("FreeFeed client initialized with application token")
        else:
            await freefeed_client.authenticate()
            logger.info("FreeFeed client initialized and authenticated")

    return freefeed_client


def _compact_user(user_data: dict) -> dict:
    fields = ["id", "username", "screenName", "type", "isPrivate", "isProtected"]
    return {key: user_data.get(key) for key in fields if key in user_data}


def _compact_whoami(payload: dict) -> dict:
    users = payload.get("users")
    subscriptions = payload.get("subscriptions")
    subscribers = payload.get("subscribers")

    compacted: dict = {}
    if isinstance(users, dict):
        compacted["users"] = _compact_user(users)

    def _compact_list(items: Any) -> list[dict]:
        if not isinstance(items, list):
            return []
        return [_compact_user(item) for item in items if isinstance(item, dict)]

    if subscriptions is not None:
        compacted["subscriptions"] = _compact_list(subscriptions)
    if subscribers is not None:
        compacted["subscribers"] = _compact_list(subscribers)

    compacted["summary"] = {
        "subscriptions": len(compacted.get("subscriptions", [])),
        "subscribers": len(compacted.get("subscribers", [])),
    }
    return compacted


def _build_post_user_map(payload: Any) -> dict[str, str]:
    """Build a map of user IDs to usernames from payload."""
    user_map: dict[str, str] = {}
    users = payload.get("users")
    if isinstance(users, list):
        for user in users:
            if isinstance(user, dict) and user.get("id") and user.get("username"):
                user_map[user["id"]] = user["username"]
    return user_map


def _apply_post_url(post: dict, base_url: str, user_map: dict[str, str]) -> None:
    """Apply post URL to a single post."""
    if not isinstance(post, dict):
        return
    author_id = post.get("createdBy")
    username = user_map.get(author_id)
    short_id = post.get("shortId")
    post_id = post.get("id")

    if username and short_id:
        post["postUrl"] = f"{base_url}/{username}/{short_id}"
    elif post_id:
        post["postUrl"] = f"{base_url}/posts/{post_id}"


def _add_post_urls(payload: Any, base_url: str) -> Any:
    if not isinstance(payload, dict):
        return payload

    user_map = _build_post_user_map(payload)
    posts = payload.get("posts")

    if isinstance(posts, list):
        for post in posts:
            _apply_post_url(post, base_url, user_map)
    elif isinstance(posts, dict):
        _apply_post_url(posts, base_url, user_map)

    return payload


def _extract_attachment_id(url: str) -> str | None:
    """Extract attachment ID from URL."""
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if "attachments" not in parts:
        return None
    idx = parts.index("attachments")
    candidate = parts[idx + 1] if idx + 1 < len(parts) else None
    if candidate and candidate.startswith("p") and idx + 2 < len(parts):
        candidate = parts[idx + 2]
    if not candidate:
        return None
    return candidate.split(".", 1)[0]


def _is_allowed_attachment_url(client: FreeFeedClient, url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    base_netloc = urlparse(client.base_url).netloc
    allowed_hosts = {base_netloc, "media.freefeed.net"}
    if parsed.netloc not in allowed_hosts:
        return False
    return _extract_attachment_id(url) is not None


def _get_fallback_urls(client: FreeFeedClient, attachment_url: str) -> list[str]:
    """Get list of fallback URLs to try for attachment."""
    attachment_id = _extract_attachment_id(attachment_url)
    if not attachment_id:
        return []

    urls: list[str] = []
    if _is_allowed_attachment_url(client, attachment_url):
        urls.append(attachment_url)

    fallback = f"{client.base_url}/attachments/{attachment_id}"
    if fallback not in urls:
        urls.append(fallback)
    media_fallback = f"https://media.freefeed.net/attachments/{attachment_id}"
    if media_fallback not in urls:
        urls.append(media_fallback)
    return urls


async def _fetch_attachment_binary(
    client: FreeFeedClient, url: str, max_bytes: int
) -> tuple[bytes | None, str | None, int | None, str | None]:
    """Fetch binary data from a single URL with size validation."""
    headers: dict[str, str] = {}
    if client.auth_token:
        headers["X-Authentication-Token"] = client.auth_token

    content_type: str | None = None
    content_length: int | None = None

    # Check headers first
    try:
        head = await client.client.head(url, headers=headers)
        if head.status_code == 404:
            return None, content_type, None, "not_found"
        if head.status_code < 400:
            content_type = head.headers.get("content-type")
            length_header = head.headers.get("content-length")
            if length_header and length_header.isdigit():
                content_length = int(length_header)
    except Exception as e:  # nosec: B110
        logger.debug("Failed to parse attachment headers: %s", e)

    if content_length is not None and content_length > max_bytes:
        return None, content_type, content_length, "too_large"

    # Fetch the actual data
    try:
        response = await client.client.get(url, headers=headers)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None, content_type, None, "not_found"
        return None, content_type, None, "http_error"

    if content_type is None:
        content_type = response.headers.get("content-type")
    data = response.content
    if len(data) > max_bytes:
        return None, content_type, len(data), "too_large"

    if content_type is None:
        content_type = guess_type(url)[0]

    return data, content_type, len(data), None


async def _try_html_preview(
    client: FreeFeedClient, url: str, max_bytes: int
) -> tuple[bytes | None, str | None, int | None, str | None] | None:
    """Try to get preview URL for HTML attachment."""
    attachment_id = _extract_attachment_id(url)
    if not attachment_id:
        return None

    try:
        preview = await client.get_attachment_preview_url(
            attachment_id, preview_type="original"
        )
        preview_url = None
        if isinstance(preview, dict):
            preview_url = preview.get("url")
        if preview_url:
            logger.info(
                "Attachment HTML fallback: using preview URL %s",
                preview_url,
            )
            return await _fetch_attachment_binary(client, preview_url, max_bytes)
    except Exception as e:  # nosec: B110
        logger.debug("Failed to get attachment preview: %s", e)
    return None


async def _handle_html_attachment(
    client: FreeFeedClient,
    url: str,
    max_bytes: int,
) -> tuple[bytes | None, str | None, int | None, str | None] | None:
    """Handle HTML attachment by attempting to fetch preview."""
    preview_result = await _try_html_preview(client, url, max_bytes)
    if preview_result:
        data, content_type, size, error = preview_result
        if not error:
            return data, content_type, size, None
    return None


async def _fetch_attachment_data(
    client: FreeFeedClient, attachment_url: str, max_bytes: int
) -> tuple[bytes | None, str | None, int | None, str | None]:
    """Fetch and validate attachment data."""
    urls = _get_fallback_urls(client, attachment_url)

    if not urls:
        return None, None, None, "invalid_url"

    for url in urls:
        data, content_type, size, error = await _fetch_attachment_binary(
            client, url, max_bytes
        )
        if error == "not_found":
            continue
        if error:
            return data, content_type, size, error
        if content_type and content_type.startswith("text/html"):
            result = await _handle_html_attachment(client, url, max_bytes)
            if result:
                return result
            return data, content_type, size, "html_response"

        return data, content_type, size, None

    return None, None, None, "not_found"


def should_skip_user(username: str, user_profile: dict) -> bool:
    """Return True if a user's content should be excluded from AI analysis."""
    config = _load_opt_out_config()
    if not config["enabled"]:
        return False

    if username in config["users"]:
        return True

    if config["respect_paused"] and user_profile.get("isGone") is True:
        return True

    if config["respect_private"] and user_profile.get("isPrivate") == "1":
        return True

    description = str(user_profile.get("description", "")).lower()
    return any(tag in description for tag in config["tags"])


def _build_user_map(payload: dict) -> dict[str, dict]:
    users = payload.get("users")
    if isinstance(users, dict):
        if users.get("id"):
            return {users["id"]: users}
        return {}

    if isinstance(users, list):
        return {
            user["id"]: user
            for user in users
            if isinstance(user, dict) and user.get("id")
        }

    return {}


def _filter_posts_by_opt_out(
    posts: list, user_map: dict, filtered_users: set, removed_post_ids: set
) -> list:
    """Filter posts by user opt-out status. Returns kept posts and updates sets."""
    kept_posts = []
    for post in posts:
        if not isinstance(post, dict):
            continue
        author_id = post.get("createdBy")
        user_profile = user_map.get(author_id, {})
        username = (
            user_profile.get("username") if isinstance(user_profile, dict) else None
        )

        if username and should_skip_user(username, user_profile):
            filtered_users.add(username)
            post_id = post.get("id")
            if post_id:
                removed_post_ids.add(post_id)
            continue

        kept_posts.append(post)
    return kept_posts


def _clean_related_content(payload: dict, removed_post_ids: set) -> None:
    """Remove comments and attachments related to filtered posts."""
    comments = payload.get("comments")
    if isinstance(comments, list):
        payload["comments"] = [
            comment
            for comment in comments
            if isinstance(comment, dict)
            and comment.get("postId") not in removed_post_ids
        ]

    attachments = payload.get("attachments")
    if isinstance(attachments, list):
        payload["attachments"] = [
            attachment
            for attachment in attachments
            if isinstance(attachment, dict)
            and attachment.get("postId") not in removed_post_ids
        ]


def _clean_timelines(payload: dict, removed_post_ids: set) -> None:
    """Remove filtered post IDs from timeline references."""
    timelines = payload.get("timelines")
    if isinstance(timelines, dict) and isinstance(timelines.get("posts"), list):
        timelines["posts"] = [
            post_id for post_id in timelines["posts"] if post_id not in removed_post_ids
        ]


def _filter_posts_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    config = _load_opt_out_config()
    if not config["enabled"]:
        return payload

    posts = payload.get("posts")
    if not isinstance(posts, list):
        return payload

    user_map = _build_user_map(payload)
    filtered_users: set[str] = set()
    removed_post_ids: set[str] = set()

    kept_posts = _filter_posts_by_opt_out(
        posts, user_map, filtered_users, removed_post_ids
    )
    payload["posts"] = kept_posts

    if removed_post_ids:
        _clean_timelines(payload, removed_post_ids)
        _clean_related_content(payload, removed_post_ids)
        payload["filtered_users"] = sorted(filtered_users)
        payload["filter_reason"] = FILTER_REASON

    return payload


def _summarize_tool_args(arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return arguments

    redacted_keys = {"password", "auth_token", "token", "data"}
    summarized: dict[str, Any] = {}

    for key, value in arguments.items():
        if key in redacted_keys:
            summarized[key] = "<redacted>"
        elif isinstance(value, list):
            summarized[key] = f"list({len(value)})"
        elif isinstance(value, dict):
            summarized[key] = "{...}"
        else:
            summarized[key] = value

    return summarized


# Tool definitions


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available FreeFeed tools."""
    return [
        # Timeline tools
        Tool(
            name="get_timeline",
            description=(
                "Get timeline feed from FreeFeed. Can get home feed, user posts, "
                "user likes, user comments, or discussions feed. "
                "Timeline types: 'home', 'posts', 'likes', 'comments', 'discussions', 'directs'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "timeline_type": {
                        "type": "string",
                        "enum": [
                            "home",
                            "posts",
                            "likes",
                            "comments",
                            "discussions",
                            "directs",
                        ],
                        "description": "Type of timeline to retrieve",
                        "default": "home",
                    },
                    "username": {
                        "type": "string",
                        "description": "Username (required for posts/likes/comments timelines)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": LIMIT_DESCRIPTION,
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "offset": {
                        "type": "integer",
                        "description": OFFSET_DESCRIPTION,
                        "minimum": 0,
                    },
                },
                "required": ["timeline_type"],
            },
        ),
        Tool(
            name="get_directs",
            description="Get direct posts timeline for current user",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": LIMIT_DESCRIPTION,
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "offset": {
                        "type": "integer",
                        "description": OFFSET_DESCRIPTION,
                        "minimum": 0,
                    },
                },
            },
        ),
        # Post tools
        Tool(
            name="get_post",
            description="Get a specific post by ID with comments and likes",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID",
                    },
                    "max_comments": {
                        "description": 'Max comments to return. Use "all" (default) for all comments, or a number to limit (shows first + last boundary comments and omits the middle).',
                        "default": "all",
                        "oneOf": [
                            {"type": "string", "enum": ["all"]},
                            {"type": "integer", "minimum": 1},
                        ],
                    },
                    "max_likes": {
                        "description": 'Max likes to return. Use "all" (default) for all likes, or a number to limit.',
                        "default": "all",
                        "oneOf": [
                            {"type": "string", "enum": ["all"]},
                            {"type": "integer", "minimum": 1},
                        ],
                    },
                },
                "required": ["post_id"],
            },
        ),
        Tool(
            name="create_post",
            description="Create a new post on FreeFeed with optional file attachments and optional group posting",
            inputSchema={
                "type": "object",
                "properties": {
                    "body": {
                        "type": "string",
                        "description": "Post text content",
                    },
                    "attachment_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to attach (will be uploaded automatically)",
                    },
                    "group_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of group usernames to post to (e.g., ['mygroup', 'anothergroup'])",
                    },
                },
                "required": ["body"],
            },
        ),
        Tool(
            name="create_direct_post",
            description="Create a direct post to one or more recipients",
            inputSchema={
                "type": "object",
                "properties": {
                    "body": {
                        "type": "string",
                        "description": "Post text content",
                    },
                    "recipients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of recipient usernames",
                    },
                    "attachment_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to attach (will be uploaded automatically)",
                    },
                },
                "required": ["body", "recipients"],
            },
        ),
        Tool(
            name="update_post",
            description="Update an existing post",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID to update",
                    },
                    "body": {
                        "type": "string",
                        "description": "New post text content",
                    },
                },
                "required": ["post_id", "body"],
            },
        ),
        Tool(
            name="delete_post",
            description="Delete a post",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID to delete",
                    },
                },
                "required": ["post_id"],
            },
        ),
        Tool(
            name="leave_direct",
            description="Leave a direct post thread",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID to leave",
                    },
                },
                "required": ["post_id"],
            },
        ),
        Tool(
            name="like_post",
            description="Like a post",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID to like",
                    },
                },
                "required": ["post_id"],
            },
        ),
        Tool(
            name="unlike_post",
            description="Remove like from a post",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID to unlike",
                    },
                },
                "required": ["post_id"],
            },
        ),
        Tool(
            name="hide_post",
            description="Hide a post from your feed",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID to hide",
                    },
                },
                "required": ["post_id"],
            },
        ),
        Tool(
            name="unhide_post",
            description="Unhide a previously hidden post",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID to unhide",
                    },
                },
                "required": ["post_id"],
            },
        ),
        # Attachment tools
        Tool(
            name="upload_attachment",
            description="Upload a file attachment (image, video, etc.) to FreeFeed. Returns attachment ID that can be used in posts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to upload",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="download_attachment",
            description=(
                "Download an attachment from a FreeFeed post. If the file is an image and "
                "small enough, returns image content plus a URL fallback; otherwise returns a URL. "
                "Can also save to file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "attachment_url": {
                        "type": "string",
                        "description": "URL of the attachment to download (from post/comment data)",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Optional path to save file. If not provided, returns base64-encoded data.",
                    },
                    "prefer_image": {
                        "type": "boolean",
                        "description": "Return image content when possible",
                        "default": True,
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Maximum bytes to return for inline image data",
                        "minimum": 256000,
                    },
                },
                "required": ["attachment_url"],
            },
        ),
        Tool(
            name="get_attachment_image",
            description=(
                "Download an attachment and return image content when possible. "
                "Returns image content plus a URL fallback; for large files, returns only the URL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "attachment_url": {
                        "type": "string",
                        "description": "URL of the attachment to download (from post/comment data)",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Maximum bytes to return for inline image data",
                        "minimum": 256000,
                    },
                },
                "required": ["attachment_url"],
            },
        ),
        Tool(
            name="get_post_attachments",
            description="Extract attachment URLs and metadata from a post. Returns list of attachments with URLs for downloading.",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID to get attachments from",
                    },
                },
                "required": ["post_id"],
            },
        ),
        # Comment tools
        Tool(
            name="add_comment",
            description="Add a comment to a post",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "Post ID to comment on",
                    },
                    "body": {
                        "type": "string",
                        "description": "Comment text",
                    },
                },
                "required": ["post_id", "body"],
            },
        ),
        Tool(
            name="update_comment",
            description="Update an existing comment",
            inputSchema={
                "type": "object",
                "properties": {
                    "comment_id": {
                        "type": "string",
                        "description": "Comment ID to update",
                    },
                    "body": {
                        "type": "string",
                        "description": "New comment text",
                    },
                },
                "required": ["comment_id", "body"],
            },
        ),
        Tool(
            name="delete_comment",
            description="Delete a comment",
            inputSchema={
                "type": "object",
                "properties": {
                    "comment_id": {
                        "type": "string",
                        "description": "Comment ID to delete",
                    },
                },
                "required": ["comment_id"],
            },
        ),
        # Search tools
        Tool(
            name="search_posts",
            description=(
                "Search posts on FreeFeed. Supports search operators: "
                "intitle:query (search in post text), "
                "incomment:query (search in comments), "
                "from:username (search by author), "
                "AND/OR (logical operators)"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query with optional operators",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "offset": {
                        "type": "integer",
                        "description": OFFSET_DESCRIPTION,
                        "minimum": 0,
                    },
                },
                "required": ["query"],
            },
        ),
        # User tools
        Tool(
            name="get_user_profile",
            description="Get user profile information",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username to get profile for",
                    },
                },
                "required": ["username"],
            },
        ),
        Tool(
            name="whoami",
            description="Get current authenticated user information",
            inputSchema={
                "type": "object",
                "properties": {
                    "compact": {
                        "type": "boolean",
                        "description": "Return a compact response to avoid large payloads",
                        "default": False,
                    }
                },
            },
        ),
        Tool(
            name="get_subscribers",
            description="Get list of user's subscribers (followers)",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username to get subscribers for",
                    },
                },
                "required": ["username"],
            },
        ),
        Tool(
            name="get_subscriptions",
            description="Get list of user's subscriptions (following)",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username to get subscriptions for",
                    },
                },
                "required": ["username"],
            },
        ),
        Tool(
            name="subscribe_user",
            description="Subscribe to (follow) a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username to subscribe to",
                    },
                },
                "required": ["username"],
            },
        ),
        Tool(
            name="unsubscribe_user",
            description="Unsubscribe from (unfollow) a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username to unsubscribe from",
                    },
                },
                "required": ["username"],
            },
        ),
        # Group tools
        Tool(
            name="get_my_groups",
            description="Get list of groups that current user is a member of",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_group_timeline",
            description="Get posts from a specific group",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_name": {
                        "type": "string",
                        "description": "Group username/name",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of posts to return",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "offset": {
                        "type": "integer",
                        "description": OFFSET_DESCRIPTION,
                        "minimum": 0,
                    },
                },
                "required": ["group_name"],
            },
        ),
        Tool(
            name="get_group_info",
            description="Get information about a specific group",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_name": {
                        "type": "string",
                        "description": "Group username/name",
                    },
                },
                "required": ["group_name"],
            },
        ),
    ]


async def _handle_tool_timeline(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle get_timeline tool."""
    result = await client.get_timeline(
        username=arguments.get("username"),
        timeline_type=arguments.get("timeline_type", "home"),
        limit=arguments.get("limit"),
        offset=arguments.get("offset"),
    )
    return _filter_posts_payload(result)


async def _handle_tool_directs(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle get_directs tool."""
    result = await client.get_directs(
        limit=arguments.get("limit"),
        offset=arguments.get("offset"),
    )
    return _filter_posts_payload(result)


async def _handle_tool_get_post(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle get_post tool."""
    result = await client.get_post(
        arguments["post_id"],
        max_comments=arguments.get("max_comments", "all"),
        max_likes=arguments.get("max_likes", "all"),
    )
    user_map = _build_user_map(result)
    post = result.get("posts") if isinstance(result, dict) else None
    if isinstance(post, dict):
        author_id = post.get("createdBy")
        user_profile = user_map.get(author_id, {})
        username = (
            user_profile.get("username") if isinstance(user_profile, dict) else None
        )
        if username and should_skip_user(username, user_profile):
            result = {
                "error": "Post author opted out of AI interactions",
                "filtered_users": [username],
                "filter_reason": FILTER_REASON,
            }
    return result


async def _handle_tool_create_post(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle create_post tool."""
    attachment_paths = arguments.get("attachment_paths")
    group_names = arguments.get("group_names")
    return await client.create_post(
        body=arguments["body"],
        attachment_files=attachment_paths if attachment_paths else None,
        group_names=group_names if group_names else None,
    )


async def _handle_tool_create_direct_post(
    client: FreeFeedClient, arguments: Any
) -> Any:
    """Handle create_direct_post tool."""
    recipients = arguments.get("recipients")
    if not isinstance(recipients, list) or not recipients:
        raise FreeFeedAPIError("Recipients list cannot be empty")
    attachment_paths = arguments.get("attachment_paths")
    return await client.create_direct_post(
        body=arguments["body"],
        recipients=recipients,
        attachment_files=attachment_paths if attachment_paths else None,
    )


async def _handle_tool_update_post(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle update_post tool."""
    return await client.update_post(
        post_id=arguments["post_id"],
        body=arguments["body"],
    )


async def _handle_tool_delete_post(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle delete_post tool."""
    return await client.delete_post(arguments["post_id"])


async def _handle_tool_leave_direct(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle leave_direct tool."""
    return await client.leave_direct(arguments["post_id"])


async def _handle_tool_like_post(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle like_post tool."""
    return await client.like_post(arguments["post_id"])


async def _handle_tool_unlike_post(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle unlike_post tool."""
    return await client.unlike_post(arguments["post_id"])


async def _handle_tool_hide_post(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle hide_post tool."""
    return await client.hide_post(arguments["post_id"])


async def _handle_tool_unhide_post(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle unhide_post tool."""
    return await client.unhide_post(arguments["post_id"])


async def _handle_tool_upload_attachment(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle upload_attachment tool."""
    return await client.upload_attachment(
        file_path=arguments["file_path"],
    )


async def _handle_tool_download_attachment(
    client: FreeFeedClient, arguments: Any
) -> tuple[Any, list[TextContent | ImageContent] | None]:
    """Handle download_attachment tool. Returns (result, early_return) tuple."""
    save_path = arguments.get("save_path")
    prefer_image = arguments.get("prefer_image", True)
    max_bytes = arguments.get("max_bytes")
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        max_bytes = _resolve_image_max_bytes()

    if save_path:
        saved_path = await client.download_attachment(
            attachment_url=arguments["attachment_url"],
            save_path=save_path,
        )
        return {
            "success": True,
            "saved_to": str(saved_path),
            "message": f"Attachment downloaded to {saved_path}",
        }, None

    attachment_url = arguments["attachment_url"]
    file_data, content_type, size, error = await _fetch_attachment_data(
        client, attachment_url, max_bytes
    )

    if error == "too_large":
        return {
            "success": False,
            "message": "Attachment is too large for inline data",
            "url": attachment_url,
            "max_bytes": max_bytes,
            "size": size,
            "content_type": content_type,
        }, None

    if error:
        return {
            "success": False,
            "message": "Attachment could not be fetched",
            "url": attachment_url,
            "error": error,
            "content_type": content_type,
        }, None

    if prefer_image and content_type and content_type.startswith("image/"):
        image_content = ImageContent(
            type="image",
            data=base64.b64encode(file_data or b"").decode("utf-8"),
            mimeType=content_type,
        )
        text_content = TextContent(
            type="text",
            text=json.dumps(
                {
                    "success": True,
                    "message": "Attachment returned as image content",
                    "url": attachment_url,
                    "size": size,
                    "content_type": content_type,
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
        return None, [image_content, text_content]

    return {
        "success": True,
        "data": base64.b64encode(file_data or b"").decode("utf-8"),
        "size": size,
        "message": "Attachment downloaded as base64 data",
        "content_type": content_type,
        "url": attachment_url,
    }, None


async def _handle_tool_get_attachment_image(
    client: FreeFeedClient, arguments: Any
) -> tuple[Any, list[TextContent | ImageContent] | None]:
    """Handle get_attachment_image tool. Returns (result, early_return) tuple."""
    attachment_url = arguments["attachment_url"]
    max_bytes = arguments.get("max_bytes")
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        max_bytes = _resolve_image_max_bytes()

    file_data, content_type, size, error = await _fetch_attachment_data(
        client, attachment_url, max_bytes
    )

    if error == "too_large":
        return {
            "success": False,
            "message": "Attachment is too large for inline image data",
            "url": attachment_url,
            "max_bytes": max_bytes,
            "size": size,
            "content_type": content_type,
        }, None

    if error:
        return {
            "success": False,
            "message": "Attachment could not be fetched",
            "url": attachment_url,
            "error": error,
            "content_type": content_type,
        }, None

    if not content_type or not content_type.startswith("image/"):
        return {
            "success": False,
            "message": "Attachment is not an image",
            "url": attachment_url,
            "size": size,
            "content_type": content_type,
        }, None

    image_content = ImageContent(
        type="image",
        data=base64.b64encode(file_data or b"").decode("utf-8"),
        mimeType=content_type,
    )
    text_content = TextContent(
        type="text",
        text=json.dumps(
            {
                "success": True,
                "message": "Attachment returned as image content",
                "url": attachment_url,
                "size": size,
                "content_type": content_type,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    return None, [image_content, text_content]


async def _handle_tool_get_post_attachments(
    client: FreeFeedClient, arguments: Any
) -> tuple[Any, list[TextContent | ImageContent] | None]:
    """Handle get_post_attachments tool. Returns (result, early_return) tuple."""
    post_data = await client.get_post(arguments["post_id"])
    user_map = _build_user_map(post_data)
    post = post_data.get("posts") if isinstance(post_data, dict) else None

    if isinstance(post, dict):
        author_id = post.get("createdBy")
        user_profile = user_map.get(author_id, {})
        username = (
            user_profile.get("username") if isinstance(user_profile, dict) else None
        )
        if username and should_skip_user(username, user_profile):
            result = {
                "error": "Post author opted out of AI interactions",
                "filtered_users": [username],
                "filter_reason": FILTER_REASON,
            }
            result = _add_post_urls(result, client.base_url)
            return None, [
                TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False),
                )
            ]

    attachments = []
    if "attachments" in post_data:
        att_list = post_data["attachments"]
        if isinstance(att_list, dict):
            att_list = [att_list]

        for att in att_list:
            attachment_info = {
                "id": att.get("id"),
                "fileName": att.get("fileName"),
                "fileSize": att.get("fileSize"),
                "mediaType": att.get("mediaType"),
                "url": client.get_attachment_url(att, "original"),
                "thumbnailUrl": client.get_attachment_url(att, "thumbnail"),
                "imageSizes": att.get("imageSizes", {}),
            }
            attachment_info = {
                k: v for k, v in attachment_info.items() if v is not None
            }
            attachments.append(attachment_info)

    return {
        "post_id": arguments["post_id"],
        "attachments": attachments,
        "count": len(attachments),
    }, None


async def _handle_tool_add_comment(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle add_comment tool."""
    return await client.add_comment(
        post_id=arguments["post_id"],
        body=arguments["body"],
    )


async def _handle_tool_update_comment(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle update_comment tool."""
    return await client.update_comment(
        comment_id=arguments["comment_id"],
        body=arguments["body"],
    )


async def _handle_tool_delete_comment(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle delete_comment tool."""
    return await client.delete_comment(arguments["comment_id"])


async def _handle_tool_search_posts(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle search_posts tool."""
    result = await client.search_posts(
        query=arguments["query"],
        limit=arguments.get("limit"),
        offset=arguments.get("offset"),
    )
    return _filter_posts_payload(result)


async def _handle_tool_get_user_profile(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle get_user_profile tool."""
    return await client.get_user_profile(arguments["username"])


async def _handle_tool_whoami(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle whoami tool."""
    result = await client.whoami()
    if arguments.get("compact"):
        result = _compact_whoami(result)
    return result


async def _handle_tool_get_subscribers(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle get_subscribers tool."""
    return await client.get_subscribers(arguments["username"])


async def _handle_tool_get_subscriptions(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle get_subscriptions tool."""
    return await client.get_subscriptions(arguments["username"])


async def _handle_tool_subscribe_user(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle subscribe_user tool."""
    return await client.subscribe_user(arguments["username"])


async def _handle_tool_unsubscribe_user(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle unsubscribe_user tool."""
    return await client.unsubscribe_user(arguments["username"])


async def _handle_tool_get_my_groups(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle get_my_groups tool."""
    return await client.get_my_groups()


async def _handle_tool_get_group_timeline(
    client: FreeFeedClient, arguments: Any
) -> Any:
    """Handle get_group_timeline tool."""
    result = await client.get_group_timeline(
        group_name=arguments["group_name"],
        limit=arguments.get("limit"),
        offset=arguments.get("offset"),
    )
    return _filter_posts_payload(result)


async def _handle_tool_get_group_info(client: FreeFeedClient, arguments: Any) -> Any:
    """Handle get_group_info tool."""
    return await client.get_group_info(arguments["group_name"])


# Tools whose responses should be slimmed to reduce token consumption
_SLIM_TOOLS = frozenset({
    "get_timeline",
    "get_directs",
    "get_post",
    "search_posts",
    "get_group_timeline",
})

# Tool handler dispatch map
_TOOL_HANDLERS = {
    "get_timeline": _handle_tool_timeline,
    "get_directs": _handle_tool_directs,
    "get_post": _handle_tool_get_post,
    "create_post": _handle_tool_create_post,
    "create_direct_post": _handle_tool_create_direct_post,
    "update_post": _handle_tool_update_post,
    "delete_post": _handle_tool_delete_post,
    "leave_direct": _handle_tool_leave_direct,
    "like_post": _handle_tool_like_post,
    "unlike_post": _handle_tool_unlike_post,
    "hide_post": _handle_tool_hide_post,
    "unhide_post": _handle_tool_unhide_post,
    "upload_attachment": _handle_tool_upload_attachment,
    "download_attachment": _handle_tool_download_attachment,
    "get_attachment_image": _handle_tool_get_attachment_image,
    "get_post_attachments": _handle_tool_get_post_attachments,
    "add_comment": _handle_tool_add_comment,
    "update_comment": _handle_tool_update_comment,
    "delete_comment": _handle_tool_delete_comment,
    "search_posts": _handle_tool_search_posts,
    "get_user_profile": _handle_tool_get_user_profile,
    "whoami": _handle_tool_whoami,
    "get_subscribers": _handle_tool_get_subscribers,
    "get_subscriptions": _handle_tool_get_subscriptions,
    "subscribe_user": _handle_tool_subscribe_user,
    "unsubscribe_user": _handle_tool_unsubscribe_user,
    "get_my_groups": _handle_tool_get_my_groups,
    "get_group_timeline": _handle_tool_get_group_timeline,
    "get_group_info": _handle_tool_get_group_info,
}


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent]:
    """Handle tool calls."""
    try:
        logger.info("MCP tool call: %s args=%s", name, _summarize_tool_args(arguments))
        start_time = time.monotonic()
        client = await get_client()

        if name not in _TOOL_HANDLERS:
            raise ValueError(f"Unknown tool: {name}")

        handler = _TOOL_HANDLERS[name]
        result = await handler(client, arguments)

        # Check if handler returned early with content
        if isinstance(result, tuple) and len(result) == 2:
            result_dict, early_return = result
            if early_return is not None:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                logger.info(MCP_TOOL_SUCCESS_LOG, name, elapsed_ms)
                return early_return
            result = result_dict

        result = _add_post_urls(result, client.base_url)
        if name in _SLIM_TOOLS:
            result = slim_response(result, keep_comments=(name == "get_post"))
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(MCP_TOOL_SUCCESS_LOG, name, elapsed_ms)
        return [
            TextContent(
                type="text", text=json.dumps(result, indent=2, ensure_ascii=False)
            )
        ]

    except FreeFeedAPIError as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error("FreeFeed API error in %s: %s", name, e)
        logger.warning(
            "MCP tool error: %s duration_ms=%.1f error=%s",
            name,
            elapsed_ms,
            e,
        )
        return [TextContent(type="text", text=f"Error: {str(e)}")]
    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "Unexpected error in %s duration_ms=%.1f error=%s",
            name,
            elapsed_ms,
            e,
            exc_info=True,
        )
        return [TextContent(type="text", text=f"Unexpected error: {str(e)}")]


async def main():
    """Run the MCP server."""
    stop_event = asyncio.Event()
    shutdown_started = False

    def _request_shutdown() -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        logger.info("FreeFeed MCP Server shutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _request_shutdown)
        loop.add_signal_handler(signal.SIGTERM, _request_shutdown)
    except NotImplementedError:
        pass

    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("FreeFeed MCP Server starting...")
            run_task = asyncio.create_task(
                app.run(
                    read_stream,
                    write_stream,
                    app.create_initialization_options(),
                )
            )
            stop_task = asyncio.create_task(stop_event.wait())
            done, _ = await asyncio.wait(
                {run_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done:
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    try:
                        await asyncio.wait_for(run_task, timeout=2.0)
                    except TimeoutError:
                        logger.warning("Server shutdown timed out; forcing exit")
            else:
                _request_shutdown()
    except KeyboardInterrupt:
        _request_shutdown()
    finally:
        if freefeed_client is not None:
            await freefeed_client.close()
        logger.info("FreeFeed MCP Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

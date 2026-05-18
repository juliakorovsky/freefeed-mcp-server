"""FreeFeed API client for MCP server."""

import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, urlparse
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

# Constants
_JSON_CONTENT_TYPE = "application/json"


def _resolve_log_level() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _configure_client_logger() -> None:
    """Ensure client logs are also written to a file."""
    default_path = os.path.join(".", "logs", "freefeed_client.log")
    log_path = os.getenv("FREEFEED_CLIENT_LOG_PATH", default_path).strip()
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


_configure_client_logger()


def _resolve_api_version() -> int:
    value = os.getenv("FREEFEED_API_VERSION", "4").strip()
    try:
        version = int(value)
        if version <= 0:
            raise ValueError
        return version
    except ValueError:
        logger.warning("Invalid FREEFEED_API_VERSION=%s, falling back to 4", value)
        return 4


class FreeFeedAPIError(Exception):
    """Base exception for FreeFeed API errors."""

    pass


class FreeFeedAuthError(FreeFeedAPIError):
    """Authentication error."""

    pass


class FreeFeedClient:
    """Client for FreeFeed API."""

    def __init__(
        self,
        base_url: str = "https://freefeed.net",
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_token: Optional[str] = None,
        api_version: Optional[int] = None,
    ):
        """Initialize FreeFeed client.

        Args:
            base_url: Base URL for FreeFeed instance
            username: FreeFeed username
            password: FreeFeed password
            auth_token: Application token or session token for auth
            api_version: FreeFeed API version (defaults to FREEFEED_API_VERSION or 4)
        """
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.auth_token: Optional[str] = auth_token
        self.api_version = (
            api_version if api_version is not None else _resolve_api_version()
        )
        self.client = httpx.AsyncClient(
            timeout=30.0,
            event_hooks={
                "request": [self._log_request],
                "response": [self._log_response],
            },
        )

    async def _log_request(self, request: httpx.Request) -> None:
        """Log outgoing requests in DEBUG mode."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        headers = dict(request.headers)
        if "X-Authentication-Token" in headers:
            headers["X-Authentication-Token"] = "<redacted>"
        if "Authorization" in headers:
            headers["Authorization"] = "<redacted>"

        logger.debug(
            "HTTP request: %s %s headers=%s",
            request.method,
            request.url,
            headers,
        )

    async def _log_response(self, response: httpx.Response) -> None:
        """Log incoming responses in DEBUG mode."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        logger.debug(
            "HTTP response: %s %s -> %s",
            response.request.method,
            response.request.url,
            response.status_code,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        if not self.auth_token and self.username and self.password:
            await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with auth token if available."""
        headers = {
            "Content-Type": _JSON_CONTENT_TYPE,
            "Accept": _JSON_CONTENT_TYPE,
        }
        if self.auth_token:
            headers["X-Authentication-Token"] = self.auth_token
        return headers

    def _api_url(self, path: str) -> str:
        """Build a versioned API URL for FreeFeed endpoints.

        The path must be a relative URL path without a scheme, host, query
        string, or fragment. Individual path segments are percent-encoded
        to safely include user-controlled values.
        """
        parsed = urlparse(path)

        # Disallow absolute URLs or paths with their own query/fragment. All
        # query parameters should be passed via the httpx request layer.
        if parsed.scheme or parsed.netloc:
            raise ValueError(f"API path must be relative, got: {path!r}")
        if parsed.params or parsed.query or parsed.fragment:
            raise ValueError(
                f"API path must not contain params, query, or fragment: {path!r}"
            )

        normalized_path = parsed.path.lstrip("/")
        if not normalized_path:
            raise ValueError("API path must not be empty")

        segments: List[str] = []
        for segment in normalized_path.split("/"):
            # Disallow empty / '.' / '..' segments to avoid ambiguous paths.
            if segment in ("", ".", ".."):
                raise ValueError(f"API path contains invalid segment: {path!r}")
            segments.append(quote(segment, safe=""))

        safe_path = "/".join(segments)
        return f"{self.base_url}/v{self.api_version}/{safe_path}"

    async def authenticate(self) -> Dict[str, Any]:
        """Authenticate with FreeFeed API.

        Returns:
            User data with auth token

        Raises:
            FreeFeedAuthError: If authentication fails
        """
        if not self.username or not self.password:
            raise FreeFeedAuthError("Username and password required for authentication")

        url = self._api_url("session")
        data = {
            "username": self.username,
            "password": self.password,
        }

        try:
            response = await self.client.post(
                url, json=data, headers={"Content-Type": _JSON_CONTENT_TYPE}
            )
            response.raise_for_status()
            result = response.json()

            # Extract auth token from response
            if "authToken" in result:
                self.auth_token = result["authToken"]
            elif "users" in result and result["users"].get("authToken"):
                self.auth_token = result["users"]["authToken"]
            else:
                raise FreeFeedAuthError("Auth token not found in response")

            logger.info(f"Successfully authenticated as {self.username}")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Authentication failed: {e}")
            raise FreeFeedAuthError(f"Authentication failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise FreeFeedAuthError(f"Authentication error: {e}")

    # Timeline methods

    async def get_timeline(
        self,
        username: Optional[str] = None,
        timeline_type: str = "posts",
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get timeline for user or home feed.

        Args:
            username: Username (None for home timeline)
            timeline_type: Type of timeline (posts, likes, comments, home, discussions, directs)
            limit: Number of posts to return
            offset: Offset for pagination

        Returns:
            Timeline data with posts
        """
        if timeline_type == "home":
            url = self._api_url("timelines/home")
        elif timeline_type == "discussions":
            url = self._api_url("timelines/filter/discussions")
        elif timeline_type == "directs":
            url = self._api_url("timelines/filter/directs")
        elif username:
            if timeline_type == "posts":
                url = self._api_url(f"timelines/{username}")
            elif timeline_type == "likes":
                url = self._api_url(f"timelines/{username}/likes")
            elif timeline_type == "comments":
                url = self._api_url(f"timelines/{username}/comments")
            else:
                raise ValueError(f"Unknown timeline type: {timeline_type}")
        else:
            raise ValueError("Username required for user timeline")

        params = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        response = await self.client.get(
            url,
            headers=self._get_headers(),
            params=params,
        )
        response.raise_for_status()
        return response.json()

    async def get_directs(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get direct posts timeline for current user."""
        return await self.get_timeline(
            timeline_type="directs",
            limit=limit,
            offset=offset,
        )

    # Attachment methods

    def _resolve_filename(
        self, file_path: Union[str, Path], filename: Optional[str]
    ) -> str:
        """Resolve the filename from path or override.

        Args:
            file_path: Path to the file
            filename: Override filename (optional)

        Returns:
            Resolved filename
        """
        if filename is not None:
            return filename

        if isinstance(file_path, Path):
            return file_path.name
        return Path(file_path).name if file_path else "attachment"

    def _resolve_mime_type(self, filename: str) -> str:
        """Resolve MIME type for a filename.

        Args:
            filename: Filename to determine MIME type for

        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"

    def _prepare_file_info(
        self,
        file_path: Union[str, Path],
        file_data: Optional[bytes],
        filename: Optional[str],
    ) -> tuple[bytes, str, str]:
        """Prepare file data, filename, and MIME type for upload.

        Args:
            file_path: Path to the file
            file_data: Raw file bytes (optional)
            filename: Override filename (optional)

        Returns:
            Tuple of (file_data, filename, mime_type)

        Raises:
            FreeFeedAPIError: If file not found
        """

        def _resolve_upload_path(raw_path: Union[str, Path]) -> Path:
            base_dir = Path(os.getenv("FREEFEED_UPLOAD_DIR", ".")).expanduser()
            base_dir = base_dir.resolve()
            candidate = Path(raw_path).expanduser()
            if candidate.is_absolute():
                resolved = candidate.resolve()
            else:
                resolved = (base_dir / candidate).resolve()
            try:
                resolved.relative_to(base_dir)
            except ValueError as exc:
                raise FreeFeedAPIError(
                    "Invalid file_path; must be within upload directory"
                ) from exc
            return resolved

        # Prepare file data
        if file_data is None:
            path = _resolve_upload_path(file_path)
            if not path.exists():
                raise FreeFeedAPIError(f"File not found: {file_path}")
            file_data = path.read_bytes()

        # Resolve filename
        resolved_filename = self._resolve_filename(file_path, filename)

        # Determine MIME type
        mime_type = self._resolve_mime_type(resolved_filename)

        return file_data, resolved_filename, mime_type

    async def upload_attachment(
        self,
        file_path: Union[str, Path],
        file_data: Optional[bytes] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload an attachment (image or other file).

        Args:
            file_path: Path to the file to upload (used if file_data is None)
            file_data: Raw file bytes (if provided, file_path is used only for filename)
            filename: Override filename (optional)

        Returns:
            Attachment data with ID

        Raises:
            FreeFeedAPIError: If upload fails
        """
        file_data, filename, mime_type = self._prepare_file_info(
            file_path, file_data, filename
        )

        url = self._api_url("attachments")

        # Prepare multipart form data
        files = {"file": (filename, file_data, mime_type)}

        # Note: For attachments, we need to use form-data, not JSON headers
        headers = {}
        if self.auth_token:
            headers["X-Authentication-Token"] = self.auth_token

        try:
            response = await self.client.post(
                url,
                files=files,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()

            logger.info(f"Successfully uploaded attachment: {filename}")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Attachment upload failed: {e}")
            raise FreeFeedAPIError(
                f"Attachment upload failed: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Attachment upload error: {e}")
            raise FreeFeedAPIError(f"Attachment upload error: {e}")

    async def download_attachment(
        self,
        attachment_url: str,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Union[bytes, Path]:
        """Download an attachment from FreeFeed.

        Args:
            attachment_url: URL of the attachment to download
            save_path: Optional path to save the file. If None, returns bytes.

        Returns:
            If save_path provided: Path to saved file
            If save_path is None: Raw bytes of the file

        Raises:
            FreeFeedAPIError: If download fails
        """

        def _is_allowed_attachment_url(raw_url: str) -> bool:
            parsed = urlparse(raw_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return False
            base_netloc = urlparse(self.base_url).netloc
            allowed_hosts = {base_netloc, "media.freefeed.net"}
            if parsed.netloc not in allowed_hosts:
                return False
            return "/attachments/" in parsed.path

        def _resolve_download_path(raw_path: Union[str, Path]) -> Path:
            base_dir = Path(
                os.getenv("FREEFEED_DOWNLOAD_DIR", "./downloads")
            ).expanduser()
            base_dir = base_dir.resolve()
            candidate = Path(raw_path).expanduser()
            if candidate.is_absolute():
                resolved = candidate.resolve()
            else:
                resolved = (base_dir / candidate).resolve()
            try:
                resolved.relative_to(base_dir)
            except ValueError as exc:
                raise FreeFeedAPIError(
                    "Invalid save_path; must be within download directory"
                ) from exc
            return resolved

        if not _is_allowed_attachment_url(attachment_url):
            raise FreeFeedAPIError("Attachment URL is not allowed")

        try:
            response = await self.client.get(attachment_url)
            response.raise_for_status()

            file_data = response.content

            if save_path:
                resolved_path = _resolve_download_path(save_path)
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                resolved_path.write_bytes(file_data)
                logger.info(f"Downloaded attachment to {resolved_path}")
                return resolved_path
            else:
                return file_data

        except httpx.HTTPStatusError as e:
            logger.error(f"Attachment download failed: {e}")
            raise FreeFeedAPIError(
                f"Attachment download failed: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Attachment download error: {e}")
            raise FreeFeedAPIError(f"Attachment download error: {e}")

    def get_attachment_url(
        self, attachment_data: Dict[str, Any], size: str = "original"
    ) -> Optional[str]:
        """Get URL for attachment from attachment data.

        Args:
            attachment_data: Attachment object from API response
            size: Size variant (original, t, t2) where t=thumbnail, t2=medium

        Returns:
            URL string or None if not available
        """
        # Try different possible structures
        if "url" in attachment_data:
            return attachment_data["url"]

        # For images with thumbnails
        if size == "thumbnail" and "thumbnailUrl" in attachment_data:
            return attachment_data["thumbnailUrl"]
        elif size == "thumbnail2" and "thumbnail2Url" in attachment_data:
            return attachment_data["thumbnail2Url"]
        elif size == "original" and "url" in attachment_data:
            return attachment_data["url"]

        # Construct URL manually if we have mediaType and id
        if "id" in attachment_data:
            # Format: https://freefeed.net/attachments/{id}
            return f"{self.base_url}/attachments/{attachment_data['id']}"

        return None

    async def get_attachment_preview_url(
        self,
        attachment_id: str,
        preview_type: str = "original",
        width: Optional[int] = None,
        height: Optional[int] = None,
        image_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get preview metadata for an attachment.

        Args:
            attachment_id: Attachment ID
            preview_type: Preview type (original, image, video, audio)
            width: Optional width for resized images
            height: Optional height for resized images
            image_format: Optional format (webp, jpeg, avif)

        Returns:
            Preview response with URL and mimeType
        """
        url = self._api_url(f"attachments/{attachment_id}/{preview_type}")
        params: Dict[str, Any] = {}
        if width is not None:
            params["width"] = width
        if height is not None:
            params["height"] = height
        if image_format:
            params["format"] = image_format

        response = await self.client.get(
            url,
            headers=self._get_headers(),
            params=params if params else None,
        )
        response.raise_for_status()
        return response.json()

    # Post methods

    async def get_post(
        self,
        post_id: str,
        max_comments: Union[str, int] = "all",
        max_likes: Union[str, int] = "all",
    ) -> Dict[str, Any]:
        """Get a specific post.

        Args:
            post_id: Post ID
            max_comments: Max comments to return ("all" or a numeric limit)
            max_likes: Max likes to return ("all" or a numeric limit)

        Returns:
            Post data with comments
        """
        url = self._api_url(f"posts/{post_id}")
        response = await self.client.get(
            url,
            headers=self._get_headers(),
            params={"maxComments": max_comments, "maxLikes": max_likes},
        )
        response.raise_for_status()
        return response.json()

    async def _upload_attachment_files(
        self, attachment_files: List[Union[str, Path]]
    ) -> List[str]:
        """Upload attachment files and extract their IDs.

        Args:
            attachment_files: List of file paths to upload

        Returns:
            List of attachment IDs
        """
        attachment_ids = []
        for file_path in attachment_files:
            logger.info(f"Uploading attachment: {file_path}")
            result = await self.upload_attachment(file_path)
            attachment_id = self._extract_attachment_id(result)
            if attachment_id:
                attachment_ids.append(attachment_id)
        return attachment_ids

    def _extract_attachment_id(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract attachment ID from upload response.

        Args:
            result: Upload response

        Returns:
            Attachment ID or None
        """
        if "attachments" in result:
            att = result["attachments"]
            if isinstance(att, dict):
                return att.get("id")
            elif isinstance(att, list) and len(att) > 0:
                return att[0].get("id")
        elif "id" in result:
            return result["id"]
        return None

    async def _resolve_feed_names(
        self,
        feeds: Optional[List[str]],
        group_names: Optional[List[str]],
    ) -> List[str]:
        """Resolve feed names from feeds and group names.

        Args:
            feeds: List of feed IDs
            group_names: List of group usernames

        Returns:
            List of feed names
        """
        feed_names = list(feeds) if feeds else []
        if group_names:
            feed_names.extend(group_names)
        if not feed_names:
            feed_names = [await self._get_default_feed_name()]
        return feed_names

    async def create_post(
        self,
        body: str,
        attachments: Optional[List[str]] = None,
        attachment_files: Optional[List[Union[str, Path]]] = None,
        feeds: Optional[List[str]] = None,
        group_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new post.

        Args:
            body: Post text
            attachments: List of attachment IDs (already uploaded)
            attachment_files: List of file paths to upload and attach
            feeds: List of feed IDs to post to (direct IDs)
            group_names: List of group usernames to post to (will be resolved to feed IDs)

        Returns:
            Created post data
        """
        attachment_ids = list(attachments) if attachments else []

        if attachment_files:
            uploaded_ids = await self._upload_attachment_files(attachment_files)
            attachment_ids.extend(uploaded_ids)

        feed_names = await self._resolve_feed_names(feeds, group_names)

        url = self._api_url("posts")
        data = {
            "post": {"body": body},
            "meta": {"feeds": feed_names},
        }

        if attachment_ids:
            data["post"]["attachments"] = attachment_ids

        response = await self.client.post(
            url,
            json=data,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def create_direct_post(
        self,
        body: str,
        recipients: List[str],
        attachments: Optional[List[str]] = None,
        attachment_files: Optional[List[Union[str, Path]]] = None,
    ) -> Dict[str, Any]:
        """Create a direct post to one or more recipients.

        Args:
            body: Post text
            recipients: List of usernames to receive the direct post
            attachments: List of attachment IDs (already uploaded)
            attachment_files: List of file paths to upload and attach

        Returns:
            Created post data
        """
        if not recipients:
            raise FreeFeedAPIError("Direct post requires at least one recipient")

        return await self.create_post(
            body=body,
            attachments=attachments,
            attachment_files=attachment_files,
            feeds=recipients,
            group_names=None,
        )

    async def leave_direct(self, post_id: str) -> Dict[str, Any]:
        """Leave a direct post thread.

        Args:
            post_id: Post ID

        Returns:
            Result payload
        """
        try:
            UUID(post_id)
        except ValueError as exc:
            raise FreeFeedAPIError("Invalid post_id format") from exc

        encoded_post_id = quote(post_id, safe="")
        url = self._api_url(f"posts/{encoded_post_id}/leave")
        response = await self.client.post(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json() if response.text else {"success": True}

    async def _get_default_feed_name(self) -> str:
        if self.username:
            return self.username

        whoami_data = await self.whoami()
        user_data = whoami_data.get("users")
        if isinstance(user_data, dict) and user_data.get("username"):
            self.username = user_data["username"]
            return self.username

        raise FreeFeedAPIError("Unable to determine username for posting")

    async def update_post(self, post_id: str, body: str) -> Dict[str, Any]:
        """Update a post.

        Args:
            post_id: Post ID
            body: New post text

        Returns:
            Updated post data
        """
        url = self._api_url(f"posts/{post_id}")
        data = {"post": {"body": body}}

        response = await self.client.put(
            url,
            json=data,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def delete_post(self, post_id: str) -> Dict[str, Any]:
        """Delete a post.

        Args:
            post_id: Post ID

        Returns:
            Deletion result
        """
        url = self._api_url(f"posts/{post_id}")
        response = await self.client.delete(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json() if response.text else {"success": True}

    async def like_post(self, post_id: str) -> Dict[str, Any]:
        """Like a post.
        Args:
            post_id: Post ID

        Returns:
            Like result
        """
        url = self._api_url(f"posts/{post_id}/like")
        response = await self.client.post(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json() if response.text else {"success": True}

    async def unlike_post(self, post_id: str) -> Dict[str, Any]:
        """Unlike a post.

        Args:
            post_id: Post ID

        Returns:
            Unlike result
        """
        url = self._api_url(f"posts/{post_id}/unlike")
        response = await self.client.post(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json() if response.text else {"success": True}

    async def hide_post(self, post_id: str) -> Dict[str, Any]:
        """Hide a post.

        Args:
            post_id: Post ID

        Returns:
            Hide result
        """
        url = self._api_url(f"posts/{post_id}/hide")
        response = await self.client.post(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json() if response.text else {"success": True}

    async def unhide_post(self, post_id: str) -> Dict[str, Any]:
        """Unhide a post.

        Args:
            post_id: Post ID

        Returns:
            Unhide result
        """
        url = self._api_url(f"posts/{post_id}/unhide")
        response = await self.client.post(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json() if response.text else {"success": True}

    # Comment methods

    async def add_comment(self, post_id: str, body: str) -> Dict[str, Any]:
        """Add a comment to a post.

        Args:
            post_id: Post ID
            body: Comment text

        Returns:
            Created comment data
        """
        url = self._api_url("comments")
        data = {
            "comment": {
                "body": body,
                "postId": post_id,
            }
        }

        response = await self.client.post(
            url,
            json=data,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def update_comment(self, comment_id: str, body: str) -> Dict[str, Any]:
        """Update a comment.

        Args:
            comment_id: Comment ID
            body: New comment text

        Returns:
            Updated comment data
        """
        url = self._api_url(f"comments/{comment_id}")
        data = {"comment": {"body": body}}

        response = await self.client.put(
            url,
            json=data,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def delete_comment(self, comment_id: str) -> Dict[str, Any]:
        """Delete a comment.

        Args:
            comment_id: Comment ID

        Returns:
            Deletion result
        """
        url = self._api_url(f"comments/{comment_id}")
        response = await self.client.delete(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json() if response.text else {"success": True}

    # Search methods

    async def search_posts(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search posts.

        Args:
            query: Search query (supports: intitle:, incomment:, from:, AND, OR)
            limit: Number of results to return
            offset: Offset for pagination

        Returns:
            Search results
        """
        params = {}
        params["query"] = query
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset

        url = self._api_url("search")
        response = await self.client.get(
            url,
            headers=self._get_headers(),
            params=params,
        )
        response.raise_for_status()
        return response.json()

    # User methods

    async def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Get user profile.

        Args:
            username: Username

        Returns:
            User profile data
        """
        url = self._api_url(f"users/{username}")
        response = await self.client.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    async def whoami(self) -> Dict[str, Any]:
        """Get current authenticated user.

        Returns:
            Current user data
        """
        url = self._api_url("users/whoami")
        response = await self.client.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    async def get_subscribers(self, username: str) -> Dict[str, Any]:
        """Get user's subscribers (followers).

        Args:
            username: Username

        Returns:
            Subscribers list
        """
        url = self._api_url(f"users/{username}/subscribers")
        response = await self.client.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    async def get_subscriptions(self, username: str) -> Dict[str, Any]:
        """Get user's subscriptions (following).

        Args:
            username: Username

        Returns:
            Subscriptions list
        """
        url = self._api_url(f"users/{username}/subscriptions")
        response = await self.client.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    async def subscribe_user(self, username: str) -> Dict[str, Any]:
        """Subscribe to a user.

        Args:
            username: Username to subscribe to

        Returns:
            Subscription result
        """
        url = self._api_url(f"users/{username}/subscribe")
        response = await self.client.post(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json() if response.text else {"success": True}

    async def unsubscribe_user(self, username: str) -> Dict[str, Any]:
        """Unsubscribe from a user.

        Args:
            username: Username to unsubscribe from

        Returns:
            Unsubscription result
        """
        url = self._api_url(f"users/{username}/unsubscribe")
        response = await self.client.post(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json() if response.text else {"success": True}

    # Group methods

    async def get_group_info(self, group_name: str) -> Dict[str, Any]:
        """Get information about a group.

        Args:
            group_name: Group username/name

        Returns:
            Group information including feeds
        """
        url = self._api_url(f"users/{group_name}")
        response = await self.client.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    async def get_group_timeline(
        self,
        group_name: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get timeline for a group.

        Args:
            group_name: Group username/name
            limit: Number of posts to return
            offset: Offset for pagination

        Returns:
            Group timeline with posts
        """
        return await self.get_timeline(
            username=group_name,
            timeline_type="posts",
            limit=limit,
            offset=offset,
        )

    async def get_my_groups(self) -> Dict[str, Any]:
        """Get list of groups that current user is member of.

        Returns:
            User data including subscriptions which contains groups
        """
        whoami_data = await self.whoami()

        # Extract groups from subscriptions
        groups = []
        if "subscriptions" in whoami_data:
            for sub in whoami_data["subscriptions"]:
                # Groups have type "group" in their user data
                if sub.get("type") == "group":
                    groups.append(
                        {
                            "id": sub.get("id"),
                            "username": sub.get("username"),
                            "screenName": sub.get("screenName"),
                            "type": sub.get("type"),
                        }
                    )

        return {"groups": groups, "count": len(groups)}

    def _extract_posts_feed_id(self, group_info: Dict[str, Any]) -> Optional[str]:
        """Extract Posts feed ID from group info.

        Args:
            group_info: Group information from API

        Returns:
            Posts feed ID or None
        """
        if "users" not in group_info:
            return None

        user_data = group_info["users"]
        if not isinstance(user_data, dict):
            return None

        subscriptions = user_data.get("subscriptions", [])
        for sub in subscriptions:
            if sub.get("name") == "Posts":
                return sub.get("id")

        return None

    async def resolve_feed_ids(
        self,
        group_names: Optional[List[str]] = None,
    ) -> List[str]:
        """Resolve group names to feed IDs.

        Args:
            group_names: List of group usernames

        Returns:
            List of feed IDs
        """
        if not group_names:
            return []

        feed_ids = []
        for group_name in group_names:
            try:
                group_info = await self.get_group_info(group_name)
                feed_id = self._extract_posts_feed_id(group_info)
                if feed_id:
                    feed_ids.append(feed_id)
            except Exception as e:
                logger.warning(f"Could not resolve group {group_name}: {e}")
                continue

        return feed_ids

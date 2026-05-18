"""PydanticAI agent for FreeFeed assistant."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel

from .client import FreeFeedAPIError, FreeFeedClient
from .filters import slim_response

logger = logging.getLogger(__name__)

DEFAULT_OPT_OUT_TAGS = ["#noai", "#opt-out-ai", "#no-bots", "#ai-free"]
FILTER_REASON = "User opted out of AI interactions"


class AssistantRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    timeline_type: Optional[str] = Field(default=None)
    username: Optional[str] = Field(default=None)
    query: Optional[str] = Field(default=None)
    limit: Optional[int] = Field(default=None, ge=1, le=100)


class AssistantResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    posts: Optional[list[dict[str, Any]]] = None
    warnings: list[str] = Field(default_factory=list)
    filtered_users: list[str] = Field(default_factory=list)


@dataclass
class AssistantDeps:
    client: FreeFeedClient
    base_url: str
    request_id: str
    opt_out_config: dict[str, Any]


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _load_config_from_file(config_path: str, config: dict[str, Any]) -> None:
    """Load opt-out configuration from a JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = handle.read()
        import json

        payload = json.loads(data)
        if not isinstance(payload, dict):
            return

        if isinstance(payload.get("enabled"), bool):
            config["enabled"] = payload["enabled"]
        if isinstance(payload.get("manual_opt_out"), list):
            config["users"] = {
                str(u).strip() for u in payload["manual_opt_out"] if str(u).strip()
            }
        if isinstance(payload.get("tags"), list):
            config["tags"] = [str(t).strip() for t in payload["tags"] if str(t).strip()]
        if isinstance(payload.get("respect_private"), bool):
            config["respect_private"] = payload["respect_private"]
        if isinstance(payload.get("respect_paused"), bool):
            config["respect_paused"] = payload["respect_paused"]
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read opt-out config %s: %s", config_path, exc)


def _load_config_from_env(config: dict[str, Any]) -> None:
    """Load opt-out configuration from environment variables."""
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


def _load_opt_out_config() -> dict[str, Any]:
    config: dict[str, Any] = {
        "enabled": False,
        "users": set(),
        "tags": list(DEFAULT_OPT_OUT_TAGS),
        "respect_private": True,
        "respect_paused": True,
    }

    config_path = os.getenv("FREEFEED_OPTOUT_CONFIG")
    if config_path:
        _load_config_from_file(config_path, config)

    _load_config_from_env(config)

    return config


def _should_skip_user(
    username: str, user_profile: dict, config: dict[str, Any]
) -> bool:
    if not config.get("enabled"):
        return False

    if username in config.get("users", set()):
        return True

    if config.get("respect_paused") and user_profile.get("isGone") is True:
        return True

    if config.get("respect_private") and user_profile.get("isPrivate") == "1":
        return True

    description = str(user_profile.get("description", "")).lower()
    return any(tag in description for tag in config.get("tags", []))


def _build_user_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
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
    posts: list[dict[str, Any]],
    user_map: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    """Filter posts and return kept posts, filtered users, and removed post IDs."""
    kept_posts: list[dict[str, Any]] = []
    filtered_users: set[str] = set()
    removed_post_ids: set[str] = set()

    for post in posts:
        if not isinstance(post, dict):
            continue
        author_id = post.get("createdBy")
        user_profile = user_map.get(author_id, {})
        username = (
            user_profile.get("username") if isinstance(user_profile, dict) else None
        )

        if username and _should_skip_user(username, user_profile, config):
            filtered_users.add(username)
            post_id = post.get("id")
            if post_id:
                removed_post_ids.add(post_id)
            continue

        kept_posts.append(post)

    return kept_posts, filtered_users, removed_post_ids


def _remove_related_content(
    payload: dict[str, Any], removed_post_ids: set[str]
) -> None:
    """Remove comments and attachments related to filtered posts."""
    timelines = payload.get("timelines")
    if isinstance(timelines, dict) and isinstance(timelines.get("posts"), list):
        timelines["posts"] = [
            post_id for post_id in timelines["posts"] if post_id not in removed_post_ids
        ]

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


def _filter_posts_payload(payload: Any, config: dict[str, Any]) -> Any:
    if not isinstance(payload, dict):
        return payload

    if not config.get("enabled"):
        return payload

    posts = payload.get("posts")
    if not isinstance(posts, list):
        return payload

    user_map = _build_user_map(payload)
    kept_posts, filtered_users, removed_post_ids = _filter_posts_by_opt_out(
        posts, user_map, config
    )

    payload["posts"] = kept_posts

    if removed_post_ids:
        _remove_related_content(payload, removed_post_ids)
        payload["filtered_users"] = sorted(filtered_users)
        payload["filter_reason"] = FILTER_REASON

    return payload


def _add_post_urls(payload: Any, base_url: str) -> Any:
    if not isinstance(payload, dict):
        return payload

    user_map = _build_user_map(payload)
    posts = payload.get("posts")

    def _apply(post: dict[str, Any]) -> None:
        if not isinstance(post, dict):
            return
        post_id = post.get("id")
        short_id = post.get("shortId")
        author_id = post.get("createdBy")
        username = None
        if isinstance(user_map.get(author_id), dict):
            username = user_map[author_id].get("username")

        if username and short_id:
            post["postUrl"] = f"{base_url}/{username}/{short_id}"
        elif post_id:
            post["postUrl"] = f"{base_url}/posts/{post_id}"

    if isinstance(posts, list):
        for post in posts:
            _apply(post)
    elif isinstance(posts, dict):
        _apply(posts)

    return payload


def _build_prompt(request: AssistantRequest) -> str:
    constraints: list[str] = []
    if request.timeline_type:
        constraints.append(f"timeline_type={request.timeline_type}")
    if request.username:
        constraints.append(f"username={request.username}")
    if request.query:
        constraints.append(f"query={request.query}")
    if request.limit:
        constraints.append(f"limit={request.limit}")

    if not constraints:
        return request.prompt

    constraints_text = ", ".join(constraints)
    return f"{request.prompt}\n\nConstraints: {constraints_text}"


def _check_user_opt_out(
    username: str, user_profile: dict[str, Any], ctx: RunContext[AssistantDeps]
) -> Optional[dict[str, Any]]:
    """Check if user opted out and return error response if so."""
    if username and _should_skip_user(username, user_profile, ctx.deps.opt_out_config):
        return {
            "error": "User opted out of AI interactions",
            "filtered_users": [username],
            "filter_reason": FILTER_REASON,
        }
    return None


def _register_timeline_tool(agent: Agent) -> None:
    @agent.tool
    async def get_timeline(
        ctx: RunContext[AssistantDeps],
        timeline_type: str = "home",
        username: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> dict[str, Any]:
        logger.info(
            "assistant_tool request_id=%s tool=get_timeline", ctx.deps.request_id
        )
        result = await ctx.deps.client.get_timeline(
            username=username,
            timeline_type=timeline_type,
            limit=limit,
            offset=offset,
        )
        result = _filter_posts_payload(result, ctx.deps.opt_out_config)
        return slim_response(_add_post_urls(result, ctx.deps.base_url))


def _register_search_tool(agent: Agent) -> None:
    @agent.tool
    async def search_posts(
        ctx: RunContext[AssistantDeps],
        query: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> dict[str, Any]:
        logger.info(
            "assistant_tool request_id=%s tool=search_posts", ctx.deps.request_id
        )
        result = await ctx.deps.client.search_posts(
            query=query,
            limit=limit,
            offset=offset,
        )
        result = _filter_posts_payload(result, ctx.deps.opt_out_config)
        return slim_response(_add_post_urls(result, ctx.deps.base_url))


def _register_post_tool(agent: Agent) -> None:
    @agent.tool
    async def get_post(
        ctx: RunContext[AssistantDeps],
        post_id: str,
        max_comments: str | int = "all",
        max_likes: str | int = "all",
    ) -> dict[str, Any]:
        """Get a specific post by ID. max_comments and max_likes can be "all" or a positive integer."""
        logger.info("assistant_tool request_id=%s tool=get_post", ctx.deps.request_id)
        result = await ctx.deps.client.get_post(
            post_id, max_comments=max_comments, max_likes=max_likes
        )
        user_map = _build_user_map(result)
        post = result.get("posts") if isinstance(result, dict) else None
        if isinstance(post, dict):
            author_id = post.get("createdBy")
            user_profile = user_map.get(author_id, {})
            username = (
                user_profile.get("username") if isinstance(user_profile, dict) else None
            )
            opt_out_response = _check_user_opt_out(username, user_profile, ctx)
            if opt_out_response:
                return opt_out_response
        return slim_response(_add_post_urls(result, ctx.deps.base_url), keep_comments=True)


def _register_profile_tool(agent: Agent) -> None:
    @agent.tool
    async def get_user_profile(
        ctx: RunContext[AssistantDeps], username: str
    ) -> dict[str, Any]:
        logger.info(
            "assistant_tool request_id=%s tool=get_user_profile", ctx.deps.request_id
        )
        result = await ctx.deps.client.get_user_profile(username)
        if isinstance(result, dict):
            user = result.get("users")
            if isinstance(user, dict):
                opt_out_response = _check_user_opt_out(username, user, ctx)
                if opt_out_response:
                    return opt_out_response
        return result


def _get_agent() -> Agent:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    model = AnthropicModel(model_name, api_key=api_key)

    system_prompt = (
        "You are a FreeFeed assistant. Use tools to retrieve posts, users, or search "
        "results before answering. Keep answers concise. If sources are available, "
        "include post URLs in sources. Respect opt-out filtering signals and do not "
        "summarize filtered content."
    )

    agent = Agent(
        model=model, result_type=AssistantResponse, system_prompt=system_prompt
    )

    _register_timeline_tool(agent)
    _register_search_tool(agent)
    _register_post_tool(agent)
    _register_profile_tool(agent)

    return agent


_agent: Optional[Agent] = None


def _get_or_create_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = _get_agent()
    return _agent


def _resolve_retry_count() -> int:
    raw = os.getenv("ASSISTANT_MAX_RETRIES", "2")
    try:
        value = int(raw)
        return max(1, min(value, 5))
    except ValueError:
        return 2


async def _run_with_retries(
    agent: Agent, prompt: str, deps: AssistantDeps
) -> AssistantResponse:
    max_retries = _resolve_retry_count()
    for attempt in range(1, max_retries + 1):
        try:
            result = await agent.run(prompt, deps=deps)
            return result.data
        except Exception as exc:
            logger.warning(
                "assistant_run_failed request_id=%s attempt=%s error=%s",
                deps.request_id,
                attempt,
                exc,
            )
            if attempt >= max_retries:
                raise
            await asyncio.sleep(0.3 * attempt)

    raise RuntimeError("Assistant failed after retries")


async def run_assistant(
    request: AssistantRequest, client: FreeFeedClient
) -> AssistantResponse:
    agent = _get_or_create_agent()
    request_id = str(uuid.uuid4())
    config = _load_opt_out_config()
    deps = AssistantDeps(
        client=client,
        base_url=client.base_url,
        request_id=request_id,
        opt_out_config=config,
    )
    prompt = _build_prompt(request)
    logger.info("assistant_run_start request_id=%s", request_id)
    try:
        result = await _run_with_retries(agent, prompt, deps)
        logger.info("assistant_run_done request_id=%s", request_id)
        return result
    except FreeFeedAPIError as exc:
        logger.error("assistant_run_api_error request_id=%s error=%s", request_id, exc)
        raise
    except Exception as exc:
        logger.error("assistant_run_error request_id=%s error=%s", request_id, exc)
        raise

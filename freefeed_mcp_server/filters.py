"""Response slimming utilities to reduce token consumption.

FreeFeed API responses include large users/subscribers/subscriptions arrays
that the MCP client doesn't need. This module strips them down to essentials.
"""

from typing import Any

_POST_FIELDS = frozenset({
    "id",
    "body",
    "createdAt",
    "updatedAt",
    "createdBy",
    "likes",
    "comments",
    "omittedComments",
    "omittedLikes",
    "postUrl",
    "backlinksCount",
})

_COMMENT_FIELDS = frozenset({
    "id",
    "body",
    "createdAt",
    "createdBy",
    "likes",
    "seqNumber",
    "postId",
})


def _build_username_map(users: Any) -> dict[str, str]:
    if isinstance(users, dict) and users.get("id"):
        return {users["id"]: users.get("username", users["id"])}
    if isinstance(users, list):
        return {
            u["id"]: u.get("username", u["id"])
            for u in users
            if isinstance(u, dict) and u.get("id")
        }
    return {}


def _slim_post(post: dict[str, Any], username_map: dict[str, str]) -> dict[str, Any]:
    slimmed = {k: post[k] for k in _POST_FIELDS if k in post}
    if "createdBy" in slimmed:
        slimmed["createdBy"] = username_map.get(slimmed["createdBy"], slimmed["createdBy"])
    if isinstance(slimmed.get("likes"), list):
        slimmed["likes"] = len(slimmed["likes"])
    if isinstance(slimmed.get("comments"), list):
        slimmed["comments"] = len(slimmed["comments"])
    return slimmed


def _slim_comment(comment: dict[str, Any], username_map: dict[str, str]) -> dict[str, Any]:
    slimmed = {k: comment[k] for k in _COMMENT_FIELDS if k in comment}
    if "createdBy" in slimmed:
        slimmed["createdBy"] = username_map.get(slimmed["createdBy"], slimmed["createdBy"])
    if isinstance(slimmed.get("likes"), list):
        slimmed["likes"] = len(slimmed["likes"])
    return slimmed


def slim_response(
    data: Any,
    keep_comments: bool = False,
    keep_attachments: bool = False,
) -> Any:
    """Strip unnecessary fields from a FreeFeed API response.

    Resolves createdBy user IDs to usernames, converts likes/comments arrays
    to counts, and drops subscribers/subscriptions/users/timelines arrays.

    keep_comments: include full comment bodies (use for get_post; omit for timelines)
    keep_attachments: include attachments array (use for get_post_attachments)
    """
    if not isinstance(data, dict):
        return data

    username_map = _build_username_map(data.get("users"))
    result: dict[str, Any] = {}

    posts = data.get("posts")
    if isinstance(posts, dict):
        result["posts"] = _slim_post(posts, username_map)
    elif isinstance(posts, list):
        result["posts"] = [_slim_post(p, username_map) for p in posts if isinstance(p, dict)]

    if keep_comments:
        comments = data.get("comments")
        if isinstance(comments, list):
            result["comments"] = [
                _slim_comment(c, username_map) for c in comments if isinstance(c, dict)
            ]

    if keep_attachments and "attachments" in data:
        result["attachments"] = data["attachments"]

    for key in ("filtered_users", "filter_reason", "error"):
        if key in data:
            result[key] = data[key]

    return result

"""
Post a thread to Bluesky using the AT Protocol (atproto SDK).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from atproto import Client
from atproto_client import models

from config import BLUESKY_POST_DELAY_SECONDS

logger = logging.getLogger(__name__)

BLUESKY_MAX_CHARS = 280


def _get_credentials() -> tuple[str, str]:
    """
    Read Bluesky handle and app password from the environment.

    Returns:
        (handle, password)

    Raises:
        RuntimeError: If variables are missing.
    """
    handle = os.environ.get("BLUESKY_HANDLE", "").strip()
    password = os.environ.get("BLUESKY_PASSWORD", "").strip()
    if not handle or not password:
        raise RuntimeError("BLUESKY_HANDLE and BLUESKY_PASSWORD must be set in the environment")
    return handle, password


def post_thread(thread_posts: list[str]) -> Optional[str]:
    """
    Post each string in sequence as a reply thread, with a delay between posts.

    Args:
        thread_posts: Post texts (first is root).

    Returns:
        AT URI of the root post, or None on failure.
    """
    posts = [(t or "").strip() for t in thread_posts if (t or "").strip()]
    if not posts:
        logger.warning("post_thread called with empty list")
        return None

    try:
        handle, password = _get_credentials()
    except RuntimeError as e:
        logger.error("%s", e)
        return None

    client = Client()
    try:
        client.login(handle, password)
    except Exception as e:
        logger.exception("Bluesky login failed: %s", e)
        return None

    root_ref: Optional[models.ComAtprotoRepoStrongRef.Main] = None
    parent_ref: Optional[models.ComAtprotoRepoStrongRef.Main] = None
    root_uri: Optional[str] = None

    try:
        for i, text in enumerate(posts):
            text = text[:BLUESKY_MAX_CHARS]
            reply_to: Optional[models.AppBskyFeedPost.ReplyRef] = None
            if i > 0 and root_ref is not None and parent_ref is not None:
                reply_to = models.AppBskyFeedPost.ReplyRef(root=root_ref, parent=parent_ref)

            res = client.send_post(text=text, reply_to=reply_to)
            ref = models.ComAtprotoRepoStrongRef.Main(uri=res.uri, cid=res.cid)
            if i == 0:
                root_ref = ref
                root_uri = res.uri
            parent_ref = ref
            if i < len(posts) - 1:
                time.sleep(float(BLUESKY_POST_DELAY_SECONDS))

        logger.info("Posted Bluesky thread (%d posts), root=%s", len(posts), root_uri)
        return root_uri
    except Exception as e:
        logger.exception("Bluesky post_thread failed: %s", e)
        return None

"""Helpers for extracting and rendering media payloads in Streamlit."""

import json
from typing import Any

import streamlit as st
from pydantic import BaseModel


def _as_dict(obj: Any) -> Any:
    """Best-effort conversion of objects (including Pydantic models) to dicts.

    Returns the original object when conversion is not supported or fails, so
    callers can decide how to handle non-dict values.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            return obj
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            return obj
    return obj


def _extract_media_content(payload: Any) -> dict | None:
    """Extract a media payload from a tool response dict.

    Accepts both ``media_content`` and ``MediaContent`` keys and normalizes the
    result to a plain ``dict`` when possible.
    """
    if not isinstance(payload, dict):
        return None
    media = payload.get("media_content") or payload.get("MediaContent")
    media = _as_dict(media)
    if isinstance(media, dict):
        return media
    return None


def _load_payload(content: Any) -> dict | None:
    """Load a tool payload from dict-like objects or JSON strings.

    Returns ``None`` when content is not a dict or valid JSON.
    """
    content = _as_dict(content)
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return None


def render_media_from_tool_result(content: Any) -> bool:
    """Render media in Streamlit from a tool result.

    Expects a top-level payload containing ``media_content`` (or ``MediaContent``)
    with ``type`` and ``url`` fields. Returns ``True`` only when something was
    rendered.
    """
    payload = _load_payload(content)
    if payload is None:
        return False
    media = _extract_media_content(payload)
    if not isinstance(media, dict):
        return False
    media_type = media.get("type")
    url = media.get("url")
    if not media_type or not url:
        return False
    return render_media_content({"type": media_type, "url": url})


def get_media_content_from_tool_result(content: Any) -> dict | None:
    """Return a normalized media dict from a tool result.

    The returned dict is shaped as ``{"type": "...", "url": "..."}`` or ``None``
    if the expected media fields are missing.
    """
    payload = _load_payload(content)
    if payload is None:
        return None
    media = _extract_media_content(payload)
    if not isinstance(media, dict):
        return None
    media_type = media.get("type")
    url = media.get("url")
    if not media_type or not url:
        return None
    return {"type": media_type, "url": url}


def render_media_content(media: dict) -> bool:
    """Render a media dict via Streamlit.

    Supported ``type`` values: ``image``, ``audio``, ``video``. Returns ``True``
    when rendering succeeds, otherwise ``False``.
    """
    media_type = media.get("type")
    url = media.get("url")
    if not media_type or not url:
        return False
    if media_type == "image":
        st.image(url)
        return True
    if media_type == "audio":
        st.audio(url)
        return True
    if media_type == "video":
        st.video(url)
        return True
    return False

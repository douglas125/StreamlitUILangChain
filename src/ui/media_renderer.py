import json
from typing import Any

import streamlit as st
from pydantic import BaseModel


def _as_dict(obj: Any) -> Any:
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
    if not isinstance(payload, dict):
        return None
    media = payload.get("media_content") or payload.get("MediaContent")
    media = _as_dict(media)
    if isinstance(media, dict):
        return media
    return None


def _load_payload(content: Any) -> dict | None:
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

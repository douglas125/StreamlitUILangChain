"""Helpers for extracting and rendering media payloads in Streamlit."""

import copy
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import altair as alt
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


def _extract_chart_content(payload: Any) -> dict | None:
    """Extract a chart payload from a tool response dict.

    Accepts both ``chart_content`` and ``ChartContent`` keys and normalizes the
    result to a plain ``dict`` when possible.
    """
    if not isinstance(payload, dict):
        return None
    chart = payload.get("chart_content") or payload.get("ChartContent")
    chart = _as_dict(chart)
    if isinstance(chart, dict):
        return chart
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


def get_chart_content_from_tool_result(content: Any) -> dict | None:
    """Return a normalized chart dict from a tool result.

    The returned dict is shaped as ``{"type": "altair", "spec": {...}}`` or
    ``None`` when the expected chart fields are missing.
    """
    payload = _load_payload(content)
    if payload is None:
        return None
    chart = _extract_chart_content(payload)
    if not isinstance(chart, dict):
        return None
    chart_type = chart.get("type")
    spec = chart.get("spec")
    if chart_type != "altair" or not isinstance(spec, dict):
        return None
    return {"type": chart_type, "spec": spec}


def _coerce_numeric_value(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    for currency in ("R$", "$", "EUR", "€", "£"):
        text = text.replace(currency, "")
    text = text.replace(" ", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return value


def _normalize_altair_spec(spec: dict[str, Any]) -> dict[str, Any]:
    encoding = spec.get("encoding")
    data = spec.get("data")
    if not isinstance(encoding, dict):
        return spec
    if not isinstance(data, dict):
        return spec
    values = data.get("values")
    if not isinstance(values, list):
        return spec

    quantitative_fields = []
    for _, enc in encoding.items():
        if not isinstance(enc, dict):
            pass
        else:
            if enc.get("type") == "quantitative":
                field = enc.get("field")
                if isinstance(field, str):
                    quantitative_fields.append(field)

    if len(quantitative_fields) == 0:
        return spec

    normalized_spec = copy.deepcopy(spec)
    normalized_values = normalized_spec["data"].get("values", [])
    for row in normalized_values:
        if not isinstance(row, dict):
            pass
        else:
            for field in quantitative_fields:
                if field in row:
                    row[field] = _coerce_numeric_value(row[field])
    return normalized_spec


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
        if isinstance(url, str):
            parsed = urlparse(url)
            is_local = parsed.scheme in ("", "file") or (len(parsed.scheme) == 1 and ":" in url)
            if is_local:
                st.caption(Path(url).name)
        return True
    if media_type == "audio":
        st.audio(url)
        return True
    if media_type == "video":
        st.video(url)
        return True
    return False


def render_chart_content(chart: dict) -> bool:
    """Render an Altair chart dict via Streamlit."""
    chart_type = chart.get("type")
    spec = chart.get("spec")
    if chart_type != "altair":
        return False
    if not isinstance(spec, dict):
        return False
    try:
        normalized_spec = _normalize_altair_spec(spec)
        chart_obj = alt.Chart.from_dict(normalized_spec)
    except Exception:
        return False
    st.altair_chart(chart_obj, width="stretch")
    return True

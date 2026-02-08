from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import inspect
import json
import xml.etree.ElementTree as ET

import streamlit as st

_NEXT_INTERACTION_TAG = "<next_interaction"
_NEXT_INTERACTION_CLOSE = "</next_interaction>"
_ALLOWED_PRESENTATION_MODES = {
    "free_text",
    "yes_no",
    "radio_box",
    "dropdown_box",
    "multi_select_checkbox",
    "date_input",
    "datetime_input",
    "data_editor",
}
_MODE_LABELS = {
    "yes_no": "Pick one",
    "radio_box": "Pick one",
    "multi_select_checkbox": "Pick one or more",
}
_DATE_LABEL = "Pick a date"
_DATETIME_LABEL = "Pick a date and time"
_SEND_LABEL = "Send"
_FREE_TEXT_HINT = "You can also type your own response below."
_PARSE_ERROR_HINT = (
    "Suggestions could not be loaded this time. You can still reply below."
)
_SUGGESTION_TOOLTIP = (
    "Why these options? They are suggested next steps based on the conversation."
)
_MULTI_COUNT_TEMPLATE = "{count} selected"
_PILLS_SUPPORTS_HELP = None


@dataclass(frozen=True)
class NextInteraction:
    """Structured hint for how the next user input should be collected."""

    presentation_mode: str
    suggested_user_follow_ups: list[str]
    params: dict[str, object]
    data: object | None
    signature_seed: str


def parse_next_interaction(raw_text):
    """Extract the <next_interaction> XML block from a response.

    Returns a tuple of (clean_text, next_interaction, parse_error). clean_text is
    the response without the XML block. If parsing fails, returns clean_text with
    next_interaction set to None and a parse_error string.
    """
    if not raw_text:
        return raw_text, None, None

    start = raw_text.rfind(_NEXT_INTERACTION_TAG)
    if start == -1:
        return raw_text, None, None

    end = raw_text.find(_NEXT_INTERACTION_CLOSE, start)
    if end == -1:
        clean_text = raw_text[:start].rstrip()
        return clean_text, None, "incomplete_next_interaction"

    end += len(_NEXT_INTERACTION_CLOSE)
    xml_text = raw_text[start:end]
    clean_text = raw_text[:start].rstrip()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return clean_text, None, "invalid_xml"

    if root.tag != "next_interaction":
        return clean_text, None, "unexpected_root"

    mode_el = root.find("presentation_mode")
    if mode_el is None or not mode_el.text:
        return clean_text, None, "missing_presentation_mode"
    presentation_mode = mode_el.text.strip()
    if presentation_mode not in _ALLOWED_PRESENTATION_MODES:
        return clean_text, None, "invalid_presentation_mode"
    if presentation_mode == "dropdown_box":
        presentation_mode = "radio_box"

    followups_el = root.find("suggested_user_follow_ups")
    suggestions = []
    if followups_el is not None:
        for q in followups_el.findall("q"):
            if q.text:
                text = q.text.strip()
                if text:
                    suggestions.append(text)

    params_raw = _parse_params_raw(root.find("params"))
    data_raw = _extract_data_raw(root.find("data"))

    if presentation_mode in {"yes_no", "free_text"}:
        suggestions = []
        signature_seed = _build_signature_seed(
            presentation_mode, suggestions, params_raw, data_raw
        )
        return (
            clean_text,
            NextInteraction(presentation_mode, suggestions, {}, None, signature_seed),
            None,
        )

    if presentation_mode in {"radio_box", "multi_select_checkbox"}:
        if not suggestions:
            return clean_text, None, "missing_suggestions"
        signature_seed = _build_signature_seed(
            presentation_mode, suggestions, params_raw, data_raw
        )
        return (
            clean_text,
            NextInteraction(presentation_mode, suggestions, {}, None, signature_seed),
            None,
        )

    if presentation_mode == "date_input":
        params, parse_error = _parse_date_params(params_raw)
        if parse_error:
            return clean_text, None, parse_error
        signature_seed = _build_signature_seed(
            presentation_mode, suggestions, params_raw, data_raw
        )
        return (
            clean_text,
            NextInteraction(presentation_mode, [], params, None, signature_seed),
            None,
        )

    if presentation_mode == "datetime_input":
        params, parse_error = _parse_datetime_params(params_raw)
        if parse_error:
            return clean_text, None, parse_error
        signature_seed = _build_signature_seed(
            presentation_mode, suggestions, params_raw, data_raw
        )
        return (
            clean_text,
            NextInteraction(presentation_mode, [], params, None, signature_seed),
            None,
        )

    if presentation_mode == "data_editor":
        params, data, parse_error = _parse_data_editor(params_raw, data_raw)
        if parse_error:
            return clean_text, None, parse_error
        signature_seed = _build_signature_seed(
            presentation_mode, suggestions, params_raw, data_raw
        )
        return (
            clean_text,
            NextInteraction(presentation_mode, [], params, data, signature_seed),
            None,
        )

    signature_seed = _build_signature_seed(
        presentation_mode, suggestions, params_raw, data_raw
    )
    return (
        clean_text,
        NextInteraction(presentation_mode, suggestions, {}, None, signature_seed),
        None,
    )


def strip_next_interaction_for_streaming(raw_text):
    """Remove any next_interaction XML (or partial tag) from display text."""
    if not raw_text:
        return raw_text

    start = raw_text.rfind(_NEXT_INTERACTION_TAG)
    if start != -1:
        return raw_text[:start]

    pending_len = _pending_tag_prefix_len(raw_text, _NEXT_INTERACTION_TAG)
    if pending_len:
        return raw_text[:-pending_len]

    return raw_text


def render_next_interaction(
    next_interaction, default_prompt="Ask away", message_count=0, parse_error=None
):
    """Render the proper Streamlit widget and return a user message if submitted.

    Always shows st.chat_input as a free-text fallback alongside any widgets.
    """
    if next_interaction is None or next_interaction.presentation_mode == "free_text":
        if parse_error:
            _render_parse_error_hint(parse_error)
            _render_free_text_hint()
        return st.chat_input(default_prompt)

    mode = next_interaction.presentation_mode
    suggestions = next_interaction.suggested_user_follow_ups

    if mode in {"radio_box", "multi_select_checkbox"} and not suggestions:
        _render_widget_hints("missing_suggestions")
        return st.chat_input(default_prompt)

    key = _interaction_key(next_interaction, message_count)

    if mode == "yes_no":
        choice = st.pills(
            _MODE_LABELS["yes_no"],
            ["Yes", "No"],
            selection_mode="single",
            default=None,
            key=f"{key}_pills",
        )
        if st.button(_SEND_LABEL, key=f"{key}_send", disabled=choice is None):
            if choice is None:
                st.warning("Please select an option.")
            else:
                return choice
        _render_widget_hints(parse_error)
        return st.chat_input(default_prompt)

    if mode == "radio_box":
        choice = st.pills(
            _MODE_LABELS["radio_box"],
            suggestions,
            selection_mode="single",
            default=None,
            key=f"{key}_pills",
            **_pills_help_kwargs(),
        )
        _render_why_options_hint()
        if st.button(_SEND_LABEL, key=f"{key}_send", disabled=choice is None):
            if choice is None:
                st.warning("Please select an option.")
            else:
                return choice
        _render_widget_hints(parse_error)
        return st.chat_input(default_prompt)

    if mode == "multi_select_checkbox":
        choices = st.pills(
            _MODE_LABELS["multi_select_checkbox"],
            suggestions,
            selection_mode="multi",
            default=None,
            key=f"{key}_pills",
            **_pills_help_kwargs(),
        )
        _render_why_options_hint()
        count = len(choices) if choices else 0
        st.caption(_MULTI_COUNT_TEMPLATE.format(count=count))
        if st.button(_SEND_LABEL, key=f"{key}_send", disabled=not choices):
            if not choices:
                st.warning("Please select at least one option.")
            elif len(choices) == 1:
                return choices[0]
            else:
                return ", ".join(choices)
        _render_widget_hints(parse_error)
        return st.chat_input(default_prompt)

    if mode == "date_input":
        params = next_interaction.params or {}
        value = st.date_input(
            _DATE_LABEL,
            value=params.get("default", date.today()),
            min_value=params.get("min"),
            max_value=params.get("max"),
            key=f"{key}_date",
        )
        if st.button(_SEND_LABEL, key=f"{key}_send"):
            return value.isoformat()
        _render_widget_hints(parse_error)
        return st.chat_input(default_prompt)

    if mode == "datetime_input":
        if not hasattr(st, "datetime_input"):
            _render_widget_hints(parse_error or "unsupported_datetime_input")
            return st.chat_input(default_prompt)
        params = next_interaction.params or {}
        value = st.datetime_input(
            _DATETIME_LABEL,
            value=params.get("default", datetime.now()),
            min_value=params.get("min"),
            max_value=params.get("max"),
            key=f"{key}_datetime",
        )
        if st.button(_SEND_LABEL, key=f"{key}_send"):
            return value.isoformat()
        _render_widget_hints(parse_error)
        return st.chat_input(default_prompt)

    if mode == "data_editor":
        if not hasattr(st, "data_editor"):
            _render_widget_hints(parse_error or "unsupported_data_editor")
            return st.chat_input(default_prompt)
        params = next_interaction.params or {}
        allow_add_rows = bool(params.get("allow_add_rows", False))
        num_rows = "dynamic" if allow_add_rows else "fixed"
        data = next_interaction.data or []
        edited = st.data_editor(
            data,
            key=f"{key}_editor",
            num_rows=num_rows,
            width="stretch",
        )
        if st.button(_SEND_LABEL, key=f"{key}_send"):
            normalized = _normalize_data_editor_output(edited)
            payload = json.dumps(normalized, default=_json_default, ensure_ascii=True)
            return f"data_editor: {payload}"
        _render_widget_hints(parse_error)
        return st.chat_input(default_prompt)

    return st.chat_input(default_prompt)


def _pending_tag_prefix_len(text, tag):
    max_len = 0
    tag_len = len(tag)
    for i in range(1, tag_len):
        if text.endswith(tag[:i]):
            max_len = i
    return max_len


def _interaction_key(next_interaction, message_count=0):
    signature = f"{message_count}|{next_interaction.signature_seed}"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
    return f"next_interaction_{digest}"


def _render_widget_hints(parse_error=None):
    _render_parse_error_hint(parse_error)
    _render_free_text_hint()


def _render_parse_error_hint(parse_error):
    if not parse_error:
        return
    st.caption(_PARSE_ERROR_HINT)
    if _debug_next_interaction_enabled():
        st.caption(f"Debug: next_interaction parse error ({parse_error}).")


def _render_free_text_hint():
    st.caption(_FREE_TEXT_HINT)


def _debug_next_interaction_enabled():
    return bool(st.session_state.get("_debug_next_interaction", False))


def _pills_help_kwargs():
    if _pills_supports_help():
        return {"help": _SUGGESTION_TOOLTIP}
    return {}


def _pills_supports_help():
    global _PILLS_SUPPORTS_HELP
    if _PILLS_SUPPORTS_HELP is None:
        try:
            _PILLS_SUPPORTS_HELP = "help" in inspect.signature(st.pills).parameters
        except (TypeError, ValueError):
            _PILLS_SUPPORTS_HELP = False
    return _PILLS_SUPPORTS_HELP


def _render_why_options_hint():
    if _pills_supports_help():
        return
    if hasattr(st, "popover"):
        with st.popover("Why these options?"):
            st.caption(
                "These are suggested next steps based on the conversation so far."
            )
    else:
        st.caption(
            "Why these options? These are suggested next steps based on the conversation."
        )


def _parse_params_raw(params_el):
    params = {}
    if params_el is None:
        return params
    for child in list(params_el):
        if child.tag:
            tag = child.tag.strip()
            if child.text is None:
                params[tag] = ""
            else:
                params[tag] = child.text.strip()
    return params


def _extract_data_raw(data_el):
    if data_el is None:
        return None
    if data_el.text is None:
        return ""
    return data_el.text.strip()


def _build_signature_seed(presentation_mode, suggestions, params_raw, data_raw):
    parts = [presentation_mode]
    if suggestions:
        parts.extend(suggestions)
    if params_raw:
        for key in sorted(params_raw.keys()):
            parts.append(f"{key}={params_raw[key]}")
    if data_raw:
        data_hash = hashlib.sha1(data_raw.encode("utf-8")).hexdigest()[:10]
        parts.append(f"data_hash={data_hash}")
    return "|".join(parts)


def _parse_date_params(params_raw):
    parsed = {"default": date.today(), "min": None, "max": None}
    default_text = params_raw.get("default")
    if default_text:
        default_value = _parse_date_value(default_text)
        if default_value is None:
            return None, "invalid_date_default"
        parsed["default"] = default_value

    min_text = params_raw.get("min")
    if min_text:
        min_value = _parse_date_value(min_text)
        if min_value is None:
            return None, "invalid_date_min"
        parsed["min"] = min_value

    max_text = params_raw.get("max")
    if max_text:
        max_value = _parse_date_value(max_text)
        if max_value is None:
            return None, "invalid_date_max"
        parsed["max"] = max_value

    min_value = parsed["min"]
    max_value = parsed["max"]
    if min_value and max_value and min_value > max_value:
        return None, "invalid_date_range"

    default_value = parsed["default"]
    if min_value and default_value < min_value:
        parsed["default"] = min_value
    if max_value and parsed["default"] > max_value:
        parsed["default"] = max_value

    return parsed, None


def _parse_datetime_params(params_raw):
    parsed = {"default": datetime.now(), "min": None, "max": None}
    default_text = params_raw.get("default")
    if default_text:
        default_value = _parse_datetime_value(default_text)
        if default_value is None:
            return None, "invalid_datetime_default"
        parsed["default"] = default_value

    min_text = params_raw.get("min")
    if min_text:
        min_value = _parse_datetime_value(min_text)
        if min_value is None:
            return None, "invalid_datetime_min"
        parsed["min"] = min_value

    max_text = params_raw.get("max")
    if max_text:
        max_value = _parse_datetime_value(max_text)
        if max_value is None:
            return None, "invalid_datetime_max"
        parsed["max"] = max_value

    min_value = parsed["min"]
    max_value = parsed["max"]
    if min_value and max_value and min_value > max_value:
        return None, "invalid_datetime_range"

    default_value = parsed["default"]
    if min_value and default_value < min_value:
        parsed["default"] = min_value
    if max_value and parsed["default"] > max_value:
        parsed["default"] = max_value

    return parsed, None


def _parse_date_value(text):
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _parse_datetime_value(text):
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is not None:
        return None
    return value


def _parse_data_editor(params_raw, data_raw):
    if data_raw is None or data_raw == "":
        return None, None, "missing_data"
    try:
        data = json.loads(data_raw)
    except json.JSONDecodeError:
        return None, None, "invalid_data_json"
    if not isinstance(data, list):
        return None, None, "invalid_data_format"
    if not data:
        return None, None, "empty_data"
    for row in data:
        if not isinstance(row, dict):
            return None, None, "invalid_data_rows"

    allow_add_rows = _parse_bool(params_raw.get("allow_add_rows", "false"))
    columns = _parse_columns(params_raw.get("columns"))
    if columns:
        data = _apply_column_order(data, columns)

    params = {"allow_add_rows": allow_add_rows, "columns": columns}
    return params, data, None


def _parse_bool(text):
    if text is None:
        return False
    normalized = text.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _parse_columns(text):
    if not text:
        return []
    parts = [part.strip() for part in text.split(",")]
    seen = set()
    columns = []
    for part in parts:
        if part:
            if part not in seen:
                seen.add(part)
                columns.append(part)
    return columns


def _apply_column_order(data, columns):
    columns_set = set(columns)
    ordered = []
    for row in data:
        reordered = {}
        for col in columns:
            if col in row:
                reordered[col] = row[col]
            else:
                reordered[col] = None
        for key in row:
            if key not in columns_set:
                reordered[key] = row[key]
        ordered.append(reordered)
    return ordered


def _normalize_data_editor_output(edited):
    if edited is None:
        return []
    if isinstance(edited, list):
        return edited
    if isinstance(edited, dict):
        return [edited]
    to_dict = getattr(edited, "to_dict", None)
    if to_dict:
        return to_dict(orient="records")
    return []


def _json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    item = getattr(value, "item", None)
    if item:
        return item()
    return str(value)

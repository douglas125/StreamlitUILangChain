from dataclasses import dataclass
import hashlib
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
}


@dataclass(frozen=True)
class NextInteraction:
    """Structured hint for how the next user input should be collected."""

    presentation_mode: str
    suggested_user_follow_ups: list[str]


def parse_next_interaction(raw_text):
    """Extract the <next_interaction> XML block from a response.

    Returns a tuple of (clean_text, next_interaction). clean_text is the
    response without the XML block. If parsing fails, returns the original
    raw_text and None.
    """
    if not raw_text:
        return raw_text, None

    start = raw_text.rfind(_NEXT_INTERACTION_TAG)
    if start == -1:
        return raw_text, None

    end = raw_text.find(_NEXT_INTERACTION_CLOSE, start)
    if end == -1:
        return raw_text, None

    end += len(_NEXT_INTERACTION_CLOSE)
    xml_text = raw_text[start:end]
    clean_text = raw_text[:start].rstrip()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return raw_text, None

    if root.tag != "next_interaction":
        return clean_text, None

    mode_el = root.find("presentation_mode")
    if mode_el is None or not mode_el.text:
        return clean_text, None
    presentation_mode = mode_el.text.strip()
    if presentation_mode not in _ALLOWED_PRESENTATION_MODES:
        return clean_text, None
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

    if presentation_mode in {"yes_no", "free_text"}:
        suggestions = []
    elif not suggestions:
        return clean_text, None

    return clean_text, NextInteraction(presentation_mode, suggestions)


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
    next_interaction, default_prompt="Ask away", message_count=0
):
    """Render the proper Streamlit widget and return a user message if submitted.

    Always shows st.chat_input as a free-text fallback alongside any widgets.
    """
    if next_interaction is None or next_interaction.presentation_mode == "free_text":
        return st.chat_input(default_prompt)

    mode = next_interaction.presentation_mode
    suggestions = next_interaction.suggested_user_follow_ups

    if mode in {"radio_box", "multi_select_checkbox"} and not suggestions:
        return st.chat_input(default_prompt)

    key = _interaction_key(next_interaction, message_count)

    if mode == "yes_no":
        col_yes, col_no = st.columns(2)
        if col_yes.button("Yes", key=f"{key}_yes"):
            return "Yes"
        if col_no.button("No", key=f"{key}_no"):
            return "No"
        return st.chat_input(default_prompt)

    if mode == "radio_box":
        with st.form(key=f"{key}_form"):
            choice = st.radio("Pick a question", suggestions, index=None)
            submitted = st.form_submit_button("Send")
        if submitted:
            if choice is None:
                st.warning("Please select an option.")
            else:
                return choice
        return st.chat_input(default_prompt)

    if mode == "multi_select_checkbox":
        with st.form(key=f"{key}_form"):
            choices = st.multiselect("Pick one or more", suggestions)
            submitted = st.form_submit_button("Send")
        if submitted:
            if not choices:
                st.warning("Please select at least one option.")
            elif len(choices) == 1:
                return choices[0]
            else:
                return ", ".join(choices)
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
    signature = f"{message_count}|{next_interaction.presentation_mode}|" + "|".join(
        next_interaction.suggested_user_follow_ups
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
    return f"next_interaction_{digest}"

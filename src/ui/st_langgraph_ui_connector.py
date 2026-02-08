import json
import uuid

import streamlit as st

from src.ui.media_renderer import get_media_content_from_tool_result
from src.ui.media_renderer import render_media_content


class _StreamState:
    """Mutable container for all state accumulated during a single streaming response.

    Holds references to lazily-created Streamlit widgets (thinking status, tools
    status, response placeholder) as well as the accumulated text for thinking and
    response output. Passed through the chain of _handle_* methods so they can
    read and mutate shared state without individual return values.

    Attributes:
        thinking_status: st.status widget for the reasoning/thinking section.
            Created on the first reasoning token; None if the model has no
            reasoning output.
        thinking_placeholder: st.empty inside thinking_status, used to
            incrementally render accumulated thinking text.
        response_container: st.chat_message context for thinking and response
            output, using the sparkles avatar. Lazily created on the first
            thinking or text token.
        response_placeholder: st.empty for the main response text. Created on
            the first text token, inside response_container.
        tools_container: st.chat_message context for tool calls, using the
            wrench avatar. Lazily created on the first tool invocation.
        tools_status: st.status widget wrapping all tool invocations and
            results, nested inside tools_container. Created on the first tool
            call; None if no tools are used.
        tool_call_count: Number of tool invocations seen so far. Used to build
            the collapsed label (e.g. "Used 2 tools").
        full_thinking: Accumulated reasoning text, re-rendered on each token.
        full_response: Accumulated response text, re-rendered on each token.
    """

    def __init__(self):
        self.thinking_status = None
        self.thinking_placeholder = None
        self.response_placeholder = None
        self.response_container = None
        self.tools_container = None
        self.tools_status = None
        self.tool_call_count = 0
        self.full_thinking = ""
        self.full_response = ""
        self.pending_media = []

    def tools_label(self):
        """Return a human-readable label summarizing how many tools were used.

        Examples: "Used 1 tool", "Used 3 tools".
        """
        return (
            f"Used {self.tool_call_count} tool{'s' if self.tool_call_count > 1 else ''}"
        )


class StLanggraphUIConnector:
    """Bridges a LangGraph agent to a Streamlit chat interface.

    Manages the full chat lifecycle: rendering conversation history from agent
    state, accepting new user input via st.chat_input, and streaming the agent's
    response in real time. Supports three concurrent UI sections during streaming:

    - Thinking: a collapsible st.status that displays reasoning tokens (for
      models that emit reasoning_content). Expands while thinking, collapses
      when the response begins.
    - Tools: a single collapsible st.status that groups all tool invocations
      (with their JSON arguments) and results. Expands while tools run,
      collapses when the response begins.
    - Response: the main answer text, streamed token-by-token into an
      st.empty placeholder.

    Args:
        agent: A compiled LangGraph agent that supports .stream() and
            .get_state(). Must accept stream_mode=["messages", "updates"].
        replacement_dict: Key-value pairs for dynamic system prompt
            placeholder replacement (e.g. {"[[DATE]]": "<date>...</date>"}).
            Passed to the agent via the context parameter on each stream call.
    """

    def __init__(self, agent, replacement_dict={}):
        self.agent = agent
        self.replacement_dict = replacement_dict
        self.thread_id = str(uuid.uuid4())

    def new_thread(self):
        """Creates a new conversation thread"""
        self.thread_id = str(uuid.uuid4())
        st.rerun()

    def display_chat(self):
        """Top-level entry point: render history then handle new user input.

        Call this once per Streamlit rerun (typically at the end of app.py).
        On each rerun it replays the full conversation from agent state, then
        if the user has submitted a new message via st.chat_input, it streams
        the agent's response. The chat input is disabled while streaming to
        prevent overlapping requests.
        """
        self._display_history()

        is_streaming = st.session_state.get("_streaming", False)
        stream_error = st.session_state.pop("_stream_error", None)
        if stream_error:
            st.error(stream_error)

        if is_streaming:
            st.chat_input("Ask away", disabled=True)
            pending_msg = st.session_state.pop("_pending_msg", "")
            with st.chat_message("user", avatar="👤"):
                st.markdown(pending_msg)
            try:
                self._stream_response(pending_msg)
            except Exception as exc:
                st.session_state["_stream_error"] = f"Streaming error: {exc}"
            finally:
                st.session_state["_streaming"] = False
                st.rerun()
        else:
            if user_msg := st.chat_input("Ask away"):
                st.session_state["_streaming"] = True
                st.session_state["_pending_msg"] = user_msg
                st.rerun()

    def _display_history(self):
        """Replay all messages from agent state as Streamlit chat bubbles.

        Iterates the message list from the agent's checkpointed state and
        renders each message appropriately:

        - Human messages: rendered as user chat bubbles.
        - AI messages with tool_calls: the tool_calls are buffered and paired
          with subsequent ToolMessages for grouped display.
        - AI messages without tool_calls: rendered as assistant chat bubbles,
          with an optional "Thinking" expander if reasoning_content is present.
        - Tool messages: buffered into a group and rendered via
          _display_tool_group once the next non-tool message is encountered
          (or at the end of the list).
        """
        state = self.agent.get_state({"configurable": {"thread_id": self.thread_id}})
        messages = state.values.get("messages", [])
        tool_buffer = []
        pending_tool_calls = []
        pending_reasoning = None
        pending_media = []
        for msg in messages:
            if msg.type == "tool":
                tool_buffer.append(msg)
            else:
                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    pending_tool_calls.extend(tool_calls)
                    reasoning = msg.additional_kwargs.get("reasoning_content")
                    if reasoning:
                        if pending_reasoning is None:
                            pending_reasoning = reasoning
                        else:
                            pending_reasoning += "\n\n" + reasoning
                else:
                    if tool_buffer:
                        media_items = self._display_tool_group(
                            pending_tool_calls, tool_buffer, pending_reasoning
                        )
                        if media_items:
                            pending_media.extend(media_items)
                        tool_buffer = []
                        pending_tool_calls = []
                        pending_reasoning = None
                    role = "user" if msg.type == "human" else "assistant"
                    avatar = "👤" if msg.type == "human" else "✨"
                    with st.chat_message(role, avatar=avatar):
                        if msg.type != "human":
                            reasoning = msg.additional_kwargs.get("reasoning_content")
                            if reasoning:
                                with st.expander("Thinking", expanded=False):
                                    st.markdown(reasoning)
                        st.markdown(
                            msg.content
                            if isinstance(msg.content, str)
                            else msg.content[0]["text"]
                        )
                        if role == "assistant" and pending_media:
                            for media in pending_media:
                                render_media_content(media)
                            pending_media = []
        if tool_buffer:
            media_items = self._display_tool_group(
                pending_tool_calls, tool_buffer, pending_reasoning
            )
            if media_items:
                pending_media.extend(media_items)
        if pending_media:
            with st.chat_message("assistant", avatar="âœ¨"):
                for media in pending_media:
                    render_media_content(media)

    def _display_tool_group(self, tool_calls, tool_msgs, reasoning=None):
        """Render a group of tool invocations and their results in a single
        collapsed status widget, optionally preceded by a thinking expander.

        If the AIMessage that triggered the tool calls also contained
        reasoning_content, it is displayed in a "Thinking" expander inside its
        own assistant chat bubble before the tool group.

        Pairs each tool_call with its corresponding ToolMessage by matching
        tool_call IDs. For each pair, renders an "Invoking: <name>" expander
        (showing the call arguments as formatted JSON) followed by a
        "Result: <name>" expander (showing the tool's response).

        Args:
            tool_calls: List of tool_call dicts from the preceding AIMessage.
                Each dict has 'name', 'args', and 'id' keys.
            tool_msgs: List of ToolMessage objects. Each has a tool_call_id
                attribute used to match against tool_calls.
            reasoning: Optional reasoning_content string from the AIMessage
                that preceded the tool calls. Displayed in a collapsed
                "Thinking" expander if present.
        """
        media_items = []
        if reasoning:
            with st.chat_message("assistant", avatar="✨"):
                with st.expander("Thinking", expanded=False):
                    st.markdown(reasoning)
        count = len(tool_msgs)
        label = f"Used {count} tool{'s' if count > 1 else ''}"
        results_by_id = {tm.tool_call_id: tm for tm in tool_msgs}
        with st.chat_message("assistant", avatar="🔧"):
            with st.status(label, state="complete", expanded=False):
                for tc in tool_calls:
                    with st.expander(f"Invoking: {tc['name']}"):
                        st.code(json.dumps(tc["args"], indent=2), language="json")
                    result_msg = results_by_id.get(tc["id"])
                    if result_msg:
                        media = get_media_content_from_tool_result(result_msg.content)
                        if media:
                            media_items.append(media)
                        else:
                            with st.expander(f"Result: {result_msg.name}"):
                                st.markdown(result_msg.content)
        return media_items

    def _stream_response(self, user_msg):
        """Stream the agent's response for a new user message.

        Sends the user message to the agent and iterates the stream, which
        yields (stream_mode, data) tuples. Two stream modes are used:

        - "updates": node-level updates containing complete tool invocations
          (from the "model" node) and tool results (from the "tools" node).
        - "messages": token-level chunks containing incremental reasoning
          and text content for real-time rendering.

        Delegates to _handle_stream_updates and _handle_stream_message for
        each mode, then calls _finalize_stream to collapse any open status
        widgets.

        Args:
            user_msg: The user's input text.
        """
        cur_user_msg = {
            "role": "user",
            "content": [{"type": "text", "text": user_msg}],
        }
        result = self.agent.stream(
            {"messages": [cur_user_msg]},
            config={"configurable": {"thread_id": self.thread_id}},
            context={"sys_prompt_replace_dict": self.replacement_dict},
            stream_mode=["messages", "updates"],
        )

        ss = _StreamState()

        try:
            for stream_mode, cur_data in result:
                if stream_mode == "updates":
                    self._handle_stream_updates(ss, cur_data)
                elif stream_mode == "messages":
                    self._handle_stream_message(ss, cur_data)
        finally:
            self._finalize_stream(ss)

    def _handle_stream_updates(self, ss, cur_data):
        """Dispatch a node-level update to the appropriate handler.

        Routes "model" node outputs (which may contain tool_calls) to
        _handle_tool_invocations and "tools" node outputs (which contain
        ToolMessages) to _handle_tool_results.

        Args:
            ss: The current _StreamState.
            cur_data: Dict mapping node names to their output dicts.
        """
        for node_name, node_output in cur_data.items():
            if node_name == "model":
                self._handle_tool_invocations(ss, node_output)
            if node_name == "tools":
                self._handle_tool_results(ss, node_output)

    def _handle_tool_invocations(self, ss, node_output):
        """Display tool invocations from the model's AIMessage.

        Extracts tool_calls from the model's output messages and renders each
        as an expander inside the shared tools_status container. The expander
        shows the tool name and its arguments as pretty-printed JSON. Lazily
        creates the tools_status st.status widget on the first invocation.

        Args:
            ss: The current _StreamState.
            node_output: The "model" node's output dict containing a
                "messages" key with a list of AIMessage objects.
        """
        model_msgs = node_output.get("messages", [])
        for model_msg in model_msgs:
            for tc in getattr(model_msg, "tool_calls", []):
                if ss.response_container is not None and ss.tools_container is None:
                    if ss.thinking_status is not None:
                        ss.thinking_status.update(
                            label="Thinking complete", state="complete", expanded=False
                        )
                        ss.thinking_status = None
                        ss.thinking_placeholder = None
                    ss.response_container = None
                    ss.response_placeholder = None
                if ss.tools_container is None:
                    ss.tools_container = st.chat_message("assistant", avatar="🔧")
                    ss.tools_status = ss.tools_container.status(
                        "Using tools...", expanded=True
                    )
                ss.tool_call_count += 1
                with ss.tools_status:
                    with st.expander(f"Invoking: {tc['name']}"):
                        st.code(json.dumps(tc["args"], indent=2), language="json")

    def _handle_tool_results(self, ss, node_output):
        """Display tool results inside the shared tools_status container.

        Renders each ToolMessage as an expander showing the tool name and its
        content. Lazily creates the tools_status st.status widget if it
        doesn't already exist (edge case where results arrive without a
        preceding invocation update).

        Args:
            ss: The current _StreamState.
            node_output: The "tools" node's output dict containing a
                "messages" key with a list of ToolMessage objects.
        """
        tool_msgs = node_output.get("messages", [])
        for tool_msg in tool_msgs:
            if ss.tools_container is None:
                ss.tools_container = st.chat_message("assistant", avatar="🔧")
                ss.tools_status = ss.tools_container.status(
                    "Using tools...", expanded=True
                )
            with ss.tools_status:
                media = get_media_content_from_tool_result(tool_msg.content)
                if media:
                    ss.pending_media.append(media)
                else:
                    with st.expander(f"Result: {tool_msg.name}"):
                        st.markdown(tool_msg.content)

    def _handle_stream_message(self, ss, cur_data):
        """Route a token-level message chunk to the appropriate handler.

        Only processes tokens from the "model" node. Each token may carry
        reasoning_content (handled by _handle_thinking_token) and/or text
        content blocks (handled by _handle_text_token). Both are checked
        independently since a single token can contain both.

        Args:
            ss: The current _StreamState.
            cur_data: A (token, metadata) tuple from the "messages" stream.
        """
        token, metadata = cur_data
        node = metadata.get("langgraph_node")
        if node == "model":
            self._handle_thinking_token(ss, token)
            self._handle_text_token(ss, token)

    def _handle_thinking_token(self, ss, token):
        """Append reasoning content to the thinking display.

        If the token carries reasoning_content in additional_kwargs, appends
        it to the accumulated thinking text and re-renders. Lazily creates the
        thinking st.status widget (expanded) and its inner st.empty placeholder
        on the first reasoning token.

        No-ops if the token has no reasoning_content or if it is empty.

        Args:
            ss: The current _StreamState.
            token: An AIMessageChunk from the "messages" stream.
        """
        if "reasoning_content" not in token.additional_kwargs:
            return
        text_content = token.additional_kwargs["reasoning_content"]
        if not text_content:
            return
        if ss.thinking_status is None:
            if ss.tools_container is not None:
                ss.tools_status.update(
                    label=ss.tools_label(), state="complete", expanded=False
                )
                ss.tools_container = None
            if ss.response_container is None:
                ss.response_container = st.chat_message("assistant", avatar="✨")
            ss.thinking_status = ss.response_container.status(
                "Thinking...", expanded=True
            )
            ss.thinking_placeholder = ss.thinking_status.empty()
        ss.full_thinking += text_content
        ss.thinking_placeholder.markdown(ss.full_thinking)

    def _handle_text_token(self, ss, token):
        """Append response text to the main response display.

        Extracts the last content block from the token and, if it is a text
        block with non-empty content, appends it to the accumulated response.
        On the first text token, collapses any open thinking or tools status
        widgets and creates the response st.empty placeholder.

        No-ops if the token has no content blocks, the last block is not text,
        or the text is empty.

        Args:
            ss: The current _StreamState.
            token: An AIMessageChunk from the "messages" stream.
        """
        if len(token.content_blocks) == 0:
            return
        block = token.content_blocks[-1]
        if block.get("type") != "text":
            return
        text_content = block["text"]
        if not text_content:
            return
        if ss.thinking_status is not None and ss.response_placeholder is None:
            ss.thinking_status.update(
                label="Thinking complete", state="complete", expanded=False
            )
        if ss.tools_status is not None and ss.response_placeholder is None:
            ss.tools_status.update(
                label=ss.tools_label(), state="complete", expanded=False
            )
        if ss.tools_container is not None and ss.response_placeholder is None:
            ss.tools_container = None
        if ss.response_placeholder is None:
            if ss.response_container is None:
                ss.response_container = st.chat_message("assistant", avatar="✨")
            ss.response_placeholder = ss.response_container.empty()
        ss.full_response += text_content
        ss.response_placeholder.markdown(ss.full_response)

    def _finalize_stream(self, ss):
        """Collapse any status widgets that are still open after streaming ends.

        Handles two edge cases:
        - The model produced only reasoning with no response text: collapses
          the thinking status.
        - Tools were invoked but the model produced no response text (e.g. the
          answer was the tool result itself): collapses the tools status.

        Also exits any open st.chat_message contexts (tools_container and
        response_container) that were entered during streaming.

        Safe to call unconditionally; checks for None before updating.

        Args:
            ss: The current _StreamState.
        """
        if ss.thinking_status is not None and ss.response_placeholder is None:
            ss.thinking_status.update(
                label="Thinking complete", state="complete", expanded=False
            )
        if ss.tools_status is not None:
            ss.tools_status.update(
                label=ss.tools_label(), state="complete", expanded=False
            )
        if ss.pending_media:
            if ss.response_container is None:
                with st.chat_message("assistant", avatar="✨"):
                    for media in ss.pending_media:
                        render_media_content(media)
            else:
                with ss.response_container:
                    for media in ss.pending_media:
                        render_media_content(media)
        if ss.tools_container is not None:
            ss.tools_container = None
        if ss.response_container is not None:
            ss.response_container = None

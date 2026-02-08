# StreamlitUILangChain

Streamlit chat UI backed by a LangGraph agent. Supports multiple LLM providers (Ollama, AWS Bedrock) and streaming with thinking-token display.

## Running

```bash
conda activate st_lc_env
streamlit run src/app.py
```

Environment is managed via Conda (`environment.yml`, env name `st_lc_env`). Python 3.13.

## Project Structure

```
src/
├── app.py                           # Streamlit entry point
├── llm/
│   ├── ollama.py                    # ChatOllama wrapper (default: gpt-oss:20b, reasoning="high")
│   └── bedrock.py                   # ChatBedrockConverse wrapper (requires model_id)
├── agent/
│   ├── basic_agent.py               # build_agent() — LangGraph agent with middleware
│   └── agent_response_structure.py  # RESPONSE_PROMPT — XML instructions for next_interaction widgets
├── ui/
│   ├── st_langgraph_ui_connector.py # Streamlit UI + LangGraph streaming + thinking tokens
│   └── next_interaction.py          # Parse, strip, and render <next_interaction> XML as widgets
└── tools/
    └── tool_do_date_math.py         # Date math tool + today_xml() helper
```

## Architecture

### LLM layer (`src/llm/`)
- `ollama.get_model()` — wraps `ChatOllama`. Default model `gpt-oss:20b`, temp 0.5, `reasoning="high"`.
- `bedrock.get_model()` — wraps `ChatBedrockConverse`. Requires explicit `model_id`. Has `max_tokens` param (default 8192).

### Agent (`src/agent/`)
- `basic_agent.py`: `build_agent(llm_model, system_prompt, tools, state_schema, checkpointer, include_anthropic_caching)`
- Uses `langchain.agents.create_agent` with a `@dynamic_prompt` middleware that replaces placeholders (e.g. `[[DATE]]`) in the system prompt at runtime using `request.runtime.context`.
- State schema `AgentStateSchema` has `messages` (with `add_messages` reducer) and `sys_prompt_replace_dict`.
- Default checkpointer: `InMemorySaver()` (in-memory, resets on restart).
- `include_anthropic_caching=True` appends a `cachePoint` block to the system message for Anthropic prompt caching.
- `agent_response_structure.py`: `RESPONSE_PROMPT` — appended to the system prompt in `app.py`. Instructs the model to emit `<next_interaction>` XML at the end of responses to drive interactive widgets (yes/no buttons, radio boxes, multi-select checkboxes). XML stays in the checkpoint so the model sees its own prior usage pattern.

### UI connector (`src/ui/st_langgraph_ui_connector.py`)
- `StLanggraphUIConnector(agent, replacement_dict)` — manages chat display and streaming.
- `display_chat()` — renders history then handles new user input. On non-streaming reruns, checks `st.session_state["_next_interaction"]` (saved during streaming) then falls back to what `_display_history()` parsed.
- `_stream_response(user_msg)` — streams agent output with containers for thinking, tools, and response. Shows `st.status("Generating suggestions...")` when `<next_interaction>` XML is being streamed (detected when stripped display text is shorter than raw text).
- `_finalize_stream(ss)` — collapses status widgets, parses `<next_interaction>` XML from raw response into `ss.next_interaction`, saved to `st.session_state` to survive the rerun.
- `_display_history()` — replays messages from agent state. Returns `(last_next_interaction, message_count)`. Parses `<next_interaction>` XML from AI messages (handles pre-existing messages that still have XML in state).
- `new_thread()` — clears `_next_interaction` from session state before rerun.
- Streams via `agent.stream()` with `stream_mode=["messages", "updates"]`. Filters for `node == "model"`.
- Context dict passed via `context={"sys_prompt_replace_dict": ...}` on each stream call.

### Next interaction widgets (`src/ui/next_interaction.py`)
- `parse_next_interaction(raw_text)` — extracts `<next_interaction>` XML from a response, returns `(clean_text, NextInteraction)`. Maps legacy `dropdown_box` → `radio_box`.
- `strip_next_interaction_for_streaming(raw_text)` — removes XML or partial tag prefixes from display text during streaming.
- `render_next_interaction(next_interaction, default_prompt, message_count)` — renders the appropriate Streamlit widget (yes/no buttons, radio, multi-select checkboxes). Always shows `st.chat_input` as a free-text fallback alongside widgets. Uses `message_count` in widget keys to prevent collisions across turns.

### Tools (`src/tools/`)
- `do_date_math` — adds/subtracts day/week/month/year intervals from a date. Uses `dateutil.relativedelta`.
- `today_xml()` — returns today's date in XML format for the system prompt `[[DATE]]` placeholder.

## Key Dependencies
- `streamlit`, `langchain`, `langchain-core`, `langgraph`, `langsmith`
- Provider packages: `langchain-ollama`, `langchain-aws`, `langchain-openai`, `langchain-anthropic`
- `boto3` (for Bedrock), `dateutil` (for date math tool)

## Conventions
- Avatars: `👤` for user, `✨` for assistant.
- System prompt placeholders use double-bracket syntax: `[[PLACEHOLDER]]`.
- `app.py` stores the `StLanggraphUIConnector` in `st.session_state.ui_connector` (created once per session).
- Do not use `continue` in loops. Use `if/else` branching instead.

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
│   └── basic_agent.py               # build_agent() — LangGraph agent with middleware
├── ui/
│   └── st_langgraph_ui_connector.py # Streamlit UI + LangGraph streaming + thinking tokens
└── tools/
    └── tool_do_date_math.py         # Date math tool + today_xml() helper
```

## Architecture

### LLM layer (`src/llm/`)
- `ollama.get_model()` — wraps `ChatOllama`. Default model `gpt-oss:20b`, temp 0.5, `reasoning="high"`.
- `bedrock.get_model()` — wraps `ChatBedrockConverse`. Requires explicit `model_id`. Has `max_tokens` param (default 8192).

### Agent (`src/agent/basic_agent.py`)
- `build_agent(llm_model, system_prompt, tools, state_schema, checkpointer, include_anthropic_caching)`
- Uses `langchain.agents.create_agent` with a `@dynamic_prompt` middleware that replaces placeholders (e.g. `[[DATE]]`) in the system prompt at runtime using `request.runtime.context`.
- State schema `AgentStateSchema` has `messages` (with `add_messages` reducer) and `sys_prompt_replace_dict`.
- Default checkpointer: `InMemorySaver()` (in-memory, resets on restart).
- `include_anthropic_caching=True` appends a `cachePoint` block to the system message for Anthropic prompt caching.

### UI connector (`src/ui/st_langgraph_ui_connector.py`)
- `StLanggraphUIConnector(agent, replacement_dict)` — manages chat display and streaming.
- `display_chat()` — renders history then handles new user input.
- `_stream_response(user_msg)` — streams agent output with two containers:
  - `st.status("Thinking...")` for `reasoning_content` (lazily created, collapses when response starts).
  - `st.empty()` for the main response text.
  - Non-thinking models produce no thinking UI elements.
- `_display_history()` — replays messages from agent state. Shows `st.expander("Thinking")` for AI messages that have `reasoning_content` in `additional_kwargs`.
- Streams via `agent.stream()` with `stream_mode=["messages"]`. Filters for `node == "model"`.
- Context dict passed via `context={"sys_prompt_replace_dict": ...}` on each stream call.

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

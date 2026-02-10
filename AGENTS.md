# StreamlitUILangChain

Streamlit chat UI backed by a LangGraph agent. Supports multiple LLM providers (Ollama, Anthropic, OpenAI, AWS Bedrock) and streaming with thinking-token display.

## Running

```bash
conda activate st_lc_env
python -m streamlit run src/app.py
```

Environment is managed via Conda (`environment.yml`, env name `st_lc_env`). Python 3.13.

## Project Structure

```
src/
  app.py                           # Streamlit entry point
  app_config.py                    # Sidebar model configuration + agent setup
  llm/
    ollama.py                      # ChatOllama wrapper (default: qwen3:8b, reasoning="low")
    anthropic.py                   # ChatAnthropic wrapper
    openai.py                      # ChatOpenAI wrapper
    bedrock.py                     # ChatBedrockConverse wrapper (requires model_id)
  agent/
    basic_agent.py                 # build_agent() - LangGraph agent with middleware
    agent_response_structure.py    # RESPONSE_PROMPT - XML instructions for next_interaction widgets
  ui/
    st_langgraph_ui_connector.py   # Streamlit UI + LangGraph streaming + thinking tokens
    next_interaction.py            # Parse, strip, and render <next_interaction> XML as widgets
    token_usage.py                 # Token usage extraction helpers
  pages/
    Token_Usage_Details.py         # Per-invocation token usage chart
  tools/
    tool_do_date_math.py           # Date math tool + today_xml() helper
    tool_show_media.py             # Media rendering tool helper
```

## Architecture

### LLM layer (`src/llm/`)
- `ollama.get_model()` - wraps `ChatOllama`. Default model `qwen3:8b`, temp 0.5, `reasoning="low"`.
- `anthropic.get_model()` - wraps `ChatAnthropic`. Requires explicit `model_id`. Has `max_tokens` param (default 8192).
- `openai.get_model()` - wraps `ChatOpenAI`. Default model `gpt-5.2`, `reasoning_effort="low"`.
- `bedrock.get_model()` - wraps `ChatBedrockConverse`. Requires explicit `model_id`. Has `max_tokens` param (default 8192).

### Agent (`src/agent/`)
- `basic_agent.py`: `build_agent(llm_model, system_prompt, tools, state_schema, checkpointer, caching_strategy)`
- Uses `langchain.agents.create_agent` with a `@dynamic_prompt` middleware that replaces placeholders (e.g. `[[DATE]]`) in the system prompt at runtime using `request.runtime.context`.
- State schema `AgentStateSchema` has `messages` (with `add_messages` reducer) and `sys_prompt_replace_dict`.
- Default checkpointer: `InMemorySaver()` (in-memory, resets on restart).
- `caching_strategy` supports `"anthropic"` and `"bedrock_anthropic"` for provider prompt caching.
- `agent_response_structure.py`: `RESPONSE_PROMPT` - appended to the system prompt when widgets are enabled.

### UI connector (`src/ui/st_langgraph_ui_connector.py`)
- `StLanggraphUIConnector(agent, replacement_dict)` manages chat display and streaming.
- `display_chat()` renders history then handles new user input.
- `_stream_response(user_msg)` streams agent output with containers for thinking, tools, and response.
- `_finalize_stream(ss)` collapses status widgets and parses `<next_interaction>` XML.
- Token usage summary appears in the sidebar; detailed per-invocation chart lives in `pages/Token_Usage_Details.py`.
- Supports user image attachments (multiple files).
- Note: Streamlit charts should use `width="stretch"` instead of `use_container_width=True` (deprecated after 2025-12-31).

### Next interaction widgets (`src/ui/next_interaction.py`)
- `parse_next_interaction(raw_text)` extracts `<next_interaction>` XML from a response, returns `(clean_text, NextInteraction)`. Maps legacy `dropdown_box` -> `radio_box`.
- `strip_next_interaction_for_streaming(raw_text)` removes XML or partial tag prefixes from display text during streaming.
- `render_next_interaction(next_interaction, default_prompt, message_count)` renders the appropriate Streamlit widget and always shows `st.chat_input` as a free-text fallback.

### Tools (`src/tools/`)
- `do_date_math` adds/subtracts day/week/month/year intervals from a date. Uses `dateutil.relativedelta`.
- `today_xml()` returns today's date in XML format for the system prompt `[[DATE]]` placeholder.
- `tool_show_media` renders image/audio/video links in the chat UI.

## Key Dependencies
- `streamlit`, `langchain`, `langchain-core`, `langgraph`, `langsmith`
- Provider packages: `langchain-ollama`, `langchain-aws`, `langchain-openai`, `langchain-anthropic`
- `boto3` (for Bedrock), `dateutil` (for date math tool), `altair` (token usage charts)

## Conventions
- Avatars: `👤` for user, `✨` for assistant, `🔧` for tool calls.
- System prompt placeholders use double-bracket syntax: `[[PLACEHOLDER]]`.
- `app.py` stores the `StLanggraphUIConnector` in `st.session_state.ui_connector` (created once per session).
- Do not use `continue` in loops. Use `if/else` branching instead.

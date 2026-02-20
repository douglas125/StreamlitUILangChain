# Streamlit UI for LangChain Agents

Drop-in Streamlit chat interface for any LangGraph agent. Handles streaming, thinking tokens, tool calls, and interactive follow-up widgets -- so you just wire up your model and go.

## Features

- Token-by-token streaming with thinking/reasoning display
- Tool call visualization (invocations + results, grouped and collapsible)
- Built-in tools for date math, media rendering, SQL queries over CSV files, and static CSV chart images
- Interactive follow-up widgets (yes/no buttons, radio, multi-select checkboxes)
- Free-text input always available alongside widgets
- System prompt placeholder replacement at runtime
- Token usage summary (demo app) with detailed per-invocation stats

## Quick Start

```python
import streamlit as st
from langchain_ollama import ChatOllama
from src.agent.basic_agent import build_agent
from src.agent import agent_response_structure
from src.ui.st_langgraph_ui_connector import StLanggraphUIConnector

st.title("My Chat App")

if "ui_connector" not in st.session_state:
    # 1. Create your model
    llm = ChatOllama(model="qwen3:8b", temperature=0.5)

    # 2. Build a LangGraph agent (append RESPONSE_PROMPT for interactive widgets)
    agent = build_agent(
        llm,
        system_prompt="You are a helpful assistant." + agent_response_structure.RESPONSE_PROMPT,
        tools=[],  # add your tools here
    )

    # 3. Initialize the UI connector
    st.session_state.ui_connector = StLanggraphUIConnector(agent)

# 4. Render the chat
st.session_state.ui_connector.display_chat()
```

That's it. The connector handles streaming, history replay, and widget rendering.

## Optional: Prompt Placeholders

Replace parts of your system prompt at runtime (e.g. injecting today's date):

```python
replacement_dict = {"[[DATE]]": "<today>2025-01-15</today>"}
st.session_state.ui_connector = StLanggraphUIConnector(
    agent, replacement_dict=replacement_dict
)
```

## Optional: New Chat Button

```python
with st.sidebar:
    if st.button("New Chat"):
        st.session_state.ui_connector.new_thread()
```

## Run the Demo

```bash
conda activate st_lc_env
python -m streamlit run src/app.py
```


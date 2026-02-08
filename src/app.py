import streamlit as st

from src.llm.ollama import get_model

# from src.llm.bedrock import get_model

from src.agent.basic_agent import build_agent
from src.ui.st_langgraph_ui_connector import StLanggraphUIConnector

from src.tools.tool_do_date_math import today_xml
from src.tools.tool_do_date_math import do_date_math
from src.tools.tool_show_media import tool_show_media

from src.agent import agent_response_structure


def main():
    st.set_page_config(
        page_title="Chat interface for Streamlit using Langgraph",
        page_icon="🧊",
    )
    st.title("Chat interface for Streamlit using Langgraph")

    # create agent and ui connector once
    if "ui_connector" not in st.session_state:
        llm = get_model()
        all_tools = [do_date_math, tool_show_media]
        agent = build_agent(
            llm,
            system_prompt="You are a helpful assistant."
            + agent_response_structure.RESPONSE_PROMPT,
            tools=all_tools,
            caching_strategy="",  # "bedrock_anthropic",
        )
        # agent = build_agent(llm, tools=all_tools, caching_strategy="bedrock_anthropic")
        # agent = build_agent(llm, tools=all_tools, caching_strategy="anthropic")
        replacement_dict = {"[[DATE]]": today_xml()}
        st.session_state.ui_connector = StLanggraphUIConnector(
            agent, replacement_dict=replacement_dict
        )

    with st.sidebar:
        if st.button("New Chat"):
            st.session_state.ui_connector.new_thread()
    st.session_state.ui_connector.display_chat()


if __name__ == "__main__":
    main()

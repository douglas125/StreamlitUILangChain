import streamlit as st

from src.llm.ollama import get_model
from src.agent.basic_agent import build_agent
from src.ui.st_langgraph_ui_connector import StLanggraphUIConnector

from src.tools.tool_do_date_math import today_xml
from src.tools.tool_do_date_math import do_date_math


def main():
    st.set_page_config(
        page_title="Chat interface for Streamlit using Langgraph",
        page_icon="🧊",
    )
    st.title("Chat interface for Streamlit using Langgraph")

    # create agent and ui connector once
    if "ui_connector" not in st.session_state:
        # llm = get_model(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
        llm = get_model()
        agent = build_agent(llm, tools=[do_date_math], include_anthropic_caching=False)
        replacement_dict = {
            "[[DATE]]": today_xml()
        }
        st.session_state.ui_connector = StLanggraphUIConnector(agent, replacement_dict=replacement_dict)

    st.session_state.ui_connector.display_chat()


if __name__ == "__main__":
    main()

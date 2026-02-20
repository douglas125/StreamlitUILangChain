import streamlit as st

from src.app_config import build_sidebar_config
from src.app_config import build_ui_connector
from src.app_config import reset_stream_state
from src.app_config import should_rebuild_connector

APP_RUNTIME_VERSION = "2026-02-20-csv-plot-static-image-fix-1"


def main():
    st.set_page_config(
        page_title="Chat interface for Streamlit using Langgraph",
        page_icon="Chat",
    )
    st.title("Chat interface for Streamlit using Langgraph")

    llm_config = build_sidebar_config()
    with st.sidebar:
        with st.expander("Model Config", expanded=False):
            system_prompt = st.text_area(
                "System prompt",
                value=st.session_state.get(
                    "system_prompt_text", "You are a helpful assistant."
                ),
                key="system_prompt_text",
                help="This is passed as the system prompt to the model.",
            )
            enable_widgets = st.checkbox(
                "Enable widget capabilities",
                value=st.session_state.get("enable_widgets", True),
                key="enable_widgets",
            )
            enable_date_math = st.checkbox(
                "Enable date math tool",
                value=st.session_state.get("enable_date_math", True),
                key="enable_date_math",
            )
            enable_media_tool = st.checkbox(
                "Enable media tool",
                value=st.session_state.get("enable_media_tool", True),
                key="enable_media_tool",
            )
            enable_csv_sql_tool = st.checkbox(
                "Enable CSV SQL tool",
                value=st.session_state.get("enable_csv_sql_tool", True),
                key="enable_csv_sql_tool",
            )
            enable_csv_plot_tool = st.checkbox(
                "Enable CSV plot tool",
                value=st.session_state.get("enable_csv_plot_tool", True),
                key="enable_csv_plot_tool",
            )
    llm_config["system_prompt"] = system_prompt
    llm_config["enable_widgets"] = enable_widgets
    llm_config["enable_date_math"] = enable_date_math
    llm_config["enable_media_tool"] = enable_media_tool
    llm_config["enable_csv_sql_tool"] = enable_csv_sql_tool
    llm_config["enable_csv_plot_tool"] = enable_csv_plot_tool
    llm_config["app_runtime_version"] = APP_RUNTIME_VERSION
    if should_rebuild_connector(llm_config):
        st.session_state.ui_connector = build_ui_connector(llm_config)
        reset_stream_state()

    with st.sidebar:
        if st.button("New Chat"):
            st.session_state.ui_connector.new_thread()
        st.session_state.ui_connector.render_sidebar_token_usage()
    st.session_state.ui_connector.display_chat()


if __name__ == "__main__":
    main()

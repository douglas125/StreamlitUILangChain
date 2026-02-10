import streamlit as st

from src.llm.ollama import get_model as get_ollama_model
from src.llm.anthropic import get_model as get_anthropic_model
from src.llm.bedrock import get_model as get_bedrock_model
from src.llm.openai import get_model as get_openai_model

from src.agent.basic_agent import build_agent
from src.ui.st_langgraph_ui_connector import StLanggraphUIConnector

from src.tools.tool_do_date_math import today_xml
from src.tools.tool_do_date_math import do_date_math
from src.tools.tool_show_media import tool_show_media

from src.agent import agent_response_structure


def build_sidebar_config():
    with st.sidebar:
        with st.expander("Model Settings", expanded=False):
            provider = st.selectbox(
                "Provider",
                ["Ollama", "Anthropic", "OpenAI", "Bedrock"],
                index=0,
                key="llm_provider",
            )
            if provider == "Ollama":
                model_id = st.text_input(
                    "Model ID", value="qwen3:8b", key="ollama_model_id"
                )
                image_support = st.checkbox(
                    "Image support",
                    value=st.session_state.get("ollama_image_support", False),
                    key="ollama_image_support",
                )
                temperature = st.number_input(
                    "Temperature", min_value=0.0, max_value=2.0, value=0.5, step=0.1
                )
                top_p = st.number_input(
                    "Top p", min_value=0.0, max_value=1.0, value=0.9, step=0.05
                )
                top_k = st.number_input(
                    "Top k", min_value=1, max_value=200, value=40, step=1
                )
                reasoning_option = st.selectbox(
                    "Reasoning", ["none", "low", "medium", "high"], index=0
                )
                reasoning = None if reasoning_option == "none" else reasoning_option
                caching_strategy = ""
                return {
                    "provider": provider,
                    "model_id": model_id,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "reasoning": reasoning,
                    "caching_strategy": caching_strategy,
                    "image_support": image_support,
                }
            if provider == "Anthropic":
                model_id = st.text_input(
                    "Model ID",
                    value="claude-haiku-4-5-20251001",
                    key="anthropic_model_id",
                )
                image_support = st.checkbox(
                    "Image support",
                    value=st.session_state.get("anthropic_image_support", False),
                    key="anthropic_image_support",
                )
                temperature = st.number_input(
                    "Temperature", min_value=0.0, max_value=2.0, value=0.5, step=0.1
                )
                max_tokens = st.number_input(
                    "Max tokens", min_value=1, max_value=200000, value=8192, step=256
                )
                caching_strategy = st.selectbox(
                    "Caching strategy", ["none", "anthropic"], index=1
                )
                return {
                    "provider": provider,
                    "model_id": model_id,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "caching_strategy": ""
                    if caching_strategy == "none"
                    else "anthropic",
                    "image_support": image_support,
                }
            if provider == "OpenAI":
                model_id = st.text_input(
                    "Model ID", value="gpt-5.2", key="openai_model_id"
                )
                image_support = st.checkbox(
                    "Image support",
                    value=st.session_state.get("openai_image_support", False),
                    key="openai_image_support",
                )
                temperature = st.number_input(
                    "Temperature", min_value=0.0, max_value=2.0, value=0.5, step=0.1
                )
                max_tokens = st.number_input(
                    "Max tokens", min_value=1, max_value=200000, value=8192, step=256
                )
                reasoning_effort = st.selectbox(
                    "Reasoning effort", ["low", "medium", "high"], index=0
                )
                return {
                    "provider": provider,
                    "model_id": model_id,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "reasoning_effort": reasoning_effort,
                    "image_support": image_support,
                }
            model_id = st.text_input(
                "Model ID",
                value="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                key="bedrock_model_id",
            )
            image_support = st.checkbox(
                "Image support",
                value=st.session_state.get("bedrock_image_support", False),
                key="bedrock_image_support",
            )
            region_name = st.text_input(
                "Region", value="us-east-1", key="bedrock_region"
            )
            temperature = st.number_input(
                "Temperature", min_value=0.0, max_value=2.0, value=0.5, step=0.1
            )
            max_tokens = st.number_input(
                "Max tokens", min_value=1, max_value=200000, value=8192, step=256
            )
            bedrock_converse = st.checkbox("Use Bedrock Converse", value=True)
            caching_strategy = st.selectbox(
                "Caching strategy", ["none", "bedrock_anthropic"], index=1
            )
            return {
                "provider": provider,
                "model_id": model_id,
                "region_name": region_name,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "bedrock_converse": bedrock_converse,
                "caching_strategy": ""
                if caching_strategy == "none"
                else "bedrock_anthropic",
                "image_support": image_support,
            }


def should_rebuild_connector(llm_config):
    current = st.session_state.get("llm_config")
    if current is None:
        st.session_state["llm_config"] = llm_config
        return True
    if current != llm_config:
        st.session_state["llm_config"] = llm_config
        return True
    return "ui_connector" not in st.session_state


def build_ui_connector(llm_config):
    provider = llm_config["provider"]
    if provider == "Ollama":
        llm = get_ollama_model(
            model_id=llm_config["model_id"],
            temperature=llm_config["temperature"],
            reasoning=llm_config["reasoning"],
            top_k=llm_config["top_k"],
            top_p=llm_config["top_p"],
        )
    elif provider == "Anthropic":
        llm = get_anthropic_model(
            model_id=llm_config["model_id"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
        )
    elif provider == "OpenAI":
        llm = get_openai_model(
            model_id=llm_config["model_id"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            reasoning_effort=llm_config["reasoning_effort"],
        )
    else:
        llm = get_bedrock_model(
            model_id=llm_config["model_id"],
            region_name=llm_config["region_name"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            bedrock_converse=llm_config["bedrock_converse"],
        )
    all_tools = []
    if llm_config.get("enable_date_math", True):
        all_tools.append(do_date_math)
    if llm_config.get("enable_media_tool", True):
        all_tools.append(tool_show_media)
    base_prompt = llm_config.get("system_prompt", "You are a helpful assistant.")
    if llm_config.get("enable_widgets", True):
        base_prompt += agent_response_structure.RESPONSE_PROMPT
    agent = build_agent(
        llm,
        system_prompt=base_prompt,
        tools=all_tools,
        caching_strategy=llm_config.get("caching_strategy", ""),
    )
    replacement_dict = {"[[DATE]]": today_xml()}
    return StLanggraphUIConnector(
        agent,
        replacement_dict=replacement_dict,
        enable_image_uploads=llm_config.get("image_support", False),
    )


def reset_stream_state():
    st.session_state.pop("_streaming", None)
    st.session_state.pop("_pending_msg", None)
    st.session_state.pop("_stream_error", None)
    st.session_state.pop("_next_interaction", None)
    st.session_state.pop("_next_interaction_parse_error", None)

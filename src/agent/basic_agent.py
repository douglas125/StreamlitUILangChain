from typing import Annotated
from typing import TypedDict

from langgraph.graph.message import add_messages

from langchain.agents import create_agent
from langchain.messages import SystemMessage
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware import dynamic_prompt

from langgraph.checkpoint.memory import InMemorySaver


class AgentStateSchema(TypedDict):
    messages: Annotated[list, add_messages]
    sys_prompt_replace_dict: dict


def build_agent(
    llm_model,
    system_prompt="You are a helpful assistant. Today is [[DATE]].",
    tools=[],
    state_schema=AgentStateSchema,
    checkpointer=None,
    include_anthropic_caching=True,
):
    @dynamic_prompt
    def replace_sys_prompt_placeholders(request: ModelRequest) -> str:
        """Replaces placeholders in the system prompt according to information in the runtime."""
        replace_dict = request.runtime.context.get("sys_prompt_replace_dict", {})
        base_prompt = system_prompt
        for k in replace_dict:
            base_prompt = base_prompt.replace(k, replace_dict[k])

        sys_msg_content = [
            {
                "type": "text",
                "text": base_prompt,
            },
        ]
        if include_anthropic_caching:
            sys_msg_content.append({"cachePoint": {"type": "default"}})

        new_prompt = SystemMessage(content=sys_msg_content)
        return new_prompt

    if checkpointer is None:
        checkpointer = InMemorySaver()

    agent = create_agent(
        model=llm_model,
        tools=tools,
        middleware=[replace_sys_prompt_placeholders],
        state_schema=state_schema,
        checkpointer=checkpointer,
    )
    return agent

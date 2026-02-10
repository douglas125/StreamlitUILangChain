from langchain_openai import ChatOpenAI


def get_model(
    model_id="gpt-5.2",
    temperature=0.5,
    max_tokens=8192,
    reasoning_effort="low",
):
    model = ChatOpenAI(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    return model

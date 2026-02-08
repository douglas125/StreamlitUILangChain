from langchain_anthropic import ChatAnthropic


def get_model(model_id="claude-haiku-4-5-20251001", temperature=0.5, max_tokens=8192):
    model = ChatAnthropic(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        # timeout=,
        # max_retries=,
        # ...
    )
    return model

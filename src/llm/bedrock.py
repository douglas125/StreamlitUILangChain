from langchain_aws import ChatBedrockConverse
from langchain_aws import ChatBedrock


def get_model(
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=8192,
    bedrock_converse=True,
):
    assert model_id is not None, "Please provide a model id"
    fn = ChatBedrockConverse if bedrock_converse else ChatBedrock
    model = fn(
        model_id=model_id,
        region_name=region_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return model

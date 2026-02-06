from langchain_aws import ChatBedrockConverse


def get_model(model_id=None, region_name="us-east-1", temperature=0.5, max_tokens=8192):
    assert model_id is not None, "Please provide a model id"
    model = ChatBedrockConverse(
        model_id=model_id,
        region_name=region_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return model

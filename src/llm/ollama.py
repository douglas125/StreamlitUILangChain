from langchain_ollama import ChatOllama


def get_model(
    model_id="gpt-oss:20b", temperature=0.5, reasoning="low", top_k=40, top_p=0.9
):
    assert model_id is not None, "Please provide a model id"
    model = ChatOllama(
        model=model_id,
        temperature=temperature,
        reasoning=reasoning,
        top_k=top_k,
        top_p=top_p,
    )
    return model

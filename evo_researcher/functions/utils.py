import tiktoken


def trim_to_n_tokens(content: str, n: int, model: str) -> str:
    encoder = tiktoken.encoding_for_model(model)
    return encoder.decode(encoder.encode(content)[:n])

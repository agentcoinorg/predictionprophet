import tiktoken


def trim_to_n_tokens(content: str, n: int, model: str, trim_batch_size: int = 128) -> str:
    encoder = tiktoken.encoding_for_model(model)
    encoded = encoder.encode(content)
    while len(encoded) > n:
        encoded = encoded[:-trim_batch_size]
    return encoder.decode(encoded)

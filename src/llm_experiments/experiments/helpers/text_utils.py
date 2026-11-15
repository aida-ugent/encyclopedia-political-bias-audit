def truncate_text(text, max_tokens=128000):
    if not isinstance(text, str):
        text = str(text)

    approx_tokens = len(text) // 4

    if approx_tokens > max_tokens:
        print(f"Truncating text from ~{approx_tokens} tokens to {max_tokens}")
        max_chars = max_tokens * 4
        return text[:max_chars]

    return text

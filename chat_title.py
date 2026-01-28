from kgpedia import KGPediaModel

def get_chat_name(user_message: str, assistant_response: str) -> tuple[str, int, int]:
    llm= KGPediaModel().get_model()
    query = f"User: {user_message}\nAssistant: {assistant_response}"
    prompt = f"""Summarize the user and assistant conversation in a few words. Give weightage to user query
    Conversation: {query}
    Instructions:
    - The title should be concise and capture the essence of the user query
    - The title should not be too long, ideally 3 to 5 words
    - The title should be more relevant to the query than the response because chat title is based on user query
    - If the user query is a simple greeting or something not related to the system prompt you can return "General Conversation" as the title
    - Do not return any other character, except the title in strings strictly. It is a serious matter, to keep the title clean and professional.
    """
    title = llm.complete(prompt)

    raw = title.raw
    if raw is None:
        output_tokens = 0
        input_tokens = 0
    else:
        usage = raw.get("usage_metadata", {})
        output_tokens = usage.get("candidates_token_count", 0)
        input_tokens = usage.get("prompt_token_count", 0)

    chat_title = title.text.strip("\n")
    return chat_title, input_tokens, output_tokens
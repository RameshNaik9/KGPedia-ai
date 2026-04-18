def get_tags(history,LLM):
    conversation_history = ""
    for item in range(len(history)):
        message = f"User asked: {history[item].content}" if item % 2 == 0 else f"Assistant replied: {history[item].content}"
        conversation_history += f"{item + 1}. {message}\n"
        
    tags_prompt_template = f"""Task: Generate up to 10 relevant tags via NER from the conversation. Include key entities like people, organizations, locations, products, topics, events, and concepts that summarize the essence.

    Conversation:
    {conversation_history}

    Instructions:
    - Extract and focus on significant entities/topics representing the main content.
    - Return as a comma-separated list (no empties).
    - For greetings/starters, ignore initial messages and return only 'General Conversation'.
    """

    tags = LLM.complete(tags_prompt_template)
    
    raw = tags.raw
    if raw is None:
        output_tokens = 0
        input_tokens = 0
    else:
        usage = raw.get("usage_metadata", {})
        output_tokens = usage.get("candidates_token_count", 0)
        input_tokens = usage.get("prompt_token_count", 0)

    tags.text.replace("\n", ", ")
    #get the tags as a list
    tags_list = tags.text.split(", ")

    #iterate through each word and remove newline characters and spaces before and after the word
    tags_list = [tag.strip() for tag in tags_list]
    
    return tags_list,tags, input_tokens, output_tokens
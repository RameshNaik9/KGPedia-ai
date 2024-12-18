def get_tags(history,LLM):
    conversation_history = ""
    for item in range(len(history)):
        message = f"User asked: {history[item].content}" if item % 2 == 0 else f"Assistant replied: {history[item].content}"
        conversation_history += f"{item + 1}. {message}\n"
        
    tags_prompt_template = f"""Task: Generate a list of relevant tags based on Named Entity Recognition (NER) from the following conversation. 
    The tags should include important people, organizations, locations, products, topics, and any other significant entities or concepts discussed.

    Conversation History:
    {conversation_history}

    Instructions:
    - Extract key entities mentioned in the conversation, including but not limited to:
    - Topic or subject discussed
    - Organizations
    - Locations
    - Important concept
    - Events or key topics
    - Focus on concepts that could summarize or represent the essence of the conversation.
    - Return 10 tags as a simple list, separating each with a comma
    - Don't return empty tags
    - If the conversation is based on starting off a conversation or some kind of greeting, you can ignore the first few messages and focus on the main content. Instead return only one tag, "General Conversation" to indicate the nature of the conversation.
    """

    tags = LLM.complete(tags_prompt_template)

    tags.text.replace("\n", ", ")
    #get the tags as a list
    tags_list = tags.text.split(", ")

    #iterate through each word and remove newline characters and spaces before and after the word
    tags_list = [tag.strip() for tag in tags_list]
    
    return tags_list,tags
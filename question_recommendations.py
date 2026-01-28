# from tags import get_tags

def question_recommendations(history, LLM):
    # if tags is None:
    #     tags, _, _, _ = get_tags(history, LLM)
    #changed by Abhirama, 

    conversation_history = ""
    for i in range(len(history)):
        if i % 2 == 0:
            # For user messages
            user_timestamp = history[i].additional_kwargs["user_timestamp"]
            if i == 0:
                message = f"User asked: {history[i].content}"
            else:
                assistant_timestamp = history[i - 1].additional_kwargs["assistant_timestamp"]
                time_diff = user_timestamp - assistant_timestamp
                message = f"User asked after {time_diff} seconds: {history[i].content}"
        else:
            # For assistant messages
            assistant_timestamp = history[i].additional_kwargs["assistant_timestamp"]
            user_timestamp = history[i - 1].additional_kwargs["user_timestamp"]
            time_diff = assistant_timestamp - user_timestamp
            message = f"Assistant replied in {time_diff} seconds: {history[i].content}"

        conversation_history += f"{i + 1}. {message}\n"

    recommendation_prompt_template = f"""
    Task: Generate 3 relevant, answerable follow-up questions based on the conversation. They should encourage exploration beyond current topics.

    Conversation History:
    {conversation_history}

    Note: The assistant knows general career guidance but lacks specific details.

    Instructions:
    1. Expand on main topics discussed.
    2. Avoid overly detailed or niche questions.
    3. Make each question clear, concise, and within the assistant's knowledge scope.
    4. Generate exactly three questions.
    5. Return as a numbered list: 
    1. First question
    2. Second question
    3. Third question
    Do not add extra text, explanations, or punctuation outside the list.

    Output Format Example:
    1. [First question]
    2. [Second question]
    3. [Third question]
    """

    questions = LLM.complete(recommendation_prompt_template)
    
    raw = questions.raw
    if raw is None:
        output_tokens = 0
        input_tokens = 0
    else:
        usage = raw.get("usage_metadata", {})
        output_tokens = usage.get("candidates_token_count", 0)
        input_tokens = usage.get("prompt_token_count", 0)
        
    questions_list = questions.text.strip().split("\n")
    questions_list = [
        question.split(". ", 1)[1].strip()
        for question in questions_list
        if question.strip()
    ]

    return questions_list, questions, input_tokens, output_tokens

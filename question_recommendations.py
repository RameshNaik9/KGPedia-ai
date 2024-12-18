from tags import get_tags

def question_recommendations(history, LLM):
    tags, _ = get_tags(history, LLM)

    conversation_history = ""
    for i in range(len(history)):
        if i % 2 == 0:
            # For user messages
            user_timestamp = history[i].additional_kwargs["user_timestamp"]
            if i == 0:
                # No previous timestamp to compare for the first message
                message = f"User asked: {history[i].content}"
            else:
                assistant_timestamp = history[i - 1].additional_kwargs[
                    "assistant_timestamp"
                ]
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
    Task: 
    Given a conversation between the user and assistant, generate 3 user queries that follow the flow of the conversation. The questions should be related to popular tags that the current user hasn't explored yet. If a user's current conversation has certain tags, recommend questions related to tags that might co-occur with them.
    
    Conversation History:
    {conversation_history}

    Tags:
    {tags}

    Instructions:
    1. Focus on questions that could summarize or represent the essence of the conversation and deep dive into it.
    2. Generate exactly three questions
    3. Return the questions as a numbered list in the following format:
            [First question]
            [Second question]
            [Third question]
    4. Ensure each question is clear, concise, and directly related to the conversation or unexplored tags.
    5. Do not include any additional text, explanations, or punctuation marks outside of the numbered list.
    
    Output Format Example:
    1. [First question]
    2. [Second question]
    3. [Third question]
    """

    questions = LLM.complete(recommendation_prompt_template)
    questions_list = questions.text.strip().split("\n")
    questions_list = [
        question.split(". ", 1)[1].strip()
        for question in questions_list
        if question.strip()
    ]

    return questions_list, questions

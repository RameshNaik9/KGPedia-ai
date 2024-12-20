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
    Given a conversation between the user and assistant, generate 3 follow-up questions that are relevant and answerable based on the available information. The questions should encourage further exploration without requiring highly detailed knowledge.    
    Conversation History:
    {conversation_history}

    Tags:
    {tags}

    Note: The assistant has knowledge about general career guidance but may not have access to specific or granular details.

    Instructions:
    1. Generate questions that expand on the main topics discussed.
    2. Avoid overly detailed or niche questions that may be unanswerable.
    3. Ensure each question is clear, concise, and within the scope of the assistant's knowledge.
    4. Generate exactly three questions.
    5. Return the questions as a numbered list in the following format:
        1. First question
        2. Second question
        3. Third question
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

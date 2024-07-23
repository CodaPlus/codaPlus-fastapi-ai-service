from langchain.schema import HumanMessage
from base_model.custom_wrapper import LangchainLLMWrapper


def suggest_remind_response(last_response):
    template = """
    Hi there,
    Just a friendly reminder of our previous message.
    """

    prompt = f"""
        Only paraphrase again your questions in {last_response}. DO NOT duplicate and remove all greeting sentences. 
        MUST SAY we'll watch out your reply.
    """

    is_chat_model, _, llm_model = LangchainLLMWrapper.load_llm_model()
    if is_chat_model:
        response = llm_model.generate([[HumanMessage(content=prompt)]])
    else:
        response = llm_model.generate([prompt])
    message = response.generations[0][0].text

    return template + message

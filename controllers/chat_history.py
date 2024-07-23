from langchain.chains.llm import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from DB.redis import r, redis_cache
from LLM import DEFAULT_MODEL
from langchain.globals import set_llm_cache
from datetime import datetime
import pytz
from base_model import remind_response, utils
from controllers.utils.chats_controller import get_document_array, append_to_array
from controllers.utils.problems_controller import retrieve_question
from base_model.RAG.data_retrivel import DataRetriever


def context_aware_conversation(user_input: str,  UUID: str, user_name: str,  SESSION_ID: str = None, question_id=None):
    chat_history = get_document_array(SESSION_ID, UUID)
    empty_array = False

    if utils.check_hello(user_input):
        chat_history = []
        empty_array = True

    if utils.check_need_to_remind(user_input) and chat_history is not None and SESSION_ID is not None:
        return {
            "response": remind_response.suggest_remind_response(chat_history[-1]),
            "session_id": str(SESSION_ID)
        }
    
    prompt = ChatPromptTemplate.from_template("""
        As a Coding Guru, focus on answering "{user_name}"'s query using the provided context. The details encapsulated within the <context> HTML tags originate from a knowledge base and previous user conversations.                         
        <context>
        User conversation history: "{context}"
        ------
        Coding Challenge given to the user: "{coding_problem}"
        ------     
        Vector Search results for the user given question: "{vector_search_context}"                              
        </context>
        Answer the question: "{input}" directly, sticking to the context. Use the given context to answer the question only, do not output anything outside of the context domain. Avoid assumptions or unrelated information, and always address "{user_name}" by name.
        If the user ask anything outside of the context, please let the user know that you are unable to provide an answer.
    """)
    set_llm_cache(redis_cache)
    vector_search_context = DataRetriever(UUID).retriever(user_input, k=4)

    chain = LLMChain(llm=DEFAULT_MODEL, prompt=prompt)
    response = chain.invoke({"context": str(chat_history[:4]), "input": user_input, "user_name": user_name, "coding_problem": retrieve_question(UUID, question_id), "vector_search_context": vector_search_context})

    _id = append_to_array(SESSION_ID, {
        "dateTime": datetime.now(pytz.timezone('UTC')),
        "User": user_input,
        "Agent": response['text']
    }, UUID, empty_array)

    return {
        "response": response['text'],
        "session_id": str(_id)
    }

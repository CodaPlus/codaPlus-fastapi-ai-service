from langchain.chains.llm import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from DB.redis import redis_cache
from LLM import FINETUNED_MODEL
from langchain.globals import set_llm_cache
import json


def generate_coding_solution(user_input: dict, UUID: str):

    template_ = """
    You are an assistant tasked with responding to user inquiries using the information provided within the specified context.
    Anything between the following /`context/`  html blocks is retrieved from a knowledge bank of user profile data vector database, not part of the conversation with the user.
    <context>
        question title: {title}
        coding question: {problem}
        code template: {code_template}
        filters: 
            question type: {question_type}
            difficulty: {difficulty}
            language: {language}
        test cases: {test_cases}
    <context/>
    Your main goal is to answer the user's coding question that is written in {language}. In order to do so, you need to provide a solution to the problem and ensure that the solution passes all the test cases provided. 
    Please only output the answer to the user's question. Do not output any other information. Make sure to use the context provided to answer the user's question as accurately as possible without fabricating information or making assumptions about the user's question.
    make sure to use the context provided to answer the user's question as accurately as possible without fabricating information or making assumptions about the user's question and please provide a in-depth explanation of the solution.

    GUIDELINES:
    - The solution should be written in the same language as the question.
    - The solution should pass all the test cases provided.
    - The solution should be efficient and optimized.
    - The explanation should be clear and concise.
    - The explanation should be in-depth and cover all aspects of the solution.

    Output the solution and explanation in the following format:
        class CodingSolution(BaseModel):
        solution: str
        explanation: str

    THE OUTPUT SHOULD BE IN THE FORM OF A JSON OBJECT ADHERING TO THE ABOVE SCHEMA.
    """

    set_llm_cache(redis_cache)
    final_prompt = ChatPromptTemplate.from_template(template_)
    final_chain = LLMChain(llm=FINETUNED_MODEL, prompt=final_prompt)
    final_chain_output = final_chain.invoke({"language": user_input.filters.language, "test_cases": user_input.test_cases, "code_template": user_input.code_template, "problem": user_input.problem_description, "title": user_input.question_title, "question_type": user_input.filters.question_type, "difficulty": user_input.filters.difficulty})
    return json.loads(final_chain_output["text"])
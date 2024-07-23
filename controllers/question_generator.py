from langchain.prompts.chat import ChatPromptTemplate
from controllers.utils.init_helepr import get_questions
from DB.mongo import mongoConnection
from datetime import datetime
import asyncio
import pytz


async def generate_coding_problem(user_input: dict, UUID: str):
    #### MONGO STATS ####
    get_problem_connection = mongoConnection.get_collection("GENERATED_USER_PROBLEM").find_one({"UUID": UUID})
    if get_problem_connection is not None:
        if (datetime.now(pytz.timezone('UTC')) - (get_problem_connection["date_time"]).replace(tzinfo=pytz.utc)).days < 1:
            filters = get_problem_connection["questions"]["filters"]
            if filters["question_type"] == user_input.filters.question_type and filters["difficulty"] == user_input.filters.difficulty and filters["language"] == user_input.filters.language:
                return get_problem_connection["questions"]       

    summarized_context = (mongoConnection.get_collection("USER_ANALYSIS").find_one({"UUID": UUID}, {"user_summary": 1, "_id": 0})).get('user_summary')
    
    template_ = f"""
        Your task as an assistant is to create CUSTOM CODING QUESTIONS tailored to the user's preferences and skill levels, based on the provided context. This involves using the context within the specified /`context/` HTML blocks, which contains user profile data from a database. Assume the user is new if no relevant details are provided.

        <context>
            {summarized_context if summarized_context else "No context found"}
        <context/>
        
        TASK OVERVIEW: Develop 1 custom coding question, including at least 4 diverse and comprehensive test cases to ensure the user's solution is correct and efficient.

        QUESTION REQUIREMENTS:
        - Provide a {user_input.filters.question_type} code template relevant to the {user_input.filters.language} programming challenge. Example template for Python:
        "
        class Solution:
            def insertionSortList(self, params):
                pass
        "
        - The template must include a "Solution" class and a method with parameters matching the test cases.

        OUTPUT STRUCTURE: Format your content as a JSON object following the below Pydantic schema CodingProblem and outputs a single object of that type:

        class TestCase(BaseModel):
            inputs: List[Union[str, int, list]]
            outputs: List[Union[int, None]]
            
        class CodingProblem(BaseModel):
            question_title: str
            problem_description: str
            code_template: str
            method_name: str  # Must match the code template
            test_cases: List[TestCase]
        Ensure the JSON object adheres to this format, with DOUBLE QUOTES for strings.
            
        GUIDELINES:
        - Diversity and Engagement: The coding questions should be diverse and engaging, tailored to match the user's programming language expertise in {user_input.filters.language} and interest in the {user_input.filters.question_type}.
        - CODE TEMPLATE FORMAT: Follow the given format closely. Use "\n" for line breaks and "\t" for tabs.
        - Include question Context and EXAMPLES under "Examples:" with correct inputs and outputs for clarity.
        - The challenge level should be {user_input.filters.difficulty}, pushing the user's abilities without being overwhelming.
        - MAKE SURE IT'S FORMATTED TO A VALID PYTHON JSON OBJECT THAT CAN BE DECODED WITH PYTHON WITHOUT ANY ERRORS.

        IMPORTANT: OUTPUT MUST BE A VALID JSON FORMATTED SINGLE OBJECT! DO NOT OUTPUT ANYTHING ELSE!
    """
    final_prompt = ChatPromptTemplate.from_template(template_)

    tasks = [get_questions(final_prompt) for _ in range(int(user_input.number_of_questions))]
    output_response = {
        "questions": await asyncio.gather(*tasks),
        "filters": {
            "question_type": user_input.filters.question_type,
            "difficulty": user_input.filters.difficulty,
            "language": user_input.filters.language
        }
    }

    mongoConnection.get_collection("GENERATED_USER_PROBLEM").update_one({"UUID": UUID}, {"$set": {"date_time": datetime.now(pytz.timezone('UTC')), "questions": output_response}}, upsert=True)
    
    return output_response
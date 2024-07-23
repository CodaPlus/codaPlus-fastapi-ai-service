import json
from base_model.custom_wrapper import LangchainLLMWrapper
from DB.mongo import mongoConnection
from DB.pg import db_conn
import base64
import pytz
from datetime import datetime
from base_model.RAG.ingest_data import DataIngestor


class UserCodeAnalysis:
    def __init__(self, UUID: str, tags: list, number_of_tags: list, rules=None):
        if not UUID:
            raise ValueError("UUID is required")
        self.UUID = UUID
        self.tags = tags
        self.rules = rules if rules else [
            "Tags should be programming language agnostic.",
            "Tags should be relevant to the question context and User Answer.",
            "Tags should be specific and not too broad.",
            "Evaluate relevance and contextual understanding of the problem.",
            "Assess mastery of the specified programming language and concepts.",
            "Evaluate problem-solving skills and algorithmic thinking.",
            "Assess code efficiency and optimization for performance.",
            "Evaluate code quality, readability, and maintainability.",
        ]   
        self.number_of_tags = number_of_tags

    
    def format_cursor_data(self, cursor_data):
        formatted_data = []
        for data in cursor_data:
            formatted_data.append({
                "question_id": data[0],
                "question_title": data[1],
                "difficulty": data[2],
                "programming_language": data[3],
                "question_description": data[4],
                "code_executed_and_tests_passed": data[5],
                "attempted_answer": base64.b64decode(data[6]).decode("utf-8")
            })
        return formatted_data


    def process(self, unique_coding_question_types:list, start_time: datetime, end_time:datetime) -> list:
        analyzed_types_data= []
        with db_conn() as conn:
            for question_type in unique_coding_question_types:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id, question_title, difficulty, programming_language, question_description, completed_status, attempted_answer FROM attempted_question WHERE created_by = %s AND question_type = %s AND completed_status = 'COMPLETED' AND created_at BETWEEN %s AND %s", (self.UUID, question_type, start_time, end_time))
                    analyzed_questions = self.assign_auto_tags(self.tags, self.format_cursor_data(cursor.fetchall()), question_type, self.number_of_tags)
                    analyzed_types_data.append({
                        "question_type": question_type,
                        "analyzed_questions": analyzed_questions
                    })
        
        self.analyze_user_coding(analyzed_types_data)
        print(f"User Coding Analysis Completed for {self.UUID}")


    def assign_auto_tags(self, tags: list, user_answered_questions: list, question_type:str, number_of_tags: int = 1) -> dict:
        _, _, llm_model = LangchainLLMWrapper.load_llm_model()

        analyzed_questions = []

        for question in user_answered_questions:
            question_prompt = f"Act as an intelligent coding guru to analyze the following programming question and its context.\n\n"
            question_prompt += f"Question Title: {question['question_title']}\n" \
                            f"Question type is : {question_type}\n" \
                            f"Difficulty level of the question: {question['difficulty']}\n" \
                            f"Programming Language: {question['programming_language']}\n" \
                            f"Description: {question['question_description']['description']}\n" \
                            f"Code Template: {question['question_description']['code_template']}\n" \
                            f"Attempted Answer: {question['attempted_answer']}\n" \
                            f"Code Executed without errors with all the tests passed: {question['code_executed_and_tests_passed']}."

            if self.rules:
                question_prompt += f"\nRules to follow when assigning tags:{','.join(self.rules)}\n"

            question_prompt += f"\nSuggest up to {number_of_tags} most fitting tags from the following list:\n{','.join(tags)}\n\n"
            question_prompt += "Use your reasoning to select the most appropriate tags and explain why they are relevant. YOU MUST ONLY RETURN THE TAGS AS A LIST! and IF NO TAGS FIT, RETURN AN EMPTY LIST.\n\n"
            response = llm_model.generate([question_prompt])

            try:
                generated_response = json.loads(response.generations[0][0].text.strip())
                suggested_tags = generated_response if isinstance(generated_response, list) else []
            except Exception as e:
                print(f"Error processing AI response: {e}")
                suggested_tags = []

            analyzed_questions.append({
                "question_id": question['question_id'],
                "suggested_tags": suggested_tags
            })

        return analyzed_questions


    def analyze_user_coding(self, analyzed_types_data):
        _, _, llm_model = LangchainLLMWrapper.load_llm_model()
        
        prompt = ("As an expert evaluator, provide a comprehensive analysis of a programmer's capabilities and areas for improvement. "
                "Consider their recent coding activities detailed below. Assess their understanding of programming concepts etc."
                "Coding Activities Overview:\n")

        for analyzed_type in analyzed_types_data:
            prompt += (f"- For questions related to {analyzed_type['question_type']}, "
                    f"the programmer attempted {len(analyzed_type['analyzed_questions'])} questions. "
                    "Tags associated with these questions include: ")
            tags = {tag for question in analyzed_type['analyzed_questions'] for tag in question['suggested_tags']}
            prompt += ', '.join(tags) + '.\n'
        
        prompt += """Given the above, analyze the programmer's skill level, areas of strength, and areas needing improvement etc.  Provide actionable advice on how they can enhance their coding proficiency. Your generated content should be formatted as a JSON object, adhering to the following Pydantic schema:      
                class SummarizedSchema(BaseModel):
                    summarized_context: str"""
        
        response = llm_model.generate([prompt])

        generated_response = None
        try:
            generated_response = json.loads(response.generations[0][0].text.strip())
        except Exception as e:
            print(f"Error processing AI response: {e}")

        self.store_analyzed_results(analyzed_types_data, generated_response)


    def store_analyzed_results(self, analyzed_questions_tags, analysis):
        data_obj = {
            "date_time": datetime.now(pytz.timezone('UTC')),
            "analyzed_questions": analyzed_questions_tags,
            "user_code_analysis": analysis
        }
        mongoConnection.get_collection("USER_CODING_ANALYSIS").update_one(
            {"UUID": self.UUID},
            {"$set": data_obj},
            upsert=True
        )
        print(f"User Coding Analysis Saved for {self.UUID}")

        ## Store the Vectors in Chroma
        try:
            DataIngestor(self.UUID).ingest(data_obj, "USER_CODING_ANALYSIS")
            print(f"User Questioner Analysis vector data storage is Completed for {self.UUID}")
        except Exception as e:
            print(f"Error in saving user questioner analysis vector data: {e}")
from base_model.custom_wrapper import LangchainLLMWrapper
from DB.mongo import mongoConnection
from DB.pg import db_conn
from base_model.RAG.ingest_data import DataIngestor
import pytz
from datetime import datetime

class UserQuestionerAnalysis:
    def __init__(self, UUID: str):
        if not UUID:
            raise ValueError("UUID is required")
        self.UUID = UUID

    def format_cursor_data(self, cursor_data):
        if not cursor_data:
            return []
        formatted_data = []
        for data in cursor_data:
            if data['question_no'] is not None:
                with db_conn() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT question FROM questioner_feedback_questions WHERE id = %s", (data['question_no'],))
                        question = cursor.fetchone()
                        if question:
                            formatted_data.append({
                                "question": question[0],
                                "answer": data['user_answer']
                            })
        return formatted_data


    def process(self):
        results = None
        with db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT questions_answers FROM user_questioner_feedback_data WHERE user_id = %s", (self.UUID,))
                fetched_data = cursor.fetchone()
                if fetched_data:
                    results = self.format_cursor_data(fetched_data[0])
        self.analyze_user_questioner(results)


    def analyze_user_questioner(self, user_questioner_data):
        _, _, llm_model = LangchainLLMWrapper.load_llm_model()

        combined_prompt = "Analyze the following list of user questions and answers to provide a comprehensive summary of the questionnaire's findings, focusing on user ambitions, insights, and more details helpful to get an understanding about the user.\n\n"        
        if user_questioner_data is not None:
            for idx, qa_pair in enumerate(user_questioner_data, start=1):
                combined_prompt += (f"{idx}. Question: {qa_pair['question']}, Answer: {qa_pair['answer']}\n")
        combined_prompt += "Based on the above questions and answers, summarize the key insights from the user's responses:\n\n"
        
        response = llm_model.generate([combined_prompt])
        summary = response.generations[0][0].text.strip() 
        print(f"Summary: {summary}") 
        self.store_analyzed_results(summary)
    
    
    def store_analyzed_results(self, summary):
        data_obj = {"date_time": datetime.now(pytz.timezone('UTC')), "summary": summary, }
        try:
            mongoConnection.get_collection("USER_QUESTIONER_ANALYSIS").update_one(
            {"UUID": self.UUID},
            {"$set": data_obj},
            upsert=True
        )
            print(f"User Questioner Analysis Completed for {self.UUID}")
        except Exception as e:
            print(f"Error in saving user questioner analysis data: {e}")

        ## Store the Vectors in Chroma
        try:
            DataIngestor(self.UUID).ingest(data_obj, "USER_QUESTIONER_ANALYSIS")
            print(f"User Questioner Analysis vector data storage is Completed for {self.UUID}")
        except Exception as e:
            print(f"Error in saving user questioner analysis vector data: {e}")
import logging
from DB.mongo import mongoConnection
from langchain.chains.llm import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from DB.redis import redis_cache
from LLM import DEFAULT_MODEL
from langchain.globals import set_llm_cache
import pytz
from datetime import datetime


class UserSummaryCreation:
    def __init__(self, UUID: str):
        if not UUID:
            raise ValueError("UUID is required")
        self.UUID = UUID


    def _fetch_data(self):
        USER_TONE = mongoConnection.get_collection("USER_TONE").find_one({"UUID": self.UUID}, {"_id": 0})
        USER_CODING_ANALYSIS = mongoConnection.get_collection("USER_CODING_ANALYSIS").find_one({"UUID": self.UUID}, {"_id": 0})
        USER_PROFILE_ANALYSIS = mongoConnection.get_collection("USER_PROFILE_ANALYSIS").find_one({"UUID": self.UUID}, {"_id": 0})
        USER_QUESTIONER_ANALYSIS = mongoConnection.get_collection("USER_QUESTIONER_ANALYSIS").find_one({"UUID": self.UUID}, {"_id": 0})

        formatted_data = f"""
                USER WRITTEN CODING ANALYSIS: {USER_CODING_ANALYSIS.get('user_code_analysis') if USER_CODING_ANALYSIS else "No data found"}
                USER ANSWERED QUESTIONER ANALYSIS: {USER_PROFILE_ANALYSIS.get('summary') if USER_PROFILE_ANALYSIS else "No data found"}
                USEE SOCIAL PROFILE ANALYSIS:
                    GITHUB: {USER_QUESTIONER_ANALYSIS.get('github') if USER_QUESTIONER_ANALYSIS.get('github') else "No data found"}
                    LEETCODE: {USER_QUESTIONER_ANALYSIS.get('leetcode') if USER_QUESTIONER_ANALYSIS.get('leetcode') else "No data found"}
                    LINKEDIN: {USER_QUESTIONER_ANALYSIS.get('linkedin') if USER_QUESTIONER_ANALYSIS.get('linkedin') else "No data found"}
                USER CHAT TONE: {USER_TONE.get('user_tone') if USER_TONE else "No data found"} 
            """
        
        return formatted_data


    def analyze_user(self):

        formatted_fetch_data = self._fetch_data()
        
        system_ = f"""
        You are an assistant tasked with writing a concise summary to extract important information and features using the information provided within the specified context. Make sure to use the context provided to summarize as accurately as possible without fabricating information or making assumptions about the context given.
        As additional instructions: anything between the following /`context/`  html blocks is retrieved from a knowledge bank of the user profile data.
        
        <context>
            { formatted_fetch_data if formatted_fetch_data else "No context found"}
        <context/>

        Your generated content should be formatted as a JSON object, adhering to the following Pydantic schema:      

        class SummarizedSchema(BaseModel):
            summarized_context: str                                                                                                                                
        """

        set_llm_cache(redis_cache)

        first_prompt = ChatPromptTemplate.from_template(system_)
        init_chain = LLMChain(llm=DEFAULT_MODEL, prompt=first_prompt)
        result = init_chain.invoke({})
        return result['text'].summarized_context


    def store_user_summary(self, analyzed_data):
        mongoConnection.get_collection("USER_ANALYSIS").update_one({"UUID": self.UUID}, {"$set": {"date_time": datetime.now(pytz.timezone('UTC')), "user_summary": analyzed_data}}, upsert=True)
        print(f"User summary stored for {self.UUID}")


    def process(self):
        logging.info(f"[Task Started] Detect Brand Tone for: {self.UUID}")
        try:
            analyzed_data=self.analyze_user()
            self.store_user_summary(analyzed_data)
        except print(0):
            pass
        finally:
            logging.info(f"[Task Finished] Detect Brand Tone for: {self.UUID}")

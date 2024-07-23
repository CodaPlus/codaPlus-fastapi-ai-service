from DB.mongo import mongoConnection
from DB.pg import db_conn
from base_model.text_classifier import ZeroShotGPTClassifier
import logging
import pytz
from datetime import datetime
from base_model.RAG.ingest_data import DataIngestor


class UserFeatureAnalysis:
    def __init__(self, UUID: str):
        if not UUID:
            raise ValueError("UUID is required")
        self.UUID = UUID


    def zeroshot_classifier(self, content: str, x_classifiers: list, y_classifiers: list) -> list:
        try:
            clf = ZeroShotGPTClassifier()
            clf.fit(x_classifiers, y_classifiers)
            label = clf.predict([content])
            return label
        except Exception as e:
            logging.exception(e)
            raise e


    def format_linkedin_features(self, linkedin_data):
        if not linkedin_data:
            return {
                "coding_skills": "None",
                "soft_skills": "None",
                "work_experience": [],
                "education_experience": [],
                "certifications_experience": [],
                "courses_experience": []
            }
    
        return {
            "coding_skills": self.zeroshot_classifier(linkedin_data, ["proficient", "intermediate", "beginner", "expert", "None"]),
            "soft_skills": self.zeroshot_classifier(linkedin_data, ["team player", "leadership", "communication", "problem solving", "None"]),
            "work_experience": linkedin_data['experience'],
            "education_experience": linkedin_data["education"],
            "certifications_experience": linkedin_data['certifications'],
            "courses_experience": linkedin_data['courses'],
        }

    def format_leetcode_features(self, leetcode_data):
        if not leetcode_data:
            return {
                "coding_skills": "None",
                "competitive_coding_skills": "None",
                "problem_solving_skills": "None",
                "data_structure_skills": "None",
                "algorithm_skills": "None",
                "programming_language_experience": [],
                "global_rank": "None"
            }
        
        return {
            "coding_skills": self.zeroshot_classifier(leetcode_data, ["proficient", "intermediate", "beginner", "expert", "None"]),
            "competitive_coding_skills": self.zeroshot_classifier(leetcode_data, ["proficient", "intermediate", "beginner", "expert", "None"]),
            "problem_solving_skills": self.zeroshot_classifier(leetcode_data, ["proficient", "intermediate", "beginner", "expert", "None"]),
            "data_structure_skills": self.zeroshot_classifier(leetcode_data, ["proficient", "intermediate", "beginner", "expert", "None"]),
            "algorithm_skills": self.zeroshot_classifier(leetcode_data, ["proficient", "intermediate", "beginner", "expert", "None"]),
            "programming_language_experience": leetcode_data['languageStats'],
            "global_rank": leetcode_data['user_global_ranking'],
        }


    def format_github_features(self, github_data):
        if not github_data:
            return {
                "open_source_skills": "None",
                "programming_language_experience": []
            }
        
        return {
            "open_source_skills": self.zeroshot_classifier(github_data, ["proficient", "intermediate", "beginner", "expert", "None"]),
            "programming_language_experience": github_data['languageFrequency'],
        }


    def process(self) -> list:
        leetcode_results = None
        linkedin_results = None
        github_results = None

        with db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT data FROM github_features WHERE 'UUID' = %s", (self.UUID,))
                github_results = self.format_github_features((cursor.fetchone())[0])
                
                cursor.execute("SELECT data FROM leet_code_features WHERE 'UUID' = %s", (self.UUID,))
                leetcode_results = self.format_leetcode_features((cursor.fetchone())[0])
                
                cursor.execute("SELECT data FROM linkedin_features WHERE 'UUID' = %s", (self.UUID,))
                linkedin_results = self.format_linkedin_features((cursor.fetchone())[0])
        
        self.store_analyzed_results(github_results, leetcode_results, linkedin_results)


    def store_analyzed_results(self, github_results, leetcode_results, linkedin_results):
        data_obj = {
            "date_time": datetime.now(pytz.timezone('UTC')),
            "github": github_results,
            "leetcode": leetcode_results,
            "linkedin": linkedin_results
        }
        try:
            mongoConnection.get_collection("USER_PROFILE_ANALYSIS").update_one(
                {"UUID": self.UUID},
                {"$set": data_obj},
                upsert=True
            )
            print(f"User Questioner Analysis Completed for {self.UUID}")
        except Exception as e:
            print(f"Error in saving user questioner analysis data: {e}")

        ## Store the Vectors in Chroma
        try:
            DataIngestor(self.UUID).ingest(data_obj, "USER_CODING_ANALYSIS")
            print(f"User Questioner Analysis vector data storage is Completed for {self.UUID}")
        except Exception as e:
            print(f"Error in saving user questioner analysis vector data: {e}")


    def analyze_specific_social(self, social: str):
        """
        Analyzes and updates user feature data for a specific social platform.

        :param social: A string indicating the social platform to analyze ('github', 'linkedin', or 'leetcode').
        """
        result = None

        with db_conn() as conn:
            with conn.cursor() as cursor:
                if social.lower() == 'github':
                    cursor.execute("SELECT data FROM github_features WHERE 'UUID' = %s", (self.UUID,))
                    result = self.format_github_features(cursor.fetchone())

                elif social.lower() == 'leetcode':
                    cursor.execute("SELECT data FROM leet_code_features WHERE 'UUID' = %s", (self.UUID,))
                    result = self.format_leetcode_features(cursor.fetchone())

                elif social.lower() == 'linkedin':
                    cursor.execute("SELECT data FROM linkedin_features WHERE 'UUID' = %s", (self.UUID,))
                    result = self.format_linkedin_features(cursor.fetchone())

        if result is not None:
            self.store_analyzed_result_for_social(social, result)
        else:
            print(f"No data found or invalid social platform specified for UUID: {self.UUID}")

    def store_analyzed_result_for_social(self, social: str, result):
        """
        Updates the analyzed result for a specific social platform in the MongoDB collection.

        :param social: The social platform ('github', 'linkedin', 'leetcode').
        :param result: The analyzed result to be stored.
        """
        try:
            update_field = {f"{social}": result}
            mongoConnection.get_collection("USER_PROFILE_ANALYSIS").update_one(
                {"UUID": self.UUID},
                {"$set": update_field},
                upsert=True
            )
            print(f"Analysis Completed for {self.UUID} on {social}.")
        except Exception as e:
            print(f"Error in updating analysis for {social}: {e}")
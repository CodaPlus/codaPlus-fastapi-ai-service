import os 
from dotenv import load_dotenv
load_dotenv()
import logging

import dramatiq
from dramatiq.brokers.redis import RedisBroker


dramatiq.set_broker(RedisBroker(url=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}/{os.getenv('REDIS_DB_WORKER', 8)}"))


from base_model.user_feature_analysis import UserFeatureAnalysis
from base_model.user_tone_detection import UserToneDetection
from base_model.user_questioner_analysis import UserQuestionerAnalysis
from base_model.user_code_analysis import UserCodeAnalysis
from base_model.user_code_embeddings import UserCodeEmbeddings
from base_model.user_generate_summary import UserSummaryCreation

@dramatiq.actor(queue_name='routine_process_queue')
def get_code_analysis(event_data):
    UUID = event_data.get('UUID', None)
    question_type = event_data.get('question_type', None)
    start_time = event_data.get('start_time', None)
    end_time = event_data.get('end_time', None)

    if not UUID:
        raise ValueError("UUID is required")
    if not question_type:
        raise ValueError("question_type is required")
    if not start_time:
        raise ValueError("start_time is required")
    if not end_time:
        raise ValueError("end_time is required")
    
    try:
        UserCodeAnalysis(UUID, [], 10).process(question_type, start_time, end_time)
    except Exception as e:
        logging.error(f"Error getting code embeddings for {UUID}: {e}")
    
    print(f"Getting code embeddings for {UUID}")


@dramatiq.actor(queue_name='routine_process_queue')
def get_code_embeddings(event_data):
    UUID = event_data.get('UUID', None)
    start_time = event_data.get('start_time', None)
    end_time = event_data.get('end_time', None)

    if not UUID:
        raise ValueError("UUID is required")
    if not start_time:
        raise ValueError("start_time is required")
    if not end_time:
        raise ValueError("end_time is required")
    
    try:
        UserCodeEmbeddings(UUID).process(start_time, end_time)
    except Exception as e:
        logging.error(f"Error getting code embeddings for {UUID}: {e}")
    
    print(f"Getting code embeddings for {UUID}")


@dramatiq.actor(queue_name='routine_process_queue')
def analyze_profile_data(event_data):
    UUID = event_data.get('UUID', None)
    
    if not UUID:
        raise ValueError("UUID is required")
    
    try:
        UserFeatureAnalysis(UUID).process()
    except Exception as e:
        logging.error(f"Error analyzing profile data for {UUID}: {e}")
    
    print(f"Analyzing profile data for {UUID}")


@dramatiq.actor(queue_name='routine_process_queue')
def analyze_user_tone(event_data):
    UUID = event_data.get('UUID', None)

    if not UUID:
        raise ValueError("UUID is required")
    
    try: 
        UserToneDetection(UUID).process()
    except Exception as e:
        logging.error(f"Error analyzing tone of the user {UUID}: {e}")

    print(f"Analyzing user tone for {UUID}")    


@dramatiq.actor(queue_name='process_queue')
def analyze_user_social(event_data):
    UUID = event_data.get('UUID', None)
    social = event_data.get('social', None)

    if not UUID:
        raise ValueError("UUID is required")
    if not social:
        raise ValueError("Social is required")
    
    try: 
        UserFeatureAnalysis(UUID).analyze_specific_social(social)
    except Exception as e:
        logging.error(f"Error analyzing user {UUID} social for {social}: {e}")

    print(f"Analyzing user {UUID} social for {social}")


@dramatiq.actor(queue_name='process_queue')
def analyze_questioner_data(event_data):
    UUID = event_data.get('UUID', None)
    
    if not UUID:
        raise ValueError("UUID is required")
    
    try:
        UserQuestionerAnalysis(UUID).process()
    except Exception as e:
        logging.error(f"Error analyzing questioner data for {UUID}: {e}")
    
    print(f"Analyzing questioner data for {UUID}")


@dramatiq.actor
def generate_profile_summary(*args, **kwargs):
    try:
        UserSummaryCreation(*args, **kwargs).process()
    except Exception as e:
        logging.error(f"Error generating profile summary: {e}")
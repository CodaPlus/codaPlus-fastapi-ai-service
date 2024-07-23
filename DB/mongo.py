import pymongo
import os 
from dotenv import load_dotenv
load_dotenv("../.env") 

mongoClient = pymongo.MongoClient(os.getenv("MONGO_DB_URL"))
mongoConnection = mongoClient["data_processing"] # Database

## SCHEMA INDEXES ##
mongoConnection.get_collection("GENERATED_USER_PROBLEM").create_index([("date_time", 1)], expireAfterSeconds=86400) # 1 day
mongoConnection.get_collection("MODEL_USER_CHATS").create_index([("date_time", 1)], expireAfterSeconds=7200) # 2 hours

## USER ##
mongoConnection.get_collection("USER_TONE").create_index([("date_time", 1), ("UUID", 1)])
mongoConnection.get_collection("USER_CODING_ANALYSIS").create_index([("date_time", 1), ("UUID", 1)])
mongoConnection.get_collection("USER_PROFILE_ANALYSIS").create_index([("date_time", 1), ("UUID", 1)])
mongoConnection.get_collection("USER_QUESTIONER_ANALYSIS").create_index([("date_time", 1), ("UUID", 1)])
mongoConnection.get_collection("USER_ANALYSIS").create_index([("date_time", 1), ("UUID", 1)])
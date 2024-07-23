from DB.mongo import mongoConnection
from bson.objectid import ObjectId


def get_document_array(key, UUID):
    document = mongoConnection.get_collection("MODEL_USER_CHATS").find_one({"_id": ObjectId(key), "UUID": UUID})
    if document:
        return document.get("data", [])
    else:
        return []


def append_to_array(session_id, obj, UUID, empty_array=False):
    collection = mongoConnection.get_collection("MODEL_USER_CHATS")
    if empty_array:
        update_result = collection.update_one(
            {"_id": ObjectId(session_id), "UUID": UUID},
            {"$set": {"data": [obj]}},
            upsert=True
        )
    else:
        update_result = collection.update_one(
            {"_id": ObjectId(session_id), "UUID": UUID},
            {"$push": {"data": {"$each": [obj]}}},
            upsert=True
        )
    return update_result.upserted_id if update_result.upserted_id else ObjectId(session_id)

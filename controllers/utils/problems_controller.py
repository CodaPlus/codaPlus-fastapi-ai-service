from DB.mongo import mongoConnection


def delete_a_question(UUID: str, question_id: str):
    collection = mongoConnection.get_collection("GENERATED_USER_PROBLEM")
    user_document = collection.find_one({"UUID": UUID})
    if not user_document:
        raise Exception("Document not found")
    
    questions = user_document.get("questions", {}).get("questions", [])
    updated_questions = [question for question in questions if question.get('question_id') != question_id]
    if len(questions) == len(updated_questions):
        raise Exception("Question not found or invalid question_id")

    result = collection.update_one({"UUID": UUID}, {"$set": {"questions.questions": updated_questions}})
    if result.matched_count == 0:
        raise Exception("Update failed")
    

def retrieve_question(UUID: str, question_id: str):
    if question_id is not None:
        user_document = mongoConnection.get_collection("GENERATED_USER_PROBLEM").find_one({"UUID": UUID})
        if user_document:
            questions = user_document.get("questions", {}).get("questions", [])
            question = next((q for q in questions if q.get('question_id') == question_id), None)
            if question:
                return {"question": question, "filters": user_document.get("filters", {})}
    return {}
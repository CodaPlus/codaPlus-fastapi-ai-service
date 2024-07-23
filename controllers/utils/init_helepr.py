import json
from LLM import DEFAULT_MODEL, FINETUNED_MODEL
from langchain.chains.llm import LLMChain
from bson.objectid import ObjectId

async def get_questions(final_prompt):

    final_chain = LLMChain(llm=FINETUNED_MODEL, prompt=final_prompt)
    final_chain_output = await final_chain.ainvoke({})
    
    try:
        final_chain_output = json.loads(final_chain_output['text'])
        final_chain_output['question_id'] = str(ObjectId())
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        error_pos = e.pos
        print(final_chain_output['text'][max(0, error_pos - 50):error_pos + 50])
        return []

    return final_chain_output
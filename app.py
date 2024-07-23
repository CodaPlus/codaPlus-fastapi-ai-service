from starlette.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, responses, encoders, Security, Depends, Request, APIRouter, BackgroundTasks
from pydantic import BaseModel, Field
from fastapi.security.api_key import APIKeyHeader
from controllers.question_generator import generate_coding_problem
from controllers.chat_history import context_aware_conversation
from controllers.solution import generate_coding_solution
from controllers.utils.problems_controller import delete_a_question
from typing import List, Union, Optional, Any
import logging
import os
from dotenv import load_dotenv
load_dotenv()

ENVIRONMENT = str(os.getenv("SERVER_ENV", "DEV"))
CORS_ORIGINS = str(os.getenv("CORS_ORIGINS", "*")).split(",")

middleware = []

API_KEY = os.getenv("API_KEY", "NO-API-KEY")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def api_authentication(api_key_header: str = Security(api_key_header), request: Request=None):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key!")


middleware.append(
    Middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
)

app = FastAPI(title="Coda Plus | AI Chain Management Backend Service",
              description="Backend Service for Coda Plus AI Chain Management", version="0.1.0", middleware=middleware)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    responses={404: {"description": "Not found"}}, 
)

class Filters(BaseModel):
    question_type: str
    difficulty: str
    language: str

class UserInput(BaseModel):
    filters: Filters
    number_of_questions: Optional[int] = 5
    difficulty_adjustment_parameter: Optional[str] = None
    UUID: str 
    user_name: str 


@router.post("/generate_coding_problem")
async def get_coding_problem(request: Request, user_input: UserInput, api_key: str = Depends(api_authentication)):
    try:
        json_compatible_data = encoders.jsonable_encoder(await generate_coding_problem(user_input, user_input.UUID))
        return responses.JSONResponse(content=json_compatible_data)
    except Exception as e:
        print(e)
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


class UserInput(BaseModel):
    question: str = Field(..., example="What is the best way to learn Python?")
    session_id: Optional[str] = Field(None, example="802e9dab-7b11-4ac3-adfc-e37c9b353de1")
    coding_question_id: Optional[str] = Field(None, example="e37c9b353de1-7b11-4ac3-adfc-802e9dab")
    UUID: str 
    user_name: str 
   
@router.post("/coding_problem_chat")
async def get_coding_problem_history(request: Request, user_input: UserInput, api_key: str = Depends(api_authentication)):
    try:
        json_compatible_data = encoders.jsonable_encoder(context_aware_conversation(str(user_input.question), user_input.UUID, user_input.user_name, user_input.session_id, user_input.coding_question_id))
        return responses.JSONResponse(content=json_compatible_data)
    except Exception as e:
        print(e)
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


class UserInput(BaseModel):
    question_title: str = Field(..., example="Code description")
    problem_description: str = Field(..., example="Code problem")
    code_template: str
    filters: Filters
    test_cases: List[Any] 
    UUID: str 
    user_name: str


@router.post("/generate_coding_solution")
async def get_coding_solution(request: Request, user_input: UserInput, api_key: str = Depends(api_authentication)):
    try:
        json_compatible_data = encoders.jsonable_encoder(generate_coding_solution(user_input, user_input.UUID))
        return responses.JSONResponse(content=json_compatible_data)
    except Exception as e:
        print(e)
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.delete("/delete_question/{UUID}/{question_id}")
async def delete_question(request: Request, UUID: str, question_id: int, background_tasks: BackgroundTasks, api_key: str = Depends(api_authentication)):
    background_tasks.add_task(delete_a_question, UUID, question_id)
    return {"message": "Deletion of question scheduled successfully"}

app.include_router(router)

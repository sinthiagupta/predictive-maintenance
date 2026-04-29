from fastapi import APIRouter
from pydantic import BaseModel
from app.chatbot.chatbot_service import generate_chat_response

router = APIRouter(prefix="/chat", tags=["AI Chatbot"])


class ChatRequest(BaseModel):
    message: str
    id: int


@router.post("/")
def maintenance_chat(request: ChatRequest):
    response = generate_chat_response(
        user_message=request.message,
        dataset_id=request.id
    )

    return {"response": response}
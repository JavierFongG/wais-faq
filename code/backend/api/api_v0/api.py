from fastapi import APIRouter

from .routers import chat

router = APIRouter()

router.include_router(chat.router, prefix="/chat", tags=["chat"])
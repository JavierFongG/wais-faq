from fastapi import APIRouter

# from .routers import chat
from .routers import whatsapp

router = APIRouter()

# router.include_router(chat.router, prefix="/chat", tags=["chat"])

router.include_router(whatsapp.router, prefix="/whatsapp", tags=["whatsapp"])
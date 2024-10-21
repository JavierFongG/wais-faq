from fastapi import FastAPI
from api.api_v0.api import router as api_router

app = FastAPI()

app.include_router(api_router, prefix="/api/v0")

@app.get("/")
async def root():
    return {"message": "Backend up and running"}
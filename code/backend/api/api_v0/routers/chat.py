import os
from fastapi import APIRouter
from api.api_v0.models.prompt import Prompt

from rag.retriever import WaisRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

model_kwargs = {'trust_remote_code' : True, 'revision' : 'main'}

# embeddings = HuggingFaceEmbeddings(model_name='Alibaba-NLP/gte-multilingual-base', model_kwargs=model_kwargs)
#
# # print(os.getenv("VECTOR_STORE_DIRECTORY"))
# # print(os.getenv("VECTOR_STORE_NAME"))
#
# vector_store = Chroma(
#     persist_directory=os.getenv("VECTOR_STORE_DIRECTORY")
#     , collection_name =os.getenv("VECTOR_STORE_NAME")
#     , embedding_function=embeddings
# )
#
# retriever = WaisRetriever(vector_store=vector_store, window_size = 2)
# client = AsyncGroq(
#             api_key= os.getenv("GROQ_API_KEY")
#         )


@router.get("/")
async def root():
    return {"message": "chat routes available"}

@router.post("/chat_completion")
async def chat_completion(prompt : Prompt):

    return {"message": prompt.prompt}


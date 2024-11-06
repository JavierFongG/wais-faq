import os
from fastapi import APIRouter, Query, Response
from api.api_v0.models.prompt import Prompt
from typing import Optional
import string
import requests
from nltk import word_tokenize
from unicodedata import normalize
import re
from groq import Groq, AsyncGroq
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List
def clean_text(text):
    """
    Cleans the input text by performing several transformations such as:
    - Removing HEX characters
    - Normalizing the text (removing accents and special characters)
    - Removing extra whitespace and newline characters
    - Removing punctuation
    - Lowercasing the text
    - Tokenizing and removing stopwords
    - Returning the cleaned text as a string of words
    """

    punctuation = string.punctuation
    punctuation = punctuation.replace('+', '').replace('-', '').replace('*', '').replace('.', '').replace(',', '')
    punctuation = punctuation + '“”«»°'

    # Example usage:
    # cleaned_text = clean_text("Este es un ejemplo... de texto! Contiene (paréntesis) y --- otros símbolos.")
    # expected_output = 'ejemplo texto contiene paréntesis -- - símbolos'

    # Step 1: Define the set of stopwords in Spanish
    # stop_w = set(stopwords.words('spanish'))

    # Step 2: Normalize the text to remove accents and special characters
    # Normalize text using 'NFKD' form which separates characters from their diacritics
    _clean_text = normalize('NFKD', text)

    # Step 3: Remove unnecessary whitespace (e.g., two or more spaces)
    _clean_text = re.sub(r'\s{2,}', ' ', _clean_text)

    # Step 4: Remove spaces before and after newline characters
    _clean_text = re.sub(r'\n\s+', '\n', _clean_text)  # Spaces after newline
    _clean_text = re.sub(r'\s+\n', '\n', _clean_text)  # Spaces before newline

    # Step 5: Limit consecutive newlines to a maximum of two
    _clean_text = re.sub(r'\n{3,}', '\n\n', _clean_text)

    # Step 6: Replace groups of consecutive periods with a single period
    _clean_text = re.sub(r'\.\s*\.+', '.', _clean_text)

    # Step 7: Convert all text to lowercase for uniformity
    _clean_text = _clean_text.lower()

    # Step 8: Remove all punctuation characters
    _clean_text = _clean_text.translate(str.maketrans('', '', punctuation))

    # Step 9: Tokenize the text into words (split text into individual tokens)
    _clean_text = word_tokenize(_clean_text)

    # Step 10: Remove stopwords from the tokenized words
    # List comprehension filters out any words that are in the stopwords set
    # _clean_text = [word for word in _clean_text if word not in stop_w]

    # Step 11: Join the remaining words back into a single string and return
    _clean_text = ' '.join(_clean_text)
    return _clean_text

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

# pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
# index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
# embeddings = HuggingFaceEmbeddings(
#         model_name= os.getenv('EMBEDDINGS_MODEL_NAME')
#         , model_kwargs={'trust_remote_code': True}
#     )
#
# vector_store = PineconeVectorStore(
#     index=index
#     , embedding=embeddings
# )


client = AsyncGroq(
            api_key= os.getenv("GROQ_API_KEY")
        )

async def chat_with_history(message : str ) -> str:
    messages = []

    # context = vector_store.similarity_search(message, k=1)
    # _context_text = ''.join([res.page_content for res in context])

    # messages.append({
    #         "role": "user",
    #         "content": f"""
    #     Busca en los archivos de equipo médico para responder con el mayor detalle posible, usando solo la información del contexto. Si no encuentras suficiente información, responde 'No lo sé'
    #     Contexto: {_context_text}
    #
    #     Pregunta: {message}""",
    # })

    messages.append(f"""{message}""")
    response_content = ""

    stream = await client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192",
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True,
    )

    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += chunk.choices[0].delta.content
        yield response_content

async def handle_whatsapp_message(body):
    bot_response = await chat_with_history(body)
    headers = {"Authorization": "Bearer " + os.environ["WHATSAPP_TOKEN"]}
    url = f"https://graph.facebook.com/v18.0/{os.environ['TEST_PHONE_ID']}/messages"
    response = {
        "messaging_product": "whatsapp",
        "to": body["entry"][0]["changes"][0]["value"]["messages"][0]["from"],
        "type": "text",
        "text": {"preview_url": False, "body": bot_response},
    }
    requests.post(url=url, json=response, headers=headers)

@router.get("/")
async def root():
    return {"message": "chat routes available"}

@router.post("/chat_completion")
async def chat_completion(prompt : Prompt):
    async for response in chat_with_history(prompt.prompt):
        continue
    return {"message": response}

# Just for webhook verification
@router.get("/webhook/")
async def verify_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[int] = Query(None, alias="hub.challenge"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
):
    print(hub_mode, hub_challenge, hub_verify_token)
    return hub_challenge

@app.post("/webhook/")
async def receive_webhook(body: dict):
    print(body)
    try:
        # info on WhatsApp text message payload:
        # https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples#text-messages
        if body.get("object"):
            if (
                body.get("entry")
                and body["entry"][0].get("changes")
                and body["entry"][0]["changes"][0].get("value")
                and body["entry"][0]["changes"][0]["value"].get("messages")
                and body["entry"][0]["changes"][0]["value"]["messages"][0]
            ):
                await handle_whatsapp_message(body)
            return {"status": "ok"}
        else:
            # if the request is not a WhatsApp API event, return an error
            return Response(content="not a WhatsApp API event", status_code=404)

    # catch all other errors and return an internal server error
    except Exception as e:
        print(f"unknown error: {e}")
        return Response(content=str(e), status_code=500)
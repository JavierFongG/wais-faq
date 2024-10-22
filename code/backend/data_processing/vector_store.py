from data_managers.downloaders import Downloader, S3Downloader
from data_managers.extractors import Extractor, PdfExtractor
import boto3
import os
import re
import string
from unicodedata import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from nlp import clean_text

class KnowledgeDocService:

    def __init__(self,downloader : Downloader, extractor : Extractor):
        self.downloader = downloader
        self.extractor = extractor

    def get_text(self, file_key : str) -> str:
        file_content = self.downloader.download(file_key)
        return self.extractor.extract_documents(file_content)


if __name__ == '__main__':

    _aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID')
    _aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    _bucket = os.getenv('BUCKETNAME')

    s3_client = boto3.client(
        's3'
        , aws_access_key_id = _aws_access_key_id
        , aws_secret_access_key = _aws_secret_access_key
    )

    loader = S3Downloader(bucket_name = _bucket)
    extractor = PdfExtractor()
    doc_servive = KnowledgeDocService(loader, extractor)

    bucket_content = s3_client.list_objects_v2(Bucket=_bucket)['Contents']
    documents_library = []
    for obj in bucket_content:
        documents_library.extend(doc_servive.get_text(obj['Key']))

    for _index, _doc in enumerate(documents_library):
        _doc.page_content = clean_text(_doc.page_content)
        documents_library[_index] = _doc

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ]
        , chunk_size=512
        , chunk_overlap=64
    )

    chunks = text_splitter.split_documents(documents_library)

    for _index, _doc in enumerate(chunks):
        _doc.metadata['position'] = _index
        chunks[_index] = _doc

    embeddings = HuggingFaceEmbeddings(
        model_name='Alibaba-NLP/gte-multilingual-base'
        , model_kwargs={'trust_remote_code': True}
    )

    vector_store = Chroma.from_documents(
        chunks
        , embeddings
        , collection_metadata={"hnsw:space": "cosine"}
        , persist_directory=os.getenv("VECTOR_STORE_DIRECTORY")
        , collection_name= os.getenv("VECTOR_STORE_NAME")
    )




from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.schema.retriever import BaseRetriever
from typing import List
from pydantic import Field

class WaisRetriever(BaseRetriever):
    vector_store: Chroma = Field(description="Vector store to use for retrieval")
    window_size: int = Field(default=3, description="Size of the window around the most relevant document")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get the most relevant document
        results = self.vector_store.similarity_search_with_score(query, k=1)

        if not results:
            return []

        doc, score = results[0]
        doc_position = doc.metadata.get('position')

        if doc_position is None:
            return results

        # Fetch all documents
        all_docs = self.vector_store.get()

        # Filter and sort the documents based on ID
        relevant_near_vectors = []
        for _metadata, _doc in zip(all_docs['metadatas'], all_docs['documents']):
            if (_metadata['position'] >= doc_position - self.window_size) and (
                    _metadata['position'] <= doc_position + self.window_size):
                _doc, _score = \
                self.vector_store.similarity_search_with_score(query, k=1, filter={'position': _metadata['position']})[0]
                if (score / _score) > .75:
                    relevant_near_vectors.append(_doc)

        relevant_near_vectors.sort(key=lambda x: x.metadata.get('id', 0))

        return relevant_near_vectors

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # For async, we just call the sync version
        return self.get_relevant_documents(query)
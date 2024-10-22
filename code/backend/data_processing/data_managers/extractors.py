from abc import ABC, abstractmethod
from PyPDF2 import PdfReader
import io
from typing import List
from langchain_core.documents import Document

class Extractor(ABC):
    @abstractmethod
    def extract_documents(self, file_content: bytes) -> List:
        pass


class PdfExtractor(Extractor):

    def extract_documents(self, file_content: bytes) -> List:

        pdf_reader = PdfReader(io.BytesIO(file_content))
        content = []

        for index, page in enumerate(pdf_reader.pages):
            _doc = Document(
                    metadata = {'source_type' : 'pdf', 'page' : index}
                    , page_content= page.extract_text()
            )
            content.append(_doc)
        return content
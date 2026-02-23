from pypdf import PdfReader
from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import uuid
from config import CHUNK_OVERLAP,CHUNK_SIZE

def extract_data_from_pdf(uploaded_file):
    pdf_reader=PdfReader(uploaded_file)
    documents=[]

    for page_num,page in enumerate(pdf_reader.pages,1):
        text=page.extract_text()

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        'source':uploaded_file.name,
                        'page':page_num,
                        'total_pages':len(pdf_reader.pages)
                    }
                )
            )
    return documents

def split_documents(documents):
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','. ', ' ',''],
        length_function=len,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks=text_splitter.split_documents(documents)

    for i,chunk in enumerate(chunks):
        chunk.metadata['chunk_index']=i

    return chunks

def process_pdf(uploaded_file):
    pdf_id=str(uuid.uuid4())
    documents=extract_data_from_pdf(uploaded_file)
    chunks=split_documents(documents)

    return pdf_id,chunks
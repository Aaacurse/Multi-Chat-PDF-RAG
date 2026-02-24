import pytest
from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from utils import pdf_processor


def test_extract_data_from_pdf(mock_uploaded_file):
    documents=pdf_processor.extract_data_from_pdf(mock_uploaded_file)
    assert len(documents)>0
    assert all(isinstance(doc,Document) for doc in documents )
    
    for doc in documents:
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert "total_pages" in doc.metadata

def test_split_pdf(sample_documents):
    chunks=pdf_processor.split_documents(sample_documents)
    assert len(chunks)>=len(sample_documents)
    assert all(isinstance(chunk,Document) for chunk in chunks)
    
    for i,chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata['chunk_index']==i
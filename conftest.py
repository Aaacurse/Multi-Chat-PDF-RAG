import os
import sys
import json
import shutil
from pathlib import Path
from io import BytesIO
import pytest
from pypdf import PdfWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_classic.schema import Document
from langchain_classic.schema.messages import HumanMessage, AIMessage


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables"""
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "test_api_key_for_testing")
    yield


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directory"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    vector_dir = data_dir / "vector_stores"
    vector_dir.mkdir()
    
    return data_dir


@pytest.fixture
def test_state_file(test_data_dir):
    """Create temporary state file"""
    state_file = test_data_dir / "app_state.json"
    return state_file


@pytest.fixture
def sample_pdf_bytes():
    """Create a simple PDF in memory"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Page 1
    c.drawString(100, 750, "This is a test PDF document.")
    c.drawString(100, 730, "It contains multiple pages with sample text.")
    c.showPage()
    
    # Page 2
    c.drawString(100, 750, "This is the second page.")
    c.drawString(100, 730, "It discusses important topics like AI and RAG.")
    c.showPage()
    
    # Page 3
    c.drawString(100, 750, "Final page with conclusions.")
    c.showPage()
    
    c.save()
    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_pdf_file(sample_pdf_bytes, tmp_path):
    """Create a temporary PDF file"""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(sample_pdf_bytes.read())
    sample_pdf_bytes.seek(0)
    return pdf_path


@pytest.fixture
def mock_uploaded_file(sample_pdf_bytes):
    """
    Mock Streamlit uploaded file using BytesIO directly.
    This works with pypdf because BytesIO already has:
    read(), seek(), tell(), close(), etc.
    """
    sample_pdf_bytes.seek(0)

    # Add attributes Streamlit normally provides
    sample_pdf_bytes.name = "test_document.pdf"
    sample_pdf_bytes.type = "application/pdf"

    return sample_pdf_bytes


@pytest.fixture
def sample_documents():
    """Create sample Document objects"""
    return [
        Document(
            page_content="This is the first page of the document.",
            metadata={"source": "test.pdf", "page": 1, "total_pages": 3}
        ),
        Document(
            page_content="This is the second page with more information.",
            metadata={"source": "test.pdf", "page": 2, "total_pages": 3}
        ),
        Document(
            page_content="This is the third and final page.",
            metadata={"source": "test.pdf", "page": 3, "total_pages": 3}
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Create sample document chunks"""
    return [
        Document(
            page_content="Chunk 1 content about AI",
            metadata={"source": "test.pdf", "page": 1, "chunk_index": 0}
        ),
        Document(
            page_content="Chunk 2 content about RAG systems",
            metadata={"source": "test.pdf", "page": 1, "chunk_index": 1}
        ),
        Document(
            page_content="Chunk 3 content about vector databases",
            metadata={"source": "test.pdf", "page": 2, "chunk_index": 2}
        ),
    ]


@pytest.fixture
def sample_messages():
    """Create sample chat messages"""
    return [
        HumanMessage(content="What is this document about?"),
        AIMessage(content="This document discusses AI and machine learning."),
        HumanMessage(content="Tell me more about RAG."),
        AIMessage(content="RAG stands for Retrieval-Augmented Generation..."),
    ]

@pytest.fixture
def chat_manager(test_state_file, monkeypatch):
    """Create ChatManager with test state file"""
    # Monkeypatch the STATE_FILE in config
    import config
    monkeypatch.setattr(config, "STATE_FILE", test_state_file)
    
    from utils.chat_manager import ChatManager
    return ChatManager()


@pytest.fixture
def vector_store_manager(test_data_dir, monkeypatch):
    """Create VectorStoreManager with test directory"""
    import config
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", test_data_dir / "vector_stores")
    
    from utils.vector_store import VectorStoreManager
    return VectorStoreManager()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Add any cleanup logic here if needed
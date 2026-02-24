import pytest
from config import *

def test_configfile():
    assert HUGGINGFACEHUB_API_TOKEN is not None
    assert EMBEDDING_MODEL is not None
    assert LLM_MODEL is not None
    assert 0<=LLM_TEMP<=1
    assert TOP_K_RESULTS>0
    assert CHUNK_OVERLAP>0
    assert CHUNK_SIZE>0
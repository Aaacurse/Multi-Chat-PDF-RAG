from typing import List
from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema.messages import BaseMessage
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.schema.runnable import RunnableParallel
from langchain_classic.schema.output_parser import StrOutputParser

from langchain_core.runnables import RunnableLambda

from config import LLM_MODEL, LLM_TEMP


# -----------------------------
# Document Formatter
# -----------------------------
def format_docs(docs):
    """Format retrieved documents into a single string"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(
            f"[Source {i} - Page {doc.metadata.get('page', 'N/A')}]\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted)


format_docs_runnable = RunnableLambda(format_docs)


# -----------------------------
# MAIN RAG CHAIN
# -----------------------------
def create_custom_rag_chain(retriever, chat_history: List[BaseMessage]):
    """
    Returns a runnable that takes:
        {"question": str}
    And returns:
        str (answer)
    """

    # LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMP
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant analyzing a PDF document.

Use the following context from the document to answer the user's question.
If you cannot find the answer in the context, say so clearly.
Always cite the page number when referencing information.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # Wrap chat history properly
    chat_history_runnable = RunnableLambda(lambda _: chat_history)

    # Build LCEL chain
    chain = (
        RunnableParallel(
            {
                "context": itemgetter("question")
                           | retriever
                           | format_docs_runnable,

                "question": itemgetter("question"),

                "chat_history": chat_history_runnable
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# -----------------------------
# RAG CHAIN WITH SOURCES
# -----------------------------
def create_custom_rag_chain_with_sources(
    retriever,
    chat_history: List[BaseMessage]
):
    """
    Returns:
        {
            "answer": str,
            "sources": List[Document]
        }
    """

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMP
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant analyzing a PDF document.

Use the following context from the document to answer the user's question.
If you cannot find the answer in the context, say so clearly.
Always cite the page number when referencing information.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chat_history_runnable = RunnableLambda(lambda _: chat_history)

    # Step 1: Retrieve documents
    retrieval_chain = RunnableParallel(
        {
            "docs": itemgetter("question") | retriever,
            "question": itemgetter("question")
        }
    )

    # Step 2: Generate answer
    answer_chain = (
        RunnableParallel(
            {
                "context": itemgetter("docs") | format_docs_runnable,
                "question": itemgetter("question"),
                "chat_history": chat_history_runnable
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 3: Combine answer + sources
    full_chain = (
        retrieval_chain
        | RunnableParallel(
            {
                "answer": answer_chain,
                "sources": itemgetter("docs")
            }
        )
    )

    return full_chain
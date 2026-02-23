"""
PDF RAG Chat Application
========================
Multi-document chat application with RAG using FAISS and Google Gemini.

Features:
- Upload multiple PDFs
- Create multiple isolated chat sessions per PDF
- Context-aware conversations using custom LCEL chains
- Source citations with page numbers
- Persistent storage of chats and vector indices
"""

import streamlit as st
from pathlib import Path

from langchain_classic.schema.messages import HumanMessage, AIMessage

from config import PAGE_TITLE, PAGE_ICON, LAYOUT
from utils import (
    process_pdf,
    VectorStoreManager,
    ChatManager,
    create_custom_rag_chain_with_sources
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=PAGE_TITLE,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Managers
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager()
    
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    
    # Current selections
    if "current_pdf_id" not in st.session_state:
        st.session_state.current_pdf_id = None
    
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    
    # UI state
    if "processing" not in st.session_state:
        st.session_state.processing = False

initialize_session_state()

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_current_pdf_data():
    """Get current PDF data dictionary"""
    if st.session_state.current_pdf_id:
        return st.session_state.chat_manager.pdfs.get(st.session_state.current_pdf_id)
    return None

def get_current_chat_data():
    """Get current chat data dictionary"""
    pdf_data = get_current_pdf_data()
    if pdf_data and st.session_state.current_chat_id:
        return pdf_data["chats"].get(st.session_state.current_chat_id)
    return None

def switch_to_pdf_chat(pdf_id: str, chat_id: str):
    """Switch to a specific PDF and chat"""
    st.session_state.current_pdf_id = pdf_id
    st.session_state.current_chat_id = chat_id

def has_chats(pdf_id: str) -> bool:
    """Check if PDF has any chats"""
    return bool(st.session_state.chat_manager.pdfs[pdf_id]["chats"])

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - PDF UPLOAD & NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📁 Documents")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PDF UPLOAD SECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    st.subheader("Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        key="pdf_uploader",
        help="Upload a PDF document to chat with"
    )
    
    if uploaded_file:
        if st.button("Process PDF", use_container_width=True, type="primary"):
            
            st.session_state.processing = True
            
            with st.spinner("Processing PDF..."):
                try:
                    # Step 1: Extract and chunk PDF
                    pdf_id, chunks = process_pdf(uploaded_file)
                    st.success(f"✅ Extracted {len(chunks)} chunks")
                    
                    # Step 2: Create vector store
                    progress_text = st.empty()
                    progress_text.text("Creating vector embeddings...")
                    
                    vector_store_path = st.session_state.vector_store_manager.create_vector_store(
                        pdf_id, chunks
                    )
                    progress_text.text("✅ Vector store created")
                    
                    # Step 3: Register in chat manager
                    first_chat_id = st.session_state.chat_manager.add_pdf(
                        pdf_id,
                        uploaded_file.name,
                        vector_store_path
                    )
                    
                    # Step 4: Switch to new PDF
                    switch_to_pdf_chat(pdf_id, first_chat_id)
                    
                    st.success(f"✅ Successfully processed: {uploaded_file.name}")
                    st.session_state.processing = False
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.session_state.processing = False
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PDF & CHAT NAVIGATION
    # ─────────────────────────────────────────────────────────────────────────
    
    if st.session_state.chat_manager.pdfs:
        st.subheader("Your Documents")
        
        for pdf_id, pdf_data in st.session_state.chat_manager.pdfs.items():
            
            # PDF Expander
            with st.expander(
                f"📄 {pdf_data['filename'][:30]}{'...' if len(pdf_data['filename']) > 30 else ''}",
                expanded=(pdf_id == st.session_state.current_pdf_id)
            ):
                
                # PDF Info
                st.caption(f"**Filename:** {pdf_data['filename']}")
                st.caption(f"**Chats:** {len(pdf_data['chats'])}")
                
                # Delete PDF button
                if st.button(
                    "🗑️ Delete PDF",
                    key=f"del_pdf_{pdf_id}",
                    help="Delete this PDF and all its chats"
                ):
                    # Delete vector store from disk
                    st.session_state.vector_store_manager.delete_vector_store(pdf_id)
                    
                    # Delete from chat manager
                    st.session_state.chat_manager.delete_pdf(pdf_id)
                    
                    # Clear current selections if deleted
                    if st.session_state.current_pdf_id == pdf_id:
                        st.session_state.current_pdf_id = None
                        st.session_state.current_chat_id = None
                    
                    st.success("PDF deleted")
                    st.rerun()
                
                st.divider()
                
                # ─────────────────────────────────────────────────────────────
                # CHAT LIST
                # ─────────────────────────────────────────────────────────────
                
                if pdf_data["chats"]:
                    st.caption("💬 **Chats:**")
                    
                    for chat_id, chat_data in pdf_data["chats"].items():
                        
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            # Chat button
                            is_current = (
                                pdf_id == st.session_state.current_pdf_id
                                and chat_id == st.session_state.current_chat_id
                            )
                            
                            if st.button(
                                chat_data["title"][:25],
                                key=f"chat_{chat_id}",
                                use_container_width=True,
                                type="primary" if is_current else "secondary"
                            ):
                                switch_to_pdf_chat(pdf_id, chat_id)
                                st.rerun()
                        
                        with col2:
                            # Delete chat button
                            if st.button(
                                "🗑️",
                                key=f"del_chat_{chat_id}",
                                help="Delete this chat"
                            ):
                                st.session_state.chat_manager.delete_chat(pdf_id, chat_id)
                                
                                # If deleted current chat, switch to another or clear
                                if st.session_state.current_chat_id == chat_id:
                                    remaining_chats = list(pdf_data["chats"].keys())
                                    if remaining_chats:
                                        st.session_state.current_chat_id = remaining_chats[0]
                                    else:
                                        st.session_state.current_chat_id = None
                                
                                st.rerun()
                
                # ─────────────────────────────────────────────────────────────
                # NEW CHAT BUTTON
                # ─────────────────────────────────────────────────────────────
                
                if st.button(
                    "➕ New Chat",
                    key=f"new_chat_{pdf_id}",
                    use_container_width=True
                ):
                    new_chat_id = st.session_state.chat_manager.create_chat(pdf_id)
                    switch_to_pdf_chat(pdf_id, new_chat_id)
                    st.rerun()
    
    else:
        st.info("👆 Upload a PDF to get started")
    
    # ─────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────────
    
    st.divider()
    st.caption("Built with 🦜 LangChain & ⚡ FAISS")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.current_pdf_id and st.session_state.current_chat_id:
    
    # ─────────────────────────────────────────────────────────────────────────
    # GET CURRENT DATA
    # ─────────────────────────────────────────────────────────────────────────
    
    pdf_id = st.session_state.current_pdf_id
    chat_id = st.session_state.current_chat_id
    
    pdf_data = get_current_pdf_data()
    chat_data = get_current_chat_data()
    
    if not pdf_data or not chat_data:
        st.error("❌ Error: Could not load PDF or chat data")
        st.stop()
    
    # ─────────────────────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────────────────────
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title(f"💬 {chat_data['title']}")
    
    with col2:
        # Rename chat button
        if st.button("✏️ Rename", key="rename_chat"):
            st.session_state.show_rename = True
    
    st.caption(f"📄 Document: **{pdf_data['filename']}**")
    
    # Rename dialog
    if st.session_state.get("show_rename", False):
        with st.form("rename_form"):
            new_title = st.text_input(
                "New chat title:",
                value=chat_data["title"],
                max_chars=50
            )
            col1, col2 = st.columns(2)
            
            with col1:
                if st.form_submit_button("Save", use_container_width=True):
                    st.session_state.chat_manager.update_chat_title(
                        pdf_id, chat_id, new_title
                    )
                    st.session_state.show_rename = False
                    st.rerun()
            
            with col2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    st.session_state.show_rename = False
                    st.rerun()
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────────────
    # GET RETRIEVER
    # ─────────────────────────────────────────────────────────────────────────
    
    try:
        retriever = st.session_state.vector_store_manager.get_retriever(
            pdf_data["vector_store_path"]
        )
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        st.stop()
    
    # ─────────────────────────────────────────────────────────────────────────
    # DISPLAY CHAT HISTORY
    # ─────────────────────────────────────────────────────────────────────────
    
    messages = st.session_state.chat_manager.get_messages(pdf_id, chat_id)
    
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
                
                # Show sources if available
                if message.additional_kwargs.get("sources"):
                    with st.expander("📚 View Sources"):
                        sources = message.additional_kwargs["sources"]
                        for i, source in enumerate(sources, 1):
                            st.caption(f"**[{i}]** {source}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHAT INPUT
    # ─────────────────────────────────────────────────────────────────────────
    
    if user_input := st.chat_input(
        "Ask a question about this document...",
        key="chat_input"
    ):
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            
            with st.spinner("Thinking..."):
                try:
                    # Create RAG chain with chat history
                    chain = create_custom_rag_chain_with_sources(
                        retriever,
                        messages
                    )
                    
                    # Invoke chain
                    result = chain.invoke({"question": user_input})
                    
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    # Display answer
                    st.write(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(sources, 1):
                                page = doc.metadata.get("page", "?")
                                chunk_idx = doc.metadata.get("chunk_index", "?")
                                
                                st.caption(
                                    f"**[{i}]** Page {page} • Chunk {chunk_idx}"
                                )
                                
                                # Show snippet
                                content_preview = doc.page_content[:300]
                                if len(doc.page_content) > 300:
                                    content_preview += "..."
                                
                                st.text(content_preview)
                                st.divider()
                    
                    # Prepare source metadata for storage
                    source_metadata = [
                        f"Page {doc.metadata.get('page', '?')}"
                        for doc in sources
                    ]
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    answer = "I encountered an error processing your question."
                    source_metadata = []
        
        # ─────────────────────────────────────────────────────────────────────
        # SAVE MESSAGES TO HISTORY
        # ─────────────────────────────────────────────────────────────────────
        
        # Save user message
        st.session_state.chat_manager.add_message(
            pdf_id,
            chat_id,
            HumanMessage(content=user_input)
        )
        
        # Save assistant message with source metadata
        st.session_state.chat_manager.add_message(
            pdf_id,
            chat_id,
            AIMessage(
                content=answer,
                additional_kwargs={"sources": source_metadata}
            )
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # AUTO-UPDATE CHAT TITLE
        # ─────────────────────────────────────────────────────────────────────
        
        if len(messages) == 0 and chat_data["title"] == "New Chat":
            # Generate title from first user message
            title = user_input[:30]
            if len(user_input) > 30:
                title += "..."
            
            st.session_state.chat_manager.update_chat_title(
                pdf_id,
                chat_id,
                title
            )
        
        # Rerun to show new messages
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN (No PDF selected)
# ══════════════════════════════════════════════════════════════════════════════

else:
    
    # Center content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("📄 PDF RAG Chat")
        
        st.markdown("""
        ### Welcome! 👋
        
        Chat with your PDF documents using AI-powered retrieval augmented generation.
        
        #### ✨ Features
        
        - 📄 **Upload multiple PDFs** — Each document gets its own vector database
        - 💬 **Multiple chat sessions** — Create unlimited conversations per PDF
        - 🧠 **Context-aware** — AI remembers your conversation history
        - 📚 **Source citations** — Every answer includes page references
        - 💾 **Persistent storage** — Your chats are saved automatically
        
        #### 🚀 Get Started
        
        1. Click **"Upload PDF"** in the sidebar
        2. Select a PDF document from your computer
        3. Click **"⚡ Process PDF"** to create the vector index
        4. Start asking questions in the chat!
        
        #### 💡 Tips
        
        - Create multiple chats to organize different topics
        - Click on source citations to see which pages were referenced
        - Rename chats to keep them organized
        - Use the sidebar to switch between different documents
        
        ---
        
        #### 🔧 Powered By
        
        - 🦜 **LangChain** — AI orchestration framework
        - ⚡ **FAISS** — Fast vector similarity search
        - 🧠 **Google Gemini** — Advanced language model
        - 🎈 **Streamlit** — Interactive web interface
        """)
        
        st.info("👈 **Upload a PDF from the sidebar to begin!**")
        
        # Stats (if any PDFs exist)
        if st.session_state.chat_manager.pdfs:
            st.divider()
            
            total_pdfs = len(st.session_state.chat_manager.pdfs)
            total_chats = sum(
                len(pdf["chats"])
                for pdf in st.session_state.chat_manager.pdfs.values()
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📄 Documents", total_pdfs)
            
            with col2:
                st.metric("💬 Total Chats", total_chats)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS (Optional)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ===== GLOBAL BACKGROUND ===== */
    .stApp {
        background-color: #0E1117;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #2A2D34;  /* Divider line */
    }

    /* Make divider stronger if needed */
    section[data-testid="stSidebar"] {
        border-right: 2px solid #2A2D34;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* Better button radius */
    .stButton button {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)
import shutil
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import VECTOR_STORE_DIR,EMBEDDING_MODEL,TOP_K_RESULTS

class VectorStoreManager:
    def __init__(self):
        self.embeddings=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


    def create_vector_store(self,pdf_id,chunks):
        vector_store_path=VECTOR_STORE_DIR/f'faiss_index_{pdf_id}'

        vector_store=FAISS.from_documents(chunks,self.embeddings)

        vector_store.save_local(str(vector_store_path))

        return str(vector_store_path)
    
    def load_vector_store(self,vector_store_path):
        return FAISS.load_local(
            vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def get_retriever(self,vector_store_path,k=TOP_K_RESULTS):
        vector_store=self.load_vector_store(vector_store_path)
        return vector_store.as_retriever(search_kwargs={'k':k})
    

    def delete_vector_store(self, pdf_id):
        vector_store_path = VECTOR_STORE_DIR / f"faiss_index_{pdf_id}"
        
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
            print(f"Deleted vector store: {vector_store_path}")
        else:
            print("Vector store not found.")
    

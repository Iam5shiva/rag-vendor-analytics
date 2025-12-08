import logging
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest_docs import load_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def create_and_store_vector_store(doc_dir:str, save_path: str):

    try:
        # Load documents
        docs = load_documents(doc_dir)
        if not docs:
            raise ValueError("No documents to create vector store")
        
        logging.info(f"Total documents for embedding: {len(docs)}")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        logging.info(f"Total chunks created: {len(chunks)}")

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        logging.info("Embeddings model loaded")

        # Create FAISS vector store
        # texts = [doc.page_content for doc in chunks]
        # vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

        logging.info("FAISS vector store created")

        # Save the vector store
        vectorstore.save_local(save_path)
        logging.info(f"Vector store created and saved at: {save_path}")

    except Exception as e:
        logging.error(f"Error creating vector store: {str(e)}")

if __name__ == "__main__":
    DATA_DIR = "data/reports"
    VECTOR_STORE_PATH = "data/vectorstore/faiss_index"
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    create_and_store_vector_store(DATA_DIR, VECTOR_STORE_PATH)    
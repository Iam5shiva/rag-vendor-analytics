from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List
import logging
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def load_documents(directory:str) -> List[Document]:
    """Load PDF and CSV documents from a directory and return a list of Document objects."""

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    all_docs = []

    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)

        if not os.path.isfile(full_path):
             continue
        
        _, extension = os.path.splitext(filename)

        try:
            if extension.lower() == ".pdf":
                loader = PyPDFLoader(full_path)
                docs = loader.load()
                for page in docs:
                    page.metadata["source"] = filename
                logging.info(f"Loaded pdf: {filename} ({len(docs)} pages)")

            elif extension.lower() == ".csv":
                loader = CSVLoader(full_path)
                docs = loader.load()
                for page in docs:
                    page.metadata["source"] = filename
                logging.info(f"Loaded csv: {filename} ({len(docs)} records)")
            
            else:
                 logging.warning(f"Skipping unsupported filetype {filename}")
            
            all_docs.extend(docs)
        
        except Exception as e:
             logging.error(f"Error loading file {filename}:{e}")

    if not all_docs:
         logging.warning("No documents were loaded from the directory")

    return all_docs

if __name__ == "__main__":
    docs = load_documents("data/reports")
    # for d in docs[:3]:
    #     print(d.metadata)
    logging.info(f"Total documents loaded: {len(docs)}")     
             
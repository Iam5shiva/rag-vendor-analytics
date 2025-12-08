# src/rag_pipeline.py
import logging
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
#from ollama_llm import get_ollama_llm
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def format_context(chunks: List[str]) -> str:
    """
    Number chunks to improve grounding and reduce hallucination.
    Example:

    [1]
    chunk text

    [2]
    chunk text
    """
    formatted = []
    for i, c in enumerate(chunks, start=1):
        formatted.append(f"[{i}]\n{c}")
    return "\n\n".join(formatted)


# ---------- Vector store loader ----------
def build_vector_store(vector_store_path: str):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        logger.info("Loading FAISS vector store...")
        vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise


# ---------- RAG chain builder ----------
def build_rag_chain(vectorstore):
    if vectorstore is None:
        logger.error("Vectorstore is None — cannot build RAG chain.")
        return None

    #llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    llm = Ollama(model="qwen2.5:7b", temperature=0.0)

    #llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.0)

    prompt = ChatPromptTemplate.from_template("""
You are a procurement analyst assistant.

Use ONLY the information from the numbered context sections below.
Do not rely on external knowledge.

If the answer is not present in the context, say:
"The answer is not available in the documents."

Context Sections:
{context}

Question:
{question}

Answer clearly and succinctly.                                             
""")
#Here is what I did find instead: <summaries>.
#At the end of your answer, add a line starting with "Sources:" and list the document names used to answer.
#Answer clearly and succinctly.
#"No document contains information related to this query. Here is what I did find instead: <summaries>."

    rag_chain = (
        RunnableLambda(lambda inp: inp)  
        | prompt
        | llm
    )

    logger.info("RAG pipeline built successfully.")
    return rag_chain


def run_rag_for_eval(vectorstore, rag_chain, query: str, top_k: int = 3, use_mock: bool = False) -> Tuple[str, List[str]]:
    """
    Run retrieval + generation in a way that's friendly for evaluation tools.
    Returns:
      answer: str (model output, which will include "Sources:" line as per Option B)
      contexts: List[str] (list of retrieved chunk strings)
    """
    try:
        retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": top_k})
        docs = retriever.invoke(query)  
        contexts = [d.page_content for d in docs]

        # if use_mock:
        #     source_names = list({d.metadata.get("source", "Unknown") for d in docs})
        #     source_str = ", ".join(source_names)
        #     mock_answer = f"[MOCK] Answer for: {query}\nSources: {source_str if source_str else 'None'}"
        #     return mock_answer, contexts

        # Build a single context string for the prompt (same behavior as pipeline)
        #context_text = "\n\n".join(contexts)
        context_text = format_context(contexts)
        prompt_inputs = {"context": context_text, "question": query}
        result = rag_chain.invoke(prompt_inputs)

        answer = result.content if hasattr(result, "content") else str(result)

        return answer, contexts

    except Exception as e:
        logger.error(f"Error in run_rag_for_eval: {e}")
        return None, []

if __name__ == "__main__":
    VECTOR_STORE_PATH = "data/vectorstore/faiss_index"
    vectorstore = build_vector_store(VECTOR_STORE_PATH)
    rag_chain = build_rag_chain(vectorstore)

    examples = [
        "Which vendors had the highest delays and penalties in Q1 2024?",
        "Summarize the key points from the procurement policy.",
    ]
    for q in examples:
        ans, ctxs = run_rag_for_eval(vectorstore, rag_chain, q)
        print("\nQUESTION:", q)
        print("ANSWER:\n", ans)
        print("RETRIEVED CONTEXTS:", len(ctxs))

# Vendor Payment Insights – RAG Pipeline with Retrieval Evaluation (RAGAS)

This project builds a Retrieval-Augmented Generation (RAG) system for answering procurement and vendor-payment questions using internal documents.  
It includes:

- a FAISS-based retriever  
- a RAG pipeline powered by LLMs (Gemini for generation, Groq/Ollama for evaluation)  
- full RAG evaluation using RAGAS  
- reproducible scripts for ingestion, retrieval, and scoring  

It’s designed to answer questions like:

- What is the payment delay in Q4?
- Which vendor has the highest payout?
- Summarize vendor contract terms.

---

## 1. Project Structure

RAG-Vendor-Analytics/
│
├── data/
│ └── vectorstore/faiss_index
| └── reports/
│
├── src/
| ├── __init__.py
│ ├── rag_pipeline.py
│ ├── eval_ragas.py
│ ├── embed_store.py
│ ├── ingest_docs.py
| ├── app.py
│
├── requirements.txt
├── README.md
└── .env

---

## 2. What the System Does

### Step 1 — Embedding and Indexing
Uses `sentence-transformers/all-mpnet-base-v2` to embed procurement files.  
Indexing is done using **FAISS** for fast vector search.

### Step 2 — Retrieval
For every user query:

1. Search FAISS  
2. Retrieve the top-k relevant chunks  
3. Feed them as context to the LLM  

The retriever is tuned for high context precision.

### Step 3 — Answer Generation (RAG)
The system uses:

- **Gemini 2.5 Flash** (generation)
- A strict context-only prompt

This helps control hallucination and improves faithfulness.

### Step 4 — RAG Evaluation (RAGAS)
RAGAS evaluates:

- **Faithfulness -** Whether the generated answer is fully supported by the retrieved context.
- **Answer Relevancy -** How well the generated answer addresses the user’s question.  
- **Context Precision -** How much of the retrieved context is actually relevant to the answer.  

These help validate the quality of the RAG pipeline.

---

## 3. RAGAS Results

| Metric             | Score  |
|--------------------|--------|
| Faithfulness       | 0.8151 |
| Answer Relevancy   | 0.5772 |
| Context Precision  | 0.8704 |

**Interpretation:**

- High precision → strong retrieval - A score of **0.87** indicates strong retrieval quality. Most retrieved chunks directly contribute to answering the question.
- Good faithfulness → grounded answers - A score of **0.81** indicates that most answers are well-supported by retrieved vendor and procurement data, with minimal fabrication.
- Relevancy is decent and can improve with chunking or better prompt tuning - A score of **0.57** suggests that answers generally stay on topic, but some responses could be more direct or better focused on the exact intent of the query.  

These scores are solid for a real-world RAG system.

---

## 4. Tech Stack

- Python 3.10+
- LangChain
- FAISS
- HuggingFace Sentence Transformers
- Gemini 2.5 Flash
- Groq Llama-3.3
- Ollama (optional)
- RAGAS

## 5. This project demonstrates:

- Real-world document QA skills  
- Understanding of retrieval and embeddings  
- Ability to measure RAG quality  
- Experience with LLM APIs (Gemini, Groq, Ollama)  
- Ability to debug, optimize, and interpret RAG pipelines 

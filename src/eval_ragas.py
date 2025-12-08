import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

from rag_pipeline import build_vector_store, build_rag_chain, run_rag_for_eval
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
#from ollama_llm import get_ollama_llm
from langchain_community.llms import Ollama

from datasets import Dataset as HFDataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper


VECTOR_STORE_PATH = "data/vectorstore/faiss_index"
vectorstore = build_vector_store(VECTOR_STORE_PATH)
rag_chain = build_rag_chain(vectorstore)

evaluation_questions = [
    "What is payment delay in Q4?",
    "Which vendor has the highest total payout?",
    "What are the key factors affecting vendor delays?",
    "Which vendors had the highest delays and penalties in Q1 2024",
    "What is Vendor Invoicing & Payment Terms?",
    "Summarize contract details for all vendors",
    "I'm a new vendor. Whom should I contact for onboarding?",
    "Summarize performance metrics for all vendors",
    "Total penalties incurred by vendors in 2024?",
    "Summarize contract terms for vendors"
]
#    "What is the average payment delay overall?","Which quarter shows the biggest payment variation?",

rows = []
for q in evaluation_questions:
    answer, source_docs = run_rag_for_eval(vectorstore, rag_chain, q)

    print("\n====================")
    print("QUESTION:", q)
    print("ANSWER:", answer)

    if isinstance(source_docs, list):
        contexts = [doc if isinstance(doc, str) else doc.page_content for doc in source_docs]
    else:
        contexts = []

    rows.append({
        "question": q,
        "answer": answer,
        "contexts": contexts,
        "reference": ""
    })

df = pd.DataFrame(rows)

dataset = HFDataset.from_pandas(df)

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

MODEL = "groq" 

if MODEL == "gemini":
    base_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
        #convert_system_message_to_human=True
    )

elif MODEL == "ollama":
    #base_llm = get_ollama_llm("qwen2.5:7b")
    base_llm = Ollama(model="qwen2.5:7b", temperature=0.0)

elif MODEL == "groq":
    base_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

llm = LangchainLLMWrapper(base_llm)


print("\nRunning Ragas...")

ragas_score = evaluate(
    dataset=dataset,
    metrics=metrics,
    embeddings=embedding_model,
    llm=llm
)

print("\n===== RAGAS SCORES =====")
print(ragas_score)
print("\nDetailed Scores:")
print(ragas_score.to_pandas())

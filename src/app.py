import rag_pipeline as rg
import streamlit as st
import time

@st.cache_resource
def load_rag_pipeline():
    VECTOR_STORE_PATH = "data/vectorstore/faiss_index"
    vectorstore = rg.build_vector_store(VECTOR_STORE_PATH)
    rag_chain = rg.build_rag_chain(vectorstore)
    return vectorstore, rag_chain

vectorstore, rag_chain = load_rag_pipeline()

def main():
    st.set_page_config(page_title="Procurement Insights RAG", layout="wide")
    st.title("Procurement Insights - RAG Assistant")
    st.caption("An AI-powered assistant to answer procurement-related queries using internal documents")    

    st.markdown(
        """
        <style>
        .answer-box {
            background-color: #ffffff;
            border: 1.5px solid #2b7de9;
            padding: 1.3rem;
            border-radius: 10px;
            font-size: 1.1rem;
            line-height: 1.65;
            color: #000000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            font-weight: 500;
        }

        .source-box {
            background-color: #f4f6fb;
            border-left: 4px solid #6b7280;
            color: #000000;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 0.7rem;
            font-size: 0.95rem;
        }
        .status {
            font-weight: 600;
            color: green;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    if rag_chain is None:
        st.error("RAG pipeline failed to initialize. Check logs.")
        return

    user_query = st.text_input("Enter your question:")
    submit = st.button("Get Answer")

    if submit and user_query.strip():

        status_placeholder = st.empty()
        #status_placeholder.info("🧠 Retrieving relevant documents...")
        status_placeholder.markdown('<p class="status">Retrieving relevant documents...</p>', unsafe_allow_html=True)
        time.sleep(0.5)  

        with st.spinner("Thinking..."):


            try:
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                docs = retriever.invoke(user_query)
                contexts = [d.page_content for d in docs]
                context_text = "\n\n".join(contexts)

                prompt_inputs = {"context": context_text, "question": user_query}

                result = rag_chain.invoke(prompt_inputs)
                status_placeholder.markdown('<p class="status">Retrieved and generated successfully</p>', unsafe_allow_html=True)

                st.markdown("### Answer")
                st.markdown(f"<div class='answer-box'>{result.content if hasattr(result, 'content') else result}</div>", unsafe_allow_html=True)
                st.divider()
                st.markdown("### Sources")
                # status_placeholder.success("Retrieved and generated successfully...")
                # st.subheader("Answer")
                # st.write(result.content)
                
                st.caption("Sources (from retrieved documents):")
                source_names = [d.metadata.get("source", "Unknown") for d in docs]
                source_str = ", ".join(source_names) if source_names else "No sources found"
                st.markdown(f"<div class='source-box'>{source_str}</div>", unsafe_allow_html=True)

                # if hasattr(result, "content"):
                #     raw_output = result.content
                # elif isinstance(result, dict):
                #     raw_output = result.get("answer", "")
                # else:
                #     raw_output = str(result)

                # Clean splitting to avoid "Sources:" leaking into the answer
                # if "Sources:" in raw_output:
                #     answer_text = raw_output.split("Sources:")[0].strip()
                #     sources_text = raw_output.split("Sources:")[1].strip()
                # else:
                #     answer_text = raw_output.strip()
                #     sources_text = None

                # Render Answer
                # st.markdown("### Answer")
                # st.markdown(f"<div class='answer-box'>{answer_text}</div>", unsafe_allow_html=True)

                # # Render Sources only if you want (optional)
                # if sources_text:
                #     st.markdown("### Sources")
                #     st.markdown(f"<div class='source-box'>{sources_text}</div>", unsafe_allow_html=True)


            except Exception as e:
                st.error(f"Error while processing your query: {e}")

if __name__ == "__main__":
    main()

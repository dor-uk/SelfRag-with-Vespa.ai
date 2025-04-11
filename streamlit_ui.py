import streamlit as st
import torch
from vespa_indexing import indexing_main
from vespa_retrieval import retrieve_documents
from generation import generation_pipeline
from answer_check import answer_check

# Streamlit Page Config
st.set_page_config(page_title="Self-RAG Chat", layout="wide")


# Title
st.title("Self-RAG System for Document Retrieval & QA")

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        - This tool retrieves relevant documents from **Vespa** and generates answers using a local **LLM (Phi-2)**.
        
        """
    )

st.subheader("Self-RAG Workflow")
st.image("Question.png", caption="Self-RAG Process Flow", width=800)

# Text Input for User Query
query = st.text_input(" Enter your question:", "")

# Button to Start Processing
if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a question first!")
    else:
        #indexing_main()

        with st.spinner("Retrieving relevant documents..."):
            retrieved_chunks = retrieve_documents(query)

        if not retrieved_chunks:
            st.error("No relevant documents found. Try a different query.")
        else:
            # Show Retrieved Chunks
            st.subheader("Retrieved Documents")
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.markdown(f"**Topic:** {chunk.get('topic', 'Unknown Topic')}")
                st.info(chunk["content"])
            # Generate Answer
            with st.spinner("Generating answer..."):
                answer, score = generation_pipeline(query, retrieved_chunks)

            
            
            # Display Answer & Score
            st.subheader("Final Answer")
            st.success(answer)

            # Show Score & Decision
            st.subheader(" Answer Quality Score")
            st.write(f"Score: {score}")

            # if score <= 2:
            #     st.warning("Low-quality answer detected! Retrying retrieval & generation...")
            #     retrieved_chunks = retrieve_documents(query)
            #     answer,score = generation_pipeline(query, retrieved_chunks)
            #     score = int(answer_check(query, answer))
            #     st.success(f"Improved Answer: {answer}")
            #     st.write(f"New Score: **{score}/5**")

            st.balloons()  

# chatbot.py

import streamlit as st
from transformers import pipeline

# Cache the pipeline so it loads only once
@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def show_chatbot():
    st.header("AI Chatbot")
    st.write("Ask any questions about your data, forecasts, or insights. Provide a block of text as context, and then type your question below.")

    # Load the QA pipeline
    qa_pipeline = load_qa_pipeline()

    # Let the user supply a context (for example, a summary of their data or insights)
    context = st.text_area("Context (paste your summary or data description here):", height=150, 
                           help="This context will be used by the QA model to answer your question. You can use the automated insights or business impact text as context.")

    # Input for the question
    question = st.text_input("Enter your question:")

    if st.button("Get Answer") and question:
        if context.strip():
            try:
                result = qa_pipeline(question=question, context=context)
                st.markdown("**Answer:** " + result['answer'])
            except Exception as e:
                st.error(f"Error generating answer: {e}")
        else:
            st.info("Please provide some context for the QA model to use.")

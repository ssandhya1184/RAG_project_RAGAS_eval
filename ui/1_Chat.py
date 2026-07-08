import streamlit as st
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
print(root_dir)

from src.rag_builder import build_pipeline
from rag_evaln import run_evaluation
from src.monitoring.metrics import (RAG_QUERY_COUNTER, RAG_QUERY_DURATION)
from prometheus_client import start_http_server

@st.cache_resource
def start_metrics():
    start_http_server(8001)

#Start Prometheus Metrics server
start_metrics()

st.set_page_config(page_title="Insurance Policy Assistant",layout="wide")

st.header("🏥 Insurance Policy Assistant")

# Build pipeline once
@st.cache_resource
def load_pipeline():
    return build_pipeline()

pipeline = load_pipeline()

summary_query = """
Summarize this insurance policy in 10 bullet points.
"""

summary, _, _ = pipeline(summary_query)

@st.cache_data
def get_summary():
    summary, _, _ = pipeline(
        "Summarize this insurance policy in 5 bullet points"
    )
    return summary

with st.sidebar:
    st.divider()

    st.subheader("📊 RAG Evaluation")

    if st.button("Run RAGAS Evaluation"):
        with st.spinner("Running RAGAS Evaluation... This may take a minute."):
            try:
                run_evaluation()
                st.success("Evaluation completed successfully.")

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
    
    if st.button("View Evaluation Dashboard"):
        st.switch_page("pages/rag_eval_ui.py")

        

# Display summary
st.markdown(get_summary())

#--------------------
# Chat History
#-------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])


#------------
# User Input
#-----------

user_question = st.chat_input("Ask a question about Insurance Policy...")

if user_question:
    # Increment the RAG_QUERY_COUNTER
    RAG_QUERY_COUNTER.inc()
    # Append user question to the state
    st.session_state.messages.append(
        {
            'role' : 'user',
            'content' : user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

        #Generate answers
        with RAG_QUERY_DURATION.time():
            answer, docs_before, docs_after = pipeline(user_question)

    #Display assistant answer
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Retrieved Context"):

            retrieved_docs = (
                docs_after
                if docs_after
                else docs_before
            )

            for i, doc in enumerate(retrieved_docs, start=1):

                st.markdown(
                    f"### Context {i}"
                )

                st.text(doc.page_content[:1000])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
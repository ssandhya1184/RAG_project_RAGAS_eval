import streamlit as st
import pandas as pd
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
print(root_dir)

st.set_page_config(
    page_title= "RAGAS Evaluation Dashboard",
    layout="wide"
    )

st.title("📊 RAG Evaluation Dashboard")

overall_metrics_file = "outputs/experiment_results.csv"
question_wise_file = "outputs/question_wise_results.csv"



# Overall REsults

st.header("Overall Metrics Summary")

if os.path.isfile(overall_metrics_file):
    overall_metrics_df = pd.read_csv(overall_metrics_file)

    st.dataframe(overall_metrics_df, width="stretch")

    latest = overall_metrics_df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Context Recall",
        round(latest["context_recall"], 3)
    )

    col2.metric(
        "Context Precision",
        round(latest["context_precision"], 3)
    )

    col3.metric(
        "Faithfulness",
        round(latest["faithfulness"], 3)
    )
   
else:
    st.warning("No overall report found.")


# Question wise Result

if os.path.isfile(question_wise_file):
    question_wise_df = pd.read_csv(question_wise_file)
    display_df = question_wise_df[['question','chunking','retrieval','reranker','context_recall','context_precision','faithfulness']]
    st.dataframe(display_df,width="stretch")
    for __, row in question_wise_df.iterrows():

        with st.expander(row["question"]):
            strategy = f"{row['chunking']} / {row['retrieval']} / {row['reranker']}"
            
            st.info(strategy)
            st.write("Ground Truth")
            st.info(row["ground_truth"])

            st.write("Answer")
            st.success(row["answer"])

            st.write("Retrieved Context")
            st.code(row["context"])
else:
    st.warning("Question wise Report not found.")

from prometheus_client import Counter, Histogram, Gauge

#-----------------
# Chat Metrics
#-----------------
RAG_QUERY_COUNTER = Counter("rag_queries_total","Total number of user questions processed")

RAG_QUERY_DURATION = Histogram("rag_queries_duration_seconds","Time taken to process one RAG Query")


# ------------------
# Retrieval Metrics
# ------------------

RETREIVAL_DURATION = Histogram("rag_retrieval_duration_seconds", "Time taken by the retriever")

RETRIEVED_DOCUMENTS = Histogram("rag_retrieved_documents","No of documents retrieved")

# -------------------------
# Pipeline Metrics
# -------------------------


RERANK_DURATION = Histogram("rag_rerank_duration_seconds","Time spent reranking documents")

LLM_DURATION = Histogram("rag_llm_duration_seconds","Time spent generating the answer")

RETRIEVED_DOC_COUNT = Histogram("rag_retrieved_doc_count","No of retrieved documents")


#---------------
# RAGAS Metrics
#---------------
RAG_CONTEXT_PRECISION = Gauge("rag_context_precision","Latest Context Precision Score")

RAG_CONTEXT_RECALL = Gauge("rag_context_recall","Latest Context Recall Score")

RAG_FAITHFULNESS = Gauge("rag_faithfulness","Latest RAG Faithfulness Score")

RAG_ANSWER_CORRECTNESS = Gauge("rag_answer_correctness","Latest RAG Answer Correctness Score")

INPUT_TOKEN_COUNT = Histogram("input_token_count","Count of Prompt + Retreived Context")

OUTPUT_TOKEN_COUNT = Histogram("output_token_count","Response Token Count")

TOTAL_TOKEN_COUNT = Histogram("total_token_count","Total Token Count")


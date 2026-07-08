from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils import count_tokens
from src.reranker import Reranker
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import time
from src.monitoring.metrics import (RETREIVAL_DURATION, 
                                    RERANK_DURATION, 
                                    LLM_DURATION, 
                                    RETRIEVED_DOC_COUNT,
                                    INPUT_TOKEN_COUNT,
                                    OUTPUT_TOKEN_COUNT,
                                    TOTAL_TOKEN_COUNT)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Build RAG chain
def build_rag_chain(retriever,llm):
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Use the context below to answer the question.
    If you don't know the answer, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """
    )

    chain = (
        {
        "context": retriever | format_docs,
        "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain 

# Build RAG chain

def build_rag_pipeline(retriever, llm, reranker, config):
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Use the context below to answer the question.
    If you don't know, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """)

    def pipeline(query):
        # 1. Retrieve
        docs_after = None

        start = time.perf_counter()
        docs_before = retriever.invoke(query)
        
        RETREIVAL_DURATION.observe(
            time.perf_counter() - start
        )

        RETRIEVED_DOC_COUNT.observe(len(docs_before))
        # 2. Rerank (if enabled)
        if reranker and config.ENABLE_RERANKING:
            
            start = time.perf_counter()
            docs_after = reranker.rerank(
                query,
                docs_before,
                top_k=config.RERANK_TOP_K
            )
            RERANK_DURATION.observe(time.perf_counter() - start)

        # 3. Format
        retrieved_docs = docs_after if docs_after else docs_before
        context = format_docs(retrieved_docs)

        input_tokens = count_tokens(context)
        print(f"Count tokens------------->{input_tokens}")
        INPUT_TOKEN_COUNT.observe(input_tokens)

        # 4. LLM
        llm_start = time.perf_counter()
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": query
        })

        response_tokens = count_tokens(response)
        OUTPUT_TOKEN_COUNT.observe(response_tokens)
        total_tokens = input_tokens + response_tokens
        TOTAL_TOKEN_COUNT.observe(total_tokens)
        LLM_DURATION.observe(time.perf_counter() - llm_start)

        return response, docs_before, docs_after

    return pipeline
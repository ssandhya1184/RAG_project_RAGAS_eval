from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.reranker import Reranker
from src.utils import count_tokens

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
        docs_before = retriever.invoke(query)
        
        # 2. Rerank (if enabled)
        if reranker and config.ENABLE_RERANKING:
            
            docs_after = reranker.rerank(
                query,
                docs_before,
                top_k=config.RERANK_TOP_K
            )

        # 3. Format
        context = format_docs(docs_before)

        used_tokens = count_tokens(context)
        print(f"Count tokens------------->{used_tokens}")

        # 4. LLM
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": query
        })

        return response, docs_before, docs_after

    return pipeline
from src.ingestion import load_pdf
from src.embedding import get_embeddings
from src.chunking import get_chunks
from src.retrieval import get_retriever
from src.pipeline import build_rag_pipeline
from src.vector_store import create_vectorstore
from src.reranker import Reranker
from src.utils import get_gemini_gemma_llm,get_llama_groq_llm
from src import config


def build_pipeline():
    print("Hello from rag-project-ragas-eval!")

    # Load the document
    documents = load_pdf(config.PDF_PATH)

    embeddings = get_embeddings(config.EMBEDDING_MODEL)


    docs = get_chunks(documents=documents, config=config, embeddings=embeddings)
    print("\n====================")
    #print(f"Strategy: {strategy}")
    print(f"Chunks: {len(docs)}")

    # Vector DB
    vector_store = create_vectorstore(docs = docs, embeddings = embeddings)

    # Retriever
    retriever = get_retriever(config = config, docs= docs, vector_store= vector_store)
    
    # Reranker
    reranker = Reranker(config.RERANK_MODEL) if config.ENABLE_RERANKING else None

    llm_llama = get_llama_groq_llm()

    # Build the chain
    #chain = build_rag_chain(retriever=retriever,llm = llm)
    pipeline = build_rag_pipeline(
            retriever,
            llm_llama,
            reranker,
            config
        )
    
    return pipeline

    
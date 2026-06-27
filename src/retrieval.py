from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


def get_similarity_retriever(vector_store, k=3):
    return vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs={'k':k}
        )

def get_mmr_retriever(vector_store, k, fetch_k, lambda_mult):
    return vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k" : k,
            "fetch_k" : fetch_k,
            "lambda_mult" : lambda_mult
        }
    )

def get_hybrid_retriever(docs, vector_store, k):
    bm25 = BM25Retriever.from_documents(documents=docs)
    bm25.k = k

    dense = vector_store.as_retriever(search_kwargs={"k":k})

    ensemble = EnsembleRetriever(retrievers = [dense,bm25], weights = [0.5,0.5])

    return ensemble


def get_retriever(config, docs, vector_store):
    strategy = config.RETRIEVAL_STRATEGY
    print(f"Retrieval strategy is -> {strategy}")

    if strategy == "similarity":
        return get_similarity_retriever(vector_store= vector_store, k = config.TOP_K)
    elif strategy == "mmr":
        return get_mmr_retriever(vector_store=vector_store,k= config.TOP_K, fetch_k=config.MMR_K, lambda_mult=config.MMR_LAMBDA )
    elif strategy == "hybrid":
        return get_hybrid_retriever(docs=docs, vector_store= vector_store, k=config.TOP_K)
    else:
        raise ValueError(f"Unknown Retrieval Strategy-> {strategy}")


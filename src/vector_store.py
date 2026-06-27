from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

def create_vectorstore(docs, embeddings):
    return FAISS.from_documents(docs,embeddings,
                                distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT)
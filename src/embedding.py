from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings(model_name="BAAI/bge-small-en"):
    return HuggingFaceEmbeddings(model_name=model_name)
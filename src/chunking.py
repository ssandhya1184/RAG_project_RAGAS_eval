from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from src import config


def get_text_splitter(chunk_size=500, chunk_overlap=100):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )

def fixed_chunking(documents,chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
    return splitter.split_documents(documents)

def semantic_chunking(documents, embeddings):
    splitter = SemanticChunker(embeddings=embeddings)
    return splitter.split_documents(documents)


def hierarchical_chunking(documents):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)

    parent_docs = parent_splitter.split_documents(documents=documents)
    child_docs = child_splitter.split_documents(documents=parent_docs)

    return child_docs


def get_chunks(documents,config, embeddings = None):
    strategy = config.CHUNKING_STRATEGY

    if strategy == "fixed":
        return fixed_chunking(documents=documents, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    elif strategy == "semantic":
        if embeddings == None:
            raise ValueError("Semantic Chunking requires Embeddings.")
        return semantic_chunking(documents=documents,embeddings=embeddings)
    elif strategy == "hierarchical":
        return hierarchical_chunking(documents=documents)
    else:
        raise ValueError(f"Unknown chunking Strategy ->{strategy}")




def split_documents(documents,splitter):
    return splitter.split_documents(documents)
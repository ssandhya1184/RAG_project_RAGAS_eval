from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    print(f"Entering load_pdf with path->{file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
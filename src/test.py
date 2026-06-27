from langchain_community.document_loaders import PyPDFLoader


#Load the document 
file_path = "src\Ramayana_Mahabharata.pdf"
#ocr_pdf()
loader = PyPDFLoader(file_path)
print("after loading")
documents = loader.load()
documents1 = documents[54:57]

#for i, doc in enumerate(documents[40:60]):
#   print(f"Page {i} length:", len(doc.page_content))

docs_list = ["This is the first line.", "This is the second line", "This is the third line."]
print(type(docs_list))
print(type(documents[50].page_content))
print("here...")

def format_docs(documents):
    print("calling format docs")
    return "\n\n".join(doc.page_content for doc in documents)
    

print(format_docs(documents1))


import tiktoken
import os
from src import config
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

def count_tokens(text,model=config.LLM_MODEL):
    try:
        print("Model-->",model)
        enc = tiktoken.encoding_for_model(model)
    except:
        enc = tiktoken.get_encoding("cl100k_base")
    
    return len(enc.encode(text))

def get_gemini_gemma_llm():
    llm = ChatGoogleGenerativeAI(model = config.LLM_MODEL, api_key = config.GEMINI_API_KEY)
    return llm

def get_llama_groq_llm():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm_llama = ChatGroq(
        model= "llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0
    )
    return llm_llama

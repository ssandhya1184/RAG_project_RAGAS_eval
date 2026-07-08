import os

import pandas as pd
import numpy as np
from datasets import Dataset

from src import config
from src.rag_builder import build_pipeline
from src.utils import get_gemini_gemma_llm, count_tokens
from src.monitoring.metrics import (RAG_CONTEXT_PRECISION,
                                    RAG_CONTEXT_RECALL,
                                    RAG_FAITHFULNESS,
                                    RAG_ANSWER_CORRECTNESS)

from ragas import evaluate
from ragas.metrics import context_recall,context_precision, faithfulness,answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFaceEmbeddings


def run_evaluation():

    gemini_chat = get_gemini_gemma_llm()
    
    judge_llm_gemini = LangchainLLMWrapper(gemini_chat)
   
    hf_embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    data = {
        "question": config.QUESTION_LIST1,

        "ground_truth": config.GROUND_TRUTH1
            
    }



    pipeline = build_pipeline()

    ##############################################
    # Load Evaluation Questions
    ##############################################
    
    
    answers = []
    contexts = []


    #print(data)
    ##############################################
    # Generate Answers
    ##############################################
   
    for question in data["question"]:        
           
        #print(f"Running Question-> {question}")
        answer, docs_before, docs_after = pipeline(question)

        retrieved_docs = (docs_after if docs_after else docs_before)

        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        
        contexts.append(retrieved_contexts)
        answers.append(answer)

    data["answer"] = answers
    data["contexts"] = contexts 
    combined_text = f'{ data["question"]} {data["contexts"]} {data["ground_truth"]} {data["answer"]}'
    #print("Final Map: ",data)
    dataset = Dataset.from_dict(data)
  
    # 5. In v0.3.8, pass the wrapper instances directly to the evaluation pipeline
    print("Executing Ragas v0.3.8 Evaluation Pipeline with Gemini...")
    print("\n\n***************************************************************")
    token_count = count_tokens(combined_text,config.JUDGE_LLM)
    print(f"#####Token count: {token_count}")
    #print(f"--- LangChain LLM max_output_tokens: {gemini_chat.max_output_tokens} ---")   

    print("\n\n***************************************************************")

    # Perform Evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_recall,
            context_precision,
            faithfulness,
            answer_correctness
            ],
        llm=judge_llm_gemini,
        embeddings=embeddings,
        batch_size=1
    )
    
    question_wise_df = pd.DataFrame({
        "chunking": config.CHUNKING_STRATEGY,
        "retrieval": config.RETRIEVAL_STRATEGY,
        "reranker": config.ENABLE_RERANKING,
        
        "context_recall": result.to_pandas()["context_recall"],
        "context_precision": result.to_pandas()["context_precision"],
        "faithfulness": result.to_pandas()["faithfulness"],
       
        "question": data["question"],
        "ground_truth": data["ground_truth"],
        "answer": data["answer"],
        "context": data["contexts"]

       
    })

    #print(question_wise_df)
    question_wise_df.to_csv(
        "outputs/question_wise_results.csv",
        mode="a",
        header=not os.path.exists("outputs/question_wise_results.csv"),
        index=False
    )
    
    avg_context_recall = np.mean(result["context_recall"])
    avg_context_precision = np.mean(result["context_precision"])
    avg_faithfulness = np.mean(result["faithfulness"])
    avg_ans_correctness = np.mean(result["answer_correctness"])

    print("avg_context_recall->",avg_context_recall)
    print("avg_context_precision->",avg_context_precision)
    print("avg_faithfulness->",avg_faithfulness)

    RAG_CONTEXT_PRECISION.set(avg_context_precision)
    RAG_CONTEXT_RECALL.set(avg_context_recall)
    RAG_FAITHFULNESS.set(avg_faithfulness)
    RAG_ANSWER_CORRECTNESS.set(avg_ans_correctness)
                                 
    experiment_result = {
        "chunking": config.CHUNKING_STRATEGY,
        "retrieval": config.RETRIEVAL_STRATEGY,
        "reranker": config.RERANK_TOP_K,
      
        "context_recall": avg_context_recall,
        "context_precision": avg_context_precision,
        "faithfulness": avg_faithfulness,
        "answer_correctness":avg_ans_correctness
       
    }

   
    results_df = pd.DataFrame([experiment_result])
    #print(results_df)
    
    results_df.to_csv(
        "outputs/experiment_results.csv",
        mode="a",
        header=not os.path.exists("outputs/experiment_results.csv"),
        index=False
    )
    

if __name__ == "__main__":
    run_evaluation()


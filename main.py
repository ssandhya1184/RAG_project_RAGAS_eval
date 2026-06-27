
print("Shree Gurubyoh Namah!")
from src.rag_builder import build_pipeline
from rag_evaln import evaluate_results


from src import config

def main():
    
    pipeline = build_pipeline()

    # Query
    queries = [
        "Summarize this document in 10-15 lines with key points",
        "Are pre-hospitalization expenses covered in the policy?",
        "Can I do preventive health checkup using this policy?"
    ]
    for query in queries:
        print("--------------------------------------------------------------")
        print(f"\n Question: {query}")
        response, docs_before,docs_after = pipeline(query)
        print("\nAnswer:\n", response)
        print("--------------------------------------------------------------")

    #EValuation
    evaluate_results(docs_before,docs_after,pipeline)

if __name__ == "__main__":
    main()



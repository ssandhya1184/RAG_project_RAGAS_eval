from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self,model_name):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_k=3):
        if not docs:
            return []
        
        pairs = [(query,doc.page_content) for doc in docs]

        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse= True)

        return [doc for doc,_ in ranked[:top_k]]

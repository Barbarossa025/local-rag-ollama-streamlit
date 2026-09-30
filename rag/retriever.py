import numpy as np
from sentence_transformers import SentenceTransformer

# Modello multilingue: puoi fare domande in italiano su documenti in inglese
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = self.model.encode(
            [c["text"] for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q  # similarità coseno
        best = np.argsort(scores)[::-1][:top_k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]


if __name__ == "__main__":
    from rag.ingest import build_chunks

    retriever = Retriever(build_chunks())
    domanda = "Com'è cambiato il costo dei prestiti alle imprese?"
    for r in retriever.search(domanda):
        print(f"\n[{r['source']} #{r['chunk_id']}] score={r['score']:.2f}")
        print(r["text"][:300])
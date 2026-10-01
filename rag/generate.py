import ollama

MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "Sei un assistente che risponde SOLO usando il contesto fornito. Regole: "
    "1) Cerca nel contesto la frase che riguarda esattamente la domanda "
    "(stessa categoria: prestiti o depositi, imprese o famiglie, stessa scadenza). "
    "2) Riporta i numeri esattamente come appaiono nel contesto, senza arrotondare "
    "né ricalcolare. "
    "3) Non confondere le variazioni (punti base) con i livelli (percentuali). "
    "4) Se il contesto non contiene la risposta, rispondi solo: "
    "'Non ho trovato l'informazione nei documenti.' "
    "Rispondi in italiano, in una o due frasi."
)


def build_prompt(question: str, contexts: list[dict]) -> str:
    blocks = [
        f"[{c['source']} #{c['chunk_id']}]\n{c['text']}" for c in contexts
    ]
    context_text = "\n\n".join(blocks)
    return f"Contesto:\n{context_text}\n\nDomanda: {question}"


def answer(question: str, contexts: list[dict], model: str = MODEL) -> str:
    response = ollama.chat(
        model=model,
        options={"temperature": 0},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, contexts)},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    from rag.ingest import build_chunks
    from rag.retriever import Retriever

    retriever = Retriever(build_chunks())
    domanda = "Com'è cambiato il costo dei prestiti alle imprese?"
    contesti = retriever.search(domanda, top_k=3)
    print(answer(domanda, contesti))
    print("\nFonti:")
    for c in contesti:
        print(f"- {c['source']} #{c['chunk_id']} (score {c['score']:.2f})")

from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(data_dir: str = "data") -> list[dict]:
    """Legge tutti i file .md e .txt nella cartella data/."""
    documents = []
    for path in sorted(Path(data_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(
                {"source": path.name, "text": path.read_text(encoding="utf-8")}
            )
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Divide il testo in blocchi senza spezzare le parole."""
    text = " ".join(text.split())  # normalizza spazi e a capo
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + overlap:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
        nxt = text.find(" ", start)
        start = nxt + 1 if nxt != -1 else len(text)
    return chunks


def build_chunks(
    data_dir: str = "data", chunk_size: int = 500, overlap: int = 100
) -> list[dict]:
    """Restituisce tutti i chunk, ciascuno con la fonte di provenienza."""
    all_chunks = []
    for doc in load_documents(data_dir):
        for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            all_chunks.append(
                {"source": doc["source"], "chunk_id": i, "text": piece}
            )
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"{len(chunks)} chunk generati")
    for c in chunks[:3]:
        print(f"\n[{c['source']} #{c['chunk_id']}]\n{c['text']}")
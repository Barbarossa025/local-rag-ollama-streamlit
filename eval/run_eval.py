import argparse
import json
from pathlib import Path

from rag.generate import MODEL, answer
from rag.ingest import build_chunks
from rag.retriever import Retriever

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "last_results.json"


def norm(text: str) -> str:
    """Minuscolo e virgola decimale -> punto, così '3,62' e '3.62' coincidono."""
    return text.lower().replace(",", ".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    retriever = Retriever(
        build_chunks(chunk_size=args.chunk_size, overlap=args.overlap)
    )

    details = []
    retrieval_ok = retrieval_total = answer_ok = 0

    for q in questions:
        keywords = [norm(k) for k in q["keywords"]]
        contexts = retriever.search(q["question"], top_k=args.top_k)
        risposta = answer(q["question"], contexts, model=args.model)

        a_ok = all(k in norm(risposta) for k in keywords)
        answer_ok += a_ok

        r_ok = None
        if not q.get("expect_refusal", False):
            joined = norm(" ".join(c["text"] for c in contexts))
            r_ok = all(k in joined for k in keywords)
            retrieval_total += 1
            retrieval_ok += r_ok

        r_label = "-" if r_ok is None else ("OK" if r_ok else "KO")
        print(f"[retrieval {r_label} | risposta {'OK' if a_ok else 'KO'}] {q['question']}")
        if not a_ok:
            print(f"    risposta: {risposta[:200]}")

        details.append(
            {
                "question": q["question"],
                "retrieval_ok": r_ok,
                "answer_ok": a_ok,
                "answer": risposta,
            }
        )

    summary = {
        "model": args.model,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "top_k": args.top_k,
        "retrieval_hit": f"{retrieval_ok}/{retrieval_total}",
        "answer_accuracy": f"{answer_ok}/{len(questions)}",
        "details": details,
    }
    RESULTS_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nRESULT model={args.model} chunk_size={args.chunk_size} top_k={args.top_k} "
        f"retrieval={retrieval_ok}/{retrieval_total} "
        f"answer={answer_ok}/{len(questions)}"
    )


if __name__ == "__main__":
    main()

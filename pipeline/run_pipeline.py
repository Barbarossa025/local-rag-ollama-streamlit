import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rag.ingest import _read_docx, _read_pdf, chunk_text, list_document_files

WAREHOUSE = Path("warehouse")
BRONZE = WAREHOUSE / "bronze_files.parquet"   # un record per file ricevuto
SILVER = WAREHOUSE / "silver_pages.parquet"   # testo estratto per pagina/sezione
GOLD = WAREHOUSE / "gold_chunks.parquet"      # chunk pronti per la ricerca
REPORT = WAREHOUSE / "quality_report.json"

BRONZE_COLS = ["file", "file_hash", "size_bytes", "ingested_at"]
SILVER_COLS = ["file", "page", "text", "n_chars"]
GOLD_COLS = ["file", "page", "chunk_id", "text", "chunk_hash", "is_duplicate"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_sections(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"source": path.name, "page": None, "text": text}] if text.strip() else []


def load_table(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)


def append(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if rows else df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="documenti")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    WAREHOUSE.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    files = {str(p): p for p in list_document_files(args.data_dir)}
    hashes = {f: sha256_bytes(p.read_bytes()) for f, p in files.items()}

    bronze = load_table(BRONZE, BRONZE_COLS)
    silver = load_table(SILVER, SILVER_COLS)
    gold = load_table(GOLD, GOLD_COLS)

    # Incrementale: elimina i file rimossi o modificati, rielabora solo i nuovi
    known = dict(zip(bronze["file"], bronze["file_hash"]))
    changed = [f for f in files if f in known and known[f] != hashes[f]]
    removed = [f for f in known if f not in files]
    drop = set(changed) | set(removed)
    if drop:
        bronze = bronze[~bronze["file"].isin(drop)]
        silver = silver[~silver["file"].isin(drop)]
        gold = gold[~gold["file"].isin(drop)]
    todo = [f for f in files if f not in set(bronze["file"])]
    print(f"File totali: {len(files)} | da elaborare: {len(todo)} | rimossi/modificati: {len(drop)}")

    new_bronze, new_silver, new_gold = [], [], []
    for f in todo:
        path = files[f]
        new_bronze.append(
            {
                "file": f,
                "file_hash": hashes[f],
                "size_bytes": path.stat().st_size,
                "ingested_at": now,
            }
        )
        try:
            sections = read_sections(path)
        except Exception as exc:  # file corrotto o protetto da password
            print(f"  ! {f}: {exc}")
            continue

        counter = 0
        for sec in sections:
            new_silver.append(
                {
                    "file": f,
                    "page": sec["page"],
                    "text": sec["text"],
                    "n_chars": len(sec["text"]),
                }
            )
            for piece in chunk_text(sec["text"], args.chunk_size, args.overlap):
                new_gold.append(
                    {
                        "file": f,
                        "page": sec["page"],
                        "chunk_id": counter,
                        "text": piece,
                        "chunk_hash": sha256_bytes(piece.encode("utf-8")),
                        "is_duplicate": False,
                    }
                )
                counter += 1

    bronze = append(bronze, new_bronze)
    silver = append(silver, new_silver)
    gold = append(gold, new_gold)

    # Qualità: marca i chunk duplicati (stesso testo in file diversi)
    if len(gold):
        gold["is_duplicate"] = gold.duplicated("chunk_hash", keep="first")
    for df in (silver, gold):
        df["page"] = df["page"].astype("Int64")

    bronze.to_parquet(BRONZE, index=False)
    silver.to_parquet(SILVER, index=False)
    gold.to_parquet(GOLD, index=False)

    report = {
        "run_at": now,
        "files": int(len(bronze)),
        "files_processed_now": len(todo),
        "files_without_text": sorted(set(bronze["file"]) - set(silver["file"])),
        "pages_almost_empty": int((pd.to_numeric(silver["n_chars"]) < 50).sum()),
        "chunks": int(len(gold)),
        "duplicate_chunks": int(gold["is_duplicate"].sum()) if len(gold) else 0,
        "avg_chunk_chars": round(float(gold["text"].str.len().mean()), 1) if len(gold) else 0,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

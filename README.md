# Local RAG con Ollama e Streamlit

RAG (Retrieval-Augmented Generation) minimale che risponde a domande sui tuoi
documenti Markdown/testo, **100% in locale**: nessuna API a pagamento.
Ogni risposta cita le fonti (chunk) da cui è stata generata.

## Come funziona

Documenti → chunking → embedding → ricerca per similarità → prompt al modello locale → risposta con fonti

1. `rag/ingest.py` legge i file `.md` e `.txt` in `data/` e li divide in chunk senza spezzare le parole
2. `rag/retriever.py` calcola gli embedding e recupera i top-k chunk più simili alla domanda
3. `rag/generate.py` costruisce il prompt con il contesto e interroga il modello via Ollama
4. `app.py` espone tutto in un'interfaccia web Streamlit

## Avvio rapido

1. Installa [Ollama](https://ollama.com) e scarica un modello: `ollama pull llama3.2`
2. Clona il repo e installa le dipendenze:
```bash
   git clone https://github.com/TUO-USERNAME/local-rag-ollama-streamlit.git
   cd local-rag-ollama-streamlit
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
```
3. Metti i tuoi documenti in `data/` e avvia:
```bash
   streamlit run app.py
```

## Scelte tecniche

- **Chunking**: 500 caratteri con overlap di 100, tagliando su confini di parola (parametri modificabili dalla sidebar)
- **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers), multilingue: domande in italiano su documenti in inglese
- **Ricerca**: similarità coseno su embedding normalizzati (NumPy, senza database vettoriale per mantenere il progetto semplice)
- **LLM**: `llama3.2` via Ollama, con prompt di sistema che impone di rispondere solo dal contesto
- **UI**: Streamlit, con fonti espandibili e punteggio di similarità per ogni chunk

## Limiti e prossimi passi

- [ ] Valutazione automatica con un set di domande di test
- [ ] Reranking dei risultati
- [ ] Supporto PDF
- [ ] Database vettoriale persistente (es. Chroma) per evitare di rifare gli embedding ad ogni avvio

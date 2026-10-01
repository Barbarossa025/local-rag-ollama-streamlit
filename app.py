import streamlit as st

from rag.generate import answer
from rag.ingest import build_chunks
from rag.retriever import Retriever

st.set_page_config(page_title="Local RAG", page_icon="📚")
st.title("📚 Local RAG")
st.caption("Fai domande sui tuoi documenti in data/ — tutto in locale con Ollama.")


@st.cache_resource(show_spinner="Indicizzo i documenti...")
def load_retriever(chunk_size: int, overlap: int) -> Retriever:
    return Retriever(build_chunks(chunk_size=chunk_size, overlap=overlap))


with st.sidebar:
    st.header("Impostazioni")
    top_k = st.slider("Chunk da recuperare (top-k)", 1, 8, 5)
    chunk_size = st.select_slider("Dimensione chunk", [300, 500, 800, 1200], 800)
    overlap = st.select_slider("Overlap", [50, 100, 150], 100)
    model = st.text_input("Modello Ollama", "llama3.2")

retriever = load_retriever(chunk_size, overlap)
st.sidebar.write(f"Chunk indicizzati: {len(retriever.chunks)}")

question = st.text_input("La tua domanda")

if st.button("Chiedi") and question.strip():
    contexts = retriever.search(question, top_k=top_k)
    with st.spinner("Il modello sta rispondendo..."):
        risposta = answer(question, contexts, model=model)

    st.subheader("Risposta")
    st.write(risposta)

    st.subheader("Fonti")
    for c in contexts:
        with st.expander(f"{c['source']} #{c['chunk_id']} — score {c['score']:.2f}"):
            st.write(c["text"])

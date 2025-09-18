import os
import io
import json
import math
import time
from typing import List, Tuple, Dict

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import requests
from dotenv import load_dotenv

load_dotenv()

# Config & constants
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))  # characters per chunk (approx.)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_ENDPOINT = os.getenv("WATSONX_ENDPOINT")  # e.g. https://<instance>.us-south.watsonx.ai
WATSONX_MODEL = os.getenv("WATSONX_MODEL", "mixtral-8x7b-instruct")

# Initialize embedding model once
@st.cache_resource
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

embedder = load_embedding_model(EMBEDDING_MODEL_NAME)

# Utilities

def extract_text_from_pdf_bytes(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract text from PDF bytes. Returns list of (page_no, page_text)."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        text = page.get_text("text")
        pages.append((pno + 1, text))
    doc.close()
    return pages


def chunk_text_with_meta(pages: List[Tuple[int, str]], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Chunk text across pages, preserving page metadata.
    Returns list of dicts: {id, page, text}
    """
    chunks = []
    chunk_id = 0
    for page_no, text in pages:
        text = text.strip()
        if not text:
            continue
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunk_txt = text[start:end].strip()
            if chunk_txt:
                chunks.append({
                    "id": f"p{page_no}-c{chunk_id}",
                    "page": page_no,
                    "text": chunk_txt,
                })
                chunk_id += 1
            if end == length:
                break
            start = end - overlap if (end - overlap) > start else end
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return numpy array of embeddings for list of texts"""
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Normalize (FAISS inner-product simulated as cosine) optional
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings.astype('float32')


class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
        self.metadatas = []  # parallel list of metadata for vectors

    def add(self, vectors: np.ndarray, metadatas: List[dict]):
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        if self.index.ntotal == 0:
            return [], []
        q = query_vector.reshape(1, -1).astype('float32')
        D, I = self.index.search(q, top_k)
        scores = D[0].tolist()
        indices = I[0].tolist()
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            meta = self.metadatas[idx]
            results.append({"score": float(score), "meta": meta})
        return results


# Watsonx generation wrapper

def call_watsonx_generate(prompt: str, model: str = WATSONX_MODEL, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Example call to IBM watsonx generative endpoint. Replace with the exact API spec for your deployment.

    This function assumes a hypothetical REST API endpoint: {WATSONX_ENDPOINT}/v1/generate
    The actual IBM service endpoint and request body may differ — check IBM docs and adapt.
    """
    if not WATSONX_API_KEY or not WATSONX_ENDPOINT:
        st.warning("WATSONX_API_KEY or WATSONX_ENDPOINT not set. Returning a placeholder response.")
        return "[watsonx response placeholder]"

    url = WATSONX_ENDPOINT.rstrip("/") + "/v1/generate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WATSONX_API_KEY}",
    }
    payload = {
        "model": model,
        "input": prompt,
        "parameters": {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # The exact shape depends on the watsonx API. Attempt to extract generated text.
        if isinstance(data, dict):
            # Common places: data['output_text'], data['generations'][0]['text'], etc.
            if "output" in data and isinstance(data["output"], str):
                return data["output"]
            if "generations" in data and len(data["generations"]) > 0:
                return data["generations"][0].get("text", str(data))
        return str(data)
    except Exception as e:
        st.error(f"Error calling watsonx endpoint: {e}")
        return "[error calling watsonx]"


# Build prompt for LLM

def build_llm_prompt(question: str, retrieved: List[dict], max_context_chars: int = 3000) -> str:
    """Construct a prompt that provides retrieved context + question to the LLM.
    We will include short citations (page numbers) and the chunk text.
    """
    header = (
        "You are an academic assistant. Use ONLY the provided context to answer the question."
        " If the answer is not contained in the context, say you don't know. Provide concise answers and cite page numbers.\n\n"
    )
    context_blocks = []
    included = 0
    total_chars = 0
    for item in retrieved:
        meta = item['meta']
        text = meta['text']
        citation = f"(page {meta['page']})"
        block = f"{text}\n—{citation}\n\n"
        if total_chars + len(block) > max_context_chars:
            break
        context_blocks.append(block)
        total_chars += len(block)
        included += 1
    context_text = "".join(context_blocks)
    prompt = f"{header}Context:\n{context_text}\nQuestion: {question}\nAnswer:"
    return prompt


# Streamlit UI

st.set_page_config(page_title="StudyMate — PDF Conversational Q&A", layout="wide")
st.title("StudyMate — Conversational Q&A from Academic PDFs")

with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")
    st.header("Indexing options")
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=3000, value=CHUNK_SIZE, step=100)
    overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=CHUNK_OVERLAP, step=50)
    top_k = st.number_input("Top K retrieved chunks", min_value=1, max_value=20, value=5)
    st.markdown("---")
    st.markdown("Watsonx settings")
    st.text_input("Watsonx model", value=WATSONX_MODEL, key="watsonx_model")
    st.text_input("Watsonx endpoint", value=WATSONX_ENDPOINT or "", key="watsonx_endpoint")
    st.text_input("(Don't paste API key here in public) Watsonx API key", value="" if WATSONX_API_KEY is None else "(loaded from env)", key="watsonx_key")

# Main area
if uploaded_files:
    all_pages = []
    for f in uploaded_files:
        bytes_data = f.read()
        pages = extract_text_from_pdf_bytes(bytes_data)
        # prefix with filename for clarity
        pages = [(pno, f"[Source: {f.name}]\n" + txt) for pno, txt in pages]
        all_pages.extend(pages)

    st.success(f"Extracted text from {len(uploaded_files)} file(s), {len(all_pages)} page(s) total.")

    # Chunk
    chunks = chunk_text_with_meta(all_pages, chunk_size=chunk_size, overlap=overlap)
    st.info(f"Created {len(chunks)} text chunks for indexing.")

    # Embedding + FAISS
    texts = [c["text"] for c in chunks]
    metadatas = [{"id": c["id"], "page": c["page"], "text": c["text"]} for c in chunks]
    with st.spinner("Computing embeddings (this may take a while for large docs)..."):
        vectors = embed_texts(texts)
    dim = vectors.shape[1]
    faiss_index = FaissIndex(dim)
    faiss_index.add(vectors, metadatas)
    st.success("Index built. Ready for questions.")

    # Query interface
    st.markdown("---")
    st.header("Ask a question")
    question = st.text_area("Enter your question", height=120)
    if st.button("Get Answer") and question.strip():
        q_emb = embed_texts([question])[0]
        retrieved = faiss_index.search(q_emb, top_k=top_k)
        if not retrieved:
            st.warning("No relevant content found in your documents.")
        else:
            st.subheader("Retrieved evidence")
            for i, item in enumerate(retrieved, start=1):
                meta = item['meta']
                st.markdown(f"**{i}. Page {meta['page']}** — score: {item['score']:.4f}")
                st.write(meta['text'][:1000] + ("..." if len(meta['text']) > 1000 else ""))

            prompt = build_llm_prompt(question, retrieved)
            st.subheader("Answer")
            with st.spinner("Generating answer from LLM..."):
                watsonx_model = st.session_state.get('watsonx_model', WATSONX_MODEL)
                # update endpoint & key if changed in sidebar
                endpoint = st.session_state.get('watsonx_endpoint', WATSONX_ENDPOINT)
                api_key = os.getenv('WATSONX_API_KEY') or WATSONX_API_KEY
                answer = call_watsonx_generate(prompt, model=watsonx_model)
            st.markdown(answer)

            st.markdown("---")
            st.markdown("**Sources (detailed)**")
            for i, item in enumerate(retrieved, start=1):
                meta = item['meta']
                st.markdown(f"- Source: Page {meta['page']} — chunk id {meta['id']} — score {item['score']:.4f}")

else:
    st.info("Upload PDFs from the sidebar to get started.")

st.markdown("---")
st.caption("StudyMate demo — extracts text with PyMuPDF, uses SentenceTransformers + FAISS for retrieval, and calls an LLM (watsonx) for answer generation.")  
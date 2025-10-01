# app.py — Rubber Duck (multi-textbook) with remote GitHub chapters
import os, glob, re, hashlib
from typing import List, Tuple
from urllib.parse import quote

import requests
import numpy as np
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
import faiss

# -------------------- BASIC CONFIG --------------------
st.set_page_config(page_title="Rubber Duck — Ask the Textbook", layout="wide")

# REQUIRED: set in Streamlit Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
client = OpenAI(api_key=OPENAI_API_KEY)

# OPTIONAL: only needed if any source repo is private
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")

# Your AP Euro chapters repo (public works out of the box)
OWNER  = "chammerberg-creator"
REPO   = "rubber-duck-APE-chapters"
BRANCH = "main"

# Generate filenames: "McKay Chapter 11 OCR.pdf" ... "McKay Chapter 30 OCR.pdf"
EURO_FILES = [f"McKay Chapter {i} OCR.pdf" for i in range(11, 31)]

# Catalog (easy to add more textbooks later)
BOOKS = {
    "ap_euro_mckay_13e": {
        "label": "AP Euro — McKay (13e)",
        "version": "v1",   # bump when you add/replace PDFs to rebuild caches
        # If repo is PUBLIC, use mode 'raw' (simplest). If PRIVATE, switch to 'github_api' and add GITHUB_TOKEN in Secrets.
        "mode": "raw",     # "raw" | "github_api"
        # For 'raw' mode we use raw.githubusercontent URLs:
        "base": f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/",
        "files": EURO_FILES,
        # For 'github_api' mode (private repos), fill these fields instead:
        "owner": OWNER,
        "repo": REPO,
        "branch": BRANCH,
        "path": "",     # e.g., "chapters" if your PDFs are inside a folder; "" for repo root
    },
    # Add more textbooks here later...
}

BOOKS_ROOT = "books"  # local cache root (per book)
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 6
CHUNK_TOKENS = 700
CHUNK_OVERLAP = 150

# -------------------- HELPERS --------------------
def _headers():
    h = {"User-Agent": "RubberDuck/1.0"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def clean(s: str) -> str:
    return re.sub(r"[ ]{2,}", " ", s.replace("\u00a0", " ").replace("\t", " ")).strip()

def extract_pages(path: str) -> List[Tuple[int, str]]:
    out = []
    reader = PdfReader(path)
    for i, p in enumerate(reader.pages):
        t = clean(p.extract_text() or "")
        if t:
            out.append((i + 1, t))
    return out

def chunk_text(text: str, fname: str, page: int):
    toks = text.split()
    i, out = 0, []
    while i < len(toks):
        j = min(len(toks), i + CHUNK_TOKENS)
        ch = " ".join(toks[i:j])
        out.append({"text": ch, "meta": {"chapter": fname, "page": page}})
        i = j - CHUNK_OVERLAP if j - CHUNK_OVERLAP > i else j
    return out

def local_book_dir(book_key: str) -> str:
    d = os.path.join(BOOKS_ROOT, book_key)
    os.makedirs(d, exist_ok=True)
    return d

def list_pdfs_via_github(owner: str, repo: str, path: str, branch: str):
    """List PDFs at a path using GitHub Contents API (useful for private repos)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path or ''}?ref={branch}"
    r = requests.get(url, headers=_headers(), timeout=60)
    r.raise_for_status()
    data = r.json()
    files = [item["name"] for item in data if item["type"] == "file" and item["name"].lower().endswith(".pdf")]
    return sorted(files)

@st.cache_data(show_spinner=True)
def ensure_book_downloaded(book_key: str):
    """Download PDFs for the selected book into books/<book_key>/ if not already present."""
    cfg = BOOKS[book_key]
    d = local_book_dir(book_key)

    mode = cfg.get("mode", "raw")
    if mode == "raw":
        base = cfg["base"].rstrip("/") + "/"
        files = cfg["files"]
        for f in files:
            dest = os.path.join(d, f)
            if not os.path.exists(dest):
                url = base + quote(f, safe="")  # handle spaces in filenames
                r = requests.get(url, headers=_headers(), timeout=120)
                r.raise_for_status()
                with open(dest, "wb") as out:
                    out.write(r.content)
    elif mode == "github_api":
        owner, repo, branch, path = cfg["owner"], cfg["repo"], cfg["branch"], cfg.get("path") or ""
        files = cfg.get("files")
        if not files:
            files = list_pdfs_via_github(owner, repo, path, branch)
        for f in files:
            dest = os.path.join(d, f)
            if not os.path.exists(dest):
                url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path + '/' if path else ''}{f}"
                r = requests.get(url, headers=_headers(), timeout=120)
                r.raise_for_status()
                with open(dest, "wb") as out:
                    out.write(r.content)
    else:
        raise ValueError("Unknown BOOKS mode (use 'raw' or 'github_api').")

    present = sorted([os.path.basename(p) for p in glob.glob(os.path.join(d, "*.pdf"))])
    return d, present

def cache_sig_for(book_key: str) -> str:
    """Stable signature to key the FAISS index cache for this book."""
    cfg = BOOKS[book_key]
    _, files = ensure_book_downloaded(book_key)
    sig = cfg.get("version", "v0") + "|" + "|".join(files)
    return hashlib.sha1(sig.encode()).hexdigest()

@st.cache_resource(show_spinner=True)
def build_index(book_key: str, cache_sig: str):
    """Build (or load) FAISS index for the selected book."""
    book_dir, files = ensure_book_downloaded(book_key)
    texts, metas = [], []
    for f in files:
        path = os.path.join(book_dir, f)
        for page_num, text in extract_pages(path):
            for ch in chunk_text(text, f, page_num):
                texts.append(ch["text"]); metas.append(ch["meta"])

    if not texts:
        dim = 1536
        arr = np.zeros((1, dim), dtype="float32")
        faiss.normalize_L2(arr)
        idx = faiss.IndexFlatIP(dim); idx.add(arr)
        return idx, texts, metas

    # Embed in batches
    vectors = []
    BATCH = 128
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        emb = client.embeddings.create(model=EMBED_MODEL, input=batch).data
        vectors.extend([e.embedding for e in emb])

    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    idx = faiss.IndexFlatIP(arr.shape[1]); idx.add(arr)
    return idx, texts, metas

def search(idx, texts, metas, query: str, k: int = TOP_K):
    qemb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q = np.array([qemb], dtype="float32")
    faiss.normalize_L2(q)
    sims, ids = idx.search(q, k)
    out = []
    for score, ix in zip(sims[0], ids[0]):
        if 0 <= ix < len(texts):
            out.append({"score": float(score), "text": texts[ix], **metas[ix]})
    return out

# -------------------- UI --------------------
st.sidebar.title("Rubber Duck")
book_key = st.sidebar.selectbox(
    "Choose a textbook",
    options=list(BOOKS.keys()),
    format_func=lambda k: BOOKS[k]["label"]
)

# Ensure chapters are present locally
with st.sidebar:
    _, present = ensure_book_downloaded(book_key)
    st.success(f"Chapters ready: {len(present)} files")
    if st.button("Refresh chapters / Rebuild index"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Build / reuse index (cached per book + version/files)
sig = cache_sig_for(book_key)
idx, texts, metas = build_index(book_key, sig)

st.title("Ask the textbook")
st.caption(f"Book: **{BOOKS[book_key]['label']}**  •  Indexed chunks: {len(texts)}  •  PDFs: {len(set(m['chapter'] for m in metas))}")

q = st.text_input("Your question:", placeholder="e.g., Identify two political effects of the Protestant Reformation.")

if q:
    with st.spinner("Retrieving…"):
        hits = search(idx, texts, metas, q, k=TOP_K)

    st.markdown("### Sources")
    if not hits:
        st.info("No matches found in this textbook. Try different phrasing or choose another book.")
    else:
        for i, h in enumerate(hits, 1):
            st.markdown(f"**{i}. {h['chapter']} • p.{h['page']}** (score {h['score']:.3f})")
            st.write(h["text"][:700] + ("..." if len(h["text"])>700 else ""))

    # Compose context and call chat model (RAG)
    if hits:
        context = "\n\n---\n\n".join([f"[{h['chapter']} p.{h['page']}]\n{h['text']}" for h in hits])
        system_msg = (
            "You are a helpful AP tutor. Answer ONLY using the provided context. "
            "Cite inline like [filename.pdf p.X]. If the answer is not in the context, say so."
        )
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Question: {q}\n\nContext:\n{context}"}
        ]
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.2)
        st.markdown("### Answer")
        st.write(resp.choices[0].message.content)

# app.py — Rubber Duck (with Quick mode + progress) pulling public PDFs from GitHub
import os, glob, re, hashlib, time
from typing import List, Tuple
from urllib.parse import quote

import requests
import numpy as np
import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError
from openai import OpenAI
import faiss

# -------------------- BASIC CONFIG --------------------
st.set_page_config(page_title="Rubber Duck — Ask the Textbook", layout="wide")

# REQUIRED: set in Streamlit Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
# Add a client timeout so embedding calls cannot stall forever
client = OpenAI(api_key=OPENAI_API_KEY, timeout=60)

# OPTIONAL: only needed if you later add private repos (kept here for future use)
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")

# Source repo for AP Euro chapters (PUBLIC)
OWNER  = "chammerberg-creator"
REPO   = "McKay-13th-Edition-History-of-Western-Society"
BRANCH = "main"

# Files are: McKay Chapter 11 - OCR.pdf ... McKay Chapter 30 - OCR.pdf
EURO_FILES = [f"McKay Chapter {i} - OCR.pdf" for i in range(11, 31)]

# Catalog (easy to add more textbooks later)
BOOKS = {
    "ap_euro_mckay_13e": {
        "label": "AP Euro — McKay (13e)",
        "version": "v1",   # bump when you add/replace PDFs to rebuild caches
        "mode": "raw",     # uses raw.githubusercontent.com (repo is public)
        "base": f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/",
        "files": EURO_FILES,

        # Fields below are unused in raw mode (kept for future private repo support)
        "owner": OWNER,
        "repo": REPO,
        "branch": BRANCH,
        "path": "",
    },
}

BOOKS_ROOT = "books"  # local cache root (per book)
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 6
CHUNK_TOKENS = 500     # lower for faster first index
CHUNK_OVERLAP = 100    # lower overlap for speed

# -------------------- SIDEBAR --------------------
st.sidebar.title("Rubber Duck")
book_key = st.sidebar.selectbox(
    "Choose a textbook",
    options=list(BOOKS.keys()),
    format_func=lambda k: BOOKS[k]["label"]
)

# Quick mode toggle to index a subset (helps avoid long first-run times)
DEV_quick = st.sidebar.toggle(
    "Quick mode (first few PDFs/pages)",
    value=True,
    help="Index a small sample to verify end-to-end. Turn off for full textbook."
)
MAX_PDFS  = 3 if DEV_quick else None     # first N PDFs
MAX_PAGES = 8 if DEV_quick else None     # first N pages per PDF

# -------------------- HELPERS --------------------
def _headers():
    h = {"User-Agent": "RubberDuck/1.0"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def clean(s: str) -> str:
    return re.sub(r"[ ]{2,}", " ", s.replace("\u00a0", " ").replace("\t", " ")).strip()

def _download_file(url: str, dest: str, max_retries: int = 3) -> bool:
    """Stream a file to disk with retries. Returns True if it looks like a valid PDF."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, headers=_headers(), timeout=120, stream=True) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp, dest)
            # quick magic-byte check
            with open(dest, "rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("Not a PDF magic header")
            return True
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)  # simple backoff
    # cleanup on failure
    try:
        if os.path.exists(dest):
            os.remove(dest)
    except Exception:
        pass
    st.warning(f"Download failed for {os.path.basename(dest)}: {last_err}")
    return False

def _looks_like_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False

def extract_pages(path: str) -> List[Tuple[int, str]]:
    out = []
    try:
        reader = PdfReader(path)
    except (PdfReadError, PdfStreamError, Exception) as e:
        st.warning(f"Cannot open {os.path.basename(path)} — skipping. ({e})")
        return out

    for i, p in enumerate(reader.pages):
        try:
            t = clean(p.extract_text() or "")
        except Exception:
            t = ""
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

@st.cache_data(show_spinner=True)
def ensure_book_downloaded(book_key: str):
    """Download PDFs for the selected book into books/<book_key>/; return valid file list."""
    cfg = BOOKS[book_key]
    d = local_book_dir(book_key)
    valid_files = []

    mode = cfg.get("mode", "raw")
    if mode == "raw":
        base = cfg["base"].rstrip("/") + "/"
        files = cfg["files"]
        for f in files:
            dest = os.path.join(d, f)
            url = base + quote(f, safe="")  # handle spaces
            need_download = (not os.path.exists(dest)) or (not _looks_like_pdf(dest))
            if need_download:
                ok = _download_file(url, dest, max_retries=3)
                if not ok:
                    continue
            # validate by opening with PdfReader
            try:
                PdfReader(dest)
                valid_files.append(f)
            except Exception as e:
                st.warning(f"Skipped unreadable PDF: {f} ({e})")
                try: os.remove(dest)
                except: pass
    else:
        # Future: add github_api branch if you later use private repos
        raise ValueError("This app is configured for 'raw' mode only right now.")

    present = sorted(valid_files)
    return d, present

def cache_sig_for(book_key: str) -> str:
    """Stable signature to key the FAISS index cache for this book + settings."""
    cfg = BOOKS[book_key]
    _, files = ensure_book_downloaded(book_key)
    extra = f"quick={DEV_quick}|mpdfs={MAX_PDFS}|mpages={MAX_PAGES}|ct={CHUNK_TOKENS}|co={CHUNK_OVERLAP}"
    sig = cfg.get("version", "v0") + "|" + "|".join(files) + "|" + extra
    return hashlib.sha1(sig.encode()).hexdigest()

@st.cache_resource(show_spinner=False)
def build_index(book_key: str, cache_sig: str):
    """Build (or load) FAISS index for the selected book with progress + retries."""
    book_dir, files = ensure_book_downloaded(book_key)
    if MAX_PDFS:
        files = files[:MAX_PDFS]

    status = st.status("Indexing textbook…", expanded=True)
    status.write("Step 1/4: Extracting & chunking pages…")
    texts, metas = [], []

    for fi, f in enumerate(files, 1):
        path = os.path.join(book_dir, f)
        pages = extract_pages(path)
        if MAX_PAGES:
            pages = pages[:MAX_PAGES]
        for page_num, text in pages:
            # chunk immediately (keeps memory modest)
            toks = text.split()
            i = 0
            while i < len(toks):
                j = min(len(toks), i + CHUNK_TOKENS)
                ch = " ".join(toks[i:j])
                texts.append(ch)
                metas.append({"chapter": f, "page": page_num})
                i = j - CHUNK_OVERLAP if j - CHUNK_OVERLAP > i else j
        status.write(f" • {fi}/{len(files)}: {f} — {len(pages)} pages")

    if not texts:
        dim = 1536
        arr = np.zeros((1, dim), dtype="float32")
        faiss.normalize_L2(arr)
        idx = faiss.IndexFlatIP(dim); idx.add(arr)
        status.update(label="Indexing complete (no text found).", state="complete")
        return idx, texts, metas

    status.write("Step 2/4: Creating embeddings…")
    BATCH = 64  # smaller batch = fewer rate-limit errors + more responsive
    vectors = []
    prog = st.progress(0.0)
    total = len(texts)
    for i in range(0, total, BATCH):
        batch = texts[i:i+BATCH]
        # retries for transient errors
        for attempt in range(4):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                vectors.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt == 3:
                    status.update(label=f"Embedding failed: {e}", state="error")
                    raise
                time.sleep(1.5 * (attempt + 1))
        prog.progress(min(1.0, (i + len(batch)) / total))

    status.write("Step 3/4: Building FAISS index…")
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    idx = faiss.IndexFlatIP(arr.shape[1])
    idx.add(arr)

    status.update(label="Step 4/4: Done! ✔️", state="complete")
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

# Ensure chapters are present locally
with st.sidebar:
    _, present = ensure_book_downloaded(book_key)
    st.success(f"Chapters ready: {len(present)} files")
    if st.button("Refresh chapters / Rebuild index"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Build / reuse index (cached per book + version/files + quick settings)
sig = cache_sig_for(book_key)
idx, texts, metas = build_index(book_key, sig)

# -------------------- UI: Q&A --------------------
st.title("Ask the textbook")
st.caption(f"Book: **{BOOKS[book_key]['label']}**  •  Indexed chunks: {len(texts)}  •  PDFs: {len(set(m['chapter'] for m in metas))}")

q = st.text_input("Your question:", placeholder="e.g., Identify two political effects of the Protestant Reformation.")

if q:
    with st.spinner("Retrieving…"):
        hits = search(idx, texts, metas, q, k=TOP_K)

    st.markdown("### Sources")
    if not hits:
        st.info("No matches found in this textbook. Try different phrasing.")
    else:
        for i, h in enumerate(hits, 1):
            st.markdown(f"**{i}. {h['chapter']} • p.{h['page']}** (score {h['score']:.3f})")
            st.write(h["text"][:700] + ("..." if len(h['text'])>700 else ""))

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

# -------------------- Diagnostics (optional) --------------------
with st.expander("Diagnostics (PDF health check)"):
    d, present = ensure_book_downloaded(book_key)
    bad = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(".pdf"):
            try:
                PdfReader(os.path.join(d, f))
            except Exception as e:
                bad.append((f, str(e)))
    st.write(f"Valid PDFs: {len(present)}")
    if bad:
        st.error("Unreadable PDFs detected:")
        for name, err in bad:
            st.write(f"• {name} — {err}")
    else:
        st.success("All downloaded PDFs open correctly.")

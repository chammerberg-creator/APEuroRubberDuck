
# Rubber Duck — Feynman Practice (MVP)

This is a minimal Streamlit app for Feynman-style practice. Students explain; **Rubber Duck** asks one short, scoped question at a time. End with a downloadable study guide (.docx).

## Quick web setup (Streamlit Community Cloud)

1. Create a GitHub repo (e.g., `rubber-duck-feynman`).
2. Upload three files into the repo:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Go to Streamlit Community Cloud → **New app** → pick your GitHub repo → file = `app.py` → **Deploy**.
4. App **⋯ → Settings → Secrets** → add:
   ```
   OPENAI_API_KEY="sk-...your key ..."
   OPENAI_CHAT_MODEL="gpt-4o-mini"
   ```
5. Share the app URL with students.

## Local run (optional)

```bash
pip install -U streamlit openai python-docx
export OPENAI_API_KEY=YOUR_KEY   # Windows: $env:OPENAI_API_KEY="YOUR_KEY"
streamlit run app.py
```

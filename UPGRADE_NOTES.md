# MedIntel v2 – what changed

## Quick start (Windows, Python 3.13 OK)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# open .env and paste:  GEMINI_API_KEY=your_key
streamlit run streamlit_app.py
```
Optional, to re-chunk your existing documents at the new size: `python scripts/rebuild_index.py`

## The three limits (all in `.env`, no code edits needed)
| What | Before | Now | Setting |
|---|---|---|---|
| Response size | 1000 tokens | 8192 (up to 65,536 on gemini-2.5-*) + auto-continue if cut off | `MAX_TOKENS`, `MAX_CONTINUATIONS` |
| FAISS retrieval | top_k 5, min score 0.75 | top_k 12 (UI up to 30), min score 0.30, over-fetch x3 + de-dup + MMR | `TOP_K_DOCUMENTS`, `MAX_TOP_K`, `RETRIEVAL_CONFIDENCE_THRESHOLD`, `FETCH_K_MULTIPLIER` |
| Context window | ~5 x 500-char chunks | each hit widened by neighbouring chunks, up to 60,000 chars, + last 6 chat turns | `NEIGHBOR_WINDOW`, `MAX_CONTEXT_CHARS`, `HISTORY_TURNS` |

Why the threshold dropped: all-MiniLM-L6-v2 scores for relevant text are ~0.30-0.65, so 0.75 discarded nearly everything and a bigger top_k had no effect.

## Other fixes
* Gemini now uses `google-genai` (old `google-generativeai` is end-of-life; kept as fallback).
* FAISS bug fixed: asking for more results than exist returned the last document by mistake.
* Duplicate chunks skipped on re-ingest and ignored at search time.
* `requirements.txt` relaxed to install on Python 3.13; openai/torch/transformers only imported if selected.
* API `/query` accepts `top_k` up to 50 and returns full retrieved passages in `contexts`.
* Hard-coded Gemini key removed from `render.yaml`, `render.yaml.backup`, `DEPLOYMENT.md` - rotate that key and set `GEMINI_API_KEY` in the Render dashboard.

## UI
Streaming answers, multi-chat sidebar, suggestion cards, sources with full passages, evidence-match badge, Copy / thumbs up / thumbs down / Regenerate, chat export, and footer notice "MedIntel is AI and can make mistakes. Please double-check responses." (link target: `AI_DISCLAIMER_URL`). Ratings go to `data/feedback/feedback.jsonl` and are summarised on the Insights page.

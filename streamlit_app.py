"""
MedIntel — Streamlit chat UI (v2)

A ChatGPT / Claude-style front end for the MedIntel RAG pipeline:
streaming answers, multi-chat sidebar, sources, 👍/👎 ratings, copy button,
regenerate, and a persistent "AI can make mistakes" notice.

Run with:
    streamlit run streamlit_app.py
"""
import html
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from src.config import get_settings
from src.data_ingestion import (
    DocumentLoader,
    MedicalCorpusBuilder,
    MedicalDocument,
    PubMedLoader,
    WHOGHOLoader,
)
from src.feedback import load_feedback, log_feedback
from src.rag_pipeline import INSUFFICIENT_MSG, RAGPipeline, max_output_limit
from src.vector_store import VectorStore

settings = get_settings()

ASSISTANT_AVATAR = "🩺"
NAV = ["💬 Chat", "📚 Knowledge base", "📊 Insights"]
PROVIDERS = ["gemini", "openai", "mistral", "qwen", "huggingface"]
TOKEN_STEPS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
DISCLAIMER_TEXT = "MedIntel is AI and can make mistakes. Please double-check responses."

SUGGESTIONS = [
    ("🧠", "Headache", "What are the common causes of a headache at the back of the head?"),
    ("🦠", "HIV", "Explain HIV: how it spreads, symptoms, and how it is managed."),
    ("🩹", "Common illnesses", "What are some common diseases and how can they be prevented?"),
    ("🚑", "Warning signs", "Which headache symptoms mean I should see a doctor urgently?"),
]

WHO_FACT_SHEETS = {
    "Diabetes": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "Hypertension": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
    "Obesity and overweight": "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
    "Tuberculosis": "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "Malaria": "https://www.who.int/news-room/fact-sheets/detail/malaria",
    "HIV": "https://www.who.int/news-room/fact-sheets/detail/hiv-aids",
    "Cardiovascular diseases": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "Mental disorders": "https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
    "Asthma": "https://www.who.int/news-room/fact-sheets/detail/asthma",
    "Anaemia": "https://www.who.int/news-room/fact-sheets/detail/anaemia",
}

st.set_page_config(
    page_title="MedIntel — Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
BASE_CSS = """
<style>
:root{
  --mi-primary:#0f766e; --mi-primary-2:#14b8a6; --mi-ink:#0f172a; --mi-muted:#64748b;
  --mi-line:#e2e8f0; --mi-card:#ffffff; --mi-soft:#f0fdfa;
}
html, body, [data-testid="stAppViewContainer"]{ font-family: "Inter","Segoe UI",system-ui,-apple-system,sans-serif; }
[data-testid="stAppViewContainer"]{ background: radial-gradient(1200px 500px at 50% -10%, #ccfbf1 0%, rgba(240,253,250,0) 60%), #f8fafc; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none !important; }
[data-testid="stHeader"]{ background: transparent; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ background:#ffffff; border-right:1px solid var(--mi-line); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"]{ padding-top:1.2rem; }
.mi-brand{ display:flex; align-items:center; gap:.65rem; margin-bottom:.9rem; }
.mi-logo{ width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-size:22px;
  background:linear-gradient(135deg,#0f766e,#14b8a6); box-shadow:0 6px 16px rgba(15,118,110,.35); }
.mi-brand b{ font-size:1.15rem; color:var(--mi-ink); letter-spacing:-.01em; }
.mi-brand span{ display:block; font-size:.72rem; color:var(--mi-muted); margin-top:-2px; }
.mi-side-label{ font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#94a3b8; margin:1rem 0 .3rem; }
[data-testid="stSidebar"] .stButton > button{ justify-content:flex-start; text-align:left; border-radius:10px; font-size:.88rem; padding:.45rem .7rem; }
[data-testid="stSidebar"] .stButton > button[kind="primary"]{ background:linear-gradient(135deg,#0f766e,#0e9488); border:none; color:#fff; }
[data-testid="stSidebar"] .stButton > button[kind="secondary"]{ background:#fff; border:1px solid var(--mi-line); color:var(--mi-ink); }
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover{ border-color:var(--mi-primary-2); background:var(--mi-soft); }

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"]{ background:transparent; padding:.35rem 0; gap:.7rem; width:100%; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){ flex-direction:row-reverse; }
[data-testid="stChatMessageAvatarUser"]{ display:none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  flex:0 1 auto; max-width:82%; margin-left:auto; color:#fff; padding:.7rem 1.05rem;
  background:linear-gradient(135deg,#0f766e,#0e9488); border-radius:20px 20px 6px 20px;
  box-shadow:0 6px 18px rgba(15,118,110,.25); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] *{ color:#fff !important; }
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) [data-testid="stChatMessageContent"]{
  background:#fff; border:1px solid var(--mi-line); border-radius:6px 20px 20px 20px; padding:1rem 1.2rem;
  box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px rgba(15,118,110,.07); }
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) > div:first-child{ background:linear-gradient(135deg,#0f766e,#14b8a6); border-radius:12px; border:none; }
[data-testid="stChatMessageContent"] p{ line-height:1.65; }

/* typing indicator */
.mi-typing{ display:flex; align-items:center; gap:.35rem; color:var(--mi-muted); font-size:.85rem; padding:.2rem 0; }
.mi-typing i{ width:7px; height:7px; border-radius:50%; background:var(--mi-primary-2); display:inline-block; animation:mi-bounce 1.2s infinite ease-in-out; }
.mi-typing i:nth-child(2){ animation-delay:.15s } .mi-typing i:nth-child(3){ animation-delay:.3s }
@keyframes mi-bounce{ 0%,80%,100%{ transform:scale(.55); opacity:.4 } 40%{ transform:scale(1); opacity:1 } }

/* badges + meta */
.mi-badge{ display:inline-flex; align-items:center; gap:.3rem; padding:.18rem .6rem; border-radius:999px; font-size:.72rem; font-weight:600; }
.mi-badge-high{ background:#dcfce7; color:#166534; } .mi-badge-med{ background:#fef9c3; color:#854d0e; } .mi-badge-low{ background:#fee2e2; color:#991b1b; }
.mi-meta{ color:var(--mi-muted); font-size:.74rem; margin-left:.5rem; }

/* sources */
.mi-source{ background:#fff; border:1px solid var(--mi-line); border-left:4px solid var(--mi-primary); border-radius:12px; padding:.65rem .85rem; margin:.45rem 0; }
.mi-source-title{ font-weight:600; font-size:.88rem; color:var(--mi-ink); }
.mi-source-meta{ color:var(--mi-muted); font-size:.74rem; margin:.1rem 0 .35rem; }
.mi-source-text{ color:#334155; font-size:.83rem; line-height:1.5; }
.mi-source details summary{ cursor:pointer; color:var(--mi-primary); font-size:.78rem; margin-top:.35rem; }

/* action row buttons */
[data-testid="stChatMessage"] .stButton > button{ border-radius:10px; border:1px solid var(--mi-line); background:#fff; padding:.15rem .5rem; min-height:34px; }
[data-testid="stChatMessage"] .stButton > button[kind="primary"]{ background:var(--mi-soft); border-color:var(--mi-primary-2); color:var(--mi-primary); }
[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"]{ flex-wrap:nowrap !important; gap:.35rem; align-items:center; }
[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] > div{ min-width:0 !important; }

/* chat input */
[data-testid="stBottom"] > div{ background:linear-gradient(180deg, rgba(248,250,252,0) 0%, #f8fafc 35%); padding-bottom:2.1rem; }
[data-testid="stChatInput"]{ border-radius:26px; border:1px solid #cbd5e1; background:#fff; box-shadow:0 8px 30px rgba(15,23,42,.08); }
[data-testid="stChatInput"]:focus-within{ border-color:var(--mi-primary-2); box-shadow:0 8px 30px rgba(20,184,166,.22); }
[data-testid="stChatInput"] textarea{ font-size:.97rem; }

/* disclaimer footer */
.mi-footer{ position:fixed; left:0; right:0; bottom:9px; text-align:center; font-size:.74rem; z-index:1000; pointer-events:none; }
.mi-footer a{ pointer-events:auto; color:var(--mi-muted); text-decoration:none; }
.mi-footer a:hover{ color:var(--mi-primary); text-decoration:underline; }
[data-testid="stAppViewContainer"]:has([data-testid="stSidebar"][aria-expanded="true"]) .mi-footer{ left:21rem; }

/* welcome screen */
.mi-hero{ text-align:center; padding:3.2rem 0 1.4rem; }
.mi-hero .mi-logo{ width:68px; height:68px; font-size:36px; margin:0 auto 1rem; border-radius:20px; }
.mi-hero h1{ font-size:2rem; letter-spacing:-.02em; margin:0; color:var(--mi-ink); }
.mi-hero p{ color:var(--mi-muted); margin:.45rem auto 0; max-width:520px; }
.mi-chips-anchor{ display:none; }
[data-testid="stMain"]:has(.mi-chips-anchor) .stButton > button{ height:auto; min-height:76px; text-align:left; justify-content:flex-start; white-space:normal;
  border-radius:16px; border:1px solid var(--mi-line); background:#fff; padding:.8rem 1rem; box-shadow:0 1px 2px rgba(15,23,42,.04); transition:all .15s; }
[data-testid="stMain"]:has(.mi-chips-anchor) .stButton > button:hover{ border-color:var(--mi-primary-2); transform:translateY(-2px); box-shadow:0 10px 24px rgba(15,118,110,.14); }

[data-testid="stMain"]:has(.mi-chips-anchor) .stButton > button p{ text-align:left; }
[data-testid="stMain"]:has(.mi-chips-anchor) .stButton > button > div{ justify-content:flex-start; width:100%; }

/* misc */
.mi-metric{ background:#fff; border:1px solid var(--mi-line); border-radius:14px; padding:.9rem; text-align:center; }
.mi-metric h3{ margin:0; color:var(--mi-primary); } .mi-metric span{ color:var(--mi-muted); font-size:.8rem; }
@media (max-width: 640px){ [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{ max-width:92%; } .mi-footer{ left:0 !important; } }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)


def apply_width(view: str):
    width = "820px" if view == NAV[0] else "1100px"
    st.markdown(
        f"<style>[data-testid='stMainBlockContainer'], .block-container{{max-width:{width}; padding-top:1.4rem; padding-bottom:9rem;}}</style>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Resources & state
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model + vector store...")
def load_vector_store():
    vs = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if os.path.exists(settings.vector_store_path):
        try:
            vs.load(settings.vector_store_path)
        except Exception as e:
            st.warning(f"Could not load existing vector store: {e}")
    return vs


def new_chat_obj():
    return {"id": uuid.uuid4().hex[:8], "title": "New chat", "messages": [], "created": time.time()}


def init_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store()
    if "chats" not in st.session_state:
        chat = new_chat_obj()
        st.session_state.chats = {chat["id"]: chat}
        st.session_state.chat_id = chat["id"]
    st.session_state.setdefault("view", NAV[0])
    st.session_state.setdefault("pending_prompt", None)


def current_chat():
    return st.session_state.chats[st.session_state.chat_id]


def cb_new_chat():
    if current_chat()["messages"]:
        chat = new_chat_obj()
        st.session_state.chats[chat["id"]] = chat
        st.session_state.chat_id = chat["id"]
    st.session_state.view = NAV[0]


def cb_open_chat(chat_id):
    st.session_state.chat_id = chat_id
    st.session_state.view = NAV[0]


def cb_suggest(text):
    st.session_state.pending_prompt = text


def cb_rate(msg_id, rating):
    chat = current_chat()
    for i, m in enumerate(chat["messages"]):
        if m["id"] == msg_id:
            new = None if m.get("rating") == rating else rating
            m["rating"] = new
            log_feedback(
                settings.feedback_path,
                {
                    "chat_id": chat["id"],
                    "message_id": msg_id,
                    "rating": new or "cleared",
                    "question": chat["messages"][i - 1]["content"] if i else "",
                    "answer": m["content"],
                    "sources": [s.title for s in (m.get("sources") or [])],
                    "model": m.get("model"),
                },
            )
            if new:
                st.toast("Thanks for the feedback!", icon="👍" if new == "up" else "📝")
            break


def cb_regenerate():
    msgs = current_chat()["messages"]
    if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and msgs[-2]["role"] == "user":
        st.session_state.pending_prompt = msgs[-2]["content"]
        del msgs[-2:]


def get_pipeline(cfg, vector_store):
    sig = (cfg["provider"], cfg["model_name"], cfg["api_key"], id(vector_store))
    if st.session_state.get("pipeline_sig") != sig:
        st.session_state.pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_provider=cfg["provider"],
            model_name=cfg["model_name"] or None,
            api_key=cfg["api_key"] or None,
            fetch_k_multiplier=settings.fetch_k_multiplier,
            history_turns=settings.history_turns,
            max_continuations=settings.max_continuations,
        )
        st.session_state.pipeline_sig = sig
    pl = st.session_state.pipeline
    pl.temperature = cfg["temperature"]
    pl.max_tokens = cfg["max_tokens"]
    pl.top_k = cfg["top_k"]
    pl.confidence_threshold = cfg["threshold"]
    pl.neighbor_window = cfg["neighbor_window"]
    pl.max_context_chars = cfg["max_context_chars"]
    return pl


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def confidence_badge(score: float) -> str:
    if score >= 0.55:
        cls, label = "mi-badge-high", "Strong evidence match"
    elif score >= 0.35:
        cls, label = "mi-badge-med", "Moderate evidence match"
    else:
        cls, label = "mi-badge-low", "Weak evidence match"
    return f'<span class="mi-badge {cls}">● {label} · {score:.2f}</span>'


def sources_html(sources) -> str:
    cards = []
    for s in sources:
        text = html.escape((s.content or s.excerpt)[:420]) + ("…" if len(s.content or s.excerpt) > 420 else "")
        full = html.escape(s.content or s.excerpt).replace("\n", "<br>")
        link = (
            f" · <a href='{html.escape(s.url)}' target='_blank' rel='noopener'>open source</a>" if s.url else ""
        )
        more = f"<details><summary>Show full passage</summary><div class='mi-source-text'>{full}</div></details>" if len(s.content or "") > 420 else ""
        cards.append(
            f"<div class='mi-source'><div class='mi-source-title'>[{html.escape(s.doc_id)}] {html.escape(s.title)}</div>"
            f"<div class='mi-source-meta'>{html.escape(str(s.year or 'n/a'))} · relevance {s.relevance_score:.2f}{link}</div>"
            f"<div class='mi-source-text'>{text}</div>{more}</div>"
        )
    return "".join(cards)


def copy_button(text: str):
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        f"""
        <style>
          body{{margin:0;background:transparent;font-family:Inter,Segoe UI,system-ui,sans-serif}}
          button{{white-space:nowrap;display:inline-flex;align-items:center;gap:6px;height:34px;padding:0 12px;border:1px solid #e2e8f0;border-radius:10px;
                 background:#fff;color:#0f172a;font-size:13px;cursor:pointer;transition:all .15s}}
          button:hover{{border-color:#14b8a6;background:#f0fdfa}}
          button.ok{{color:#0f766e;border-color:#14b8a6;background:#f0fdfa}}
        </style>
        <button id="b" title="Copy response">📋 Copy</button>
        <script>
          const text = {payload};
          const b = document.getElementById('b');
          b.onclick = async () => {{
            try {{ await navigator.clipboard.writeText(text); }}
            catch (e) {{
              const t = document.createElement('textarea'); t.value = text; document.body.appendChild(t);
              t.select(); document.execCommand('copy'); t.remove();
            }}
            b.textContent = '✓ Copied'; b.classList.add('ok');
            setTimeout(() => {{ b.textContent = '📋 Copy'; b.classList.remove('ok'); }}, 1600);
          }};
        </script>
        """,
        height=40,
    )


def render_actions(msg, is_last):
    c_copy, c_up, c_down, c_regen, _ = st.columns([2.3, 1, 1, 1, 4])
    with c_copy:
        copy_button(msg["content"])
    with c_up:
        st.button("👍", key=f"up_{msg['id']}", help="Good answer", on_click=cb_rate, args=(msg["id"], "up"),
                  type="primary" if msg.get("rating") == "up" else "secondary")
    with c_down:
        st.button("👎", key=f"down_{msg['id']}", help="Bad answer", on_click=cb_rate, args=(msg["id"], "down"),
                  type="primary" if msg.get("rating") == "down" else "secondary")
    with c_regen:
        if is_last:
            st.button("🔄", key=f"regen_{msg['id']}", help="Regenerate response", on_click=cb_regenerate)


def render_assistant_extras(msg, is_last):
    if msg.get("sources"):
        meta = ""
        if msg.get("confidence") is not None:
            meta += confidence_badge(msg["confidence"])
        meta += (
            f"<span class='mi-meta'>{len(msg['sources'])} passages · retrieval {msg.get('retrieval_ms', 0):.0f} ms · "
            f"answer {msg.get('generation_ms', 0) / 1000:.1f} s · {msg.get('ts', '')}</span>"
        )
        st.markdown(meta, unsafe_allow_html=True)
        with st.expander(f"📚 Sources ({len(msg['sources'])})", expanded=False):
            st.markdown(sources_html(msg["sources"]), unsafe_allow_html=True)
    render_actions(msg, is_last)


def render_history(chat):
    msgs = chat["messages"]
    last_idx = len(msgs) - 1
    for i, msg in enumerate(msgs):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.markdown(msg["content"])
                render_assistant_extras(msg, is_last=(i == last_idx))


def render_welcome(vs):
    n_chunks = len(vs.documents)
    n_docs = len({d["doc_id"] for d in vs.documents}) if n_chunks else 0
    st.markdown(
        f"""
        <div class="mi-hero">
          <div class="mi-logo">🩺</div>
          <h1>How can I help you today?</h1>
          <p>Ask a medical question and get a cited answer grounded in
             {n_docs} verified document{'s' if n_docs != 1 else ''} ({n_chunks} passages).</p>
        </div>
        <div class="mi-chips-anchor"></div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (icon, title, question) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            st.button(f"{icon}  **{title}**\n\n{question}", key=f"sug_{i}", on_click=cb_suggest, args=(question,), use_container_width=True)


def render_footer():
    st.markdown(
        f"<div class='mi-footer'><a href='{html.escape(settings.ai_disclaimer_url)}' target='_blank' rel='noopener'>{DISCLAIMER_TEXT}</a></div>",
        unsafe_allow_html=True,
    )


def nearest_step(value):
    return min(TOKEN_STEPS, key=lambda s: abs(s - value))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def render_sidebar(vs):
    with st.sidebar:
        st.markdown(
            "<div class='mi-brand'><div class='mi-logo'>🩺</div><div><b>MedIntel</b><span>Cited medical answers</span></div></div>",
            unsafe_allow_html=True,
        )
        st.button("➕  New chat", on_click=cb_new_chat, use_container_width=True, type="primary")
        st.radio("Navigation", NAV, key="view", label_visibility="collapsed")

        chats = sorted(st.session_state.chats.values(), key=lambda c: -c["created"])
        chats = [c for c in chats if c["messages"]][:8]
        if chats:
            st.markdown("<div class='mi-side-label'>Recent chats</div>", unsafe_allow_html=True)
            for c in chats:
                st.button(
                    ("💬 " if c["id"] != st.session_state.chat_id else "▶ ") + c["title"],
                    key=f"chat_{c['id']}",
                    on_click=cb_open_chat,
                    args=(c["id"],),
                    use_container_width=True,
                )

        st.markdown("<div class='mi-side-label'>Settings</div>", unsafe_allow_html=True)
        with st.expander("🤖 Model", expanded=False):
            provider = st.selectbox(
                "LLM provider", PROVIDERS,
                index=PROVIDERS.index(settings.llm_provider) if settings.llm_provider in PROVIDERS else 0,
            )
            defaults = {
                "gemini": settings.llm_model if settings.llm_provider == "gemini" else "gemini-2.5-flash",
                "openai": settings.llm_model if settings.llm_provider == "openai" else "gpt-4o",
                "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
                "qwen": "Qwen/Qwen1.5-7B-Chat",
                "huggingface": "",
            }
            model_name = st.text_input("Model name", value=defaults[provider], key=f"model_{provider}")
            env_key = settings.gemini_api_key or os.getenv("GOOGLE_API_KEY", "") if provider == "gemini" else settings.openai_api_key
            key_in = st.text_input(f"{provider.upper()} API key", type="password", placeholder="Uses .env value if blank")
            api_key = key_in or env_key
            temperature = st.slider("Temperature", 0.0, 1.0, float(settings.llm_temperature), 0.05)
            cap = max_output_limit(provider, model_name)
            steps = [s for s in TOKEN_STEPS if s <= cap] or [cap]
            max_tokens = st.select_slider(
                "Max answer length (tokens)", options=steps,
                value=min(steps, key=lambda s: abs(s - settings.max_tokens)),
                help=f"Upper limit for this model: {cap:,} tokens. Cut-off answers are continued automatically.",
            )
        with st.expander("🔍 Retrieval & context", expanded=False):
            top_k = st.slider("Passages to retrieve (FAISS top_k)", 1, settings.max_top_k, min(settings.top_k_documents, settings.max_top_k))
            threshold = st.slider("Min. similarity", 0.0, 1.0, float(settings.retrieval_confidence_threshold), 0.05,
                                  help="MiniLM cosine scores for relevant text are typically 0.30–0.65.")
            neighbor_window = st.slider("Neighbouring chunks per hit", 0, 4, settings.neighbor_window,
                                        help="Widens each hit with the chunks before/after it → larger context for the model.")
            max_context_chars = st.select_slider("Max context sent to model (chars)",
                                                 options=[8000, 15000, 30000, 60000, 120000, 250000],
                                                 value=min([8000, 15000, 30000, 60000, 120000, 250000], key=lambda s: abs(s - settings.max_context_chars)))

        msgs = current_chat()["messages"]
        if msgs:
            transcript = "\n\n".join(f"**{'You' if m['role'] == 'user' else 'MedIntel'}:** {m['content']}" for m in msgs)
            st.download_button("⬇️  Export this chat", transcript, file_name="medintel_chat.md", use_container_width=True)
        if st.button("🔄  Reload index from disk", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.vector_store = load_vector_store()
            st.rerun()

    return {
        "provider": provider, "model_name": model_name, "api_key": api_key, "temperature": temperature,
        "max_tokens": max_tokens, "top_k": top_k, "threshold": threshold,
        "neighbor_window": neighbor_window, "max_context_chars": max_context_chars,
    }


# ----------------------------------------------------------------------------
# Chat view
# ----------------------------------------------------------------------------
def handle_prompt(prompt, cfg, vs):
    chat = current_chat()
    history = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]
    chat["messages"].append({"id": uuid.uuid4().hex[:8], "role": "user", "content": prompt, "ts": datetime.now().strftime("%H:%M")})
    if chat["title"] == "New chat":
        chat["title"] = prompt[:38] + ("…" if len(prompt) > 38 else "")

    with st.chat_message("user"):
        st.markdown(prompt)

    reply = {"id": uuid.uuid4().hex[:8], "role": "assistant", "content": "", "ts": datetime.now().strftime("%H:%M"),
             "model": f"{cfg['provider']}:{cfg['model_name']}"}
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        typing = st.empty()
        typing.markdown("<div class='mi-typing'><i></i><i></i><i></i>&nbsp;Searching medical sources…</div>", unsafe_allow_html=True)
        try:
            if not vs.documents:
                reply["content"] = "I don't have any indexed medical documents yet, so I can't answer safely. Add some in **📚 Knowledge base** first."
                typing.empty(); st.markdown(reply["content"])
            elif cfg["provider"] in ("gemini", "openai") and not cfg["api_key"]:
                reply["content"] = (f"🔑 No **{cfg['provider'].upper()}** API key found. Put `{cfg['provider'].upper()}_API_KEY=...` "
                                    "in your `.env` file (then restart) or paste a key under **Settings → Model**.")
                typing.empty(); st.markdown(reply["content"])
            else:
                pipeline = get_pipeline(cfg, vs)
                prepared = pipeline.prepare(prompt, top_k=cfg["top_k"], history=history,
                                            neighbor_window=cfg["neighbor_window"], max_context_chars=cfg["max_context_chars"])
                if prepared.insufficient:
                    typing.empty()
                    reply["content"] = INSUFFICIENT_MSG
                    st.markdown(reply["content"])
                    reply["retrieval_ms"] = prepared.retrieval_ms
                else:
                    typing.markdown("<div class='mi-typing'><i></i><i></i><i></i>&nbsp;Writing a grounded answer…</div>", unsafe_allow_html=True)
                    started = time.time()

                    def stream():
                        first = True
                        for piece in pipeline.stream_answer(prepared):
                            if first:
                                typing.empty(); first = False
                            yield piece
                        if first:
                            typing.empty()

                    answer = st.write_stream(stream())
                    response = pipeline.finalize(prepared, answer, (time.time() - started) * 1000)
                    reply.update(content=response.answer, sources=response.sources, confidence=response.confidence,
                                 retrieval_ms=response.retrieval_time_ms, generation_ms=response.generation_time_ms)
        except Exception as e:  # keep the chat alive on any provider error
            typing.empty()
            reply["content"] = f"⚠️ Something went wrong while generating the answer:\n\n`{e}`"
            st.markdown(reply["content"])

    chat["messages"].append(reply)
    st.rerun()


def render_chat_view(cfg, vs):
    chat = current_chat()
    typed = st.chat_input("Ask a medical question…")
    prompt = st.session_state.pop("pending_prompt", None) or typed

    if not chat["messages"] and not prompt:
        render_welcome(vs)
    else:
        render_history(chat)
    if prompt:
        handle_prompt(prompt, cfg, vs)
    render_footer()


# ----------------------------------------------------------------------------
# Knowledge base view (data ingestion)
# ----------------------------------------------------------------------------
def save_docs(vs, docs, filename):
    corpus_builder = MedicalCorpusBuilder()
    vs.add_documents(docs)
    vs.save(settings.vector_store_path)
    corpus_builder.save_corpus(docs, filename)


def render_knowledge_view(vs):
    st.markdown("## 📚 Knowledge base")
    st.caption("Every source added here becomes evidence the chatbot can retrieve and cite.")

    t_files, t_pubmed, t_gho, t_url, t_text = st.tabs(
        ["📄 Upload files", "🔬 PubMed", "🌍 WHO indicators", "🌐 WHO / CDC / NIH URL", "✍️ Paste text"]
    )

    with t_files:
        st.write("Upload textbook chapters, guideline PDFs, DOCX, TXT, or JSON (`[{title, content, source, year, url}]`).")
        uploaded = st.file_uploader("Choose file(s)", type=["pdf", "docx", "txt", "json"], accept_multiple_files=True)
        if st.button("📥 Ingest uploaded files", disabled=not uploaded):
            new_docs = []
            with st.spinner("Parsing files..."):
                for f in uploaded:
                    suffix = Path(f.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        if suffix == ".pdf":
                            for i, page_text in enumerate(DocumentLoader.load_pdf(tmp_path)):
                                if page_text.strip():
                                    new_docs.append(MedicalDocument(content=page_text, title=f"{Path(f.name).stem} — Page {i + 1}",
                                                                    source=f.name, metadata={"uploaded_as": f.name, "page": i + 1}))
                        elif suffix in (".docx", ".txt"):
                            content = DocumentLoader.load_docx(tmp_path) if suffix == ".docx" else DocumentLoader.load_txt(tmp_path)
                            if content.strip():
                                new_docs.append(MedicalDocument(content=content, title=Path(f.name).stem, source=f.name))
                        elif suffix == ".json":
                            for item in DocumentLoader.load_json(tmp_path):
                                if isinstance(item, dict) and item.get("content"):
                                    new_docs.append(MedicalDocument(content=item["content"], title=item.get("title", Path(f.name).stem),
                                                                    source=item.get("source", f.name), year=item.get("year"),
                                                                    url=item.get("url"), metadata=item.get("metadata", {})))
                    finally:
                        os.unlink(tmp_path)
            if new_docs:
                with st.spinner(f"Embedding {len(new_docs)} sections..."):
                    save_docs(vs, new_docs, f"upload_{int(time.time())}.json")
                st.success(f"Added {len(new_docs)} section(s) ✅")
                st.rerun()
            else:
                st.warning("No readable content found.")

    with t_pubmed:
        st.write("Fetch peer-reviewed abstracts from PubMed (NCBI E-utilities).")
        q = st.text_input("PubMed search query", placeholder="e.g. type 2 diabetes management")
        email = st.text_input("Contact email (required by NCBI)", value=settings.pubmed_email or os.getenv("PUBMED_EMAIL", ""))
        n = st.slider("Max abstracts", 5, 100, 20)
        if st.button("📥 Fetch & ingest from PubMed", disabled=not (q and email)):
            with st.spinner(f"Searching PubMed for '{q}'..."):
                loader = PubMedLoader(email=email)
                pmids = loader.search_pubmed(q, max_results=n)
                docs = loader.fetch_abstracts(pmids) if pmids else []
            if docs:
                with st.spinner("Embedding abstracts..."):
                    save_docs(vs, docs, f"pubmed_{int(time.time())}.json")
                st.success(f"Added {len(docs)} abstract(s) ✅")
                st.rerun()
            else:
                st.warning("No abstracts found.")

    with t_gho:
        st.write("Official statistics from WHO's Global Health Observatory API (no key needed).")
        loader = WHOGHOLoader()
        label = st.selectbox("Indicator", list(loader.COMMON_INDICATORS.values()))
        code = [k for k, v in loader.COMMON_INDICATORS.items() if v == label][0]
        st.caption(f"Indicator code: `{code}`")
        iso = st.text_input("Country ISO3 (optional)", placeholder="e.g. IND, USA, GBR")
        latest_n = st.slider("Max data points", 10, 200, 50)
        if st.button("📥 Fetch & ingest WHO GHO data"):
            with st.spinner("Querying WHO GHO..."):
                docs = loader.to_documents(code, label, iso or None, latest_n)
            if docs:
                with st.spinner("Embedding..."):
                    save_docs(vs, docs, f"who_gho_{code}_{int(time.time())}.json")
                st.success(f"Added {len(docs)} document(s) ✅")
                st.rerun()
            else:
                st.warning("No data returned.")

    with t_url:
        import requests
        from bs4 import BeautifulSoup

        def fetch_page(url):
            resp = requests.get(url, timeout=15, headers={"User-Agent": "MedIntelBot/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            return title, " ".join(soup.get_text(separator=" ").split())

        st.markdown("**Quick-add official WHO fact sheets**")
        picks = st.multiselect("Topics", list(WHO_FACT_SHEETS))
        if st.button("📥 Fetch selected fact sheets", disabled=not picks):
            added = 0
            with st.spinner("Fetching..."):
                for topic in picks:
                    try:
                        _, text = fetch_page(WHO_FACT_SHEETS[topic])
                        if len(text) > 200:
                            save_docs(vs, [MedicalDocument(content=text[:40000], title=f"WHO Fact Sheet: {topic}",
                                                           source="World Health Organization", url=WHO_FACT_SHEETS[topic])],
                                      f"who_factsheet_{topic}_{int(time.time())}.json")
                            added += 1
                    except Exception as e:
                        st.warning(f"Could not fetch {topic}: {e}")
            if added:
                st.success(f"Added {added} fact sheet(s) ✅")
                st.rerun()

        st.markdown("**Or any public URL**")
        url = st.text_input("Page URL", placeholder="https://www.who.int/news-room/fact-sheets/detail/diabetes")
        if st.button("📥 Fetch & ingest URL", disabled=not url):
            try:
                with st.spinner("Fetching..."):
                    title, text = fetch_page(url)
                if len(text) > 200:
                    save_docs(vs, [MedicalDocument(content=text[:40000], title=title,
                                                   source=url.split("/")[2] if "//" in url else url, url=url)],
                              f"webpage_{int(time.time())}.json")
                    st.success(f"Added '{title}' ✅")
                    st.rerun()
                else:
                    st.warning("Could not extract enough readable text.")
            except Exception as e:
                st.error(f"Failed to fetch page: {e}")

    with t_text:
        with st.form("paste_text_form"):
            title = st.text_input("Title")
            source = st.text_input("Source name", placeholder="e.g. Harrison's Internal Medicine, 21st ed.")
            year = st.text_input("Year (optional)")
            link = st.text_input("URL (optional)")
            content = st.text_area("Content", height=200)
            submitted = st.form_submit_button("📥 Ingest text")
        if submitted:
            if title and content:
                save_docs(vs, [MedicalDocument(content=content, title=title, source=source or "Manual entry",
                                               year=year or None, url=link or None)], f"manual_{int(time.time())}.json")
                st.success(f"Added '{title}' ✅")
                st.rerun()
            else:
                st.warning("Title and content are required.")


# ----------------------------------------------------------------------------
# Insights view
# ----------------------------------------------------------------------------
def render_insights_view(vs, cfg):
    import pandas as pd

    st.markdown("## 📊 Insights")
    if not vs.documents:
        st.info("No documents indexed yet.")
        return
    stats = vs.get_stats()
    cols = st.columns(4)
    for col, (value, label) in zip(cols, [(stats["total_chunks"], "Chunks"), (stats["unique_documents"], "Documents"),
                                          (stats["dimension"], "Embedding dim"), (f"{cfg['top_k']}", "Passages / question")]):
        col.markdown(f"<div class='mi-metric'><h3>{value}</h3><span>{label}</span></div>", unsafe_allow_html=True)

    st.markdown("#### Answer feedback")
    rows = [r for r in load_feedback(settings.feedback_path) if r.get("rating") in ("up", "down")]
    if rows:
        df = pd.DataFrame(rows)
        up, down = int((df.rating == "up").sum()), int((df.rating == "down").sum())
        c1, c2 = st.columns(2)
        c1.metric("👍 Helpful", up)
        c2.metric("👎 Not helpful", down)
        if down:
            st.caption("Recent answers rated 👎 — good candidates for new sources or prompt fixes:")
            st.dataframe(df[df.rating == "down"][["ts", "question", "model"]].tail(10), use_container_width=True, hide_index=True)
    else:
        st.caption("No ratings yet — use 👍 / 👎 under any answer.")

    st.markdown("#### Indexed documents")
    seen = {}
    for d in vs.documents:
        seen.setdefault(d["doc_id"], {"Title": d["title"], "Source": d["source"], "Year": d.get("year"), "Chunks": 0})["Chunks"] += 1
    st.dataframe(pd.DataFrame(list(seen.values())), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
init_state()
vector_store = st.session_state.vector_store
config = render_sidebar(vector_store)
view = st.session_state.view
apply_width(view)

if view == NAV[0]:
    render_chat_view(config, vector_store)
elif view == NAV[1]:
    render_knowledge_view(vector_store)
else:
    render_insights_view(vector_store, config)

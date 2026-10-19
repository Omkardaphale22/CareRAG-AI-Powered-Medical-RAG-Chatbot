"""
Configuration management for MedIntel RAG Chatbot

Every value here can be overridden from the environment / .env file
(e.g. MAX_TOKENS=16384, TOP_K_DOCUMENTS=15).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # ------------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------------
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "gemini"  # gemini, openai, mistral, qwen, huggingface
    # gemini-2.5-flash: 1,048,576-token input window, 65,536-token output limit.
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.1

    # ------------------------------------------------------------------
    # RESPONSE SIZE  (was 1000)
    # Maximum tokens the model may write per answer. Gemini 2.5 models allow up
    # to 65,536; gemini-2.0-* allow 8,192 (the pipeline caps it automatically).
    # ------------------------------------------------------------------
    max_tokens: int = 8192
    # If an answer is cut off by the token limit, ask the model to continue
    # (up to this many extra calls) so users never see a half-finished answer.
    max_continuations: int = 2

    # ------------------------------------------------------------------
    # FAISS RETRIEVAL SIZE  (top_k was 5, threshold was 0.75)
    # ------------------------------------------------------------------
    top_k_documents: int = 12        # default chunks retrieved per question
    max_top_k: int = 30              # upper bound exposed in the UI / API
    fetch_k_multiplier: int = 3      # over-fetch from FAISS, then de-duplicate + diversify (MMR)
    # all-MiniLM-L6-v2 cosine scores for genuinely relevant text are usually
    # 0.30 - 0.65, so 0.75 filtered out almost everything.
    retrieval_confidence_threshold: float = 0.30

    # ------------------------------------------------------------------
    # CONTEXT WINDOW  (what actually gets sent to the LLM)
    # ------------------------------------------------------------------
    neighbor_window: int = 1         # also pull N chunks before/after every hit
    max_context_chars: int = 60000   # ~15k tokens of evidence per question
    history_turns: int = 6           # previous chat turns included for follow-ups

    # ------------------------------------------------------------------
    # API configuration
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # ------------------------------------------------------------------
    # Vector store configuration (applies to newly ingested documents)
    # ------------------------------------------------------------------
    vector_store_path: str = "./data/vector_store"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    feedback_path: str = "./data/feedback/feedback.jsonl"
    ai_disclaimer_url: str = (
        "https://support.anthropic.com/en/articles/8525154-claude-is-providing-"
        "incorrect-or-misleading-responses-what-s-going-on"
    )
    pubmed_email: str = ""

    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    app_name: str = "MedIntel - Medical RAG Chatbot"
    app_version: str = "2.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # unknown keys in .env must not crash the app


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

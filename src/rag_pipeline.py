"""
RAG Pipeline with LLM integration (Gemini, OpenAI, Mistral, Qwen, HuggingFace)

Upgrades over v1
----------------
* Larger context: retrieved chunks are widened with their neighbours, merged
  into passages and fed to the model together with the recent chat history.
* Larger answers: max output tokens raised (capped per model), and answers that
  hit the token limit are automatically continued instead of being cut off.
* Larger retrieval: higher top_k, FAISS over-fetching + de-duplication + MMR.
* Streaming: prepare() -> stream_answer() -> finalize() lets a UI show the
  answer token by token. query() still works as a single blocking call.
* Gemini now uses the maintained `google-genai` SDK (falls back to the old
  `google-generativeai` package if that is the only one installed).
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

from src.models import QueryResponse, Source
from src.vector_store import VectorStore

# --- Gemini SDKs (imported defensively; only one of them is needed) ----------
try:
    from google import genai as genai_new
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai_new = None
    genai_types = None

DISCLAIMER = (
    "⚠️ This information is for educational purposes only and is not a "
    "substitute for professional medical advice."
)
ERROR_PREFIX = "Error generating response"
INSUFFICIENT_MSG = (
    "I'm sorry, I don't have enough verified information to answer that safely. "
    "Please consult with a healthcare professional for accurate medical advice."
)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    MISTRAL = "mistral"
    QWEN = "qwen"
    HUGGINGFACE = "huggingface"
    GEMINI = "gemini"


def max_output_limit(provider: str, model_name: str) -> int:
    """Largest output size (tokens) we will request from a given model."""
    name = (model_name or "").lower()
    if provider == LLMProvider.GEMINI.value:
        if any(tag in name for tag in ("gemini-2.0", "gemini-1.5", "gemini-1.0")):
            return 8192
        return 65536  # gemini-2.5-* / gemini-3-*
    if provider == LLMProvider.OPENAI.value:
        if any(tag in name for tag in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")):
            return 16384
        return 4096
    return 4096  # local HuggingFace models


class _GeminiBackend:
    """Thin wrapper hiding the difference between google-genai and google-generativeai."""

    def __init__(self, model_name: str, api_key: Optional[str]):
        self.model_name = model_name
        self.mode = "new" if genai_new is not None else "legacy"

        if self.mode == "new":
            self.client = genai_new.Client(api_key=api_key) if api_key else genai_new.Client()
        else:
            try:
                import google.generativeai as legacy
            except ImportError as exc:
                raise ImportError(
                    "Install the Gemini SDK:  pip install google-genai"
                ) from exc
            self._legacy = legacy
            if api_key:
                legacy.configure(api_key=api_key)
            self.model = legacy.GenerativeModel(model_name)

    @staticmethod
    def _reason_name(reason: Any) -> Optional[str]:
        if reason is None:
            return None
        return getattr(reason, "name", None) or str(reason).split(".")[-1]

    def stream(self, prompt: str, temperature: float, max_tokens: int, state: Dict[str, Any]) -> Iterator[str]:
        """Yield text pieces. `state["finish"]` / `state["blocked"]` are filled in as we go."""
        if self.mode == "new":
            config = genai_types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=max_tokens
            )
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name, contents=prompt, config=config
            ):
                feedback = getattr(chunk, "prompt_feedback", None)
                if feedback is not None and getattr(feedback, "block_reason", None):
                    state["blocked"] = self._reason_name(feedback.block_reason)
                candidates = getattr(chunk, "candidates", None) or []
                if candidates and getattr(candidates[0], "finish_reason", None):
                    state["finish"] = self._reason_name(candidates[0].finish_reason)
                text = chunk.text
                if text:
                    yield text
        else:
            legacy = self._legacy
            response = self.model.generate_content(
                prompt,
                generation_config=legacy.types.GenerationConfig(
                    temperature=temperature, max_output_tokens=max_tokens
                ),
                stream=True,
            )
            for chunk in response:
                try:
                    text = chunk.text
                except ValueError:  # chunk without text parts (e.g. safety stop)
                    text = ""
                if text:
                    yield text
            try:
                state["finish"] = self._reason_name(response.candidates[0].finish_reason)
            except Exception:
                pass


@dataclass
class PreparedQuery:
    """Everything needed to generate an answer (result of the retrieval step)."""
    question: str
    prompt: str = ""
    passages: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_ms: float = 0.0
    insufficient: bool = False


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline"""

    SYSTEM_PROMPT = """You are MedIntel, a medical AI assistant designed to provide fact-based, reliable, and explainable answers.
You are connected to a retrieval system that provides verified medical documents.
Your task is to answer medical queries using ONLY the information retrieved below.

CRITICAL RULES:
1. Use clear, simple language while maintaining medical accuracy
2. Every factual claim MUST have a citation in the format [DOC_X] where X is the document number
3. At the end of your response, list all sources as: "Sources: [DOC_1: Title], [DOC_2: Title], ..."
4. If the retrieved documents do not answer the question, reply: "I'm sorry, I don't have enough verified information to answer that safely."
5. NEVER fabricate or infer medical facts not present in the retrieved documents
6. Do NOT diagnose users or prescribe treatments. Your purpose is to inform, not diagnose.
7. Always include this disclaimer: "⚠️ This information is for educational purposes only and is not a substitute for professional medical advice."
8. Be thorough: cover every relevant point the documents support (for example causes, symptoms, warning signs, when to seek care, prevention) and organise longer answers with short headings or bullet points. Do not stop early.
9. The conversation history is only for understanding follow-up questions. It is NOT a source of medical facts - every fact must come from the retrieved context.

Conversation so far:
{history}

Retrieved Context:
{context}

User Query: {question}

Provide your answer with inline citations:"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        top_k: int = 12,
        confidence_threshold: float = 0.30,
        neighbor_window: int = 1,
        max_context_chars: int = 60000,
        fetch_k_multiplier: int = 3,
        history_turns: int = 6,
        max_continuations: int = 2,
    ):
        self.vector_store = vector_store
        self.llm_provider = LLMProvider(llm_provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.neighbor_window = neighbor_window
        self.max_context_chars = max_context_chars
        self.fetch_k_multiplier = fetch_k_multiplier
        self.history_turns = history_turns
        self.max_continuations = max_continuations
        self.client = None  # OpenAI client
        self.gemini: Optional[_GeminiBackend] = None
        self.model = None  # HuggingFace model
        self.tokenizer = None  # HuggingFace tokenizer

        if self.llm_provider == LLMProvider.OPENAI:
            import openai  # lazy: only needed for this provider

            self.model_name = model_name or "gpt-4-turbo-preview"
            self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

        elif self.llm_provider == LLMProvider.GEMINI:
            self.model_name = model_name or "gemini-2.5-flash"
            self.gemini = _GeminiBackend(self.model_name, api_key)
            print(f"✅ Gemini model initialized: {self.model_name} (SDK: {self.gemini.mode})")

        elif self.llm_provider == LLMProvider.MISTRAL:
            self.model_name = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
            self._load_huggingface_model()

        elif self.llm_provider == LLMProvider.QWEN:
            self.model_name = model_name or "Qwen/Qwen1.5-7B-Chat"
            self._load_huggingface_model()

        elif self.llm_provider == LLMProvider.HUGGINGFACE:
            self.model_name = model_name
            if not model_name:
                raise ValueError("model_name required for HuggingFace provider")
            self._load_huggingface_model()

    # ------------------------------------------------------------------
    # Model loading / generation backends
    # ------------------------------------------------------------------
    def _load_huggingface_model(self):
        """Load HuggingFace model and tokenizer (imports are lazy: they are slow)"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        print("Model loaded successfully")

    def output_cap(self) -> int:
        """Max output tokens actually requested (user setting, capped by the model's limit)."""
        return max(256, min(self.max_tokens, max_output_limit(self.llm_provider.value, self.model_name)))

    def _generate_openai(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are MedIntel, a medical AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.output_cap(),
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"{ERROR_PREFIX}: {str(e)}"

    def _generate_huggingface(self, prompt: str) -> str:
        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.output_cap(),
                    temperature=max(self.temperature, 0.01),
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            return response
        except Exception as e:
            return f"{ERROR_PREFIX}: {str(e)}"

    def _gemini_stream(self, prompt: str) -> Iterator[str]:
        """Stream from Gemini; if the answer hits the token limit, ask the model to continue."""
        produced = ""
        current_prompt = prompt

        for _ in range(self.max_continuations + 1):
            state: Dict[str, Any] = {}
            got = ""
            try:
                for piece in self.gemini.stream(current_prompt, self.temperature, self.output_cap(), state):
                    got += piece
                    yield piece
            except Exception as e:
                yield ("\n\n" if (produced or got) else "") + f"{ERROR_PREFIX}: {e}"
                return

            produced += got
            if state.get("finish") != "MAX_TOKENS":
                if not produced.strip():
                    reason = state.get("blocked") or state.get("finish")
                    yield (
                        "I couldn't produce an answer for that request"
                        + (f" (model stop reason: {reason})" if reason else "")
                        + ". Please rephrase your question or consult a healthcare professional."
                    )
                return

            # Cut off by the length limit -> continue where we stopped
            current_prompt = (
                prompt
                + "\n\n---\nYour answer so far (it was cut off by the length limit):\n"
                + produced[-6000:]
                + "\n---\nContinue EXACTLY where the answer stopped. Do not repeat earlier text and do not "
                "restart. Finish the remaining points, keep the [DOC_X] citation style, and end with the "
                "Sources line and the disclaimer."
            )

        yield (
            "\n\n*(The answer was shortened because it reached the output limit — "
            "raise “Max answer length” in the settings for more detail.)*"
        )

    def _generate_stream(self, prompt: str) -> Iterator[str]:
        if self.llm_provider == LLMProvider.GEMINI:
            yield from self._gemini_stream(prompt)
        elif self.llm_provider == LLMProvider.OPENAI:
            yield self._generate_openai(prompt)
        else:
            yield self._generate_huggingface(prompt)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    @staticmethod
    def _format_context(passages: List[Dict[str, Any]]) -> str:
        """Format retrieved passages as numbered context blocks"""
        parts = []
        for i, p in enumerate(passages, 1):
            doc = p["doc"]
            parts.append(
                f"[DOC_{i}] {doc['title']} ({doc['source']}, {doc.get('year') or 'N/A'})\n"
                f"Content: {doc['content']}\n"
                f"Relevance Score: {p['score']:.3f}\n"
            )
        return "\n".join(parts)

    def _format_history(self, history: Optional[List[Dict[str, Any]]]) -> str:
        if not history:
            return "(none)"
        turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        turns = turns[-(self.history_turns * 2):]
        lines = []
        for m in turns:
            text = str(m["content"]).replace(DISCLAIMER, "").strip()
            if m["role"] == "assistant" and len(text) > 1500:
                text = text[:1500] + " …"
            lines.append(f"{'User' if m['role'] == 'user' else 'MedIntel'}: {text}")
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _retrieval_query(question: str, history: Optional[List[Dict[str, Any]]]) -> str:
        """Short follow-ups ("what about children?") borrow the previous user question for retrieval."""
        q = question.strip()
        if history and len(q.split()) <= 8:
            last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
            if last_user:
                return f"{last_user}\n{q}"
        return q

    def _calculate_confidence(self, passages: List[Dict[str, Any]], answer: str) -> float:
        """Evidence-match score: mean similarity of the best passages, adjusted for citations."""
        if not passages:
            return 0.0
        top = sorted((p["score"] for p in passages), reverse=True)[:3]
        confidence = sum(top) / len(top)

        if "[DOC_" in answer:
            confidence *= 1.1
        uncertainty_phrases = ["don't have enough", "cannot answer", "insufficient information", "not enough verified"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases) and confidence < self.confidence_threshold:
            confidence *= 0.7
        return min(confidence, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(
        self,
        question: str,
        top_k: Optional[int] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        neighbor_window: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ) -> PreparedQuery:
        """Retrieval step: FAISS search -> context expansion -> prompt."""
        k = top_k or self.top_k
        window = self.neighbor_window if neighbor_window is None else neighbor_window
        budget = max_context_chars or self.max_context_chars

        start = time.time()
        hits = self.vector_store.search(
            self._retrieval_query(question, history),
            top_k=k,
            score_threshold=self.confidence_threshold,
            fetch_k=k * max(1, self.fetch_k_multiplier),
            diversify=True,
        )
        passages = self.vector_store.expand_hits(hits, window=window, max_chars=budget)
        retrieval_ms = (time.time() - start) * 1000

        if not passages:
            return PreparedQuery(question=question, retrieval_ms=retrieval_ms, insufficient=True)

        prompt = self.SYSTEM_PROMPT.format(
            history=self._format_history(history),
            context=self._format_context(passages),
            question=question,
        )
        return PreparedQuery(question=question, prompt=prompt, passages=passages, retrieval_ms=retrieval_ms)

    def stream_answer(self, prepared: PreparedQuery) -> Iterator[str]:
        """Yield the answer in pieces; the medical disclaimer is appended if the model omitted it."""
        collected = ""
        for piece in self._generate_stream(prepared.prompt):
            collected += piece
            yield piece
        if "educational purposes" not in collected.lower() and not collected.lstrip().startswith(ERROR_PREFIX):
            yield "\n\n" + DISCLAIMER

    def finalize(self, prepared: PreparedQuery, answer: str, generation_ms: float) -> QueryResponse:
        """Wrap a finished answer into a QueryResponse (sources, confidence, timings)."""
        if prepared.insufficient:
            return QueryResponse(
                question=prepared.question,
                answer=INSUFFICIENT_MSG,
                contexts=[],
                sources=[],
                confidence=0.0,
                retrieval_time_ms=prepared.retrieval_ms,
                generation_time_ms=0.0,
                total_time_ms=prepared.retrieval_ms,
                warning="⚠️ Insufficient verified information available. Please consult a healthcare professional.",
            )

        sources = []
        for i, p in enumerate(prepared.passages, 1):
            doc, content = p["doc"], p["doc"]["content"]
            sources.append(
                Source(
                    doc_id=f"DOC_{i}",
                    title=doc["title"],
                    year=doc.get("year"),
                    url=doc.get("url"),
                    relevance_score=p["score"],
                    excerpt=content[:300] + "..." if len(content) > 300 else content,
                    content=content,
                )
            )

        if "educational purposes" not in answer.lower() and not answer.lstrip().startswith(ERROR_PREFIX):
            answer += "\n\n" + DISCLAIMER

        return QueryResponse(
            question=prepared.question,
            answer=answer,
            contexts=[s.content for s in sources],
            sources=sources,
            confidence=self._calculate_confidence(prepared.passages, answer),
            retrieval_time_ms=prepared.retrieval_ms,
            generation_time_ms=generation_ms,
            total_time_ms=prepared.retrieval_ms + generation_ms,
            warning="Always consult with qualified healthcare professionals for medical decisions.",
        )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> QueryResponse:
        """Process a query through the RAG pipeline (blocking)."""
        prepared = self.prepare(question, top_k=top_k, history=history)
        if prepared.insufficient:
            return self.finalize(prepared, "", 0.0)

        start = time.time()
        answer = "".join(self.stream_answer(prepared))
        return self.finalize(prepared, answer, (time.time() - start) * 1000)

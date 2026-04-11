import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List

from src.data_loader import load_medical_corpus, load_medqa_data
from src.evaluation.metrics import RAGEvaluator
from src.generation.llm_generators import GroqGenerator
from src.retrieval.langchain_faiss_store import LangchainFAISSStore

logger = logging.getLogger(__name__)


class MedRAGService:
    def __init__(self, config):
        self.config = config
        self._lock = threading.RLock()
        self.vector_store = None
        self.llm_gen = None
        self.evaluator = RAGEvaluator()
        self.is_initialized = False
        self._retrieval_cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._retrieval_cache_size = 128
        self._load_runtime_state()

    def _save_runtime_state(self, payload: Dict[str, Any]):
        try:
            with open(self.config.runtime_state_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            logger.warning("Failed to persist runtime state: %s", exc)

    def _load_runtime_state(self):
        if not os.path.exists(self.config.runtime_state_path):
            return
        try:
            with open(self.config.runtime_state_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.is_initialized = bool(payload.get("is_initialized", False))
        except Exception as exc:
            logger.warning("Failed to load runtime state: %s", exc)

    def _cache_get(self, key: str):
        if key not in self._retrieval_cache:
            return None
        self._retrieval_cache.move_to_end(key)
        return self._retrieval_cache[key]

    def _cache_set(self, key: str, value):
        self._retrieval_cache[key] = value
        self._retrieval_cache.move_to_end(key)
        if len(self._retrieval_cache) > self._retrieval_cache_size:
            self._retrieval_cache.popitem(last=False)

    def initialize(self, corpus_size: int, force_reindex: bool = False):
        with self._lock:
            if corpus_size < 1 or corpus_size > self.config.max_corpus_size:
                raise ValueError(
                    f"corpus_size must be between 1 and {self.config.max_corpus_size}"
                )

            if self.is_initialized and not force_reindex and self.vector_store and self.llm_gen:
                return {"status": "success", "message": "System already initialized."}

            start_time = time.time()
            logger.info("Initializing Medical Knowledge Base (Size: %s)...", corpus_size)

            _, full_dataset = load_medqa_data(num_eval_questions=5)
            medical_corpus = load_medical_corpus(
                dataset_to_fallback=full_dataset, max_docs=corpus_size
            )

            self.vector_store = LangchainFAISSStore(
                db_dir=self.config.faiss_db_dir,
                device="cpu",
            )

            has_existing = (
                self.vector_store.vector_store is not None
                and getattr(self.vector_store.vector_store.index, "ntotal", 0) > 0
            )
            if force_reindex or not has_existing:
                self.vector_store.add_documents(medical_corpus)

            self.llm_gen = GroqGenerator(self.config.api_key, max_retries=self.config.llm_max_retries)
            self.is_initialized = True
            self._retrieval_cache.clear()

            duration = time.time() - start_time
            self._save_runtime_state(
                {
                    "is_initialized": True,
                    "initialized_at": int(time.time()),
                    "corpus_size": corpus_size,
                    "duration_seconds": round(duration, 2),
                }
            )

            return {
                "status": "success",
                "message": f"Loaded {len(medical_corpus)} docs in {duration:.2f}s.",
            }

    def health_ready(self):
        return bool(self.is_initialized and self.vector_store is not None and self.llm_gen is not None)

    def _retrieve(self, query: str, top_k: int):
        cached = self._cache_get(query)
        if cached is not None:
            return cached
        docs = self.vector_store.retrieve(query, top_k=top_k)
        self._cache_set(query, docs)
        return docs

    def chat(self, question: str, reference: str = ""):
        if not self.is_initialized:
            raise RuntimeError("Initialize system first")
        if not question or not question.strip():
            raise ValueError("No query provided")
        if len(question) > self.config.max_query_chars:
            raise ValueError(
                f"Query exceeds max length ({self.config.max_query_chars} characters)"
            )

        start_time = time.time()
        with self._lock:
            retrieved_docs = self._retrieve(question, top_k=self.config.retrieval_top_k)
            answer = self.llm_gen.generate(question, retrieved_docs)
            baseline_answer = self.llm_gen.generate_no_context(question)

        latency = time.time() - start_time
        metrics = {"Latency": f"{latency:.2f}s"}

        eval_reference = reference
        if not eval_reference and retrieved_docs:
            eval_reference = retrieved_docs[0].get("text", "")[:1000]

        if eval_reference:
            rag_scores = self.evaluator.compute_rouge_scores(answer, eval_reference)
            base_scores = self.evaluator.compute_rouge_scores(baseline_answer, eval_reference)
            metrics["RAG_ROUGE_L"] = round(rag_scores["rougeL"], 4)
            metrics["Baseline_ROUGE_L"] = round(base_scores["rougeL"], 4)
            diff = rag_scores["rougeL"] - base_scores["rougeL"]
            metrics["Accuracy_Improvement"] = f"{diff * 100:.1f}%"

        sources = [
            {
                "title": d.get("metadata", {}).get("title", "Source"),
                "text": f"{d.get('text', '')[:400]}...",
            }
            for d in retrieved_docs
        ]

        return {
            "status": "success",
            "answer": answer,
            "baseline_answer": baseline_answer,
            "sources": sources,
            "metrics": metrics,
        }

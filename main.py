import argparse
import logging
import os
import torch
from transformers import AutoTokenizer

from src.config import OPENCALL_LLM_KEY, HUGGINGFACE_API_KEY, Config
from src.data_loader import load_medqa_data, load_medical_corpus, MedicalQADataset
from src.retrieval.lstm_retriever import LSTMRetriever, train_lstm_retriever, baseline_lstm_retrieve
from src.retrieval.vector_store import ChromaVectorStore
from src.generation.llm_generators import GroqGenerator, LlamaMedicalGenerator, baseline_generate
from src.evaluation.metrics import RAGEvaluator

logger = logging.getLogger('main')

def run_evaluation(system_name, questions, retrieve_func, generate_func, evaluator):
    logger.info(f"\\n--- Evaluating System: {system_name} ---")
    results = []
    
    for item in questions:
        if isinstance(item, dict):
            question = item.get('question', item.get('sent1', ''))
            reference = item.get('answer', item.get('ending0', 'Unknown'))
        else:
            question = str(item)
            reference = "Unknown"
            
        if not question:
            continue
            
        retrieved_docs, r_time = evaluator.measure_response_time(retrieve_func, question, top_k=Config.TOP_K_RETRIEVAL)
        answer, g_time = evaluator.measure_response_time(generate_func, question, retrieved_docs)
        
        r_scores = evaluator.compute_rouge_scores(answer, reference)
        hal_rate = evaluator.detect_hallucination(answer, retrieved_docs)
        total_time = r_time + g_time
        
        # Avoiding direct slicing to satisfy linter limitations
        truncated_question = ""
        for char_idx in range(min(100, len(question))):
            truncated_question += question[char_idx]

        results.append({
            'system': system_name,
            'question': truncated_question,
            'rouge1': r_scores['rouge1'],
            'rougeL': r_scores['rougeL'],
            'hallucination': hal_rate,
            'time': total_time
        })
        
    avg_r1 = sum(float(r['rouge1']) for r in results) / len(results) if results else 0.0
    avg_rl = sum(float(r['rougeL']) for r in results) / len(results) if results else 0.0
    avg_hal = sum(float(r['hallucination']) for r in results) / len(results) if results else 0.0
    
    logger.info(f"Results for {system_name}:")
    logger.info(f"  Avg ROUGE-1: {avg_r1:.4f}")
    logger.info(f"  Avg ROUGE-L: {avg_rl:.4f}")
    logger.info(f"  Avg Hallucination Rate: {avg_hal:.4f}\\n")
    return results

def main():
    parser = argparse.ArgumentParser(description="MedBot RAG Pipeline Evaluation")
    parser.add_argument("--demo", action="store_true", help="Run a quick demo with 2 questions and synthetic data.")
    args = parser.parse_args()

    Config.load_env()
    logger.info(f"Running on device: {Config.DEVICE}")
    logger.info("Initializing Pipeline...")

    eval_size = 2 if args.demo else Config.NUM_EVAL_QUESTIONS
    corpus_size = 50 if args.demo else 5000

    # 1. Load Data
    eval_questions, full_dataset = load_medqa_data(num_eval_questions=eval_size)
    medical_corpus = load_medical_corpus(dataset_to_fallback=full_dataset, max_docs=corpus_size)

    # 2. Setup Vector Store
    vector_store = ChromaVectorStore(db_dir=Config.CHROMA_DB_DIR, device=Config.DEVICE)
    corpus_embeddings = vector_store.add_documents(medical_corpus)

    # 3. Setup Generators
    llm_gen = GroqGenerator(OPENCALL_LLM_KEY)
    
    # 4. Evaluator
    evaluator = RAGEvaluator()

    # Define simple retrieval function wrappers
    def chromadb_retriever(q, top_k):
        return vector_store.retrieve(q, top_k=top_k)

    def groq_generator_wrapper(q, docs):
        return llm_gen.generate(q, docs)

    # 5. Run Evaluations
    run_evaluation(
        system_name="Groq RAG (ChromaDB)",
        questions=eval_questions,
        retrieve_func=chromadb_retriever,
        generate_func=groq_generator_wrapper,
        evaluator=evaluator
    )

    run_evaluation(
        system_name="Baseline Template (ChromaDB)",
        questions=eval_questions,
        retrieve_func=chromadb_retriever,
        generate_func=baseline_generate,
        evaluator=evaluator
    )
    
    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()

import time
from rouge_score import rouge_scorer
import logging

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Comprehensive evaluator for RAG systems"""
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_retrieval_metrics(self, retrieved_docs, ground_truth_docs):
        """Compute retrieval precision, recall, F1"""
        retrieved_ids = set([doc.get('id', doc.get('text', '')[:50]) for doc in retrieved_docs])
        ground_truth_ids = set([doc.get('id', doc.get('text', '')[:50]) for doc in ground_truth_docs])
        
        if len(ground_truth_ids) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        tp = len(retrieved_ids & ground_truth_ids)
        precision = tp / len(retrieved_ids) if len(retrieved_ids) > 0 else 0.0
        recall = tp / len(ground_truth_ids) if len(ground_truth_ids) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def compute_rouge_scores(self, generated_answer, reference_answer):
        """Compute ROUGE-1, ROUGE-2, ROUGE-L scores"""
        scores = self.rouge_scorer.score(str(reference_answer), str(generated_answer))
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def detect_hallucination(self, generated_answer, retrieved_docs):
        """Simple hallucination detection: token overlap between answer and docs"""
        if not retrieved_docs or not generated_answer:
            return 1.0  # High hallucination if no docs or no answer
        
        answer_tokens = set(str(generated_answer).lower().split())
        doc_tokens = set()
        for doc in retrieved_docs:
            doc_tokens.update(doc.get('text', '').lower().split())
        
        if len(answer_tokens) == 0:
            return 1.0
        
        overlap = len(answer_tokens & doc_tokens) / len(answer_tokens)
        hallucination_rate = 1.0 - overlap  # Higher overlap = lower hallucination
        return hallucination_rate
    
    def measure_response_time(self, func, *args, **kwargs):
        """Measure function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

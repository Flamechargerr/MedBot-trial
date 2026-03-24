import logging
from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def load_medqa_data(num_eval_questions=100, split="test"):
    """
    Loads the MedQA USMLE dataset or a fallback.
    """
    logger.info("Loading MedQA dataset...")
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        logger.info(f"Loaded {len(dataset)} questions from MedQA USMLE")
    except Exception as e:
        logger.warning(f"Error loading MedQA: {e}. Attempting fallback to medmcqa.")
        dataset = load_dataset("medmcqa", split="validation")
        logger.info(f"Loaded {len(dataset)} questions from MedMCQA")

    # Sample evaluation questions
    eval_questions = dataset.shuffle(seed=42).select(range(min(num_eval_questions, len(dataset))))
    return eval_questions, dataset


def load_medical_corpus(dataset_to_fallback: Dataset = None, max_docs=5000):
    """
    Loads medical documents serving as the retrieval corpus.
    """
    logger.info("Loading medical corpus...")
    try:
        corpus_dataset = load_dataset("pubmed", split=f"train[:{max_docs}]")
        medical_corpus = [
            {
                "text": item['MedlineCitation']['Article']['Abstract']['AbstractText'][0] if item.get('MedlineCitation', {}).get('Article', {}).get('Abstract', {}).get('AbstractText') else "",
                "id": str(i),
                "title": item.get('MedlineCitation', {}).get('Article', {}).get('ArticleTitle', "Untitled")
            }
            for i, item in enumerate(corpus_dataset)
        ]
        medical_corpus = [doc for doc in medical_corpus if doc['text']]
    except Exception as e:
        logger.warning(f"PubMed corpus loading failed: {e}. Building synthetic corpus.")
        if dataset_to_fallback is None:
            raise ValueError("No fallback dataset provided to generate synthetic corpus.")
            
        medical_corpus = [
            {
                "text": item['question'] if 'question' in item else item.get('sent1', ''),
                "id": str(i),
                "title": f"Medical Question {i}"
            }
            for i, item in enumerate(dataset_to_fallback)
        ]

    logger.info(f"Medical corpus prepared with {len(medical_corpus)} documents.")
    return medical_corpus


class MedicalQADataset(Dataset):
    """
    PyTorch Dataset wrapper for LSTM training
    """
    def __init__(self, questions, tokenizer, max_length=128):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        if isinstance(question, dict):
            text = question.get('question', question.get('sent1', ''))
        else:
            text = str(question)
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'text': text
        }

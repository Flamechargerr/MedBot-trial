# MedBot Phase 3 - Quick Reference Card

## 🚀 Quick Start (30 seconds)

### Google Colab
1. Upload `MedBot_Phase3.ipynb` to Colab
2. Runtime → Change runtime type → GPU
3. Run all cells

### Local
```bash
bash setup_phase3.sh
python test_environment.py
jupyter notebook MedBot_Phase3.ipynb
```

## 📋 What You Get

### 3 Models Evaluated
- ✅ Baseline LSTM Retriever
- ✅ ChatGPT RAG (GPT-3.5)
- ✅ Llama-2 Medical RAG

### 4 Metrics Computed
- ✅ Retrieval F1
- ✅ ROUGE-1/2/L
- ✅ Hallucination Rate
- ✅ Response Time

### 8 Output Files
- `lstm_retriever_model.pt` (model)
- `chroma_db/` (vector DB)
- `phase3_results.csv` (detailed)
- `phase3_aggregate_stats.csv` (summary)
- `phase3_summary.txt` (report)
- `*.png` (4 visualizations)
- `medbot_phase3_results.zip` (package)

## 🔑 API Keys

### Pre-configured (No action needed)
```python
EMERGENT_LLM_KEY = 'sk-emergent-56016CcDc780e503a4'
```

### Optional (for Llama-2)
```python
os.environ['HUGGINGFACE_API_KEY'] = 'your_token'
```

## ⚙️ Key Configuration

```python
CONFIG = {
    'num_epochs': 5,              # Training epochs
    'top_k_retrieval': 5,         # Documents to retrieve
    'num_eval_questions': 100,    # Questions to evaluate
}
```

## 🎯 Expected Runtime

| Hardware | Time |
|----------|------|
| Colab T4 GPU | ~30-45 min |
| Local GPU (16GB) | ~20-30 min |
| CPU only | ~2-3 hours |

## 📊 Expected Results

| Model | ROUGE-1 | Speed |
|-------|---------|-------|
| Baseline | ~0.25 | Fast |
| ChatGPT | ~0.45 | Medium |
| Llama-2 | ~0.40 | Slow |

## 🔧 Quick Fixes

### Out of Memory?
```python
CONFIG['batch_size'] = 8
CONFIG['num_eval_questions'] = 50
```

### Too Slow?
```python
CONFIG['num_epochs'] = 3
CONFIG['num_eval_questions'] = 50
```

### API Errors?
- Check internet connection
- Verify API key
- Check rate limits

## 📁 Files You Need

**Required:**
- `MedBot_Phase3.ipynb` ← Main file

**Optional:**
- `requirements.txt` (local setup)
- `PHASE3_README.md` (full guide)
- `test_environment.py` (verification)

## ✅ Success Indicators

During run, you should see:
- ✅ GPU detected message
- ✅ Models loading progress
- ✅ Training progress bars
- ✅ Evaluation progress bars
- ✅ Plots displaying
- ✅ Files created

## ⚠️ Warning Signs

Watch for:
- ❌ "CUDA out of memory" → Reduce batch size
- ❌ "API rate limit" → Add delays
- ❌ "Dataset not found" → Will use fallback
- ❌ "Model not available" → Will skip

## 🎓 Understanding Output

### phase3_results.csv
- Each row = one question
- Columns = all metrics per system

### phase3_aggregate_stats.csv
- Each row = one system
- Statistics (mean, std) per metric

### Plots
1. **ROUGE comparison** → Answer quality
2. **Hallucination** → Factual accuracy
3. **Response time** → Speed analysis
4. **Radar chart** → Overall comparison

## 📞 Need Help?

1. **Setup issues** → Check `test_environment.py`
2. **Usage help** → See `PHASE3_README.md`
3. **Errors** → Read error message carefully
4. **Results** → Check `phase3_summary.txt`

## 🎯 Next Actions

After running:
1. Review `phase3_summary.txt`
2. Analyze visualizations
3. Examine `phase3_results.csv`
4. Download `medbot_phase3_results.zip`

## 💡 Pro Tips

1. **Use GPU** → 10x faster
2. **Start small** → Test with 10 questions first
3. **Monitor progress** → Watch tqdm bars
4. **Save often** → Checkpoint models
5. **Document results** → Keep notes

## 🌟 Key Success Metrics

Your implementation is successful if:
- ✅ All 3 models run without errors
- ✅ Evaluation completes on 100+ questions
- ✅ Results CSV has valid data
- ✅ Plots are generated
- ✅ ROUGE scores are > 0

## 📈 Benchmarks

Compare your results to:
- Baseline LSTM: ROUGE-1 ~0.25
- ChatGPT RAG: ROUGE-1 ~0.45
- Published MedRAG: ROUGE-1 ~0.50

## 🎉 You're Ready!

Everything is set up. Just run the notebook and wait for results!

---

**Quick Links:**
- Main Notebook: `MedBot_Phase3.ipynb`
- Full Guide: `PHASE3_README.md`
- Delivery Summary: `DELIVERY_SUMMARY.md`

**Version**: 1.0 | **Status**: ✅ Ready

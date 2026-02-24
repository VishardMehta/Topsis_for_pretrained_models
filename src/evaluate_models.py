from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import numpy as np
import time
import torch


def evaluate_model(model_name, num_samples=30):
    print(f"Evaluating {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
    articles, references = [], []
    for i, s in enumerate(dataset):
        if i >= num_samples:
            break
        articles.append(s["article"])
        references.append(s["highlights"])

    is_t5 = "t5" in model_name.lower()
    preds = []

    start = time.time()
    for a in articles:
        inp = tokenizer(f"summarize: {a}" if is_t5 else a, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            out = model.generate(inp["input_ids"], max_length=150, min_length=30, num_beams=4, length_penalty=2.0)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
    elapsed = time.time() - start

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, references):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)

    return {"Model": model_name, "ROUGE-1": round(np.mean(r1), 4), "ROUGE-2": round(np.mean(r2), 4),
            "ROUGE-L": round(np.mean(rl), 4), "Throughput": round(len(articles) / elapsed, 2), "Size": round(size_mb, 2)}
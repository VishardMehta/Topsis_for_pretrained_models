from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
import numpy as np
import time
import torch


def evaluate_model(model_name, num_samples=30):

    print(f"Evaluating {model_name}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    # Load CNN/DailyMail dataset in streaming mode (no full download)
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)

    articles = []
    references = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        articles.append(sample["article"])
        references.append(sample["highlights"])

    # T5 and Flan-T5 models require "summarize: " prefix
    is_t5 = "t5" in model_name.lower()

    pred_summaries = []

    # Measure inference time
    start = time.time()
    for article in articles:
        input_text = f"summarize: {article}" if is_t5 else article

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        pred_summaries.append(summary)
    end = time.time()

    throughput = len(articles) / (end - start)

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(pred_summaries, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    # Calculate model size (MB)
    size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)

    print(f"  -> ROUGE-1: {np.mean(rouge1_scores):.4f}, ROUGE-2: {np.mean(rouge2_scores):.4f}, ROUGE-L: {np.mean(rougeL_scores):.4f}, Throughput: {throughput:.2f}, Size: {size_mb:.2f} MB")

    return {
        "Model": model_name,
        "ROUGE-1": round(np.mean(rouge1_scores), 4),
        "ROUGE-2": round(np.mean(rouge2_scores), 4),
        "ROUGE-L": round(np.mean(rougeL_scores), 4),
        "Throughput": round(throughput, 2),
        "Size": round(size_mb, 2),
    }
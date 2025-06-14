import torch
from transformers import pipeline
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score
from rich.progress import Progress


def evaluate_perplexity(model, tokenizer, text):
    """
    Menghitung perplexity dari sebuah teks menggunakan model causal LM.
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()


def evaluate_accuracy(model_name, dataset, labels=None):
    """
    Calculate accuracy using zero-shot-classification pipeline.
    labels: list of labels for classification (default: all unique labels in dataset)
    """
    if labels is None:
        labels = sorted(list({d["label"] for d in dataset}))
    nlp = pipeline("zero-shot-classification", model=model_name)
    correct = 0
    print(f"[INFO] Starting accuracy evaluation for model: {model_name}")
    with Progress() as progress:
        task = progress.add_task("Evaluating accuracy...", total=len(dataset))
        for data in dataset:
            text = data["text"]
            true_label = data["label"]
            prediction = nlp(text, candidate_labels=labels)
            predicted_label = prediction["labels"][0]
            if predicted_label == true_label:
                correct += 1
            progress.update(task, advance=1)
    accuracy = correct / len(dataset)
    print(f"[INFO] Finished accuracy evaluation for model: {model_name}")
    return accuracy


def evaluate_bleu(model, tokenizer, dataset):
    """
    Calculate BLEU score for generative (seq2seq) dataset.
    """
    generated_texts = []
    reference_texts = []
    print(f"[INFO] Starting BLEU evaluation for model: {getattr(model, 'name_or_path', str(model))}")
    with Progress() as progress:
        task = progress.add_task("Evaluating BLEU...", total=len(dataset))
        for data in dataset:
            source_text = data.get("text", "")
            target_text = data.get("label", "")
            if not source_text or not target_text:
                progress.update(task, advance=1)
                continue
            inputs = tokenizer(
                source_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = model.generate(
                inputs["input_ids"], max_length=512, num_beams=5, early_stopping=True
            )
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(predicted_text)
            reference_texts.append([target_text])
            progress.update(task, advance=1)
    generated_tokens = [tokenizer.tokenize(text) for text in generated_texts]
    reference_tokens = [tokenizer.tokenize(ref[0]) for ref in reference_texts]
    bleu_score = float(corpus_bleu(reference_tokens, generated_tokens))
    print(f"[INFO] Finished BLEU evaluation for model: {getattr(model, 'name_or_path', str(model))}")
    return bleu_score


# Modular evaluation entry points

def run_causal_lm_evaluation(model, tokenizer, dataset, metrics):
    results = {}
    if 'perplexity' in metrics:
        results['perplexity'] = evaluate_perplexity(model, tokenizer, dataset[0]['text'])
    return results

def run_seq2seq_evaluation(model, tokenizer, dataset, metrics):
    results = {}
    if 'bleu' in metrics:
        results['bleu'] = evaluate_bleu(model, tokenizer, dataset)
    return results

def run_zero_shot_classification_evaluation(model_name, dataset, metrics):
    results = {}
    if 'accuracy' in metrics:
        results['accuracy'] = evaluate_accuracy(model_name, dataset)
    return results


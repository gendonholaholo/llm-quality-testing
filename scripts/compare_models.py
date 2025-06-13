import yaml
import json
import csv
from pathlib import Path
from llm_eval.evaluation import evaluate_perplexity, evaluate_accuracy, evaluate_bleu
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table


def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_sample_data(filepath):
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_csv(results, csv_path):
    if not results:
        return
    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def save_json(results, json_path):
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

def print_leaderboard(results):
    table = Table(title="Comparative Model Leaderboard")
    if not results:
        print("No results to display.")
        return
    for key in results[0].keys():
        table.add_column(key, style="bold")
    for row in results:
        table.add_row(*[str(row[k]) for k in results[0].keys()])
    console = Console()
    console.print(table)

def load_model_and_tokenizer(model_name, task_type):
    if task_type == 'seq2seq':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def main():
    config = load_yaml_config('configs/default_config.yaml')
    models = config['models']
    dataset_path = config['dataset']
    output_csv = config['output_csv']
    output_json = config['output_json']
    dataset = load_sample_data(dataset_path)
    results = []
    for model_name in models:
        # Perplexity (Causal LM)
        model_causal, tokenizer_causal = load_model_and_tokenizer(model_name, 'causal')
        perplexity = evaluate_perplexity(model_causal, tokenizer_causal, dataset[0]['text'])
        # Accuracy (Zero-shot)
        accuracy = None
        try:
            accuracy = evaluate_accuracy(model_name, dataset)
        except Exception:
            accuracy = 'N/A'
        # BLEU (Seq2Seq)
        try:
            model_seq2seq, tokenizer_seq2seq = load_model_and_tokenizer(model_name, 'seq2seq')
            bleu = evaluate_bleu(model_seq2seq, tokenizer_seq2seq, dataset)
        except Exception:
            bleu = 'N/A'
        results.append({
            'model_name': model_name,
            'perplexity': round(perplexity, 4) if isinstance(perplexity, float) else perplexity,
            'accuracy': round(accuracy, 4) if isinstance(accuracy, float) else accuracy,
            'bleu': round(bleu, 4) if isinstance(bleu, float) else bleu
        })
    print_leaderboard(results)
    save_csv(results, output_csv)
    save_json(results, output_json)
    print(f"\nLeaderboard saved to: {output_csv} and {output_json}")

if __name__ == "__main__":
    main() 
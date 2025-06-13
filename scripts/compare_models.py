import yaml
import json
import csv
from pathlib import Path
from llm_eval.evaluation import evaluate_perplexity, evaluate_accuracy, evaluate_bleu
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
import click


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

@click.group()
def cli():
    """LLM Quality Testing CLI"""
    pass

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--model-name', '-m', multiple=True, help='Override model names (can specify multiple times)')
@click.option('--dataset-path', '-d', type=click.Path(), help='Override dataset path')
@click.option('--output-csv', type=click.Path(), help='Override output CSV file path')
@click.option('--output-json', type=click.Path(), help='Override output JSON file path')
@click.option('--metrics', '-M', multiple=True, type=click.Choice(['perplexity', 'accuracy', 'bleu']), help='Metrics to compute (default: all)')
def evaluate(config_path, model_name, dataset_path, output_csv, output_json, metrics):
    """Evaluate and compare LLMs using a config YAML file."""
    config = load_yaml_config(config_path)
    # Override config with CLI options if provided
    if model_name:
        config['models'] = list(model_name)
    if dataset_path:
        config['dataset'] = dataset_path
    if output_csv:
        config['output_csv'] = output_csv
    if output_json:
        config['output_json'] = output_json
    selected_metrics = set(metrics) if metrics else {'perplexity', 'accuracy', 'bleu'}
    models = config['models']
    dataset_path = config['dataset']
    output_csv = config['output_csv']
    output_json = config['output_json']
    dataset = load_sample_data(dataset_path)
    results = []
    for model_name in models:
        row = {'model_name': model_name}
        if 'perplexity' in selected_metrics:
            model_causal, tokenizer_causal = load_model_and_tokenizer(model_name, 'causal')
            try:
                perplexity = evaluate_perplexity(model_causal, tokenizer_causal, dataset[0]['text'])
                row['perplexity'] = round(perplexity, 4) if isinstance(perplexity, float) else perplexity
            except Exception:
                row['perplexity'] = 'N/A'
        if 'accuracy' in selected_metrics:
            try:
                accuracy = evaluate_accuracy(model_name, dataset)
                row['accuracy'] = round(accuracy, 4) if isinstance(accuracy, float) else accuracy
            except Exception:
                row['accuracy'] = 'N/A'
        if 'bleu' in selected_metrics:
            try:
                model_seq2seq, tokenizer_seq2seq = load_model_and_tokenizer(model_name, 'seq2seq')
                bleu = evaluate_bleu(model_seq2seq, tokenizer_seq2seq, dataset)
                row['bleu'] = round(bleu, 4) if isinstance(bleu, float) else bleu
            except Exception:
                row['bleu'] = 'N/A'
        results.append(row)
    print_leaderboard(results)
    save_csv(results, output_csv)
    save_json(results, output_json)
    print(f"\nLeaderboard saved to: {output_csv} and {output_json}")

def main(args=None):
    if args is not None:
        cli.main(args=args, standalone_mode=False)
    else:
        cli.main(standalone_mode=False)

if __name__ == "__main__":
    main() 
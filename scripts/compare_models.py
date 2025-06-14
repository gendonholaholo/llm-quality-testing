import yaml
import json
import csv
from pathlib import Path
from llm_eval.evaluation import (
    run_causal_lm_evaluation,
    run_seq2seq_evaluation,
    run_zero_shot_classification_evaluation,
)
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from rich.console import Console
from rich.table import Table
import click
import sys


def load_yaml_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] File konfigurasi '{config_path}' tidak ditemukan. Harap buat file konfigurasi terlebih dahulu.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] File konfigurasi '{config_path}' tidak bisa di-parse YAML: {e}")
        sys.exit(1)

def load_sample_data(filepath):
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_csv(results, csv_path):
    if not results:
        return
    # Collect all possible keys from all rows
    all_keys = set()
    for row in results:
        all_keys.update(row.keys())
    keys = list(all_keys)
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
        table.add_row(*[str(row.get(k, 'N/A')) for k in results[0].keys()])
    console = Console()
    console.print(table)

def load_model_and_tokenizer(model_name, task_type):
    try:
        if not isinstance(model_name, str):
            raise ValueError(f"model_name harus string, dapat: {repr(model_name)}")
        if task_type == 'seq2seq':
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif task_type == 'causal_lm':
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = None
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"[ERROR] Gagal memuat model/tokenizer '{model_name}': {e}")
        return None, None

@click.group()
def cli():
    """LLM Quality Testing CLI"""
    pass

@cli.command()
@click.argument('config_path', type=click.Path(exists=False))
def evaluate(config_path):
    """Evaluate and compare LLMs using a config YAML file."""
    config = load_yaml_config(config_path)
    models = config['models']
    dataset_path = config['dataset']
    output_csv = config['output_csv']
    output_json = config['output_json']
    dataset = load_sample_data(dataset_path)
    results = []
    print(f"[INFO] Loaded config from {config_path}")
    print(f"[INFO] Loaded dataset from {dataset_path} with {len(dataset)} samples")
    for model_cfg in models:
        if isinstance(model_cfg, str):
            model_name = model_cfg
            task_type = None
            metrics = []
        elif isinstance(model_cfg, dict):
            model_name = model_cfg.get('name')
            if not isinstance(model_name, str):
                print(f"[ERROR] Field 'name' in model config must be a string. Skipping: {model_cfg}")
                continue
            task_type = model_cfg.get('task_type')
            metrics = model_cfg.get('metrics', [])
        else:
            print(f"[ERROR] Unrecognized model config format: {model_cfg}")
            continue
        # Auto-detect architecture if not specified
        if not task_type:
            try:
                config_obj = AutoConfig.from_pretrained(model_name)
                archs = getattr(config_obj, 'architectures', [])
                if any('CausalLM' in a for a in archs):
                    task_type = 'causal_lm'
                elif any('Seq2Seq' in a for a in archs):
                    task_type = 'seq2seq'
                elif any('Classification' in a for a in archs):
                    task_type = 'zero_shot_classification'
                else:
                    task_type = 'causal_lm'  # Default fallback
            except Exception as e:
                print(f"[ERROR] Failed to detect model architecture for '{model_name}': {e}")
                continue
        row = {'model_name': model_name, 'task_type': task_type}
        print(f"[INFO] Starting evaluation for model: {model_name} (task: {task_type})")
        if task_type == 'causal_lm':
            model, tokenizer = load_model_and_tokenizer(model_name, 'causal_lm')
            if model is None or tokenizer is None:
                row['error'] = 'Failed to load model/tokenizer.'
            else:
                eval_results = run_causal_lm_evaluation(model, tokenizer, dataset, metrics)
                row.update(eval_results)
        elif task_type == 'seq2seq':
            model, tokenizer = load_model_and_tokenizer(model_name, 'seq2seq')
            if model is None or tokenizer is None:
                row['error'] = 'Failed to load model/tokenizer.'
            else:
                eval_results = run_seq2seq_evaluation(model, tokenizer, dataset, metrics)
                row.update(eval_results)
        elif task_type == 'zero_shot_classification':
            eval_results = run_zero_shot_classification_evaluation(model_name, dataset, metrics)
            row.update(eval_results)
        else:
            row['error'] = f'Unknown task_type: {task_type}'
        print(f"[INFO] Finished evaluation for model: {model_name}")
        results.append(row)
    print("[INFO] Printing leaderboard...")
    print_leaderboard(results)
    print(f"[INFO] Saving results to {output_csv} and {output_json}")
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
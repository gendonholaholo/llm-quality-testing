import os
import yaml
import json
import csv
import pytest
from unittest.mock import patch
from scripts import compare_models
import subprocess

def test_comparative_leaderboard_workflow(tmp_path, monkeypatch):
    # Siapkan config dummy dengan struktur baru
    config = {
        'models': [
            {'name': 'dummy-model', 'task_type': 'causal_lm', 'metrics': ['perplexity']},
            {'name': 'dummy-model-2', 'task_type': 'seq2seq', 'metrics': ['bleu']}
        ],
        'dataset': str(tmp_path / 'dummy_data.json'),
        'output_csv': str(tmp_path / 'leaderboard.csv'),
        'output_json': str(tmp_path / 'leaderboard.json')
    }
    with open(tmp_path / 'dummy_config.yaml', 'w') as f:
        yaml.dump(config, f)
    dataset = [{"text": "a", "label": "b"}]
    with open(tmp_path / 'dummy_data.json', 'w') as f:
        json.dump(dataset, f)
    # Patch model loading dan evaluator
    monkeypatch.setattr(compare_models, 'load_yaml_config', lambda x: config)
    monkeypatch.setattr(compare_models, 'load_sample_data', lambda x: dataset)
    monkeypatch.setattr(compare_models, 'load_model_and_tokenizer', lambda m, t: (None, None))
    monkeypatch.setattr(compare_models, 'print_leaderboard', lambda x: None)
    monkeypatch.setattr(compare_models, 'run_causal_lm_evaluation', lambda m, t, d, metrics: {'perplexity': 1.0} if 'perplexity' in metrics else {})
    monkeypatch.setattr(compare_models, 'run_seq2seq_evaluation', lambda m, t, d, metrics: {'bleu': 0.5} if 'bleu' in metrics else {})
    monkeypatch.setattr(compare_models, 'run_zero_shot_classification_evaluation', lambda m, d, metrics: {'accuracy': 1.0} if 'accuracy' in metrics else {})
    # Jalankan main
    with patch('builtins.print'):
        compare_models.main(['evaluate', str(tmp_path / 'dummy_config.yaml')])
    # Cek file output
    assert os.path.exists(config['output_csv'])
    assert os.path.exists(config['output_json'])
    with open(config['output_csv']) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert rows[0]['model_name'] == 'dummy-model'
        assert rows[1]['model_name'] == 'dummy-model-2'
    with open(config['output_json']) as f:
        data = json.load(f)
        assert data[0]['model_name'] == 'dummy-model'
        assert data[1]['model_name'] == 'dummy-model-2'

def test_invalid_model_config_handling(tmp_path, monkeypatch):
    # Config dengan model config salah format
    config = {
        'models': [
            {'task_type': 'causal_lm', 'metrics': ['perplexity']},  # missing 'name'
            123,  # not a string or dict
            {'name': None, 'task_type': 'causal_lm', 'metrics': ['perplexity']},
        ],
        'dataset': str(tmp_path / 'dummy_data.json'),
        'output_csv': str(tmp_path / 'leaderboard.csv'),
        'output_json': str(tmp_path / 'leaderboard.json')
    }
    with open(tmp_path / 'dummy_config.yaml', 'w') as f:
        yaml.dump(config, f)
    dataset = [{"text": "a", "label": "b"}]
    with open(tmp_path / 'dummy_data.json', 'w') as f:
        json.dump(dataset, f)
    monkeypatch.setattr(compare_models, 'load_yaml_config', lambda x: config)
    monkeypatch.setattr(compare_models, 'load_sample_data', lambda x: dataset)
    monkeypatch.setattr(compare_models, 'load_model_and_tokenizer', lambda m, t: (None, None))
    monkeypatch.setattr(compare_models, 'print_leaderboard', lambda x: None)
    monkeypatch.setattr(compare_models, 'run_causal_lm_evaluation', lambda m, t, d, metrics: {'perplexity': 1.0} if 'perplexity' in metrics else {})
    monkeypatch.setattr(compare_models, 'run_seq2seq_evaluation', lambda m, t, d, metrics: {'bleu': 0.5} if 'bleu' in metrics else {})
    monkeypatch.setattr(compare_models, 'run_zero_shot_classification_evaluation', lambda m, d, metrics: {'accuracy': 1.0} if 'accuracy' in metrics else {})
    # Jalankan main, pastikan tidak crash
    with patch('builtins.print') as mock_print:
        compare_models.main(['evaluate', str(tmp_path / 'dummy_config.yaml')])
        # Pastikan error handling terpanggil
        printed = '\n'.join(str(a[0]) for a in mock_print.call_args_list)
        assert "Field 'name' pada model config harus string" in printed or "Format model config tidak dikenali" in printed 

def test_missing_config_file(tmp_path):
    # Path file config yang tidak ada
    missing_config = tmp_path / 'tidak_ada.yaml'
    result = subprocess.run([
        'python', 'scripts/compare_models.py', 'evaluate', str(missing_config)
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "tidak ditemukan" in result.stdout or "tidak ditemukan" in result.stderr 
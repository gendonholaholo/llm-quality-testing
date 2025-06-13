import os
import yaml
import json
import csv
import pytest
from unittest.mock import patch
from scripts import compare_models

def test_comparative_leaderboard_workflow(tmp_path, monkeypatch):
    # Siapkan config dummy
    config = {
        'models': ['dummy-model'],
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
    monkeypatch.setattr(compare_models, 'evaluate_perplexity', lambda m, t, text: 1.0)
    monkeypatch.setattr(compare_models, 'evaluate_accuracy', lambda m, d: 1.0)
    monkeypatch.setattr(compare_models, 'evaluate_bleu', lambda m, t, d: 0.5)
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
    with open(config['output_json']) as f:
        data = json.load(f)
        assert data[0]['model_name'] == 'dummy-model' 
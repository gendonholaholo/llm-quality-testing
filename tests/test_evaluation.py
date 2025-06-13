import pytest
from llm_eval import evaluation
from unittest.mock import MagicMock
import torch

@pytest.fixture
def dummy_tokenizer():
    class DummyTokenizer:
        def __call__(self, text, return_tensors=None, **kwargs):
            return {'input_ids': [0, 1, 2]}
        def tokenize(self, text):
            return text.split()
        def decode(self, ids, skip_special_tokens=True):
            return "decoded text"
    return DummyTokenizer()

@pytest.fixture
def dummy_model():
    class DummyModel:
        def __call__(self, **kwargs):
            class Output:
                loss = torch.tensor(1.0)
            return Output()
        def generate(self, input_ids, **kwargs):
            return [[0, 1, 2]]
    return DummyModel()

def test_evaluate_perplexity(dummy_model, dummy_tokenizer):
    result = evaluation.evaluate_perplexity(dummy_model, dummy_tokenizer, "test")
    assert isinstance(result, float)

def test_evaluate_accuracy(monkeypatch):
    # Patch pipeline to return always correct label
    def dummy_pipeline(task, model):
        def fn(text, candidate_labels):
            return {'labels': [candidate_labels[0]]}
        return fn
    monkeypatch.setattr(evaluation, 'pipeline', dummy_pipeline)
    dataset = [{"text": "a", "label": "class_1"}, {"text": "b", "label": "class_1"}]
    acc = evaluation.evaluate_accuracy("dummy", dataset, labels=["class_1"])
    assert acc == 1.0

def test_evaluate_bleu(dummy_model, dummy_tokenizer):
    dataset = [{"text": "a", "label": "b"}]
    result = evaluation.evaluate_bleu(dummy_model, dummy_tokenizer, dataset)
    assert isinstance(result, float)

# Tes regresi: BLEU = 0 tetap float, tidak error
def test_evaluate_bleu_zero(dummy_model, dummy_tokenizer):
    # Simulasikan dataset yang pasti BLEU = 0
    dataset = [{"text": "a", "label": "completelydifferent"}]
    result = evaluation.evaluate_bleu(dummy_model, dummy_tokenizer, dataset)
    assert isinstance(result, float)
    assert result == 0.0 
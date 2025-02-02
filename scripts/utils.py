from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer  

def load_model(model_name: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_sample_data(filepath: str):
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

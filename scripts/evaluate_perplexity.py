from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name: str):
    """Memuat model dan tokenizer dari Hugging Face."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def calculate_perplexity(model, tokenizer, text: str):
    """Menghitung perplexity dari sebuah teks."""
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    return perplexity.item()

if __name__ == "__main__":
    model_name = "gpt2"  
    model, tokenizer = load_model(model_name)
    
    text = "This is a sample text to calculate perplexity."
    perplexity = calculate_perplexity(model, tokenizer, text)
    
    print(f"Perplexity: {perplexity}")

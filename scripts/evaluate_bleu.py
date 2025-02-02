from nltk.translate.bleu_score import corpus_bleu  
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import load_model, load_sample_data

def evaluate_bleu(model, tokenizer, dataset):
    generated_texts = []
    reference_texts = []
    
    for data in dataset:
        source_text = data.get("text", "")  
        target_text = data.get("label", "")
        
        if not source_text or not target_text:
            continue
        
        inputs = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        outputs = model.generate(inputs["input_ids"], max_length=512, num_beams=5, early_stopping=True)
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generated_texts.append(predicted_text)
        reference_texts.append([target_text])  
    
    generated_tokens = [tokenizer.tokenize(text) for text in generated_texts]
    reference_tokens = [tokenizer.tokenize(ref[0]) for ref in reference_texts]  
    
    bleu_score = corpus_bleu(reference_tokens, generated_tokens)
    return bleu_score

if __name__ == "__main__":
    model_name = "facebook/bart-large-cnn"
    model, tokenizer = load_model(model_name) 
    
    dataset = load_sample_data("data/sample_data.json")
    bleu = evaluate_bleu(model, tokenizer, dataset)
    print(f"BLEU Score: {bleu}")

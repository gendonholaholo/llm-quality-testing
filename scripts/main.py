from evaluate_perplexity import calculate_perplexity
from evaluate_accuracy import evaluate_accuracy
from evaluate_bleu import evaluate_bleu
from utils import load_model, load_sample_data

def main():
    model_name = "gpt2"  
    model, tokenizer = load_model(model_name)

    text = "This is a test sentence."
    perplexity = calculate_perplexity(model, tokenizer, text)
    print(f"Perplexity: {perplexity}")

    dataset = load_sample_data("data/sample_data.json")
    accuracy = evaluate_accuracy(model, dataset)  
    print(f"Accuracy: {accuracy}")

    bleu = evaluate_bleu(model, tokenizer, dataset)  
    print(f"BLEU Score: {bleu}")

if __name__ == "__main__":
    main()

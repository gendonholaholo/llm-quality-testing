from sklearn.metrics import accuracy_score
from transformers import pipeline
from utils import load_sample_data

def evaluate_accuracy(model_name, dataset):
    nlp = pipeline('zero-shot-classification', model=model_name)
    labels = ["class_1", "class_2"]
    correct = 0
    for data in dataset:
        text = data["text"]
        true_label = data["label"]
        
        prediction = nlp(text, candidate_labels=labels)
        
        print(prediction)  

        predicted_label = prediction['labels'][0]
        
        if predicted_label == true_label:
            correct += 1
    accuracy = correct / len(dataset)
    return accuracy

if __name__ == "__main__":
    model_name = "facebook/bart-large-mnli"  
    dataset = load_sample_data("data/sample_data.json")
    accuracy = evaluate_accuracy(model_name, dataset)
    print(f"Accuracy: {accuracy}")

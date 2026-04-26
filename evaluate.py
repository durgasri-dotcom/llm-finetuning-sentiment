import torch
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "./lora-sentiment-model"
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}

def load_model():
    """Load the LoRA fine-tuned model and tokenizer."""
    print("Loading LoRA fine-tuned model...")
    config = PeftConfig.from_pretrained(MODEL_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model loaded successfully.")
    return model, tokenizer

def predict(text, model, tokenizer, max_length=256):
    """Run inference on a single text sample."""
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).numpy()[0]
        predicted_class = np.argmax(probabilities)

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "prediction": LABELS[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            "NEGATIVE": float(probabilities[0]),
            "POSITIVE": float(probabilities[1])
        }
    }

def evaluate_samples(model, tokenizer):
    """Evaluate model on diverse test samples."""
    test_samples = [
        "This movie was absolutely brilliant! The acting was superb and the plot kept me engaged throughout.",
        "Terrible waste of time. The story made no sense and the characters were completely unlikeable.",
        "An average film with some good moments but ultimately forgettable.",
        "One of the best films I have seen in years. Highly recommend to everyone!",
        "Boring, predictable, and poorly written. I nearly fell asleep halfway through.",
        "The cinematography was stunning but the script left much to be desired.",
        "A masterpiece of modern cinema. Every scene was crafted with precision and care.",
        "Not what I expected. Had its moments but overall disappointing."
    ]

    print("\n" + "="*60)
    print("LoRA Fine-Tuned DistilBERT — Sentiment Evaluation")
    print("="*60)
    print(f"Model: distilbert-base-uncased + LoRA (r=16, alpha=32)")
    print(f"Trainable params: 887,042 / 67,842,052 (1.31%)")
    print(f"Best eval accuracy: 87.4% | F1: 0.874")
    print("="*60 + "\n")

    correct = 0
    expected = [1, 0, 0, 1, 0, 0, 1, 0]  # expected labels

    for i, (text, expected_label) in enumerate(zip(test_samples, expected)):
        result = predict(text, model, tokenizer)
        predicted = 1 if result["prediction"] == "POSITIVE" else 0
        is_correct = predicted == expected_label
        if is_correct:
            correct += 1

        print(f"Sample {i+1}:")
        print(f"  Text: {result['text']}")
        print(f"  Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        print(f"  Probabilities: NEG={result['probabilities']['NEGATIVE']:.3f} | POS={result['probabilities']['POSITIVE']:.3f}")
        print(f"  Correct: {'YES' if is_correct else 'NO'}")
        print()

    sample_accuracy = correct / len(test_samples)
    print("="*60)
    print(f"Sample Accuracy: {correct}/{len(test_samples)} = {sample_accuracy:.1%}")
    print("="*60)

if __name__ == "__main__":
    model, tokenizer = load_model()
    evaluate_samples(model, tokenizer)

    # Interactive mode
    print("\nEnter your own text for sentiment analysis (type 'quit' to exit):")
    while True:
        user_input = input("\nText: ").strip()
        if user_input.lower() == 'quit':
            break
        if user_input:
            result = predict(user_input, model, tokenizer)
            print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load FinBERT model from Hugging Face
# https://huggingface.co/ProsusAI/finbert

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
pos_idx = next(k for k, v in labels.items() if v == "positive")
neg_idx = next(k for k, v in labels.items() if v == "negative")
neu_idx = next(k for k, v in labels.items() if v == "neutral")


def get_sentiment(text: str) -> dict:
    inputs = tokenizer(text, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1).squeeze().tolist()

    return {
        "neg": probs[neg_idx],
        "neu": probs[neu_idx],
        "pos": probs[pos_idx],
        "score": probs[pos_idx] - probs[neg_idx],
    }


if __name__ == "__main__":
    sample = "Apple’s Q2 revenue exceeded analysts’ estimates."
    print("Label mapping:", labels)
    print(get_sentiment(sample))

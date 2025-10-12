from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load FinBERT model from Hugging face
# https://huggingface.co/ProsusAI/finbert
# For details: https://github.com/ProsusAI/finBERT

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

def get_sentiment(text: str) -> dict:
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    return {"neg": probs[0], "neu": probs[1], "pos": probs[2], "score": probs[2] - probs[0]}

if __name__ == "__main__":
    sample = "Apple’s Q2 revenue exceeded analysts’ estimates."
    print(get_sentiment(sample))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

def load_model():
    # Define paths
    MODEL_CHECKPOINT = './experiments/distilbert-base-uncased'
    SAVED_MODEL_PATH = "./experiments/bank_fineuned_model.pt"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=7  # Adjust this based on your training labels
    )

    # Load LoRA fine-tuned model
    model = PeftModel.from_pretrained(base_model, SAVED_MODEL_PATH)

    # Move to device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("LoRA Fine-tuned Model Loaded Successfully!")

    return model, tokenizer 

def predict(model, tokenizer, text_list, device="cuda" if torch.cuda.is_available() else "cpu"):

    id2label = {0: "ACCOUNT", 1: "ATM", 2:"CARD", 3:"CONTACT", 4:"FEES", 5:"LOAN", 6:"TRANSFER"}
    label2id = {"ACCOUNT":0, "ATM":1, "CARD":2, "CONTACT":3, "FEES":4, "LOAN":5, "TRANSFER":6}

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    print("Trained Model Predictions:")
    print("--------------------------")
    
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Move input tensors to the correct device
        
        with torch.no_grad():  # Disable gradient computation for inference
            logits = model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=1).item()
        print(f"{text} - {id2label[predictions]}")


text_samples = [
    "What is the eligibility criteria for a home loan",
    "How to apply for a credit card",
    "What is the process of getting a loan approved.",
    "Can I get a loan",
    "I want to transfer money"
]
model, tokenizer = load_model()
predict(model, tokenizer, text_samples)


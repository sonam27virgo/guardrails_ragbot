import torch
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
    TrainingArguments, Trainer
)
from peft import get_peft_model, LoraConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import evaluate

# Load dataset
def load_and_preprocess_data(file_path):
    df = pd.read_parquet(file_path)
    df = df.rename(columns={'instruction': 'text'})
    
    # Sample 10% per category
    df = df.groupby("category").apply(lambda x: x.sample(frac=0.1, random_state=42)).reset_index(drop=True)
    
    # Filter relevant categories
    categories = ['CARD', 'LOAN', 'TRANSFER', 'FEES', 'ACCOUNT', 'CONTACT', 'ATM']
    df = df[df['category'].isin(categories)][['text', 'category']].reset_index(drop=True)
    
    return df

# Encode labels
def encode_labels(df):
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    return df, label_encoder

# Split dataset
def split_dataset(df):
    xtrain, xtest = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    return xtrain.reset_index(drop=True), xtest.reset_index(drop=True)

# Convert to Hugging Face Dataset
def convert_to_hf_dataset(xtrain, xtest):
    return DatasetDict({
        "train": Dataset.from_pandas(xtrain),
        "validation": Dataset.from_pandas(xtest)
    })

# Tokenization
def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, return_tensors="np"
    )

# Load model and tokenizer
def load_model_and_tokenizer(model_checkpoint, num_labels, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

# Compute accuracy
def compute_metrics(p):
    accuracy = evaluate.load("accuracy")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

# Fine-tune model using PEFT
def fine_tune_with_lora(model):
    peft_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=['q_lin'])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

# Training setup
def train_model(model, tokenizer, tokenized_dataset):
    training_args = TrainingArguments(
        output_dir="distilbert-lora-text-classification",
        learning_rate=1e-3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    return model

# Make predictions
def predict(model, tokenizer, id2label, text_list, device="mps"):
    model.to(device)
    print("Trained Model Predictions:")
    print("--------------------------")
    
    for text in text_list:
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        logits = model(inputs).logits
        predictions = torch.argmax(logits).item()
        print(f"{text} - {id2label[predictions]}")

# Save Model
def save_model(model, save_path):
    model.save_pretrained(save_path)


if __name__ == '__main__':
    FILE_PATH = "hf://datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset/bitext-retail-banking-llm-chatbot-training-dataset.parquet"
    MODEL_CHECKPOINT = 'distilbert-base-uncased'

    df = load_and_preprocess_data(FILE_PATH)
    df, label_encoder = encode_labels(df)
    xtrain, xtest = split_dataset(df)
    dataset = convert_to_hf_dataset(xtrain, xtest)
    dataset

    id2label = {0: "ACCOUNT", 1: "ATM", 2:"CARD", 3:"CONTACT", 4:"FEES", 5:"LOAN", 6:"TRANSFER"}
    label2id = {"ACCOUNT":0, "ATM":1, "CARD":2, "CONTACT":3, "FEES":4, "LOAN":5, "TRANSFER":6}

    model, tokenizer = load_model_and_tokenizer(MODEL_CHECKPOINT, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    model = fine_tune_with_lora(model)
    model = train_model(model, tokenizer, tokenized_dataset)

    model

    text_samples = [
        "What is the eligibility criteria for a home loan", 
        "How to apply for a credit card", 
        "What is the process of getting a loan approved.", 
        "Can I get a loan", 
        "I want to transfer money"
    ]

    predict(model, tokenizer, id2label, text_samples)
    save_model(model, "bank_fineuned_model.pt")
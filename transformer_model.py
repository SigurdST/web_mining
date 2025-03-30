from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


def prepare_transformer_data(df):
    df = df[['text', 'annotation_postPriority']].dropna().copy()
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['annotation_postPriority'])
    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    return train_dataset, test_dataset, label_encoder, test_df[['text']], test_df['label']


def train_transformer_model(train_dataset, test_dataset, label_encoder):
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=len(label_encoder.classes_)
    )

    args = TrainingArguments(
        output_dir="transformer_model",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        save_strategy="epoch",
        learning_rate=2e-5,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save model
    trainer.save_model("transformer_model")
    tokenizer.save_pretrained("transformer_model")

    return trainer

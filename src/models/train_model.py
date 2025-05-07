from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback)
from datasets import load_dataset
from pathlib import Path
import numpy as np
import evaluate


MODEL_NAME = "distilbert/distilbert-base-uncased"

dataset = load_dataset("AliAfkhamii/hf_emotion_generation_texts")

labels = dataset["train"].unique("label")
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in id2label.items()}


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

accuracy_metric = evaluate.load("accuracy")

def compute_accuracy(preds_labels: tuple[np.array, np.array]):

    preds, labels = preds_labels

    if len(preds.shape) >= 2:
        preds = np.argmax(preds, axis=1)

    return accuracy_metric.compute(predictions=preds, references=labels)


def labels_to_ids(sample):
    sample['label'] = label2id[sample['label']]
    return sample

def tokenize_text(sample):
    return tokenizer(sample["text"], truncation=True, padding=True)

dataset = dataset['train'].map(labels_to_ids).\
           train_test_split(test_size=0.2, shuffle=True).\
           map(tokenize_text, batched=True, batch_size=1000)



model_path = Path("../models")
model_path.mkdir(parents=True, exist_ok=True)

model_save_name = "hf_emotion_detector_text_classifier-distilbert-base-uncased"
model_save_dir = model_path / model_save_name


training_args = TrainingArguments(
    output_dir=str(model_save_dir),
    learning_rate=0.0005,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    use_cpu=False,
    load_best_model_at_end=True,
    weight_decay=0.1,
    logging_strategy="epoch",
    report_to="none",
    hub_private_repo=False,
)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=dataset["train"],
                  eval_dataset=dataset["test"],
                  processing_class=tokenizer,
                  compute_metrics=compute_accuracy,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                  )


trainer.train()

# you must log in first
trainer.push_to_hub(
    commit_message="push best model to hub",
)

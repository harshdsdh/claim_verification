import sys

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.append("/Users/harshitmishra/Documents/FEVEROUS")  # TODO

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
accuracy = evaluate.load("accuracy")
id2label = {0: "REFUTES", 1: "SUPPORTS", 2: "NOT ENOUGH INFO"}

label2id = {"REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2}


# references = https://huggingface.co/docs/transformers/tasks/sequence_classification
class Train:
    if __name__ == "__main__":

        def load_dataset(type_data):
            if type_data == "train":
                path = "src/augment_train_data.csv"
            elif type_data == "eval":
                path = "src/augment_dev_data.csv"
            return pd.read_csv(path)

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        def encode_labels(example):
            example["label"] = label2id[example["label"]]
            return example

        df = load_dataset("train")
        df_eval = load_dataset("eval")

        df["text"] = df["text"].fillna("")
        ds = Dataset.from_pandas(df[1:])

        df_eval["text"] = df_eval["text"].fillna("")
        ds_eval = Dataset.from_pandas(df_eval[1:])

        tokenized_data = ds.map(preprocess_function, batched=True)
        tokenized_data_eval = ds_eval.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        tokenized_data = tokenized_data.map(encode_labels)
        tokenized_data_eval = tokenized_data_eval.map(encode_labels)

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
        ).to(device)

        training_args = TrainingArguments(
            output_dir="model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data,
            eval_dataset=tokenized_data_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        predictions = trainer.predict(tokenized_data_eval)
        print(predictions.predictions.shape, predictions.label_ids.shape)
        preds = np.argmax(predictions.predictions, axis=1)
        acc = accuracy.compute(
            predictions=preds, references=predictions.label_ids
        )
        print(f"dev accuracy : {acc}")
        print("script complete")

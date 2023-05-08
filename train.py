import evaluate
import numpy as np
from pandas import DataFrame
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer

from src.utils import visible_print, get_config_yaml
from src.data import tokenize

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    disable_caching()

    # Load config
    config = get_config_yaml()

    visible_print("Dataset")
    dataset = load_dataset("csv", data_files=config["data"]["train"], split="train")
    dataset = dataset.remove_columns(["id", "keyword", "location"])
    dataset = dataset.rename_column("target", "label")
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=config["data"]["valid_size"], shuffle=True, stratify_by_column="label")
    print(dataset)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_name"])
    dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # id2label and label2id mapping
    id2label = {0: "NOT DISASTER", 1: "DISASTER"}
    label2id = {"NOT DISASTER": 0, "DISASTER": 1}

    visible_print("Model")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"], 
        num_labels=config["model"]["num_labels"], 
        id2label=id2label, 
        label2id=label2id
    )
    print(model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        optim="adamw_torch",
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    visible_print("Training")
    trainer.train()

    visible_print("Save model")
    trainer.save_model("./output/model")
    print("Model saved to ./output/model")

    visible_print("Run inference")
    # Load dataset
    test_dataset = load_dataset("csv", data_files=config["data"]["test"], split="train")
    ids = test_dataset["id"]
    test_dataset = test_dataset.remove_columns(["id", "keyword", "location"])
    test_dataset = test_dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # Run inference and save predictions
    predictions = trainer.predict(test_dataset).predictions.argmax(axis=1)

    # Save predictions to csv
    submission = DataFrame({"id": ids, "target": predictions})
    submission.to_csv("./output/submission.csv", index=False)

    visible_print("Done")
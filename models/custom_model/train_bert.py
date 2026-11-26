#!/usr/bin/env python3
"""Retrain Daniel's BERT on the frozen train split, with the settings from models/daniel_bert/original_notebook.ipynb.

Run from the repo root:  python -m models.custom_model.train_bert [checkpoint_dir]
Same as the notebook: bert-base-uncased, 512-token truncation, batch 8, up to 10 epochs,
eval/save each epoch, keep the best validation ROC-AUC, early stopping patience 2.
Changes: frozen train/validation instead of his own split; bf16 (speed); dynamic padding (same logits as max_length
padding); fast tokenizer (same WordPiece tokens); lr 2e-5 + 6% warmup + weight decay 0.01, because the notebook's
default lr 5e-5 collapsed to predicting only the majority class (val AUC 0.49) on this split.
No checkpoint rotation: deleting old checkpoints failed on Hopper's NFS scratch (.nfs busy files).
Final model -> models/custom_model/artifacts/bert/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from models.yash_5th_place.train import load_splits

BASE = "bert-base-uncased"
OUT = Path(__file__).resolve().parent / "artifacts" / "bert"


def main() -> int:
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "bert_checkpoints"
    splits, manifest = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(BASE)

    def to_dataset(frame):
        ds = Dataset.from_dict({"text": frame.text.fillna("").astype(str).tolist(),
                                "labels": frame.labels.astype(int).tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=512),
                      batched=True, remove_columns=["text"])

    train, val = to_dataset(splits["train"]), to_dataset(splits["validation"])
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)

    def compute_metrics(pred):
        return {"auc": roc_auc_score(pred.label_ids, softmax(pred.predictions, axis=1)[:, 1])}

    args = TrainingArguments(output_dir=ckpt_dir, num_train_epochs=10, eval_strategy="epoch", save_strategy="epoch",
                             load_best_model_at_end=True, metric_for_best_model="auc", greater_is_better=True,
                             learning_rate=2e-5, warmup_ratio=0.06, weight_decay=0.01,  # 5e-5 collapsed to majority class
                             bf16=True, dataloader_num_workers=4, seed=747,
                             logging_steps=500, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train(resume_from_checkpoint=any(Path(ckpt_dir).glob("checkpoint-*")))
    metrics = trainer.evaluate()

    OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    (OUT / "configuration.json").write_text(json.dumps({
        "model_id": "daniel_bert", "variant": "retrained on frozen train for models/custom_model",
        "base_model": BASE, "max_length": 512, "positive_class_id": 1,
        "dataset_hash": manifest["dataset_hash"], "split_fingerprints": manifest["split_fingerprints"],
        "validation_use": "epoch selection and early stopping only",
        "best_checkpoint": trainer.state.best_model_checkpoint, "val_auc": metrics["eval_auc"],
        "training_args": {"lr": args.learning_rate, "warmup_ratio": args.warmup_ratio,
                          "weight_decay": args.weight_decay, "batch": args.per_device_train_batch_size,
                          "max_epochs": 10, "patience": 2, "bf16": True, "seed": 747},
    }, indent=2), encoding="utf-8")
    print("RESULTS " + json.dumps({"val_auc": metrics["eval_auc"], "best": trainer.state.best_model_checkpoint}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

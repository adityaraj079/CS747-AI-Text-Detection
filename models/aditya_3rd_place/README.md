# Aditya Raj — Submission #3 adaptation

This directory packages a runnable, low-memory adaptation of the Submission #3 workflow for the shared CS747 model-integration contract. It is not a verified reproduction of the original competition submission: its original hyperparameters and ensemble weights were not available.

## Source material

- `original_notebook.ipynb`: the project notebook used for the original submission replication.
- `cs747_submission3.py`: the helper module that exposes the picklable BPE tokenizer wrapper.
- `artifacts/`: fitted tokenizer, TF-IDF vectorizer, SGD classifier, CatBoost classifier, and training configuration.

## Model status

The low-memory adaptation uses BPE tokenization, word-level TF-IDF with 3- and 4-grams, an SGD classifier, and a CatBoost classifier. The ensemble weight is selected by ROC-AUC on validation data. The saved configuration records the settings and source split fingerprints.

The fit uses only `data/processed/train.csv`. `validation.csv` is used only to select the ensemble weight; standard test and unseen test are not used in fitting or selection. The result is a project-specific adaptation, not the original trained submission weights.

## Packaging contract

The shared project expects a `ModelAdapter` implementation with a `predict_proba(texts)` method returning a 1D NumPy float array shaped like `(n_samples,)` and oriented with `1 = AI-generated`.

## Usage

Recreate the artifacts locally with:

```bash
python scripts/train_aditya_adaptation.py
```

Then evaluate against the frozen partitions with `scripts/evaluate_registered_model.py`. The processed CSVs remain local and ignored by Git; the fitted model artifacts are packaged for inference.

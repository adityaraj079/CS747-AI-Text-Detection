# Aditya Raj — 3rd-place efficiency solution

This directory packages the Submission #3 replication from the project notebook and helper module, adapted to the shared CS747 model-integration contract.

## Source material

- `original_notebook.ipynb`: the project notebook used for the original submission replication.
- `cs747_submission3.py`: the helper module that exposes the picklable BPE tokenizer wrapper.
- `artifacts/bpe_tokenizer.json`: the tokenizer artifact saved by the notebook.

## Model status

This repository does not include a final fitted TF-IDF vectorizer or trained classifier artifact for the full Submission #3 model. The available project artifact is the tokenizer file only. The adapter is therefore structured to load the trained vectorizer and classifier when those artifacts are placed in `artifacts/`, without modifying the original notebook or retraining the model.

## Packaging contract

The shared project expects a `ModelAdapter` implementation with a `predict_proba(texts)` method returning a 1D NumPy float array shaped like `(n_samples,)` and oriented with `1 = AI-generated`.

## Usage

1. Place the fitted vectorizer and classifier artifacts in `artifacts/`.
2. Keep the helper module and tokenizer artifact in the same directory.
3. Use `adapter.py` with the shared evaluator.

The package intentionally does not fabricate model outputs or retrain any model.

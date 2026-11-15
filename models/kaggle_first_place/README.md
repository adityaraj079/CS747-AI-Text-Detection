# Kaggle main-leaderboard first-place solution

This package is reserved for the winner of the competition's **main private leaderboard**. It is not the first-place Efficiency Prize model.

Published reference: [Comprehensive 1st Place Write-Up](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/comprehensive-1st-place-write-up)

## Current status

The original notebook, verified architecture details, fitted model artifacts, and an implementation compatible with the project's adapter contract have not been supplied. No leaderboard ROC-AUC or local result is asserted here. The app marks this model unavailable until its inference pipeline and artifacts are integrated.

The adapter raises `ModelNotAvailableError`; it does not return placeholder probabilities. No training or artifact loading occurs during app startup.

## Where to add the verified solution

- `original_notebook.ipynb`: original source notebook, if released and permitted to redistribute.
- `adapter.py`: inference-only wrapper implementing `predict_proba(texts)` and returning `P(labels == 1)`.
- `artifacts/`: fitted tokenizer, feature extractors, model weights, and ensemble configuration required for inference.
- `requirements.txt`: model-specific, pinned dependencies; do not add heavy model libraries to the project-wide requirements unless necessary.
- `config.json`: verified source metadata, architecture, provenance, artifact paths, CPU compatibility, and integration status.

Only load trusted, repository-owned model artifacts. Do not upload student essays, restricted data, or model weights to the public repository without permission. Do not retrain the solution as part of packaging unless explicitly authorized.

## Registering the model

After implementing and testing the adapter, add it explicitly to `app_components/model_registry.py`. The dashboard does not discover or load arbitrary files. Run the package smoke tests and standardized evaluation before marking the model available.

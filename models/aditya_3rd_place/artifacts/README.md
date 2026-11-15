# Model artifacts

This folder is reserved for fitted model assets required for inference.

## Current status

The project archive currently includes only the BPE tokenizer artifact:

- `bpe_tokenizer.json`

## Expected additional artifacts

Before running real `predict_proba` inference, add the trained fitted objects used by the submission notebook, such as:

- TF-IDF vectorizer (`*.joblib` or `*.pkl`)
- trained classifier(s) (`*.joblib` or `*.pkl`)

These files remain local to this folder and are not automatically downloaded or recreated here.

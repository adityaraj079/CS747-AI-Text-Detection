This directory contains only the final inference artifacts referenced by
`configuration.json`:

- `tokenizer.json`: the BPE tokenizer.
- `model.joblib`: the TF-IDF vocabulary and IDF, the MultinomialNB log
  probabilities and priors, the SGD coefficients and intercept, and the
  ensemble weights, all stored as plain numpy arrays and a dict.

`../export_artifacts.py` exported these files from the trained `run_7th.py`
pipeline. No raw data, predictions, training matrices, or checkpoints are
included.

`model.joblib` is a trusted object generated for this repository. Don't
replace it or load a replacement from an untrusted source.

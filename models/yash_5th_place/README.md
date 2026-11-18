# Yash fifth-place efficiency model

This package contains a reproducible adaptation of the lightweight BPE/TF-IDF
ensemble associated with Yash Koli's reported fifth-place efficiency result.
The original notebook and fitted competition artifacts were not in the supplied
repository, so this package does not claim to reproduce the original submission.

## Training and evaluation

Run `python models/yash_5th_place/train.py` from the repository root after
generating the ignored `data/processed/*.csv` files with the repository's frozen
split process. The script verifies all four split fingerprints before fitting.
It learns the tokenizer, n-gram vocabulary, TF-IDF weights, and classifiers from
the frozen training set only. Validation is used for SGD epoch selection and
LightGBM early stopping. The standard and unseen-source test sets are reserved
for the shared evaluator and are not used for fitting or selection.

The architecture has byte-level BPE, 3-to-5-token TF-IDF, three ComplementNB
models, three modified-Huber SGD models, and one LightGBM model. Fixed ensemble
weights are `[0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.40]`. The exact original
settings, notebook provenance, and original leaderboard metrics were not
available and are not represented as verified facts here.

The dataset ZIP lists Kaggle and competition sources with different usage terms.
The model owner stated that written rights cover all sources and the derived
model artifacts before these weights were prepared for public repository use.
The dataset itself and processed partitions are not included in this package.

Inference uses `adapter.py`, which implements the shared `ModelAdapter` API and
returns `P(label == 1)` (AI-generated) on CPU. Generated assets are stored in
`artifacts/`: `configuration.json`, `tokenizer.json`, `tfidf_vectorizer.joblib`,
`classifiers.joblib`, and `lightgbm.txt`. The sklearn joblib files are trusted,
repository-generated artifacts; do not load replacements from untrusted sources.
Install this model's `requirements.txt` before inference: the adapter checks for
the exact scikit-learn and LightGBM versions recorded in `configuration.json`
and reports the model unavailable if either differs.

The locally supplied CSV reserializes to a different dataframe hash under the
available pandas version. Its four record counts and split fingerprints match
the frozen repository manifest. The manifest hash, local dataframe hash, and
this limitation are recorded in `artifacts/configuration.json`.

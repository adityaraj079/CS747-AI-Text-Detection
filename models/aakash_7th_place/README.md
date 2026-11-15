# Aakash 7th-place efficiency model

This package holds Aakash Patil's trained reproduction of the 7th-place Efficiency
Prize solution (author davin11 / Kaggle user wcqyfly, MIT licence) for the
LLM - Detect AI Generated Text competition.

## Architecture

The pipeline has four steps:

1. A byte-level BPE tokenizer with a vocabulary of 50,275.
2. TF-IDF over 3-5 token n-grams, with sublinear tf. The vocabulary has
   1,537,124 n-grams.
3. Two classifiers: `MultinomialNB(alpha=0.0225)` and
   `SGDClassifier(loss="modified_huber", tol=1e-4, random_state=6743)`.
4. A soft vote between them, with weights `[0.43, 0.57]`.

The model definition (`original_solution/model_ensemble.py`) and the tokenizer
and TF-IDF settings are the authors' own, unchanged.

## Leakage warning: this model was not trained on the frozen split

The model was trained before the repository froze its partitions. It uses its
own stratified 80/10/10 split of `CS747data.csv` (seed 747, 123,661 training
essays). That split draws about 80% of every source into training, including
the sources the repository holds out for `unseen_test` (Mistral7B-CME-*,
persuade_2.0). As a result, most repository `test` and `unseen_test` essays were
seen in training.

Shared evaluator scores for this model are optimistic. Don't compare them with
models trained on `data/processed/train.csv`. Retraining on the frozen split
would remove this caveat.

## What was adapted from the original solution

The adapted training script is `original_solution/run_7th.py`, and its run
notebook is `original_solution/reproduce.ipynb`. Compared with the original
solution:

- **Data.** It trains on `CS747data.csv` instead of the competition's
  DAIGT/argugpt/commonlit sources. Essays shorter than 50 characters, duplicates,
  and conflicting labels were removed.
- **Tokenizer and vocabulary.** The BPE tokenizer and TF-IDF vocabulary were fit
  on the own-split validation essays. The original fit them on the competition
  test essays. IDF was fit on the own-split training essays.
- **SGD training.** SGD was trained one epoch at a time with `fit()`'s stopping
  rule, and stopped after 6 epochs.
- **Threshold.** A decision threshold of 0.32 was selected on validation. It is
  recorded in the configuration but not used by the adapter.

On the model's own 80/10/10 test split, the results were:

- ROC-AUC: 0.99962
- Accuracy at 0.5: 0.9863
- Accuracy at 0.32: 0.9948

The full record is in `artifacts/configuration.json`.

## Artifacts and inference

`export_artifacts.py` converted `run_7th.py`'s `pipeline.joblib` into
version-independent artifacts. It stores plain numpy arrays and a dict, with no
sklearn objects. It drops `MultinomialNB.feature_count_`, which inference does
not use.

The export also asserts that the adapter matches the original
scikit-learn 1.2.2 pipeline. The maximum absolute difference on 500 essays was
1.2e-14.

The artifacts are:

- `artifacts/tokenizer.json`: the BPE tokenizer.
- `artifacts/model.joblib`: the TF-IDF vocabulary and IDF, the NB log
  probabilities and priors, the SGD coefficients and intercept, and the ensemble
  weights.
- `artifacts/configuration.json`: provenance, settings, and metrics.

`adapter.py` (`AakashSeventhPlaceAdapter`) implements the shared `ModelAdapter`
API. It returns `P(label == 1)` (AI-generated) and runs on CPU:

```bash
python scripts/evaluate_registered_model.py --model-id aakash_7th_place --dataset-type standard_test
```

## Licence

The original code is MIT-licensed (`original_solution/LICENSE`). The model
owner states that written rights cover all source datasets and the derived
model artifacts. Neither the dataset nor the processed partitions are included.

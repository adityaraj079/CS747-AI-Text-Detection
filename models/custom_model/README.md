# Custom model: a stacked ensemble of the four team models

The custom model combines the four team detectors. It averages their log-odds with equal weights and applies a
threshold selected on validation:

```
P(AI) = sigmoid( mean_i logit(p_i) - threshold_logit )
```

| Member | Source | Training |
|---|---|---|
| `aakash_7th_place` | `artifacts/seventh/` | Retrained on the frozen `train.csv` (`train_seventh.py`) |
| `aditya_3rd_place` | `models/aditya_3rd_place/` | Used unchanged; already fit on the frozen train split |
| `yash_5th_place` | `models/yash_5th_place/` | Used unchanged; already fit on the frozen train split |
| `daniel_bert` | Hugging Face, revision pinned in `config.json` | Retrained on the frozen `train.csv` (`train_bert.py`) |

Aakash's 7th-place model and Daniel's BERT were retrained because their original packages used their own
80/10/10 splits. Those splits overlap the repository's test and unseen_test sets.

## Data contract

- All fitting used only `data/processed/train.csv`. The split is checked against `data/manifests/split_manifest.json`
  (dataset hash `e6f09bce…`, train fingerprint `d061b2ab…`).
- Validation was used only to pick BERT's best epoch, to choose the ensemble rule, and to set the threshold.
- Test and unseen_test were scored once, after every choice was fixed.

## How the combiner was chosen (`build_ensemble.py`)

Six candidates were compared: each member alone, an equal mean of logits, and logistic regression on logits.
Each was scored by ROC-AUC on validation with `GroupKFold(5)` over `source`, so every source is scored by a combiner
that never saw it. The equal mean (0.99980) won, and the threshold maximises held-out balanced accuracy.

## Results (one-time, from `artifacts/combiner.json`)

| ROC-AUC | Validation | Test | Unseen test |
|---|---|---|---|
| 7th place (retrained) | 0.9995 | 0.9996 | **0.9997** |
| Aditya 3rd place | 0.9994 | 0.9995 | 0.9993 |
| Yash 5th place | 0.9994 | 0.9996 | 0.9994 |
| BERT (retrained) | 0.9998 | 0.9997 | 0.9692 |
| **Ensemble** | 0.9998 | **0.9999** | 0.9953 |

The ensemble's balanced accuracy at the validation threshold is 0.998 on validation, 0.998 on test, and 0.960 on
unseen test.

### Shared evaluator (`scripts/evaluate_registered_model.py`, run once)

| | ROC-AUC | Accuracy | Precision | Recall | F1 | Human essays flagged as AI |
|---|---|---|---|---|---|---|
| standard_test (18,478) | 0.99991 | 0.9976 | 0.9996 | 0.9976 | 0.9986 | 6 of 2,544 |
| unseen_test (31,400) | 0.99534 | 0.9561 | 0.8141 | 0.9654 | 0.8833 | 1,192 of 25,994 |

The records are in `results/metrics/standardized/custom_model_*_20261126T*.json`, and their ROC-AUCs match
`combiner.json`. The latency in these records (about 17 ms per essay) was measured on Hopper with an A100 3g.40gb
GPU slice available, so it is **not** comparable with the CPU-only latency of the other models.

**Known limitation.** On unseen sources the ensemble scores below three of its members, because the retrained
BERT generalizes much worse. Validation could not detect this: every candidate scored about 0.9995 there. The rule
was not re-selected after seeing unseen_test, because the team protocol forbids tuning on test data.

## Adaptations (verified versus changed)

- **7th place:** the model settings are unchanged (MNB alpha 0.0225, modified-Huber SGD, weights 0.43/0.57). The
  tokenizer, vocabulary, and IDF are fit on train instead of target text, and the vocabulary keeps the top 500,000
  n-grams. The adapter reproduces sklearn's output (max diff 1.2e-14).
- **BERT:** the run uses bert-base-uncased, 512 tokens, batch 8, up to 10 epochs, the best validation AUC, and
  patience 2, as in Daniel's notebook. It uses learning rate 2e-5 with 6% warmup and weight decay 0.01, because the
  notebook's default 5e-5 collapsed to always predicting "AI" (val AUC 0.49). It also uses bf16, dynamic padding,
  and the fast tokenizer. The best epoch was 1.
- **No original notebook:** this is a new team model, so the training scripts stand in for it.

## Files

- `adapter.py` contains `CustomEnsembleAdapter`. It loads only the four trusted members and skips any member with
  zero weight. BERT loads from `artifacts/bert/` when the weights are present; otherwise it downloads them from the
  pinned Hugging Face revision.
- `train_seventh.py`, `train_bert.py`, and `build_ensemble.py` reproduce the training. The `*.sbatch` files are the
  Slurm jobs used on GMU Hopper (partition `gpuq` or `normal`), and `evaluate.sbatch` ran the shared evaluator.
- `tests/test_adapter.py` covers empty input, missing artifacts, and output shape, range, and orientation. It uses
  fakes and needs no large files.

## Usage

```bash
python scripts/evaluate_registered_model.py --model-id custom_model --dataset-type standard_test
python scripts/evaluate_registered_model.py --model-id custom_model --dataset-type unseen_test
```

Use a GPU if one is available; the BERT member is slow on CPU.

## Licensing

bert-base-uncased is Apache-2.0. The retrained BERT member is published under Apache-2.0 at
[JaCkKnite12/cs747-custom-bert](https://huggingface.co/JaCkKnite12/cs747-custom-bert) and pinned to revision
`2710a1b04978f4bd6a24b980f749759f4e56c822`. The model owner confirmed the right to publish weights trained on the
team dataset. A download of the pinned revision produces the same predictions as the local weights (max diff 0.0
on 20 essays).

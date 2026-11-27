# Kaggle submission requirements: LLM - Detect AI Generated Text

Competition: https://www.kaggle.com/competitions/llm-detect-ai-generated-text

This file lists what we need in order to get an official Kaggle score for a team model, such as `custom_model`.

## Facts checked with the Kaggle API

- **Status:** the competition is closed. The deadline was 2024-01-22 23:59 UTC, so any submission now is a **late
  submission**.
- **Files:** `train_essays.csv`, `train_prompts.csv`, `test_essays.csv`, and `sample_submission.csv`.
- **Placeholder test set:** the public `test_essays.csv` is only 90 bytes, a few placeholder rows. The real test
  essays are hidden.
- **Rules first:** each team member must accept the competition rules on the website ("Join" / "Late Submission")
  before the data can be downloaded or a notebook submitted. Until then, the API returns `403 Forbidden`.

## Competition format

The rules pages could not be read automatically, so this section comes from the competition's known format and the
original 7th-place code (`train_predict.py`). Check it against the
[Overview](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview) and
[Rules](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/rules) pages before relying on it.

| Requirement | Detail |
|---|---|
| Submission type | **Code competition.** You submit a Kaggle notebook, not a CSV. Kaggle re-runs it on the hidden test set. |
| Hidden test set | About 9,000 essays. They replace the placeholder `test_essays.csv` at scoring time. |
| Input path | `/kaggle/input/llm-detect-ai-generated-text/test_essays.csv` (columns `id`, `prompt_id`, `text`) |
| Output | The notebook must write `submission.csv` with columns `id,generated` |
| `generated` | P(AI-generated) for each essay, a float in [0, 1] |
| Metric | ROC-AUC |
| Internet | **Off** during the scoring run |
| Runtime limit | 9 hours (CPU or GPU notebook) |
| Late submission score | A public and a private score are shown, but no leaderboard rank |

Example `submission.csv`:

```csv
id,generated
0000aaaa,0.12
1111bbbb,0.97
```

## What the team must do to get scores

1. **Accept the rules** on the competition page, logged in to Kaggle.
2. **Upload a private Kaggle Dataset.** The notebook has no internet, so everything must be bundled in it:
   - the repository's `models/` and `evaluation/` code;
   - the artifacts of each model being scored (for `custom_model`: `artifacts/seventh/`, `artifacts/combiner.json`,
     and the Aditya and Yash artifacts);
   - the BERT weights (`models/custom_model/artifacts/bert/`, about 420 MB). The notebook cannot download them from
     Hugging Face;
   - pip wheels for any version pins: `scikit-learn==1.6.1` and `lightgbm==4.6.0`, which Yash's adapter requires.
3. **Create a notebook** that:
   - attaches the competition data and our dataset;
   - installs the wheels offline (`pip install --no-index --find-links ...`);
   - loads the model adapter from local paths, with `HF_HUB_OFFLINE=1`;
   - scores `test_essays.csv` and writes `submission.csv` (`id`, `generated`).
4. **Test it:** run it once on the placeholder test file to check that it finishes and writes the right columns.
5. **Submit:** Notebook → Submit to Competition. Use a GPU accelerator so the BERT branch finishes in time.
6. **Record** the public and private ROC-AUC for the final report.

## Notes for the report

- Our models were trained on the team dataset (`CS747data.csv`), not on Kaggle's `train_essays.csv`. The Kaggle score
  therefore measures transfer to the competition's hidden test set.
- The scores are late submissions, so they are not comparable to leaderboard ranks from the competition period.

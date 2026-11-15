# CS747 — AI Generated Text Detection

This repository contains the shared evaluation, dataset preparation, and integration scaffolding for the CS747 project at George Mason University.

## Dataset snapshot

- Original records: 165086
- Accepted records after cleaning and deduplication: 154582
- Removed due to empty text, normalized duplicates, and conflicting labels: 10504
- Label distribution in accepted data: {0: 42955, 1: 111627}
- Held-out unseen evaluation sources: Mistral7B-CME-v1, Mistral7B-CME-v2, Mistral7B-CME-v3, Mistral7B-CME-v4, Mistral7B-CME-v5, Mistral7B-CME-v6, Mistral7B-CME-v7, Mistral7B-CME-v7-15-percent-corruption, persuade_2.0_human_scores_demo_id_github

## Repository structure

- `data/`: validated dataset, manifests, figures, and documentation
- `evaluation/`: reusable dataset, metric, and model-contract utilities
- `models/`: team-specific model package stubs and packaging prompt
- `results/`: evaluation outputs and benchmark artifacts
- `scripts/`: dataset preparation and verification utilities
- `app.py`, `app_pages/`, `app_components/`: two-page Streamlit research dashboard

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run or deploy the Streamlit app

Run the dashboard locally:

```bash
streamlit run app.py
```

To publish it with Streamlit Community Cloud:

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from that repository and select the `main` branch and `app.py`.
3. Deploy. The app installs dependencies from the root `requirements.txt`.

The dashboard has exactly two views: **Kaggle Champion vs. Our Custom Model** and **Team Model Comparison**. It reads saved result files and loads registered inference adapters only when requested. Aditya's low-memory Submission #3 adaptation is packaged for inference; its weights are not a verified reproduction of the original competition submission. The Kaggle first-place and other team adapters remain unavailable until their verified inference packages and fitted artifacts are integrated. The app does not load or expose essay CSVs on Streamlit Community Cloud.

Dataset CSVs are excluded by `.gitignore`; keep them out of the public GitHub repository unless redistribution is approved.

## Dataset preparation

Use the original CSV supplied locally and run:

```bash
python scripts/prepare_dataset.py --source-csv "/absolute/path/to/CS747data.csv" --rebuild
python scripts/generate_data_report.py
python scripts/verify_splits.py
```

The default source path is discovered from the sibling `Progress/data/CS747data.csv` location if present.

## Current split summary

- Train: 86227
- Validation: 18477
- Test: 18478
- Unseen test: 31400

## Model integration

Each member should package a model using `models/model-prompt.md` and the adapter contract in `evaluation/model_adapters.py`. The main leaderboard first-place scaffold lives in `models/kaggle_first_place/`; it is distinct from the Efficiency Prize solutions.

### Results and standardized evaluation

The dashboard reads small JSON or CSV result records from `results/metrics/standardized/` and `results/metrics/historical/`. Set `CS747_RESULTS_DIR` to import another trusted result directory. Standardized comparisons require matching dataset hash, split ID, and dataset type; CPU comparisons additionally require matching CPU and thread settings. Historical results are visibly labeled and never silently treated as a controlled comparison.

Aditya's adaptation was trained only from the frozen standard train partition. Team members preparing other models should follow `models/model-prompt.md`, which requires the same frozen train/validation/test protocol and explains how to package final artifacts.

After an adapter and fitted artifacts are available, run a standardized evaluation locally:

```bash
python scripts/evaluate_registered_model.py --model-id aditya_3rd_place --dataset-type standard_test
python scripts/evaluate_registered_model.py --model-id aditya_3rd_place --dataset-type unseen_test
```

The script uses the frozen local CSVs, shared evaluator, and CPU benchmark, then writes a timestamped result under `results/metrics/standardized/`. It does not train a model. Processed essay CSVs remain local and ignored by Git; the fitted adaptation artifacts are committed for inference.

## Notes

- The dashboard does not train models at startup; team training and packaging follow `models/model-prompt.md`.
- The dataset is validated before splitting.
- The unseen evaluation set is derived from defensible source-family holdout groups, not arbitrary labels.

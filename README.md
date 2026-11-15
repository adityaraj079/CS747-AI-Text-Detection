# CS747 — AI Generated Text Detection

This repository contains the shared evaluation, dataset preparation, and integration scaffolding for the CS747 project at George Mason University.

## Dataset snapshot

- Original records: 165086
- Accepted records after cleaning and deduplication: 154582
- Removed due to empty text, normalized duplicates, and conflicting labels: 10504
- Label distribution in accepted data: {0: 43312, 1: 121774}
- Held-out unseen evaluation sources: Mistral7B-CME-v1, Mistral7B-CME-v2, Mistral7B-CME-v3, Mistral7B-CME-v4, Mistral7B-CME-v5, Mistral7B-CME-v6, Mistral7B-CME-v7, Mistral7B-CME-v7-15-percent-corruption, persuade_2.0_human_scores_demo_id_github

## Repository structure

- `data/`: validated dataset, manifests, figures, and documentation
- `evaluation/`: reusable dataset, metric, and model-contract utilities
- `models/`: team-specific model package stubs and packaging prompt
- `results/`: evaluation outputs and benchmark artifacts
- `scripts/`: dataset preparation and verification utilities
- `app.py`: placeholder Streamlit application

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run or deploy the Streamlit app

Run the current placeholder locally:

```bash
streamlit run app.py
```

To publish it with Streamlit Community Cloud:

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from that repository and select the `main` branch and `app.py`.
3. Deploy. The app installs dependencies from the root `requirements.txt`.

The current app shows the project overview only; model integrations, prediction, and benchmark results are not available yet. Dataset CSVs are excluded by `.gitignore`; keep them out of the public GitHub repository unless redistribution is approved.

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

Each team member should package a model using the guidance in `models/model-prompt.md` and the expected adapter contract in `evaluation/model_adapters.py`.

## Notes

- No model training is performed in this phase.
- The dataset is validated before splitting.
- The unseen evaluation set is derived from defensible source-family holdout groups, not arbitrary labels.

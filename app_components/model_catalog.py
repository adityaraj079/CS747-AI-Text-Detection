from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIRST_PLACE_WRITEUP = (
    "https://www.kaggle.com/competitions/llm-detect-ai-generated-text/"
    "writeups/comprehensive-1st-place-write-up"
)
COMPETITION_URL = "https://www.kaggle.com/competitions/llm-detect-ai-generated-text/"


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "kaggle_first_place": {
        "name": "Kaggle main leaderboard champion",
        "short_name": "Kaggle #1",
        "group": "champion",
        "member": "Published competition solution",
        "architecture": "Not yet verified from the published write-up.",
        "version": "Not packaged",
        "reference": FIRST_PLACE_WRITEUP,
        "reference_label": "Official first-place write-up",
        "availability": "Not integrated",
        "status_detail": "The original fitted artifacts and compatible adapter have not been supplied.",
    },
    "custom_model": {
        "name": "Our custom model",
        "short_name": "Custom",
        "group": "champion",
        "member": "CS747 team",
        "architecture": "Equal-weight average of the log-odds of four team models (7th-place BPE TF-IDF NB+SGD, "
                        "Aditya 3rd-place, Yash 5th-place, BERT), all trained on the frozen train split.",
        "version": "Stacked ensemble v1 (combiner selected on validation with source-held-out folds)",
        "reference": None,
        "reference_label": None,
        "availability": "Packaged",
        "status_detail": "Validation-selected equal-weight ensemble. On unseen_test it scores below its three TF-IDF "
                         "members because the retrained BERT generalizes worse; reported as-is, not re-tuned.",
    },
    "yash_5th_place": {
        "name": "Yash Koli — 5th-place Efficiency Prize",
        "short_name": "Yash",
        "group": "team",
        "member": "Yash Koli",
        "architecture": "BPE + TF-IDF with ComplementNB, SGD, and LightGBM ensemble.",
        "version": "Adapted reproduction; original settings unverified",
        "reference": COMPETITION_URL,
        "reference_label": "Kaggle competition",
        "availability": "Adapted package",
        "status_detail": "Adapted CPU inference package; original submission settings and leaderboard metrics are unverified.",
    },
    "aakash_7th_place": {
        "name": "Aakash Patil — 7th-place Efficiency Prize",
        "short_name": "Aakash",
        "group": "team",
        "member": "Aakash Patil",
        "architecture": "BPE + TF-IDF with MultinomialNB and SGD ensemble.",
        "version": "Adapted reproduction; frozen split not used",
        "reference": COMPETITION_URL,
        "reference_label": "Kaggle competition",
        "availability": "Adapted package",
        "status_detail": (
            "CPU inference adapter and fitted artifacts are available. The model used a separate "
            "training split, so repository test scores are optimistic and not directly comparable."
        ),
    },
    "aditya_3rd_place": {
        "name": "Aditya Raj — Submission #3 adaptation",
        "short_name": "Aditya",
        "group": "team",
        "member": "Aditya Raj",
        "architecture": "BPE + TF-IDF with SGD and CatBoost; memory-optimized adaptation.",
        "version": "Trained low-memory adaptation; original settings unverified",
        "reference": COMPETITION_URL,
        "reference_label": "Kaggle competition",
        "availability": "Adaptation artifacts available",
        "status_detail": "Fitted on the frozen standard training split; not a verified reproduction of the original submission.",
    },
    "daniel_bert": {
        "name": "Daniel Stiller — BERT-base classifier",
        "short_name": "Daniel BERT",
        "group": "team",
        "member": "Daniel Stiller",
        "architecture": "BERT-base sequence classifier with WordPiece tokenization.",
        "version": "Original 80/10/10 model; split differs from frozen benchmark",
        "reference": "https://huggingface.co/grumpydan99/BERT",
        "reference_label": "Hugging Face model",
        "availability": "Hugging Face checkpoint integrated",
        "status_detail": (
            "Loads the original Hugging Face checkpoint. Its separate 80/10/10 split may overlap "
            "repository holdouts, so evaluations are not leakage-free or directly comparable."
        ),
    },
}


def load_model_catalog(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    catalog = {model_id: dict(spec) for model_id, spec in MODEL_SPECS.items()}
    config_path = Path(repo_root) / "models" / "kaggle_first_place" / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        champion = catalog["kaggle_first_place"]
        champion["competition_track"] = config.get("competition_track")
        champion["solution_name"] = config.get("team_or_solution_name")
        champion["architecture"] = config.get("architecture_summary") or champion["architecture"]
        champion["original_leaderboard_roc_auc"] = config.get("original_private_leaderboard_roc_auc")
        champion["availability"] = config.get("reproduction_status", champion["availability"])
        champion["status_detail"] = config.get("status_detail", champion["status_detail"])
    return catalog

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_components.data_loader import compare_results_compatibility, latest_result, partition_metadata
from app_components.inference import show_live_predictions

TEAM_MODEL_IDS = ("aakash_7th_place", "aditya_3rd_place", "daniel_bert", "yash_5th_place")
PROJECT_SPLIT_MODELS = ("aditya_3rd_place", "yash_5th_place")
TEST_SETS = {
    "External essays": "ghostbuster_essay",
    "Standard test": "standard_test",
    "Unseen sources": "unseen_test",
}


def _table(records, specs, original=False):
    rows = []
    for record in records:
        row = {"Detector": specs[record["model_id"]]["short_name"]}
        if original:
            row["Test essays"] = record["record_count"]
        row.update({
            "ROC-AUC": record.get("roc_auc"),
            "Accuracy": record.get("accuracy"),
            "F1": record.get("f1"),
        })
        if not original and record.get("confusion_matrix"):
            matrix = record["confusion_matrix"]
            row.update({"Humans flagged": matrix[0][1], "AI essays missed": matrix[1][0]})
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows).set_index("Detector")
        st.table(df.style.format({"ROC-AUC": "{:.4f}", "Accuracy": "{:.2%}",
                                  "F1": "{:.4f}", "Test essays": "{:,}"}, na_rep="—"))


def render_page(repo_root, results, model_specs, split_manifest, unseen_manifest, adapter_loader):
    st.title("AI essay detection")
    st.subheader("Performance comparison")
    label = st.radio("Test set", list(TEST_SETS), horizontal=True, label_visibility="collapsed")
    dataset = TEST_SETS[label]
    external = dataset == "ghostbuster_essay"
    ids = TEAM_MODEL_IDS if external else PROJECT_SPLIT_MODELS
    records = [latest_result(results, mid, dataset, "standardized") for mid in ids]
    if not external:
        metadata = partition_metadata(split_manifest, dataset) or {}
        records = [r for r in records if r and r["split_id"] == metadata.get("split_id")
                   and r["dataset_hash"] == metadata.get("dataset_hash")]
    records = [r for r in records if r]
    if records and len(records) == len(ids) and all(
        compare_results_compatibility(records[0], r)[0]
        and r.get("threshold") == records[0].get("threshold")
        and r.get("record_count") == records[0].get("record_count") for r in records
    ):
        first = records[0]
        matrix = first["confusion_matrix"]
        st.caption(f"{first['record_count']:,} essays · {sum(matrix[0]):,} human · "
                   f"{sum(matrix[1]):,} AI · decision threshold {first['threshold']:.2f}")
        _table(records, model_specs)
        if external:
            st.caption("All four frozen models scored the same Ghostbuster essay sample. "
                       "This pilot covers one collection and one GPT condition.")
        else:
            st.caption("Aditya and Yash trained on the project split. Aakash and Daniel used independent splits.")
    else:
        st.info("A complete set of compatible scores is unavailable for this test.")

    st.subheader("Results on each model’s original test set")
    st.caption("Each team member used a different test split. Values are from the original progress report.")
    originals = [latest_result(results, mid, "original_test", "historical") for mid in TEAM_MODEL_IDS]
    if any(originals):
        _table([record for record in originals if record], model_specs, original=True)
    else:
        st.info("Original test results are unavailable.")

    with st.expander("Models and evaluation details"):
        for mid in TEAM_MODEL_IDS:
            spec = model_specs[mid]
            st.markdown(f"**{spec['member']}** · {spec['architecture']}")
        st.write("The external pilot samples matching human and GPT file IDs with seed 747. "
                 "Each pair was screened for matching 32-word passages against the full project corpus "
                 "before any predictions were made. This check cannot detect all paraphrases.")
        st.markdown("[Ghostbuster essay dataset](https://github.com/vivek3141/ghostbuster-data/tree/86ebd72590556a81622986fab736ab9227a948af/essay)")
        for r in records:
            st.caption(f"{model_specs[r['model_id']]['short_name']} · {r['source_file']}")
        if not external and records:
            st.dataframe(pd.DataFrame([{"Detector": model_specs[r["model_id"]]["short_name"],
                                        "CPU ms / essay": r.get("latency_ms_per_essay"),
                                        "Essays / second": r.get("throughput")}
                                       for r in records]), hide_index=True)
    with st.expander("Try an essay"):
        # Render the lightweight CPU detectors before loading the larger BERT checkpoint.
        live_order = (*PROJECT_SPLIT_MODELS, "aakash_7th_place", "daniel_bert")
        show_live_predictions(live_order, model_specs, adapter_loader)

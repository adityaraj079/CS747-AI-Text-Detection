from __future__ import annotations

import pandas as pd
import streamlit as st

from app_components.data_loader import compare_results_compatibility, latest_result, partition_metadata
from app_components.table_graphs import show_architecture, show_bars, show_test_graph
from app_components.inference import show_live_predictions

TEAM_MODEL_IDS = ("aakash_7th_place", "aditya_3rd_place", "daniel_bert", "yash_5th_place")
PROJECT_SPLIT_MODELS = ("aditya_3rd_place", "yash_5th_place")
TEST_SETS = {
    "Project standard test": "standard_test",
    "Project unseen sources": "unseen_test",
    "Supplemental · Ghostbuster": "ghostbuster_essay",
    "Supplemental · HC3": "hc3_answers",
    "Supplemental · GPT-wiki-intro": "wiki_intro_50",
}


def _table(records, specs, original=False, unit="essays"):
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
            row.update({"Humans flagged": matrix[0][1], f"AI {unit} missed": matrix[1][0]})
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows).set_index("Detector")
        st.table(df.style.format({"ROC-AUC": "{:.4f}", "Accuracy": "{:.2%}",
                                  "F1": "{:.4f}", "Test essays": "{:,}"}, na_rep="—"))
        show_test_graph(records, specs, original, unit)


def render_page(repo_root, results, model_specs, split_manifest, unseen_manifest, adapter_loader):
    st.title("AI essay detection")
    st.subheader("Performance comparison")
    st.caption("The project corpus supplies the training data and both project tests. Each supplemental dataset tests all four frozen detectors.")
    label = st.radio("Test set", list(TEST_SETS), horizontal=True, label_visibility="collapsed")
    dataset = TEST_SETS[label]
    external = dataset not in {"standard_test", "unseen_test"}
    unit = {"hc3_answers": "answers", "wiki_intro_50": "passages"}.get(dataset, "essays")
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
        st.caption(f"{first['record_count']:,} {unit} · {sum(matrix[0]):,} human · "
                   f"{sum(matrix[1]):,} AI · decision threshold {first['threshold']:.2f}")
        _table(records, model_specs, unit=unit)
        if dataset == "hc3_answers":
            st.caption("Each English HC3 question contributes one human answer and one ChatGPT answer. "
                       "This measures performance on question answering, a different task from essay writing.")
        elif dataset == "wiki_intro_50":
            st.caption("Each topic contributes 50 words from Wikipedia and 50 from a GPT Curie continuation.")
        elif external:
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
            st.markdown(f"**{spec['member']}**")
            show_architecture(mid)
        if external:
            if dataset == "hc3_answers":
                st.write("Using seed 747, we sampled questions with at least 32 words in both answers. "
                         "Detectors received only the answers. We retained the first 1,000 eligible pairs after screening.")
                st.markdown("[HC3 dataset · Guo et al., CC BY-SA 4.0](https://huggingface.co/datasets/Hello-SimpleAI/HC3)")
            elif dataset == "wiki_intro_50":
                st.write("Using seed 747, we selected 50,000 distinct article IDs after duplicate and overlap screening. "
                         "For each topic we scored the last 50 words of the Wikipedia introduction and the last 50 words "
                         "of the generated continuation, excluding the prompt. Results describe short encyclopedia passages.")
                st.markdown("[GPT-wiki-intro dataset · Aaditya Bhat](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro)")
            else:
                st.write("The Ghostbuster pilot samples matching human and GPT essay IDs with seed 747.")
                st.markdown("[Ghostbuster essay dataset](https://github.com/vivek3141/ghostbuster-data/tree/86ebd72590556a81622986fab736ab9227a948af/essay)")
            st.caption("We screened pairs for duplicate texts and matching 32-word passages against the full project corpus before inference. This check can miss paraphrases.")
        for r in records:
            st.caption(f"{model_specs[r['model_id']]['short_name']} · {r['source_file']}")
        if not external and records:
            timings = pd.DataFrame([{"Detector": model_specs[r["model_id"]]["short_name"],
                                     "CPU ms / essay": r.get("latency_ms_per_essay"),
                                     "Essays / second": r.get("throughput")} for r in records])
            st.dataframe(timings, hide_index=True)
            measured = timings.dropna(subset=["CPU ms / essay"])
            show_bars("CPU time per essay in the benchmark batch", measured["Detector"].tolist(),
                      {"ms per essay": measured["CPU ms / essay"].tolist()})
            if len(measured) > 1:
                latency = measured["CPU ms / essay"]
                st.caption(f"On this batch, CPU time per essay ranged from {latency.min():.2f} to {latency.max():.2f} ms.")
    with st.expander("Try an essay"):
        # Render the lightweight CPU detectors before loading the larger BERT checkpoint.
        live_order = (*PROJECT_SPLIT_MODELS, "aakash_7th_place", "daniel_bert")
        show_live_predictions(live_order, model_specs, adapter_loader)

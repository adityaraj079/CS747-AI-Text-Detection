# Dataset report

## Dataset provenance

The source dataset is read from the local CSV path provided for the project, with the expected schema: `text`, `labels`, and `source`. The dataset contains 165086 records and was validated before splitting.

## Validation and cleaning steps

1. Required columns were checked explicitly.
2. Binary labels were validated as 0/1 only.
3. Empty or missing essays were quarantined.
4. Exact duplicate texts were removed.
5. Normalized duplicates were detected using whitespace-normalized text hashing.
6. Rows with identical normalized essays but conflicting labels were quarantined and excluded from accepted partitions.
7. Stable `essay_id` values were created using deterministic SHA-256 text hashes.
8. The accepted dataset was split using a fixed random seed of 747.

## Quality summary

- Missing or empty text rows: 4
- Exact duplicate rows: 0
- Normalized duplicate rows: 10501
- Conflicting-label rows: 2
- Dataset SHA-256: `e6f09bceef8d6bd2334034ebcd0cc85f388e116f5abe9d9983d4037f6b611982`

## Accepted data summary

- Human-written essays: 42955
- AI-generated essays: 111627
- Number of sources retained: 24

## Split summary

The unseen-source holdout is set aside first. The remaining 123182 records are split 70/15/15 into train, validation, and standard test using seed 747. Partition percentages below are shares of the complete accepted dataset (154582 records).

| Partition | Count | % of accepted | Human | AI |
| --- | ---: | ---: | ---: | ---: |
| Train | 86227 | 55.8% | 11873 | 74354 |
| Validation | 18477 | 12.0% | 2544 | 15933 |
| Test | 18478 | 12.0% | 2544 | 15934 |
| Unseen test | 31400 | 20.3% | 25994 | 5406 |

## Unseen-source holdout

The unseen evaluation set intentionally holds out the following source groups: Mistral7B-CME-v1, Mistral7B-CME-v2, Mistral7B-CME-v3, Mistral7B-CME-v4, Mistral7B-CME-v5, Mistral7B-CME-v6, Mistral7B-CME-v7, Mistral7B-CME-v7-15-percent-corruption, persuade_2.0_human_scores_demo_id_github. This creates a stricter out-of-distribution evaluation while ensuring that human and AI labels remain present in the unseen data.

## Reproducibility

The splits are deterministic with `RANDOM_SEED = 747` and stable per-record `essay_id` assignments. The manifest is saved at `data/manifests/split_manifest.json`.

## Limitations

- The dataset contains source-level imbalance and varying quality across generators.
- The unseen evaluation set is source-aware and intentionally constraint-driven rather than arbitrary.
- No external or synthetic essays were created for evaluation.

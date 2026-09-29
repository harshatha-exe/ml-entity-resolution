# ML Entity Resolution System

This local machine-learning application identifies whether business records in S2 and S3 refer to the same entity as a selected S1 record. It is designed for demonstration and coursework: all included data is synthetic.

## Workflow

```text
S1 record
  -> normalize names, addresses, and countries
  -> block likely S2/S3 candidates
  -> calculate pair features
  -> trained model predict_proba()
  -> saved validation threshold
  -> ranked accepted and rejected candidates in Streamlit
```

The matcher preserves original business text for display, while normalized fields are used internally. Each candidate result includes its source, match probability, acceptance decision, and explanation features.

## Repository layout

```text
app/
  matching.py          Model-backed matching API
  streamlit_app.py     Interactive Streamlit UI
data/
  s1.csv               Source entities to match
  s2.csv, s3.csv       Candidate entity sources
  links.csv            Synthetic ground-truth links
  candidates_*.csv     Blocked train/validation/test candidate pairs
  README.md            Dataset details and demo IDs
ml/
  seed_data.py         Fixed-seed synthetic data generator
  normalize.py         Text normalization utilities
  blocking.py          Candidate generation and blocking-recall reporting
  features.py          Pair feature extraction
  train.py             Model training and artifact generation
  evaluate.py          Threshold selection and test evaluation
  model.joblib         Saved trained model
  metrics.json         Saved threshold and evaluation metrics
tests/                 Automated tests
```

## Requirements

- Python 3.10 or newer is recommended.
- `pip` and a virtual-environment tool.

## Fresh-clone setup

From the repository root, create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the complete pipeline

Run these commands from the project root after installing dependencies:

```bash
python -m ml.seed_data
python -m ml.blocking
python -m ml.train
python -m ml.evaluate
pytest -q
streamlit run app/streamlit_app.py
```

`ml.seed_data` regenerates the deterministic source data and `ml.blocking` regenerates the candidate-pair CSV files. Training writes `ml/model.joblib`. Evaluation selects a cutoff only on validation data and writes it, together with evaluation results, to `ml/metrics.json`. The Streamlit command opens a local browser session; stop it with `Ctrl+C`.

If generated data and model artifacts already exist, launch the UI directly:

```bash
streamlit run app/streamlit_app.py
```

## Matching pipeline

1. **Generate synthetic data** — creates approximately 100 S1 entities and 300 combined S2/S3 candidate records with deterministic IDs.
2. **Normalize** — lowercases text, removes punctuation and extra whitespace, handles selected company suffixes and address abbreviations, and standardizes a small set of country forms.
3. **Block candidates** — unions normalized name, suffix-free name, and address keys, then uses a limited name-token fallback where needed. Candidate blocks are capped to control comparisons.
4. **Extract features** — uses name similarity, address similarity, exact-name agreement, country agreement, and name-length difference.
5. **Train** — evaluates classifiers on train/validation candidate pairs and saves the selected model and feature order.
6. **Evaluate** — chooses the probability threshold on validation data and reports performance on the untouched S1-level test split.
7. **Serve matches** — `get_matches(s1_id)` blocks candidates, scores them with `predict_proba`, ranks them, and applies the saved threshold.

## Demo IDs

Use these IDs in the Streamlit selector or matching API:

| S1 ID | Scenario | Expected behaviour |
|---|---|---|
| `S1_001` | True match | Accepts `S2_001` and `S3_001`. |
| `S1_002` | Same-name collision | Shows `S2_002` and `S3_002` as candidates but reject them because address/country evidence conflicts. |
| `S1_003` | Singleton | Has no true link and should produce no accepted IDs. |

## Programmatic usage

```python
from app.matching import get_matches, list_entities

entities = list_entities()
result = get_matches("S1_001")

print(result["accepted_ids"])
```

`get_matches()` returns the original S1 record, saved threshold, ranked candidates, and `accepted_ids`. Each candidate includes original source text, the model probability, acceptance flag, and explanation features. An unknown S1 ID raises `ValueError` with a clear message.

## Entity Matching Application

The Streamlit application lets you select an S1 entity, inspect ranked S2/S3 candidates, and see the ML score, saved threshold, and feature-level explanation for every result.

### Running the Streamlit application

From the project root:

```bash
streamlit run app/streamlit_app.py
```

## Data provenance and privacy

All records in `data/` are synthetic and generated locally with fixed seed 42. They are not private challenge data, customer data, or production records. See [data/README.md](data/README.md) for dataset counts, scenario details, and evaluation terminology.

## Limitations

- This is a small, synthetic benchmark; reported metrics do not establish production performance.
- The normalization rules and country/address mappings are intentionally limited.
- Blocking can omit a true match if it shares none of the blocking signals; inspect the per-split recall emitted by `python -m ml.blocking`.
- A probability is a model score, not proof of identity. Review accepted candidates before any real-world decision.
- Retrain and evaluate the model if the data schema, feature set, or entity distribution changes.

## Tests

Run the full suite with:

```bash
pytest -q
```

The tests cover fixed-seed data generation, normalization, split/blocking behaviour, feature extraction and model persistence, threshold evaluation, and the true-match, collision, and singleton matching scenarios.

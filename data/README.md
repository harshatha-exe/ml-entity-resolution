# Synthetic Entity Resolution Dataset

- **Generated with seed**: 42
- **Provenance**: 100% Synthetic data generated for entity resolution benchmark testing.
- **Record counts**:
  - `s1.csv`: 100 rows
  - `s2.csv`: 150 rows (Unique IDs)
  - `s3.csv`: 150 rows (Unique IDs)
  - Total candidate records (S2 + S3): 300
  - `links.csv`: 90 true match links

## Demo S1 Entities
- **Match Demo ID**: `S1_001` (`Acme Global Solutions`) - Has true links in S2 (`S2_001`) and S3 (`S3_001`).
- **Collision Demo ID**: `S1_002` (`Apex Logistics`) - Has same-name records in S2 (`S2_002`) and S3 (`S3_002`) at different addresses/countries, which are NOT true links.
- **Singleton Demo ID**: `S1_003` (`Unique BioTech Corp`) - Has no true matches in S2 or S3.

## Metrics & Evaluation Methodology
- **Pair F0.5**: Evaluated at candidate-pair level favoring precision (beta = 0.5).
- **Mean Linked Per-S1 F0.5**: Calculated strictly on S1 entities that possess at least 1 true positive link in ground truth (denominator stated in `ml/metrics.json`).
- **Singleton No-Match Accuracy**: Evaluates no-link S1 entities separately. Correct abstention occurs when 0 candidates are falsely accepted.

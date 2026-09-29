# ML Entity Resolution System

A machine-learning based entity resolution system for identifying
whether records from different data sources refer to the same
real-world business/entity.

The project is designed as a college ML project and uses a local
dataset with Python-based data processing, feature engineering,
machine learning, and a Streamlit interface.

---

# 1. Project Objective

The objective of this project is to match records representing the
same business/entity across multiple data sources.

For example, the same business may appear as:

Source 1:

    Acme Trading Co
    14 Anna Salai, Chennai
    IN

Source 2:

    ACME Trading Company
    14 Anna Salai, Chennai
    IN

Source 3:

    Acme Trdg Co
    14 Anna Salai Chennai
    IN

Although the names and addresses are not exactly identical, these
records may represent the same real-world entity.

The system should:

1. Take an entity from S1.
2. Generate candidate entities from S2 and S3.
3. Calculate matching features.
4. Calculate a matching score/prediction.
5. Rank the candidates.
6. Identify confident matches.
7. Return the results to the Streamlit UI.

---

# 2. Technology Stack

The project uses:

- Python
- pandas
- scikit-learn
- RapidFuzz
- joblib
- Streamlit
- pytest

The project does NOT use:

- External APIs
- Cloud APIs
- A database
- API keys
- Secrets

The system is intended to run locally.

---

# 3. Project Structure

```text
ml-entity-resolution/
│
├── app/
│   ├── __init__.py
│   └── matching.py
│
├── data/
│   ├── s1.csv
│   ├── s2.csv
│   ├── s3.csv
│   └── links.csv
│
├── ml/
│   └──
│
├── notebooks/
│   └──
│
├── tests/
│   ├── __init__.py
│   └── test_matching.py
│
├── docs/
│   └──
│
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
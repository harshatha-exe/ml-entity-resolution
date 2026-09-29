"""
Streamlit UI for ML-based entity resolution.

Run from project root:

    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.matching import list_entities, get_matches


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

METRICS_FILE = PROJECT_ROOT / "ml" / "metrics.json"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Entity Resolution",
    page_icon="🔎",
    layout="wide",
)


# ---------------------------------------------------------
# Cached API calls
# ---------------------------------------------------------

@st.cache_data
def load_entities():
    return list_entities()


@st.cache_data
def load_matches(s1_id):
    return get_matches(s1_id)


@st.cache_data
def load_metrics():
    if not METRICS_FILE.exists():
        return None

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def format_percentage(value):
    return f"{float(value) * 100:.2f}%"


def explain_feature(name, value):
    """
    Plain-English explanation for one feature.
    """

    if name == "name_sim":
        if value >= 0.9:
            return "Very similar business names."
        if value >= 0.7:
            return "Business names are reasonably similar."
        if value >= 0.5:
            return "Business names have some similarity."
        return "Business names are quite different."

    if name == "addr_sim":
        if value >= 0.9:
            return "Addresses are almost identical."
        if value >= 0.7:
            return "Addresses are strongly similar."
        if value >= 0.5:
            return "Addresses have moderate similarity."
        return "Addresses are substantially different."

    if name == "exact_name":
        if value == 1:
            return "The normalized business names are exactly the same."
        return "The normalized business names are not exactly the same."

    if name == "country_match":
        if value == 1:
            return "The normalized countries match."
        return "The countries do not match."

    if name == "len_diff":
        return (
            f"The business-name lengths differ by "
            f"{int(value)} characters."
        )

    return ""


def display_feature(label, value, feature_name):
    """
    Display a feature with a simple explanation.
    """

    if feature_name in {"name_sim", "addr_sim"}:
        display_value = format_percentage(value)
    elif feature_name in {"exact_name", "country_match"}:
        display_value = "Yes" if value == 1 else "No"
    else:
        display_value = str(int(value))

    st.write(
        f"**{label}:** {display_value}"
    )

    explanation = explain_feature(
        feature_name,
        value
    )

    st.caption(explanation)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🔎 Entity Resolution")

st.write(
    "Match an S1 business entity against candidate records "
    "using the trained machine-learning model."
)

st.info(
    "Pipeline: normalize → block candidates → extract features "
    "→ ML probability → saved threshold → ranked matches"
)


# ---------------------------------------------------------
# Load S1 entities
# ---------------------------------------------------------

try:
    entities = load_entities()

except Exception as error:
    st.error(
        f"Could not load entities: {error}"
    )
    st.stop()


if not entities:
    st.warning("No S1 entities were found.")
    st.stop()


# ---------------------------------------------------------
# S1 selector
# ---------------------------------------------------------

entity_options = {
    f"{entity['entity_id']} — {entity['business_name']}":
        entity["entity_id"]
    for entity in entities
}

selected_label = st.selectbox(
    "Select an S1 entity",
    list(entity_options.keys())
)

selected_id = entity_options[selected_label]


# ---------------------------------------------------------
# Run matching
# ---------------------------------------------------------

with st.spinner("Running entity matching..."):

    try:
        result = load_matches(selected_id)

    except ValueError as error:
        st.error(str(error))
        st.stop()

    except Exception as error:
        st.error(
            f"Matching failed: {error}"
        )
        st.stop()


# ---------------------------------------------------------
# S1 information
# ---------------------------------------------------------

s1 = result["s1"]
threshold = result["threshold"]
candidates = result["candidates"]
accepted_ids = result["accepted_ids"]


st.subheader("Selected S1 Entity")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Business name**")
    st.write(s1["business_name"])

with col2:
    st.write("**Address**")
    st.write(s1["business_address"])

with col3:
    st.write("**Country**")
    st.write(s1["country"])


# ---------------------------------------------------------
# Match status
# ---------------------------------------------------------

st.subheader("Match Result")

if accepted_ids:

    st.success(
        f"Match found — {len(accepted_ids)} confident "
        f"candidate(s) accepted."
    )

else:

    st.info(
        "No confident match found."
    )


st.caption(
    f"Acceptance threshold: {threshold:.2f}"
)


# ---------------------------------------------------------
# Candidates
# ---------------------------------------------------------

st.subheader("Ranked Candidates")


if not candidates:

    st.info(
        "The blocking stage returned zero candidates."
    )

else:

    for index, candidate in enumerate(candidates, start=1):

        score = candidate["score"]
        accepted = candidate["accepted"]

        if accepted:
            status = "✅ Accepted"
        else:
            status = "Not accepted"

        with st.expander(
            f"{index}. "
            f"{candidate['candidate_id']} — "
            f"{candidate['business_name']} — "
            f"{score:.4f} — {status}",
            expanded=(index == 1)
        ):

            # -------------------------------------------------
            # Candidate original data
            # -------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Source**")
                st.write(candidate["source"])

                st.write("**Business name**")
                st.write(candidate["business_name"])

            with col2:
                st.write("**Address**")
                st.write(candidate["business_address"])

            with col3:
                st.write("**Country**")
                st.write(candidate["country"])

            st.divider()

            # -------------------------------------------------
            # Score
            # -------------------------------------------------

            score_col, threshold_col = st.columns(2)

            with score_col:
                st.metric(
                    "ML match probability",
                    f"{score:.4f}"
                )

            with threshold_col:
                st.metric(
                    "Saved threshold",
                    f"{threshold:.2f}"
                )

            if accepted:
                st.success(
                    "This candidate is above the saved "
                    "acceptance threshold."
                )
            else:
                st.warning(
                    "This candidate is below the saved "
                    "acceptance threshold."
                )

            # -------------------------------------------------
            # Features
            # -------------------------------------------------

            st.markdown("### Explanation Features")

            features = candidate["features"]

            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:

                display_feature(
                    "Name similarity",
                    features["name_sim"],
                    "name_sim"
                )

                display_feature(
                    "Exact name",
                    features["exact_name"],
                    "exact_name"
                )

                display_feature(
                    "Name length difference",
                    features["len_diff"],
                    "len_diff"
                )

            with feature_col2:

                display_feature(
                    "Address similarity",
                    features["addr_sim"],
                    "addr_sim"
                )

                display_feature(
                    "Country match",
                    features["country_match"],
                    "country_match"
                )


# ---------------------------------------------------------
# Optional metrics
# ---------------------------------------------------------

st.divider()

with st.expander("📊 Model Evaluation Metrics"):

    metrics = load_metrics()

    if metrics is None:

        st.warning(
            "ml/metrics.json was not found."
        )

    else:

        st.write(
            f"**Selected model:** "
            f"{metrics.get('selected_model', 'Unknown')}"
        )

        st.write(
            f"**Saved threshold:** "
            f"{metrics.get('threshold', 'Unknown')}"
        )

        st.markdown("### Validation")

        val_metrics = metrics.get(
            "val_metrics",
            {}
        )

        val_col1, val_col2, val_col3 = st.columns(3)

        with val_col1:
            st.metric(
                "Accuracy",
                format_percentage(
                    val_metrics.get("accuracy", 0)
                )
            )

        with val_col2:
            st.metric(
                "Precision",
                format_percentage(
                    val_metrics.get("precision", 0)
                )
            )

        with val_col3:
            st.metric(
                "F0.5",
                format_percentage(
                    val_metrics.get("f0_5", 0)
                )
            )

        st.markdown("### Untouched Test Set")

        test_metrics = metrics.get(
            "test_metrics",
            {}
        )

        test_col1, test_col2, test_col3 = st.columns(3)

        with test_col1:
            st.metric(
                "Pair F0.5",
                format_percentage(
                    test_metrics.get("f0_5", 0)
                )
            )

        with test_col2:
            st.metric(
                "Precision",
                format_percentage(
                    test_metrics.get("precision", 0)
                )
            )

        with test_col3:
            st.metric(
                "Recall",
                format_percentage(
                    test_metrics.get("recall", 0)
                )
            )

        counts = test_metrics.get(
            "counts",
            {}
        )

        if counts:

            st.markdown("### Per-S1 Test Results")

            st.write(
                f"Test S1 entities: "
                f"**{counts.get('total_test_s1', 0)}**"
            )

            st.write(
                f"Linked S1 entities: "
                f"**{counts.get('total_linked_s1', 0)}**"
            )

            st.write(
                f"No-link S1 entities: "
                f"**{counts.get('total_no_link_s1', 0)}**"
            )

            st.write(
                f"Correct accepted links: "
                f"**{counts.get('correct_accepted_links', 0)}**"
            )

            st.write(
                f"Missed links: "
                f"**{counts.get('missed_links', 0)}**"
            )

            st.write(
                f"False matches: "
                f"**{counts.get('false_matches', 0)}**"
            )

        st.caption(
            metrics.get("notes", "")
        )
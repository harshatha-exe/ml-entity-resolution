import re
import pandas as pd

COMPANY_SUFFIXES = [
    r"\bincorporated\b", r"\binc\b", r"\bllc\b", r"\bcorp\b", r"\bcorporation\b",
    r"\blimited\b", r"\bltd\b", r"\bco\b", r"\bcompany\b", r"\bgroup\b",
    r"\bplc\b", r"\bgmbh\b", r"\bsa\b", r"\bholdings\b", r"\bservices\b"
]

ADDRESS_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\brd\b": "road",
    r"\bblvd\b": "boulevard",
    r"\bste\b": "suite",
    r"\bdr\b": "drive",
    r"\bapt\b": "apartment",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bpl\b": "place",
    r"\bct\b": "court",
    r"\bln\b": "lane"
}

COUNTRY_MAP = {
    "ca": "canada",
    "canada": "canada",
    "usa": "united states",
    "us": "united states",
    "united states": "united states",
    "uk": "united kingdom",
    "united kingdom": "united kingdom",
    "de": "germany",
    "germany": "germany",
    "fr": "france",
    "france": "france",
    "aus": "australia",
    "australia": "australia",
    "jpn": "japan",
    "japan": "japan"
}

def clean_text(text: str) -> str:
    """Lowercases, removes punctuation, normalizes whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_name(name: str) -> str:
    """Returns normalized name string."""
    return clean_text(name)

def strip_company_suffixes(name: str) -> str:
    """Removes common company suffixes from clean normalized name."""
    norm = clean_text(name)
    for sfx in COMPANY_SUFFIXES:
        norm = re.sub(sfx, "", norm)
    return re.sub(r"\s+", " ", norm).strip()

def normalize_address(address: str) -> str:
    """Normalizes address string and expands common address abbreviations."""
    norm = clean_text(address)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        norm = re.sub(pattern, replacement, norm)
    return re.sub(r"\s+", " ", norm).strip()

def normalize_country(country: str) -> str:
    """Standardizes country codes/names."""
    clean = clean_text(country)
    return COUNTRY_MAP.get(clean, clean)

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds normalized fields while retaining original strings."""
    df_out = df.copy()
    if "business_name" in df_out.columns:
        df_out["norm_name"] = df_out["business_name"].apply(normalize_name)
        df_out["suffix_free_name"] = df_out["business_name"].apply(strip_company_suffixes)
    if "business_address" in df_out.columns:
        df_out["norm_address"] = df_out["business_address"].apply(normalize_address)
    if "country" in df_out.columns:
        df_out["norm_country"] = df_out["country"].apply(normalize_country)
    return df_out

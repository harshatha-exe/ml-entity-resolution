import os
import random
import pandas as pd

def generate_seed_data(data_dir: str = "data", seed: int = 42) -> None:
    random.seed(seed)
    os.makedirs(data_dir, exist_ok=True)

    s1_rows = []
    s2_rows = []
    s3_rows = []
    links_rows = []

    s2_counter = 1
    s3_counter = 1

    def next_s2_id():
        nonlocal s2_counter
        cid = f"S2_{s2_counter:03d}"
        s2_counter += 1
        return cid

    def next_s3_id():
        nonlocal s3_counter
        cid = f"S3_{s3_counter:03d}"
        s3_counter += 1
        return cid

    # 1. Fixed hand-checkable cases (1 to 12)
    # S1_001: Match demo
    s1_001_s2 = next_s2_id() # S2_001
    s1_001_s3 = next_s3_id() # S3_001
    s1_rows.append({"entity_id": "S1_001", "business_name": "Acme Global Solutions", "business_address": "123 Innovation Way Suite 400", "country": "USA"})
    s2_rows.append({"entity_id": s1_001_s2, "business_name": "Acme Global Solutions Inc.", "business_address": "123 Innovation Way Ste 400", "country": "USA"})
    s3_rows.append({"entity_id": s1_001_s3, "business_name": "Acme Global Solutions LLC", "business_address": "123 Innovation Way", "country": "USA"})
    links_rows.append({"s1_id": "S1_001", "other_id": s1_001_s2, "source": "s2"})
    links_rows.append({"s1_id": "S1_001", "other_id": s1_001_s3, "source": "s3"})

    # S1_002: Collision demo (same name, different address/country, unlinked)
    s1_002_s2 = next_s2_id() # S2_002
    s1_002_s3 = next_s3_id() # S3_002
    s1_rows.append({"entity_id": "S1_002", "business_name": "Apex Logistics", "business_address": "456 Main St", "country": "USA"})
    s2_rows.append({"entity_id": s1_002_s2, "business_name": "Apex Logistics", "business_address": "789 Market St", "country": "UK"})
    s3_rows.append({"entity_id": s1_002_s3, "business_name": "Apex Logistics", "business_address": "100 Industrial Pkwy", "country": "CA"})

    # S1_003: Singleton demo (no candidates/matches)
    s1_rows.append({"entity_id": "S1_003", "business_name": "Unique BioTech Corp", "business_address": "789 Research Blvd", "country": "GERMANY"})

    # S1_004: Typo variant match
    s1_004_s2 = next_s2_id()
    s1_rows.append({"entity_id": "S1_004", "business_name": "Starlight Retailers", "business_address": "12 Star Ave", "country": "USA"})
    s2_rows.append({"entity_id": s1_004_s2, "business_name": "Starlite Retailers", "business_address": "12 Star Avenue", "country": "USA"})
    links_rows.append({"s1_id": "S1_004", "other_id": s1_004_s2, "source": "s2"})

    # S1_005: Suffix variant match
    s1_005_s3 = next_s3_id()
    s1_rows.append({"entity_id": "S1_005", "business_name": "Vortex Technologies Ltd", "business_address": "50 Tech Park", "country": "UK"})
    s3_rows.append({"entity_id": s1_005_s3, "business_name": "Vortex Technologies Group", "business_address": "50 Tech Park", "country": "UK"})
    links_rows.append({"s1_id": "S1_005", "other_id": s1_005_s3, "source": "s3"})

    # S1_006: Address abbreviation match
    s1_006_s2 = next_s2_id()
    s1_rows.append({"entity_id": "S1_006", "business_name": "Summit Health Systems", "business_address": "300 Doctors Dr Apt 2B", "country": "USA"})
    s2_rows.append({"entity_id": s1_006_s2, "business_name": "Summit Health Systems", "business_address": "300 Doctors Drive Apartment 2B", "country": "USA"})
    links_rows.append({"s1_id": "S1_006", "other_id": s1_006_s2, "source": "s2"})

    # S1_007: Country variant match
    s1_007_s3 = next_s3_id()
    s1_rows.append({"entity_id": "S1_007", "business_name": "Global Energy Partners", "business_address": "10 Oil Plaza", "country": "CA"})
    s3_rows.append({"entity_id": s1_007_s3, "business_name": "Global Energy Partners", "business_address": "10 Oil Plaza", "country": "Canada"})
    links_rows.append({"s1_id": "S1_007", "other_id": s1_007_s3, "source": "s3"})

    # S1_008: Pacific Financial Services match
    s1_008_s2 = next_s2_id()
    s1_rows.append({"entity_id": "S1_008", "business_name": "Pacific Financial Services", "business_address": "500 Wall Street", "country": "USA"})
    s2_rows.append({"entity_id": s1_008_s2, "business_name": "Pacific Financial", "business_address": "500 Wall St", "country": "USA"})
    links_rows.append({"s1_id": "S1_008", "other_id": s1_008_s2, "source": "s2"})

    # S1_009: Second singleton
    s1_rows.append({"entity_id": "S1_009", "business_name": "Orion Robotics Inc", "business_address": "99 Future Way", "country": "JPN"})

    # S1_010: Second collision (unlinked decoy)
    s1_010_s2 = next_s2_id()
    s1_rows.append({"entity_id": "S1_010", "business_name": "Horizon Media", "business_address": "11 Broad St", "country": "USA"})
    s2_rows.append({"entity_id": s1_010_s2, "business_name": "Horizon Media", "business_address": "22 Broad St", "country": "USA"})

    # S1_011: Match
    s1_011_s2 = next_s2_id()
    s1_rows.append({"entity_id": "S1_011", "business_name": "Alpha Biotech", "business_address": "80 Science Rd", "country": "USA"})
    s2_rows.append({"entity_id": s1_011_s2, "business_name": "Alpha Biotech LLC", "business_address": "80 Science Road", "country": "USA"})
    links_rows.append({"s1_id": "S1_011", "other_id": s1_011_s2, "source": "s2"})

    # S1_012: Match
    s1_012_s3 = next_s3_id()
    s1_rows.append({"entity_id": "S1_012", "business_name": "Beta Systems", "business_address": "90 Tech Lane", "country": "UK"})
    s3_rows.append({"entity_id": s1_012_s3, "business_name": "Beta Systems Corp", "business_address": "90 Tech Lane", "country": "UK"})
    links_rows.append({"s1_id": "S1_012", "other_id": s1_012_s3, "source": "s3"})

    # 2. Generate remaining S1 entities up to 100
    prefixes = ["Alpha", "Beta", "Delta", "Echo", "Gamma", "Nova", "Omni", "Prime", "Quantum", "Solar", "Terra", "Apex", "Zenith", "Atlas", "Beacon"]
    domains = ["Tech", "Logistics", "Health", "Financial", "Energy", "Retail", "Systems", "Media", "Solutions", "Industries", "Group", "Labs", "Capital", "Ventures"]
    suffixes = ["Inc.", "LLC", "Corp", "Ltd", "Co", "Group", "Holdings"]
    street_names = ["Main St", "Market St", "Park Ave", "Broadway", "Fifth Ave", "Oak Rd", "Pine St", "Washington Blvd", "Lincoln Way", "Commercial Dr"]
    countries = ["USA", "UK", "CA", "DE", "FR", "AUS", "JPN"]

    for idx in range(13, 101):
        s1_id = f"S1_{idx:03d}"
        p = random.choice(prefixes)
        d = random.choice(domains)
        sfx = random.choice(suffixes)
        name = f"{p} {d} {sfx}"
        street_num = random.randint(100, 9999)
        st = random.choice(street_names)
        addr = f"{street_num} {st}"
        country = random.choice(countries)

        s1_rows.append({"entity_id": s1_id, "business_name": name, "business_address": addr, "country": country})

        # 75% match chance
        if random.random() < 0.75:
            source = random.choice(["s2", "s3", "both"])
            if source in ["s2", "both"]:
                cid = next_s2_id()
                cname = f"{p} {d} {random.choice(suffixes)}"
                caddr = addr.replace(" St", " Street").replace(" Ave", " Avenue").replace(" Rd", " Road").replace(" Blvd", " Boulevard")
                s2_rows.append({"entity_id": cid, "business_name": cname, "business_address": caddr, "country": country})
                links_rows.append({"s1_id": s1_id, "other_id": cid, "source": "s2"})
            if source in ["s3", "both"]:
                cid = next_s3_id()
                cname = f"{p} {d} {random.choice(suffixes)}"
                caddr = addr.replace(" St", " Street").replace(" Ave", " Avenue").replace(" Rd", " Road").replace(" Blvd", " Boulevard")
                s3_rows.append({"entity_id": cid, "business_name": cname, "business_address": caddr, "country": country})
                links_rows.append({"s1_id": s1_id, "other_id": cid, "source": "s3"})

    # 3. Add extra unlinked noise candidate records to grow S2 and S3 to ~150 each (total 300 candidate records)
    while len(s2_rows) < 150:
        cid = next_s2_id()
        p = random.choice(prefixes)
        d = random.choice(domains)
        name = f"{p} {d} {random.choice(suffixes)}"
        addr = f"{random.randint(100, 9999)} {random.choice(street_names)}"
        s2_rows.append({"entity_id": cid, "business_name": name, "business_address": addr, "country": random.choice(countries)})

    while len(s3_rows) < 150:
        cid = next_s3_id()
        p = random.choice(prefixes)
        d = random.choice(domains)
        name = f"{p} {d} {random.choice(suffixes)}"
        addr = f"{random.randint(100, 9999)} {random.choice(street_names)}"
        s3_rows.append({"entity_id": cid, "business_name": name, "business_address": addr, "country": random.choice(countries)})

    s1_df = pd.DataFrame(s1_rows)
    s2_df = pd.DataFrame(s2_rows)
    s3_df = pd.DataFrame(s3_rows)
    links_df = pd.DataFrame(links_rows)

    # Assert strict uniqueness
    assert s2_df["entity_id"].is_unique, "S2 entity_ids are not unique!"
    assert s3_df["entity_id"].is_unique, "S3 entity_ids are not unique!"

    s1_df.to_csv(os.path.join(data_dir, "s1.csv"), index=False)
    s2_df.to_csv(os.path.join(data_dir, "s2.csv"), index=False)
    s3_df.to_csv(os.path.join(data_dir, "s3.csv"), index=False)
    links_df.to_csv(os.path.join(data_dir, "links.csv"), index=False)

    readme_content = f"""# Synthetic Entity Resolution Dataset

- **Generated with seed**: {seed}
- **Provenance**: 100% Synthetic data generated for entity resolution benchmark testing.
- **Record counts**:
  - `s1.csv`: {len(s1_df)} rows
  - `s2.csv`: {len(s2_df)} rows (Unique IDs)
  - `s3.csv`: {len(s3_df)} rows (Unique IDs)
  - Total candidate records (S2 + S3): {len(s2_df) + len(s3_df)}
  - `links.csv`: {len(links_df)} true match links

## Demo S1 Entities
- **Match Demo ID**: `S1_001` (`Acme Global Solutions`) - Has true links in S2 (`S2_001`) and S3 (`S3_001`).
- **Collision Demo ID**: `S1_002` (`Apex Logistics`) - Has same-name records in S2 (`S2_002`) and S3 (`S3_002`) at different addresses/countries, which are NOT true links.
- **Singleton Demo ID**: `S1_003` (`Unique BioTech Corp`) - Has no true matches in S2 or S3.
"""
    with open(os.path.join(data_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Generated synthetic dataset: S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)}, Links={len(links_df)}")

if __name__ == "__main__":
    generate_seed_data()

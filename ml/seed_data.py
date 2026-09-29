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

    # 1. Fixed hand-checkable cases (1 to 12)
    hand_checkable = [
        # Match demo ID (S1_001)
        {
            "s1_id": "S1_001",
            "s1_name": "Acme Global Solutions",
            "s1_addr": "123 Innovation Way Suite 400",
            "s1_country": "USA",
            "matches": [
                {"id": "S2_001", "source": "s2", "name": "Acme Global Solutions Inc.", "addr": "123 Innovation Way Ste 400", "country": "USA"},
                {"id": "S3_001", "source": "s3", "name": "Acme Global Solutions LLC", "addr": "123 Innovation Way", "country": "USA"}
            ],
            "decoys": []
        },
        # Collision demo ID (S1_002) - same name, different address/country, NOT linked
        {
            "s1_id": "S1_002",
            "s1_name": "Apex Logistics",
            "s1_addr": "456 Main St",
            "s1_country": "USA",
            "matches": [],
            "decoys": [
                {"id": "S2_002", "source": "s2", "name": "Apex Logistics", "addr": "789 Market St", "country": "UK"},
                {"id": "S3_002", "source": "s3", "name": "Apex Logistics", "addr": "100 Industrial Pkwy", "country": "CA"}
            ]
        },
        # Singleton demo ID (S1_003) - no match
        {
            "s1_id": "S1_003",
            "s1_name": "Unique BioTech Corp",
            "s1_addr": "789 Research Blvd",
            "s1_country": "GERMANY",
            "matches": [],
            "decoys": []
        },
        # Case 4: Typo variant
        {
            "s1_id": "S1_004",
            "s1_name": "Starlight Retailers",
            "s1_addr": "12 Star Ave",
            "s1_country": "USA",
            "matches": [
                {"id": "S2_004", "source": "s2", "name": "Starlite Retailers", "addr": "12 Star Avenue", "country": "USA"}
            ],
            "decoys": []
        },
        # Case 5: Suffix variant
        {
            "s1_id": "S1_005",
            "s1_name": "Vortex Technologies Ltd",
            "s1_addr": "50 Tech Park",
            "s1_country": "UK",
            "matches": [
                {"id": "S3_005", "source": "s3", "name": "Vortex Technologies Group", "addr": "50 Tech Park", "country": "UK"}
            ],
            "decoys": []
        },
        # Case 6: Address abbreviation variant
        {
            "s1_id": "S1_006",
            "s1_name": "Summit Health Systems",
            "s1_addr": "300 Doctors Dr Apt 2B",
            "s1_country": "USA",
            "matches": [
                {"id": "S2_006", "source": "s2", "name": "Summit Health Systems", "addr": "300 Doctors Drive Apartment 2B", "country": "USA"}
            ],
            "decoys": []
        },
        # Case 7: Country variant
        {
            "s1_id": "S1_007",
            "s1_name": "Global Energy Partners",
            "s1_addr": "10 Oil Plaza",
            "s1_country": "CA",
            "matches": [
                {"id": "S3_007", "source": "s3", "name": "Global Energy Partners", "addr": "10 Oil Plaza", "country": "Canada"}
            ],
            "decoys": []
        },
        # Case 8: Decoy match
        {
            "s1_id": "S1_008",
            "s1_name": "Pacific Financial Services",
            "s1_addr": "500 Wall Street",
            "s1_country": "USA",
            "matches": [
                {"id": "S2_008", "source": "s2", "name": "Pacific Financial", "addr": "500 Wall St", "country": "USA"}
            ],
            "decoys": []
        },
        # Case 9: Second singleton
        {
            "s1_id": "S1_009",
            "s1_name": "Orion Robotics Inc",
            "s1_addr": "99 Future Way",
            "s1_country": "JPN",
            "matches": [],
            "decoys": []
        },
        # Case 10: Second collision
        {
            "s1_id": "S1_010",
            "s1_name": "Horizon Media",
            "s1_addr": "11 Broad St",
            "s1_country": "USA",
            "matches": [],
            "decoys": [
                {"id": "S2_010", "source": "s2", "name": "Horizon Media", "addr": "22 Broad St", "country": "USA"}
            ]
        },
        # Case 11: Match
        {
            "s1_id": "S1_011",
            "s1_name": "Alpha Biotech",
            "s1_addr": "80 Science Rd",
            "s1_country": "USA",
            "matches": [
                {"id": "S2_011", "source": "s2", "name": "Alpha Biotech LLC", "addr": "80 Science Road", "country": "USA"}
            ],
            "decoys": []
        },
        # Case 12: Match
        {
            "s1_id": "S1_012",
            "s1_name": "Beta Systems",
            "s1_addr": "90 Tech Lane",
            "s1_country": "UK",
            "matches": [
                {"id": "S3_012", "source": "s3", "name": "Beta Systems Corp", "addr": "90 Tech Lane", "country": "UK"}
            ],
            "decoys": []
        }
    ]

    for item in hand_checkable:
        s1_rows.append({
            "entity_id": item["s1_id"],
            "business_name": item["s1_name"],
            "business_address": item["s1_addr"],
            "country": item["s1_country"]
        })
        for m in item["matches"]:
            row = {"entity_id": m["id"], "business_name": m["name"], "business_address": m["addr"], "country": m["country"]}
            if m["source"] == "s2":
                s2_rows.append(row)
            else:
                s3_rows.append(row)
            links_rows.append({"s1_id": item["s1_id"], "other_id": m["id"], "source": m["source"]})
        for d in item["decoys"]:
            row = {"entity_id": d["id"], "business_name": d["name"], "business_address": d["addr"], "country": d["country"]}
            if d["source"] == "s2":
                s2_rows.append(row)
            else:
                s3_rows.append(row)

    # 2. Synthetic generation for S1_013 through S1_100
    prefixes = ["Alpha", "Beta", "Delta", "Echo", "Gamma", "Nova", "Omni", "Prime", "Quantum", "Solar", "Terra", "Apex", "Zenith", "Atlas", "Beacon"]
    domains = ["Tech", "Logistics", "Health", "Financial", "Energy", "Retail", "Systems", "Media", "Solutions", "Industries", "Group", "Labs", "Capital", "Ventures"]
    suffixes = ["Inc.", "LLC", "Corp", "Ltd", "Co", "Group", "Holdings"]
    street_names = ["Main St", "Market St", "Park Ave", "Broadway", "Fifth Ave", "Oak Rd", "Pine St", "Washington Blvd", "Lincoln Way", "Commercial Dr"]
    countries = ["USA", "UK", "CA", "DE", "FR", "AUS", "JPN"]

    s2_counter = len(s2_rows) + 1
    s3_counter = len(s3_rows) + 1

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

        s1_rows.append({
            "entity_id": s1_id,
            "business_name": name,
            "business_address": addr,
            "country": country
        })

        # 75% match chance
        if random.random() < 0.75:
            num_matches = random.choice([1, 1, 2])
            for _ in range(num_matches):
                source = random.choice(["s2", "s3"])
                if source == "s2":
                    cand_id = f"S2_{s2_counter:03d}"
                    s2_counter += 1
                else:
                    cand_id = f"S3_{s3_counter:03d}"
                    s3_counter += 1

                # Variant name/address
                alt_sfx = random.choice(suffixes)
                cand_name = f"{p} {d} {alt_sfx}"
                # Slight variation in address abbreviation
                cand_addr = addr.replace(" St", " Street").replace(" Ave", " Avenue").replace(" Rd", " Road").replace(" Blvd", " Boulevard")
                cand_country = country

                row = {"entity_id": cand_id, "business_name": cand_name, "business_address": cand_addr, "country": cand_country}
                if source == "s2":
                    s2_rows.append(row)
                else:
                    s3_rows.append(row)

                links_rows.append({"s1_id": s1_id, "other_id": cand_id, "source": source})
        else:
            # Singleton or decoy
            if random.random() < 0.5:
                # Decoy
                source = random.choice(["s2", "s3"])
                cand_id = f"S2_{s2_counter:03d}" if source == "s2" else f"S3_{s3_counter:03d}"
                if source == "s2":
                    s2_counter += 1
                else:
                    s3_counter += 1
                decoy_addr = f"{random.randint(100, 9999)} {random.choice(street_names)}"
                row = {"entity_id": cand_id, "business_name": name, "business_address": decoy_addr, "country": random.choice(countries)}
                if source == "s2":
                    s2_rows.append(row)
                else:
                    s3_rows.append(row)

    # Convert to DataFrames and save
    s1_df = pd.DataFrame(s1_rows)
    s2_df = pd.DataFrame(s2_rows)
    s3_df = pd.DataFrame(s3_rows)
    links_df = pd.DataFrame(links_rows)

    s1_df.to_csv(os.path.join(data_dir, "s1.csv"), index=False)
    s2_df.to_csv(os.path.join(data_dir, "s2.csv"), index=False)
    s3_df.to_csv(os.path.join(data_dir, "s3.csv"), index=False)
    links_df.to_csv(os.path.join(data_dir, "links.csv"), index=False)

    # Create data/README.md
    readme_content = f"""# Synthetic Entity Resolution Dataset

- **Generated with seed**: {seed}
- **Provenance**: 100% Synthetic data generated for entity resolution benchmark testing. Contains zero restricted or challenge data.
- **Record counts**:
  - `s1.csv`: {len(s1_df)} rows
  - `s2.csv`: {len(s2_df)} rows
  - `s3.csv`: {len(s3_df)} rows
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

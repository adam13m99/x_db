# Step 1 - sf_code_extractor.py
"""
sf_code_extracter.py — concise overview
---------------------------------------
Purpose:
- Consolidate Snappfood vendor identifiers by extracting `sf_code`/`vendor_code` from scraped files,
  merging them with `prelinked_sf_code` from `outputs/tf_vendors.csv`, de-duplicating, and writing
  the final list to `data/scraped/snappfood_vendor_codes.csv`.

Main steps:
1) Scan matching CSVs (see SCRAPED_PATTERNS) and collect `sf_code`/`vendor_code`.
2) Read `prelinked_sf_code` (and `tf_code`) from `outputs/tf_vendors.csv`.
3) Merge new codes with any existing output, de-duplicate, and clean blanks/NaNs.
4) Filter out any `sf_code` that equals or contains a `tf_code` (exact + substring).
5) Save the final sorted codes to `data/scraped/snappfood_vendor_codes.csv` and print stats.

Inputs:
- Scraped/vendor files matching:
  - data/scraped/V_sf_vendor_dual_grading_*.csv
  - data/scraped/V_sf_vendor_scrape_*.csv
  - data/scraped/extra_sf_codes*.csv
  - data/scraped/extra_sf_codes.csv
  - data/scraped/snappfood_vendor_codes.csv (existing output, if present)
  - data/extra_matches/extra_matches.csv
- Optional: outputs/tf_vendors.csv providing `prelinked_sf_code` and `tf_code`.

Outputs:
- data/scraped/snappfood_vendor_codes.csv with one column: `sf_code` (deduplicated, filtered, sorted).

Run:
    python sf_code_extracter.py
"""


import glob
import os
import time
from datetime import timedelta
from typing import List, Set, Tuple
import pandas as pd


OUTPUT_FILE = "data/scraped/snappfood_vendor_codes.csv"
SCRAPED_PATTERNS = [
    'data/scraped/V_sf_vendor_dual_grading_*.csv',
    'data/scraped/V_sf_vendor_scrape_*.csv',
    'data/scraped/extra_sf_codes*.csv',
    'data/scraped/extra_sf_codes.csv',
    "data/scraped/snappfood_vendor_codes.csv",
    "data/extra_matches/extra_matches.csv"
]


def _dedupe_clean(codes: List[str]) -> List[str]:
    cleaned = []
    for c in codes:
        if pd.isna(c):
            continue
        s = str(c).strip()
        if not s or s.lower() == "nan":
            continue
        cleaned.append(s)
    return sorted(set(cleaned))


def _read_scraped_codes() -> List[str]:
    print("--- Scanning scraped vendor files ---")
    all_files: List[str] = []
    for pattern in SCRAPED_PATTERNS:
        matched = glob.glob(pattern)
        print(f"Pattern '{pattern}' -> {len(matched)} file(s)")
        all_files.extend(matched)

    scraped_codes: List[str] = []
    for fp in all_files:
        try:
            df = pd.read_csv(fp)
            col = 'sf_code' if 'sf_code' in df.columns else ('vendor_code' if 'vendor_code' in df.columns else None)
            if col is None:
                print(f"  {os.path.basename(fp)}: SKIPPED (no sf_code/vendor_code column)")
                continue
            cleaned = _dedupe_clean(df[col].tolist())
            scraped_codes.extend(cleaned)
            print(f"  {os.path.basename(fp)}: {len(cleaned)} code(s)")
        except Exception as e:
            print(f"  {os.path.basename(fp)}: ERROR - {e}")

    scraped_codes = _dedupe_clean(scraped_codes)
    print(f"Total unique codes from scraped files: {len(scraped_codes)}")
    return scraped_codes


def _tf_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "outputs", "tf_vendors.csv")


def _read_tf_prelinked_codes() -> List[str]:
    tf_path = _tf_path()
    if not os.path.exists(tf_path):
        print(f"--- tf_vendors.csv not found at {tf_path}; skipping prelinked_sf_code merge.")
        return []

    print(f"--- Reading prelinked_sf_code from: {tf_path}")
    try:
        df = pd.read_csv(tf_path)
    except Exception as e:
        print(f"ERROR reading tf_vendors.csv: {e}")
        return []

    if 'prelinked_sf_code' not in df.columns:
        print("tf_vendors.csv has no 'prelinked_sf_code' column; skipping.")
        return []

    prelinked = _dedupe_clean(df['prelinked_sf_code'].tolist())
    print(f"Total unique prelinked_sf_code from tf_vendors: {len(prelinked)}")
    return prelinked


def _read_tf_codes() -> List[str]:
    """Read tf_code list to filter out any occurrences in sf_code."""
    tf_path = _tf_path()
    if not os.path.exists(tf_path):
        print(f"--- tf_vendors.csv not found at {tf_path}; cannot filter sf_code against tf_code.")
        return []

    print(f"--- Reading tf_code list from: {tf_path}")
    try:
        df = pd.read_csv(tf_path)
    except Exception as e:
        print(f"ERROR reading tf_vendors.csv: {e}")
        return []

    if 'tf_code' not in df.columns:
        print("tf_vendors.csv has no 'tf_code' column; skipping sf_code vs tf_code filtering.")
        return []

    tf_codes = _dedupe_clean(df['tf_code'].tolist())
    print(f"Total unique tf_code from tf_vendors: {len(tf_codes)}")
    return tf_codes


def _filter_out_sf_codes_with_tf(
    sf_codes: List[str],
    tf_codes: List[str],
) -> Tuple[List[str], int, int]:
    """
    Returns (filtered_sf_codes, removed_exact_count, removed_substring_count)
    - Exact removal: sf_code == tf_code
    - Substring removal: tf_code is a substring of sf_code
    """
    if not tf_codes or not sf_codes:
        return sf_codes, 0, 0

    tf_set = set(tf_codes)

    # Remove exact matches first
    remaining = [c for c in sf_codes if c not in tf_set]
    removed_exact = len(sf_codes) - len(remaining)

    # Then remove any where a tf_code appears as a substring
    tf_codes_sorted = sorted(tf_codes, key=len, reverse=True)

    filtered: List[str] = []
    removed_substring = 0
    for c in remaining:
        hit = False
        for t in tf_codes_sorted:
            if t and t in c:
                hit = True
                break
        if hit:
            removed_substring += 1
        else:
            filtered.append(c)

    return filtered, removed_exact, removed_substring


def _read_existing_output() -> List[str]:
    if not os.path.exists(OUTPUT_FILE):
        return []
    try:
        df = pd.read_csv(OUTPUT_FILE)
        if 'sf_code' not in df.columns:
            return []
        existing = _dedupe_clean(df['sf_code'].tolist())
        print(f"Existing output file has {len(existing)} code(s).")
        return existing
    except Exception as e:
        print(f"WARNING: Could not read existing {OUTPUT_FILE}: {e}")
        return []


def extract_sf_codes() -> bool:
    print("--- Extracting SF Codes ---")

    scraped_codes = _read_scraped_codes()
    tf_prelinked_codes = _read_tf_prelinked_codes()
    existing_codes = _read_existing_output()

    before_merge_count = len(existing_codes)
    merged: Set[str] = set(existing_codes)
    merged.update(scraped_codes)
    added_from_scraped = len(merged) - before_merge_count

    before_tf_merge = len(merged)
    merged.update(tf_prelinked_codes)
    added_from_tf = len(merged) - before_tf_merge

    total_unique = len(merged)
    if total_unique == 0:
        print("ERROR: No sf_codes found from any source.")
        return False

    # Convert to sorted list
    merged_list = sorted(merged)

    # Remove any sf_code that equals/contains a tf_code
    tf_codes = _read_tf_codes()
    filtered_list, removed_exact, removed_substring = _filter_out_sf_codes_with_tf(merged_list, tf_codes)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    final_df = pd.DataFrame({'sf_code': filtered_list})
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(filtered_list)} unique code(s) to: {OUTPUT_FILE}")
    print(f"  Newly added from scraped files: {added_from_scraped}")
    print(f"  Newly added from tf_vendors prelinked_sf_code: {added_from_tf}")
    print(f"  Removed for tf_code conflicts (exact): {removed_exact}")
    print(f"  Removed for tf_code conflicts (substring): {removed_substring}")

    return True


def _format_duration(seconds: float) -> str:
    """Return human-friendly HH:MM:SS.mmm string."""
    return str(timedelta(seconds=round(seconds, 3)))


if __name__ == "__main__":
    start = time.perf_counter()
    success = extract_sf_codes()
    if success:
        print("\n[SUCCESS] SF code consolidation completed!")
    else:
        print("\n[FAILED] SF code consolidation failed.")
    elapsed = time.perf_counter() - start
    print(f"\n[STATS] Total runtime: {_format_duration(elapsed)} ({elapsed:.3f}s)")

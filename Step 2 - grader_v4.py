# grader_sf_only_configurable_timed.py
# - Computes sf_grade (city_id × business_line) with configurable thresholds.
# - Vendors with comment_count < MIN_COMMENTS -> INELIGIBLE_LABEL.
# - Top/Bottom percentages by comment_count are configurable.
# - Reads and writes sf_vendors.csv (adds sf_grade column back to the same file).
# - Prints a summary and total elapsed time at the end.

from __future__ import annotations
import sys
import glob
import time
from typing import List
import numpy as np
import pandas as pd

# ==================== CONFIG (edit these) ====================
# IO
FILE_GLOB = "outputs/sf_vendors.csv"     # Input file or glob
OUTPUT_CSV = "outputs/sf_vendors.csv"    # Output file (same file to "add" sf_grade)

# Eligibility
MIN_COMMENTS = 300               # comment_count < MIN_COMMENTS -> ineligible
INELIGIBLE_LABEL = "Not Enough Rate"

# Percentages (fractions: 0.25 means 25%)
TOP_PERCENT = 0.25               # top X% get level-up
BOTTOM_PERCENT = 0.25            # bottom X% get a '-'

# Rating bands (right-closed intervals, except A uses strict >)
# A: r > RATING_A_GT
RATING_A_GT = 9.0
# B: RATING_B_LO < r <= RATING_B_HI
RATING_B_LO, RATING_B_HI = 8.4, 9.0
# C: RATING_C_LO < r <= RATING_C_HI
RATING_C_LO, RATING_C_HI = 7.9, 8.4
# D: RATING_D_LO < r <= RATING_D_HI
RATING_D_LO, RATING_D_HI = 7.0, 7.9
# E: RATING_E_LO < r <= RATING_E_HI
RATING_E_LO, RATING_E_HI = 6.0, 7.0
# F: everything else (including NaN)
# ============================================================

REQUIRED_COLUMNS = {
    "sf_code", "city_id", "business_line",
    "comment_count", "rating"
}

def load_input_frames(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)

def ensure_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default)

def base_letter_from_rating(r: float) -> str:
    """
    Uses configurable rating bands (edit in CONFIG section).
    """
    if pd.isna(r):
        return "F"
    if r > RATING_A_GT:                         return "A"
    if RATING_B_LO < r <= RATING_B_HI:          return "B"
    if RATING_C_LO < r <= RATING_C_HI:          return "C"
    if RATING_D_LO < r <= RATING_D_HI:          return "D"
    if RATING_E_LO < r <= RATING_E_HI:          return "E"
    return "F"

LEVEL_UP = {"E": "D", "D": "C", "C": "B", "B": "A"}  # A handled separately (A -> A+)

def apply_rules_with_levelup(
    df: pd.DataFrame,
    group_cols: List[str],
    out_col: str
) -> pd.DataFrame:
    """
    - If comment_count < MIN_COMMENTS -> INELIGIBLE_LABEL
    - Else:
        Base letter by rating bands (A..F).
        Top TOP_PERCENT by comments (eligible-only in each group):
          A  -> A+   ;  B -> A ; C -> B ; D -> C ; E -> D ; F -> F
        Bottom BOTTOM_PERCENT by comments:
          A..E -> letter + '-' ; F unchanged
        Middle -> base letter unchanged
    Uses a frozen 'base' snapshot to avoid cascading upgrades.
    """
    out = df.copy()
    out["comment_count"] = to_numeric(out["comment_count"], 0).clip(lower=0)
    out["rating"] = to_numeric(out["rating"], 0)

    eligible = out["comment_count"] >= MIN_COMMENTS

    # Base letters (snapshot)
    base = out["rating"].apply(base_letter_from_rating)

    # Percentile ranks of comments among eligible rows within each group
    comment_rank = pd.Series(np.nan, index=out.index)
    if eligible.any():
        comment_rank.loc[eligible] = (
            out.loc[eligible]
               .groupby(group_cols, observed=True)["comment_count"]
               .transform(lambda s: s.rank(pct=True, method="average"))
        )

    final = base.copy()

    # Translate percentages to rank thresholds
    top_cut = 1.0 - TOP_PERCENT
    bot_cut = BOTTOM_PERCENT

    # Quartile masks (only eligible)
    top_q = eligible & (comment_rank >= top_cut)
    bot_q = eligible & (comment_rank <= bot_cut)

    # Top: A -> A+ ; B/C/D/E -> one step up ; F unchanged
    final.loc[top_q & (base == "A")] = "A+"
    for src, dst in LEVEL_UP.items():
        final.loc[top_q & (base == src)] = dst

    # Bottom: A..E -> add '-' ; F unchanged
    for letter in list("ABCDE"):
        final.loc[bot_q & (base == letter)] = letter + "-"

    # Ineligible -> INELIGIBLE_LABEL
    final.loc[~eligible] = INELIGIBLE_LABEL

    out[out_col] = final
    return out[[*group_cols, "sf_code", out_col]]

def _format_duration(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def main(pattern: str = FILE_GLOB, output_csv: str = OUTPUT_CSV) -> None:
    t0 = time.perf_counter()

    df = load_input_frames(pattern)
    ensure_columns(df, REQUIRED_COLUMNS)

    # Normalize numeric types
    df["comment_count"] = to_numeric(df["comment_count"], 0).clip(lower=0)
    df["rating"] = to_numeric(df["rating"], 0)

    # SF rule-based grade (with one-step level-up)
    sf = apply_rules_with_levelup(
        df,
        ["city_id", "business_line"],
        out_col="sf_grade",
    )

    # Merge sf_grade back into the original columns
    out = df.merge(sf[["sf_code", "sf_grade"]], on="sf_code", how="left")

    # Write back to the same CSV by default (adds sf_grade column)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved → {output_csv}")

    # Quick summary
    total = len(out)
    not_enough = (out["sf_grade"] == INELIGIBLE_LABEL).sum()
    print(f"\nTotal vendors: {total}")
    print(f"SF '{INELIGIBLE_LABEL}': {not_enough} ({not_enough/total:.1%})")

    vc = out["sf_grade"].value_counts(dropna=False).sort_index()
    print(f"\nsf_grade distribution:\n{vc}")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.3f}s ({_format_duration(elapsed)})")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        pat = sys.argv[1]
    else:
        pat = FILE_GLOB
    if len(sys.argv) >= 3:
        out_csv = sys.argv[2]
    else:
        out_csv = OUTPUT_CSV
    main(pat, out_csv)

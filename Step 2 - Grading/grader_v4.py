# grade_vendors_rules_levelup_algo_min250.py
# - BL grade (city_id × business_line): fixed bands + one-step level-up for top 25% by comments,
#   '-' for bottom 25%, and 'Not Enough Rate' if comments < 400.
# - MA grade (city_id × business_line × marketing_area): same rules.
# - grade_algo (within city_id × business_line): composite A..F by percentiles, but vendors
#   with comment_count ≤ 250 are 'Not Enough Rate' and excluded from the ranking pool.

from __future__ import annotations
import sys
import glob
from typing import List
import numpy as np
import pandas as pd

# -------------------- Config --------------------
FILE_GLOB = "V_sf_vendor_RENEWED_*.csv"
OUTPUT_CSV = "sf_vendors_graded.csv"

# Composite knobs (used only for grade_algo)
K_PRIOR = 25
HIGH_THRESH = 8.9
LOW_THRESH  = 7.75
BOOST_FACTOR = 1.10
PENALTY_FACTOR = 0.85

REQUIRED_COLUMNS = {
    "sf_code", "city_id", "business_line", "marketing_area",
    "comment_count", "rating"
}

# -------------------- IO helpers --------------------
def load_input_frames(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out

def ensure_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default)

# -------------------- Your hard rules (one-step level-up) --------------------
def base_letter_from_rating(r: float) -> str:
    """
    > 9.0 -> A
    (8.5, 9.0] -> B
    (8.0, 8.5] -> C
    (7.0, 8.0] -> D
    (6.0, 7.0] -> E
    else -> F
    """
    if pd.isna(r):
        return "F"
    if r > 9.0:                 return "A"
    if 8.4 < r <= 9.0:          return "B"
    if 7.9 < r <= 8.4:          return "C"
    if 7.0 < r <= 7.9:          return "D"
    if 6.0 < r <= 7.0:          return "E"
    return "F"

LEVEL_UP = {"E": "D", "D": "C", "C": "B", "B": "A"}  # A handled separately (A -> A+)

def apply_rules_with_levelup(
    df: pd.DataFrame,
    group_cols: List[str],
    out_col: str,
    min_comments: int = 200
) -> pd.DataFrame:
    """
    - If comment_count < min_comments -> 'Not Enough Rate'
    - Else:
        Base letter by rating bands (A..F).
        Top 25% by comments (among eligible rows in the group):
          A  -> A+   ;  B -> A ; C -> B ; D -> C ; E -> D ; F -> F
        Bottom 25% by comments:
          A..E -> letter + '-' ; F unchanged
        Middle -> base letter unchanged
    Uses a frozen 'base' snapshot to avoid cascading upgrades.
    """
    out = df.copy()
    out["comment_count"] = to_numeric(out["comment_count"], 0).clip(lower=0)
    out["rating"] = to_numeric(out["rating"], 0)

    eligible = out["comment_count"] >= min_comments

    # Base letters (snapshot)
    base = out["rating"].apply(base_letter_from_rating)

    # Percentile ranks of comments among eligible rows within each group
    comment_rank = pd.Series(np.nan, index=out.index)
    if eligible.any():
        comment_rank.loc[eligible] = (
            out[eligible].groupby(group_cols, observed=True)["comment_count"]
               .transform(lambda s: s.rank(pct=True, method="average"))
        )

    final = base.copy()

    # Quartile masks (only eligible)
    top_q = eligible & (comment_rank >= 0.75)
    bot_q = eligible & (comment_rank <= 0.25)

    # Top quartile: A -> A+ ; B/C/D/E -> one step up ; F unchanged
    final.loc[top_q & (base == "A")] = "A+"
    for src, dst in LEVEL_UP.items():
        mask = top_q & (base == src)
        final.loc[mask] = dst

    # Bottom quartile: A..E -> add '-' ; F unchanged
    for letter in list("ABCDE"):
        mask = bot_q & (base == letter)
        final.loc[mask] = letter + "-"

    # Ineligible -> Not Enough Rate
    final.loc[~eligible] = "Not Enough Rate"

    out[out_col] = final
    return out[[*group_cols, "sf_code", out_col]]

# -------------------- My composite (grade_algo) within BL groups --------------------
def minmax_by_group(df: pd.DataFrame, group_cols: List[str], src_col: str) -> pd.Series:
    def _scale(s: pd.Series) -> pd.Series:
        smin, smax = s.min(), s.max()
        if pd.isna(smin) or pd.isna(smax) or smax == smin:
            return pd.Series(0.0, index=s.index)
        return (s - smin) / (smax - smin)
    return df.groupby(group_cols, observed=True)[src_col].transform(_scale)

def build_algo_score(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Composite score per BL group:
      - Bayesian-adjusted rating toward group mean with prior K_PRIOR
      - Small global boost/penalty at thresholds
      - Comment feature: log1p -> logistic
      - Normalize both within group
      - Adaptive weighting by group std
    """
    out = df.copy()
    out["comment_count"] = to_numeric(out["comment_count"], 0).clip(lower=0)
    out["rating"] = to_numeric(out["rating"], 0).clip(lower=0)

    group_mean = out.groupby(group_cols, observed=True)["rating"].transform("mean").fillna(0.0)
    denom = out["comment_count"] + K_PRIOR
    out["rating_adj"] = ((out["rating"] * out["comment_count"]) + (group_mean * K_PRIOR)) / denom.replace(0, np.nan)
    out["rating_adj"] = out["rating_adj"].fillna(group_mean)

    weight = np.where(out["rating_adj"] > HIGH_THRESH, BOOST_FACTOR,
             np.where(out["rating_adj"] < LOW_THRESH, PENALTY_FACTOR, 1.0))
    out["rating_adj_w"] = out["rating_adj"] * weight

    out["log_comments"] = np.log1p(out["comment_count"])
    out["comment_feat"] = 1.0 / (1.0 + np.exp(-out["log_comments"]))

    out["rating_norm"]  = minmax_by_group(out, group_cols, "rating_adj_w")
    out["comment_norm"] = minmax_by_group(out, group_cols, "comment_feat")

    stds = (
        out.groupby(group_cols, observed=True)[["rating_norm", "comment_norm"]]
           .std(ddof=0)
           .replace(0, 1.0)
    )
    out = out.join(stds, on=group_cols, rsuffix="_std")
    den = (out["rating_norm_std"] + out["comment_norm_std"]).replace(0, np.nan)
    out["score"] = (
        (out["rating_norm"]  * out["rating_norm_std"]) +
        (out["comment_norm"] * out["comment_norm_std"])
    ) / den
    out["score"] = out["score"].fillna(0.5)

    return out

def algo_grade_within_bl(df: pd.DataFrame) -> pd.DataFrame:
    """
    grade_algo within (city_id × business_line):
      - Vendors with comment_count ≤ 250 -> 'Not Enough Rate'
      - Eligible vendors are ranked by composite score percentiles (A..F)
        computed ONLY among eligible vendors in each group.
    """
    group_cols = ["city_id", "business_line"]
    comp = build_algo_score(df, group_cols=group_cols)

    # Eligibility for algo: strictly greater than 250
    eligible = comp["comment_count"] > 250

    # Compute percentile ranks only on eligible rows within groups
    ranks = pd.Series(np.nan, index=comp.index)
    if eligible.any():
        ranks.loc[eligible] = (
            comp.loc[eligible].groupby(group_cols, observed=True)["score"]
                .transform(lambda s: s.rank(pct=True, method="average"))
        )

    # Bin eligible ranks into A..F
    labels = ["F","E","D","C","B","A"]
    bins = np.linspace(0.0, 1.0, 7)
    graded = pd.Series("Not Enough Rate", index=comp.index, dtype=object)
    elig_bins = pd.cut(ranks.loc[eligible], bins=bins, labels=labels,
                       include_lowest=True, right=True)
    graded.loc[eligible] = elig_bins.astype(object)

    return pd.DataFrame({"sf_code": comp["sf_code"].values, "grade_algo": graded.values})

# -------------------- Main --------------------
def main(pattern: str = FILE_GLOB, output_csv: str = OUTPUT_CSV) -> None:
    df = load_input_frames("data/V_sf_vendor_RENEWED_*.csv")
    ensure_columns(df, REQUIRED_COLUMNS)

    df["comment_count"] = to_numeric(df["comment_count"], 0).clip(lower=0)
    df["rating"] = to_numeric(df["rating"], 0)

    # BL rule-based grade (with one-step level-up; min comments 400)
    bl = apply_rules_with_levelup(df, ["city_id", "business_line"], out_col="bl_grade", min_comments=200)

    # MA rule-based grade (with one-step level-up; min comments 400)
    ma = apply_rules_with_levelup(df, ["city_id", "business_line", "marketing_area"], out_col="ma_grade", min_comments=200)

    # Algo grade (with min comments 250; excluded from ranking if not eligible)
    algo = algo_grade_within_bl(df)

    out = df.merge(bl[["sf_code","bl_grade"]], on="sf_code", how="left") \
            .merge(ma[["sf_code","ma_grade"]], on="sf_code", how="left") \
            .merge(algo, on="sf_code", how="left")

    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved → {output_csv}")

    total = len(out)
    bl_not = (out["bl_grade"] == "Not Enough Rate").sum()
    ma_not = (out["ma_grade"] == "Not Enough Rate").sum()
    algo_not = (out["grade_algo"] == "Not Enough Rate").sum()
    print(f"\nTotal vendors: {total}")
    print(f"BL 'Not Enough Rate':  {bl_not} ({bl_not/total:.1%})")
    print(f"MA 'Not Enough Rate':  {ma_not} ({ma_not/total:.1%})")
    print(f"Algo 'Not Enough Rate': {algo_not} ({algo_not/total:.1%})")

    for col in ["bl_grade","ma_grade","grade_algo"]:
        vc = out[col].value_counts(dropna=False).sort_index()
        print(f"\n{col} distribution:\n{vc}")

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

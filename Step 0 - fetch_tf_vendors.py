# Step 0 -fetch_tf_vendors.py
"""
fetch_tf_vendors.py — brief overview
------------------------------------
What it does:
- Fetches TF vendor data from Metabase (question_id=6257) via `mini.fetch_question_data`.
- Normalizes the result to the exact EXPECTED_COLS CSV schema.
- Cleans placeholder/invalid dates (e.g., '1970-01-01*' -> '0') and fills selected numeric fields with 0.
- Re-assigns `marketing_area` using point-in-polygon (Shapely STRtree) from WKT polygons.
- Saves the final dataset to `outputs/tf_vendors.csv`.

Inputs:
- Polygon files: CSVs in `data/polygons/*_polygons.csv` with columns: `WKT`, `name`.

Outputs:
- `outputs/tf_vendors.csv` (UTF-8 with BOM), columns in EXACT order defined by EXPECTED_COLS.
"""
from pathlib import Path
import sys
import numbers
import numpy as np
import pandas as pd
from mini import fetch_question_data 

# ---- Expected final CSV schema (exact order) ----
EXPECTED_COLS = [
    "prelinked_sf_code", "tf_code","tf_name", "business_line", "tf_latitude", "tf_longitude",
    "city_id", "marketing_area", "own_delivery", "ofood_delivery",
    "available_H", "shift_H", "availability", "vendor_fault", "vms_percent",
    "nfc_percent", "osr", "avg_biker_wait_time", "zero_orders", "status_id",
    "vendor_status", "own_delivery_rate", "ofood_delivery_rate",
    "rate_count_vendor", "average_rate_order", "total_net_order_all_time",
    "first_day_live", "first_order_datetime", "past_30_days_net_orders",
]

# ---- Columns to fill with zeros if missing/NaN (per your rules) ----
FILL_ZERO_COLS = [
    "own_delivery_rate", "ofood_delivery_rate",
    "rate_count_vendor", "average_rate_order",
    "total_net_order_all_time",
]

DATE_COLS = ["first_day_live", "first_order_datetime"]


# ---------------- Polygon utilities (robust to Shapely returning arrays of indices OR geometries) ----------------
def load_polygons(polygons_dir: Path):
    """
    Load polygons from all *_polygons.csv in polygons_dir. Each CSV must have: WKT, name
    Returns (tree, geoms, names, name_by_id) or (None, [], [], {}) if Shapely missing / no files.
    """
    try:
        from shapely import wkt
        from shapely.strtree import STRtree
    except Exception:
        print("⚠️ shapely not available. Install with: pip install shapely", file=sys.stderr)
        return None, [], [], {}

    files = sorted(polygons_dir.glob("*_polygons.csv"))
    if len(files) == 0:
        print(f"⚠️ No polygon files found in {polygons_dir}", file=sys.stderr)
        return None, [], [], {}

    geoms, names = [], []
    for f in files:
        try:
            dfp = pd.read_csv(f)
        except Exception as e:
            print(f"⚠️ Failed to read {f}: {e}", file=sys.stderr)
            continue
        if not {"WKT", "name"}.issubset(dfp.columns):
            print(f"⚠️ Skipping {f.name}: missing WKT/name columns", file=sys.stderr)
            continue
        for _, r in dfp.iterrows():
            try:
                geom = wkt.loads(r["WKT"])
                geoms.append(geom)
                names.append(str(r["name"]))
            except Exception:
                continue

    if len(geoms) == 0:
        print("⚠️ No valid polygons loaded.", file=sys.stderr)
        return None, [], [], {}

    tree = STRtree(geoms)
    name_by_id = {id(g): n for g, n in zip(geoms, names)}
    return tree, geoms, names, name_by_id


def _normalize_to_list(arr):
    """
    Normalize STRtree.query(...) return to a Python list.
    Handles: None, single object, list/tuple, numpy arrays, shapely arrays.
    """
    if arr is None:
        return []
    try:
        arr_np = np.asarray(arr)
        if arr_np.size == 0:
            return []
        return arr_np.ravel().tolist()
    except Exception:
        pass
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return [arr]


def assign_marketing_area(df: pd.DataFrame, polygons_dir: Path) -> pd.DataFrame:
    """
    Overwrite df['marketing_area'] using point-in-polygon with tf_longitude/tf_latitude.
    Handles STRtree.query returning either geometries OR integer indices (version-dependent).
    """
    try:
        from shapely.geometry import Point
        from shapely.geometry.base import BaseGeometry
    except Exception:
        return df  # shapely missing; already warned in loader

    tree, geoms, names, name_by_id = load_polygons(polygons_dir)
    if tree is None:
        return df

    def _candidates(pt):
        try:
            try:
                cand = tree.query(pt, predicate="intersects")  # Shapely 2.x (preferred)
            except TypeError:
                cand = tree.query(pt)  # Fallback (older builds)
        except Exception:
            return []

        cand_list = _normalize_to_list(cand)
        if len(cand_list) == 0:
            return []

        first = cand_list[0]
        out = []
        # Case A: indices (pygeos / some builds)
        if isinstance(first, numbers.Integral) or isinstance(first, np.integer):
            for idx in cand_list:
                ii = int(np.asarray(idx).item()) if hasattr(idx, "__array__") else int(idx)
                poly = geoms[ii]
                out.append((poly, names[ii]))
            return out
        # Case B: geometries
        for poly in cand_list:
            if isinstance(poly, BaseGeometry):
                out.append((poly, name_by_id.get(id(poly))))
        return out

    def _lookup_area(lon, lat, current):
        if pd.isna(lon) or pd.isna(lat):
            return current
        try:
            x = float(lon)
            y = float(lat)
        except Exception:
            return current

        pt = Point(x, y)  # (x=lon, y=lat)
        for poly, area_name in _candidates(pt):
            try:
                if poly.intersects(pt):  # includes boundary
                    return area_name if area_name is not None else current
            except Exception:
                continue
        return current

    df["marketing_area"] = [
        _lookup_area(lon, lat, cur)
        for lon, lat, cur in zip(df.get("tf_longitude"), df.get("tf_latitude"), df.get("marketing_area"))
    ]
    return df


# ---------------- Cleaning utilities ----------------
def sanitize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For DATE_COLS, replace any value starting with '1970-01-01' with '0'.
    Also ensure empty/NaT/None/nan become '0'.
    """
    for col in DATE_COLS:
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        s = s.where(~s.str.startswith("1970-01-01", na=False), other="0")
        s = s.replace({"NaT": "0", "None": "0", "nan": "0", "": "0"})
        df[col] = s
    return df


def fill_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    For FILL_ZERO_COLS, coerce to numeric and fill missing/invalid with 0.
    """
    for col in FILL_ZERO_COLS:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def ensure_expected_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map current-output-style columns to the EXPECTED schema:
      * sf_code         -> prelinked_sf_code  (if expected col missing)
      * chain_fault     -> vendor_fault       (if expected col missing)
    Ensures all EXPECTED_COLS exist, returns df in exact EXPECTED_COLS order.
    """
    rename_map = {}

    # Prefer whatever already matches the expected names.
    if "prelinked_sf_code" not in df.columns and "sf_code" in df.columns:
        rename_map["sf_code"] = "prelinked_sf_code"
    if "vendor_fault" not in df.columns and "chain_fault" in df.columns:
        rename_map["chain_fault"] = "vendor_fault"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Create any missing expected columns
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # Reorder to exact expected schema
    return df[EXPECTED_COLS]


# ---------------- Main ----------------
def main():
    tf_df = fetch_question_data(
        question_id=6257,
        metabase_url="https://metabase.ofood.cloud",
        username="a.mehmandoost@OFOOD.CLOUD",
        password="Fff322666@",
        team="data",
        workers=8,
        page_size=50000,
    )
    if tf_df is None:
        raise SystemExit("Failed to fetch data from Metabase (returned None).")

    # 1) Convert schema to exactly the expected download format
    tf_df = ensure_expected_schema(tf_df)

    # 2) Fix placeholder epoch dates -> "0"
    tf_df = sanitize_dates(tf_df)

    # 3) Fill specified numeric metrics with zeros where null/none
    tf_df = fill_zeros(tf_df)

    # 4) Re-assign marketing_area using polygons WKT
    base_dir = Path(__file__).resolve().parent
    polygons_dir = base_dir / "data" / "polygons"
    tf_df = assign_marketing_area(tf_df, polygons_dir)

    # 5) Save
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tf_vendors.csv"
    tf_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ Saved {len(tf_df):,} rows to {out_path}")
    print(f"ℹ️ Columns: {', '.join(tf_df.columns)}")


if __name__ == "__main__":
    main()

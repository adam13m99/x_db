# scrape.py
# Run: python scrape.py
# Requirements: requests, pandas, numpy, shapely

from __future__ import annotations

import math
import time
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from shapely import wkt

# ───────────────────────── Hardcoded settings ─────────────────────────

SF_CODE = "gnzz6x"  # or whatever you test with
CSV_PATH = Path("data/sf_vendors.csv")        # ← was "sf_vendors.csv"
POLYGONS_DIR = Path("data/polygons")    # polygons folder in same directory

# ───────────────────────── API config ─────────────────────────

DETAIL_API_URL = "https://snappfood.ir/mobile/v2/restaurant/details/dynamic"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/107.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

COMMON_PARAMS = {
    "optionalClient": "WEBSITE",
    "client": "WEBSITE",
    "deviceType": "WEBSITE",
    "appVersion": "8.1.1",
    "UDID": "",
    "locale": "fa",
}

# Supported cities (for probing + city assignment)
CITY_ID_MAP = {"tehran": 2, "mashhad": 1, "shiraz": 5}
CITY_ID_TO_NAME = {2: "tehran", 1: "mashhad", 5: "shiraz"}

CITY_COORDS: Dict[int, Tuple[float, float]] = {
    2: (35.6892, 51.3890),  # Tehran
    1: (36.2605, 59.6168),  # Mashhad
    5: (29.5918, 52.5836),  # Shiraz
}

# Multiple probe points per city to resolve location-dependent API responses
CITY_MULTIPLE_COORDS = {
    2: [  # Tehran
        (35.6892, 51.3890), (35.7219, 51.3347), (35.6961, 51.4231),
        (35.6515, 51.3680), (35.7058, 51.3570), (35.7297, 51.4015),
        (35.6736, 51.3185), (35.6403, 51.4180), (35.7456, 51.3750),
        (35.6234, 51.3456),
    ],
    1: [  # Mashhad
        (36.2605, 59.6168), (36.2297, 59.5657), (36.2915, 59.6543),
        (36.2456, 59.6789), (36.2123, 59.5987), (36.3012, 59.6234),
        (36.2789, 59.7012), (36.1987, 59.6456),
    ],
    5: [  # Shiraz
        (29.5918, 52.5836), (29.6234, 52.5456), (29.5654, 52.6123),
        (29.5543, 52.5234), (29.6012, 52.5678), (29.6345, 52.5987),
        (29.5876, 52.4987), (29.5432, 52.6345),
    ],
}

BUSINESS_LINE_MAP = {
    "RESTAURANT": "Restaurant",
    "CAFFE": "Cafe",
    "CONFECTIONERY": "Pastry",
    "BAKERY": "Bakery",
    "GROCERY": "Fruit Shop",
    "PROTEIN": "Meat Shop",
    "JUICE": "Ice Cream and Juice Shop",
}

# ───────────────────────── Logging ─────────────────────────

def get_logger() -> logging.Logger:
    log = logging.getLogger("SFAddOne")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log

logger = get_logger()

# ───────────────────────── Utils ─────────────────────────

def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default)

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def fmt(v):
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        return f"{v:.4g}"
    return str(v)

# ───────────────────────── Parsers ─────────────────────────

def extract_business_line(v: dict) -> Optional[str]:
    """
    Try to normalize vendor category to a business line using BUSINESS_LINE_MAP.
    Looks at multiple possible fields (varies across API payloads).
    """
    candidates: List[Optional[str]] = []
    mc = v.get("mainCategory")
    if isinstance(mc, dict):
        candidates.extend([mc.get("alias"), mc.get("nameEn"), mc.get("name")])
    elif isinstance(mc, str):
        candidates.append(mc)

    candidates.extend([
        v.get("superTypeAlias"),
        v.get("vendorType"),
        v.get("childType"),
        v.get("category"),
    ])

    for c in candidates:
        if not c:
            continue
        key = str(c).strip().upper()
        if key in BUSINESS_LINE_MAP:
            return BUSINESS_LINE_MAP[key]
    return None

def extract_is_express(v: dict) -> Optional[bool]:
    for k in ("is_express", "isExpress"):
        if k in v and v[k] is not None:
            return bool(v[k])

    exp = v.get("express")
    if isinstance(exp, dict):
        for k in ("is_express", "isExpress", "enabled"):
            if k in exp and exp[k] is not None:
                return bool(exp[k])

    vend = v.get("vendor")
    if isinstance(vend, dict):
        for k in ("is_express", "isExpress"):
            if k in vend and vend[k] is not None:
                return bool(vend[k])

    return None

def extract_is_zf_express(v: dict) -> Optional[bool]:
    candidates = [v, v.get("express", {}), v.get("vendor", {})]
    keys = ("isZFExpress", "isZfExpress", "is_zf_express", "zfIsExpress")
    for d in candidates:
        if not isinstance(d, dict):
            continue
        for k in keys:
            if k in d and d[k] is not None:
                return bool(d[k])
    return None

def get_is_express_from_vendor(v: dict) -> int:
    """
    STRICT PRIORITY:
      1) Normalized (vendor-centered) response: v['_normalized'] → isZFExpress / legacy
      2) First response: isZFExpress / legacy
    Returns 0/1 integer.
    """
    # 1) Try normalized payload first
    v2 = v.get("_normalized") or {}
    for src in (v2,):
        zf = extract_is_zf_express(src)
        if zf is not None:
            return int(bool(zf))
        legacy = extract_is_express(src)
        if legacy is not None:
            return int(bool(legacy))

    # 2) Fall back to initial payload
    for src in (v,):
        zf = extract_is_zf_express(src)
        if zf is not None:
            return int(bool(zf))
        legacy = extract_is_express(src)
        if legacy is not None:
            return int(bool(legacy))

    return 0

# ───────────────────────── Polygons ─────────────────────────

def load_polygons(polygons_dir: Path) -> pd.DataFrame:
    files = {
        2: polygons_dir / "tehran_polygons.csv",
        1: polygons_dir / "mashhad_polygons.csv",
        5: polygons_dir / "shiraz_polygons.csv",
    }
    frames = []
    for cid, p in files.items():
        if not p.exists():
            logger.warning(f"Polygon file not found: {p}")
            continue
        df = pd.read_csv(p)
        if "WKT" not in df.columns or "name" not in df.columns:
            logger.warning(f"Polygon file missing columns (name,WKT): {p}")
            continue
        df = df[["name", "WKT"]].copy()
        df["city_id"] = cid
        df["geometry"] = df["WKT"].apply(wkt.loads)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def point_to_marketing_area(lat: float, lon: float, city_id: int | float, polygons_df: pd.DataFrame) -> str:
    if polygons_df.empty or pd.isna(lat) or pd.isna(lon) or pd.isna(city_id):
        return "Unknown"
    pt = Point(float(lon), float(lat))
    subset = polygons_df[polygons_df["city_id"] == int(city_id)]
    for _, row in subset.iterrows():
        if row["geometry"].contains(pt):
            return str(row["name"])
    return "Outside Defined Areas"

def city_id_from_persian_or_coords(city_persian: Optional[str], lat: Optional[float], lon: Optional[float]) -> Optional[int]:
    if isinstance(city_persian, str):
        pmap = {"تهران": 2, "مشهد": 1, "شیراز": 5}
        c = city_persian.strip()
        if c in pmap:
            return pmap[c]
    try:
        if not (pd.isna(lat) or pd.isna(lon)):
            return min(CITY_COORDS, key=lambda k: haversine(float(lat), float(lon), *CITY_COORDS[k]))
    except Exception:
        return None
    return None

# ───────────────────────── Scrape vendor ─────────────────────────

def fetch_vendor(sf_code: str, max_retries: int = 10, timeout: int = 10) -> dict:
    """
    Resolve a Snappfood vendor by probing across city coordinates to get a
    stable payload. Then re-call the details API at the vendor's own lat/lon
    and store that normalized payload under key '_normalized'.

    Returns: dict with original fields + (optional) '_normalized' key.
    Raises: RuntimeError if resolution fails.
    """
    sess = requests.Session()
    sess.headers.update(HEADERS)
    headers = {"Referer": "https://snappfood.ir/"}

    candidates = [2, 1, 5]  # tehran, mashhad, shiraz

    for attempt in range(1, max_retries + 1):
        for cid in candidates:
            for (lat, lon) in CITY_MULTIPLE_COORDS[cid]:
                try:
                    params = {**COMMON_PARAMS, "lat": str(lat), "long": str(lon), "vendorCode": sf_code}
                    resp = sess.get(DETAIL_API_URL, params=params, headers=headers, timeout=timeout)
                    resp.raise_for_status()
                    vendor = (resp.json() or {}).get("data", {}).get("vendor", {}) or {}
                    if not vendor.get("vendorCode"):
                        continue

                    # second call at vendor coords to normalize location-dependent fields
                    v_lat, v_lon = vendor.get("lat"), vendor.get("lon")
                    if v_lat and v_lon:
                        try:
                            params2 = {
                                **COMMON_PARAMS,
                                "lat": str(v_lat),
                                "long": str(v_lon),
                                "vendorCode": sf_code,
                                "locationCacheKey": f"lat={v_lat}&long={v_lon}",
                                "show_party": 1,
                                "fetch-static-data": 1,
                            }
                            resp2 = sess.get(DETAIL_API_URL, params=params2, headers=headers, timeout=timeout)
                            resp2.raise_for_status()
                            v2 = (resp2.json() or {}).get("data", {}).get("vendor", {}) or {}
                            vendor["_normalized"] = v2
                        except Exception as e:
                            logger.warning(f"Vendor-centered recheck failed for {sf_code}: {e}")

                    logger.info(f"Resolved vendor {sf_code} on attempt {attempt}.")
                    return vendor

                except requests.HTTPError:
                    pass
                except Exception as e:
                    logger.warning(f"Probe error ({lat},{lon}) attempt {attempt}: {e}")

        time.sleep(0.6)  # small backoff between attempts

    raise RuntimeError(f"Could not resolve vendor '{sf_code}' via details endpoint.")

# ───────────────────────── Grading (level-up rules) ─────────────────────────

def base_letter_from_rating(r: float) -> str:
    if pd.isna(r):
        return "F"
    if r > 9.0:
        return "A"
    if 8.4 < r <= 9.0:
        return "B"
    if 7.9 < r <= 8.4:
        return "C"
    if 7.0 < r <= 7.9:
        return "D"
    if 6.0 < r <= 7.0:
        return "E"
    return "F"

def compute_levelup_grade_for_vendor(
    df_full: pd.DataFrame,
    vendor_row: pd.Series,
    group_cols: List[str],
    min_comments: int = 250
) -> str:
    # must have group keys
    if any(pd.isna(vendor_row.get(c)) for c in group_cols):
        return "Not Enough Rate"

    cc = pd.to_numeric(vendor_row.get("comment_count"), errors="coerce")
    if pd.isna(cc) or cc < min_comments:
        return "Not Enough Rate"

    base = base_letter_from_rating(pd.to_numeric(vendor_row.get("rating"), errors="coerce"))

    # group pool mask
    mask = np.ones(len(df_full), dtype=bool)
    for c in group_cols:
        mask &= (df_full[c] == vendor_row[c])

    pool = df_full.loc[mask, ["comment_count"]].copy()
    pool["comment_count"] = to_numeric(pool["comment_count"], 0)

    # ensure vendor included
    pool = pd.concat([pool, pd.DataFrame({"comment_count": [cc]})], ignore_index=True)

    elig = pool["comment_count"] >= min_comments
    if not elig.any():
        return base

    ranks = pool.loc[elig, "comment_count"].rank(pct=True, method="average")
    v_rank = ranks.iloc[-1]  # last appended row is vendor

    if v_rank >= 0.75:
        if base == "A":
            return "A+"
        return {"B": "A", "C": "B", "D": "C", "E": "D"}.get(base, base)

    if v_rank <= 0.25 and base in list("ABCDE"):
        return base + "-"

    return base

# ───────────────────────── Change logging ─────────────────────────

def log_field_changes(old_row: Optional[pd.Series], new_row: dict, context: str):
    """
    Logs only changed fields (old -> new). If old_row is None, logs inserted fields.
    """
    tracked = [
        "sf_name","city_name","city_id","marketing_area","business_line",
        "sf_latitude","sf_longitude","min_order","chain_title",
        "comment_count","rating","is_express","bl_grade","ma_grade"
    ]
    if old_row is None:
        logger.info(f"{context} Inserted fields:")
        msg = "; ".join([f"{k}={fmt(new_row.get(k))}" for k in tracked if k in new_row])
        logger.info(f"{context} {msg}")
        return

    changes = []
    for k in tracked:
        old_v = old_row.get(k) if k in old_row else None
        new_v = new_row.get(k)
        same = (pd.isna(old_v) and pd.isna(new_v)) or (old_v == new_v)
        if not same:
            changes.append(f"{k}: {fmt(old_v)} → {fmt(new_v)}")

    if changes:
        logger.info(f"{context} Updated fields:")
        for chunk in changes:
            logger.info(f"{context} - {chunk}")
    else:
        logger.info(f"{context} No field changes.")

# ───────────────────────── Main flow (standalone) ─────────────────────────

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH.resolve()}")

    polygons_df = load_polygons(POLYGONS_DIR)

    # 1) Validate + scrape vendor
    vendor_raw = fetch_vendor(SF_CODE)

    # 2) Parse & enrich to MATCH sf_vendors.csv schema
    # Prefer normalized payload for sensitive, location-dependent fields (esp. is_express)
    v_norm = vendor_raw.get("_normalized") or {}

    sf_name = vendor_raw.get("title", "")
    # rating/comment_count pulled from normalized if available
    rating = float((v_norm.get("rating") if v_norm.get("rating") is not None else vendor_raw.get("rating")) or 0.0)
    comment_count = int((v_norm.get("commentCount") if v_norm.get("commentCount") is not None else vendor_raw.get("commentCount")) or 0)

    # position itself should be stable
    sf_latitude = float(vendor_raw.get("lat") or np.nan)
    sf_longitude = float(vendor_raw.get("lon") or np.nan)

    # min order, chain
    min_order_src = v_norm if "minOrder" in v_norm else vendor_raw
    min_order = min_order_src.get("minOrder")
    try:
        min_order = int(min_order) if min_order is not None else 0
    except Exception:
        min_order = 0
    chain_title = (v_norm.get("chainTitle") if v_norm.get("chainTitle") is not None else vendor_raw.get("chainTitle"))

    # business line
    business_line = extract_business_line(v_norm if v_norm else vendor_raw)

    # is_express with STRICT vendor-centered priority
    is_express = get_is_express_from_vendor(vendor_raw)

    # city inference
    city_persian = (v_norm.get("city") if v_norm.get("city") is not None else vendor_raw.get("city"))
    cid = city_id_from_persian_or_coords(city_persian, sf_latitude, sf_longitude)
    city_id = int(cid) if cid is not None else np.nan
    city_name = CITY_ID_TO_NAME.get(city_id) if not pd.isna(city_id) else np.nan

    # marketing area via polygons
    marketing_area = point_to_marketing_area(sf_latitude, sf_longitude, city_id, polygons_df)

    # 3) Load CSV and add/update row
    df = pd.read_csv(CSV_PATH)

    # Ensure schema columns exist
    all_cols = [
        "sf_code","sf_name","city_name","city_id","marketing_area","business_line",
        "sf_latitude","sf_longitude","min_order","chain_title","comment_count",
        "rating","is_express","bl_grade","ma_grade","grade_algo"
    ]
    for c in all_cols:
        if c not in df.columns:
            df[c] = np.nan

    new_row = {
        "sf_code": SF_CODE,
        "sf_name": sf_name,
        "city_name": city_name,
        "city_id": city_id,
        "marketing_area": marketing_area,
        "business_line": business_line,
        "sf_latitude": sf_latitude,
        "sf_longitude": sf_longitude,
        "min_order": min_order,
        "chain_title": chain_title,
        "comment_count": comment_count,
        "rating": rating,
        "is_express": is_express,  # grades will be set below
        "bl_grade": np.nan,
        "ma_grade": np.nan,
        "grade_algo": np.nan,      # untouched
    }

    idx = df.index[df["sf_code"].astype(str).str.strip() == SF_CODE]
    if len(idx) > 0:
        logger.info(f"Vendor {SF_CODE} exists — updating.")
        target_idx = idx[0]
        old_row = df.loc[target_idx].copy()
        # fill row (without grades first)
        for k, v in new_row.items():
            df.loc[target_idx, k] = v
    else:
        logger.info(f"Appending vendor {SF_CODE}.")
        old_row = None
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        target_idx = df.index[-1]

    # 4) Compute bl_grade & ma_grade for THIS vendor only (min comments = 250)
    vrow = df.loc[target_idx].copy()
    bl_grade = compute_levelup_grade_for_vendor(df, vrow, ["city_id", "business_line"], min_comments=250)
    ma_grade = compute_levelup_grade_for_vendor(df, vrow, ["city_id", "business_line", "marketing_area"], min_comments=250)

    df.loc[target_idx, "bl_grade"] = bl_grade
    df.loc[target_idx, "ma_grade"] = ma_grade

    # Prepare final row (with grades) for change logging
    final_row = df.loc[target_idx].to_dict()
    final_row["bl_grade"] = bl_grade
    final_row["ma_grade"] = ma_grade

    # 5) Daily backup + overwrite CSV
    day_stamp = pd.Timestamp.now().strftime("%Y%m%d")
    backup = CSV_PATH.with_name(f"{CSV_PATH.stem}_backup_{day_stamp}.csv")
    df.to_csv(backup, index=False, encoding="utf-8-sig")
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    # 6) Log changes summary
    logger.info("────────────────────────────────────────────────")
    logger.info(f"✅ Vendor '{SF_CODE}' processed.")
    log_field_changes(old_row, final_row, context="↻")
    logger.info(f"↻ Grades → BL: {bl_grade} | MA: {ma_grade}")
    logger.info(f"↻ Saved: {CSV_PATH.name} | Backup: {backup.name}")
    logger.info("────────────────────────────────────────────────")

# ───────────────────────── Module entry ─────────────────────────

if __name__ == "__main__":
    main()

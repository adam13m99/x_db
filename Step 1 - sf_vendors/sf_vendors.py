# M_snappfood_vendor_scrape.py - FIXED FOR RELIABLE is_express & business_line + EXTRA sf_codes support (NO GRADING)
"""
--------------------------------------------------------------------------------
-- VENDOR DATA RENEWAL PIPELINE (FIXED, NO GRADING) --
--------------------------------------------------------------------------------
This script performs a two-stage process and **fixes location-biased is_express**.
It also supports **adding extra vendors from sf_code-only CSVs** and probing
**all cities** when a vendor's city is unknown.

1) AGGREGATE
   - Finds local CSVs matching 'V_sf_vendor_dual_grading_*.csv',
     'V_sf_vendor_scrape_*.csv', and optional 'extra_sf_codes*.csv'.
   - Combines & dedupes to create a master list of 'sf_code' (+ city_name if any).

2) RENEW & SCRAPE (Location-normalized)
   - Scrapes vendor details from SnappFood API using robust multi-coordinate
     probing. If city is unknown, probes **all configured cities**.
   - **Then re-calls the details endpoint using the vendor's own lat/lon** to
     neutralize location-dependent fields (notably isZFExpress).
   - Extracts business_line from details (multiple candidate fields) and
     **is_express from vendor-centered `isZFExpress`** (falling back to legacy fields only if needed).
   - Builds a fallback index from the list API to fill missing
     business_line and (optionally) is_express when not resolvable.
   - Enriches with city_id/city_name and assigns marketing areas (Shapely polygons).
   - Saves final, renewed dataset to timestamped CSV. (All grading logic removed.)
--------------------------------------------------------------------------------
"""

import glob
import math
import os
import threading
import time
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np

# This script requires the 'shapely' library. Install it using: pip install shapely
try:
    from shapely.geometry import Point
    from shapely import wkt
except ImportError:
    print("Error: The 'shapely' library is required. Please install it using: pip install shapely")
    raise

# ─── Configuration ───────────────────────────────────────────────────────────

LIST_API_URL   = "https://snappfood.ir/search/api/v1/desktop/vendors-list"
DETAIL_API_URL = "https://snappfood.ir/mobile/v2/restaurant/details/dynamic"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/107.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
COMMON_PARAMS = {
    "optionalClient": "WEBSITE",
    "client":         "WEBSITE",
    "deviceType":     "WEBSITE",
    "appVersion":     "8.1.1",
    "UDID":           "",
}

# Page sizes/workers
PAGE_SIZE_LIST      = 100   # for list indexing fallback
LIST_MAX_WORKERS    = 20
DETAIL_MAX_WORKERS  = 20
DETAIL_MAX_RETRIES  = 10

BUSINESS_LINE_MAP = {
    "RESTAURANT":    "Restaurant",
    "CAFFE":         "Cafe",
    "CONFECTIONERY": "Pastry",
    "BAKERY":        "Bakery",
    "GROCERY":       "Fruit Shop",
    "PROTEIN":       "Meat Shop",
    "JUICE":         "Ice Cream and Juice Shop",
}

# This set defines the aliases we will query from the list endpoint
BUSINESS_ALIASES = list(BUSINESS_LINE_MAP.keys())

CITIES      = ["tehran", "mashhad", "shiraz"]
CITY_ID_MAP = {"tehran": 2, "mashhad": 1, "shiraz": 5}

# Single reference coordinates for city assignment
CITY_COORDS = {
    2: (35.6892, 51.3890),  # Tehran
    1: (36.2605, 59.6168),  # Mashhad
    5: (29.5918, 52.5836),  # Shiraz
}

# Multiple coordinate points for comprehensive coverage of each city (used in details probing)
CITY_MULTIPLE_COORDS = {
    2: [  # Tehran - covering different districts
        (35.6892, 51.3890), (35.7219, 51.3347), (35.6961, 51.4231),
        (35.6515, 51.3680), (35.7058, 51.3570), (35.7297, 51.4015),
        (35.6736, 51.3185), (35.6403, 51.4180), (35.7456, 51.3750),
        (35.6234, 51.3456),
    ],
    1: [  # Mashhad - covering different areas
        (36.2605, 59.6168), (36.2297, 59.5657), (36.2915, 59.6543),
        (36.2456, 59.6789), (36.2123, 59.5987), (36.3012, 59.6234),
        (36.2789, 59.7012), (36.1987, 59.6456),
    ],
    5: [  # Shiraz - covering different areas
        (29.5918, 52.5836), (29.6234, 52.5456), (29.5654, 52.6123),
        (29.5543, 52.5234), (29.6012, 52.5678), (29.6345, 52.5987),
        (29.5876, 52.4987), (29.5432, 52.6345),
    ]
}

# Additional sf_code-only CSVs (optional). Must contain a column named 'sf_code'.
# Example: extra_sf_codes.csv, extra_sf_codes_2025-08-15.csv, etc.
EXTRA_VENDOR_PATTERNS = ['data/scraped/extra_sf_codes.csv']

# Updated paths
# Base & output directories
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR  # "same directory" as the script
POLYGON_DIR = BASE_DIR / "data" / "polygons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POLYGON_FILES = {
    2: POLYGON_DIR / "tehran_polygons.csv",
    1: POLYGON_DIR / "mashhad_polygons.csv",
    5: POLYGON_DIR / "shiraz_polygons.csv",
}

_thread_local = threading.local()

# ─── logging & session ───────────────────────────────────────────────────────

def configure_logging() -> logging.Logger:
    logger = logging.getLogger("SnappFoodScraper")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def get_session(extra_headers: dict = None) -> requests.Session:
    if not hasattr(_thread_local, "session"):
        sess = requests.Session()
        sess.headers.update(HEADERS)
        _thread_local.session = sess
    if extra_headers:
        sess = _thread_local.session
        sess.headers.update(extra_headers)
    return _thread_local.session


# ─── STAGE 1: Aggregate Vendor Codes from Local Files ────────────────────────

def merge_source_vendor_files() -> pd.DataFrame:
    """
    Finds all local vendor CSVs, combines them, and returns a de-duplicated
    DataFrame ready for scraping. It preserves existing metadata like city and
    alias if available. Also supports extra files that contain only 'sf_code'.
    """
    logger = configure_logging()
    logger.info("--- STAGE 1: AGGREGATING VENDOR CODES FROM LOCAL CSVs ---")

    # Include the extra sf_code-only CSVs
    file_patterns = [
        'data/scraped/V_sf_vendor_dual_grading_*.csv',
        'data/scraped/V_sf_vendor_scrape_*.csv',  # fixed 'scapred' -> 'scraped'
        *EXTRA_VENDOR_PATTERNS,
    ]
    all_files = []
    for pattern in file_patterns:
        matched = glob.glob(pattern)
        if matched:
            logger.info(f"Pattern '{pattern}' → {len(matched)} files")
        all_files.extend(matched)

    if not all_files:
        logger.error("Fatal: No source vendor CSV files found in the current directory.")
        logger.error("Please add files matching "
                     "'V_sf_vendor_dual_grading_*.csv', 'V_sf_vendor_scrape_*.csv', "
                     "or 'extra_sf_codes*.csv' (sf_code-only).")
        raise SystemExit(1)

    logger.info(f"Found {len(all_files)} source files to combine: {', '.join(map(os.path.basename, all_files))}")

    all_dfs = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            if 'sf_code' in df.columns:
                # Keep sf_code and city_name if present
                cols = ['sf_code'] + ([c for c in ('city_name',) if c in df.columns])
                all_dfs.append(df[cols])
            else:
                logger.warning(f"Skipping '{os.path.basename(file_path)}' as it lacks 'sf_code' column.")
        except Exception as e:
            logger.error(f"Could not read or process file '{os.path.basename(file_path)}': {e}")

    if not all_dfs:
        logger.error("Fatal: None of the source files could be processed or contained 'sf_code'.")
        raise SystemExit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    initial_count = len(combined_df)
    combined_df.drop_duplicates(subset=['sf_code'], keep='first', inplace=True)
    final_count = len(combined_df)

    logger.info(f"Combined {initial_count} rows into {final_count} unique vendors based on 'sf_code'.")

    if 'city_name' not in combined_df.columns:
        logger.warning("Column 'city_name' not found in source files. The scraper will try multiple cities.")
        combined_df['city_name'] = None

    return combined_df[['sf_code', 'city_name']]


# ─── Helpers to robustly extract business_line & is_express ──────────────────

def _extract_business_line_from_detail(v: dict) -> str | None:
    """
    Try multiple fields from the details payload and map to BUSINESS_LINE_MAP values.
    """
    candidates = []

    # mainCategory can be dict or str
    mc = v.get("mainCategory")
    if isinstance(mc, dict):
        candidates.extend([mc.get("alias"), mc.get("nameEn"), mc.get("name")])
    elif isinstance(mc, str):
        candidates.append(mc)

    # Other plausible fields
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


def _extract_is_express_from_detail(v: dict) -> bool | None:
    """
    Pull is_express from any plausible spot in details payload.
    """
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


# NEW: Prefer vendor-centered isZFExpress

def _extract_is_zf_express_from_detail(v: dict) -> bool | None:
    """Extract 'isZFExpress' regardless of casing/nesting differences."""
    candidates = [v, v.get("express", {}), v.get("vendor", {})]
    keys = ("isZFExpress", "isZfExpress", "is_zf_express", "zfIsExpress")
    for d in candidates:
        if not isinstance(d, dict):
            continue
        for k in keys:
            if k in d and d[k] is not None:
                return bool(d[k])
    return None


# ─── Build a fallback index from the list endpoint ───────────────────────────

def _city_latlon_for_name(city_name: str) -> tuple[float, float]:
    """Return a representative (lat, lon) for a lowercase city name."""
    cid = CITY_ID_MAP.get(city_name.lower())
    if cid in CITY_COORDS:
        return CITY_COORDS[cid]
    # default to Tehran
    return CITY_COORDS[2]


def build_list_index_for_cities(cities: list[str]) -> pd.DataFrame:
    """
    Query the vendors-list endpoint for each (alias, city) to build a map:
    vendor_code -> is_express_list, business_line_list
    """
    logger = logging.getLogger("SnappFoodScraper")
    rows = []
    headers = {"Referer": "https://snappfood.ir/"}

    city_norm = []
    for c in cities:
        if not isinstance(c, str) or not c:
            city_norm.append("tehran")
        else:
            c = c.strip().lower()
            city_norm.append(c if c in CITY_ID_MAP else "tehran")
    cities = sorted(set(city_norm))

    for alias in BUSINESS_ALIASES:
        for city in cities:
            lat, lon = _city_latlon_for_name(city)
            page = 0
            while True:
                params = {
                    **COMMON_PARAMS,
                    "lat": f"{lat}",
                    "long": f"{lon}",
                    "page": page,
                    "page_size": PAGE_SIZE_LIST,
                    "filters": "{}",
                    "query": "",
                    "sp_alias": alias,     # IMPORTANT: alias is e.g. 'RESTAURANT'
                    "city_name": city,
                    "locale": "fa",
                }
                try:
                    sess = get_session(headers)
                    resp = sess.get(LIST_API_URL, params=params, timeout=12)
                    resp.raise_for_status()
                    data = resp.json().get("data", {})
                    items = data.get("finalResult") or []
                    if not items:
                        break

                    for entry in items:
                        d = entry.get("data", entry)
                        code = d.get("code") or d.get("vendorCode")
                        if not code:
                            continue
                        rows.append({
                            "vendor_code": code,
                            "is_express_list": d.get("is_express"),
                            "business_line_list": BUSINESS_LINE_MAP.get(alias),
                        })

                    if len(items) < PAGE_SIZE_LIST:
                        break
                    page += 1

                except Exception as e:
                    logger.error(f"List index fetch error for alias={alias} city={city} page={page}: {e}")
                    break

    if not rows:
        logger.warning("List index builder returned no rows; fallback data will be unavailable.")
        return pd.DataFrame(columns=["vendor_code", "is_express_list", "business_line_list"])

    # Consolidate duplicates per vendor (any True for is_express wins; most frequent business_line)
    df_rows = pd.DataFrame(rows)

    def _mode_or_first(s: pd.Series):
        m = s.mode()
        return m.iat[0] if not m.empty else (s.iloc[0] if len(s) else None)

    df_idx = (
        df_rows
        .groupby("vendor_code", as_index=False)
        .agg(
            is_express_list=("is_express_list", lambda s: bool(np.nanmax(pd.to_numeric(s, errors='coerce'))) if s.notna().any() else np.nan),
            business_line_list=("business_line_list", _mode_or_first)
        )
    )

    logger.info(f"List index built with {len(df_idx)} unique vendor_code rows.")
    return df_idx


# ─── STAGE 2: Vendor Detail Scraping ─────────────────────────────────────────

class VendorDetailScraper:
    def __init__(self):
        self.max_retries = DETAIL_MAX_RETRIES
        self.max_workers = DETAIL_MAX_WORKERS
        self.logger      = configure_logging()
        self.failed_vendors = []

    def _get_detail(self, code: str, city_id: int = None, city_name: str = None, patient_mode: bool = False) -> dict | None:
        """Fetch vendor details. First try to find the vendor by probing coords; if found,
        re-query using the vendor's own lat/lon to neutralize location-dependent fields (e.g., isZFExpress)."""

        # Build candidate city IDs: prefer explicit city_id/city_name; otherwise probe all known cities
        candidates: list[int] = []
        if isinstance(city_id, int) and city_id in CITY_MULTIPLE_COORDS:
            candidates.append(city_id)
        if city_name and isinstance(city_name, str):
            cid_from_name = CITY_ID_MAP.get(city_name.lower())
            if cid_from_name and cid_from_name in CITY_MULTIPLE_COORDS and cid_from_name not in candidates:
                candidates.append(cid_from_name)
        if not candidates:
            candidates = list(CITY_MULTIPLE_COORDS.keys())  # probe all cities

        headers = {"Referer": "https://snappfood.ir/"}
        timeout = 15 if patient_mode else 10
        max_retries = self.max_retries * 2 if patient_mode else self.max_retries

        for attempt in range(1, max_retries + 1):
            for cand_city_id in candidates:
                coord_points = CITY_MULTIPLE_COORDS.get(cand_city_id, CITY_MULTIPLE_COORDS[2])
                for coord_idx, (lat, lon) in enumerate(coord_points):
                    try:
                        params = {**COMMON_PARAMS, "lat": str(lat), "long": str(lon), "vendorCode": code, "locale": "fa"}
                        sess = get_session(headers)
                        resp = sess.get(DETAIL_API_URL, params=params, timeout=timeout)
                        resp.raise_for_status()
                        vendor_data = resp.json().get("data", {}).get("vendor", {})
                        if vendor_data and vendor_data.get("vendorCode"):
                            if coord_idx > 0 or len(candidates) > 1:
                                self.logger.info(
                                    f"Found vendor {code} using city {cand_city_id} point #{coord_idx+1}"
                                )

                            # ── second call: force our location to the VENDOR's coords ──
                            v_lat, v_lon = vendor_data.get("lat"), vendor_data.get("lon")
                            if v_lat and v_lon:
                                try:
                                    params2 = {
                                        **COMMON_PARAMS,
                                        "lat": str(v_lat),
                                        "long": str(v_lon),
                                        "vendorCode": code,
                                        "locationCacheKey": f"lat={v_lat}&long={v_lon}",
                                        "show_party": 1,
                                        "fetch-static-data": 1,
                                        "locale": "fa",
                                    }
                                    resp2 = sess.get(DETAIL_API_URL, params=params2, timeout=timeout)
                                    resp2.raise_for_status()
                                    v2 = resp2.json().get("data", {}).get("vendor", {}) or {}
                                    zf = _extract_is_zf_express_from_detail(v2)
                                    if zf is not None:
                                        vendor_data["__zf_express_override__"] = bool(zf)
                                except Exception as e:
                                    self.logger.warning(f"Vendor-centered recheck failed for {code}: {e}")

                            return vendor_data
                    except requests.HTTPError as e:
                        if getattr(e.response, "status_code", None) in (400, 404, 500):
                            continue
                        self.logger.error(f"HTTP error for {code} at ({lat}, {lon}), attempt {attempt}: {e}")
                    except Exception as e:
                        self.logger.error(f"Network error for {code} at ({lat}, {lon}), attempt {attempt}: {e}")
                    time.sleep(0.5)
            if attempt < max_retries:
                time.sleep(1.0)

        if not patient_mode:
            # store the originally provided city hints; may be None
            self.failed_vendors.append((code, city_id, city_name))

        self.logger.error(f"Failed to fetch details for {code} after {max_retries} attempts.")
        return None

    def _parse(self, v: dict) -> dict:
        persian_weekdays = {1: "دوشنبه", 2: "سه‌شنبه", 3: "چهارشنبه", 4: "پنجشنبه", 5: "جمعه", 6: "شنبه", 7: "یکشنبه"}
        sched_lines = []
        for s in v.get("schedules", []) or []:
            day = persian_weekdays.get(s.get("weekday"), str(s.get("weekday")))
            start, stop  = s.get("startHour", ""), s.get("stopHour", "")
            sched_lines.append(f"{day} - {start} - {stop}")

        # Robust extractions
        business_line = _extract_business_line_from_detail(v)

        # NEW: choose is_express with this priority:
        # 1) override from vendor-centered call (normalized isZFExpress)
        # 2) isZFExpress found on this payload
        # 3) legacy extraction (is_express/isExpress/etc.)
        is_express_override = v.get("__zf_express_override__")
        zf_here             = _extract_is_zf_express_from_detail(v)
        legacy_express      = _extract_is_express_from_detail(v)

        if is_express_override is not None:
            final_is_express = bool(is_express_override)
        elif zf_here is not None:
            final_is_express = bool(zf_here)
        else:
            final_is_express = legacy_express if legacy_express is not None else None

        return {
            "vendor_code":  v.get("vendorCode"),
            "sf_name":      v.get("title", ""),
            "latitude":     v.get("lat"),
            "longitude":    v.get("lon"),
            "city_persian": v.get("city"),
            "min_order":    v.get("minOrder"),
            "review_stars": v.get("reviewStars"),
            "vendor_state": v.get("vendorState"),
            "chain_title":  v.get("chainTitle"),
            "branch_title": v.get("branchTitle"),
            "rating":       v.get("rating"),
            "comment_count": v.get("commentCount"),
            "is_express":   final_is_express,         # ← now stable & vendor-centered
            "work_schedules": "\n".join(sched_lines),
            "business_line": business_line,
            "logo":         v.get("logo"),            # ← logo from DETAIL_API_URL
            "cover":        v.get("coverPath"),       # ← cover from DETAIL_API_URL
        }

    def get_failure_count(self) -> int:
        return len(self.failed_vendors)

    def run(self, vendor_list_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Fetching details for {len(vendor_list_df)} vendors from the master list.")
        tasks = []
        for _, row in vendor_list_df.iterrows():
            code = row["sf_code"]
            city_name = row["city_name"]
            city_id = CITY_ID_MAP.get(city_name.lower()) if isinstance(city_name, str) else None
            tasks.append((code, city_id, city_name))

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(self._get_detail, c, cid, cn): c for c, cid, cn in tasks}
            for fut in concurrent.futures.as_completed(futures):
                raw = fut.result()
                if raw:
                    try:
                        info = self._parse(raw)
                        results.append(info)
                        self.logger.info(f"Parsed details for {info['vendor_code']} - {info.get('sf_name','')}")
                    except Exception as e:
                        self.logger.error(f"Error parsing data for vendor {futures[fut]}: {e}")

        df_initial = pd.DataFrame(results).drop_duplicates(subset=["vendor_code"])
        self.logger.info(f"Scraping completed: {df_initial.shape[0]} unique records.")

        if self.failed_vendors:
            self.logger.warning(f"⚠️ {len(self.failed_vendors)} vendors failed to scrape.")
        else:
            self.logger.info("🎉 All vendors successfully scraped!")

        return df_initial


# ─── Data Enrichment Functions (no grading) ──────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def assign_city_id(row) -> int | None:
    # Prefer Persian city name from scraped details
    city_persian_map = {"تهران": 2, "مشهد": 1, "شیراز": 5}
    cp = row.get("city_persian")
    if pd.notna(cp) and cp in city_persian_map:
        return city_persian_map[cp]
    try:
        return min(
            CITY_COORDS,
            key=lambda k: haversine(
                float(row["latitude"]), float(row["longitude"]),
                *CITY_COORDS[k]
            )
        )
    except (ValueError, TypeError):
        return None


def assign_city_name(row) -> str | None:
    city_id_to_name = {2: "tehran", 1: "mashhad", 5: "shiraz"}
    return city_id_to_name.get(row.get("city_id"))


def load_and_prepare_polygons(polygon_files: dict) -> pd.DataFrame:
    logger = logging.getLogger("SnappFoodScraper")
    all_polygons = []
    for city_id, file_path in polygon_files.items():
        if not file_path.exists():
            logger.warning(f"Polygon file not found, skipping: {file_path}")
            continue
        try:
            poly_df = pd.read_csv(file_path)
            poly_df["city_id"] = city_id
            poly_df["geometry"] = poly_df["WKT"].apply(wkt.loads)
            all_polygons.append(poly_df)
            logger.info(f"Loaded {len(poly_df)} polygons for city_id {city_id}")
        except Exception as e:
            logger.error(f"Failed to load or parse {file_path}: {e}")

    return pd.concat(all_polygons, ignore_index=True) if all_polygons else pd.DataFrame()


def assign_marketing_area(df: pd.DataFrame, polygons_df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger("SnappFoodScraper")
    if polygons_df.empty:
        logger.warning("Polygon data is empty. Skipping marketing area assignment.")
        df["marketing_area"] = "Unknown"
        return df

    polygons_by_city = {cid: g.to_records() for cid, g in polygons_df.groupby("city_id")}

    def find_area(row):
        try:
            point = Point(float(row["longitude"]), float(row["latitude"]))
            city_polygons = polygons_by_city.get(row["city_id"], [])
            for poly in city_polygons:
                if poly["geometry"].contains(point):
                    return poly["name"]
            return "Outside Defined Areas"
        except (ValueError, TypeError):
            return "Invalid Coordinates"

    logger.info("Starting marketing area assignment...")
    df["marketing_area"] = df.apply(find_area, axis=1)
    logger.info("Marketing area assignment complete.")
    return df


# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    logger = configure_logging()

    # STAGE 1: Get the master list of vendors to scrape from local files
    df_list = merge_source_vendor_files()

    logger.info("\n--- STAGE 2: RENEWING VENDOR DATA VIA SCRAPING ---")

    # Load polygon data
    polygons_df = load_and_prepare_polygons(POLYGON_FILES)

    # Step 1: Fetch details for every vendor in the master list (vendor-centered is_express)
    scraper = VendorDetailScraper()
    df_details = scraper.run(df_list)

    if df_details.empty:
        logger.error("Fatal: No vendor details could be scraped. Exiting.")
        return

    # Step 2: Build list-index for only the cities we actually have (fallbacks for is_express/business_line)
    cities_needed = (
        df_list["city_name"]
        .fillna("tehran")
        .astype(str).str.lower()
        .map(lambda c: c if c in CITY_ID_MAP else "tehran")
        .dropna()
        .unique()
        .tolist()
    )
    df_index = build_list_index_for_cities(cities_needed)

    # Step 3: Merge list-index into details to fill business_line & (optionally) is_express gaps
    if not df_index.empty:
        df_details = df_details.merge(df_index, on="vendor_code", how="left")

        if "business_line_list" in df_details.columns:
            df_details["business_line"] = df_details["business_line"].where(
                df_details["business_line"].notna(), df_details["business_line_list"]
            )

        # Keep is_express backfill as LAST resort (vendor-centered value has priority)
        if "is_express_list" in df_details.columns:
            df_details["is_express"] = df_details["is_express"].where(
                df_details["is_express"].notna(), df_details["is_express_list"]
            )
        
        # Note: Cover image now comes directly from DETAIL_API_URL, no list index merge needed

    # Step 4: Enrich with city_id, city_name and marketing area
    df_details["city_id"] = df_details.apply(assign_city_id, axis=1)
    df_details["city_name"] = df_details.apply(assign_city_name, axis=1)
    df_enriched = assign_marketing_area(df_details, polygons_df)

    # Step 5: Select, rename, and order final columns (no grading columns)
    final_columns_map = {
        "vendor_code": "sf_code",
        "sf_name": "sf_name",
        "city_name": "city_name",
        "city_id": "city_id",
        "marketing_area": "marketing_area",
        "business_line": "business_line",  
        "latitude": "sf_latitude",
        "longitude": "sf_longitude",
        "min_order": "min_order",
        "chain_title": "chain_title",
        "comment_count": "comment_count",
        "rating": "rating",
        "is_express": "is_express",
        "cover": "cover",
        "logo": "logo",
    }

    # Ensure the columns exist before selection (in case some are missing from responses)
    missing = [c for c in final_columns_map.keys() if c not in df_enriched.columns]
    for m in missing:
        df_enriched[m] = np.nan

    df_final = df_enriched[final_columns_map.keys()].rename(columns=final_columns_map)

    # Optional: enforce dtype for is_express
    df_final["is_express"] = df_final["is_express"].astype("boolean")

    # Step 6: Save the comprehensive output file (no grading)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"V_sf_vendor_RENEWED_{timestamp}.csv"
    df_final.to_csv(out_file, index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info(f"✅ PIPELINE COMPLETE! Renewed data for {len(df_final)} vendors.")
    logger.info(f"   Output saved to: {out_file}")
    logger.info("=" * 60)

    # Final failed vendors report
    final_failed_codes = [code for code, _, _ in scraper.failed_vendors if code not in df_details["vendor_code"].values]
    if final_failed_codes:
        failed_file = OUTPUT_DIR / f"failed_vendors_{timestamp}.txt"
        with open(failed_file, 'w', encoding="utf-8") as f:
            f.write("\n".join(final_failed_codes))
        logger.warning(f"⚠️  {len(final_failed_codes)} vendors could not be scraped after all attempts.")
        logger.warning(f"   A list of failed sf_codes has been saved to: {failed_file.name}")
    else:
        logger.info("🎉 Perfect run! All vendors were successfully scraped!")


if __name__ == "__main__":
    main()

# sf_vendors_scraper.py — reads consolidated sf_codes, cycles failed list, logs elapsed time
"""
--------------------------------------------------------------------------------
VENDOR DATA RENEWAL PIPELINE (DETAIL-first; reads consolidated codes)
--------------------------------------------------------------------------------
- Reads sf_codes from: data/scraped/consolidated_sf_codes.csv
- Skips any codes listed in: data/failed_to_scrape.csv (header: sf_code)
- Scrapes DETAIL_API_URL with vendor-centered recheck for isZFExpress;
  uses LIST_API_URL only to backfill business_line and (last-resort) is_express.
- Enriches city_id/name and marketing_area (Shapely).
- Saves final CSV to: outputs/sf_vendors.csv
- Writes current run's failures to: data/failed_to_scrape.csv (header: sf_code)
- Logs total elapsed time for the whole run.
--------------------------------------------------------------------------------
"""

import math
import os
import threading
import time
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
import pandas as pd
import numpy as np

# Requires: pip install shapely
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

# Workers & retries
PAGE_SIZE_LIST      = 100
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
BUSINESS_ALIASES = list(BUSINESS_LINE_MAP.keys())

# Cities in use
CITY_ID_MAP = {"tehran": 2, "mashhad": 1, "shiraz": 5}
CITY_COORDS = {
    2: (35.6892, 51.3890),  # Tehran
    1: (36.2605, 59.6168),  # Mashhad
    5: (29.5918, 52.5836),  # Shiraz
}
CITY_MULTIPLE_COORDS = {
    2: [
        (35.6892, 51.3890), (35.7219, 51.3347), (35.6961, 51.4231),
        (35.6515, 51.3680), (35.7058, 51.3570), (35.7297, 51.4015),
        (35.6736, 51.3185), (35.6403, 51.4180), (35.7456, 51.3750),
        (35.6234, 51.3456),
    ],
    1: [
        (36.2605, 59.6168), (36.2297, 59.5657), (36.2915, 59.6543),
        (36.2456, 59.6789), (36.2123, 59.5987), (36.3012, 59.6234),
        (36.2789, 59.7012), (36.1987, 59.6456),
    ],
    5: [
        (29.5918, 52.5836), (29.6234, 52.5456), (29.5654, 52.6123),
        (29.5543, 52.5234), (29.6012, 52.5678), (29.6345, 52.5987),
        (29.5876, 52.4987), (29.5432, 52.6345),
    ]
}

# Paths
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
SCRAPED_DIR = DATA_DIR / "scraped"
OUTPUTS_DIR = BASE_DIR / "outputs"
POLYGON_DIR = DATA_DIR / "polygons"

CONSOLIDATED_CODES_FILE = SCRAPED_DIR / "snappfood_vendor_codes.csv"
FAILED_LIST_FILE = SCRAPED_DIR / "failed_to_scrape.csv"
OUTPUT_FILE = OUTPUTS_DIR / "sf_vendors.csv"

# Ensure directories exist
for d in [DATA_DIR, SCRAPED_DIR, OUTPUTS_DIR, POLYGON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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


# ─── STAGE 1: Load consolidated sf_codes & prune known failed ────────────────

def load_consolidated_vendor_codes(logger: logging.Logger) -> pd.DataFrame:
    """
    Reads sf_codes from data/scraped/consolidated_sf_codes.csv (one column: 'sf_code').
    If a 'city_name' column exists, it will be used; otherwise set to None.
    Removes any sf_codes present in data/failed_to_scrape.csv before scraping.
    """
    path = CONSOLIDATED_CODES_FILE
    if not path.exists():
        logger.error(f"Fatal: Could not find consolidated codes file at: {path}")
        raise SystemExit(1)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to read {path.name}: {e}")
        raise SystemExit(1)

    if "sf_code" not in df.columns:
        logger.error(f"Fatal: {path.name} must contain a 'sf_code' column.")
        raise SystemExit(1)

    df["sf_code"] = (
        df["sf_code"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
    )
    df = df.dropna(subset=["sf_code"]).drop_duplicates(subset=["sf_code"]).copy()

    if "city_name" not in df.columns:
        df["city_name"] = None
    else:
        df["city_name"] = (
            df["city_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .where(lambda s: s.isin(CITY_ID_MAP.keys()), None)
        )

    total_before = len(df)

    # Remove previously failed codes (if file exists)
    failed_set: Set[str] = set()
    if FAILED_LIST_FILE.exists():
        try:
            df_failed = pd.read_csv(FAILED_LIST_FILE)
            if "sf_code" in df_failed.columns:
                failed_set = set(
                    df_failed["sf_code"].astype(str).str.strip().replace({"": np.nan}).dropna().unique()
                )
            else:
                logger.warning(f"{FAILED_LIST_FILE.name} has no 'sf_code' column; ignoring it.")
        except Exception as e:
            logger.warning(f"Could not read {FAILED_LIST_FILE.name}: {e}")

    if failed_set:
        df = df[~df["sf_code"].isin(failed_set)].copy()
        logger.info(f"Excluded {total_before - len(df)} previously failed vendor(s) from this run.")

    logger.info(f"Loaded {len(df)} unique sf_code(s) from {path.name}")
    return df[["sf_code", "city_name"]]


# ─── Helpers to extract business_line & is_express from details ──────────────

def _extract_business_line_from_detail(v: dict) -> Optional[str]:
    candidates: List[Any] = []

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


def _extract_is_express_from_detail(v: dict) -> Optional[bool]:
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


def _extract_is_zf_express_from_detail(v: dict) -> Optional[bool]:
    candidates = [v, v.get("express", {}), v.get("vendor", {})]
    keys = ("isZFExpress", "isZfExpress", "is_zf_express", "zfIsExpress")
    for d in candidates:
        if not isinstance(d, dict):
            continue
        for k in keys:
            if k in d and d[k] is not None:
                return bool(d[k])
    return None


def _get_first(v: dict, keys: List[str]) -> Any:
    for k in keys:
        if k in v and v.get(k) is not None:
            return v.get(k)
    return None


# ─── LIST endpoint index (fallbacks) ─────────────────────────────────────────

def _city_latlon_for_name(city_name: str) -> Tuple[float, float]:
    cid = CITY_ID_MAP.get(city_name.lower())
    if cid in CITY_COORDS:
        return CITY_COORDS[cid]
    return CITY_COORDS[2]


def build_list_index_for_cities(cities: List[str]) -> pd.DataFrame:
    logger = logging.getLogger("SnappFoodScraper")
    rows: List[Dict[str, Any]] = []
    headers = {"Referer": "https://snappfood.ir/"}

    city_norm: List[str] = []
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
                    "sp_alias": alias,
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

    df_rows = pd.DataFrame(rows)

    def _mode_or_first(s: pd.Series):
        m = s.mode()
        return m.iat[0] if not m.empty else (s.iloc[0] if len(s) else None)

    df_idx = (
        df_rows
        .groupby("vendor_code", as_index=False)
        .agg(
            is_express_list=("is_express_list",
                             lambda s: bool(np.nanmax(pd.to_numeric(s, errors='coerce'))) if s.notna().any() else np.nan),
            business_line_list=("business_line_list", _mode_or_first)
        )
    )

    logging.getLogger("SnappFoodScraper").info(f"List index built with {len(df_idx)} unique vendor_code rows.")
    return df_idx


# ─── STAGE 2: Vendor Detail Scraping ─────────────────────────────────────────

class VendorDetailScraper:
    def __init__(self):
        self.max_retries = DETAIL_MAX_RETRIES
        self.max_workers = DETAIL_MAX_WORKERS
        self.logger      = configure_logging()
        self.failed_vendors: List[Tuple[str, Optional[int], Optional[str]]] = []

    def _get_detail(self, code: str, city_id: int = None, city_name: str = None, patient_mode: bool = False) -> Optional[dict]:
        """Fetch vendor details. Probe coords; once found, re-query using the vendor's
        own lat/lon to neutralize location-dependent fields (e.g., isZFExpress)."""

        candidates: List[int] = []
        if isinstance(city_id, int) and city_id in CITY_MULTIPLE_COORDS:
            candidates.append(city_id)
        if city_name and isinstance(city_name, str):
            cid_from_name = CITY_ID_MAP.get(city_name.lower())
            if cid_from_name and cid_from_name in CITY_MULTIPLE_COORDS and cid_from_name not in candidates:
                candidates.append(cid_from_name)
        if not candidates:
            candidates = list(CITY_MULTIPLE_COORDS.keys())

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
                                self.logger.info(f"Found vendor {code} using city {cand_city_id} point #{coord_idx+1}")

                            # vendor-centered recheck
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

        business_line = _extract_business_line_from_detail(v)

        # is_express precedence
        is_express_override = v.get("__zf_express_override__")
        zf_here             = _extract_is_zf_express_from_detail(v)
        legacy_express      = _extract_is_express_from_detail(v)
        if is_express_override is not None:
            final_is_express = bool(is_express_override)
        elif zf_here is not None:
            final_is_express = bool(zf_here)
        else:
            final_is_express = legacy_express if legacy_express is not None else None

        vendor_state = _get_first(v, ["vendorState", "state"])
        tag_names = v.get("tagNames")
        if isinstance(tag_names, list):
            try:
                tag_names = ", ".join(map(lambda x: str(x).strip(), tag_names))
            except Exception:
                tag_names = str(tag_names)

        return {
            # IDs / names
            "vendor_code":   v.get("vendorCode"),
            "sf_name":       v.get("title", ""),

            # Location
            "latitude":      v.get("lat"),
            "longitude":     v.get("lon"),
            "city_persian":  v.get("city"),
            "address":       v.get("address"),

            # Ops & logistics
            "min_order":     v.get("minOrder"),
            "is_express":    final_is_express,
            "is_open":       v.get("isOpen"),
            "is_open_now":   v.get("isOpenNow"),
            "vendor_status": v.get("vendorStatus"),
            "vendor_state":  vendor_state,
            "vendor_sub_type": v.get("vendorSubType"),
            "in_place_delivery": v.get("inPlaceDelivery"),

            # Ratings & engagement
            "rating":        v.get("rating"),
            "review_stars":  _get_first(v, ["reviewStars", "review_stars"]),
            "comment_count": v.get("commentCount"),

            # Visuals
            "logo":          v.get("logo"),
            "cover":         v.get("coverPath"),

            # Chain / branch
            "chain_title":   v.get("chainTitle"),
            "branch_title":  v.get("branchTitle"),

            # Biz line
            "business_line": business_line,

            # Financial / tax
            "service_fee":   v.get("serviceFee"),
            "tax_enabled":   v.get("taxEnabled"),
            "tax_included":  v.get("taxIncluded"),
            "tax_enabled_in_products":     v.get("taxEnabledInProducts"),
            "tax_enabled_in_packaging":    v.get("taxEnabledInPackaging"),
            "tax_enabled_in_delivery_fee": v.get("taxEnabledInDeliveryFee"),

            # Promotions / flags
            "has_coupon":    v.get("has_coupon"),
            "has_packaging": v.get("has_packaging"),
            "tag_names":     tag_names,
            "is_pro":        v.get("isPro"),
            "is_economical": v.get("isEconomical"),

            # Schedule
            "work_schedules": "\n".join(sched_lines),
        }

    def get_failure_count(self) -> int:
        return len(self.failed_vendors)

    def run(self, vendor_list_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Fetching details for {len(vendor_list_df)} vendors from consolidated list.")
        tasks: List[Tuple[str, Optional[int], Optional[str]]] = []
        for _, row in vendor_list_df.iterrows():
            code = row["sf_code"]
            city_name = row["city_name"]
            city_id = CITY_ID_MAP.get(city_name.lower()) if isinstance(city_name, str) else None
            tasks.append((code, city_id, city_name))

        results: List[Dict[str, Any]] = []
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


# ─── Data Enrichment (no grading) ────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def assign_city_id(row) -> Optional[int]:
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


def assign_city_name(row) -> Optional[str]:
    city_id_to_name = {2: "tehran", 1: "mashhad", 5: "shiraz"}
    return city_id_to_name.get(row.get("city_id"))


def load_and_prepare_polygons(polygon_files: dict) -> pd.DataFrame:
    logger = logging.getLogger("SnappFoodScraper")
    all_polygons: List[pd.DataFrame] = []
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


# ─── Utility ─────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    # HH:MM:SS.ss
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    logger = configure_logging()
    start_ts = time.perf_counter()

    try:
        # STAGE 1: Load sf_codes & prune known fails
        df_list = load_consolidated_vendor_codes(logger)

        if df_list.empty:
            logger.warning("No vendors left to scrape after excluding failed list. Exiting.")
            # Ensure failed_to_scrape.csv exists with header
            if not FAILED_LIST_FILE.exists():
                pd.DataFrame(columns=["sf_code"]).to_csv(FAILED_LIST_FILE, index=False, encoding="utf-8-sig")
            elapsed = time.perf_counter() - start_ts
            logger.info(f"⏱️ Total elapsed: {_fmt_duration(elapsed)}")
            return

        logger.info("\n--- STAGE 2: RENEWING VENDOR DATA VIA SCRAPING ---")

        # Polygons (optional enrichment)
        polygons_df = load_and_prepare_polygons(POLYGON_FILES)

        # Details scraping (vendor-centered is_express)
        scraper = VendorDetailScraper()
        df_details = scraper.run(df_list)

        if df_details.empty:
            logger.error("Fatal: No vendor details could be scraped. Writing failures and exiting.")
            # On complete failure, mark all attempted as failed
            failed_codes = df_list["sf_code"].tolist()
            pd.DataFrame({"sf_code": failed_codes}).to_csv(FAILED_LIST_FILE, index=False, encoding="utf-8-sig")
            elapsed = time.perf_counter() - start_ts
            logger.info(f"⏱️ Total elapsed: {_fmt_duration(elapsed)}")
            return

        # Build list-index for only the cities we actually have (fallbacks)
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

        # Merge list-index to backfill business_line & (last-resort) is_express
        if not df_index.empty:
            df_details = df_details.merge(df_index, on="vendor_code", how="left")

            if "business_line_list" in df_details.columns:
                df_details["business_line"] = df_details["business_line"].where(
                    df_details["business_line"].notna(), df_details["business_line_list"]
                )

            if "is_express_list" in df_details.columns:
                df_details["is_express"] = df_details["is_express"].where(
                    df_details["is_express"].notna(), df_details["is_express_list"]
                )

        # City & marketing area enrichment
        df_details["city_id"] = df_details.apply(assign_city_id, axis=1)
        df_details["city_name"] = df_details.apply(assign_city_name, axis=1)
        df_enriched = assign_marketing_area(df_details, polygons_df)

        # Final column selection & ordering
        final_columns_map = {
            # Identification
            "vendor_code": "sf_code",
            "sf_name": "sf_name",

            # Location
            "city_name": "city_name",
            "city_id": "city_id",
            "marketing_area": "marketing_area",
            "latitude": "sf_latitude",
            "longitude": "sf_longitude",
            "address": "address",

            # Business
            "business_line": "business_line",
            "chain_title": "chain_title",
            "branch_title": "branch_title",
            "vendor_sub_type": "vendor_sub_type",

            # Operations
            "is_open": "is_open",
            "is_open_now": "is_open_now",
            "vendor_status": "vendor_status",
            "vendor_state": "vendor_state",
            "is_express": "is_express",

            # Performance
            "rating": "rating",
            "review_stars": "review_stars",
            "comment_count": "comment_count",
            "min_order": "min_order",

            # Promotions / tags
            "has_coupon": "has_coupon",
            "has_packaging": "has_packaging",
            "tag_names": "tag_names",
            "is_pro": "is_pro",
            "is_economical": "is_economical",

            # Logistics / tax / fees
            "in_place_delivery": "in_place_delivery",
            "service_fee": "service_fee",
            "tax_enabled": "tax_enabled",
            "tax_included": "tax_included",
            "tax_enabled_in_products": "tax_enabled_in_products",
            "tax_enabled_in_packaging": "tax_enabled_in_packaging",
            "tax_enabled_in_delivery_fee": "tax_enabled_in_delivery_fee",

            # Media
            "cover": "cover",
            "logo": "logo",

            # Schedules
            "work_schedules": "work_schedules",
        }

        missing = [c for c in final_columns_map.keys() if c not in df_enriched.columns]
        for m in missing:
            df_enriched[m] = np.nan

        df_final = df_enriched[final_columns_map.keys()].rename(columns=final_columns_map)

        # Normalize bool-like columns to nullable boolean
        bool_cols = [
            "is_express", "is_open", "is_open_now", "vendor_status",
            "in_place_delivery", "tax_enabled", "tax_included",
            "tax_enabled_in_products", "tax_enabled_in_packaging",
            "tax_enabled_in_delivery_fee", "has_coupon", "has_packaging",
            "is_pro", "is_economical",
        ]
        for bc in bool_cols:
            if bc in df_final.columns:
                try:
                    df_final[bc] = df_final[bc].astype("boolean")
                except Exception:
                    df_final[bc] = (
                        pd.to_numeric(df_final[bc], errors="coerce")
                        .astype("Int64")
                        .astype("boolean")
                    )

        # Save final output (overwrite each run)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        logger.info("=" * 60)
        logger.info(f"✅ PIPELINE COMPLETE! Renewed data for {len(df_final)} vendors.")
        logger.info(f"   Output saved to: {OUTPUT_FILE}")
        logger.info("=" * 60)

        # Failed vendors report for next cycle (overwrite with only this run's failures)
        scraped_codes = set(df_details["vendor_code"].astype(str))
        attempted_codes = set(df_list["sf_code"].astype(str))
        current_failures = sorted(attempted_codes - scraped_codes)
        pd.DataFrame({"sf_code": current_failures}).to_csv(FAILED_LIST_FILE, index=False, encoding="utf-8-sig")
        if current_failures:
            logger.warning(f"⚠️  {len(current_failures)} vendors failed; written to {FAILED_LIST_FILE}")
        else:
            logger.info("🎉 Perfect run! No failures; failed_to_scrape.csv now empty.")
    finally:
        elapsed = time.perf_counter() - start_ts
        logger.info(f"⏱️ Total elapsed: {_fmt_duration(elapsed)}")


if __name__ == "__main__":
    main()

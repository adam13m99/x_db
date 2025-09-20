# dual_matcher_v18.py
from __future__ import annotations

import glob
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.neighbors import BallTree

# --------------------------------------------------------------------------------------
#                                  CONFIG
# --------------------------------------------------------------------------------------

# Base layout (as requested)
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Inputs (moved into /outputs per your note)
TF_CSV = OUTPUTS_DIR / "tf_vendors.csv"
SF_CSV = OUTPUTS_DIR / "sf_vendors.csv"

# Extra matches (moved inside data/extra_matches/extra_matches.csv)
EXTRA_MATCHES_CSV = DATA_DIR / "extra_matches" / "extra_matches.csv"

# Optional polygons for scraper (unchanged path; only read if available)
SCRAPER_POLYGONS_DIR = DATA_DIR / "polygons"

# Outputs (exactly these CSVs, no others)
DUAL_MATCHED_CSV = OUTPUTS_DIR / "dual_matched_vendors.csv"
TF_ENRICHED_CSV = OUTPUTS_DIR / "tf_vendors_enriched.csv"
SF_ENRICHED_CSV = OUTPUTS_DIR / "sf_vendors_enriched.csv"
UNMATCHED_CSV = OUTPUTS_DIR / "unmatched.csv"
TF_PRO_CSV = OUTPUTS_DIR / "tf_vendors_pro.csv"
X_MAP_GRADE_CSV = OUTPUTS_DIR / "x_map_grade.csv"

# Prelinked failure TXT (only auxiliary output allowed)
PRELINKED_FAIL_TXT_DIR = DATA_DIR / "scraped"

# Snappfood vendor codes file
SNAPPFOOD_VENDOR_CODES_CSV = DATA_DIR / "scraped" / "snappfood_vendor_codes.csv"

# Matching knobs
PRELINKED_RADIUS_KM = 2.0
ALGO_RADIUS_KM = 0.2
POSSIBLE_RADIUS_KM = 0.14
EARTH_R = 6371.0088

FUZZY_THRESH = 95
MIN_SCORE = 95
POSSIBLE_FUZZY = 86

# Known city map for pretty names
CITY_MAP = {'1': 'Mashhad', '2': 'Tehran', '5': 'Shiraz'}

# Keep columns from SF where available (updated to single sf_grade)
SF_KEEP_COLS = [
    'sf_code', 'sf_name', 'city_id', 'marketing_area', 'business_line',
    'comment_count', 'rating', 'is_express', 'cover', 'logo', 'sf_grade',
    'sf_latitude', 'sf_longitude'
]

# Scraper options
ENABLE_SCRAPER_ENRICH = True
SCRAPE_MAX_PER_RUN = 500
SCRAPE_MAX_WORKERS = 15


# --------------------------------------------------------------------------------------
#                                  UTILITIES
# --------------------------------------------------------------------------------------

def configure_logging() -> logging.Logger:
    logger = logging.getLogger("DualMatcherV18")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


@lru_cache(50_000)
def _simplify(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    t = (txt.lower()
         .replace('ي', 'ی').replace('ك', 'ک').replace('ؤ', 'و')
         .replace('ئ', 'ی').replace('ة', 'ه'))
    t = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in t)
    return ' '.join(t.split())


def haversine_vec(lat1, lon1, lat2, lon2) -> np.ndarray:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def _tf_sf_series(df: pd.DataFrame) -> pd.Series:
    """Return a Series of the TF->SF link using sf_code if present, else prelinked_sf_code, else all NA."""
    if 'sf_code' in df.columns:
        return df['sf_code']
    if 'prelinked_sf_code' in df.columns:
        return df['prelinked_sf_code']
    return pd.Series(pd.NA, index=df.index)


def _ensure_numeric_radians(df: pd.DataFrame, lat_col: str, lon_col: str, lat_rad_col: str, lon_rad_col: str) -> pd.DataFrame:
    df[[lat_col, lon_col]] = df[[lat_col, lon_col]].apply(pd.to_numeric, errors='coerce')
    df[lat_rad_col] = np.radians(df[lat_col])
    df[lon_rad_col] = np.radians(df[lon_col])
    df.dropna(subset=[lat_rad_col, lon_rad_col], inplace=True)
    return df


def _write_csv(df: pd.DataFrame, path: Path, log: logging.Logger, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info(f"Wrote {label}: {len(df)} rows  →  {path}")


def _write_txt(lines: List[str], path: Path, log: logging.Logger, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"Wrote {label}: {path}")


def _norm_code_series(s: pd.Series) -> pd.Series:
    s_str = s.astype('string')
    return s_str.str.strip().str.lower()


def _norm_city_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors='coerce')
    s_int = pd.Series(s_num, dtype="Int64")
    return s_int.astype('string')


def _norm_bl_series(s: pd.Series) -> pd.Series:
    return s.astype('string').str.strip().str.casefold()


def _make_sf_lookup(df_sf: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SF_KEEP_COLS if c in df_sf.columns]
    return df_sf[cols].drop_duplicates(subset=['sf_code'])


def _ensure_unique_columns(df: pd.DataFrame, log: logging.Logger, label: str) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    dup_counts = df.columns.to_series().value_counts()
    dups = dup_counts[dup_counts > 1].index.tolist()
    log.warning(f"{label}: duplicate columns detected {dups} — auto-renaming with __dupN")
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    out = df.copy()
    out.columns = new_cols
    return out


def order_columns(df: pd.DataFrame, first_cols: List[str]) -> pd.DataFrame:
    first = [c for c in first_cols if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    # Keep the remainder in a stable, human-friendly order (alphabetical)
    return df[first + sorted(rest, key=lambda x: x.lower())]


def _deduplicate_by_tf_code_distance(df: pd.DataFrame, log: logging.Logger, label: str = "") -> pd.DataFrame:
    if df.empty or 'tf_code' not in df.columns:
        return df
    initial_count = len(df)
    df_work = df.copy()
    df_work['distance_km_sort'] = df_work['distance_km'].fillna(float('inf'))
    if 'source' in df_work.columns:
        priority = {'extra': -1, 'prelinked': 0, 'algorithmic': 1, 'matched': 1, 'possible': 2, 'llm': 0}
        df_work['source_priority'] = df_work['source'].map(priority).fillna(99).astype(int)
        sort_cols = ['tf_code', 'distance_km_sort', 'source_priority']
        sort_ascending = [True, True, True]
        if 'fuzzy_score' in df_work.columns:
            sort_cols.append('fuzzy_score')
            sort_ascending.append(False)
        df_work = df_work.sort_values(sort_cols, ascending=sort_ascending).drop(columns=['source_priority'])
    else:
        df_work = df_work.sort_values(['tf_code', 'distance_km_sort'], ascending=[True, True])
    df_dedup = df_work.drop_duplicates(subset=['tf_code'], keep='first').drop(columns=['distance_km_sort'])
    removed_count = initial_count - len(df_dedup)
    if removed_count > 0:
        log.info(f"{label}: removed {removed_count} duplicate tf_codes, keeping closest.")
    return df_dedup


def _deduplicate_by_sf_code_distance(df: pd.DataFrame, log: logging.Logger, label: str = "", scraper=None) -> pd.DataFrame:
    if df.empty or 'sf_code' not in df.columns:
        return df
    initial_count = len(df)
    df_work = df.copy()

    # Attempt to fill missing distances via scraper lat/lon, if available
    duplicates_mask = df_work['sf_code'].duplicated(keep=False)
    duplicates_groups = df_work[duplicates_mask].groupby('sf_code')
    scraped_count = 0
    calculated_count = 0
    for sf_code, group in duplicates_groups:
        if group['distance_km'].notna().any():
            continue
        if scraper is None or not ENABLE_SCRAPER_ENRICH:
            continue
        tf_coord_cols = [col for col in ['tf_latitude', 'tf_longitude'] if col in group.columns]
        if len(tf_coord_cols) < 2:
            continue
        has_tf_coords = (
            group[tf_coord_cols].notna().all(axis=1) |
            group[tf_coord_cols].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
        ).any()
        if not has_tf_coords:
            continue
        try:
            vendor_data = scraper.fetch_vendor(sf_code)
            sf_lat = vendor_data.get('lat')
            sf_lon = vendor_data.get('lon')
            if sf_lat is None or sf_lon is None:
                continue
            scraped_count += 1
            group_indices = group.index
            for idx in group_indices:
                row = df_work.loc[idx]
                tf_lat = pd.to_numeric(row.get('tf_latitude'), errors='coerce')
                tf_lon = pd.to_numeric(row.get('tf_longitude'), errors='coerce')
                if pd.notna(tf_lat) and pd.notna(tf_lon):
                    distance = haversine_vec(
                        np.array([np.radians(tf_lat)]),
                        np.array([np.radians(tf_lon)]),
                        np.array([np.radians(sf_lat)]),
                        np.array([np.radians(sf_lon)])
                    )[0]
                    df_work.loc[idx, 'distance_km'] = distance
                    calculated_count += 1
        except Exception:
            continue

    if scraped_count > 0:
        log.info(f"{label}: scraped {scraped_count} vendors, calculated {calculated_count} distances")

    df_work['distance_km_sort'] = df_work['distance_km'].fillna(float('inf'))
    if 'source' in df_work.columns:
        priority = {'extra': -1, 'prelinked': 0, 'algorithmic': 1, 'matched': 1, 'possible': 2, 'llm': 0}
        df_work['source_priority'] = df_work['source'].map(priority).fillna(99).astype(int)
        sort_cols = ['sf_code', 'distance_km_sort', 'source_priority']
        sort_ascending = [True, True, True]
        if 'fuzzy_score' in df_work.columns:
            sort_cols.append('fuzzy_score')
            sort_ascending.append(False)
        df_work = df_work.sort_values(sort_cols, ascending=sort_ascending).drop(columns=['source_priority'])
    else:
        df_work = df_work.sort_values(['sf_code', 'distance_km_sort'], ascending=[True, True])

    df_dedup = df_work.drop_duplicates(subset=['sf_code'], keep='first').drop(columns=['distance_km_sort'])
    removed_count = initial_count - len(df_dedup)
    if removed_count > 0:
        log.info(f"{label}: removed {removed_count} duplicate sf_codes, keeping closest.")
    return df_dedup


# --------------------------------------------------------------------------------------
#                               MATCHER CLASS (V18)
# --------------------------------------------------------------------------------------

class DualMatcherV18:
    def __init__(self):
        self.log = configure_logging()
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Optional scraper import
        self.scraper = None
        if ENABLE_SCRAPER_ENRICH:
            try:
                import scraper as _scraper
                self.scraper = _scraper
                self.log.info("Scraper module imported: scraper.py")
            except Exception:
                try:
                    import scrape as _scraper
                    self.scraper = _scraper
                    self.log.info("Scraper module imported: scrape.py")
                except Exception:
                    self.log.warning("Scraper module not found (scraper.py/scrape.py). Enrich disabled.")
                    self.scraper = None

        self.polygons_df = pd.DataFrame()
        if self.scraper is not None:
            try:
                self.polygons_df = self.scraper.load_polygons(Path(SCRAPER_POLYGONS_DIR))
                if self.polygons_df.empty:
                    self.log.warning(f"No polygons loaded from {SCRAPER_POLYGONS_DIR}. marketing_area will be 'Unknown'.")
                else:
                    self.log.info(f"Loaded polygons for scraper enrich: {len(self.polygons_df)} rows")
            except Exception as e:
                self.log.warning(f"Failed to load polygons: {e}")
                self.polygons_df = pd.DataFrame()

    # ----------------------------- PIPELINE ---------------------------------

    def run(self):
        start_time = time.time()
        self.log.info("Starting DualMatcher V18 pipeline")
        # Load inputs
        if not TF_CSV.exists() or not SF_CSV.exists():
            self.log.error(f"Missing input CSVs. Expected:\n - {TF_CSV}\n - {SF_CSV}")
            return

        df_tf = pd.read_csv(TF_CSV, dtype=str)
        df_sf_raw = pd.read_csv(SF_CSV, dtype=str)

        self.log.info(f"Loaded TF={len(df_tf)} rows  SF={len(df_sf_raw)} rows")

        # Normalize SF (add city_name if missing, type conversions)
        df_sf = df_sf_raw.copy()
        df_sf['_city_key'] = _norm_city_series(df_sf['city_id'])
        if 'city_name' not in df_sf.columns:
            df_sf['city_name'] = df_sf['_city_key'].map(CITY_MAP)

        # Normalize TF
        df_tf['city_name'] = _norm_city_series(df_tf['city_id']).map(CITY_MAP)

        # Numeric coords for SF
        df_sf = _ensure_numeric_radians(df_sf, 'sf_latitude', 'sf_longitude', 'lat_rad', 'lon_rad')
        sf_lookup = _make_sf_lookup(df_sf)

        # Phase 1: status_id == 5
        df_tf_status_5, df_tf_remaining = self._filter_by_status_id(df_tf, 5)
        self.log.info("=== Phase 1: Processing status_id = 5 ===")
        df_pre_final_phase1, df_dist_fail_phase1 = self._verify_prelinked(df_tf_status_5, df_sf)

        # Load extra matches if present (new path)
        df_extra_in = self._load_extra_matches()
        df_extra_phase1 = pd.DataFrame()
        if df_extra_in is not None and not df_tf_status_5.empty:
            status_5_tf_codes = set(df_tf_status_5['tf_code'].dropna())
            df_extra_filtered = df_extra_in[df_extra_in['tf_code'].isin(status_5_tf_codes)]
            df_extra_phase1 = self._prepare_extra_records(df_extra_filtered, df_tf_status_5, df_sf_raw) if not df_extra_filtered.empty else pd.DataFrame()
            if not df_extra_phase1.empty:
                self.log.info(f"Phase 1 - Extra matches: processed {len(df_extra_phase1)}")

        # Algorithmic on status_5 vendors that are still unlinked (excluding ones covered by extra)
        sf_link1 = _tf_sf_series(df_tf_status_5)
        df_unlinked_phase1 = df_tf_status_5[sf_link1.isna()].copy()
    
        if not df_extra_phase1.empty:
            extra_tf_codes_phase1 = set(df_extra_phase1['tf_code'].dropna())
            df_unlinked_phase1 = df_unlinked_phase1[~df_unlinked_phase1['tf_code'].isin(extra_tf_codes_phase1)].copy()

        df_algo_phase1 = self._algorithmic_match(df_unlinked_phase1, df_sf)

        # Consolidate Phase 1
        frames_phase1 = [x for x in [df_pre_final_phase1, df_extra_phase1, df_algo_phase1] if x is not None and not x.empty]
        if frames_phase1:
            df_matched_phase1 = pd.concat(frames_phase1, ignore_index=True)
        else:
            df_matched_phase1 = pd.DataFrame()
        if not df_matched_phase1.empty:
            df_matched_phase1 = self._backfill_sf_cols(df_matched_phase1, sf_lookup)
            df_matched_phase1 = _deduplicate_by_tf_code_distance(df_matched_phase1, self.log, "Phase 1 - tf_code")
            df_matched_phase1 = _deduplicate_by_sf_code_distance(df_matched_phase1, self.log, "Phase 1 - sf_code", self.scraper)
        self.log.info(f"Phase 1 completed: {len(df_matched_phase1)} matches")

        # Phase 2: remaining vendors (non-status 5) & SF after removing Phase1 sf_codes
        self.log.info("=== Phase 2: Processing remaining vendors !== 5 ===")
        phase1_sf_codes = set(df_matched_phase1['sf_code'].dropna()) if not df_matched_phase1.empty else set()
        df_sf_phase2 = df_sf[~df_sf['sf_code'].isin(phase1_sf_codes)].copy()
        df_sf_raw_phase2 = df_sf_raw[~df_sf_raw['sf_code'].isin(phase1_sf_codes)].copy()

        df_pre_final_phase2, df_dist_fail_phase2 = self._verify_prelinked(df_tf_remaining, df_sf_phase2)

        df_extra_phase2 = pd.DataFrame()
        if df_extra_in is not None and not df_tf_remaining.empty and not df_sf_phase2.empty:
            remaining_tf_codes = set(df_tf_remaining['tf_code'].dropna())
            available_sf_codes = set(df_sf_phase2['sf_code'].dropna())
            df_extra_filtered = df_extra_in[df_extra_in['tf_code'].isin(remaining_tf_codes) & df_extra_in['sf_code'].isin(available_sf_codes)]
            df_extra_phase2 = self._prepare_extra_records(df_extra_filtered, df_tf_remaining, df_sf_raw_phase2) if not df_extra_filtered.empty else pd.DataFrame()
            if not df_extra_phase2.empty:
                self.log.info(f"Phase 2 - Extra matches: processed {len(df_extra_phase2)}")

        sf_link2 = _tf_sf_series(df_tf_remaining)
        df_unlinked_phase2 = df_tf_remaining[sf_link2.isna()].copy()

        if not df_extra_phase2.empty:
            extra_tf_codes_phase2 = set(df_extra_phase2['tf_code'].dropna())
            df_unlinked_phase2 = df_unlinked_phase2[~df_unlinked_phase2['tf_code'].isin(extra_tf_codes_phase2)].copy()

        df_algo_phase2 = self._algorithmic_match(df_unlinked_phase2, df_sf_phase2)

        frames_phase2 = [x for x in [df_pre_final_phase2, df_extra_phase2, df_algo_phase2] if x is not None and not x.empty]
        if frames_phase2:
            df_matched_phase2 = pd.concat(frames_phase2, ignore_index=True)
        else:
            df_matched_phase2 = pd.DataFrame()
        if not df_matched_phase2.empty:
            sf_lookup_phase2 = _make_sf_lookup(df_sf_phase2)
            df_matched_phase2 = self._backfill_sf_cols(df_matched_phase2, sf_lookup_phase2)
            df_matched_phase2 = _deduplicate_by_tf_code_distance(df_matched_phase2, self.log, "Phase 2 - tf_code")
            df_matched_phase2 = _deduplicate_by_sf_code_distance(df_matched_phase2, self.log, "Phase 2 - sf_code", self.scraper)
        self.log.info(f"Phase 2 completed: {len(df_matched_phase2)} matches")

        # Combine & finalize one-to-one dual matches across phases
        df_matched = self._combine_dedup_across_phases(df_matched_phase1, df_matched_phase2)

        # Possible matches (for any still-unlinked TF against remaining SF)
        df_possible = self._find_possible_for_unlinked(df_tf, df_matched, df_sf)

        # Build final combined (matched + possible), dedup by tf and sf code with source priority, and WRITE REQUESTED OUTPUTS
        self._build_and_write_outputs(
            df_matched=df_matched,
            df_possible=df_possible,
            df_sf_raw=df_sf_raw,
            df_tf=df_tf,
            df_prelinked_fail=self._safe_concat([df_dist_fail_phase1, df_dist_fail_phase2])
        )

        # Print total runtime
        end_time = time.time()
        runtime_seconds = end_time - start_time
        runtime_minutes = runtime_seconds / 60
        self.log.info(f"DualMatcher V18 completed successfully. Total runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

        # Generate comprehensive analysis and report
        self._generate_comprehensive_report()

    # ----------------------------- STEPS ------------------------------------

    def _filter_by_status_id(self, df_tf: pd.DataFrame, status_id_value: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if 'status_id' not in df_tf.columns:
            self.log.warning(f"status_id not found in TF — treating all as non-status_{status_id_value}")
            return pd.DataFrame(), df_tf.copy()
        status_numeric = pd.to_numeric(df_tf['status_id'], errors='coerce')
        mask = status_numeric == status_id_value
        return df_tf[mask].copy(), df_tf[~mask].copy()

    def _verify_prelinked(self, df_tf: pd.DataFrame, df_sf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sf_link = _tf_sf_series(df_tf)
        df_pre = df_tf[sf_link.notna()].copy()
        self.log.info(f"Pre-linked input: {len(df_pre)} rows with sf_code/prelinked_sf_code")

        # Ensure we have a column literally named 'sf_code' for merging, without polluting original df_tf
        df_pre = df_pre.assign(sf_code=sf_link.loc[df_pre.index])
        self.log.info(f"Pre-linked input: {len(df_pre)} rows with sf_code")

        sf_cols_for_merge = [
            'sf_code', 'lat_rad', 'lon_rad', 'city_id', 'city_name',
            'marketing_area', 'business_line', 'comment_count', 'rating',
            'is_express', 'sf_grade', 'sf_latitude', 'sf_longitude'
        ]
        sf_cols_for_merge = [c for c in sf_cols_for_merge if c in df_sf.columns]

        pre = df_pre.merge(df_sf[sf_cols_for_merge], on='sf_code', how='left', suffixes=('_tf', '_sf'))

        pre[['tf_latitude', 'tf_longitude']] = pre[['tf_latitude', 'tf_longitude']].apply(pd.to_numeric, errors='coerce')
        pre['tf_lat_rad'] = np.radians(pre['tf_latitude'])
        pre['tf_lon_rad'] = np.radians(pre['tf_longitude'])

        both = pre['lat_rad'].notna() & pre['lon_rad'].notna() & pre['tf_lat_rad'].notna() & pre['tf_lon_rad'].notna()
        dist = np.full(len(pre), np.nan)
        dist[both.to_numpy()] = haversine_vec(pre.loc[both, 'tf_lat_rad'], pre.loc[both, 'tf_lon_rad'],
                                              pre.loc[both, 'lat_rad'], pre.loc[both, 'lon_rad'])
        pre['distance_km'] = dist
        dist_fail = both & (pre['distance_km'] > PRELINKED_RADIUS_KM)

        df_pre_ok = pre[~dist_fail].copy()
        self.log.info(f"Pre-linked verified (≤{PRELINKED_RADIUS_KM} km): {len(df_pre_ok)}  |  dropped: {int(dist_fail.sum())}")

        df_dist_fail = pre.loc[dist_fail, ['tf_code', 'sf_code', 'distance_km']].copy()
        df_dist_fail['reason'] = f"distance > {PRELINKED_RADIUS_KM} km"

        # Shape prelinked records to unified schema
        df_pre_ok['sf_name'] = df_pre_ok['sf_code'].map(df_sf.set_index('sf_code')['sf_name'])
        rename_map = {
            'city_id_sf': 'sf_city_id', 'city_name_sf': 'sf_city_name',
            'marketing_area_sf': 'sf_marketing_area', 'business_line_sf': 'sf_business_line'
        }
        for old, new in rename_map.items():
            if old in df_pre_ok.columns:
                df_pre_ok.rename(columns={old: new}, inplace=True)

        tf_rename = {
            'city_id_tf': 'tf_city_id', 'city_name_tf': 'tf_city_name',
            'marketing_area_tf': 'tf_marketing_area', 'business_line_tf': 'tf_business_line'
        }
        for old, new in tf_rename.items():
            if old in df_pre_ok.columns:
                df_pre_ok.rename(columns={old: new}, inplace=True)

        kept_cols = [
            'sf_code', 'sf_name', 'sf_city_id', 'sf_city_name', 'sf_marketing_area', 'sf_business_line',
            'city_id', 'marketing_area', 'business_line', 'comment_count', 'rating', 'is_express',
            'sf_grade', 'sf_latitude', 'sf_longitude',
            'tf_code', 'tf_name', 'tf_city_id', 'tf_city_name', 'tf_marketing_area', 'tf_business_line',
            'tf_latitude', 'tf_longitude', 'zero_orders', 'available_H', 'availability', 'status_id', 'vendor_status',
            'distance_km'
        ]
        kept_cols = [c for c in kept_cols if c in df_pre_ok.columns]
        df_pre_final = df_pre_ok.loc[:, kept_cols].copy()
        df_pre_final['source'] = 'prelinked'
        return df_pre_final, df_dist_fail

    def _algorithmic_match(self, df_tf_sub: pd.DataFrame, df_sf: pd.DataFrame) -> pd.DataFrame:
        df = df_tf_sub.copy()
        sf = df_sf.copy()

        for d in (df, sf):
            d['_city_key'] = _norm_city_series(d['city_id'])
            d['_bl_key'] = _norm_bl_series(d['business_line'])

        before_tf = len(df)
        bl = df['_bl_key']
        keep_mask = bl.notna() & (bl != '') & (bl != 'other')
        df = df[keep_mask].copy()
        dropped_other_blank = before_tf - len(df)

        sf_keys = set(map(tuple, sf.loc[sf['_city_key'].notna() & sf['_bl_key'].notna(), ['_city_key', '_bl_key']].drop_duplicates().to_numpy()))
        df['_grp_key'] = list(zip(df['_city_key'], df['_bl_key']))
        df = df[df['_grp_key'].isin(sf_keys)].copy()
        aligned_tf = len(df)

        self.log.info(f"Algorithmic: removed 'Other'/blank BL: {dropped_other_blank}; aligned TF rows: {aligned_tf}")
        if df.empty:
            return pd.DataFrame()

        df = _ensure_numeric_radians(df, 'tf_latitude', 'tf_longitude', 'lat_rad', 'lon_rad')
        sf = _ensure_numeric_radians(sf, 'sf_latitude', 'sf_longitude', 'lat_rad', 'lon_rad')

        df['norm'] = df['tf_name'].map(_simplify)
        sf['norm'] = sf['sf_name'].map(_simplify)

        tf_grps: Dict[Tuple[str, str], pd.DataFrame] = {k: g.reset_index(drop=True) for k, g in df.groupby(['_city_key', '_bl_key'])}
        sf_grps: Dict[Tuple[str, str], pd.DataFrame] = {k: g.reset_index(drop=True) for k, g in sf.groupby(['_city_key', '_bl_key'])}
        common_keys = set(tf_grps.keys()) & set(sf_grps.keys())
        if not common_keys:
            self.log.info("Algorithmic: no common (city_id, business_line) groups.")
            return pd.DataFrame()

        trees = {k: BallTree(np.c_[tf_grps[k]['lat_rad'], tf_grps[k]['lon_rad']], metric='haversine')
                 for k in common_keys if not tf_grps[k].empty}

        candidates = []
        rad = ALGO_RADIUS_KM / EARTH_R
        for grp in common_keys:
            tf_g = tf_grps.get(grp)
            sf_g = sf_grps.get(grp)
            if tf_g is None or sf_g is None or tf_g.empty or sf_g.empty:
                continue

            name_mat = process.cdist(sf_g['norm'], tf_g['norm'], scorer=fuzz.token_set_ratio)
            idxs, dists = trees[grp].query_radius(np.c_[sf_g['lat_rad'], sf_g['lon_rad']], r=rad, return_distance=True, sort_results=True)

            for i, (inds, dlist) in enumerate(zip(idxs, dists)):
                for j, dr in zip(inds, dlist):
                    sc = name_mat[i, j]
                    if sc >= FUZZY_THRESH:
                        candidates.append((grp, i, j, int(sc), float(dr * EARTH_R)))

        matches = [c for c in candidates if c[3] >= MIN_SCORE]

        recs = []
        for grp, i, j, sc, dkm in matches:
            sf_g, tf_g = sf_grps[grp], tf_grps[grp]
            city_key, bl_key = grp
            recs.append({
                'sf_code': sf_g.at[i, 'sf_code'],
                'sf_name': sf_g.at[i, 'sf_name'],
                'sf_city_id': city_key,
                'sf_city_name': sf_g.at[i, 'city_name'] if 'city_name' in sf_g.columns else CITY_MAP.get(str(city_key), None),
                'sf_marketing_area': sf_g.at[i, 'marketing_area'],
                'sf_business_line': sf_g.at[i, 'business_line'],
                'city_id': city_key,
                'marketing_area': sf_g.at[i, 'marketing_area'],
                'business_line': sf_g.at[i, 'business_line'],
                'comment_count': sf_g.at[i, 'comment_count'] if 'comment_count' in sf_g.columns else None,
                'rating': sf_g.at[i, 'rating'] if 'rating' in sf_g.columns else None,
                'is_express': sf_g.at[i, 'is_express'] if 'is_express' in sf_g.columns else None,
                'sf_grade': sf_g.at[i, 'sf_grade'] if 'sf_grade' in sf_g.columns else None,
                'sf_latitude': sf_g.at[i, 'sf_latitude'],
                'sf_longitude': sf_g.at[i, 'sf_longitude'],

                'tf_code': tf_g.at[j, 'tf_code'],
                'tf_name': tf_g.at[j, 'tf_name'],
                'tf_city_id': city_key,
                'tf_city_name': tf_g.at[j, 'city_name'] if 'city_name' in tf_g.columns else None,
                'tf_marketing_area': tf_g.at[j, 'marketing_area'],
                'tf_business_line': tf_g.at[j, 'business_line'],
                'zero_orders': tf_g.at[j, 'zero_orders'] if 'zero_orders' in tf_g.columns else None,
                'available_H': tf_g.at[j, 'available_H'] if 'available_H' in tf_g.columns else None,
                'availability': tf_g.at[j, 'availability'] if 'availability' in tf_g.columns else None,
                'status_id': tf_g.at[j, 'status_id'] if 'status_id' in tf_g.columns else None,
                'vendor_status': tf_g.at[j, 'vendor_status'] if 'vendor_status' in tf_g.columns else None,

                'distance_km': round(dkm, 4),
                'fuzzy_score': sc,
                'source': 'algorithmic'
            })

        out = pd.DataFrame(recs)
        self.log.info(f"Algorithmic matches: {len(out)} (≥{MIN_SCORE} fuzzy & ≤{ALGO_RADIUS_KM} km)")
        return out

    def _find_possible_for_unlinked(self, df_tf: pd.DataFrame, df_matched: pd.DataFrame, df_sf: pd.DataFrame) -> pd.DataFrame:
        matched_tf_codes = set(df_matched['tf_code'].dropna()) if not df_matched.empty else set()
        df_unlinked = df_tf[~df_tf['tf_code'].isin(matched_tf_codes)].copy()
        if df_unlinked.empty:
            return pd.DataFrame()
        # Limit SF to those not already claimed by matched
        used_sf_codes = set(df_matched['sf_code'].dropna()) if not df_matched.empty else set()
        df_sf_remaining = df_sf[~df_sf['sf_code'].isin(used_sf_codes)].copy()
        return self._find_possible(df_unlinked, df_sf_remaining)

    def _find_possible(self, df_tf_sub: pd.DataFrame, df_sf: pd.DataFrame) -> pd.DataFrame:
        df = df_tf_sub.copy()
        df['norm'] = df['tf_name'].map(_simplify)
        df = _ensure_numeric_radians(df, 'tf_latitude', 'tf_longitude', 'lat_rad', 'lon_rad')

        sf = df_sf.copy()
        sf['norm'] = sf['sf_name'].map(_simplify)

        if sf.empty:
            return pd.DataFrame()

        tree = BallTree(np.c_[sf['lat_rad'], sf['lon_rad']], metric='haversine')
        rad = POSSIBLE_RADIUS_KM / EARTH_R

        possibles = []
        for _, tf in df.iterrows():
            idxs, dists = tree.query_radius([[tf['lat_rad'], tf['lon_rad']]], r=rad, return_distance=True, sort_results=True)
            idxs, dists = idxs[0], (dists[0] * EARTH_R)
            if not len(idxs):
                continue
            norms = sf['norm'].iloc[idxs].tolist()
            scores = process.cdist([tf['norm']], norms, scorer=fuzz.token_set_ratio)[0]
            for pos_idx, sc in enumerate(scores):
                sc = int(sc)
                if sc >= POSSIBLE_FUZZY:
                    row = sf.iloc[idxs[pos_idx]]
                    possibles.append({
                        'tf_code': tf['tf_code'],
                        'tf_name': tf['tf_name'],
                        'tf_city_id': tf['city_id'],
                        'tf_city_name': tf.get('city_name'),
                        'tf_marketing_area': tf['marketing_area'],
                        'tf_business_line': tf['business_line'],
                        'tf_latitude': tf['tf_latitude'],
                        'tf_longitude': tf['tf_longitude'],
                        'zero_orders': tf.get('zero_orders'),
                        'available_H': tf.get('available_H'),
                        'availability': tf.get('availability'),
                        'status_id': tf.get('status_id'),
                        'vendor_status': tf.get('vendor_status'),

                        'sf_possible_code': row['sf_code'],
                        'sf_possible_name': row['sf_name'],
                        'sf_city_id': row['city_id'],
                        'sf_city_name': row.get('city_name'),
                        'sf_marketing_area': row['marketing_area'],
                        'sf_business_line': row['business_line'],
                        'city_id': row['city_id'],
                        'marketing_area': row['marketing_area'],
                        'business_line': row['business_line'],
                        'comment_count': row.get('comment_count'),
                        'rating': row.get('rating'),
                        'is_express': row.get('is_express'),
                        'sf_grade': row.get('sf_grade'),
                        'sf_latitude': row['sf_latitude'],
                        'sf_longitude': row['sf_longitude'],

                        'distance_km': round(float(dists[pos_idx]), 4),
                        'fuzzy_score': sc
                    })

        out_cols = [
            'tf_code', 'tf_name', 'tf_city_id', 'tf_city_name', 'tf_marketing_area', 'tf_business_line',
            'sf_possible_code', 'sf_possible_name', 'sf_city_id', 'sf_city_name', 'sf_marketing_area', 'sf_business_line',
            'city_id', 'marketing_area', 'business_line', 'comment_count', 'rating', 'is_express', 'sf_grade',
            'sf_latitude', 'sf_longitude', 'distance_km', 'fuzzy_score',
            'zero_orders', 'available_H', 'availability', 'status_id', 'vendor_status'
        ]
        out = pd.DataFrame(possibles)
        out = out[[c for c in out_cols if c in out.columns]]
        self.log.info(f"Possible matches (≥{POSSIBLE_FUZZY} fuzzy & ≤{POSSIBLE_RADIUS_KM} km): {len(out)}")
        return out

    def _load_extra_matches(self) -> Optional[pd.DataFrame]:
        if not EXTRA_MATCHES_CSV.exists():
            self.log.info("No extra matches file found (skipping).")
            return None
        df = pd.read_csv(EXTRA_MATCHES_CSV, dtype=str)
        canon = {c.lower().strip(): c for c in df.columns}
        if "tf_code" not in canon or "sf_code" not in canon:
            self.log.warning(f"Extra matches file missing tf_code/sf_code; skipping.")
            return None

        cols = []
        for k in ["tf_code", "tf_name", "sf_code", "sf_name"]:
            if k in df.columns:
                cols.append(k)
            elif k in canon:
                cols.append(canon[k])

        df = df[cols].rename(columns={
            canon.get("tf_code", "tf_code"): "tf_code",
            canon.get("tf_name", "tf_name"): "tf_name",
            canon.get("sf_code", "sf_code"): "sf_code",
            canon.get("sf_name", "sf_name"): "sf_name"
        })
        df = df[df["tf_code"].notna() & (df["tf_code"].astype(str).str.strip() != "") &
                df["sf_code"].notna() & (df["sf_code"].astype(str).str.strip() != "")]
        self.log.info(f"Loaded {len(df)} extra matches")
        return df.reset_index(drop=True)

    def _prepare_extra_records(self, df_extra: pd.DataFrame, df_tf: pd.DataFrame, df_sf_raw: pd.DataFrame) -> pd.DataFrame:
        if df_extra is None or df_extra.empty:
            return pd.DataFrame()

        tf = df_tf.copy()
        sf = df_sf_raw.copy()

        sf['_city_key'] = _norm_city_series(sf['city_id'])
        if 'city_name' not in sf.columns:
            sf['city_name'] = sf['_city_key'].map(CITY_MAP)

        tf['_city_key'] = _norm_city_series(tf['city_id'])
        if 'city_name' not in tf.columns:
            tf['city_name'] = tf['_city_key'].map(CITY_MAP)

        sf_cols_base = ['sf_code', 'city_id', 'marketing_area', 'business_line', 'city_name', *SF_KEEP_COLS]
        sf_cols = [c for c in dict.fromkeys(sf_cols_base) if c in sf.columns]
        ex = df_extra.merge(sf[sf_cols], on='sf_code', how='left', suffixes=('', '__sf'))

        tf_cols = ['tf_code', 'tf_name', 'city_id', 'city_name', 'marketing_area', 'business_line',
                   'tf_latitude', 'tf_longitude', 'zero_orders', 'available_H', 'availability',
                   'status_id', 'vendor_status']
        tf_cols = [c for c in tf_cols if c in tf.columns]
        ex = ex.merge(tf[tf_cols], on='tf_code', how='left', suffixes=('', '__tf'))

        def _to_rad(x):
            v = pd.to_numeric(x, errors='coerce')
            return np.radians(v)

        lat_sf = _to_rad(ex.get('sf_latitude'))
        lon_sf = _to_rad(ex.get('sf_longitude'))
        lat_tf = _to_rad(ex.get('tf_latitude'))
        lon_tf = _to_rad(ex.get('tf_longitude'))

        mask = lat_sf.notna() & lon_sf.notna() & lat_tf.notna() & lon_tf.notna()
        dist = np.full(len(ex), np.nan, dtype=float)
        if len(ex):
            dist[mask.to_numpy()] = haversine_vec(lat_tf[mask], lon_tf[mask], lat_sf[mask], lon_sf[mask])
        ex['distance_km'] = np.round(dist, 4)

        ex['__tf_norm'] = ex['tf_name'].map(_simplify)
        sf_name_best = ex['sf_name'].copy()
        if 'sf_name__sf' in ex.columns:
            need = sf_name_best.isna() | (sf_name_best.astype(str).str.strip() == '')
            sf_name_best.loc[need] = ex.loc[need, 'sf_name__sf']
        ex['__sf_norm'] = sf_name_best.map(_simplify)

        def _nz(x): return x if isinstance(x, str) else ""
        try:
            ex['fuzzy_score'] = ex.apply(lambda r: fuzz.token_set_ratio(_nz(r['__tf_norm']), _nz(r['__sf_norm'])),
                                         axis=1).astype('Int64')
        except Exception:
            ex['fuzzy_score'] = pd.NA

        out = pd.DataFrame({
            'sf_code': ex['sf_code'],
            'sf_name': sf_name_best,
            'sf_city_id': _norm_city_series(ex['city_id']),
            'sf_city_name': ex.get('city_name'),
            'sf_marketing_area': ex.get('marketing_area'),
            'sf_business_line': ex.get('business_line'),
            'city_id': ex.get('city_id'),
            'marketing_area': ex.get('marketing_area'),
            'business_line': ex.get('business_line'),
            'comment_count': ex.get('comment_count'),
            'rating': ex.get('rating'),
            'is_express': ex.get('is_express'),
            'sf_grade': ex.get('sf_grade'),
            'sf_latitude': ex.get('sf_latitude'),
            'sf_longitude': ex.get('sf_longitude'),

            'tf_code': ex['tf_code'],
            'tf_name': ex['tf_name'],
            'tf_city_id': _norm_city_series(ex.get('city_id__tf') if 'city_id__tf' in ex.columns else ex.get('city_id')),
            'tf_city_name': ex.get('city_name__tf') if 'city_name__tf' in ex.columns else ex.get('city_name'),
            'tf_marketing_area': ex.get('marketing_area__tf') if 'marketing_area__tf' in ex.columns else ex.get('marketing_area'),
            'tf_business_line': ex.get('business_line__tf') if 'business_line__tf' in ex.columns else ex.get('business_line'),
            'zero_orders': ex.get('zero_orders'),
            'available_H': ex.get('available_H'),
            'availability': ex.get('availability'),
            'status_id': ex.get('status_id'),
            'vendor_status': ex.get('vendor_status'),

            'distance_km': ex['distance_km'],
            'fuzzy_score': ex['fuzzy_score'],
            'source': 'extra'
        })

        # Resolve intra-extra conflicts (keep closest, highest fuzzy)
        if out.empty:
            return out
        out = self._deduplicate_extra_matches(out)
        return out

    def _deduplicate_extra_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        initial_count = len(df)
        dfw = df.copy()
        if 'distance_km' not in dfw.columns:
            dfw['distance_km'] = 0.1
        if 'fuzzy_score' not in dfw.columns:
            dfw['fuzzy_score'] = 100
        if 'source' not in dfw.columns:
            dfw['source'] = 'extra'

        dfw['distance_sort'] = dfw['distance_km'].fillna(float('inf'))
        dfw['fuzzy_sort'] = dfw['fuzzy_score'].fillna(0)

        # First: keep best SF for each TF
        dfw = dfw.sort_values(['tf_code', 'distance_sort', 'fuzzy_sort'], ascending=[True, True, False])
        df_tf_dedup = dfw.drop_duplicates(subset=['tf_code'], keep='first')

        # Then: keep best TF per SF with a simple priority
        source_priority = {'extra': 1, 'prelinked': 2, 'algorithmic': 3, 'possible': 4, 'llm': 5}
        df_tf_dedup['source_priority'] = df_tf_dedup['source'].map(source_priority).fillna(99)
        df_tf_dedup = df_tf_dedup.sort_values(
            ['sf_code', 'source_priority', 'distance_sort', 'fuzzy_sort'],
            ascending=[True, True, True, False]
        )
        df_final = df_tf_dedup.drop_duplicates(subset=['sf_code'], keep='first').drop(
            columns=['distance_sort', 'fuzzy_sort', 'source_priority']
        )
        if len(df_final) < initial_count:
            self.log.info(f"Extra matches optimization: {initial_count} → {len(df_final)}")
        return df_final.reset_index(drop=True)

    def _backfill_sf_cols(self, df_any: pd.DataFrame, sf_lookup: pd.DataFrame,
                          code_col: str = 'sf_code', name_col: Optional[str] = 'sf_name') -> pd.DataFrame:
        if code_col not in df_any.columns:
            return df_any
        out = df_any.merge(sf_lookup, left_on=code_col, right_on='sf_code', how='left', suffixes=('', '__sfref'))
        for col in SF_KEEP_COLS:
            ref = f"{col}__sfref"
            if col in out.columns and ref in out.columns:
                missing = out[col].isna() | (out[col].astype(str).str.strip() == '')
                out.loc[missing, col] = out.loc[missing, ref]
                out.drop(columns=[ref], inplace=True, errors='ignore')
        if name_col and name_col in out.columns and 'sf_name__sfref' in out.columns:
            missing = out[name_col].isna() | (out[name_col].astype(str).str.strip() == '')
            out.loc[missing, name_col] = out.loc[missing, 'sf_name__sfref']
            out.drop(columns=['sf_name__sfref'], inplace=True, errors='ignore')
        if 'sf_code__sfref' in out.columns:
            out.drop(columns=['sf_code__sfref'], inplace=True, errors='ignore')
        return out

    def _combine_dedup_across_phases(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        frames = [f for f in [df1, df2] if f is not None and not f.empty]
        if frames:
            df_matched = pd.concat(frames, ignore_index=True)
        else:
            df_matched = pd.DataFrame()
        if df_matched.empty:
            return df_matched

        # Prefer phase 1 for same tf_code/sf_code by adding a phase priority (status=5 wins)
        phase1_tf_codes = set(df1['tf_code'].dropna()) if df1 is not None and not df1.empty else set()
        df_matched['phase_priority'] = df_matched['tf_code'].apply(lambda x: 1 if x in phase1_tf_codes else 2)

        # Deduplicate across tf_code then sf_code
        df_matched = df_matched.sort_values(['tf_code', 'phase_priority', 'distance_km'],
                                            ascending=[True, True, True]).drop_duplicates(['tf_code'], keep='first')
        df_matched = df_matched.sort_values(['sf_code', 'phase_priority', 'distance_km'],
                                            ascending=[True, True, True]).drop_duplicates(['sf_code'], keep='first')
        df_matched = df_matched.drop(columns=['phase_priority'])
        return df_matched

    # ------------------ OUTPUT CONSTRUCTION (ONLY THE REQUESTED FILES) ------------------

    def _build_and_write_outputs(self,
                                 df_matched: pd.DataFrame,
                                 df_possible: pd.DataFrame,
                                 df_sf_raw: pd.DataFrame,
                                 df_tf: pd.DataFrame,
                                 df_prelinked_fail: pd.DataFrame):

        # 1) Build unified pool (matched + possible normalized) → then dedup by TF & SF with source priority
        df_possible_norm = df_possible.copy()
        if not df_possible_norm.empty:
            df_possible_norm.drop(columns=['sf_code', 'sf_name'], errors='ignore', inplace=True)
            df_possible_norm = df_possible_norm.rename(columns={
                'sf_possible_code': 'sf_code',
                'sf_possible_name': 'sf_name'
            })
            df_possible_norm['source'] = 'possible'

        # Ensure schemas
        common_cols = [
            'sf_code', 'sf_name', 'sf_city_id', 'sf_city_name', 'sf_marketing_area', 'sf_business_line',
            'city_id', 'marketing_area', 'business_line', 'comment_count', 'rating', 'is_express', 'cover', 'logo',
            'sf_grade', 'sf_latitude', 'sf_longitude',
            'tf_code', 'tf_name', 'tf_city_id', 'tf_city_name', 'tf_marketing_area', 'tf_business_line',
            'zero_orders', 'available_H', 'availability', 'status_id', 'vendor_status',
            'distance_km', 'fuzzy_score', 'source'
        ]

        for _df in (df_matched, df_possible_norm):
            if _df is not None and not _df.empty:
                for c in common_cols:
                    if c not in _df.columns:
                        _df[c] = pd.NA

        # Combine and dedup (keep "extra" > prelinked > algorithmic > possible)
        frames = [d for d in [df_matched, df_possible_norm] if d is not None and not d.empty]
        if frames:
            df_all = pd.concat(frames, ignore_index=True)
        else:
            df_all = pd.DataFrame()

        if df_all.empty:
            self.log.warning("No matches or possibles found. Outputs will reflect that.")
            # Even so, proceed to create minimal enrichment and unmatched files.
        else:
            df_all = _ensure_unique_columns(df_all, self.log, "df_all (combined)")
            # tf-level dedup
            df_all = _deduplicate_by_tf_code_distance(df_all, self.log, "combined by tf_code")
            # sf-level dedup
            df_all = _deduplicate_by_sf_code_distance(df_all, self.log, "combined by sf_code", self.scraper)

        # 1) dual_matched_vendors.csv
        dual_cols_order = [
            # TF side first (helps downstream)
            'tf_code', 'tf_name', 'tf_city_id', 'tf_city_name', 'tf_marketing_area', 'tf_business_line',
            'zero_orders', 'available_H', 'availability', 'status_id', 'vendor_status',
            # SF side
            'sf_code', 'sf_name', 'sf_city_id', 'sf_city_name', 'sf_marketing_area', 'sf_business_line',
            'city_id', 'marketing_area', 'business_line', 'sf_latitude', 'sf_longitude',
            'rating', 'comment_count', 'is_express', 'sf_grade', 'cover', 'logo',
            # Matching meta
            'distance_km', 'fuzzy_score', 'source'
        ]
        df_dual = df_all.copy() if not df_all.empty else pd.DataFrame(columns=dual_cols_order)
        df_dual = order_columns(df_dual, dual_cols_order)
        _write_csv(df_dual, DUAL_MATCHED_CSV, self.log, "dual_matched_vendors.csv")

        # Convenience maps for enrichment
        sf_cols_for_enrich = ['sf_code', 'sf_name', 'sf_business_line', 'sf_latitude', 'sf_longitude',
                              'rating', 'comment_count', 'sf_grade']
        tf_cols_all = list(df_tf.columns)

        # 2) tf_vendors_enriched.csv
        df_tf_enriched = df_tf.copy()

        if 'prelinked_sf_code' in df_tf_enriched.columns:
            df_tf_enriched.drop(columns=['prelinked_sf_code'], inplace=True)

        # Attach SF info for matched tf_codes
        if not df_dual.empty:
            tf_to_sf = df_dual[['tf_code'] + sf_cols_for_enrich].drop_duplicates('tf_code')
            df_tf_enriched = df_tf_enriched.merge(tf_to_sf, on='tf_code', how='left')
            df_tf_enriched['dual_match'] = df_tf_enriched['sf_code'].notna().astype(int)
        else:
            for c in sf_cols_for_enrich:
                df_tf_enriched[c] = pd.NA
            df_tf_enriched['dual_match'] = 0

        # Set null sf_grade values to 'Not Found on Snappfood'
        if 'sf_grade' in df_tf_enriched.columns:
            df_tf_enriched['sf_grade'] = df_tf_enriched['sf_grade'].fillna('Not Found on Snappfood')

        # Column order: original TF columns first, then enrichment
        tf_enrich_order = [c for c in tf_cols_all if c != 'prelinked_sf_code'] + sf_cols_for_enrich + ['dual_match']
        df_tf_enriched = order_columns(df_tf_enriched, tf_enrich_order)
        _write_csv(df_tf_enriched, TF_ENRICHED_CSV, self.log, "tf_vendors_enriched.csv")

        # 3) sf_vendors_enriched.csv
        df_sf_enriched = df_sf_raw.copy()

        # Merge TF columns onto SF by the dual matches
        if not df_dual.empty:
            sf_to_tf = df_dual[['sf_code', 'tf_code', 'tf_name', 'tf_city_id', 'tf_city_name',
                                'tf_marketing_area', 'tf_business_line', 'tf_latitude', 'tf_longitude',
                                'zero_orders', 'available_H', 'availability', 'status_id', 'vendor_status']].drop_duplicates('sf_code')
            df_sf_enriched = df_sf_enriched.merge(sf_to_tf, on='sf_code', how='left')
            df_sf_enriched['dual_match'] = df_sf_enriched['tf_code'].notna().astype(int)
        else:
            # Add empty TF columns and dual flag
            add_tf_cols = ['tf_code', 'tf_name', 'tf_city_id', 'tf_city_name',
                           'tf_marketing_area', 'tf_business_line', 'tf_latitude', 'tf_longitude',
                           'zero_orders', 'available_H', 'availability', 'status_id', 'vendor_status']
            for c in add_tf_cols:
                df_sf_enriched[c] = pd.NA
            df_sf_enriched['dual_match'] = 0

        # Column order: original SF columns first, then all TF columns, then dual flag
        tf_cols_from_tf = [c for c in tf_cols_all if c != 'prelinked_sf_code']
        # Ensure we only add TF columns once and preserve order using tf file as baseline
        tf_cols_for_sf = [c for c in tf_cols_from_tf if c in df_sf_enriched.columns] + \
                         [c for c in ['tf_code', 'tf_name', 'tf_city_id', 'tf_city_name',
                                      'tf_marketing_area', 'tf_business_line', 'tf_latitude', 'tf_longitude',
                                      'zero_orders', 'available_H', 'availability', 'status_id', 'vendor_status']
                          if c in df_sf_enriched.columns and c not in tf_cols_from_tf]

        sf_enrich_order = list(df_sf_raw.columns) + [c for c in tf_cols_for_sf if c not in df_sf_raw.columns] + ['dual_match']
        df_sf_enriched = order_columns(df_sf_enriched, sf_enrich_order)
        _write_csv(df_sf_enriched, SF_ENRICHED_CSV, self.log, "sf_vendors_enriched.csv")

        # 4) unmatched.csv (with reasons & attempts)
        dual_tf_codes = set(df_dual['tf_code'].dropna()) if not df_dual.empty else set()
        df_unmatched = df_tf[~df_tf['tf_code'].isin(dual_tf_codes)].copy()

        def analyze_unmatched_reason(row):
            reasons = []
            if pd.isna(row.get('tf_name')) or str(row.get('tf_name')).strip() == '':
                reasons.append("MISSING_NAME")
            if (pd.isna(row.get('tf_latitude')) or pd.isna(row.get('tf_longitude')) or
                    pd.to_numeric(row.get('tf_latitude'), errors='coerce') == 0 or
                    pd.to_numeric(row.get('tf_longitude'), errors='coerce') == 0):
                reasons.append("MISSING_OR_ZERO_COORDINATES")
            business_line = str(row.get('business_line', '')).strip().lower()
            if business_line in ['other', '']:
                reasons.append("OTHER_OR_MISSING_BUSINESS_LINE")
            status_id = pd.to_numeric(row.get('status_id'), errors='coerce')
            vendor_status = pd.to_numeric(row.get('vendor_status'), errors='coerce')
            if pd.isna(status_id) or status_id != 5:
                reasons.append(f"NON_PRIORITY_STATUS_ID_{int(status_id) if not pd.isna(status_id) else 'UNKNOWN'}")
            if pd.isna(vendor_status) or vendor_status != 1:
                reasons.append(f"INACTIVE_VENDOR_STATUS_{int(vendor_status) if not pd.isna(vendor_status) else 'UNKNOWN'}")
            zero_orders = pd.to_numeric(row.get('zero_orders'), errors='coerce')
            if not pd.isna(zero_orders) and zero_orders > 0:
                reasons.append("HAS_ZERO_ORDERS")
            available_h = pd.to_numeric(row.get('available_H'), errors='coerce')
            if not pd.isna(available_h) and available_h == 0:
                reasons.append("ZERO_AVAILABLE_HOURS")
            city_id = pd.to_numeric(row.get('city_id'), errors='coerce')
            if pd.isna(city_id) or city_id not in [1, 2, 5]:
                reasons.append("UNSUPPORTED_CITY_ID")
            if not reasons:
                reasons.append("NO_SUITABLE_CANDIDATES_IN_SF")
            return " | ".join(reasons)

        def get_match_attempts_summary(row):
            attempts = []
            has_coords = (not pd.isna(row.get('tf_latitude')) and not pd.isna(row.get('tf_longitude')) and
                          pd.to_numeric(row.get('tf_latitude'), errors='coerce') != 0)
            has_name = not pd.isna(row.get('tf_name')) and str(row.get('tf_name')).strip() != ''
            has_valid_bl = str(row.get('business_line', '')).strip().lower() not in ['other', '']
            if has_coords and has_name and has_valid_bl:
                attempts.append("ALGORITHMIC_ELIGIBLE")
            else:
                attempts.append("ALGORITHMIC_INELIGIBLE")
            return " | ".join(attempts) if attempts else "NO_ATTEMPTS"

        if not df_unmatched.empty:
            df_unmatched['unmatched_reason'] = df_unmatched.apply(analyze_unmatched_reason, axis=1)
            df_unmatched['match_attempts'] = df_unmatched.apply(get_match_attempts_summary, axis=1)

        unmatched_cols = [
            'tf_code', 'tf_name', 'city_id', 'city_name', 'marketing_area', 'business_line',
            'tf_latitude', 'tf_longitude', 'zero_orders', 'available_H', 'availability', 'status_id', 'vendor_status',
            'unmatched_reason', 'match_attempts'
        ]
        unmatched_cols = [c for c in unmatched_cols if c in df_unmatched.columns]
        df_unmatched_out = df_unmatched[unmatched_cols].copy()
        df_unmatched_out = order_columns(df_unmatched_out, unmatched_cols)
        _write_csv(df_unmatched_out, UNMATCHED_CSV, self.log, "unmatched.csv")

        # 5) tf_vendors_pro.csv (prelinked_sf_code dropped, add sf_grade or 'Not Found on Snappfood')
        df_tf_pro = df_tf.copy()
        if 'prelinked_sf_code' in df_tf_pro.columns:
            df_tf_pro.drop(columns=['prelinked_sf_code'], inplace=True)

        if not df_dual.empty:
            tf_sf_grade = df_dual[['tf_code', 'sf_grade']].drop_duplicates('tf_code')
            df_tf_pro = df_tf_pro.merge(tf_sf_grade, on='tf_code', how='left')
            df_tf_pro['sf_grade'] = df_tf_pro['sf_grade'].fillna('Not Found on Snappfood')
        else:
            df_tf_pro['sf_grade'] = 'Not Found on Snappfood'

        tf_pro_order = [c for c in df_tf.columns if c != 'prelinked_sf_code'] + ['sf_grade']
        df_tf_pro = order_columns(df_tf_pro, tf_pro_order)
        _write_csv(df_tf_pro, TF_PRO_CSV, self.log, "tf_vendors_pro.csv")

        # 6) x_map_grade.csv - Now includes ALL sf_vendors, not just dual matched ones
        # Columns: sf_vendor_code, sf_vendor_name, city_id, business_line, grade, is_dual, tf_vendor_code, tf_vendor_name

        # Start with all SF vendors
        map_df = df_sf_raw[['sf_code', 'sf_name', 'city_id', 'business_line', 'sf_grade']].copy()
        map_df['tf_code'] = pd.NA
        map_df['tf_name'] = pd.NA
        map_df['is_dual'] = 0

        # Update with dual match information where available
        if not df_dual.empty:
            dual_info = df_dual[['sf_code', 'tf_code', 'tf_name']].drop_duplicates('sf_code')
            map_df = map_df.merge(dual_info, on='sf_code', how='left', suffixes=('', '_dual'))

            # Update tf_code and tf_name for matched vendors
            matched_mask = map_df['tf_code_dual'].notna()
            map_df.loc[matched_mask, 'tf_code'] = map_df.loc[matched_mask, 'tf_code_dual']
            map_df.loc[matched_mask, 'tf_name'] = map_df.loc[matched_mask, 'tf_name_dual']
            map_df.loc[matched_mask, 'is_dual'] = 1

            # Clean up temporary columns
            map_df.drop(columns=['tf_code_dual', 'tf_name_dual'], inplace=True)

        map_df = map_df.rename(columns={
            'sf_code': 'sf_vendor_code',
            'sf_name': 'sf_vendor_name',
            'sf_grade': 'grade',
            'tf_code': 'tf_vendor_code',
            'tf_name': 'tf_vendor_name'
        })
        xmap_order = ['sf_vendor_code', 'sf_vendor_name', 'city_id', 'business_line', 'grade', 'is_dual',
                      'tf_vendor_code', 'tf_vendor_name']
        map_df = order_columns(map_df, xmap_order)
        _write_csv(map_df, X_MAP_GRADE_CSV, self.log, "x_map_grade.csv")

        # Check for null/blank sf_grade in dual_matched_vendors.csv and add sf_codes to snappfood_vendor_codes.csv
        self._update_snappfood_vendor_codes(df_dual)

        # Prelinked failures TXT (as requested) in data/scraped
        if df_prelinked_fail is not None and not df_prelinked_fail.empty:
            lines = [f"Prelinked failures @ {self.ts}",
                     "-" * 60,
                     f"Max allowed distance: {PRELINKED_RADIUS_KM} km",
                     ""]
            for _, r in df_prelinked_fail.iterrows():
                lines.append(f"tf_code={r.get('tf_code')}  sf_code={r.get('sf_code')}  "
                             f"distance_km={r.get('distance_km')}  reason={r.get('reason')}")
            prelinked_fail_path = PRELINKED_FAIL_TXT_DIR / f"prelinked_failures_{self.ts}.txt"
            _write_txt(lines, prelinked_fail_path, self.log, "prelinked failures")

    # ------------------------------------------------------------------------

    def _enrich_from_scraper(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        # (Kept for completeness but not writing any extra outputs.)
        if not ENABLE_SCRAPER_ENRICH or self.scraper is None or df is None or df.empty:
            return df
        need_cols_any = ['sf_name', 'comment_count', 'rating', 'is_express',
                         'sf_latitude', 'sf_longitude', 'city_id',
                         'sf_marketing_area', 'marketing_area',
                         'business_line', 'sf_business_line', 'sf_grade']
        for c in need_cols_any:
            if c not in df.columns:
                df[c] = pd.NA

        def _is_missing_row(r) -> bool:
            checks = [
                (r.get('sf_name'), True),
                (r.get('comment_count'), True),
                (r.get('rating'), True),
                (r.get('is_express'), True),
                (r.get('sf_latitude'), True),
                (r.get('sf_longitude'), True),
                (r.get('city_id'), True),
                (r.get('sf_marketing_area'), True),
                (r.get('business_line'), True),
                (r.get('sf_business_line'), True),
            ]
            for val, _ in checks:
                s = pd.Series([val])
                if s.isna().iloc[0] or str(val).strip() == '':
                    return True
            return False

        grp = df[df['sf_code'].notna()].copy()
        if grp.empty:
            return df
        missing_mask = grp.apply(_is_missing_row, axis=1)
        missing_codes = pd.Series(grp.loc[missing_mask, 'sf_code'], dtype=str).dropna().str.strip().unique().tolist()
        if not missing_codes:
            self.log.info(f"{label}: no SF rows need scraping.")
            return df
        if len(missing_codes) > SCRAPE_MAX_PER_RUN:
            self.log.info(f"{label}: {len(missing_codes)} sf_codes missing; limiting to first {SCRAPE_MAX_PER_RUN}.")
            missing_codes = missing_codes[:SCRAPE_MAX_PER_RUN]

        def _scrape_single_code(code: str) -> Tuple[str, Dict | str]:
            try:
                v = self.scraper.fetch_vendor(code)
                v_norm = v.get("_normalized") or {}
                sf_name = v.get("title") or v.get("name") or pd.NA
                rating = v_norm.get("rating", v.get("rating", None))
                comment_count = v_norm.get("commentCount", v.get("commentCount", None))

                try:
                    rating = float(rating) if rating is not None else np.nan
                except Exception:
                    rating = np.nan
                try:
                    comment_count = int(comment_count) if comment_count is not None else np.nan
                except Exception:
                    comment_count = np.nan

                lat = v.get("lat"); lon = v.get("lon")
                try:
                    lat = float(lat) if lat is not None else np.nan
                except Exception:
                    lat = np.nan
                try:
                    lon = float(lon) if lon is not None else np.nan
                except Exception:
                    lon = np.nan

                is_express = int(bool(self.scraper.get_is_express_from_vendor(v))) if hasattr(self.scraper, 'get_is_express_from_vendor') else pd.NA
                business_line = self.scraper.extract_business_line(v_norm if v_norm else v) if hasattr(self.scraper, 'extract_business_line') else pd.NA
                city_persian = v_norm.get("city") if v_norm.get("city") is not None else v.get("city")
                if hasattr(self.scraper, 'city_id_from_persian_or_coords'):
                    cid = self.scraper.city_id_from_persian_or_coords(city_persian, lat, lon)
                else:
                    cid = None
                city_id = int(cid) if cid is not None else np.nan

                if hasattr(self.scraper, 'point_to_marketing_area'):
                    m_area = self.scraper.point_to_marketing_area(lat, lon, city_id, self.polygons_df)
                else:
                    m_area = "Unknown"

                # No source for sf_grade from scraper; leave as-is
                return code, {
                    'sf_name': sf_name,
                    'sf_latitude': lat,
                    'sf_longitude': lon,
                    'comment_count': comment_count,
                    'rating': rating,
                    'is_express': is_express,
                    'city_id': city_id,
                    'marketing_area': m_area,
                    'business_line': business_line
                }
            except Exception as e:
                return code, str(e)

        patched = 0
        with ThreadPoolExecutor(max_workers=min(SCRAPE_MAX_WORKERS, len(missing_codes))) as executor:
            future_to_code = {executor.submit(_scrape_single_code, code): code for code in missing_codes}
            for future in as_completed(future_to_code):
                code, result = future.result()
                if isinstance(result, str):
                    self.log.warning(f"Scrape enrich failed for {code}: {result}")
                    continue
                idxs = df.index[df['sf_code'] == code].tolist()
                if not idxs:
                    continue

                # patch safely
                for k, v in result.items():
                    if k in ['rating', 'comment_count', 'is_express', 'sf_latitude', 'sf_longitude',
                             'city_id', 'marketing_area', 'business_line', 'sf_name']:
                        df.loc[idxs, k] = df.loc[idxs, k].fillna(v)

                # distance recompute if tf coords exist
                if 'tf_latitude' in df.columns and 'tf_longitude' in df.columns:
                    tf_lat = pd.to_numeric(df.loc[idxs, 'tf_latitude'], errors='coerce')
                    tf_lon = pd.to_numeric(df.loc[idxs, 'tf_longitude'], errors='coerce')
                    sf_lat = pd.to_numeric(df.loc[idxs, 'sf_latitude'], errors='coerce')
                    sf_lon = pd.to_numeric(df.loc[idxs, 'sf_longitude'], errors='coerce')
                    mask = tf_lat.notna() & tf_lon.notna() & sf_lat.notna() & sf_lon.notna()
                    if mask.any():
                        dkm = np.full(len(mask), np.nan, dtype=float)
                        for pos, kidx in enumerate(df.loc[idxs].index):
                            tlat = tf_lat.loc[kidx]; tlon = tf_lon.loc[kidx]; slat = sf_lat.loc[kidx]; slon = sf_lon.loc[kidx]
                            if not (pd.isna(tlat) or pd.isna(tlon) or pd.isna(slat) or pd.isna(slon)):
                                dkm[pos] = float(haversine_vec(np.radians(tlat), np.radians(tlon),
                                                               np.radians(slat), np.radians(slon)))
                        for pos, kidx in enumerate(df.loc[idxs].index):
                            if not np.isnan(dkm[pos]):
                                df.at[kidx, 'distance_km'] = round(dkm[pos], 4)
                patched += 1

        self.log.info(f"{label}: scraper enrich — patched={patched}")
        return df

    def _update_snappfood_vendor_codes(self, df_dual: pd.DataFrame):
        """Check for null/blank sf_grade in dual matches and add sf_codes to snappfood_vendor_codes.csv"""
        if df_dual.empty:
            return

        # Find sf_codes with null or blank sf_grade
        null_grade_mask = (
            df_dual['sf_grade'].isna() |
            (df_dual['sf_grade'].astype(str).str.strip() == '') |
            (df_dual['sf_grade'].astype(str).str.lower() == 'nan')
        )
        sf_codes_with_null_grade = df_dual.loc[null_grade_mask, 'sf_code'].dropna().unique().tolist()

        if not sf_codes_with_null_grade:
            self.log.info("No sf_codes with null/blank sf_grade found in dual matches.")
            return

        self.log.info(f"Found {len(sf_codes_with_null_grade)} sf_codes with null/blank sf_grade")

        # Load existing snappfood_vendor_codes.csv if it exists
        existing_codes = set()
        if SNAPPFOOD_VENDOR_CODES_CSV.exists():
            try:
                existing_df = pd.read_csv(SNAPPFOOD_VENDOR_CODES_CSV, dtype=str)
                if 'sf_code' in existing_df.columns:
                    existing_codes = set(existing_df['sf_code'].dropna().astype(str).str.strip())
                elif len(existing_df.columns) > 0:
                    # Assume first column contains sf_codes
                    existing_codes = set(existing_df.iloc[:, 0].dropna().astype(str).str.strip())
            except Exception as e:
                self.log.warning(f"Could not read existing snappfood_vendor_codes.csv: {e}")

        # Add new sf_codes that aren't already present
        new_codes = [code for code in sf_codes_with_null_grade if str(code).strip() not in existing_codes]

        if new_codes:
            # Append new codes to the file
            all_codes = list(existing_codes) + new_codes
            codes_df = pd.DataFrame({'sf_code': sorted(all_codes)})

            # Ensure directory exists
            SNAPPFOOD_VENDOR_CODES_CSV.parent.mkdir(parents=True, exist_ok=True)

            codes_df.to_csv(SNAPPFOOD_VENDOR_CODES_CSV, index=False, encoding="utf-8-sig")
            self.log.info(f"Added {len(new_codes)} new sf_codes to {SNAPPFOOD_VENDOR_CODES_CSV}")
        else:
            self.log.info("All sf_codes with null/blank sf_grade are already in snappfood_vendor_codes.csv")

    def _safe_concat(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Safely concatenate DataFrames, avoiding FutureWarning for empty/all-NA DataFrames"""
        # Filter out None and empty DataFrames
        valid_frames = [df for df in dataframes if df is not None and not df.empty]

        if not valid_frames:
            return pd.DataFrame()
        elif len(valid_frames) == 1:
            return valid_frames[0].copy()
        else:
            return pd.concat(valid_frames, ignore_index=True)

    def _generate_comprehensive_report(self):
        """Generate comprehensive analysis and report of matches from various sources"""
        self.log.info("\n" + "="*80)
        self.log.info("COMPREHENSIVE MATCHING ANALYSIS REPORT")
        self.log.info("="*80)

        try:
            # Load the output files for analysis
            df_dual = pd.read_csv(DUAL_MATCHED_CSV) if DUAL_MATCHED_CSV.exists() else pd.DataFrame()
            df_tf_enriched = pd.read_csv(TF_ENRICHED_CSV) if TF_ENRICHED_CSV.exists() else pd.DataFrame()
            df_sf_enriched = pd.read_csv(SF_ENRICHED_CSV) if SF_ENRICHED_CSV.exists() else pd.DataFrame()
            df_unmatched = pd.read_csv(UNMATCHED_CSV) if UNMATCHED_CSV.exists() else pd.DataFrame()
            df_x_map = pd.read_csv(X_MAP_GRADE_CSV) if X_MAP_GRADE_CSV.exists() else pd.DataFrame()

            # Overall Statistics
            self.log.info("\n1. OVERALL STATISTICS:")
            self.log.info("-" * 40)
            self.log.info(f"Total Dual Matches: {len(df_dual)}")
            self.log.info(f"Total TF Vendors: {len(df_tf_enriched)}")
            self.log.info(f"Total SF Vendors: {len(df_sf_enriched)}")
            self.log.info(f"Total Unmatched TF: {len(df_unmatched)}")

            if not df_tf_enriched.empty and 'dual_match' in df_tf_enriched.columns:
                match_rate = (df_tf_enriched['dual_match'].sum() / len(df_tf_enriched)) * 100
                self.log.info(f"TF Match Rate: {match_rate:.2f}%")

            if not df_sf_enriched.empty and 'dual_match' in df_sf_enriched.columns:
                sf_match_rate = (df_sf_enriched['dual_match'].sum() / len(df_sf_enriched)) * 100
                self.log.info(f"SF Match Rate: {sf_match_rate:.2f}%")

            # Source Analysis
            if not df_dual.empty and 'source' in df_dual.columns:
                self.log.info("\n2. MATCHES BY SOURCE:")
                self.log.info("-" * 40)
                source_counts = df_dual['source'].value_counts()
                for source, count in source_counts.items():
                    percentage = (count / len(df_dual)) * 100
                    self.log.info(f"{source.upper()}: {count} matches ({percentage:.1f}%)")

            # City Analysis
            if not df_dual.empty and 'city_id' in df_dual.columns:
                self.log.info("\n3. MATCHES BY CITY:")
                self.log.info("-" * 40)
                city_counts = df_dual['city_id'].value_counts()
                city_map = {'1': 'Mashhad', '2': 'Tehran', '5': 'Shiraz'}
                for city_id, count in city_counts.items():
                    city_name = city_map.get(str(city_id), f"City {city_id}")
                    percentage = (count / len(df_dual)) * 100
                    self.log.info(f"{city_name} (ID: {city_id}): {count} matches ({percentage:.1f}%)")

            # Business Line Analysis
            if not df_dual.empty and 'business_line' in df_dual.columns:
                self.log.info("\n4. MATCHES BY BUSINESS LINE:")
                self.log.info("-" * 40)
                bl_counts = df_dual['business_line'].value_counts().head(10)
                for bl, count in bl_counts.items():
                    percentage = (count / len(df_dual)) * 100
                    self.log.info(f"{bl}: {count} matches ({percentage:.1f}%)")

            # Distance Analysis
            if not df_dual.empty and 'distance_km' in df_dual.columns:
                distances = pd.to_numeric(df_dual['distance_km'], errors='coerce').dropna()
                if not distances.empty:
                    self.log.info("\n5. DISTANCE ANALYSIS:")
                    self.log.info("-" * 40)
                    self.log.info(f"Average Distance: {distances.mean():.4f} km")
                    self.log.info(f"Median Distance: {distances.median():.4f} km")
                    self.log.info(f"Max Distance: {distances.max():.4f} km")
                    self.log.info(f"Min Distance: {distances.min():.4f} km")

                    # Distance ranges
                    range_0_01 = (distances <= 0.01).sum()
                    range_01_05 = ((distances > 0.01) & (distances <= 0.05)).sum()
                    range_05_1 = ((distances > 0.05) & (distances <= 0.1)).sum()
                    range_1_2 = ((distances > 0.1) & (distances <= 2.0)).sum()
                    range_2_plus = (distances > 2.0).sum()

                    self.log.info(f"≤ 0.01 km: {range_0_01} matches")
                    self.log.info(f"0.01-0.05 km: {range_01_05} matches")
                    self.log.info(f"0.05-0.1 km: {range_05_1} matches")
                    self.log.info(f"0.1-2.0 km: {range_1_2} matches")
                    self.log.info(f"> 2.0 km: {range_2_plus} matches")

            # Fuzzy Score Analysis
            if not df_dual.empty and 'fuzzy_score' in df_dual.columns:
                fuzzy_scores = pd.to_numeric(df_dual['fuzzy_score'], errors='coerce').dropna()
                if not fuzzy_scores.empty:
                    self.log.info("\n6. FUZZY SCORE ANALYSIS:")
                    self.log.info("-" * 40)
                    self.log.info(f"Average Fuzzy Score: {fuzzy_scores.mean():.2f}")
                    self.log.info(f"Median Fuzzy Score: {fuzzy_scores.median():.2f}")
                    self.log.info(f"Min Fuzzy Score: {fuzzy_scores.min():.2f}")

                    # Score ranges
                    perfect = (fuzzy_scores == 100).sum()
                    excellent = ((fuzzy_scores >= 95) & (fuzzy_scores < 100)).sum()
                    good = ((fuzzy_scores >= 90) & (fuzzy_scores < 95)).sum()
                    acceptable = ((fuzzy_scores >= 85) & (fuzzy_scores < 90)).sum()

                    self.log.info(f"Perfect (100): {perfect} matches")
                    self.log.info(f"Excellent (95-99): {excellent} matches")
                    self.log.info(f"Good (90-94): {good} matches")
                    self.log.info(f"Acceptable (85-89): {acceptable} matches")

            # Grade Analysis
            if not df_x_map.empty and 'grade' in df_x_map.columns:
                self.log.info("\n7. SF GRADE DISTRIBUTION:")
                self.log.info("-" * 40)
                grade_counts = df_x_map['grade'].value_counts()
                for grade, count in grade_counts.items():
                    percentage = (count / len(df_x_map)) * 100
                    self.log.info(f"{grade}: {count} vendors ({percentage:.1f}%)")

            # Unmatched Analysis
            if not df_unmatched.empty:
                self.log.info("\n8. UNMATCHED TF VENDORS ANALYSIS:")
                self.log.info("-" * 40)
                if 'unmatched_reason' in df_unmatched.columns:
                    reason_counts = df_unmatched['unmatched_reason'].value_counts().head(10)
                    for reason, count in reason_counts.items():
                        percentage = (count / len(df_unmatched)) * 100
                        self.log.info(f"{reason}: {count} vendors ({percentage:.1f}%)")

            # Status Analysis
            if not df_dual.empty and 'status_id' in df_dual.columns:
                self.log.info("\n9. MATCHED TF VENDORS BY STATUS:")
                self.log.info("-" * 40)
                status_counts = df_dual['status_id'].value_counts()
                for status, count in status_counts.items():
                    percentage = (count / len(df_dual)) * 100
                    self.log.info(f"Status {status}: {count} matches ({percentage:.1f}%)")

            # Quality Metrics
            self.log.info("\n10. QUALITY METRICS:")
            self.log.info("-" * 40)

            if not df_dual.empty:
                # High quality matches (close distance + high fuzzy score)
                if 'distance_km' in df_dual.columns and 'fuzzy_score' in df_dual.columns:
                    distances = pd.to_numeric(df_dual['distance_km'], errors='coerce')
                    fuzzy_scores = pd.to_numeric(df_dual['fuzzy_score'], errors='coerce')

                    high_quality = ((distances <= 0.05) & (fuzzy_scores >= 95)).sum()
                    medium_quality = ((distances <= 0.2) & (fuzzy_scores >= 90)).sum()

                    hq_percentage = (high_quality / len(df_dual)) * 100
                    mq_percentage = (medium_quality / len(df_dual)) * 100

                    self.log.info(f"High Quality Matches (≤0.05km & ≥95% fuzzy): {high_quality} ({hq_percentage:.1f}%)")
                    self.log.info(f"Medium Quality Matches (≤0.2km & ≥90% fuzzy): {medium_quality} ({mq_percentage:.1f}%)")

            self.log.info("\n" + "="*80)
            self.log.info("END OF ANALYSIS REPORT")
            self.log.info("="*80)

        except Exception as e:
            self.log.error(f"Error generating comprehensive report: {e}")


# --------------------------------------------------------------------------------------
#                                      MAIN
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    DualMatcherV18().run()

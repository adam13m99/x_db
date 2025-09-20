# X-Database: Vendor Management & Matching System

A comprehensive data pipeline for collecting, processing, and matching vendor information between TapsiFood (TF) and Snappfood (SF) platforms. This system implements a 5-step workflow that fetches vendor data, extracts codes, scrapes detailed information, applies grading algorithms, and performs intelligent vendor matching.

## 🏗️ Architecture Overview

The system follows a sequential processing pipeline designed around the database schema shown in `X_DATABASE_SCHEMA.png`:

```
Step 0: TF Data Fetching → Step 1A: SF Code Extraction → Step 1B: SF Data Scraping → Step 2: Vendor Grading → Step 3: Dual Matching
```

### Core Components

1. **Data Fetcher** (`Step 0`): Retrieves TapsiFood vendor data from Metabase
2. **Code Extractor** (`Step 1A`): Consolidates Snappfood vendor codes from multiple sources
3. **Vendor Scraper** (`Step 1B`): Scrapes detailed Snappfood vendor information
4. **Grader System** (`Step 2`): Applies configurable grading algorithms to vendors
5. **Dual Matcher** (`Step 3`): Matches vendors across platforms using fuzzy logic and geospatial analysis

## 📋 Prerequisites

### Required Dependencies
```bash
pip install pandas numpy requests shapely scikit-learn rapidfuzz
```

### System Requirements
- Python 3.7+
- 8GB+ RAM (recommended for large datasets)
- Stable internet connection for API calls
- Access to Metabase instance

### Configuration Setup
1. **Metabase Credentials**: Update `config.py` with your Metabase connection details
2. **API Access**: Ensure access to Snappfood APIs
3. **Polygon Data**: Place marketing area polygon files in `data/polygons/`

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and navigate to project
cd "X - Database"

# Install dependencies
pip install -r requirements.txt  # if available, or install manually

# Verify configuration
python -c "import config; print('Config loaded successfully')"
```

### 2. Execute Pipeline
```bash
# Run complete pipeline
python "Step 0 - fetch_tf_vendors.py"
python "Step 1 - sf_code_extractor.py"
python "Step 1 - sf_vendors_scraper.py"
python "Step 2 - grader_v4.py"
python "Step 3 - dual_matcher_v18.py"
```

## 📁 Directory Structure

```
X - Database/
├── README.md                           # This documentation
├── config.py                          # Metabase configuration
├── mini.py                            # Metabase client utility
├── X_DATABASE_SCHEMA.png              # Visual database schema
│
├── Step 0 - fetch_tf_vendors.py       # TapsiFood data fetcher
├── Step 1 - sf_code_extractor.py      # Snappfood code consolidator
├── Step 1 - sf_vendors_scraper.py     # Snappfood data scraper
├── Step 2 - grader_v4.py              # Vendor grading system
├── Step 3 - dual_matcher_v18.py       # Cross-platform matcher
│
├── data/                              # Input data and configurations
│   ├── polygons/                     # Marketing area polygon files
│   ├── extra_matches/                # Manual vendor matches
│   └── scraped/                      # Scraped and processed data
│
└── outputs/                          # Generated results
    ├── tf_vendors.csv                # TapsiFood vendor data
    ├── sf_vendors.csv                # Snappfood vendor data
    ├── dual_matched_vendors.csv      # Successfully matched vendors
    ├── tf_vendors_enriched.csv       # Enhanced TF data
    ├── sf_vendors_enriched.csv       # Enhanced SF data
    └── unmatched.csv                 # Unmatched vendors
```

## 🔄 Detailed Workflow

### Step 0: TapsiFood Data Fetching
**File**: `Step 0 - fetch_tf_vendors.py`

Fetches vendor data from Metabase and performs initial data processing:

- **Data Source**: Metabase question ID 6257
- **Processing**: Schema normalization, date sanitization, geographic assignment
- **Output**: `outputs/tf_vendors.csv` with standardized vendor information

**Key Features**:
- Point-in-polygon marketing area assignment using Shapely
- Automatic date cleaning (removes placeholder '1970-01-01' dates)
- Zero-filling for missing numeric fields
- UTF-8 with BOM encoding for Excel compatibility

### Step 1A: Snappfood Code Extraction
**File**: `Step 1 - sf_code_extractor.py`

Consolidates Snappfood vendor codes from multiple sources:

- **Input Sources**:
  - Scraped vendor files (`V_sf_vendor_*.csv`)
  - Extra code files (`extra_sf_codes*.csv`)
  - Previous extractions (`snappfood_vendor_codes.csv`)
  - Manual matches (`extra_matches.csv`)
- **Processing**: Deduplication, cleaning, TF code conflict resolution
- **Output**: `data/scraped/snappfood_vendor_codes.csv`

**Filtering Logic**:
- Removes codes that exactly match TapsiFood codes
- Eliminates codes containing TapsiFood codes as substrings
- Maintains audit trail of removed conflicts

### Step 1B: Snappfood Data Scraping
**File**: `Step 1 - sf_vendors_scraper.py`

Scrapes detailed vendor information from Snappfood APIs:

- **API Endpoints**:
  - Detail API: `snappfood.ir/mobile/v2/restaurant/details/dynamic`
  - List API: `snappfood.ir/search/api/v1/desktop/vendors-list` (fallback)
- **Rate Limiting**: Configurable delays to respect API limits
- **Error Handling**: Failed scrapes logged to `data/failed_to_scrape.csv`
- **Geographic Enhancement**: City and marketing area enrichment

**Data Enrichment**:
- Business line classification
- Express delivery status detection
- Rating and comment count extraction
- Geographic coordinate validation

### Step 2: Vendor Grading
**File**: `Step 2 - grader_v4.py`

Applies configurable grading algorithm to Snappfood vendors:

**Grading Criteria**:
- **Eligibility**: Minimum 300 comments required
- **Rating Bands**: A (>9.0) → B (8.4-9.0) → C (7.9-8.4) → D (7.0-7.9) → E (6.0-7.0) → F (<6.0)
- **Volume Adjustments**: Top/bottom 25% by comment count get grade modifications
- **City-Business Segmentation**: Separate grading within each city-business line combination

**Configuration Options**:
- Adjustable comment thresholds
- Customizable rating band boundaries
- Configurable top/bottom percentages
- Flexible eligibility criteria

### Step 3: Dual Platform Matching
**File**: `Step 3 - dual_matcher_v18.py`

Matches vendors across TapsiFood and Snappfood platforms using advanced algorithms:

**Matching Strategies**:
1. **Prelinked Matches**: Direct SF code matches from TF data
2. **Fuzzy Name Matching**: RapidFuzz-based name similarity (threshold: 85%)
3. **Geographic Proximity**: BallTree-based coordinate matching (500m radius)
4. **Manual Overrides**: User-defined matches from `extra_matches.csv`

**Output Generation**:
- **Dual Matched**: Successfully matched vendor pairs
- **TF Enriched**: TapsiFood vendors with Snappfood data
- **SF Enriched**: Snappfood vendors with TapsiFood data
- **Unmatched**: Vendors without cross-platform matches
- **TF Pro**: Enhanced TapsiFood dataset with grades
- **X Map Grade**: Grade mapping reference table

## ⚙️ Configuration

### Metabase Settings (`config.py`)
```python
METABASE_URL = "https://metabase.ofood.cloud"
METABASE_USERNAME = "your_username@domain.com"
METABASE_PASSWORD = "your_password"
```

### Grading Parameters (`Step 2 - grader_v4.py`)
```python
MIN_COMMENTS = 300              # Minimum comments for eligibility
TOP_PERCENT = 0.25              # Top 25% get grade boost
BOTTOM_PERCENT = 0.25           # Bottom 25% get grade penalty
RATING_A_GT = 9.0              # A grade threshold
# Additional rating band configurations...
```

### Matching Thresholds (`Step 3 - dual_matcher_v18.py`)
```python
FUZZY_THRESHOLD = 85            # Name similarity threshold
DISTANCE_THRESHOLD_M = 500      # Geographic proximity (meters)
MAX_WORKERS = 4                 # Concurrent processing threads
```

## 📊 Output Files

### Primary Outputs

| File | Description | Key Columns |
|------|-------------|-------------|
| `tf_vendors.csv` | TapsiFood vendor master data | `tf_code`, `tf_name`, `tf_latitude`, `tf_longitude` |
| `sf_vendors.csv` | Snappfood vendor master data | `sf_code`, `name`, `latitude`, `longitude`, `sf_grade` |
| `dual_matched_vendors.csv` | Cross-platform matched pairs | `tf_code`, `sf_code`, `match_method`, `confidence` |
| `tf_vendors_enriched.csv` | TF vendors with SF data | TF columns + `sf_grade`, `sf_rating` |
| `sf_vendors_enriched.csv` | SF vendors with TF data | SF columns + TF operational metrics |
| `unmatched.csv` | Unmatched vendors | `code`, `platform`, `name`, `reason` |

### Auxiliary Outputs

| File | Description | Usage |
|------|-------------|-------|
| `tf_vendors_pro.csv` | Enhanced TF dataset | Advanced analytics |
| `x_map_grade.csv` | Grade mapping table | Grade interpretation |
| `snappfood_vendor_codes.csv` | Consolidated SF codes | Code management |

## 🔧 Troubleshooting

### Common Issues

**1. Metabase Connection Failed**
```bash
# Verify credentials and network access
python -c "from mini import fetch_question_data; print('Testing connection...')"
```

**2. Shapely Import Error**
```bash
# Install or reinstall Shapely
pip uninstall shapely
pip install shapely
```

**3. API Rate Limiting**
- Increase delay between requests in scraper configuration
- Monitor `data/failed_to_scrape.csv` for persistent failures
- Consider running scraper in smaller batches

**4. Memory Issues with Large Datasets**
- Process data in chunks (modify `page_size` in fetcher)
- Increase system memory allocation
- Use streaming processing for very large files

**5. Geographic Assignment Failures**
- Verify polygon files exist in `data/polygons/`
- Check WKT format validity in polygon CSV files
- Ensure coordinate system consistency (WGS84)

### Performance Optimization

**For Large Datasets**:
- Increase `workers` parameter in fetcher (default: 8)
- Adjust `MAX_WORKERS` in matcher (default: 4)
- Use SSD storage for faster I/O operations
- Consider database indexing for frequent queries

**For API Efficiency**:
- Batch API requests where possible
- Implement caching for repeated calls
- Use connection pooling for sustained operations
- Monitor and respect API rate limits

## 🔄 Maintenance

### Regular Tasks

**Daily**:
- Monitor failed scrape logs
- Check data freshness timestamps
- Verify output file generation

**Weekly**:
- Update vendor code extractions
- Review matching accuracy metrics
- Clean up temporary files

**Monthly**:
- Audit and update manual matches
- Review and adjust grading parameters
- Update polygon boundaries if needed
- Performance optimization review

### Data Quality Checks

```bash
# Check for missing required columns
python -c "import pandas as pd; df=pd.read_csv('outputs/tf_vendors.csv'); print(df.isnull().sum())"

# Verify geographic coordinates
python -c "import pandas as pd; df=pd.read_csv('outputs/sf_vendors.csv'); print(f'Invalid coords: {((df.latitude < -90) | (df.latitude > 90) | (df.longitude < -180) | (df.longitude > 180)).sum()}')"

# Check matching statistics
python -c "import pandas as pd; df=pd.read_csv('outputs/dual_matched_vendors.csv'); print(f'Total matches: {len(df)}, Methods: {df.match_method.value_counts()}')"
```

**Last Updated**: September 2024
**Pipeline Version**: v18
**Compatibility**: Python 3.7+
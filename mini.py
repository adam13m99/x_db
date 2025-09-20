import requests
import pandas as pd
import logging
import concurrent.futures
from typing import Optional
from dataclasses import dataclass
from config import METABASE_URL, METABASE_USERNAME, METABASE_PASSWORD

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MetabaseConfig:
    """Configuration for Metabase connection"""
    url: str
    username: str
    password: str
    database_name: str
    database_id: Optional[int] = None

    @classmethod
    def create_with_team_db(cls, url: str, username: str, password: str, team: str):
        """Creates config with a team-specific database name for convenience."""
        team_databases = {
            'growth': 'Growth Team Clickhouse Connection',
            'data': 'Data Team Clickhouse Connection',
            'product': 'Product Team Clickhouse Connection'
        }
        if team.lower() not in team_databases:
            raise ValueError(f"Invalid team. Choose from: {list(team_databases.keys())}")
        
        return cls(
            url=url,
            username=username,
            password=password,
            database_name=team_databases[team.lower()]
        )

# --- Core Metabase Client ---
class MetabaseClient:
    """An optimized client for interacting with the Metabase API."""
    
    def __init__(self, config: MetabaseConfig):
        self.config = config
        self.session = requests.Session()
        self.session_token = None
        self.database_id = config.database_id
    
    def authenticate(self) -> bool:
        """Authenticate with Metabase and store the session token."""
        try:
            auth_url = f"{self.config.url}/api/session"
            auth_data = {"username": self.config.username, "password": self.config.password}
            response = self.session.post(auth_url, json=auth_data, timeout=30)
            response.raise_for_status()
            
            self.session_token = response.json().get('id')
            self.session.headers.update({'X-Metabase-Session': self.session_token})
            
            logger.info("Successfully authenticated with Metabase.")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def get_database_id(self) -> Optional[int]:
        """Find the database ID using its name."""
        if self.database_id:
            return self.database_id
        try:
            databases_url = f"{self.config.url}/api/database"
            response = self.session.get(databases_url)
            response.raise_for_status()
            
            databases = response.json().get('data', [])
            for db in databases:
                if db.get('name') == self.config.database_name:
                    self.database_id = db.get('id')
                    logger.info(f"Found database ID: {self.database_id} for '{self.config.database_name}'")
                    return self.database_id
            
            logger.error(f"Database '{self.config.database_name}' not found.")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get database ID: {e}")
            return None

    def execute_query(self, sql_query: str, timeout: int = 300, max_results: int = 100000) -> Optional[pd.DataFrame]:
        """Execute a single, raw SQL query. Assumes prior authentication."""
        try:
            query_payload = {
                "type": "native",
                "native": {"query": sql_query},
                "database": self.database_id,
                "constraints": {"max-results": max_results, "max-results-bare-rows": max_results}
            }
            
            query_url = f"{self.config.url}/api/dataset"
            response = self.session.post(query_url, json=query_payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            if result.get('status') != 'completed':
                logger.error(f"Query failed. Status: {result.get('status')}. Error: {result.get('error')}")
                return None
            
            data = result.get('data', {})
            rows = data.get('rows', [])
            columns = [col['name'] for col in data.get('cols', [])]
            return pd.DataFrame(rows, columns=columns)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Query execution failed: {e}")
            return None
            
    def execute_query_with_parallel_pagination(self, sql_query: str, page_size: int = 50000, max_workers: int = 8) -> Optional[pd.DataFrame]: # CHANGED: Increased default workers
        """Fetch all results for a query using an OPTIMIZED parallel pagination method."""
        logger.info(f"🚀 Executing with OPTIMIZED parallel pagination ({max_workers} workers)...")
        
        # We need the database_id for the count query.
        if not self.database_id and not self.get_database_id():
            return None

        # 1. Get total row count to calculate pages
        count_query = f"SELECT COUNT(*) as total_rows FROM ({sql_query.rstrip(';')}) as subquery"
        count_df = self.execute_query(count_query)
        if count_df is None or count_df.empty:
            logger.error("Failed to get total row count. Cannot proceed with parallel fetch.")
            return None
            
        total_rows = count_df.iloc[0]['total_rows']
        if total_rows == 0:
            logger.info("Query returned 0 rows.")
            return pd.DataFrame()
        
        total_pages = (total_rows + page_size - 1) // page_size
        logger.info(f"📊 Total rows: {total_rows:,}, Pages: {total_pages}, Page size: {page_size:,}")
        
        # CHANGED: The worker function now accepts the session token and db_id to prevent re-authentication.
        def fetch_page(page_num: int, session_token: str, db_id: int) -> Optional[pd.DataFrame]:
            """Fetch a single page of data using pre-authenticated credentials."""
            try:
                # Use a new session for thread safety, but with the existing token.
                with requests.Session() as thread_session:
                    thread_session.headers.update({'X-Metabase-Session': session_token})
                    
                    offset = page_num * page_size
                    paginated_query = f"{sql_query.rstrip(';')} LIMIT {page_size} OFFSET {offset}"
                    
                    query_payload = {
                        "type": "native",
                        "native": {"query": paginated_query},
                        "database": db_id,
                        "constraints": {"max-results": page_size, "max-results-bare-rows": page_size}
                    }

                    query_url = f"{self.config.url}/api/dataset"
                    response = thread_session.post(query_url, json=query_payload, timeout=300)
                    response.raise_for_status()

                    result = response.json()
                    data = result.get('data', {})
                    rows = data.get('rows', [])
                    columns = [col['name'] for col in data.get('cols', [])]
                    df = pd.DataFrame(rows, columns=columns)

                    logger.info(f"✅ Page {page_num + 1}/{total_pages} fetched ({len(df):,} rows)")
                    return df
            except Exception as e:
                logger.error(f"Error fetching page {page_num + 1}: {e}")
            return None

        # 2. Execute all page fetches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # CHANGED: Pass the token and db_id to each worker.
            futures = [executor.submit(fetch_page, i, self.session_token, self.database_id) for i in range(total_pages)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 3. Combine results
        valid_dataframes = [df for df in results if df is not None]
        if not valid_dataframes:
            logger.error("No data retrieved from any page.")
            return None
            
        final_df = pd.concat(valid_dataframes, ignore_index=True)
        logger.info(f"🎉 Parallel fetch complete. Total rows retrieved: {len(final_df):,}")
        
        if len(final_df) != total_rows:
            logger.warning(f"⚠️ Mismatch in row count! Expected {total_rows}, got {len(final_df)}. Some pages may have failed.")
            
        return final_df

    def get_question_details(self, question_id: int) -> Optional[dict]:
        """Retrieve the details (including the SQL) of a saved question."""
        try:
            url = f"{self.config.url}/api/card/{question_id}"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get details for question {question_id}: {e}")
            return None

    def logout(self):
        """Log out from the Metabase session."""
        if self.session_token:
            try:
                self.session.delete(f"{self.config.url}/api/session")
                logger.info("Successfully logged out.")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Logout failed: {e}")
            finally:
                self.session_token = None

# --- Main High-Level Function ---

def fetch_question_data(
    question_id: int, 
    metabase_url: str, 
    username: str, 
    password: str, 
    team: str = "growth",
    workers: int = 8,  
    page_size: int = 50000
) -> Optional[pd.DataFrame]:
    """
    Fetches all data for a given Metabase saved question using OPTIMIZED parallel processing.
    """
    logger.info(f"🚀 Starting optimized fetch for Metabase question ID: {question_id}")
    config = MetabaseConfig.create_with_team_db(url=metabase_url, username=username, password=password, team=team)
    client = MetabaseClient(config)

    try:
        if not client.authenticate(): return None
        details = client.get_question_details(question_id)
        if not details: return None

        native_query_data = details.get('dataset_query', {}).get('native')
        if not native_query_data or 'query' not in native_query_data:
            logger.error(f"Question {question_id} is not a native SQL query. Cannot extract SQL.")
            return None
        
        sql_query = native_query_data['query']
        logger.info(f"Successfully extracted SQL from question '{details.get('name', 'N/A')}'.")

        return client.execute_query_with_parallel_pagination(sql_query, page_size=page_size, max_workers=workers)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return None
    finally:
        client.logout()

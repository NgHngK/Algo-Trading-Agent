import gc
import hashlib
import json
import math
import os
import re
import time
from datetime import time as clock_time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm


# --- CONFIG ---
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_REVISION = os.getenv("FINBERT_REVISION", "4556d13015211d73dccd3fdd39d39232506f3e43")
MARKET_TZ = "America/New_York"
MARKET_CLOSE = clock_time(16, 0)
MAX_COMPANIES = 50
MIN_TRAIN_NEWS = 40
MIN_TRAIN_PRICE_DAYS = 250
LOOKBACK_CANDIDATES = [1, 3, 7, 15, 30]
MAX_LOOKBACK = max(LOOKBACK_CANDIDATES)
# Price-change features need one previous row, so every lookback starts after the same warm-up.
REQUIRED_HISTORY = MAX_LOOKBACK + 1
CACHE_VERSION = 6

# FNSPID news is a modern dataset. A malformed historical timestamp must not
# shift the whole train split decades backwards. Override for another dataset.
NEWS_MIN_DATE = pd.Timestamp(os.getenv("FNSPID_NEWS_MIN_DATE", "1998-01-01"))
NEWS_MAX_DATE = pd.Timestamp.now().normalize() + pd.Timedelta(days=366)
MAX_NEWS_TO_TRADING_GAP_DAYS = 7

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_DIR / "outputs"

UNIVERSE_CACHE = CACHE_DIR / "company_universe.csv"
NEWS_CACHE = CACHE_DIR / "panel_news.parquet"
SENTIMENT_CACHE = CACHE_DIR / "panel_news_sentiment.parquet"
DAILY_CACHE = CACHE_DIR / "panel_daily_sentiment.parquet"
PANEL_CACHE = CACHE_DIR / "panel_base.parquet"
META_FILE = CACHE_DIR / "cache_meta.json"


# --- SYSTEM ---
def get_system_info():
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    ram = psutil.virtual_memory()

    thread_cap = max(1, int(os.getenv("MAX_CPU_THREADS", "8")))
    safe_threads = max(1, min(physical, logical - 1 if logical > 2 else logical, thread_cap))
    if ram.available < 4 * 1024**3:
        safe_threads = min(safe_threads, 2)

    info = {
        "physical_cores": physical,
        "logical_cores": logical,
        "ram_total_gb": ram.total / 1024**3,
        "ram_available_gb": ram.available / 1024**3,
        "safe_threads": safe_threads,
        "cuda": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_free_gb": 0.0,
        "vram_total_gb": 0.0,
    }

    if info["cuda"]:
        free_vram, total_vram = torch.cuda.mem_get_info()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_free_gb"] = free_vram / 1024**3
        info["vram_total_gb"] = total_vram / 1024**3

    return info


def print_system_info():
    info = get_system_info()
    print("\nSystem")
    print("------")
    print(f"CPU physical: {info['physical_cores']}")
    print(f"CPU logical: {info['logical_cores']}")
    print(f"RAM available: {info['ram_available_gb']:.2f} GB")
    print(f"CUDA: {info['cuda']}")
    print(f"GPU: {info['gpu_name'] or 'None'}")
    print(f"VRAM free: {info['vram_free_gb']:.2f} GB")
    print(f"Safe CPU threads: {info['safe_threads']}")
    return info


# --- PATHS ---
def find_fnspid_dir():
    env_path = os.getenv("FNSPID_DIR")
    candidates = []

    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([
        DATA_DIR / "FNSPID",
        PROJECT_DIR / "FNSPID",
        PROJECT_DIR.parent / "data" / "FNSPID",
    ])

    for path in candidates:
        news_file = path / "Stock_news" / "nasdaq_exteral_data.csv"
        price_dir = path / "Stock_price" / "full_history"
        if news_file.exists() and price_dir.exists():
            return path

    raise FileNotFoundError(
        "FNSPID was not found. Put it in data/FNSPID or set FNSPID_DIR."
    )


def get_data_paths(fnspid_dir=None):
    if fnspid_dir is None:
        fnspid_dir = find_fnspid_dir()

    fnspid_dir = Path(fnspid_dir)
    price_dir = fnspid_dir / "Stock_price" / "full_history"

    # The official FNSPID archive may contain one extra full_history folder:
    # Stock_price/full_history/full_history/*.csv
    # Support both layouts so users do not need to move thousands of files.
    if not any(price_dir.glob("*.csv")):
        nested_price_dir = price_dir / "full_history"
        if nested_price_dir.is_dir() and any(nested_price_dir.glob("*.csv")):
            price_dir = nested_price_dir

    return {
        "root": fnspid_dir,
        "news": fnspid_dir / "Stock_news" / "nasdaq_exteral_data.csv",
        "price_dir": price_dir,
    }


def get_huggingface_cache_dir(fnspid_dir=None):
    if fnspid_dir is None:
        fnspid_dir = find_fnspid_dir()

    cache_dir = Path(fnspid_dir) / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# --- SMALL HELPERS ---
def _normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_column(columns, aliases, required=True):
    normalized = {_normalize_name(col): col for col in columns}
    for alias in aliases:
        key = _normalize_name(alias)
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(f"Could not find a column matching: {aliases}")
    return None


def _parse_datetime(values, utc=False):
    try:
        return pd.to_datetime(values, errors="coerce", utc=utc, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce", utc=utc)


def _load_meta():
    if not META_FILE.exists():
        return {}
    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_meta(meta):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _source_signature(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _directory_signature(path, pattern="*.csv"):
    hasher = hashlib.blake2b(digest_size=16)
    count = 0
    for file in sorted(path.glob(pattern), key=lambda x: x.name):
        stat = file.stat()
        row = f"{file.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8")
        hasher.update(row)
        count += 1
    return {"path": str(path.resolve()), "files": count, "digest": hasher.hexdigest()}


def _dataframe_fingerprint(df, columns=None):
    if columns is None:
        columns = list(df.columns)
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return {"rows": int(len(df)), "columns": [], "digest": "empty"}

    normalized = df[columns].copy()
    for col in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[col]):
            normalized[col] = pd.to_datetime(normalized[col], errors="coerce").astype("string")
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.blake2b(hashed.tobytes(), digest_size=16).hexdigest()
    return {"rows": int(len(df)), "columns": columns, "digest": digest}


def _save_table(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        fallback = path.with_suffix(".pkl")
        if fallback.exists():
            fallback.unlink()
        return path
    except (ImportError, ModuleNotFoundError):
        fallback = path.with_suffix(".pkl")
        df.to_pickle(fallback)
        return fallback


def _load_table(path):
    if path.exists():
        return pd.read_parquet(path)
    fallback = path.with_suffix(".pkl")
    if fallback.exists():
        return pd.read_pickle(fallback)
    raise FileNotFoundError(path)


def _table_exists(path):
    return path.exists() or path.with_suffix(".pkl").exists()


def choose_chunk_size():
    ram_gb = psutil.virtual_memory().available / 1024**3
    if ram_gb < 4:
        return 50_000
    if ram_gb < 8:
        return 100_000
    if ram_gb < 16:
        return 200_000
    return 300_000


def _valid_news_dates(dates):
    naive = dates.dt.tz_convert(None)
    return naive.notna() & (naive >= NEWS_MIN_DATE) & (naive <= NEWS_MAX_DATE)


# --- INPUT SCHEMA ---
def inspect_news_schema(news_file, nrows=8):
    sample = pd.read_csv(news_file, nrows=nrows, low_memory=False)
    print("\nNews CSV")
    print("--------")
    print("Columns:", sample.columns.tolist())
    print("Dtypes:", sample.dtypes.astype(str).to_dict())
    print(sample.head())
    return sample


def detect_news_columns(news_file):
    columns = pd.read_csv(news_file, nrows=2, low_memory=False).columns.tolist()

    detected = {
        "index": _find_column(columns, ["Unnamed: 0", "index"], False),
        "date": _find_column(columns, ["Date", "publishedAt", "timestamp"]),
        "title": _find_column(columns, ["Article_title", "title", "headline"]),
        "symbol": _find_column(columns, ["Stock_sym", "Stock_symbol", "symbol", "ticker"]),
        "url": _find_column(columns, ["Url", "url", "link"], False),
        "publisher": _find_column(columns, ["Publisher", "publisher", "source"], False),
        "author": _find_column(columns, ["Author", "author"], False),
        "article": _find_column(columns, ["Article", "article", "content", "text"], False),
        "lsa_summary": _find_column(columns, ["Lsa_summary"], False),
        "luhn_summary": _find_column(columns, ["Luhn_summary"], False),
        "textrank_summary": _find_column(columns, ["Textrank_summary"], False),
        "lexrank_summary": _find_column(columns, ["Lexrank_summary"], False),
    }

    print("\nDetected news columns")
    print("---------------------")
    for key, value in detected.items():
        print(f"{key}: {value}")

    return detected


def inspect_price_schema(price_file, nrows=8):
    sample = pd.read_csv(price_file, nrows=nrows)
    print("\nPrice CSV")
    print("---------")
    print("Columns:", sample.columns.tolist())
    print("Dtypes:", sample.dtypes.astype(str).to_dict())
    print(sample.head())
    return sample


# --- UNIVERSE ---
def _duckdb_date_expr(date_col):
    safe = date_col.replace('"', '""')
    return f"TRY_CAST(REPLACE(CAST(\"{safe}\" AS VARCHAR), ' UTC', '') AS TIMESTAMP)"


def _news_date_range_duckdb(news_file, columns, threads):
    try:
        import duckdb
    except ImportError:
        print("DuckDB is not installed in this Jupyter kernel; using slower pandas chunks.")
        return None

    path_sql = str(news_file).replace("'", "''")
    date_expr = _duckdb_date_expr(columns["date"])

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={max(1, threads)}")
    memory_mb = max(256, int(psutil.virtual_memory().available * 0.55 / 1024**2))
    con.execute(f"SET memory_limit='{memory_mb}MB'")

    query = f"""
        SELECT MIN(ts) AS min_date, MAX(ts) AS max_date
        FROM (
            SELECT {date_expr} AS ts
            FROM read_csv_auto(
                '{path_sql}', header=true, all_varchar=true,
                ignore_errors=true, sample_size=200000, parallel=true
            )
        )
        WHERE ts IS NOT NULL
          AND ts >= TIMESTAMP '{NEWS_MIN_DATE.strftime("%Y-%m-%d %H:%M:%S")}'
          AND ts <= TIMESTAMP '{NEWS_MAX_DATE.strftime("%Y-%m-%d %H:%M:%S")}'
    """

    try:
        row = con.execute(query).fetchone()
        con.close()
        if row and row[0] is not None and row[1] is not None:
            return pd.Timestamp(row[0]), pd.Timestamp(row[1])
    except Exception as e:
        con.close()
        print(f"DuckDB date scan failed: {type(e).__name__}: {e}")
    return None


def _news_date_range_pandas(news_file, columns):
    date_col = columns["date"]
    min_date = None
    max_date = None
    ignored_out_of_range = 0
    chunk_size = choose_chunk_size()

    for chunk in tqdm(
        pd.read_csv(
            news_file,
            usecols=[date_col],
            chunksize=chunk_size,
            dtype=str,
            on_bad_lines="skip",
        ),
        desc="Scan news dates",
    ):
        parsed = _parse_datetime(chunk[date_col], utc=True)
        valid = _valid_news_dates(parsed)
        ignored_out_of_range += int((parsed.notna() & ~valid).sum())
        if not valid.any():
            continue
        naive = parsed.dt.tz_convert(None)
        cmin = naive[valid].min()
        cmax = naive[valid].max()
        min_date = cmin if min_date is None else min(min_date, cmin)
        max_date = cmax if max_date is None else max(max_date, cmax)

    if min_date is None:
        raise ValueError("Could not parse any news dates inside the valid date range.")
    if ignored_out_of_range:
        print(f"Ignored out-of-range news timestamps: {ignored_out_of_range:,}")
        print(f"Valid date window: {NEWS_MIN_DATE.date()} to {NEWS_MAX_DATE.date()}")
    return pd.Timestamp(min_date), pd.Timestamp(max_date)


def get_news_date_range(news_file, columns=None):
    if columns is None:
        columns = detect_news_columns(news_file)

    info = get_system_info()
    result = _news_date_range_duckdb(news_file, columns, info["safe_threads"])
    if result is None:
        result = _news_date_range_pandas(news_file, columns)

    print(f"News start: {result[0]}")
    print(f"News end: {result[1]}")
    return result


def _training_cutoff(start, end, train_ratio=0.70):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    return start + (end - start) * train_ratio


def _ticker_counts_duckdb(news_file, columns, train_end, limit, threads):
    try:
        import duckdb
    except ImportError:
        print("DuckDB is not installed in this Jupyter kernel; using slower pandas chunks.")
        return None

    path_sql = str(news_file).replace("'", "''")
    sym = columns["symbol"].replace('"', '""')
    date_expr = _duckdb_date_expr(columns["date"])
    cutoff = pd.Timestamp(train_end).strftime("%Y-%m-%d %H:%M:%S")

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={max(1, threads)}")
    memory_mb = max(256, int(psutil.virtual_memory().available * 0.55 / 1024**2))
    con.execute(f"SET memory_limit='{memory_mb}MB'")

    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    query = f"""
        SELECT ticker, COUNT(*) AS train_news
        FROM (
            SELECT UPPER(TRIM(CAST(\"{sym}\" AS VARCHAR))) AS ticker,
                   {date_expr} AS ts
            FROM read_csv_auto(
                '{path_sql}', header=true, all_varchar=true,
                ignore_errors=true, sample_size=200000, parallel=true
            )
        )
        WHERE ts IS NOT NULL
          AND ts >= TIMESTAMP '{NEWS_MIN_DATE.strftime("%Y-%m-%d %H:%M:%S")}'
          AND ts <= TIMESTAMP '{cutoff}'
          AND REGEXP_MATCHES(ticker, '^[A-Z][A-Z0-9.-]{{0,9}}$')
        GROUP BY ticker
        HAVING COUNT(*) >= {int(MIN_TRAIN_NEWS)}
        ORDER BY train_news DESC
        {limit_sql}
    """

    try:
        result = con.execute(query).df()
        con.close()
        return result
    except Exception as e:
        con.close()
        print(f"DuckDB ticker scan failed: {type(e).__name__}: {e}")
        return None


def _ticker_counts_pandas(news_file, columns, train_end, limit):
    usecols = [columns["symbol"], columns["date"]]
    chunk_size = choose_chunk_size()
    counts = {}
    cutoff = pd.Timestamp(train_end, tz="UTC") if pd.Timestamp(train_end).tzinfo is None else pd.Timestamp(train_end)

    for chunk in tqdm(
        pd.read_csv(
            news_file,
            usecols=usecols,
            chunksize=chunk_size,
            dtype=str,
            on_bad_lines="skip",
        ),
        desc="Count train news",
    ):
        ticker = chunk[columns["symbol"]].astype(str).str.upper().str.strip()
        dates = _parse_datetime(chunk[columns["date"]], utc=True)
        mask = _valid_news_dates(dates) & (dates <= cutoff) & ticker.str.match(r"^[A-Z][A-Z0-9.\-]{0,9}$")
        vc = ticker[mask].value_counts()
        for key, value in vc.items():
            counts[key] = counts.get(key, 0) + int(value)

    items = [(ticker, count) for ticker, count in counts.items() if count >= MIN_TRAIN_NEWS]
    items.sort(key=lambda x: x[1], reverse=True)
    if limit:
        items = items[:limit]
    return pd.DataFrame(items, columns=["ticker", "train_news"])


def _price_info(price_file, train_start, train_end):
    try:
        df = pd.read_csv(price_file, usecols=lambda c: _normalize_name(c) == "date")
    except Exception:
        df = pd.read_csv(price_file)

    date_col = _find_column(df.columns, ["date"])
    dates = _parse_datetime(df[date_col], utc=True).dt.tz_convert(None).dropna()
    train_mask = (dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))
    train_days = int(dates[train_mask].nunique())
    return train_days, dates.min(), dates.max()


def _build_price_file_index(price_dir):
    index = {}
    for file in Path(price_dir).glob("*.csv"):
        stem = file.stem.upper().strip()
        for alias in {stem, stem.replace("-", "."), stem.replace(".", "-")}:
            index.setdefault(alias, file)
    return index


def _find_price_file(price_index, ticker):
    ticker = str(ticker).upper().strip()
    return (
        price_index.get(ticker)
        or price_index.get(ticker.replace(".", "-"))
        or price_index.get(ticker.replace("-", "."))
    )


def select_company_universe(fnspid_dir=None, mode="full", force=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = get_data_paths(fnspid_dir)
    news_file = paths["news"]
    price_dir = paths["price_dir"]
    columns = detect_news_columns(news_file)
    meta = _load_meta()

    source_sig = _source_signature(news_file)
    cache_key = {
        "cache_version": CACHE_VERSION,
        "source": source_sig,
        "price_dir": _directory_signature(price_dir),
        "max_companies": MAX_COMPANIES,
        "min_train_news": MIN_TRAIN_NEWS,
        "min_train_price_days": MIN_TRAIN_PRICE_DAYS,
        "news_min_date": str(NEWS_MIN_DATE.date()),
        "mode": mode,
    }

    if not force and UNIVERSE_CACHE.exists() and meta.get("universe", {}).get("key") == cache_key:
        print("Using cached company universe.")
        return pd.read_csv(UNIVERSE_CACHE), pd.Timestamp(meta["universe"]["train_end"])

    start_date, end_date = get_news_date_range(news_file, columns)
    train_end = _training_cutoff(start_date, end_date)
    print(f"Universe selection uses news only through: {train_end.date()}")

    info = get_system_info()
    # The news CSV must be scanned anyway. In full mode keep every ticker that
    # passes the news threshold, then check price availability/history.
    limit = 200 if mode == "smoke" else None
    counts = _ticker_counts_duckdb(news_file, columns, train_end, limit, info["safe_threads"])
    if counts is None:
        counts = _ticker_counts_pandas(news_file, columns, train_end, limit)

    rows = []
    target_count = 8 if mode == "smoke" else MAX_COMPANIES
    price_index = _build_price_file_index(price_dir)
    missing_price = 0
    short_history = 0
    bad_price = 0

    for _, row in counts.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        price_file = _find_price_file(price_index, ticker)
        if price_file is None:
            missing_price += 1
            continue

        try:
            train_days, price_start, price_end = _price_info(price_file, start_date, train_end)
        except Exception as e:
            bad_price += 1
            if bad_price <= 5:
                print(f"Skipping {ticker}: bad price file ({type(e).__name__}: {e})")
            continue
        if train_days < MIN_TRAIN_PRICE_DAYS and mode != "smoke":
            short_history += 1
            continue

        rows.append({
            "ticker": ticker,
            "train_news": int(row["train_news"]),
            "train_price_days": int(train_days),
            "price_start": str(price_start),
            "price_end": str(price_end),
        })

        if len(rows) >= target_count:
            break

    print("\nUniverse filter audit")
    print("---------------------")
    print(f"News-qualified candidates checked: {len(counts):,}")
    print(f"Rejected - missing price file: {missing_price:,}")
    print(f"Rejected - < {MIN_TRAIN_PRICE_DAYS} train price days: {short_history:,}")
    print(f"Rejected - unreadable price file: {bad_price:,}")
    print(f"Accepted: {len(rows):,} / target {target_count}")

    universe = pd.DataFrame(rows)
    if len(universe) < 3:
        raise ValueError(
            "Too few eligible companies were found after date/news/price filters. "
            "Check the Universe filter audit above."
        )

    universe.to_csv(UNIVERSE_CACHE, index=False)
    meta["universe"] = {
        "key": cache_key,
        "train_end": str(train_end),
    }
    _save_meta(meta)

    print("\nCompany universe")
    print("----------------")
    print(f"Companies: {len(universe)}")
    print(universe.head(20).to_string(index=False))
    return universe, train_end


# --- NEWS EXTRACTION ---
def _extract_news_duckdb(news_file, columns, tickers, mode, threads):
    try:
        import duckdb
    except ImportError:
        return None

    path_sql = str(news_file).replace("'", "''")
    tickers_sql = ", ".join("'" + str(t).replace("'", "''") + "'" for t in tickers)

    selected = []
    mapping = {
        "date": "published_raw",
        "title": "title",
        "symbol": "ticker",
        "url": "url",
    }

    for key, out in mapping.items():
        col = columns.get(key)
        if col:
            safe = col.replace('"', '""')
            selected.append(f'CAST("{safe}" AS VARCHAR) AS "{out}"')

    symbol = columns["symbol"].replace('"', '""')
    limit_sql = " LIMIT 5000" if mode == "smoke" else ""

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={max(1, threads)}")
    memory_mb = max(256, int(psutil.virtual_memory().available * 0.55 / 1024**2))
    con.execute(f"SET memory_limit='{memory_mb}MB'")

    query = f"""
        SELECT {', '.join(selected)}
        FROM read_csv_auto(
            '{path_sql}', header=true, all_varchar=true,
            ignore_errors=true, sample_size=200000, parallel=true
        )
        WHERE UPPER(TRIM(CAST(\"{symbol}\" AS VARCHAR))) IN ({tickers_sql})
        {limit_sql}
    """

    try:
        df = con.execute(query).df()
        con.close()
        return df
    except Exception as e:
        con.close()
        print(f"DuckDB extraction failed: {type(e).__name__}: {e}")
        return None


def _extract_news_pandas(news_file, columns, tickers, mode):
    usecols = [columns[key] for key in ["date", "title", "symbol", "url"] if columns.get(key)]
    chunk_size = choose_chunk_size()
    ticker_set = {str(t).upper() for t in tickers}
    parts = []
    total = 0

    rename = {
        columns["date"]: "published_raw",
        columns["title"]: "title",
        columns["symbol"]: "ticker",
    }
    if columns.get("url"):
        rename[columns["url"]] = "url"

    reader = pd.read_csv(
        news_file,
        usecols=usecols,
        chunksize=chunk_size,
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    )

    for chunk in tqdm(reader, desc="Extract panel news"):
        ticker = chunk[columns["symbol"]].astype(str).str.upper().str.strip()
        small = chunk[ticker.isin(ticker_set)]
        if small.empty:
            continue
        parts.append(small.rename(columns=rename))
        total += len(small)
        if mode == "smoke" and total >= 5000:
            break

    if not parts:
        raise ValueError("No panel news was found.")
    return pd.concat(parts, ignore_index=True)


def _clean_panel_news(df):
    before = len(df)
    df = df.copy()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.strip()
    df["published_raw"] = df["published_raw"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()

    if "url" not in df.columns:
        df["url"] = ""
    df["url"] = df["url"].fillna("").astype(str)

    df = df[(df["ticker"] != "") & (df["published_raw"] != "") & (df["title"] != "")].copy()
    df["_published_ts"] = _parse_datetime(df["published_raw"], utc=True)
    valid_dates = _valid_news_dates(df["_published_ts"])
    bad_dates = int((~valid_dates).sum())
    if bad_dates:
        print(f"Dropped implausible/unparsed panel-news dates: {bad_dates:,}")
    df = df[valid_dates].copy()
    df = df.sort_values(["ticker", "_published_ts"]).drop_duplicates()

    has_url = df["url"].str.strip() != ""
    # Keep the earliest occurrence, not whichever duplicate happens to appear
    # first in the raw CSV.
    with_url = df[has_url].drop_duplicates(["ticker", "url"], keep="first")

    no_url = df[~has_url].copy()
    no_url["title_key"] = no_url["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    no_url["dedup_key"] = no_url["ticker"] + "|" + no_url["title_key"] + "|" + no_url["published_raw"]
    no_url = no_url.drop_duplicates("dedup_key", keep="first")
    no_url = no_url.drop(columns=["title_key", "dedup_key"])

    df = pd.concat([with_url, no_url], ignore_index=True)
    df = df.sort_values(["ticker", "_published_ts"]).reset_index(drop=True)
    df = df.drop(columns=["_published_ts"])

    print("\nPanel news")
    print("----------")
    print(f"Rows before cleaning: {before:,}")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Rows removed: {before - len(df):,}")
    print(f"Companies: {df['ticker'].nunique()}")
    return df


def extract_panel_news(fnspid_dir=None, mode="full", force=False):
    paths = get_data_paths(fnspid_dir)
    news_file = paths["news"]
    columns = detect_news_columns(news_file)
    universe, _ = select_company_universe(fnspid_dir, mode=mode, force=False)
    tickers = universe["ticker"].tolist()

    meta = _load_meta()
    cache_key = {
        "cache_version": CACHE_VERSION,
        "source": _source_signature(news_file),
        "tickers": tickers,
        "mode": mode,
    }

    if not force and _table_exists(NEWS_CACHE) and meta.get("panel_news") == cache_key:
        print("Using cached panel news.")
        return _load_table(NEWS_CACHE)

    info = get_system_info()
    start = time.perf_counter()
    df = _extract_news_duckdb(news_file, columns, tickers, mode, info["safe_threads"])
    method = "DuckDB"
    if df is None:
        method = "pandas chunks"
        df = _extract_news_pandas(news_file, columns, tickers, mode)

    df = _clean_panel_news(df)
    saved = _save_table(df, NEWS_CACHE)
    elapsed = time.perf_counter() - start

    print(f"Extraction method: {method}")
    print(f"Saved: {saved}")
    print(f"Extraction time: {elapsed:.2f} sec")

    meta["panel_news"] = cache_key
    _save_meta(meta)
    return df


# --- PRICES ---
def load_price_file(price_file, ticker):
    df = pd.read_csv(price_file)
    columns = df.columns.tolist()

    rename = {
        _find_column(columns, ["date"]): "date",
        _find_column(columns, ["open"]): "open",
        _find_column(columns, ["high"]): "high",
        _find_column(columns, ["low"]): "low",
        _find_column(columns, ["close"]): "close",
        _find_column(columns, ["volume"]): "volume",
    }

    adj = _find_column(columns, ["adj close", "adj_close", "adjusted close"], False)
    if adj:
        rename[adj] = "adj_close"

    df = df[list(rename.keys())].rename(columns=rename)
    df["date"] = _parse_datetime(df["date"], utc=True).dt.tz_convert(None).dt.normalize()

    for col in ["open", "high", "low", "close", "volume", "adj_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("date", keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    df["ticker"] = ticker
    return df


def load_panel_prices(fnspid_dir=None, mode="full"):
    paths = get_data_paths(fnspid_dir)
    universe, _ = select_company_universe(fnspid_dir, mode=mode, force=False)
    parts = []
    price_index = _build_price_file_index(paths["price_dir"])

    for ticker in tqdm(universe["ticker"], desc="Load prices"):
        price_file = _find_price_file(price_index, ticker)
        if price_file is None:
            continue
        part = load_price_file(price_file, ticker)
        parts.append(part)

    if not parts:
        raise ValueError("No stock price files were loaded.")

    prices = pd.concat(parts, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices


# --- NEWS TIME ALIGNMENT ---
def _is_date_only_value(raw):
    raw = str(raw).strip()
    if not raw:
        return False
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", raw):
        return True
    return bool(re.fullmatch(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]00:00:00(?:\.0+)?(?:\s*(?:UTC|Z|[+-]\d{2}:?\d{2}))?",
        raw,
        flags=re.I,
    ))


def audit_timestamp_quality(news):
    raw = news["published_raw"].fillna("").astype(str).str.strip()
    parsed = _parse_datetime(raw, utc=True)
    valid = parsed.notna()
    date_only = raw.map(_is_date_only_value)
    date_only_ratio = float(date_only[valid].mean()) if valid.any() else 0.0

    hours = parsed[valid & ~date_only].dt.hour.value_counts().sort_index()
    print("\nTimestamp audit")
    print("---------------")
    print(f"Rows: {len(news):,}")
    print(f"Parsed: {int(valid.sum()):,}")
    print(f"Unparsed: {int((~valid).sum()):,}")
    print(f"Date-only ratio: {date_only_ratio:.2%}")
    print("UTC hour counts for precise timestamps:")
    print(hours.to_string())

    return {
        "rows": int(len(news)),
        "parsed": int(valid.sum()),
        "unparsed": int((~valid).sum()),
        "date_only_ratio": date_only_ratio,
        "distinct_precise_hours": int(hours.size),
    }


def _next_trading_date(day, trading_dates, side="right"):
    day = pd.Timestamp(day).normalize()
    pos = trading_dates.searchsorted(day, side=side)
    if pos >= len(trading_dates):
        return pd.NaT
    next_day = pd.Timestamp(trading_dates[pos]).normalize()
    if next_day - day > pd.Timedelta(days=MAX_NEWS_TO_TRADING_GAP_DAYS):
        return pd.NaT
    return trading_dates[pos]


def _safe_feature_date(raw_value, trading_dates):
    raw = str(raw_value).strip()
    if not raw:
        return pd.NaT

    # Date-only or exact-midnight values have unknown publication time.
    # Keep them conservative and use the next trading session.
    if _is_date_only_value(raw):
        day = pd.to_datetime(raw[:10], errors="coerce")
        if pd.isna(day):
            return pd.NaT
        return _next_trading_date(day, trading_dates, side="right")

    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(ts):
        day = pd.to_datetime(raw[:10], errors="coerce")
        if pd.isna(day):
            return pd.NaT
        return _next_trading_date(day, trading_dates, side="right")

    local = ts.tz_convert(MARKET_TZ)
    local_day = pd.Timestamp(local.date())
    local_time = local.time().replace(tzinfo=None)

    # Exactly at the close is treated as too late for the close-based forecast.
    side = "left" if local_time < MARKET_CLOSE else "right"
    return _next_trading_date(local_day, trading_dates, side=side)


def assign_feature_dates(news, prices):
    price_dates = {
        ticker: pd.DatetimeIndex(group["date"].sort_values().unique())
        for ticker, group in prices.groupby("ticker", sort=False)
    }

    parts = []
    for ticker, group in tqdm(news.groupby("ticker", sort=False), desc="Align news dates"):
        dates = price_dates.get(ticker)
        if dates is None or len(dates) == 0:
            continue
        part = group.copy()
        part["feature_date"] = [
            _safe_feature_date(raw, dates) for raw in part["published_raw"].tolist()
        ]
        parts.append(part.dropna(subset=["feature_date"]))

    if not parts:
        return news.iloc[0:0].assign(feature_date=pd.Series(dtype="datetime64[ns]"))
    out = pd.concat(parts, ignore_index=True)
    out["feature_date"] = pd.to_datetime(out["feature_date"])
    return out


# --- FINBERT ---
def load_finbert():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise ImportError("Install transformers before running FinBERT.") from e

    info = get_system_info()
    device = torch.device("cuda" if info["cuda"] else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(info["safe_threads"])
    hf_cache_dir = get_huggingface_cache_dir()

    tokenizer = AutoTokenizer.from_pretrained(
        FINBERT_MODEL,
        revision=FINBERT_REVISION,
        use_fast=True,
        cache_dir=str(hf_cache_dir),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL,
        revision=FINBERT_REVISION,
        cache_dir=str(hf_cache_dir),
    )
    model.to(device)
    model.eval()

    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    pos_idx = next(k for k, v in labels.items() if v == "positive")
    neg_idx = next(k for k, v in labels.items() if v == "negative")
    neu_idx = next(k for k, v in labels.items() if v == "neutral")

    return tokenizer, model, device, labels, pos_idx, neg_idx, neu_idx


def choose_max_length(texts, tokenizer):
    sample = texts[: min(1000, len(texts))]
    lengths = [len(tokenizer.encode(x, add_special_tokens=True, truncation=False)) for x in sample]
    p99 = int(np.percentile(lengths, 99)) if lengths else 64
    return int(min(128, max(32, math.ceil(p99 / 8) * 8)))


def _finbert_batch(model, tokenizer, texts, device, max_length):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**encoded).logits
        else:
            logits = model(**encoded).logits
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()

    del encoded
    return probs


def benchmark_finbert_batch(model, tokenizer, texts, device, max_length):
    info = get_system_info()
    if device.type == "cuda":
        free = info["vram_free_gb"]
        candidates = [8, 16, 32, 64] if free < 6 else [16, 32, 64, 128]
    else:
        candidates = [2, 4, 8] if info["ram_available_gb"] < 4 else [4, 8, 16, 32]

    sample = texts[: min(256, len(texts))]
    results = []

    for batch_size in candidates:
        if batch_size > len(sample):
            continue
        try:
            batch = sample[:batch_size]
            _finbert_batch(model, tokenizer, batch, device, max_length)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _finbert_batch(model, tokenizer, batch, device, max_length)
            if device.type == "cuda":
                torch.cuda.synchronize()
            speed = batch_size / max(time.perf_counter() - start, 1e-6)
            print(f"Batch {batch_size}: {speed:.1f} articles/sec")
            results.append((batch_size, speed))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if device.type == "cuda" and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise

    return max(results, key=lambda x: x[1])[0] if results else 1


def run_finbert(news, prices, mode="full", force=False):
    meta = _load_meta()
    cache_key = {
        "cache_version": CACHE_VERSION,
        "model": FINBERT_MODEL,
        "model_revision": FINBERT_REVISION,
        "news": _dataframe_fingerprint(news, ["ticker", "published_raw", "title", "url"]),
        "price_calendar": _dataframe_fingerprint(prices, ["ticker", "date"]),
        "mode": mode,
        "market_tz": MARKET_TZ,
        "market_close": MARKET_CLOSE.strftime("%H:%M"),
    }

    if not force and _table_exists(SENTIMENT_CACHE) and meta.get("sentiment") == cache_key:
        print("Using cached FinBERT sentiment.")
        return _load_table(SENTIMENT_CACHE)

    aligned = assign_feature_dates(news, prices)
    texts = aligned["title"].fillna("").astype(str).str.strip()
    aligned = aligned[texts != ""].reset_index(drop=True)
    texts = aligned["title"].tolist()

    if mode == "smoke":
        aligned = aligned.head(300).copy()
        texts = aligned["title"].tolist()

    tokenizer, model, device, labels, pos_idx, neg_idx, neu_idx = load_finbert()
    print("FinBERT labels:", labels)
    print("FinBERT device:", device)

    max_length = choose_max_length(texts, tokenizer)
    print(f"Tokenizer max length: {max_length}")
    batch_size = benchmark_finbert_batch(model, tokenizer, texts, device, max_length)
    print(f"FinBERT batch size: {batch_size}")

    pos, neg, neu = [], [], []
    start = time.perf_counter()
    i = 0

    with tqdm(total=len(texts), desc="FinBERT") as bar:
        while i < len(texts):
            batch = texts[i:i + batch_size]
            try:
                probs = _finbert_batch(model, tokenizer, batch, device, max_length)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if device.type == "cuda" and "out of memory" in str(e).lower() and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    print(f"CUDA OOM. New batch size: {batch_size}")
                    continue
                raise

            pos.extend(probs[:, pos_idx].tolist())
            neg.extend(probs[:, neg_idx].tolist())
            neu.extend(probs[:, neu_idx].tolist())
            i += len(batch)
            bar.update(len(batch))

    elapsed = time.perf_counter() - start
    aligned["sentiment_pos"] = np.asarray(pos, dtype=np.float32)
    aligned["sentiment_neg"] = np.asarray(neg, dtype=np.float32)
    aligned["sentiment_neu"] = np.asarray(neu, dtype=np.float32)
    aligned["sentiment_score"] = (aligned["sentiment_pos"] - aligned["sentiment_neg"]).astype(np.float32)

    check = _finbert_batch(
        model,
        tokenizer,
        ["Apple's Q2 revenue exceeded analysts' estimates."],
        device,
        max_length,
    )[0]
    print("FinBERT sanity check:", {
        "positive": float(check[pos_idx]),
        "negative": float(check[neg_idx]),
        "neutral": float(check[neu_idx]),
    })

    saved = _save_table(aligned, SENTIMENT_CACHE)
    print(f"Saved: {saved}")
    print(f"FinBERT time: {elapsed:.2f} sec")
    print(f"FinBERT speed: {len(aligned) / max(elapsed, 1e-6):.1f} articles/sec")

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    meta["sentiment"] = cache_key
    _save_meta(meta)
    return aligned


# --- DAILY DATA ---
def build_daily_sentiment(sentiment_df, force=False, mode="full"):
    meta = _load_meta()
    cache_key = {
        "cache_version": CACHE_VERSION,
        "mode": mode,
        "sentiment": _dataframe_fingerprint(
            sentiment_df,
            ["ticker", "feature_date", "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_score"],
        ),
    }

    if not force and _table_exists(DAILY_CACHE) and meta.get("daily") == cache_key:
        print("Using cached daily sentiment.")
        return _load_table(DAILY_CACHE)

    daily = (
        sentiment_df.groupby(["ticker", "feature_date"])
        .agg(
            sentiment_sum=("sentiment_score", "sum"),
            sentiment_pos_sum=("sentiment_pos", "sum"),
            sentiment_neg_sum=("sentiment_neg", "sum"),
            sentiment_neu_sum=("sentiment_neu", "sum"),
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            news_count=("sentiment_score", "size"),
        )
        .reset_index()
        .rename(columns={"feature_date": "date"})
    )

    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    saved = _save_table(daily, DAILY_CACHE)
    print(f"Saved: {saved}")

    meta["daily"] = cache_key
    _save_meta(meta)
    return daily


def build_panel_base(prices, daily_sentiment, force=False, mode="full"):
    meta = _load_meta()
    cache_key = {
        "cache_version": CACHE_VERSION,
        "mode": mode,
        "prices": _dataframe_fingerprint(prices, ["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]),
        "daily": _dataframe_fingerprint(
            daily_sentiment,
            ["ticker", "date", "sentiment_sum", "sentiment_pos_sum", "sentiment_neg_sum", "sentiment_neu_sum",
             "sentiment_mean", "sentiment_std", "news_count"],
        ),
    }

    if not force and _table_exists(PANEL_CACHE) and meta.get("panel") == cache_key:
        print("Using cached panel dataset.")
        panel = _load_table(PANEL_CACHE)
        panel["date"] = pd.to_datetime(panel["date"])
        panel["target_date"] = pd.to_datetime(panel["target_date"], errors="coerce")
        return panel

    df = prices.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", group_keys=False)

    # Price features stay inside each company.
    df["prev_close"] = g["close"].shift(1)
    df["prev_volume"] = g["volume"].shift(1)
    df["daily_return"] = df["close"] / df["prev_close"] - 1
    df["open_gap"] = df["open"] / df["prev_close"] - 1
    df["intraday_return"] = df["close"] / df["open"] - 1
    df["high_low_range"] = df["high"] / df["low"] - 1
    df["volume_change"] = df["volume"] / df["prev_volume"] - 1
    df["history_count"] = g.cumcount() + 1

    # Target stays inside each company.
    df["target_open"] = g["open"].shift(-1)
    df["target_date"] = g["date"].shift(-1)
    df["target_return"] = df["target_open"] / df["close"] - 1

    keep_daily = daily_sentiment.copy()
    keep_daily["date"] = pd.to_datetime(keep_daily["date"]).dt.normalize()
    df = df.merge(keep_daily, on=["ticker", "date"], how="left")

    fill_zero = [
        "sentiment_sum", "sentiment_pos_sum", "sentiment_neg_sum", "sentiment_neu_sum",
        "sentiment_mean", "sentiment_std", "news_count",
    ]
    for col in fill_zero:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    saved = _save_table(df, PANEL_CACHE)
    print(f"Saved: {saved}")
    print(f"Panel rows: {len(df):,}")
    print(f"Panel companies: {df['ticker'].nunique()}")

    meta["panel"] = cache_key
    _save_meta(meta)
    return df


# --- LOOKBACK FEATURES ---
def model_feature_columns():
    return [
        "daily_return",
        "open_gap",
        "intraday_return",
        "high_low_range",
        "volume_change",
        "sentiment_mean",
        "sentiment_std",
        "news_count",
        "return_mean_lb",
        "return_vol_lb",
        "volume_change_mean_lb",
        "sentiment_mean_lb",
        "sentiment_std_lb",
        "news_count_lb",
    ]


def make_lookback_features(panel, lookback, min_history=REQUIRED_HISTORY):
    df = panel.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    group = df.groupby("ticker", group_keys=False)

    df["return_mean_lb"] = group["daily_return"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).mean()
    )
    df["return_vol_lb"] = group["daily_return"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).std(ddof=0)
    )
    df["volume_change_mean_lb"] = group["volume_change"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).mean()
    )
    df["sentiment_roll_sum"] = group["sentiment_sum"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).sum()
    )
    df["news_count_lb"] = group["news_count"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).sum()
    )
    df["sentiment_mean_lb"] = np.where(
        df["news_count_lb"] > 0,
        df["sentiment_roll_sum"] / df["news_count_lb"],
        0.0,
    )
    df["sentiment_std_lb"] = group["sentiment_mean"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).std(ddof=0)
    )

    df = df[df["history_count"] >= min_history].copy()
    df = df.drop(columns=["sentiment_roll_sum"])
    return df


# --- FULL PREPROCESSING ---
def prepare_all(mode="full", force=False, run_sentiment=True):
    start = time.perf_counter()
    print_system_info()
    fnspid_dir = find_fnspid_dir()
    paths = get_data_paths(fnspid_dir)

    print("\nInput files")
    print("-----------")
    print("News:", paths["news"])
    print("Price folder:", paths["price_dir"])

    inspect_news_schema(paths["news"])
    columns = detect_news_columns(paths["news"])
    universe, train_end = select_company_universe(fnspid_dir, mode=mode, force=force)
    inspect_price_schema(paths["price_dir"] / f"{universe.iloc[0]['ticker']}.csv")

    news = extract_panel_news(fnspid_dir, mode=mode, force=force)
    timestamp_info = audit_timestamp_quality(news)
    prices = load_panel_prices(fnspid_dir, mode=mode)

    if run_sentiment:
        sentiment = run_finbert(news, prices, mode=mode, force=force)
        daily = build_daily_sentiment(sentiment, force=force, mode=mode)
    else:
        raise ValueError("run_sentiment=False is only for custom testing.")

    panel = build_panel_base(prices, daily, force=force, mode=mode)

    summary = {
        "mode": mode,
        "companies": int(panel["ticker"].nunique()),
        "rows": int(len(panel)),
        "date_start": str(panel["date"].min()),
        "date_end": str(panel["date"].max()),
        "news_rows": int(len(news)),
        "universe_train_end": str(train_end),
        "timestamp_audit": timestamp_info,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPreprocessing summary")
    print("---------------------")
    for key, value in summary.items():
        if key != "timestamp_audit":
            print(f"{key}: {value}")
    print(f"Total time: {time.perf_counter() - start:.2f} sec")
    return panel


if __name__ == "__main__":
    mode = os.getenv("MODE", "full").lower()
    prepare_all(mode=mode)

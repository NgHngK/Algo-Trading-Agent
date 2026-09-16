import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from sentiment import get_sentiment
from sklearn.ensemble import RandomForestRegressor

# --- CONFIG ---
TICKER = "AAPL"
NEWS_FILE = "news_AAPL_OR_Apple_Inc_2026-08-16_2026-09-16.json"
DATE_START = "2026-08-16"
DATE_END = "2026-09-16"
MARKET_CLOSE = time(16, 0)


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns into single-level strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(part) for part in col if part and str(part) != "nan")
            for col in df.columns.to_list()
        ]
    return df


def make_text(row):
    parts = [row.get("title"), row.get("description"), row.get("content")]
    parts = [x for x in parts if isinstance(x, str) and x.strip()]
    return " ".join(parts)


def make_model():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def walk_forward(data, features, start_index, end_index, name):
    model_errors = []
    baseline_errors = []

    print(f"\n=== {name} walk-forward ===")

    for i in range(start_index, end_index):
        train = data.iloc[:i]
        test = data.iloc[[i]]

        model = make_model()
        model.fit(train[features], train["target_return"])

        pred_return = model.predict(test[features])[0]
        current_close = test["close"].iloc[0]
        pred_open = current_close * (1 + pred_return)
        actual_open = test["target_open"].iloc[0]
        baseline_open = current_close

        model_errors.append((pred_open - actual_open) ** 2)
        baseline_errors.append((baseline_open - actual_open) ** 2)

        feature_date = test["date"].iloc[0].date()
        target_date = test["target_date"].iloc[0].date()

        print(
            f"{feature_date} -> {target_date}: "
            f"Pred ${pred_open:.2f}, Act ${actual_open:.2f}, "
            f"Baseline ${baseline_open:.2f}"
        )

    model_rmse = np.sqrt(np.mean(model_errors))
    baseline_rmse = np.sqrt(np.mean(baseline_errors))

    print(f"{name} model RMSE: ${model_rmse:.4f}")
    print(f"{name} baseline RMSE: ${baseline_rmse:.4f}")

    return model_rmse, baseline_rmse


# 1) Load news
articles = pd.read_json(NEWS_FILE)["articles"]
df = pd.json_normalize(articles)

if df.empty:
    raise ValueError("No articles found in the news file.")

if df["publishedAt"].astype(str).str.len().max() <= 10:
    raise ValueError(
        "This news file lost the original timestamps. "
        "Run fetch_news.py again with the updated code."
    )

# 2) Keep news known by market close and compute sentiment
df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
df = df.dropna(subset=["publishedAt"]).copy()
df["published_ny"] = df["publishedAt"].dt.tz_convert("America/New_York")
df = df[df["published_ny"].dt.time <= MARKET_CLOSE].copy()
df["date"] = df["published_ny"].dt.tz_localize(None).dt.normalize()

df["text"] = df.apply(make_text, axis=1)
df = df[df["text"].str.len() > 0].copy()
df["sentiment"] = df["text"].apply(get_sentiment)

sent_df = pd.DataFrame(df["sentiment"].tolist(), index=df.index)
df = pd.concat([df, sent_df], axis=1)

# 3) Aggregate daily sentiment
daily_sent = (
    df.groupby("date")
    .agg(
        score_mean=("score", "mean"),
        score_count=("score", "count"),
        pos_mean=("pos", "mean"),
        neg_mean=("neg", "mean"),
    )
    .reset_index()
)
print(f"Daily sentiment days: {len(daily_sent)} rows")

# 4) Download price data
price_end = datetime.strptime(DATE_END, "%Y-%m-%d") + timedelta(days=8)
raw_prices = yf.download(
    TICKER,
    start=DATE_START,
    end=price_end.strftime("%Y-%m-%d"),
    auto_adjust=False,
    progress=False,
)
raw_prices = flatten_cols(raw_prices)

if f"Open_{TICKER}" in raw_prices.columns:
    price_cols = {
        f"Open_{TICKER}": "open",
        f"High_{TICKER}": "high",
        f"Low_{TICKER}": "low",
        f"Close_{TICKER}": "close",
        f"Volume_{TICKER}": "volume",
    }
else:
    price_cols = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }

prices = (
    raw_prices.reset_index()[["Date"] + list(price_cols.keys())]
    .rename(columns={"Date": "date", **price_cols})
    .sort_values("date")
    .reset_index(drop=True)
)
prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)

# 5) Create next-trading-day target before merging news
prices["target_open"] = prices["open"].shift(-1)
prices["target_date"] = prices["date"].shift(-1)
prices["target_return"] = prices["target_open"] / prices["close"] - 1

start_dt = pd.to_datetime(DATE_START)
end_dt = pd.to_datetime(DATE_END)
prices = prices[(prices["date"] >= start_dt) & (prices["date"] <= end_dt)].copy()

now_ny = datetime.now(ZoneInfo("America/New_York"))
today_ny = pd.Timestamp(now_ny.date())

if now_ny.time() < MARKET_CLOSE:
    prices = prices[prices["date"] < today_ny].copy()
else:
    prices = prices[prices["date"] <= today_ny].copy()

# 6) Keep every trading day, even if there was no news
data = pd.merge(prices, daily_sent, on="date", how="left").sort_values("date")

sentiment_cols = ["score_mean", "score_count", "pos_mean", "neg_mean"]
data[sentiment_cols] = data[sentiment_cols].fillna(0)
data = data.reset_index(drop=True)

print(f"Trading days with features: {len(data)} rows")

features = [
    "score_mean",
    "score_count",
    "pos_mean",
    "neg_mean",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# 7) Chronological train / validation / test split
labeled = data.dropna(subset=["target_return", "target_open", "target_date"]).copy()
labeled = labeled.reset_index(drop=True)

if len(labeled) < 10:
    raise ValueError("Not enough labeled trading days for train/validation/test.")

n = len(labeled)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

if train_end < 5 or val_end - train_end < 2 or n - val_end < 2:
    raise ValueError("The dataset is too small for this split.")

print(f"\nTrain rows: {train_end}")
print(f"Validation rows: {val_end - train_end}")
print(f"Test rows: {n - val_end}")

print(
    f"Train: {labeled['date'].iloc[0].date()} -> "
    f"{labeled['date'].iloc[train_end - 1].date()}"
)
print(
    f"Validation: {labeled['date'].iloc[train_end].date()} -> "
    f"{labeled['date'].iloc[val_end - 1].date()}"
)
print(
    f"Test: {labeled['date'].iloc[val_end].date()} -> "
    f"{labeled['date'].iloc[-1].date()}"
)

walk_forward(labeled, features, train_end, val_end, "Validation")
walk_forward(labeled, features, val_end, n, "Test")

# 8) Final next-session prediction
future_rows = data[data["target_open"].isna()].copy()

if len(future_rows) > 0:
    current = future_rows.iloc[[-1]]

    final_model = make_model()
    final_model.fit(labeled[features], labeled["target_return"])

    pred_return = final_model.predict(current[features])[0]
    pred_open = current["close"].iloc[0] * (1 + pred_return)
    feature_date = current["date"].iloc[0].date()

    print(f"\nUsing completed features on {feature_date}")
    print(f"Predicted OPEN for the next trading session: ${pred_open:.2f}")
else:
    print(
        "\nNo future prediction is available yet. "
        "Run again after the latest US trading session has closed."
    )

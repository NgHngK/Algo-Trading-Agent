import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sentiment import get_sentiment
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- CONFIG ---
TICKER     = "AAPL"
NEWS_FILE  = "news_AAPL_OR_Apple_Inc_2025-07-01_2025-07-24.json"
DATE_START = "2025-07-01"
DATE_END   = "2025-07-24"  

def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns into single-level strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(part) for part in col if part and str(part) != "nan")
            for col in df.columns.to_list()
        ]
    return df

# 1) Load & normalize all fetched articles
articles = pd.read_json(NEWS_FILE)["articles"]
df       = pd.json_normalize(articles)

# 2) Compute FinBERT sentiment per article
df["publishedAt"] = pd.to_datetime(df["publishedAt"]).dt.date
df["sentiment"]   = df["content"].fillna("").apply(get_sentiment)
sent_df          = pd.DataFrame(df["sentiment"].tolist())
df                = pd.concat([df, sent_df], axis=1)

# 3) Aggregate into daily sentiment features
daily_sent = (
    df
    .groupby("publishedAt")
    .agg(
        score_mean  = ("score", "mean"),
        score_count = ("score", "count"),
        pos_mean    = ("pos",   "mean"),
        neg_mean    = ("neg",   "mean"),
    )
    .reset_index()
    .rename(columns={"publishedAt": "date"})
)
daily_sent["date"] = pd.to_datetime(daily_sent["date"])
print(f"Daily sentiment days: {len(daily_sent)} rows")

# 4) Download price data through the day after DATE_END
end_dt     = datetime.strptime(DATE_END, "%Y-%m-%d") + timedelta(days=1)
raw_prices = yf.download(
    TICKER,
    start=DATE_START,
    end=end_dt.strftime("%Y-%m-%d"),
    progress=False
)
raw_prices = flatten_cols(raw_prices)

# 5) Build price-feature DataFrame
prices = (
    raw_prices
    .reset_index()[[
        "Date",
        f"Open_{TICKER}", f"High_{TICKER}",
        f"Low_{TICKER}",  f"Close_{TICKER}",
        f"Volume_{TICKER}"
    ]]
    .rename(columns={
        "Date": "date",
        f"Open_{TICKER}":   "open",
        f"High_{TICKER}":   "high",
        f"Low_{TICKER}":    "low",
        f"Close_{TICKER}":  "close",
        f"Volume_{TICKER}": "volume",
    })
)
prices["date"] = pd.to_datetime(prices["date"])

# 6) Merge sentiment & price features
data = pd.merge(daily_sent, prices, on="date") \
         .sort_values("date") \
         .reset_index(drop=True)
print(f"Merged data days: {len(data)} rows")

# 7) Create next-day target (open on d+1)
data["target_open"] = data["open"].shift(-1)
data = data.dropna(subset=["target_open"]).reset_index(drop=True)
print(f"After adding target_open: {len(data)} rows")

# 8) Walk-forward evaluation with per-day prints (now including current open)
features = [
    "score_mean","score_count","pos_mean","neg_mean",
    "open","high","low","close","volume"
]

errors = []
print("\n=== Walk-forward results ===")
for i in range(1, len(data)):
    train = data.iloc[:i]
    test  = data.iloc[i : i+1]

    train_end   = train["date"].iloc[-1].date()   # current day D
    test_date   = test["date"].iloc[0].date()     # next day D+1
    open_current = train["open"].iloc[-1]         # open on current day D

    X_tr, y_tr = train[features], train["target_open"]
    X_te, y_te = test[features],  test["target_open"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    p       = model.predict(X_te)[0]
    actual  = y_te.values[0]
    abs_err = abs(p - actual)

    errors.append((p - actual) ** 2)

    print(
        f"Trained through {train_end} (Open ${open_current:.2f}) → "
        f"Test on {test_date}: "
        f"Pred ${p:.2f}, Act ${actual:.2f}, AbsErr ${abs_err:.2f}"
    )

# 9) Overall RMSE
rmse = (sum(errors) / len(errors)) ** 0.5
print(f"\nOverall walk-forward RMSE: ${rmse:.4f}")

# 10) Final one-day-ahead prediction for July 24
X_all        = data[features]
y_all        = data["target_open"]
final_model  = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X_all, y_all)

X_current    = data.iloc[[-1]][features]  
X_next       = final_model.predict(X_current)[0]

end_dt       = datetime.strptime(DATE_END, "%Y-%m-%d").date()
feature_date = end_dt - timedelta(days=1)     # 2025-07-23
pred_date    = end_dt                         # 2025-07-24

print(f"\nUsing X_current (features on {feature_date}):")
print(X_current)
print(f"\nPredicted OPEN on {pred_date}: ${X_next:.2f}")

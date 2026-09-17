# FNSPID multi-company overnight-return model

This project trains one panel model across multiple companies.

One row means:

`ticker + trading day -> next trading day's opening return`

Target:

`target_return = next_open / current_close - 1`

The predicted next opening price is reconstructed from the predicted return.

## Files

```text
preprocessing.py
preprocessing.ipynb
train.ipynb
requirements.txt
README.md
```

Generated `data/cache/` and `outputs/` folders are not included in the final ZIP.

## Input data

Put FNSPID here:

```text
data/FNSPID/
├── Stock_news/
│   └── nasdaq_exteral_data.csv
└── Stock_price/
    └── full_history/
        └── full_history/
            ├── AAPL.csv
            ├── MSFT.csv
            └── ...
```

You may also set `FNSPID_DIR` to the FNSPID root folder.

## Run

```bash
pip install -r requirements.txt
```

Run `preprocessing.ipynb` from top to bottom, then run `train.ipynb` from top to bottom.

Both notebooks default to:

```python
MODE = "full"
```

Use `MODE = "smoke"` for a quick pipeline test.

## Low-resource behavior

The preprocessing pipeline:

- streams the large news CSV,
- uses DuckDB when available and pandas chunks as a fallback,
- extracts only the news fields needed by the model,
- caches filtered news, FinBERT results, daily sentiment and the panel,
- fingerprints cache inputs so same-row-count but different data does not reuse stale results,
- detects CPU, RAM, CUDA and free VRAM,
- benchmarks FinBERT batch size,
- halves the FinBERT batch size after CUDA OOM,
- limits model CPU parallelism.

The CPU cap defaults to at most 8 physical cores. Override it with `MAX_CPU_THREADS`.

Parquet is preferred for cache files. If a parquet engine is unavailable, preprocessing falls back to pickle cache files.

## Data-quality guardrails

The preprocessing code rejects implausible FNSPID news timestamps before it computes the chronological cutoff. The default lower bound is `1998-01-01`; override it with `FNSPID_NEWS_MIN_DATE` only when using another dataset. News is also prevented from being shifted more than 7 calendar days to a later trading session.

Universe selection checks every news-qualified ticker in full mode instead of discarding candidates before price-file validation. Tickers containing digits, dots, or dashes are supported, and common `BRK.B` / `BRK-B` filename differences are matched automatically.

If the notebook prints `DuckDB is not installed ...`, install `requirements.txt` in the active Jupyter kernel. Otherwise the 20+ GB news CSV falls back to slower pandas chunk scans.

## GPU acceleration

FinBERT uses CUDA automatically when the installed PyTorch build can access an NVIDIA GPU. XGBoost probes its own CUDA support independently and verifies the device actually selected, because PyTorch and XGBoost may have different CUDA builds. Random Forest and Extra Trees remain CPU models in scikit-learn.

## FinBERT reproducibility

The model is `ProsusAI/finbert` and the default model revision is pinned to:

```text
4556d13015211d73dccd3fdd39d39232506f3e43
```

You may override it with `FINBERT_REVISION`.

## Company design

The preprocessing stage first creates a practical candidate universe. The training notebook then applies eligibility rules again using the final TRAIN period only.

Eligible companies are split into:

- seen companies: may appear in TRAIN, VALIDATION and TEST-SEEN,
- unseen companies: excluded from TRAIN and VALIDATION and used only in TEST-UNSEEN.

Ticker identity is not a model feature.

## Timestamp rule

Date-only values and exact-midnight values are treated conservatively because their publication time is unknown. Examples include:

```text
2023-12-12
2023/12/12
2023-12-12 00:00:00 UTC
2023-12-12T00:00:00Z
```

They are assigned to the next trading session.

Precise timestamps are converted to `America/New_York`:

- before 16:00 -> the same trading session if it exists,
- exactly 16:00 or later -> the next trading session.

This prevents after-close information from entering a close-based forecast for the same day.

## Price and target leakage protection

All lag, rolling and target operations are grouped by ticker.

For each ticker:

```text
prev_close   = previous row close
next_open    = next row open
target_date  = next row trading date
```

They never cross company boundaries.

The chronological split is:

```text
TRAIN -> purge date -> VALIDATION -> purge date -> TEST
```

TRAIN rows must have `target_date < validation_start`.
VALIDATION rows must have `target_date < test_start`.

Inner CV is based on unique trading dates, not random rows, and also purges training labels that reach the validation block.

## Common lookback warm-up

Lookback candidates are:

```text
1, 3, 7, 15, 30 trading sessions
```

Price-change features need one prior price row. Therefore every lookback comparison starts only after `MAX_LOOKBACK + 1 = 31` price-history rows. This gives every lookback the same eligible row set.

Lookback selection uses TRAIN date-based CV only.

## Hyperparameter search

After the lookback is fixed, the notebook tunes:

- Random Forest,
- Extra Trees,
- XGBoost.

The XGBoost learning-rate schedule is selected using TRAIN CV only. XGBoost stopping rounds are also frozen from TRAIN CV before VALIDATION is used for final model selection.

VALIDATION compares Random Forest, Extra Trees, XGBoost and the zero-return Naive baseline.

The Naive baseline is allowed to win. If no ML model beats it on VALIDATION, the selected model is `Naive` and the TEST/future-prediction path remains valid.

TEST is not used for hyperparameter or model selection.

## Walk-forward TEST

TEST evaluation is online/prequential.

At each test date, only labels whose `target_date <= current_date` can enter history. Unseen-company rows are never added to fitting history.

The selected ML model is refit periodically. If `Naive` was selected, no model fitting is performed.

The notebook saves `walk_forward_audit.csv` so these history boundaries can be inspected directly.

## Outputs

Main outputs include:

```text
lookback_search.csv
lookback_fold_results.csv
lr_search_summary.csv
lr_search_epoch_history.csv
rf_trials.csv
extra_trees_trials.csv
xgb_trials.csv
xgb_optuna_epoch_history.csv
xgb_validation_epoch_history.csv
validation_results.csv
test_predictions_seen.csv
test_predictions_unseen.csv
walk_forward_audit.csv
feature_importance.csv
future_forecasts.csv
metrics.json
split_summary.json
preprocessing_summary.json
```

## Interpretation

This project evaluates next-open return forecasting. It is not by itself a complete trading-strategy backtest. Claims about trading profitability would additionally require transaction costs, slippage, turnover, position rules and risk metrics such as drawdown and Sharpe ratio.

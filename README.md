# FNSPID Multi-Company Overnight Return Forecasting

## About this project

This project studies whether market data and financial news can help predict how a stock will open on the next trading day. Instead of predicting the next opening price directly, the model first predicts the **overnight return**:

```text
target_return = next_open / current_close - 1
```

The predicted opening price is then calculated from the current closing price:

```text
predicted_next_open = current_close * (1 + predicted_return)
```

The project uses stock price data, trading volume, recent return features, and financial news from the FNSPID dataset. News titles are scored with FinBERT, and the final feature set is tested with Random Forest, Extra Trees, and XGBoost. The main goal is to see whether one shared model can learn useful patterns across many companies and still work on companies that were not used during training.

## Model inputs

Each row represents one company on one trading day. The market features include daily return, opening gap, intraday return, high-low range, volume change, recent average return, return volatility, and recent volume change. The news features include average sentiment, sentiment variation, news count, rolling sentiment, and rolling news count.

A simple Naive model is also used as a baseline. It always predicts a return of zero, which means it assumes that the next opening price will stay close to the current closing price. This baseline is useful because overnight stock returns are usually small, so a machine learning model should clearly beat this simple rule before we can say that it adds useful prediction power.

## Pipeline

The full pipeline is:

```text
FNSPID stock prices
        +
FNSPID financial news
        ↓
clean and align the data
        ↓
FinBERT sentiment scoring
        ↓
market + news features
        ↓
lookback search
        ↓
time-based training and validation
        ↓
Random Forest / Extra Trees / XGBoost
        ↓
final model selection
        ↓
walk-forward test
        ↓
seen-company and unseen-company evaluation
```

The data is split by time rather than randomly. This is important because financial data is ordered in time, and random splitting could allow future information to leak into training. Ticker identity is also not used as a model feature, so the model must learn from market and news patterns instead of memorizing company names.

## Data summary

The processed dataset contains 50 companies, 424,859 panel rows, and 348,410 cleaned news rows. The price history runs from 1962 to 2023. After the final training rules were applied, 40 companies were used in the model experiment: 32 seen companies and 8 unseen companies.

The unseen companies were:

```text
BIIB, BSX, CI, FCX, GILD, QCOM, TXN, WFC
```

The final model datasets were:

| Split | Rows |
|---|---:|
| Train | 152,207 |
| Validation | 74,720 |
| Test Seen | 73,842 |
| Test Unseen | 17,739 |

The chronological split was:

```text
Train      : 1962-02-13 to 2005-06-02
Validation : 2005-06-06 to 2014-09-12
Test       : 2014-09-16 to 2023-12-27
```

## Important note about news data

The price history starts much earlier than the usable news history. The stock data goes back to 1962, while the usable FinBERT news features begin around 2009. This matters because the current training period ends in 2005. As a result, the model in this run mainly learns from price, return, volatility, and volume features rather than from news sentiment.

This also explains why the final Extra Trees model gives zero importance to all news-related features. This should not be read as proof that financial news is useless. A better interpretation is that the current time split is not a fair test of the news signal. A future experiment should use a period where both price and news data are available during training.

# Results

## 1. Lookback window

The project tested lookback windows of 1, 3, 7, 15, and 30 trading days.

| Lookback | CV MAE |
|---:|---:|
| **15** | **0.005629** |
| 30 | 0.005638 |
| 7 | 0.005801 |
| 3 | 0.005992 |
| 1 | 0.006049 |

The best result came from a **15-day lookback**. Compared with a 1-day lookback, the 15-day window reduced CV MAE by about **6.95%**, which shows that the model benefits from using more than a few days of recent history. However, the difference between 15 and 30 days was very small, so the exact choice between these two windows was not very important in this run.

The fold results also became worse in later periods. For the 15-day lookback, the MAE increased from about 0.00398 in the first fold to about 0.00760 in the third fold. This suggests that later market periods were harder to predict than earlier ones, which is common in financial data because market behavior changes over time.

## 2. XGBoost learning-rate search

Several learning-rate strategies were tested, including constant, exponential, step decay, and cosine schedules. The best selected choice was a constant learning rate of **0.10**, but the results of the different strategies were almost identical.

This means that the learning-rate schedule did not have a large effect on model quality in this experiment. XGBoost also stopped improving after only a small number of boosting rounds, which suggests that the stable prediction signal in the data was limited.

## 3. Model tuning

The best training cross-validation results were:

| Model | Best CV MAE |
|---|---:|
| **XGBoost** | **0.005444** |
| Extra Trees | 0.005604 |
| Random Forest | 0.005658 |

XGBoost looked strongest during training cross-validation. However, the final model was not selected from training CV alone. All tuned models were tested again on the later validation period. This is important because a model that performs well on older data may not be the best model in a later market period.

That is exactly what happened here. XGBoost had the best training CV score, but Extra Trees had the best validation return MAE.

## 4. Validation results

| Model | Return MAE | Return RMSE | Return R² | Direction Accuracy |
|---|---:|---:|---:|---:|
| **Extra Trees** | **0.007440** | **0.014019** | **0.00065** | 50.91% |
| XGBoost | 0.007440 | 0.014027 | -0.00051 | 50.89% |
| Naive | 0.007443 | 0.014031 | -0.00108 | — |
| Random Forest | 0.007449 | 0.014046 | -0.00324 | 50.93% |

Extra Trees had the lowest validation MAE and was therefore selected as the final model. However, the improvement over the Naive baseline was very small. Naive had an MAE of 0.0074428, while Extra Trees had an MAE of 0.0074397, which is only about a **0.04% improvement**.

The validation R² was also almost zero, and direction accuracy stayed close to 50%. This means that the model found only a weak signal in overnight returns. Extra Trees technically performed best, but the difference was too small to describe the model as highly accurate.

## 5. Test on seen companies

The final Extra Trees model was then tested with walk-forward evaluation on companies that had already appeared in the training data.

| Metric | Extra Trees | Naive |
|---|---:|---:|
| Return MAE | **0.007732** | 0.007742 |
| Return RMSE | **0.016283** | 0.016440 |
| Return R² | **0.01872** | -0.00035 |
| Direction Accuracy | 51.78% | — |
| Open MAE | **0.468621** | 0.469356 |
| Open RMSE | **1.004967** | 1.006134 |

This is one of the better parts of the experiment. Extra Trees beat the Naive baseline on return MAE, return RMSE, return R², opening-price MAE, and opening-price RMSE. The improvement was still small, but the positive return R² of 0.0187 shows that the model captured some information for companies it had already seen before.

Direction accuracy reached 51.78%, which is only slightly above 50%. Overall, the seen-company test suggests that there is a small predictive signal, but not a strong one.

## 6. Test on unseen companies

The harder test used eight companies that were never included in training or validation.

| Metric | Extra Trees | Naive |
|---|---:|---:|
| Return MAE | **0.00721354** | 0.00721397 |
| Return RMSE | 0.01362454 | **0.01361987** |
| Return R² | -0.00124 | **-0.00056** |
| Direction Accuracy | 52.33% | — |
| Open MAE | 0.765842 | **0.764550** |
| Open RMSE | 2.259682 | **2.252562** |

The unseen-company result is mixed. Extra Trees had a slightly better return MAE, but Naive was slightly better on return RMSE, return R², opening-price MAE, and opening-price RMSE. Direction accuracy reached 52.33%, but this small improvement in direction did not lead to better overall price error.

The main conclusion is that the model does not yet generalize strongly to completely new companies. This is an important result because unseen-company testing is much harder than testing only on companies that were already present in training.

# Feature importance

The final Extra Trees feature importance was:

| Feature | Importance |
|---|---:|
| `volume_change_mean_lb` | **23.44%** |
| `high_low_range` | **18.16%** |
| `return_vol_lb` | **17.92%** |
| `intraday_return` | **12.79%** |
| `return_mean_lb` | **10.93%** |
| `daily_return` | **9.88%** |
| `open_gap` | **5.21%** |
| `volume_change` | **1.66%** |
| News features | **0%** |

The model relied mainly on volume, volatility, and recent price behavior. The strongest feature was the rolling average of volume change, followed by the high-low range and return volatility. All news features had zero importance in this run because the training period ends before the usable FinBERT news period begins.

The correct conclusion is therefore not that news does not matter. Instead, this run mainly tests market-based features, while the news contribution still needs a new experiment with a better time split.

# Why the price charts look very good

Some next-opening-price charts look almost perfect, even though the return results are weak. The reason is that the predicted opening price is rebuilt from the real current closing price:

```text
predicted_next_open = current_close * (1 + predicted_return)
```

For example, if the current close is 300, the real next open is 301, and the model predicts a return of zero, the predicted next open is still 300. The return prediction is weak, but the price error is only 1 dollar. On a chart where the stock price moves between 200 and 400, the two lines can look almost identical.

This is why price-level R² can be very high while return R² stays close to zero.

## Example: BIIB unseen test

| Model | Next-Open R² | Return R² |
|---|---:|---:|
| XGBoost | 0.98683 | -0.00040 |
| Extra Trees | 0.98679 | -0.00335 |
| Random Forest | 0.97140 | -1.41882 |

The BIIB example makes this difference very clear. XGBoost and Extra Trees both produce next-opening-price R² values close to 0.99, but their return R² values are around zero. This means the models follow the overall price level well because they start from the latest real closing price, but they still have difficulty predicting the actual overnight move.

For this reason, the most useful metrics in this project are return MAE, return RMSE, return R², direction accuracy, and comparison with the Naive baseline.

# Future forecasts

The final pipeline produced one forecast row for each selected ticker. Most predicted overnight returns were very close to zero. The average predicted return was about **+0.0348%**, while the largest was about **+0.1063%**.

These small values are consistent with the rest of the results. The model behaves conservatively because it has not found a strong signal for large overnight moves. The forecast file should also be read carefully: each row is based on the last available row for that ticker, so it is not a live forecast for the current market date.

# Main findings

The experiment shows that a 15-day lookback works better than very short windows, and Extra Trees gives the best validation return MAE. However, the improvement over the Naive baseline is very small, which means that the prediction signal is weak. The model performs slightly better on companies that were already present in training, but its advantage almost disappears on unseen companies.

The final model mainly learns from price, volume, and volatility features. The current experiment does not properly test the value of news because the training period ends before the usable FinBERT news period begins. Another important finding is that strong-looking price charts can be misleading, so return-level metrics are more useful for judging the real quality of the forecast.

# Contribution

The main value of this project is the full evaluation pipeline rather than a claim of high prediction accuracy. The project combines multi-company modeling, FinBERT news processing, time-based splitting, leakage protection, rolling features, lookback search, model tuning, Naive baseline comparison, seen-company testing, unseen-company testing, and walk-forward evaluation.

It also shows why financial machine learning results need to be interpreted carefully. A model can produce a price chart that looks very accurate while still having very little ability to predict the actual next-day return. Testing against a simple baseline and checking unseen companies gives a more realistic view of model quality.

# Next step

The next experiment should test the news signal with a time range where both market data and news data are available during training. The clearest setup would compare three versions of the model:

```text
1. Market features only
2. News features only
3. Market + news features
```

This would directly answer whether FinBERT news sentiment improves prediction beyond price and volume data.

# Example outputs

### Extra Trees — BIIB next-opening price

![Extra Trees BIIB next opening price](outputs/images/individual_models/next_open/test_unseen_biib_extra_trees_next_open_1000dpi.png)

### Extra Trees — BIIB target return

![Extra Trees BIIB target return](outputs/images/individual_models/target_return/test_unseen_biib_extra_trees_target_return_1000dpi.png)

More plots are stored in:

```text
outputs/images/individual_models/
```

# Project structure

```text
.
├── README.md
├── requirements.txt
├── preprocessing.py
├── preprocessing.ipynb
├── train.ipynb
├── download_FNSPID.ipynb
├── folder_structure.py
└── outputs/
    ├── checkpoints/
    ├── images/
    │   └── individual_models/
    │       ├── next_open/
    │       ├── target_return/
    │       ├── residuals/
    │       ├── absolute_errors/
    │       ├── scatter/
    │       ├── residual_histograms/
    │       └── model_overviews/
    ├── validation_results.csv
    ├── test_predictions_seen.csv
    ├── test_predictions_unseen.csv
    ├── feature_importance.csv
    ├── future_forecasts.csv
    ├── walk_forward_audit.csv
    ├── metrics.json
    ├── split_summary.json
    └── preprocessing_summary.json
```

# How to run

Install the packages:

```bash
pip install -r requirements.txt
```

Then run:

```text
1. preprocessing.ipynb
2. train.ipynb
```

Run both notebooks from top to bottom.

# Final note

This project tests forecasting, not a complete trading strategy. It does not include transaction costs, slippage, position sizing, trading rules, Sharpe ratio, or maximum drawdown. These parts are needed before making any claim about real trading profit.

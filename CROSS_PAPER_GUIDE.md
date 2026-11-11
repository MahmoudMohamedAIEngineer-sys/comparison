# Cross-paper comparison: edits for each notebook

Use this guide to re-run the seven paper notebooks so that every model is scored on **the same task, test set and metrics**. You add a small number of cells and change a few lines in each notebook. You don't rewrite any of them.

Files:

| File | What it is |
|---|---|
| `cb_benchmark.py` | The shared module. Upload it once as a Kaggle Dataset named `cb-benchmark`. |
| `cb_comparison.ipynb` | The final notebook. It loads every saved prediction and ranks models and feature sets. |
| `CROSS_PAPER_GUIDE.md` | This guide. |

Cell numbers below count from 0 (the first cell is cell 0), the same numbering as the evidence map. Each edit also quotes the cell's first line so you can find it in Kaggle.

---

## 1. The common protocol (CB-24)

| Item | Rule | Why |
|---|---|---|
| Task | Day-ahead. The origin is the hour whose **local** hour is 23. The forecast covers the next 24 hourly rows. | It matches what P6 and P7 already do and how a real day-ahead forecast is issued. |
| Split | Uses the local year of the first target hour: **train < 2024, validation = 2024, test = 2025**. | Every country is tested on the same calendar year. It's P7's split. |
| Test origins | Only local-23 origins in 2025 whose 24 targets are all observed. | This gives exactly one forecast per country per day. |
| Train/val origins | Any hour, or local-23 only. Each notebook keeps its own sampling. | More training data. Training choices stay with each paper. |
| Inputs | Anything observed at or before the origin, plus calendar values of the target hours. | This is what you would actually know at forecast time. |
| Weather | **No temperature after the origin** in the `honest` track. If you keep a model that uses observed future temperature, save it with `track="oracle"`. Oracle runs are reported separately. | It removes the oracle-weather problem. |
| Scaling | Fitted on training rows only. All notebooks already do this. | It prevents leakage. |
| Output | 24 values in MW per origin, saved with `cb_save`. | |
| Metrics | Computed **centrally** against load re-read from the CSV. Primary metric: mean of per-country MAPE (`Country_MAPE`). Also reported: MAPE, WAPE, RMSE, MAE, bias, Diebold–Mariano tests. | Notebook-side bugs, scaling mistakes and P2's capping can't affect the scores. |
| Seeds | Deep models: seeds 42, 43 and 44 if time allows. At minimum use seed 42. | You get the run-to-run spread. |

The comparison notebook keeps only the (country, origin) pairs that **every** honest run predicted. All models are therefore scored on exactly the same days. It also prints each run's coverage. A run below 50 % coverage is listed but left out of the leaderboard.

### Two comparisons in one benchmark

1. **Which model is best.** Each notebook's own models, re-run on CB-24.
2. **Which feature engineering is best.** Each notebook also trains two **fixed reference models**, `REF-Ridge` and `REF-MLP` (2 × 256, fixed hyper-parameters), on **its own input representation**. The model stays the same and only the features change, so differences between feature sets come from the features. This is the controlled feature-engineering comparison you couldn't make before.

---

## 2. Setup cell for every notebook

Add this as a **new cell right after the data-loading cell** (the cell that creates `df`). Change the two strings for each notebook.

```python
import sys
sys.path.append("/kaggle/input/cb-benchmark")
from cb_benchmark import *

CB["NB"] = "P1"                        # P1 … P7
CB["FE"] = "FE-1 engineered (27)"      # default feature-set label for this notebook
CB["SEED"] = 42
CB_TABLE = cb_origin_table(df)
```

> In P2, `df` is modified by the IQR capping cell later. Build `CB_TABLE` **before** that cell, which is where the setup cell sits anyway.

Feature-set labels to use, so the comparison groups them correctly:

| Notebook | `CB["FE"]` label(s) |
|---|---|
| P1 | `FE-1 engineered (27) · 168h` |
| P2 | `FE-1 engineered (27) · 7h`, `FE-2 Pearson-5 · 7h`, `FE-2 SHAP-5 · 7h` |
| P3 | `FE-3a raw load+temp · 60h`, `FE-3b VMD K4 · 60h` |
| P4 | `FE-4a tabular 10`, `FE-4b univariate 24h` |
| P5 | `FE-4c PCC weather (4ch) · 24h`, `FE-4c all weather (5ch) · 24h` |
| P6 | `FE-5 similar days` |
| P7 | `FE-6 raw 672h` |

At the end of every notebook, add `cb_manifest()` to list what was saved.

---

## 3. P1 (paper1.ipynb): already day-ahead, three small edits

**3.1 Split.** Replace the whole of **cell 52** (starts `split_parts = []`) with:

```python
valid_origins = cb_apply_split(valid_origins, CB_TABLE)
```

**3.2 Test origins.** In **cell 64** (starts `SEED = 42`), replace

```python
eval_samples = samples[
    (samples["split"] == "test") &
    (samples["Hour (UTC)"].dt.hour == ORIGIN_HOUR)
].reset_index(drop=True)
```

with

```python
eval_samples = samples[samples["split"] == "test"].reset_index(drop=True)
```

In the same cell, add one line at the **end of `evaluate()`**, after `np.save(...)`:

```python
    cb_save(name, eval_countries, eval_samples["Hour (UTC)"], predictions_mw, y_true_mw=eval_actual_mw)
```

All six models (ARIMA, ARIMAX, SARIMA, XGBoost, SES and LSTM) go through `evaluate()`, so this single line saves all of them.

**3.3 Honest ARIMAX.** In **cell 65** (starts `def fill_nan_exog`), replace the `else:` branch of `get_windows` with

```python
        else:
            exog = fill_nan_exog(data["X"][end - INPUT_LENGTH:end, exog_idx])
            yield history, exog, exog[-FORECAST_HORIZON:]
```

The future temperature becomes "same as the last 24 hours" (persistence) instead of the observed future.

**3.4 Seeds (optional).** Before the LSTM training cell, set `CB["SEED"] = s` and re-run that cell for each `s` in `CB["SEEDS"]`.

**3.5 Reference models.** Add a new cell at the end:

```python
CB["FE"] = "FE-1 engineered (27) · 168h"
rng = np.random.default_rng(42)
def ds_arrays(ds, n):
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False) if n else range(len(ds))
    X, y = zip(*[ds[i] for i in idx])
    return np.stack([x.numpy() for x in X]), np.stack([t.numpy() for t in y])

Xtr, ytr = ds_arrays(train_dataset, CB["N_REF"])
Xva, yva = ds_arrays(validation_dataset, CB["N_VAL"])
Xte, _ = ds_arrays(eval_dataset, None)
cb_reference_suite(Xtr, ytr, Xva, yva, Xte, eval_countries, eval_samples["Hour (UTC)"],
                   lambda p, c: cb_to_mw(p, c, country_target_scalers))
```

> Memory: 50,000 × 168 × 27 floats is about 0.9 GB, which is fine on a Kaggle GPU session. If memory is tight, lower `CB["N_REF"]` to 30,000.

---

## 4. P2 (paper2.ipynb): change the horizon from 1 h to 24 h

**4.1 Horizon.** In **cell 36** (starts `HISTORY = 7`), set `HORIZON = 24`.

**4.2 IQR capping from training years only.** In **cell 40** (starts `capped_columns = ...`), replace the two quantile lines with

```python
    train_part = df["year"] < CB["VAL_YEAR"]
    grouped = df[column].where(train_part).groupby(df["Country code"])
    q1 = grouped.transform(lambda x: x.quantile(0.25))
    q3 = grouped.transform(lambda x: x.quantile(0.75))
```

Delete the `grouped = df.groupby("Country code")[column]` line above them. The scores are computed later against the **uncapped** CSV load. Capping therefore stays a training-time choice and can't change the test target.

**4.3 Split.** Replace **cell 53** (starts `split_parts = []`) with

```python
valid_origins = cb_apply_split(valid_origins, CB_TABLE)
```

**4.4 Model output.** In **cell 66** (starts `class Forecaster`), change both `nn.Linear(..., 1)` output layers to `nn.Linear(..., FORECAST_HORIZON)`. In `predict`, change

```python
    return torch.cat(outputs).cpu().numpy().ravel()
```

to

```python
    return torch.cat(outputs).cpu().numpy()
```

Make `compute_metrics` work on (n, 24) by inserting these as its first lines:

```python
    k = np.asarray(scaled_actual).reshape(len(countries), -1).shape[1]
    countries = np.repeat(countries, k)
    scaled_actual, scaled_predicted = np.ravel(scaled_actual), np.ravel(scaled_predicted)
```

In **cell 65**, change the two target arrays to stay 2-D:

```python
y_train_scaled = y_train.cpu().numpy()
y_test_scaled = y_test.cpu().numpy()
```

**4.5 Save.** In `run_experiment` (cell 66), right after `predicted = predict(model, X_te)`, add

```python
    cb_save(name, test_country_codes, test_dataset.samples["Hour (UTC)"],
            cb_to_mw(predicted, test_country_codes, country_target_scalers),
            feature_set=f"{'FE-1 engineered (27)' if feature_set in ('All', 'Pearson') else 'FE-2 ' + feature_set} · 7h")
```

**4.6 CV (optional, saves time).** The shuffled 5-fold CV does not affect the test predictions. You can set `N_FOLDS = 2` or remove the CV loop. If you keep CV, replace `KFold(..., shuffle=True)` with `TimeSeriesSplit(n_splits=N_FOLDS)` on chronologically sorted training windows.

**4.7 SHAP with 24 outputs.** In **cell 72**, wrap the model so SHAP explains the mean of the 24 outputs:

```python
class MeanHorizon(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x).mean(dim=1, keepdim=True)
```

Then change `model = copy.deepcopy(...).cpu().eval()` to `model = MeanHorizon(copy.deepcopy(...).cpu()).eval()`.

**4.8 Reference models.** Add a new cell at the end. It runs the reference suite on all three P2 feature sets:

```python
val_idx = rng.choice(len(validation_dataset), min(CB["N_VAL"], len(validation_dataset)), replace=False)
X_val, y_val = load_windows(validation_dataset, val_idx)
to_mw_cb = lambda p, c: cb_to_mw(p, c, country_target_scalers)
for label, feats in [("FE-1 engineered (27) · 7h", feature_columns),
                     ("FE-2 Pearson-5 · 7h", pearson5_features),
                     ("FE-2 SHAP-5 · 7h", shap5_features)]:
    idx = [feature_columns.index(f) for f in feats]
    cb_reference_suite(X_train[:, :, idx].cpu().numpy(), y_train.cpu().numpy(),
                       X_val[:, :, idx].cpu().numpy(), y_val.cpu().numpy(),
                       X_test[:, :, idx].cpu().numpy(), test_country_codes,
                       test_dataset.samples["Hour (UTC)"], to_mw_cb, feature_set=label)
```

---

## 5. P3 (paper3.ipynb): change the horizon from 1 h to 24 h, including the VMD components

**5.1** In **cell 22** (starts `HISTORY = 60`), set `HORIZON = 24`.

**5.2 Valid origins need all 24 targets.** In **cell 23**, replace the two `target_available` lines with

```python
    finite = np.isfinite(load_values[rows]).astype(int)
    cf = np.concatenate([[0], np.cumsum(finite)])
    target_available = np.zeros(len(rows), dtype=bool)
    target_available[:-HORIZON] = (cf[1 + HORIZON:] - cf[1:-HORIZON]) == HORIZON
```

**5.3 Split.** Replace **cell 24** (starts `valid_origins = valid_origins.sort_values`) with

```python
valid_origins = cb_apply_split(valid_origins, CB_TABLE)
```

**5.4 Windows.** In **cell 29**, function `build_windows`, change

```python
    load_target = scaled_y[rows + HORIZON]
```

to

```python
    load_target = scaled_y[rows[:, None] + np.arange(1, HORIZON + 1)[None, :]]
```

and change the two component-target lines to

```python
    extended = np.concatenate([window_y, load_target], axis=1)
    component_target = scale_components(decompose(extended, K, ALPHA)[:, -HORIZON:, :], countries).reshape(len(rows), -1)
```

In the `targets = {...}` dict, remove the `[:, None]` from `y_train_load` and `y_validation_load`.

**5.5 Reconstruction.** Replace `reconstruct` (in the cell with `compute_metrics`) with

```python
def reconstruct(component_predictions, countries):
    p = component_predictions.reshape(len(countries), HORIZON, K)
    out = np.empty((len(countries), HORIZON))
    for c in np.unique(countries):
        m = countries == c
        out[m] = imf_scalers[c].inverse_transform(p[m].reshape(-1, K)).reshape(-1, HORIZON, K).sum(axis=2)
    return out
```

**5.6 Outputs.** In **cell 31**, set `self.output = nn.Linear(HIDDEN_SIZE, HORIZON)` in `Recurrent`, and `nn.Linear(D_MODEL, HORIZON)` in `TransformerNet`. `BP` and `VCAG` already take `output_size` from the target shape.

**5.7 Save.** In `run_model`, change `scaled_load = predicted.ravel()` to `scaled_load = predicted`, then replace the metrics line with

```python
    row = {"Model": name, **compute_metrics(y_test_load.ravel(), scaled_load.ravel(), np.repeat(test_country_codes, HORIZON))}
    cb_save(name, test_country_codes, test_sample["Hour (UTC)"],
            cb_to_mw(scaled_load, test_country_codes, country_target_scalers),
            feature_set="FE-3b VMD K4 · 60h" if use_vmd else "FE-3a raw load+temp · 60h")
```

The PSO cell can stay as it is. The test set is now about 15,000 local-23 origins instead of one million windows, so P3 becomes **much faster**.

**5.8 Reference models.**

```python
to_mw_cb = lambda p, c: cb_to_mw(p, c, country_target_scalers)
for kind, label in [("raw", "FE-3a raw load+temp · 60h"), ("vmd", "FE-3b VMD K4 · 60h")]:
    Xtr, Xva, Xte = (a.cpu().numpy() for a in inputs[kind])
    cb_reference_suite(Xtr, y_train_load, Xva, y_validation_load, Xte,
                       test_country_codes, test_sample["Hour (UTC)"], to_mw_cb, feature_set=label)
```

In the VMD case the reference model predicts load directly from the components. That is the right comparison for "is VMD a better input representation".

---

## 6. P4 (paper4.ipynb): change the horizon from 1 h to 24 h. Rows become origins.

In P4, `row` currently points at the **target** hour. After these edits it points at the **origin** hour.

**6.1** In **cell 22** (starts `HISTORY = 24`), set `HORIZON = 24` and `N_XGB = CB["N_TRAIN"]`.

**6.2 Samples and split.** Replace **cell 24** (starts `country_rows = ...`) with

```python
country_rows = df.groupby("Country code", sort=False).indices
load_values = df[TARGET].to_numpy()
window = REQUIRED_HISTORY + HORIZON
sample_parts = []
for country, rows in country_rows.items():
    cumulative = np.concatenate([[0], np.cumsum(np.isfinite(load_values[rows]))])
    complete_end = np.zeros(len(rows), dtype=bool)
    complete_end[window - 1:] = (cumulative[window:] - cumulative[:-window]) == window
    origin_pos = np.flatnonzero(complete_end) - HORIZON
    sample_parts.append(pd.DataFrame({"Country code": country, "row": rows[origin_pos]}))
samples = pd.concat(sample_parts, ignore_index=True)
samples["Hour (UTC)"] = df["Hour (UTC)"].to_numpy()[samples["row"].to_numpy()]
samples = cb_apply_split(samples, CB_TABLE)
```

**6.3 Features relative to the origin.** In **cell 27**, replace `history_window`, `tabular_data`, `sequence_data` and `to_mw` with

```python
FUT = np.arange(1, HORIZON + 1)[None, :]

def history_window(rows):
    return scaled_y[rows[:, None] + np.arange(-HISTORY + 1, 1)[None, :]]

def tabular_data(rows):
    t = rows + 1                                  # first target hour
    previous = history_window(rows)
    X = np.column_stack([hour[t], day_of_week[t], month[t], is_weekend[t],
                         scaled_y[t - 1], scaled_y[t - 2], scaled_y[t - 24], scaled_y[t - 168],
                         previous.mean(axis=1), previous.std(axis=1, ddof=1)])
    y = scaled_y[rows[:, None] + FUT]
    assert np.isfinite(X).all() and np.isfinite(y).all()
    return X.astype(np.float32), y.astype(np.float32)

def sequence_data(rows):
    X, y = history_window(rows), scaled_y[rows[:, None] + FUT]
    assert np.isfinite(X).all() and np.isfinite(y).all()
    return X[:, :, None].astype(np.float32), y.astype(np.float32)

def to_mw(scaled, countries):
    return cb_to_mw(scaled, countries, country_target_scalers)
```

At the end of `evaluate()`, add

```python
    cb_save(name, test_country_codes, test_rows["Hour (UTC)"], predicted,
            feature_set="FE-4a tabular 10" if name == "XGBoost" else "FE-4b univariate 24h")
```

**6.4 Outputs.** In **cell 28**, change `nn.Linear(25, 1)` to `nn.Linear(25, HORIZON)` in `Recurrent`, and `nn.Linear(50, 1)` to `nn.Linear(50, HORIZON)` in `CNN1D`. XGBoost accepts the 2-D `y` as it is, as a multi-output model.

**6.5 Seasonal naive.** **Cell 33** can be deleted, because the comparison notebook computes both naive baselines itself.

**6.6 SARIMA.** Replace the loop in **cell 34** with

```python
sarima_predictions = np.empty((len(test_rows), HORIZON))
for country in np.unique(test_country_codes):
    mask = test_country_codes == country
    first = test_rows.loc[mask, "row"].min()
    series = scaled_y[max(country_rows[country][0], first + 1 - SARIMA_FIT_HOURS):first + 1]
    result = ARIMA(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24),
                   enforce_stationarity=False, enforce_invertibility=False).fit()
    sarima_predictions[mask] = np.stack([
        result.apply(scaled_y[row + 1 - SARIMA_HISTORY:row + 1], refit=False).forecast(HORIZON)
        for row in test_rows.loc[mask, "row"]])
evaluate("SARIMA", sarima_predictions)
```

**6.7 Prophet.** In **cell 35**, predict the 24 target timestamps of each origin:

```python
    target_index = (test_rows.loc[mask, "row"].to_numpy()[:, None] + FUT).ravel()
    future = pd.DataFrame({"ds": timestamps[target_index], "is_holiday": holiday_flag[target_index]})
    prophet_predictions[mask] = model.predict(future)["yhat"].to_numpy().reshape(-1, HORIZON)
```

Also initialise it as `prophet_predictions = np.empty((len(test_rows), HORIZON))`.

**6.8 Significance tests (cell 36).** Flatten both prediction arrays with `.ravel()` before the paired tests, or remove the tests. The comparison notebook runs a Diebold–Mariano test on daily losses, which is more appropriate.

**6.9 Reference models.**

```python
to_mw_cb = lambda p, c: cb_to_mw(p, c, country_target_scalers)
tr = cb_sample(samples, "train", CB["N_REF"]); va = cb_sample(samples, "validation", CB["N_VAL"])
for label, fn in [("FE-4a tabular 10", tabular_data), ("FE-4b univariate 24h", sequence_data)]:
    Xtr, ytr = fn(tr["row"].to_numpy()); Xva, yva = fn(va["row"].to_numpy())
    Xte, _ = fn(test_rows["row"].to_numpy())
    cb_reference_suite(Xtr, ytr, Xva, yva, Xte, test_country_codes, test_rows["Hour (UTC)"], to_mw_cb, feature_set=label)
```

---

## 7. P5 (paper5-update.ipynb): change the horizon from 1 h to 24 h. The weather window ends at the origin.

**7.1** In **cell 22** (starts `HISTORY = 24`), add `HORIZON = 24`, and set `N_TRAIN = CB["N_TRAIN"]` and `N_TEST = 10**9` (use all test origins).

**7.2 Samples.** In **cell 23**, replace the `for` loop and the `rank`/`split` lines with

```python
for country, rows in country_rows.items():
    load_ok = window_complete(np.isfinite(load_values[rows]), HISTORY + HORIZON)
    meteo_ok = window_complete(meteo_observed[rows], HISTORY)
    n = len(rows)
    valid = np.zeros(n, dtype=bool)
    valid[:n - HORIZON] = load_ok[HORIZON:] & meteo_ok[:n - HORIZON]
    sample_parts.append(pd.DataFrame({"Country code": country, "row": rows[valid]}))

samples = pd.concat(sample_parts, ignore_index=True)
samples["Hour (UTC)"] = df["Hour (UTC)"].to_numpy()[samples["row"].to_numpy()]
samples = cb_apply_split(samples, CB_TABLE)
```

`row` is now the origin. The existing train-only scaler and PCC cells keep working (they use `split == "train"`).

**7.3 Windows (honest weather).** In **cell 27**, replace `window_data` and `to_mw` with

```python
def window_data(rows):
    past = rows[:, None] + np.arange(-HISTORY + 1, 1)[None, :]
    X = np.concatenate([scaled_y[past][:, :, None], scaled_meteo[past]], axis=2)
    y = scaled_y[rows[:, None] + np.arange(1, HORIZON + 1)[None, :]]
    assert np.isfinite(X).all() and np.isfinite(y).all()
    return X.astype(np.float32), y.astype(np.float32)

def to_mw(scaled, countries):
    return cb_to_mw(scaled, countries, country_target_scalers)
```

In the same cell, change `y_train_t = to_tensor(y_train[:, None])` to `to_tensor(y_train)`. At the end of `evaluate()`, add

```python
    cb_save(name, test_country_codes, test_rows["Hour (UTC)"], to_mw(scaled_predictions, test_country_codes),
            train_seconds=seconds, feature_set=CB["FE"])
```

**7.4 Outputs.** In **cell 29**, change every `nn.Linear(..., 1)` in `TCN` and `Recurrent` to `nn.Linear(..., HORIZON)`. In `run_deep_model`, change `.cpu().numpy().ravel()` to `.cpu().numpy()`.

**7.5 Feature-set label.** Set `CB["FE"] = "FE-4c PCC weather (4ch) · 24h"` in the setup cell. In **cell 42** (starts `all_channels = ...`), make this the first line inside the `for label, channels in ...` loop:

```python
    CB["FE"] = "FE-4c all weather (5ch) · 24h" if label == "all features" else "FE-4c PCC weather (4ch) · 24h"
```

Add `CB["FE"] = "FE-4c PCC weather (4ch) · 24h"` after the loop. The XGBoost in this cell takes the 2-D `y` without changes.

**7.6 Stacking variants.** The stacking learners use the last time step only, and SVR and GBDT are single-output. To include them, wrap the single-output learners in `sklearn.multioutput.MultiOutputRegressor` and give the stacking channel 24 columns. That is roughly 24× more training. **Recommendation: run the deep models (GRU, LSTM, BiGRU, BiLSTM, TCN, TCN-GRU) on CB-24 and skip cells 36–41 (Stacking, BiStacking and the four stacking hybrids).** Those cells would fail with a 24-column target anyway. They were 5× slower and worse in the original P5, so leaving them out doesn't change the conclusion. Say this explicitly when you present.

**7.7 Reference models.**

```python
va = cb_sample(samples, "validation", CB["N_VAL"])
X_val, y_val = window_data(va["row"].to_numpy())
to_mw_cb = lambda p, c: to_mw(p, c)
for label, ch in [("FE-4c PCC weather (4ch) · 24h", selected_channels), ("FE-4c all weather (5ch) · 24h", list(range(X_train.shape[2])))]:
    cb_reference_suite(X_train[:, :, ch], y_train, X_val[:, :, ch], y_val, X_test[:, :, ch],
                       test_country_codes, test_rows["Hour (UTC)"], to_mw_cb, feature_set=label)
```

---

## 8. P6 (paper6.ipynb): already day-ahead. Change the split and use honest weather.

In P6, `row` is local hour 0 of the target day, so the **origin is `row − 1`**.

**8.1 Split.** In **cell 25**, replace the `day_rank` and `samples["split"]` lines with

```python
samples = cb_apply_split(samples, CB_TABLE, origin_times=samples["Hour (UTC)"] - pd.Timedelta(hours=1))
```

**8.2 Honest temperature: use the previous day.** In **cell 25**, after the line `valid = ...` inside the loop, add

```python
    prev = index - 24
    valid &= prev[:, 0] >= 0
    valid &= np.isfinite(temperature[rows][np.clip(prev, 0, None)]).all(axis=1)
```

In **cell 27**, change `temperature_stats` to read the day **before** the target day:

```python
def temperature_stats(rows):
    window = temperature[day_index(rows - 24)]
    return np.column_stack([window.mean(axis=1), window.max(axis=1), window.min(axis=1)])
```

(If you also want the paper's original setup, run it once without this change and save with `CB["TRACK"] = "oracle"`.)

**8.3 Save.** At the end of `evaluate()` (the cell with `def evaluate(name, scaled_predictions)`), add

```python
    cb_save(name, test_country_codes, test_samples["Hour (UTC)"] - pd.Timedelta(hours=1), predicted,
            y_true_mw=actual_mw)
```

The similar-day features (`xday_1`–`xday_4`) only use earlier days, so they are already honest.

**8.4 Reference models.**

```python
X_val_ref, y_val_ref = window_data(ssa_validation_rows)
cb_reference_suite(X_train.cpu().numpy() if hasattr(X_train, "cpu") else X_train,
                   y_train.cpu().numpy() if hasattr(y_train, "cpu") else y_train,
                   X_val_ref, y_val_ref, X_test_array, test_country_codes,
                   test_samples["Hour (UTC)"] - pd.Timedelta(hours=1),
                   lambda p, c: cb_to_mw(p, c, country_target_scalers), feature_set="FE-5 similar days")
```

---

## 9. P7 (paper7.ipynb): already CB-24. Only add saving.

P7's origin (local hour 23) and split (target year 2024/2025) are already the CB protocol.

**9.1 Keep the test origin times.** In **cell 26**, inside the `for (country, split), group in ...` loop, add

```python
    if split == "test":
        cb_test_origins[country] = group["Hour (UTC)"].to_numpy()
```

Initialise `cb_test_origins = {}` above the loop.

**9.2 Collect and save predictions.** Add a new cell after cell 28:

```python
cb_buffer = {}
def cb_collect(setup, country, models):
    X_test, _, _ = country_data(country, "test")
    p = ensemble_predict(models, X_test)
    cb_buffer.setdefault(setup, []).append((country, cb_test_origins[country],
                                            cb_to_mw(p, [country] * len(p), country_target_scalers)))
def cb_flush(setup):
    parts = cb_buffer.pop(setup)
    cb_save(f"MLP-{setup}", np.concatenate([[c] * len(o) for c, o, _ in parts]),
            np.concatenate([o for _, o, _ in parts]), np.concatenate([p for *_, p in parts]))
```

- **Baseline loop (cell 30):** after `baseline_mape[country] = test_mape(country, models)`, add `cb_collect("Baseline", country, models)`. After the loop, add `cb_flush("Baseline")`.
- **`run_transfer`:** before `return`, add `cb_collect(CB_SETUP, country, models)`.
- **AbO loop (cell 31):** put `CB_SETUP = "AbO"` before the loop and `cb_flush("AbO")` after it. Do the same with `"CbO"` for the CbO loop.

**9.3 Reference models (global, without transfer).**

```python
Xtr, ytr = stack_data(countries, "train"); Xva, yva = stack_data(countries, "validation")
Xte = torch.cat([data[(c, "test")][0] for c in countries])
te_c = np.concatenate([[c] * len(data[(c, "test")][0]) for c in countries])
te_o = np.concatenate([cb_test_origins[c] for c in countries])
cb_reference_suite(Xtr.cpu().numpy(), ytr.cpu().numpy(), Xva.cpu().numpy(), yva.cpu().numpy(),
                   Xte.cpu().numpy(), te_c, te_o, lambda p, c: cb_to_mw(p, c, country_target_scalers))
```

---

## 10. Run order on Kaggle

1. Upload `cb_benchmark.py` as the dataset **cb-benchmark**.
2. In each paper notebook, add the setup cell and the edits, then **Save Version → Save & Run All**. Predictions go to `/kaggle/working/cb_predictions/`.
3. Upload `cb_comparison.ipynb`. Add as inputs: `cb-benchmark`, the seven notebook **outputs**, and the preprocessed data. Run it.
4. Read the results:
   - **Section 2 leaderboard**: which model is best (mean ± sd over seeds) and its skill relative to the 168-h seasonal naive.
   - **Section 3**: whether the winner is **significantly** better than each other model (Diebold–Mariano).
   - **Section 5**: which feature engineering wins with the model held fixed (REF-MLP and REF-Ridge), plus the best native model per feature set.
   - **Sections 6–7**: error by horizon and by country, and the oracle track kept separate.

---

## 11. What stays different, and how to describe it

- **Training budget per paper** (sample size, epochs) still follows each paper. The ranking therefore compares *paper pipelines as specified*. That's fair if you say so. For a strict architecture comparison, set every notebook's training sample to `CB["N_TRAIN"]`, as done above for P4 and P5.
- **Coverage.** P6 drops daylight-saving days and countries with incomplete days, and P7 needs at least 60 test days per country. The comparison notebook scores only origins that every run covers and prints each run's coverage.
- **Country set.** The test year is 2025, so countries whose data ends earlier (e.g. GBR in 2021, DZA in 2020) can't be in the benchmark. This is printed as "Countries in common set".
- **Sanity checks built in.** The comparison notebook asserts that every saved origin is a local-23 2025 origin. It also reports any notebook whose own `y_true` differs from the CSV by more than 1 MW, which exposes capping or scaling mistakes.

## 12. Tests run on the module

`cb_benchmark.py` and `cb_comparison.ipynb` were run end to end on **synthetic data** with the same schema (4 countries, 2022–2025, different UTC offsets and missing values). The tests checked:

- that local-23 origins map to local 0 of the next day;
- that a 2024-12-31 23:00 local origin is labelled test;
- target completeness against an independent lookup;
- saving and loading;
- the reference Ridge and MLP;
- naive baselines, the common set, metrics and seed aggregation;
- Diebold–Mariano tests and the Excel export.

The leaderboard recovered the known noise ranking of the fake models.

The per-notebook edits were written against the saved notebook code and **have not been executed** against your real data. A line may need small adjustments where a variable name differs from what is shown (for example in the cells I couldn't see in full, such as P5's stacking learners or P6's day-type masks). Each `cb_save` call asserts shapes and finite values, so a mismatch fails loudly instead of silently.

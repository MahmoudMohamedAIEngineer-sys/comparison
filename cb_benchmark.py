import glob
import math
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

CB = {
    "NB": None,
    "FE": None,
    "SEED": 42,
    "SEEDS": [42, 43, 44],
    "TRACK": "honest",
    "HORIZON": 24,
    "ORIGIN_HOUR": 23,
    "VAL_YEAR": 2024,
    "TEST_YEAR": 2025,
    "N_TRAIN": 50000,
    "N_REF": 50000,
    "N_VAL": 10000,
    "TIME": "Hour (UTC)",
    "COUNTRY": "Country code",
    "TARGET": "Load_MW_hourly",
    "LOCAL_HOUR": "hour",
    "PRED_DIR": "cb_predictions",
}


def _asarray(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def cb_origin_table(df):
    frame = pd.DataFrame({
        "country": df[CB["COUNTRY"]].to_numpy(),
        "origin": pd.to_datetime(df[CB["TIME"]]).to_numpy(),
        "local_hour": df[CB["LOCAL_HOUR"]].to_numpy(),
        "load": pd.to_numeric(df[CB["TARGET"]], errors="coerce").to_numpy(dtype=np.float64),
    }).sort_values(["country", "origin"], kind="stable")

    h = CB["HORIZON"]
    parts = []

    for country, group in frame.groupby("country", sort=False):
        group = group.reset_index(drop=True)
        times = group["origin"].to_numpy()
        n = len(group)
        observed = np.isfinite(group["load"].to_numpy()).astype(int)
        cumulative = np.concatenate([[0], np.cumsum(observed)])

        targets_ok = np.zeros(n, dtype=bool)
        targets_ok[:n - h] = (cumulative[1 + h:] - cumulative[1:-h]) == h

        consecutive = np.zeros(n, dtype=bool)
        consecutive[:n - h] = (times[h:] - times[:-h]) == np.timedelta64(h, "h")

        utc_hour = pd.to_datetime(times).hour.to_numpy()
        local_hour = group["local_hour"].to_numpy()
        offset = (local_hour - utc_hour) % 24
        offset = np.where(offset > 14, offset - 24, offset)
        local_time = times + offset.astype("timedelta64[h]")
        target_year = (local_time + np.timedelta64(1, "h")).astype("datetime64[Y]").astype(int) + 1970

        split = np.select(
            [target_year == CB["TEST_YEAR"], target_year == CB["VAL_YEAR"], target_year < CB["VAL_YEAR"]],
            ["test", "validation", "train"],
            default="unused",
        )

        parts.append(pd.DataFrame({
            "country": country,
            "origin": times,
            "local_hour": local_hour,
            "target_year": target_year,
            "cb_origin": (local_hour == CB["ORIGIN_HOUR"]) & targets_ok & consecutive,
            "cb_split": split,
        }))

    return pd.concat(parts, ignore_index=True)


def cb_label(tab, countries, origins):
    lookup = tab.drop_duplicates(["country", "origin"]).set_index(
        ["country", "origin"]
    )[["cb_origin", "cb_split"]]
    index = pd.MultiIndex.from_arrays([
        np.asarray(countries),
        pd.to_datetime(np.asarray(origins)).to_numpy(),
    ], names=["country", "origin"])
    labelled = lookup.reindex(index)

    return pd.DataFrame({
        "cb_origin": labelled["cb_origin"].fillna(False).astype(bool).to_numpy(),
        "cb_split": labelled["cb_split"].fillna("unused").to_numpy(),
    })


def cb_apply_split(samples, table, origin_times=None):
    samples = samples.copy()
    times = samples[CB["TIME"]] if origin_times is None else origin_times
    labelled = cb_label(table, samples[CB["COUNTRY"]].to_numpy(), np.asarray(times))
    split = labelled["cb_split"].to_numpy()
    keep = (split != "unused") & ((split != "test") | labelled["cb_origin"].to_numpy())

    samples = samples.loc[keep].copy()
    samples["split"] = split[keep]

    per_country = samples.groupby(CB["COUNTRY"])["split"].agg(set)
    complete = per_country[per_country.apply(lambda s: {"train", "validation", "test"} <= s)].index
    dropped = sorted(set(per_country.index) - set(complete))
    if dropped:
        print("Countries without train, validation and CB-24 test origins:", dropped)

    return samples[samples[CB["COUNTRY"]].isin(complete)].reset_index(drop=True)


def cb_to_mw(scaled, countries, country_target_scalers):
    values = _asarray(scaled).astype(np.float64)
    shape = values.shape
    flat = values.reshape(-1, 1)
    labels = np.repeat(np.asarray(countries), shape[1]) if values.ndim == 2 else np.asarray(countries)
    mw = np.empty(len(flat), dtype=np.float64)

    for country in np.unique(labels):
        mask = labels == country
        mw[mask] = country_target_scalers[country].inverse_transform(flat[mask]).ravel()

    return mw.reshape(shape)


def cb_save(name, countries, origins, predictions_mw, y_true_mw=None, feature_set=None,
            track=None, train_seconds=None, n_params=None):
    predictions = _asarray(predictions_mw).astype(np.float64)
    countries = np.asarray(countries)
    times = pd.to_datetime(np.asarray(origins)).to_numpy()
    h = CB["HORIZON"]
    n = len(countries)

    assert predictions.ndim == 2 and predictions.shape == (n, h), f"{name}: expected {n} x {h} predictions"
    assert len(times) == n, f"{name}: countries and origins have different lengths"
    assert np.isfinite(predictions).all(), f"{name}: non-finite predictions"

    feature_set = CB["FE"] if feature_set is None else feature_set
    track = CB["TRACK"] if track is None else track
    run = "|".join([str(CB["NB"]), name, feature_set, track, str(CB["SEED"])])

    frame = pd.DataFrame({
        "notebook": str(CB["NB"]),
        "model": name,
        "feature_set": feature_set,
        "family": None,
        "track": track,
        "seed": CB["SEED"],
        "run": run,
        "country": np.repeat(countries, h),
        "origin": np.repeat(times, h),
        "h": np.tile(np.arange(1, h + 1), n),
        "y_pred": predictions.ravel(),
    })

    if y_true_mw is not None:
        actual = _asarray(y_true_mw).astype(np.float64)
        assert actual.shape == predictions.shape, f"{name}: wrong y_true_mw shape"
        assert np.isfinite(actual).all(), f"{name}: non-finite y_true_mw"
        frame["y_true_notebook"] = actual.ravel()

    frame["train_seconds"] = np.nan if train_seconds is None else train_seconds
    frame["n_params"] = np.nan if n_params is None else n_params

    os.makedirs(CB["PRED_DIR"], exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9]+", "_", run).strip("_") + ".parquet"
    frame.to_parquet(os.path.join(CB["PRED_DIR"], filename), index=False)
    return frame


def cb_manifest():
    paths = sorted(glob.glob(os.path.join(CB["PRED_DIR"], "*.parquet")))
    rows = []

    for path in paths:
        frame = pd.read_parquet(path)
        rows.append({
            "run": frame["run"].iloc[0],
            "origins": frame[["country", "origin"]].drop_duplicates().shape[0],
            "rows": len(frame),
            "file": os.path.basename(path),
        })

    manifest = pd.DataFrame(rows)
    print(manifest.to_string(index=False))
    return manifest


def cb_sample(samples, split, n):
    part = samples[samples["split"] == split]
    return part.sample(min(n, len(part)), random_state=CB["SEED"])


def cb_reference_suite(X_train, y_train, X_validation, y_validation, X_test,
                       countries, origins, to_mw, feature_set=None):
    Xtr = _asarray(X_train).astype(np.float64).reshape(len(X_train), -1)
    Xva = _asarray(X_validation).astype(np.float64).reshape(len(X_validation), -1)
    Xte = _asarray(X_test).astype(np.float64).reshape(len(X_test), -1)
    ytr = _asarray(y_train).astype(np.float64).reshape(len(y_train), -1)
    yva = _asarray(y_validation).astype(np.float64).reshape(len(y_validation), -1)

    assert np.isfinite(Xtr).all() and np.isfinite(Xva).all() and np.isfinite(Xte).all()
    assert np.isfinite(ytr).all() and np.isfinite(yva).all()
    assert ytr.shape[1] == CB["HORIZON"], "reference targets must have 24 columns"
    assert Xte.shape[0] == len(countries)

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)

    models = {
        "REF-Ridge": Ridge(alpha=1.0),
        "REF-MLP": MLPRegressor(
            hidden_layer_sizes=(256, 256),
            max_iter=200,
            batch_size=256,
            learning_rate_init=1e-3,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=CB["SEED"],
        ),
    }

    for name, model in models.items():
        model.fit(Xtr, ytr)
        predictions = model.predict(Xte)
        cb_save(name, countries, origins, to_mw(predictions, countries), feature_set=feature_set)


def cb_load_predictions(pred_dirs):
    paths = sorted(
        path for folder in pred_dirs for path in glob.glob(os.path.join(folder, "*.parquet"))
    )
    pred = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    pred["origin"] = pd.to_datetime(pred["origin"])
    pred["h"] = pred["h"].astype(int)
    pred["seed"] = pred["seed"].astype(int)

    for column in ["family", "train_seconds", "n_params"]:
        if column not in pred:
            pred[column] = np.nan

    return pred


def _actual_lookup(raw):
    frame = pd.DataFrame({
        "country": raw[CB["COUNTRY"]].to_numpy(),
        "time": pd.to_datetime(raw[CB["TIME"]]).to_numpy(),
        "load": raw[CB["TARGET"]].to_numpy(),
    }).drop_duplicates(["country", "time"])
    return frame.set_index(["country", "time"])["load"]


def cb_attach_actuals(pred, raw):
    frame = pred.copy()
    lookup = _actual_lookup(raw)
    target = pd.to_datetime(frame["origin"]).to_numpy() + frame["h"].to_numpy().astype("timedelta64[h]")
    frame["y_true"] = lookup.reindex(
        pd.MultiIndex.from_arrays([frame["country"].to_numpy(), target])
    ).to_numpy()

    if "y_true_notebook" in frame:
        difference = (frame["y_true_notebook"] - frame["y_true"]).abs()
        worst = difference.groupby(frame["run"]).max().dropna()
        wrong = worst[worst > 1]

        if len(wrong):
            print("Runs whose own y_true differs from the CSV by more than 1 MW:")
            print(wrong.round(3).to_string())

    return frame


def cb_seasonal_naive(keys, raw, lag, name):
    lookup = _actual_lookup(raw)
    h = CB["HORIZON"]
    countries = np.repeat(keys["country"].to_numpy(), h)
    origins = np.repeat(pd.to_datetime(keys["origin"]).to_numpy(), h)
    horizon = np.tile(np.arange(1, h + 1), len(keys))
    target = origins + horizon.astype("timedelta64[h]")
    naive_time = target - np.timedelta64(int(lag), "h")
    y_pred = lookup.reindex(pd.MultiIndex.from_arrays([countries, naive_time])).to_numpy()

    return pd.DataFrame({
        "notebook": "CB",
        "model": name,
        "feature_set": "naive",
        "family": "naive",
        "track": "honest",
        "seed": 0,
        "run": name,
        "country": countries,
        "origin": origins,
        "h": horizon,
        "y_pred": y_pred,
        "train_seconds": np.nan,
        "n_params": np.nan,
    })


def cb_common_set(honest, min_coverage=0.5):
    frame = honest.dropna(subset=["y_pred", "y_true"]).copy()
    union = frame[["country", "origin"]].drop_duplicates()
    sizes = frame.groupby(["run", "country", "origin"]).size()
    pairs = sizes[sizes == CB["HORIZON"]].reset_index()[["run", "country", "origin"]]

    per_run = pairs.groupby("run").size()
    coverage = per_run.reindex(frame["run"].unique()).fillna(0) / max(len(union), 1)
    kept = coverage[coverage >= min_coverage].index

    common_sizes = pairs[pairs["run"].isin(kept)].groupby(["country", "origin"]).size()
    common = common_sizes[common_sizes == len(kept)].reset_index()[["country", "origin"]]
    com = frame[frame["run"].isin(kept)].merge(common, on=["country", "origin"], how="inner")
    assert len(com) == len(kept) * len(common) * CB["HORIZON"], "runs do not share the same test origins"

    return com, common, coverage


def cb_metrics(com):
    rows = []

    for run, group in com.groupby("run", sort=False):
        actual = group["y_true"].to_numpy(dtype=np.float64)
        predicted = group["y_pred"].to_numpy(dtype=np.float64)
        error = predicted - actual
        nonzero = np.abs(actual) > 1e-8
        ape = np.abs(error[nonzero] / actual[nonzero])
        per_country = pd.DataFrame({
            "country": group["country"].to_numpy()[nonzero],
            "ape": ape,
        }).groupby("country")["ape"].mean()
        first = group.iloc[0]

        rows.append({
            "run": run,
            "notebook": first["notebook"],
            "model": first["model"],
            "feature_set": first["feature_set"],
            "family": first["family"],
            "seed": first["seed"],
            "Country_MAPE": per_country.mean() * 100,
            "MAPE": ape.mean() * 100,
            "WAPE": np.abs(error).sum() / np.abs(actual).sum() * 100,
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "MAE": np.mean(np.abs(error)),
            "Bias_pct": error.mean() / actual.mean() * 100,
            "train_seconds": first.get("train_seconds", np.nan),
            "n_params": first.get("n_params", np.nan),
        })

    return pd.DataFrame(rows)


def cb_aggregate_seeds(m, primary):
    keys = ["notebook", "model", "feature_set", "family"]
    values = [
        column for column in
        ["Country_MAPE", "MAPE", "WAPE", "RMSE", "MAE", "Bias_pct", "train_seconds", "n_params"]
        if column in m
    ]
    grouped = m.groupby(keys, dropna=False)
    board = grouped[values].mean()
    board[f"{primary}_sd"] = grouped[primary].std(ddof=0).fillna(0)
    board["n_seeds"] = grouped.size()
    return board.reset_index().sort_values(primary).reset_index(drop=True)


def cb_skill(board, baseline, primary):
    rows = board.loc[board["model"] == baseline, primary]

    if len(rows):
        board[f"Skill_vs_{baseline}"] = (1 - board[primary] / rows.iloc[0]) * 100

    return board


def cb_daily_loss(com, run):
    group = com[com["run"] == run]
    loss = (group["y_pred"] - group["y_true"]).abs() / group["y_true"].abs().replace(0, np.nan)
    return loss.groupby([group["country"], group["origin"]]).mean().rename(run)


def cb_dm_test(best_loss, other_loss):
    aligned = pd.concat([best_loss, other_loss], axis=1, join="inner").dropna()
    difference = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).to_numpy(dtype=np.float64)
    n = len(difference)

    if n < 2:
        return float("nan"), float("nan")

    mean = difference.mean()
    centred = difference - mean
    lag = 7
    variance = np.dot(centred, centred) / n

    for k in range(1, min(lag, n - 1) + 1):
        gamma = np.dot(centred[k:], centred[:-k]) / n
        variance += 2 * (1 - k / (lag + 1)) * gamma

    standard_error = math.sqrt(max(variance, 0) / n)

    if standard_error == 0:
        return float("nan"), float("nan")

    dm = mean / standard_error
    p_value = math.erfc(abs(dm) / math.sqrt(2))
    return dm, p_value


def cb_per_horizon(com, runs):
    columns = {}

    for run in runs:
        group = com[com["run"] == run]
        ape = 100 * (group["y_pred"] - group["y_true"]).abs() / group["y_true"].abs().replace(0, np.nan)
        columns[run] = ape.groupby(group["h"]).mean()

    return pd.DataFrame(columns)


def cb_per_country(com, runs):
    columns = {}

    for run in runs:
        group = com[com["run"] == run]
        ape = 100 * (group["y_pred"] - group["y_true"]).abs() / group["y_true"].abs().replace(0, np.nan)
        columns[run] = ape.groupby(group["country"]).mean()

    return pd.DataFrame(columns)

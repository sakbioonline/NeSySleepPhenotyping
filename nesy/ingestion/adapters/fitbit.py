"""
Fitbit intraday → EmbracePlus adapter.

Converts Fitbit intraday heart rate and steps exports into the columnar
format expected by the NeSy ingestion layer (load_embraceplus output schema).

Fitbit intraday HR and steps are available at 1-minute resolution from the
Fitbit Web API or research data export. The adapter handles three input
formats:

  Format A — single combined CSV per participant:
    columns: datetime, heart_rate, steps
    (produced by Fitbit research export or custom API download script)

  Format B — separate HR and steps files per participant:
    hr_file:    datetime, heart_rate
    steps_file: datetime, steps

  Format C — Fitbit JSON export (raw API format):
    "activities-heart-intraday": {"dataset": [{"time": "HH:MM:SS", "value": N}]}
    "activities-steps-intraday": {"dataset": [{"time": "HH:MM:SS", "value": N}]}

Output schema matches load_embraceplus():
    participant_id, datetime, hr_bpm, steps, prv_ms
    indexed by (participant_id, datetime)

Notes:
  - Fitbit does not provide PRV (pulse rate variability) at 1-min resolution.
    prv_ms is set to NaN throughout; EEG symbols and RAR metrics are unaffected.
  - Non-wear detection: epochs with hr_bpm < 30 or hr_bpm > 220 are flagged
    as non-wear (same threshold as EmbracePlus adapter).
  - ROCKET will use hr_bpm + steps as the activity signal, matching the
    EmbracePlus path. The absence of PRV reduces the feature space slightly
    but does not prevent ROCKET from running.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


# ── Column name variants across Fitbit export formats ────────────────────────
_HR_ALIASES = [
    "heart_rate", "heart rate", "Heart Rate", "HeartRate",
    "bpm", "BPM", "hr", "HR", "value",
]
_STEPS_ALIASES = [
    "steps", "Steps", "STEPS", "step_count", "StepCount",
]
_DT_ALIASES = [
    "datetime", "Datetime", "DateTime", "timestamp", "Timestamp",
    "time", "Time", "date_time",
]


def _find_col(df, aliases):
    for a in aliases:
        if a in df.columns:
            return a
    return None


def _parse_combined_csv(path, date_hint=None):
    """Parse Format A — single CSV with datetime, heart_rate, steps."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    dt_col  = _find_col(df, _DT_ALIASES)
    hr_col  = _find_col(df, _HR_ALIASES)
    stp_col = _find_col(df, _STEPS_ALIASES)

    if dt_col is None:
        raise ValueError(
            f"No datetime column found in {Path(path).name}. "
            f"Available: {list(df.columns[:8])}"
        )

    df["_dt"] = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
    if date_hint:
        df["_dt"] = df["_dt"].fillna(
            pd.to_datetime(df[dt_col].astype(str).str.strip(),
                           format="%H:%M:%S", errors="coerce")
            .apply(lambda t: pd.Timestamp(date_hint).replace(
                hour=t.hour, minute=t.minute, second=t.second)
                if pd.notna(t) else pd.NaT)
        )

    df["hr_bpm"]  = pd.to_numeric(df[hr_col],  errors="coerce") \
                    if hr_col  else np.nan
    df["steps"]   = pd.to_numeric(df[stp_col], errors="coerce") \
                    if stp_col else 0.0

    return df[["_dt", "hr_bpm", "steps"]].rename(columns={"_dt": "datetime"})


def _parse_separate_csvs(hr_path, steps_path):
    """Parse Format B — separate HR and steps files."""
    hr_df  = pd.read_csv(hr_path)
    stp_df = pd.read_csv(steps_path)

    hr_df.columns  = [str(c).strip() for c in hr_df.columns]
    stp_df.columns = [str(c).strip() for c in stp_df.columns]

    hr_dt  = _find_col(hr_df,  _DT_ALIASES)
    stp_dt = _find_col(stp_df, _DT_ALIASES)
    hr_col = _find_col(hr_df,  _HR_ALIASES)
    st_col = _find_col(stp_df, _STEPS_ALIASES)

    hr_df["datetime"]  = pd.to_datetime(hr_df[hr_dt],
                                         dayfirst=True, errors="coerce")
    stp_df["datetime"] = pd.to_datetime(stp_df[stp_dt],
                                         dayfirst=True, errors="coerce")

    hr_df["hr_bpm"] = pd.to_numeric(hr_df[hr_col], errors="coerce")
    stp_df["steps"] = pd.to_numeric(stp_df[st_col], errors="coerce")

    merged = pd.merge(
        hr_df[["datetime", "hr_bpm"]],
        stp_df[["datetime", "steps"]],
        on="datetime", how="outer"
    ).sort_values("datetime").reset_index(drop=True)

    merged["steps"] = merged["steps"].fillna(0)
    return merged


def _parse_json(path, date_str):
    """Parse Format C — Fitbit API JSON for a single day."""
    with open(path) as f:
        data = json.load(f)

    def _extract(key, val_key="value"):
        try:
            dataset = data[key]["dataset"]
            times   = [d["time"] for d in dataset]
            values  = [d[val_key] for d in dataset]
            dts     = pd.to_datetime(
                [f"{date_str} {t}" for t in times], errors="coerce"
            )
            return pd.Series(values, index=dts, dtype=float)
        except (KeyError, TypeError):
            return pd.Series(dtype=float)

    hr_series  = _extract("activities-heart-intraday")
    stp_series = _extract("activities-steps-intraday")

    df = pd.DataFrame({
        "datetime": hr_series.index,
        "hr_bpm":   hr_series.values,
    })
    df = df.merge(
        pd.DataFrame({"datetime": stp_series.index, "steps": stp_series.values}),
        on="datetime", how="outer"
    ).sort_values("datetime").reset_index(drop=True)
    df["steps"] = df["steps"].fillna(0)
    return df


def load_fitbit(
    path,
    participant_id,
    hr_path=None,
    steps_path=None,
    date_hint=None,
):
    """
    Load Fitbit intraday data and return an EmbracePlus-compatible DataFrame.

    Parameters
    ----------
    path           : str or Path — combined CSV (Format A) or JSON (Format C).
                     Pass None if using separate hr_path / steps_path.
    participant_id : str — participant identifier
    hr_path        : str or Path — separate HR CSV (Format B only)
    steps_path     : str or Path — separate steps CSV (Format B only)
    date_hint      : str — ISO date string "YYYY-MM-DD" used when parsing
                     time-only columns (Format A with time not datetime)

    Returns
    -------
    DataFrame indexed by (participant_id, datetime) with columns:
        hr_bpm, steps, prv_ms (NaN throughout for Fitbit)
    """
    if hr_path and steps_path:
        df = _parse_separate_csvs(hr_path, steps_path)
    elif path and str(path).endswith(".json"):
        date_str = date_hint or str(Path(path).stem)[:10]
        df = _parse_json(path, date_str)
    else:
        df = _parse_combined_csv(path, date_hint=date_hint)

    # clean
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # resample to 1-minute grid — forward fill gaps up to 5 minutes
    df = df.set_index("datetime").resample("1min").mean()
    df["hr_bpm"] = df["hr_bpm"].interpolate(method="time", limit=5)
    df["steps"]  = df["steps"].fillna(0)
    df = df.reset_index()

    # non-wear flag: HR outside physiological range
    df["non_wear"] = (
        (df["hr_bpm"] < 30) | (df["hr_bpm"] > 220) | df["hr_bpm"].isna()
    )

    # PRV not available from Fitbit intraday
    df["prv_ms"] = np.nan

    # add participant_id and set index to match EmbracePlus schema
    df["participant_id"] = participant_id
    df = df.set_index(["participant_id", "datetime"])

    return df[["hr_bpm", "steps", "prv_ms", "non_wear"]]


def load_fitbit_cohort(data_dir, participant_ids=None):
    """
    Load all Fitbit files from a directory.

    Expects one of:
        data_dir/{pid}.csv          — combined Format A
        data_dir/{pid}_hr.csv +
        data_dir/{pid}_steps.csv    — separate Format B
        data_dir/{pid}/*.json       — daily Format C files

    Returns concatenated DataFrame in EmbracePlus schema.
    """
    data_dir = Path(data_dir)
    dfs = []

    # discover participant files
    combined = list(data_dir.glob("*.csv"))
    pids_found = set()

    for f in combined:
        stem = f.stem
        if stem.endswith("_hr") or stem.endswith("_steps"):
            continue
        pid = stem
        if participant_ids and pid not in participant_ids:
            continue
        pids_found.add(pid)

        hr_f  = data_dir / f"{pid}_hr.csv"
        stp_f = data_dir / f"{pid}_steps.csv"
        try:
            if hr_f.exists() and stp_f.exists():
                df = load_fitbit(None, pid, hr_path=hr_f, steps_path=stp_f)
            else:
                df = load_fitbit(f, pid)
            dfs.append(df)
        except Exception as e:
            print(f"  Fitbit load failed for {pid}: {e}")

    # JSON format
    for pid_dir in data_dir.iterdir():
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        if participant_ids and pid not in participant_ids:
            continue
        if pid in pids_found:
            continue
        json_files = sorted(pid_dir.glob("*.json"))
        if not json_files:
            continue
        day_dfs = []
        for jf in json_files:
            date_str = jf.stem[:10]
            try:
                day_dfs.append(load_fitbit(jf, pid, date_hint=date_str))
            except Exception as e:
                print(f"  Fitbit JSON load failed {pid}/{jf.name}: {e}")
        if day_dfs:
            dfs.append(pd.concat(day_dfs))

    if not dfs:
        raise FileNotFoundError(
            f"No Fitbit files found in {data_dir}. "
            "Expected *.csv, *_hr.csv/*_steps.csv, or pid/YYYY-MM-DD.json"
        )

    return pd.concat(dfs).sort_index()

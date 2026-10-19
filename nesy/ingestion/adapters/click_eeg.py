"""
Click EEG → Dreem adapter.

Maps Click EEG output column names (snake_case) to the Dreem summary schema
used throughout the NeSy pipeline. Click EEG produces the same sleep stage
variables as Dreem but uses lowercase underscore-separated names.

Column mapping (Click EEG → Dreem):
    participant_id  → participant_id   (unchanged)
    rec_date        → rec_date         (unchanged — already yyyy-mm-dd)
    tst_min         → TST_min
    sol_min         → SOL_min
    waso_min        → WASO_min
    se_pct          → SE_pct
    n1_pct          → N1_pct
    n2_pct          → N2_pct
    n3_pct          → N3_pct
    rem_pct         → REM_pct
    ahi             → AHI              (if present)
    epoch_file      → epoch_file       (path to per-epoch staging CSV)

Per-epoch staging CSV (Click EEG):
    time, stage
    where stage is one of: wake, n1, n2, n3, rem (lowercase)

    Maps to Dreem per-epoch format:
    time, sleep_stage
    where sleep_stage is one of: 0=wake, 1=N1, 2=N2, 3=N3, 4=REM

Usage:
    from nesy.ingestion.adapters.click_eeg import load_click_eeg_summary
    from nesy.ingestion.adapters.click_eeg import load_click_eeg_epochs

    dreem_summary = load_click_eeg_summary("22q_click_eeg_summary.csv")
    dreem_epochs  = load_click_eeg_epochs("22q_click_eeg_epochs/")
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Column mapping: Click EEG snake_case → Dreem PascalCase ─────────────────
SUMMARY_COL_MAP = {
    # identity (already correct)
    "participant_id":  "participant_id",
    "rec_date":        "rec_date",

    # sleep architecture
    "tst_min":         "TST_min",
    "sol_min":         "SOL_min",
    "waso_min":        "WASO_min",
    "se_pct":          "SE_pct",
    "n1_pct":          "N1_pct",
    "n2_pct":          "N2_pct",
    "n3_pct":          "N3_pct",
    "rem_pct":         "REM_pct",

    # optional
    "ahi":             "AHI",
    "epoch_count":     "epoch_count",
    "lights_off":      "lights_off",
    "lights_on":       "lights_on",
    "recording_duration_min": "recording_duration_min",
}

# Stage label mapping: Click EEG string → integer (Dreem convention)
STAGE_LABEL_MAP = {
    "wake":    0,
    "w":       0,
    "0":       0,
    "n1":      1,
    "1":       1,
    "nrem1":   1,
    "n2":      2,
    "2":       2,
    "nrem2":   2,
    "n3":      3,
    "3":       3,
    "nrem3":   3,
    "sws":     3,   # slow wave sleep = N3
    "rem":     4,
    "r":       4,
    "4":       4,
    "unknown": -1,
    "artefact":-1,
    "artifact":-1,
    "unscored":-1,
    "-1":      -1,
}


def load_click_eeg_summary(path):
    """
    Load Click EEG nightly summary CSV and return in Dreem summary schema.

    Parameters
    ----------
    path : str or Path — Click EEG summary CSV

    Returns
    -------
    DataFrame indexed by (participant_id, night_date) with Dreem column names.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower().replace(" ", "_")
                  for c in df.columns]

    # check participant_id column exists
    if "participant_id" not in df.columns:
        # try common alternatives
        for alt in ["pid", "id", "subject", "subject_id", "participant"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "participant_id"})
                break
        if "participant_id" not in df.columns:
            raise ValueError(
                f"No participant_id column found in {Path(path).name}. "
                f"Available: {list(df.columns[:10])}"
            )

    # rename columns
    df = df.rename(columns={k: v for k, v in SUMMARY_COL_MAP.items()
                             if k in df.columns})

    # normalise rec_date to yyyy-mm-dd
    df["rec_date"] = pd.to_datetime(
        df["rec_date"], dayfirst=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # add night_date alias (Dreem convention: date of recording morning)
    df["night_date"] = df["rec_date"]

    # ensure numeric columns
    numeric_cols = ["TST_min", "SOL_min", "WASO_min", "SE_pct",
                    "N1_pct", "N2_pct", "N3_pct", "REM_pct"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # set index
    df = df.set_index(["participant_id", "night_date"]).sort_index()

    return df


def load_click_eeg_epochs(epochs_dir, summary_df=None):
    """
    Load Click EEG per-epoch staging files and return in Dreem epoch schema.

    Expects one CSV per night, named either:
        {participant_id}_{YYYY-MM-DD}_epochs.csv
        {participant_id}_{YYYY-MM-DD}.csv
        {YYYY-MM-DD}_{participant_id}.csv

    Each file has columns: time (HH:MM:SS), stage (string label)

    Parameters
    ----------
    epochs_dir : str or Path — directory containing per-night epoch CSVs
    summary_df : DataFrame — optional Click EEG summary (used to cross-check
                 participant IDs and dates)

    Returns
    -------
    DataFrame indexed by (participant_id, night_date, epoch_number) with
    columns: sleep_stage (int), datetime (Timestamp)
    Matches load_dreem_hypnogram() output schema.
    """
    epochs_dir = Path(epochs_dir)
    epoch_files = sorted(
        list(epochs_dir.glob("*.csv")) +
        list(epochs_dir.glob("*.tsv"))
    )

    if not epoch_files:
        raise FileNotFoundError(
            f"No epoch files found in {epochs_dir}. "
            "Expected files named {pid}_{YYYY-MM-DD}_epochs.csv"
        )

    records = []

    for f in epoch_files:
        stem  = f.stem.replace("_epochs", "")
        parts = stem.split("_")

        # parse pid and date from filename
        pid       = None
        night_str = None
        for part in parts:
            try:
                pd.to_datetime(part)
                night_str = part
            except Exception:
                if pid is None:
                    pid = part
                else:
                    pid = pid + "_" + part  # handle underscores in pid

        if pid is None or night_str is None:
            print(f"  Click EEG: cannot parse pid/date from {f.name} — skipping")
            continue

        try:
            sep = "\t" if f.suffix == ".tsv" else ","
            ep  = pd.read_csv(f, sep=sep, low_memory=False)
            ep.columns = [str(c).strip().lower().replace(" ", "_")
                          for c in ep.columns]

            # find stage column
            stage_col = None
            for candidate in ["stage", "sleep_stage", "label", "scoring",
                               "epoch_stage", "annotation"]:
                if candidate in ep.columns:
                    stage_col = candidate
                    break
            if stage_col is None:
                print(f"  Click EEG: no stage column in {f.name} — skipping")
                continue

            # map stage labels to integers
            ep["sleep_stage"] = (
                ep[stage_col].astype(str).str.strip().str.lower()
                .map(STAGE_LABEL_MAP)
                .fillna(-1)
                .astype(int)
            )

            # parse time column if present
            time_col = None
            for tc in ["time", "epoch_time", "start_time", "timestamp"]:
                if tc in ep.columns:
                    time_col = tc
                    break

            night_date = pd.to_datetime(night_str).strftime("%Y-%m-%d")

            if time_col:
                # combine night date with epoch time
                ep["datetime"] = pd.to_datetime(
                    night_date + " " + ep[time_col].astype(str),
                    errors="coerce"
                )
            else:
                # generate timestamps from epoch number (30s epochs assumed)
                base = pd.Timestamp(night_date + " 21:00:00")
                ep["datetime"] = [base + pd.Timedelta(seconds=30 * i)
                                   for i in range(len(ep))]

            ep["participant_id"] = pid
            ep["night_date"]     = night_date
            ep["epoch_number"]   = range(len(ep))

            records.append(ep[["participant_id", "night_date", "epoch_number",
                                "datetime", "sleep_stage"]])

        except Exception as e:
            print(f"  Click EEG: failed to load {f.name}: {e}")

    if not records:
        raise ValueError("No Click EEG epoch files loaded successfully.")

    result = pd.concat(records, ignore_index=True)
    result = result.set_index(
        ["participant_id", "night_date", "epoch_number"]
    ).sort_index()

    return result


def load_click_eeg(summary_path, epochs_dir=None):
    """
    Convenience loader: load both summary and epochs in one call.

    Returns
    -------
    summary_df : DataFrame in Dreem summary schema
    epochs_df  : DataFrame in Dreem epoch schema (None if epochs_dir not provided)
    """
    summary_df = load_click_eeg_summary(summary_path)

    epochs_df = None
    if epochs_dir is not None:
        try:
            epochs_df = load_click_eeg_epochs(epochs_dir, summary_df)
        except FileNotFoundError as e:
            print(f"  Click EEG: epochs not loaded — {e}")

    return summary_df, epochs_df

"""
Raw data loaders for the three SSP modalities.

Each loader reads one source, normalises column names and types,
and returns a clean dataframe satisfying its schema contract.
Dreem epoch loading transparently runs the EpochAligner when
the device config declares pre_aligned: false.

File conventions observed in the SSP cohort:

    EmbracePlus:  data/ssp/embraceplus/SP0013553.csv
    Dreem summary: data/ssp/dreem/dreem_summary.csv
    Dreem hypno:  data/ssp/dreem/dreem_hypno_SP0013553_2023-12-14.csv
    Sleep diary:  data/ssp/diary/ssp_sleep_diary.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# column contracts and stage encodings
EMBRACEPLUS_TIMESERIES_COLS = [
    "participant_id", "datetime", "activity_counts", "step_counts",
    "sleep_stage_ep", "pulse_rate_bpm", "prv_rmssd_ms", "respiratory_rate_brpm",
]

DREEM_SUMMARY_COLS = [
    "participant_id", "night_date", "TST_min", "SE_pct", "SOL_min", "WASO_min",
    "N1_min", "N2_min", "N3_min", "REM_min",
    "N1_pct", "N2_pct", "N3_pct", "REM_pct", "recording_quality",
]

DREEM_EPOCH_COLS = [
    "participant_id", "night_date", "epoch_start", "sleep_stage", "duration_sec",
]

DIARY_COLS = [
    "participant_id", "night_date", "ANX", "DIFF", "SQ",
    "num_awakenings", "how_energetic", "feeling_physically", "mood", "diary_missing",
]

NIGHT_RECORD_REQUIRED_COLS = [
    "participant_id", "night_date",
    "SE_pct", "WASO_min", "SOL_min", "TST_min",
    "N1_pct", "N2_pct", "N3_pct", "REM_pct",
    "ANX", "DIFF", "SQ", "diary_missing",
]

EMBRACEPLUS_STAGE_MAP = {0.0: "wake", 101.0: "light_sleep", 102.0: "deep_sleep", 300.0: "unknown"}

DREEM_STAGE_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "": -1}

from nesy.ingestion.align_epochs import EpochAligner, needs_alignment, AlignmentReport

# ------------------------------------------------------------------
# Default device alignment config
# Used when devices.yaml is not loaded by the caller
# ------------------------------------------------------------------

_DREEM_DEFAULT_ALIGNMENT = {
    "pre_aligned": False,
    "method": "majority_overlap",
    "min_overlap_sec": 15,
}

_FITBIT_DEFAULT_ALIGNMENT = {
    "pre_aligned": True,
}


# ------------------------------------------------------------------
# EmbracePlus loader
# ------------------------------------------------------------------

def load_embraceplus(file_path: str | Path) -> pd.DataFrame:
    """
    Load a single participant's EmbracePlus CSV into a clean time series.

    Parameters
    ----------
    file_path:
        Path to the participant CSV, e.g. embraceplus/SP0013553.csv.
        Participant ID is derived from the filename stem.

    Returns
    -------
    DataFrame indexed by datetime with columns matching
    EMBRACEPLUS_TIMESERIES_COLS.
    """
    path = Path(file_path)
    participant_id = path.stem

    raw = pd.read_csv(path, dtype=str)
    raw.columns = raw.columns.str.strip()

    raw["datetime"] = pd.to_datetime(
        raw["date"].str.strip() + " " + raw["time"].str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        utc=False,
    )

    df = pd.DataFrame({
        "participant_id": participant_id,
        "datetime": raw["datetime"],
        "activity_counts": pd.to_numeric(raw["activity_counts"], errors="coerce"),
        "step_counts": pd.to_numeric(raw["step_counts"], errors="coerce"),
        "sleep_stage_ep": pd.to_numeric(raw["sleep_detection_stage"], errors="coerce"),
        "pulse_rate_bpm": pd.to_numeric(raw["pulse_rate_bpm"], errors="coerce"),
        "prv_rmssd_ms": pd.to_numeric(raw["prv_rmssd_ms"], errors="coerce"),
        "respiratory_rate_brpm": pd.to_numeric(raw["respiratory_rate_brpm"], errors="coerce"),
    })

    return df.set_index("datetime").sort_index()


def load_embraceplus_directory(directory: str | Path) -> pd.DataFrame:
    """
    Load all EmbracePlus CSVs in a directory and concatenate them.

    Returns
    -------
    Concatenated DataFrame across all participants, sorted by datetime.
    """
    directory = Path(directory)
    files = sorted(directory.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in EmbracePlus directory: {directory}"
        )

    return pd.concat(
        [load_embraceplus(f) for f in files]
    ).sort_index()


# ------------------------------------------------------------------
# Dreem summary loader
# ------------------------------------------------------------------

def load_dreem_summary(file_path: str | Path) -> pd.DataFrame:
    """
    Load the cohort-wide Dreem sleep summary CSV.

    Maps raw column names to canonical framework names and parses
    rec_date (DD/MM/YYYY) to a proper date string (YYYY-MM-DD).

    Returns
    -------
    DataFrame indexed by (participant_id, night_date).
    """
    path = Path(file_path)
    raw = pd.read_csv(path, dtype=str)
    raw.columns = raw.columns.str.strip()

    night_date = pd.to_datetime(
        raw["sleep_date"].str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    tst = pd.to_numeric(raw["tst"], errors="coerce")
    trt_raw = pd.to_numeric(raw["trt"], errors="coerce")
    se = pd.to_numeric(raw["se"], errors="coerce")

    # TRT unit-inconsistency correction: TRT is in seconds for the vast
    # majority of rows, but a small subset (verified via a SE=TST/TRT*100
    # cross-check) have TRT already in minutes. Detect via the SE mismatch
    # under the seconds assumption and correct those rows to seconds too,
    # so TRT_min is uniform across the whole dataframe.
    se_check_sec = tst / (trt_raw / 60) * 100
    unit_bug = (se - se_check_sec).abs() > 5
    trt_sec = trt_raw.where(~unit_bug, trt_raw * 60)
    trt_min = trt_sec / 60

    df = pd.DataFrame({
        "participant_id": raw["id"].str.strip(),
        "night_date": night_date,
        "TST_min": tst,
        "SE_pct": se,
        "SOL_min": pd.to_numeric(raw["sol"], errors="coerce"),
        "WASO_min": pd.to_numeric(raw["waso"], errors="coerce"),
        "N1_min": pd.to_numeric(raw["n1"], errors="coerce"),
        "N2_min": pd.to_numeric(raw["n2"], errors="coerce"),
        "N3_min": pd.to_numeric(raw["n3"], errors="coerce"),
        "REM_min": pd.to_numeric(raw["rem"], errors="coerce"),
        "N1_pct": pd.to_numeric(raw["p_n1"], errors="coerce"),
        "N2_pct": pd.to_numeric(raw["p_n2"], errors="coerce"),
        "N3_pct": pd.to_numeric(raw["p_n3"], errors="coerce"),
        "REM_pct": pd.to_numeric(raw["p_rem"], errors="coerce"),
        "recording_quality": pd.to_numeric(raw["qual"], errors="coerce"),
        "TRT_min": trt_min,
    })

    return df.dropna(
        subset=["participant_id", "night_date"]
    ).set_index(["participant_id", "night_date"]).sort_index()


# ------------------------------------------------------------------
# Dreem night-level quality flags
# ------------------------------------------------------------------

def flag_dreem_quality(dreem, long_session_hours=11.0, implausible_rem_pct=40.0):
    """
    Add long_session_flag and implausible_rem_flag to a Dreem summary
    dataframe (as returned by load_dreem_summary).

    long_session_flag: TRT_min > long_session_hours*60 minutes — recording
        spans more than one sleep episode (merged/duplicate session), not
        a genuinely long single night. Architecture check (elevated WASO
        and REM%, flat N3 despite higher TST) supports exclusion rather
        than treating these as real long nights.
    implausible_rem_flag: REM_pct > implausible_rem_pct — physiologically
        implausible REM share (normal ~20-25%). Independent of
        recording_quality (qual): a clean signal can still be
        catastrophically mis-staged, so qual alone does not catch this.

    Returns a copy of dreem with both flag columns added (int 0/1).
    """
    out = dreem.copy()
    out["long_session_flag"] = (out["TRT_min"] > long_session_hours * 60).astype(int)
    out["implausible_rem_flag"] = (out["REM_pct"] > implausible_rem_pct).astype(int)
    return out


def dreem_clean_nights(dreem):
    """Return only nights passing both quality flags (0 on both)."""
    flagged = flag_dreem_quality(dreem)
    return flagged[
        (flagged["long_session_flag"] == 0) & (flagged["implausible_rem_flag"] == 0)
        ].drop(columns=["long_session_flag", "implausible_rem_flag"])


# ------------------------------------------------------------------
# Dreem hypnogram loader — with transparent alignment
# ------------------------------------------------------------------

# def load_dreem_hypnogram(
#         file_path: str | Path,
#         alignment_config: dict | None = None,
# ) -> pd.DataFrame:
#     """
#     Load a single night's Dreem epoch hypnogram.
#
#     Participant ID and night date are parsed from the filename
#     (dreem_hypno_{participant_id}_{YYYY-MM-DD}.csv).
#
#     If alignment_config declares pre_aligned: false (the default for
#     Dreem), the raw epochs are passed through the EpochAligner before
#     being returned. The output is always on the canonical :00/:30 grid.
#
#     Parameters
#     ----------
#     file_path:
#         Path to a single hypnogram file.
#     alignment_config:
#         Alignment sub-dict from devices.yaml. Defaults to Dreem's
#         standard config (pre_aligned: false, min_overlap_sec: 15).
#
#     Returns
#     -------
#     DataFrame indexed by grid_start (UTC, :00/:30 grid) with columns:
#     participant_id, night_date, sleep_stage, duration_sec,
#     assignment_type.
#     """
#     path = Path(file_path)
#     participant_id, night_date = _parse_hypno_filename(path)
#     cfg = alignment_config or _DREEM_DEFAULT_ALIGNMENT
#
#     raw = pd.read_csv(path, dtype=str)
#     raw.columns = raw.columns.str.strip()
#
#     epoch_start = pd.to_datetime(
#         raw["time"].str.strip(), utc=True, errors="coerce"
#     )
#     stage_raw = raw["Sleep Stage"].str.strip().fillna("")
#     sleep_stage = stage_raw.map(DREEM_STAGE_MAP).fillna(-1).astype(int)
#
#     # ── Derive clinical night date from epoch timestamps ─────────────────────────
#     valid_epochs = epoch_start.dropna()
#     print(f"lenght epoch for {participant_id} {len(valid_epochs)}")
#
#     if not valid_epochs.empty:
#         first_epoch = valid_epochs.iloc[0]
#         print(f"First valid epoch for {participant_id} {first_epoch}")
#         # Optional: convert to local participant timezone first if timestamps are in UTC
#         # first_epoch = first_epoch.tz_convert("Europe/London")
#
#         # Shift back by 12 hours so 00:00-11:59 falls on T-1, 12:00-23:59 stays on T
#         clinical_night_date = (first_epoch - pd.Timedelta(hours=12)).date()
#     else:
#         # Fallback to T-1 from filename if timestamps are completely missing/corrupt
#         parsed_date = pd.to_datetime(night_date).date()
#         clinical_night_date = parsed_date - pd.Timedelta(days=1)
#
#     night_date = clinical_night_date
#     epochs = pd.DataFrame({
#         "participant_id": participant_id,
#         "night_date": night_date,
#         "epoch_start": epoch_start,
#         "sleep_stage": sleep_stage,
#         "duration_sec": pd.to_numeric(
#             raw["Duration[s]"], errors="coerce"
#         ).fillna(30).astype(int),
#     }).dropna(subset=["epoch_start"])
#
#     if not needs_alignment(cfg):
#         # Device already on the canonical grid — return as-is
#         return epochs.set_index("epoch_start").sort_index()
#
#     # Run the aligner
#     aligner = EpochAligner(
#         grid_resolution_sec=30,
#         min_overlap_sec=cfg.get("min_overlap_sec", 15),
#         fill_limit=cfg.get("fill_limit", 4),
#     )
#
#     aligned, report = aligner.align(
#         epochs,
#         participant_id=participant_id,
#         night_date=night_date,
#         device="dreem",
#     )
#
#     # Attach metadata columns and index on grid_start
#     aligned["participant_id"] = participant_id
#     aligned["night_date"] = night_date
#     aligned["duration_sec"] = 30
#
#     return aligned.set_index("grid_start").sort_index(), report
def load_dreem_hypnogram(
    file_path: str | Path,
    alignment_config: dict | None = None,
) -> tuple[pd.DataFrame, object]:
    """
    Load a single night's Dreem epoch hypnogram using local wall-clock time.

    Parameters
    ----------
    file_path:
        Path to a single hypnogram file.
    alignment_config:
        Alignment sub-dict from devices.yaml. Defaults to Dreem's
        standard config (pre_aligned: false, min_overlap_sec: 15).

    Returns
    -------
    tuple[pd.DataFrame, AlignmentReport]
        - DataFrame indexed by local grid_start (:00/:30 grid) with columns:
          participant_id, night_date, sleep_stage, duration_sec, assignment_type.
        - Alignment report dataclass.
    """
    path = Path(file_path)
    participant_id, filename_date = _parse_hypno_filename(path)
    cfg = alignment_config or _DREEM_DEFAULT_ALIGNMENT

    # 1. Ingest raw CSV
    raw = pd.read_csv(path, dtype=str)
    raw.columns = raw.columns.str.strip()

    # 2. Stage mapping (-1 for blank / NaN)
    # stage_raw = raw["Sleep Stage"].str.strip().fillna("")
    # sleep_stage = stage_raw.map(DREEM_STAGE_MAP).fillna(-1).astype(int)
    sleep_stage = (
        pd.to_numeric(raw["Sleep Stage"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    # 3. Truncate trailing unrecorded buffer epochs
    valid_mask = sleep_stage >= 0
    if valid_mask.any():
        last_valid_idx = np.where(valid_mask)[0].max()
        raw = raw.iloc[: last_valid_idx + 1].copy()
        sleep_stage = sleep_stage.iloc[: last_valid_idx + 1]

    # 4. Parse timestamps in pure local time (strip timezone offset)
    raw_time_str = raw["time"].str.strip()
    clean_time_str = raw_time_str.astype(str).str.replace(r"([+-]\d{2}:?\d{2}|Z)$", "", regex=True)
    #epoch_start = pd.to_datetime(raw_time_str, utc=True, errors="coerce").dt.tz_localize(None)
    epoch_start = pd.to_datetime(clean_time_str, errors="coerce")

    valid_epochs = epoch_start.dropna()
    if not valid_epochs.empty:
        first_local_dt = valid_epochs.iloc[0]
        clinical_night_date = (first_local_dt - pd.Timedelta(hours=12)).date()
    else:
        parsed_date = pd.to_datetime(filename_date).date()
        clinical_night_date = parsed_date - pd.Timedelta(days=1)

    night_date_str = str(clinical_night_date)

    # 5. Build cleaned epoch DataFrame
    epochs = pd.DataFrame({
        "participant_id": participant_id,
        "night_date": night_date_str,
        "epoch_start": epoch_start,
        "sleep_stage": sleep_stage,
        "duration_sec": pd.to_numeric(
            raw.get("Duration[s]", 30), errors="coerce"
        ).fillna(30).astype(int),
    }).dropna(subset=["epoch_start"])

    # 6. Grid alignment
    if not needs_alignment(cfg):
        return epochs.set_index("epoch_start").sort_index(), None

    aligner = EpochAligner(
        grid_resolution_sec=30,
        min_overlap_sec=cfg.get("min_overlap_sec", 15),
        fill_limit=cfg.get("fill_limit", 4),
    )

    aligned, report = aligner.align(
        epochs,
        participant_id=participant_id,
        night_date=night_date_str,
        device="dreem",
    )

    aligned["participant_id"] = participant_id
    aligned["night_date"] = night_date_str
    aligned["duration_sec"] = 30

    return aligned.set_index("grid_start").sort_index(), report


def load_dreem_hypnogram_directory(
        directory: str | Path,
        alignment_config: dict | None = None,
) -> tuple[pd.DataFrame, list[AlignmentReport]]:
    """
    Load all Dreem hypnogram files in a directory.

    Expects files named dreem_hypno_{participant_id}_{YYYY-MM-DD}.csv.

    Returns
    -------
    Tuple of (concatenated aligned epochs DataFrame, list of
    AlignmentReport objects — one per file).
    """
    directory = Path(directory)
    pattern = re.compile(r"dreem_hypno_.+_\d{4}-\d{2}-\d{2}\.csv$")
    files = sorted(
        f for f in directory.glob("*.csv") if pattern.match(f.name)
    )

    if not files:
        raise FileNotFoundError(
            f"No hypnogram files matching dreem_hypno_*_YYYY-MM-DD.csv "
            f"found in: {directory}"
        )

    frames = []
    reports = []

    for i, f in enumerate(files, 1):
        if i % 50 == 0 or i == len(files):
            print(f"    aligning epochs: {i}/{len(files)} files", end="\r")
        result = load_dreem_hypnogram(f, alignment_config)
        if isinstance(result, tuple):
            df, report = result
            reports.append(report)
        else:
            df = result
        frames.append(df)

    print()  # newline after progress
    return pd.concat(frames).sort_index(), reports


# ------------------------------------------------------------------
# Sleep diary loader
# ------------------------------------------------------------------

def load_sleep_diary(file_path: str | Path) -> pd.DataFrame:
    """
    Load the cohort-wide sleep diary CSV.

    sleep_date is the night being reported (diary filled next morning).
    We key on sleep_date to match Dreem and EmbracePlus conventions.

    Returns
    -------
    DataFrame indexed by (participant_id, night_date) with canonical
    column names. diary_missing is always False here — missing nights
    are flagged by align.py during the join.
    """
    path = Path(file_path)
    raw = pd.read_csv(path, dtype=str)
    raw.columns = raw.columns.str.strip()

    night_date = pd.to_datetime(
        raw["sleep_date"].str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    df = pd.DataFrame({
        "participant_id": raw["pid"].str.strip(),
        "night_date": night_date,
        "ANX": pd.to_numeric(raw["anxious_restless"], errors="coerce"),
        "DIFF": pd.to_numeric(raw["sleep_difficulty"], errors="coerce"),
        "SQ": pd.to_numeric(raw["morningFeeling"], errors="coerce"), #10.0 -
        "num_awakenings": pd.to_numeric(raw["num_awake"], errors="coerce"),
        "how_energetic": pd.to_numeric(raw["how_energetic"], errors="coerce"),
        "feeling_physically": pd.to_numeric(raw["feeling_physically"], errors="coerce"),
        "mood": pd.to_numeric(raw["mood"], errors="coerce"),
        "diary_missing": False,
    })

    return df.dropna(
        subset=["participant_id", "night_date"]
    ).set_index(["participant_id", "night_date"]).sort_index()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_HYPNO_PATTERN = re.compile(
    r"dreem_hypno_(?P<pid>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.csv$"
)


def _parse_hypno_filename(path: Path) -> tuple[str, str]:
    """
    Extract participant ID and night date from a hypnogram filename.

    Expected format: dreem_hypno_{participant_id}_{YYYY-MM-DD}.csv
    Raises ValueError for non-conforming filenames.
    """
    match = _HYPNO_PATTERN.match(path.name)
    if not match:
        raise ValueError(
            f"Hypnogram filename does not match expected pattern "
            f"'dreem_hypno_{{pid}}_{{YYYY-MM-DD}}.csv': {path.name}"
        )
    return match.group("pid"), match.group("date")


def load_demographics(filepath):
    """
    Load participant demographics from ssp_participant_data.csv.

    Returns DataFrame indexed by participant_id with columns:
        age, age_band, sex_encoded, autism_status, family_id
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={"pid": "participant_id"})

    # age — use real_age (age at study), cast to int, flag missing
    df["age"] = pd.to_numeric(df["real_age"], errors="coerce")
    df["age_band"] = df["age"].apply(
        lambda a: 0 if 10 <= a <= 13 else (1 if 14 <= a <= 17 else -1)
        if pd.notna(a) else -1
    )

    # sex: Male=0, Female=1
    df["sex_encoded"] = df["sex"].map({"Male": 0, "Female": 1}).fillna(-1).astype(int)

    # autism status: True=1, False=0
    df["autism_status"] = df["asd"].astype(int)

    return df.set_index("participant_id")[
        ["age", "age_band", "sex_encoded", "autism_status", "family_id"]
    ]


def prune_candidate_outlier_nights(
        dreem_clean: pd.DataFrame,
        min_nights: int = 5,
        qual_threshold: float = 0.50,
        candidate_floor: float = 0.40,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Selective Night-Pruning Quality Recovery.

  Identifies borderline participants (0.45 <= qual_mean < 0.50), iteratively
  drops their lowest-quality outlier night(s), and returns the updated
  dreem_clean table alongside a pruning summary dict.
  """
    df = dreem_clean.copy()

    # 1. Compute initial participant-level mean quality
    p_qual = df.groupby(level="participant_id")["recording_quality"].agg(
        ["count", "mean"]
    )

    # 2. Identify candidates with enough nights to spare (> min_nights) in [0.45, 0.50)
    candidates = p_qual[
        (p_qual["count"] > min_nights)
        & (p_qual["mean"] >= candidate_floor)
        & (p_qual["mean"] < qual_threshold)
        ].index.tolist()

    nights_to_drop = []
    pruned_counts = {}

    # 3. Iteratively prune lowest-quality night per candidate
    for pid in candidates:
        p_nights = df.xs(pid, level="participant_id").sort_values(
            "recording_quality", ascending=True
        )
        dropped = 0

        while (
                len(p_nights) > min_nights
                and p_nights["recording_quality"].mean() < qual_threshold
        ):
            lowest_night_date = p_nights.index[0]
            nights_to_drop.append((pid, lowest_night_date))
            p_nights = p_nights.iloc[1:]
            dropped += 1

        if dropped > 0:
            pruned_counts[pid] = dropped

    # 4. Drop pruned outlier nights from dreem_clean
    if nights_to_drop:
        drop_idx = pd.MultiIndex.from_tuples(
            nights_to_drop, names=["participant_id", "night_date"]
        )
        df = df.drop(index=drop_idx, errors="ignore")

    return df, pruned_counts


def apply_analytic_sample_filter(
        ep_ts,
        dreem,
        dreem_clean,
        diary,
        rar_participant,
        rar_daily,
        min_rar_days=5,
        min_aligned_nights=5,
        min_qual_mean=0.5,
        qual_mean_exceptions=None,
):
    # ------------------------------------------------------------------
    # STEP A: Apply targeted outlier pruning to dreem_clean
    # ------------------------------------------------------------------
    pruned_summary = {}
    if dreem_clean is not None:
        dreem_clean, pruned_summary = prune_candidate_outlier_nights(
            dreem_clean,
            min_nights=min_aligned_nights,
            qual_threshold=min_qual_mean,
            candidate_floor=0.40,
        )

    # ------------------------------------------------------------------
    # STEP B: Existing quality counts & alignment logic (unchanged)
    # ------------------------------------------------------------------
    rar_counts = (
        rar_participant["n_days"]
        if "n_days" in rar_participant.columns
        else pd.Series(0, index=rar_participant.index)
    )

    if dreem_clean is not None and "recording_quality" in dreem_clean.columns:
        # Recalculates qual_mean post-pruning
        qual_counts = dreem_clean.groupby(level="participant_id")[
            "recording_quality"
        ].mean()
    else:
        qual_counts = pd.Series(dtype=float)

    qual_mean_exceptions = set(qual_mean_exceptions or [])

    from nesy.ingestion.align import aggregate_embraceplus_daily, align_modalities
    from nesy.ingestion.rar import merge_rar_into_night_record

    ep_daily = aggregate_embraceplus_daily(ep_ts) if ep_ts is not None else None
    aligned = (
        align_modalities(dreem_clean, diary, ep_daily)
        if dreem_clean is not None
        else None
    )

    if aligned is not None and rar_daily is not None:
        aligned = merge_rar_into_night_record(aligned, rar_participant, rar_daily)
        rar_daily_valid = aligned[["IV", "L5", "M10"]].notna().all(axis=1)
        aligned["fully_aligned"] = (~aligned["diary_missing"]) & rar_daily_valid
        aligned_counts = (
            aligned[aligned["fully_aligned"]].groupby(level="participant_id").size()
        )
    else:
        aligned_counts = pd.Series(dtype=int)

    all_pids = set()
    if dreem is not None:
        all_pids |= set(dreem.index.get_level_values("participant_id"))
    if diary is not None:
        all_pids |= set(diary.index.get_level_values("participant_id"))
    if ep_ts is not None and "participant_id" in ep_ts.columns:
        all_pids |= set(ep_ts["participant_id"].unique())
    all_pids |= set(rar_participant.index)

    records = []
    excluded = set()
    for pid in sorted(all_pids):
        rar_n = int(rar_counts.get(pid, 0))
        aligned_n = int(aligned_counts.get(pid, 0))
        qual_m = float(qual_counts.get(pid, float("nan")))

        reasons = []
        if rar_n < min_rar_days:
            reasons.append(f"RAR={rar_n}<{min_rar_days}")
        if aligned_n < min_aligned_nights:
            reasons.append(f"Aligned={aligned_n}<{min_aligned_nights}")
        qual_fails = pd.isna(qual_m) or qual_m < min_qual_mean
        if qual_fails and pid not in qual_mean_exceptions:
            reasons.append(
                f"qual_mean={qual_m:.3g}<{min_qual_mean}"
                if not pd.isna(qual_m)
                else "qual_mean=NaN"
            )

        records.append({
            "participant_id": pid,
            "rar_days": rar_n,
            "aligned_nights": aligned_n,
            "qual_mean": round(qual_m, 3) if not pd.isna(qual_m) else None,
            "nights_pruned_for_recovery": pruned_summary.get(pid, 0),
            "manual_exception": pid in qual_mean_exceptions and qual_fails,
            "included": len(reasons) == 0,
            "exclusion_reason": "; ".join(reasons) if reasons else "",
        })
        if reasons:
            excluded.add(pid)

    report = pd.DataFrame(records).set_index("participant_id")
    included = sorted(all_pids - excluded)

    ep_f = ep_ts[ep_ts["participant_id"].isin(included)] if ep_ts is not None else None
    aligned_f = (
        aligned.loc[
            aligned.index.get_level_values("participant_id").isin(included)
        ]
        if aligned is not None
        else None
    )
    diary_f = (
        diary.loc[
            diary.index.get_level_values("participant_id").isin(included)
        ]
        if diary is not None
        else None
    )
    rar_f = (
        rar_participant.loc[rar_participant.index.isin(included)]
        if not rar_participant.empty
        else rar_participant
    )

    assert len(included) + len(excluded) == len(all_pids), (
        f"Accounting mismatch: {len(included)} included + {len(excluded)} "
        f"excluded != {len(all_pids)} total participants"
    )

    return ep_f, aligned_f, diary_f, rar_f, report

# def apply_analytic_sample_filter(ep_ts, dreem, dreem_clean, diary,
#                                  rar_participant, rar_daily,
#                                  min_rar_days=5, min_aligned_nights=5,
#                                  min_qual_mean=0.5, qual_mean_exceptions=None):
#     """
#     Filter participants to the analytic sample based on data completeness.
#
#     min_aligned_nights defaults to 5, matching min_rar_days, rather than
#     the original per-modality value of 3. The two thresholds answer
#     different questions (RAR: is the whole-recording IS estimate
#     trustworthy; aligned: how many usable per-night training rows does
#     this participant contribute) but are set to the same value here
#     deliberately — the aligned-night check already requires valid RAR
#     daily metrics on that date, so a participant's RAR-valid days and
#     their aligned (Dreem+diary+RAR) days should be comparable in number;
#     using 5 for both keeps that comparison meaningful rather than
#     grading the two on different scales.
#
#     Parameters
#     ----------
#     dreem:
#         Raw Dreem summary table (all rows). Used only for the all_pids
#         universe; NOT used for any threshold check.
#     dreem_clean:
#         Dreem summary table restricted to nights passing the internal-
#         consistency filter (SPT vs recording-duration check). This is
#         the anchor for the aligned-night join, matching align.py's
#         Dreem-as-anchor design.
#     diary:
#         Diary table indexed by (participant_id, night_date), with a
#         diary_missing column (see loaders.py / align.py conventions).
#     rar_daily:
#         Output of rar.compute_rar_daily_features() — one row per
#         (participant_id, date) with IV/L5/M10/L5_midpoint/M10_midpoint
#         already restricted to rar.py's own per-day validity gate.
#
#     Returns
#     -------
#     Tuple of (filtered_ep, filtered_aligned_nights, filtered_diary,
#               filtered_rar_participant, exclusion_report)
#     A night in the aligned table now requires: Dreem-clean present,
#     diary present, AND valid RAR daily metrics (IV/L5/M10 non-null) —
#     all three on the same date.
#     """
#     rar_counts = rar_participant["n_days"] if "n_days" in rar_participant.columns \
#         else pd.Series(0, index=rar_participant.index)
#
#     # Participant-level recording-quality gate. Uses dreem_clean's own
#     # recording_quality (qual) column, meaned per participant over their
#     # clean nights only — a bad-duration/implausible-REM night shouldn't
#     # drag down (or prop up) the quality average used to judge the rest.
#     if dreem_clean is not None and "recording_quality" in dreem_clean.columns:
#         qual_counts = dreem_clean.groupby(level="participant_id")["recording_quality"].mean()
#     else:
#         qual_counts = pd.Series(dtype=float)
#     qual_mean_exceptions = set(qual_mean_exceptions or [])
#
#     # --- build the aligned night-level table (mirrors align.py's join,
#     #     then joins on the RAR daily metrics the same way
#     #     merge_rar_into_night_record() does) ---
#     from nesy.ingestion.align import align_modalities, aggregate_embraceplus_daily
#     from nesy.ingestion.rar import merge_rar_into_night_record
#
#     ep_daily = aggregate_embraceplus_daily(ep_ts) if ep_ts is not None else None
#     aligned = align_modalities(dreem_clean, diary, ep_daily) if dreem_clean is not None else None
#
#     if aligned is not None and rar_daily is not None:
#         aligned = merge_rar_into_night_record(aligned, rar_participant, rar_daily)
#         rar_daily_valid = aligned[["IV", "L5", "M10"]].notna().all(axis=1)
#         aligned["fully_aligned"] = (~aligned["diary_missing"]) & rar_daily_valid
#         aligned_counts = (aligned[aligned["fully_aligned"]]
#                           .groupby(level="participant_id").size())
#     else:
#         aligned_counts = pd.Series(dtype=int)
#
#     # FIX 1: union across all modality sources, not just rar_participant.index —
#     # a participant with 0 valid RAR days must still be visible here.
#     all_pids = set()
#     if dreem is not None:
#         all_pids |= set(dreem.index.get_level_values("participant_id"))
#     if diary is not None:
#         all_pids |= set(diary.index.get_level_values("participant_id"))
#     if ep_ts is not None and "participant_id" in ep_ts.columns:
#         all_pids |= set(ep_ts["participant_id"].unique())
#     all_pids |= set(rar_participant.index)
#
#     records = []
#     excluded = set()
#     for pid in sorted(all_pids):
#         rar_n = int(rar_counts.get(pid, 0))
#         aligned_n = int(aligned_counts.get(pid, 0))  # FIX 2: joined, not marginal
#         qual_m = float(qual_counts.get(pid, float("nan")))
#
#         reasons = []
#         if rar_n < min_rar_days:
#             reasons.append(f"RAR={rar_n}<{min_rar_days}")
#         if aligned_n < min_aligned_nights:
#             reasons.append(f"Aligned={aligned_n}<{min_aligned_nights}")
#         qual_fails = pd.isna(qual_m) or qual_m < min_qual_mean
#         if qual_fails and pid not in qual_mean_exceptions:
#             reasons.append(f"qual_mean={qual_m:.3g}<{min_qual_mean}"
#                            if not pd.isna(qual_m) else "qual_mean=NaN")
#
#         records.append({
#             "participant_id": pid,
#             "rar_days": rar_n,
#             "aligned_nights": aligned_n,
#             "qual_mean": round(qual_m, 3) if not pd.isna(qual_m) else None,
#             "manual_exception": pid in qual_mean_exceptions and qual_fails,
#             "included": len(reasons) == 0,
#             "exclusion_reason": "; ".join(reasons) if reasons else "",
#         })
#         if reasons:
#             excluded.add(pid)
#
#     report = pd.DataFrame(records).set_index("participant_id")
#     included = sorted(all_pids - excluded)
#
#     ep_f = ep_ts[ep_ts["participant_id"].isin(included)] if ep_ts is not None else None
#     aligned_f = aligned.loc[aligned.index.get_level_values("participant_id").isin(included)] \
#         if aligned is not None else None
#     diary_f = diary.loc[diary.index.get_level_values("participant_id").isin(included)] \
#         if diary is not None else None
#     rar_f = rar_participant.loc[rar_participant.index.isin(included)] \
#         if not rar_participant.empty else rar_participant
#
#     # sanity check: included + excluded should equal len(all_pids), always
#     assert len(included) + len(excluded) == len(all_pids), (
#         f"Accounting mismatch: {len(included)} included + {len(excluded)} "
#         f"excluded != {len(all_pids)} total participants"
#     )
#
#     return ep_f, aligned_f, diary_f, rar_f, report

# def apply_analytic_sample_filter_old(ep_ts, dreem, diary, rar_participant,
#                                   min_rar_days=5, min_dreem_nights=3,
#                                   min_diary_nights=3):
#     """
#     Filter participants to the analytic sample based on data completeness.
#
#     The binding constraint is RAR valid days — participants with fewer
#     than min_rar_days have unreliable IS estimates and are excluded from
#     the primary analytic sample.
#
#     Returns
#     -------
#     Tuple of (filtered_ep, filtered_dreem, filtered_diary,
#               filtered_rar_participant, exclusion_report)
#     where exclusion_report is a DataFrame summarising why each
#     participant was excluded.
#     """
#     import pandas as pd
#
#     # count valid days per modality
#     rar_counts   = rar_participant["n_days"] if "n_days" in rar_participant.columns \
#                    else pd.Series(0, index=rar_participant.index)
#     dreem_counts = dreem.groupby(level="participant_id").size() \
#                    if dreem is not None else pd.Series(dtype=int)
#     diary_counts = diary.groupby(level="participant_id").size() \
#                    if diary is not None else pd.Series(dtype=int)
#
#     all_pids = set(rar_participant.index)
#
#     records = []
#     excluded = set()
#     for pid in all_pids:
#         rar_n   = int(rar_counts.get(pid, 0))
#         dreem_n = int(dreem_counts.get(pid, 0))
#         diary_n = int(diary_counts.get(pid, 0))
#
#         reasons = []
#         if rar_n   < min_rar_days:      reasons.append(f"RAR={rar_n}<{min_rar_days}")
#         if dreem_n < min_dreem_nights:  reasons.append(f"Dreem={dreem_n}<{min_dreem_nights}")
#         if diary_n < min_diary_nights:  reasons.append(f"Diary={diary_n}<{min_diary_nights}")
#
#         records.append({
#             "participant_id": pid,
#             "rar_days":       rar_n,
#             "dreem_nights":   dreem_n,
#             "diary_nights":   diary_n,
#             "included":       len(reasons) == 0,
#             "exclusion_reason": "; ".join(reasons) if reasons else "",
#         })
#         if reasons:
#             excluded.add(pid)
#
#     report = pd.DataFrame(records).set_index("participant_id")
#     included = sorted(all_pids - excluded)
#
#     # filter each modality
#     ep_f    = ep_ts[ep_ts["participant_id"].isin(included)]
#     dreem_f = dreem.loc[dreem.index.get_level_values("participant_id").isin(included)] \
#               if dreem is not None else None
#     diary_f = diary.loc[diary.index.get_level_values("participant_id").isin(included)] \
#               if diary is not None else None
#     rar_f   = rar_participant.loc[rar_participant.index.isin(included)]
#
#     return ep_f, dreem_f, diary_f, rar_f, report

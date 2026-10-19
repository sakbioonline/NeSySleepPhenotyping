"""
REST-activity rhythm (RAR) feature extraction.

Computes seven nonparametric circadian metrics from the EmbracePlus
activity_counts time series. All metrics follow the 24-hour computation
window convention from Van Someren et al. (1999).

IS  — interdaily stability. Requires the full multi-day recording by
      definition. One value per participant.

RA  — relative amplitude (M10-L5)/(M10+L5). Empirically stable within
      participants in EmbracePlus data (CV ~4% in SSP, N=184).
      Stored as participant mean.

IV, L5, M10, L5_midpoint, M10_midpoint — computed per 24-hour day,
      then stored as both daily values and participant means.
      These show genuine within-participant daily variation in
      EmbracePlus micro-activity data (CV: IV ~20%, L5 ~44%,
      L5_midpoint std ~1.07h in SSP). Unlike step-count devices
      (Fitbit), micro-activity counts detect nocturnal restlessness
      and small movements that drive real daily variation.

      NHANES and UK Biobank average these to participant-level because
      their research question is between-person (cross-sectional
      epidemiology), not because daily variation is absent.

Outputs
-------
compute_rar_daily_features(ep_ts)
    One row per (participant, date). IS broadcast from participant level.

compute_rar_participant_features(ep_ts)
    One row per participant. Means of daily values. IS as single value.

merge_rar_into_night_record(nr, rar_participant, rar_daily)
    Adds daily IV, L5, L5_midpoint, M10, M10_midpoint to the night
    record. IS and RA_mean are NOT added — they live in rar_participant
    and are used as knowledge priors by the symbolic rules.

Non-wear imputation
-------------------
Applied before RAR computation. Tiered strategy:
    missing < 5h AND mostly daytime → hourly-mean fill
    otherwise                        → forward-fill + flag day

Zero-fill is not used — it inflates IV and distorts L5/L5_midpoint.

References
----------
Van Someren EJW et al. (1999). IEEE Trans Rehab Eng, 7(4):455-458.
Blume C et al. (2016). MethodsX, 3:100-107.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

_NIGHT_START = 22
_NIGHT_END = 6
_MAX_MISSING = 300  # 5 hours
_MAX_NOCT = 60  # 1 hour nocturnal


@dataclass
class ImputationReport:
    participant_id: str
    total_missing_minutes: int
    days_hourly_filled: list = field(default_factory=list)
    days_forward_filled: list = field(default_factory=list)
    days_flagged_high_missing: list = field(default_factory=list)

    @property
    def flagged_day_count(self):
        return len(self.days_flagged_high_missing)

    def __repr__(self):
        return (f"ImputationReport(pid={self.participant_id!r}, "
                f"missing={self.total_missing_minutes}min, "
                f"hourly={len(self.days_hourly_filled)}d, "
                f"fwd={len(self.days_forward_filled)}d, "
                f"flagged={self.flagged_day_count}d)")


def impute_non_wear(ep_ts, max_missing=_MAX_MISSING, max_noct=_MAX_NOCT):
    parts = []
    reports = []
    for pid, group in ep_ts.groupby("participant_id"):
        imputed, report = _impute_participant(group.copy(), pid, max_missing, max_noct)
        parts.append(imputed)
        reports.append(report)
    if not parts:
        return ep_ts.copy(), reports
    return pd.concat(parts).sort_index(), reports


def _impute_participant(group, pid, max_missing, max_noct):
    group = group.sort_index()
    report = ImputationReport(pid, int(group["activity_counts"].isna().sum()))
    if report.total_missing_minutes == 0:
        return group, report

    hourly_means = _hourly_means(group)
    dates = group.index.date.astype(str)
    act = group["activity_counts"].copy()

    for date in pd.unique(dates):
        mask_d = dates == date
        day_act = act[mask_d]
        missing = day_act.isna()
        n_miss = missing.sum()
        if n_miss == 0:
            continue

        day_idx = group.index[mask_d]
        n_noct = (missing & _is_nocturnal(day_idx)).sum()

        if n_miss < max_missing and n_noct < max_noct:
            fill_vals = pd.Series(
                [float(hourly_means.get(ts.hour, 0.0)) for ts in day_idx],
                index=day_idx
            )
            act[mask_d] = day_act.fillna(fill_vals)
            report.days_hourly_filled.append(date)
        else:
            filled = day_act.ffill().bfill()
            if filled.isna().any():
                fill_vals = pd.Series(
                    [float(hourly_means.get(ts.hour, 0.0)) for ts in day_idx],
                    index=day_idx
                )
                filled = filled.fillna(fill_vals)
            act[mask_d] = filled
            report.days_forward_filled.append(date)
            if n_miss >= max_missing:
                report.days_flagged_high_missing.append(date)

    group = group.copy()
    group["activity_counts"] = act
    return group, report


def _hourly_means(group):
    valid = group["activity_counts"].dropna()
    if valid.empty:
        return pd.Series(0.0, index=range(24))
    by_hour = valid.groupby(valid.index.hour).mean()
    return by_hour.reindex(range(24), fill_value=valid.mean())


def _is_nocturnal(index):
    h = index.hour
    return pd.Series((h >= _NIGHT_START) | (h < _NIGHT_END), index=index)


def imputation_reports_to_dataframe(reports):
    return pd.DataFrame([{
        "participant_id": r.participant_id,
        "total_missing_minutes": r.total_missing_minutes,
        "days_hourly_filled": len(r.days_hourly_filled),
        "days_forward_filled": len(r.days_forward_filled),
        "days_flagged_high_missing": r.flagged_day_count,
    } for r in reports]).set_index("participant_id")


# Validity thresholds — match reference implementation (ssp_npcra_all_v3.csv)
_MIN_VALID_TOTAL_MIN = 300  # max missing minutes per day
_MIN_VALID_SPT_MIN = 180  # max missing minutes in sleep-labelled epochs
_SPT_CODE = 101.0  # EmbracePlus sleep detection stage code
_MIN_DAY_MINUTES = 1440  # require full 24h day


def _valid_dates(df):
    """
    Return dates passing the quality gate used in the reference implementation:
      total_missing < 300 min  AND  spt_missing <= 180 min.
    Days failing this are excluded from IS and daily metric computation.
    """
    valid = []
    df["_date"] = df.index.date
    for day, g in df.groupby("_date"):
        total_miss = g["activity_counts"].isna().sum()
        spt_miss = g.loc[g["sleep_stage_ep"] == _SPT_CODE, "activity_counts"].isna().sum()
        if total_miss < _MIN_VALID_TOTAL_MIN and spt_miss <= _MIN_VALID_SPT_MIN:
            valid.append(day)
    return valid


def _is_from_valid(df, valid_dates):
    """
    IS computed across valid days only.
    Formula matches reference: n * sum(h_mean - mean)^2 / (24 * sum(xi - mean)^2)
    """
    valid_set = set(str(d) for d in valid_dates)
    v = df[[str(d) in valid_set for d in df.index.date]]["activity_counts"]
    if v.empty or v.std() == 0:
        return np.nan
    overall_mean = v.mean()
    h_mean = v.groupby(v.index.hour).mean()
    n = len(v)
    num = n * float(((h_mean - overall_mean) ** 2).sum())
    den = 24 * float(((v - overall_mean) ** 2).sum())
    return float(num / den) if den != 0 else np.nan


def _iv(activity):
    # matches reference: iv_num/iv_den where
    # iv_num = sum(diff^2) / (N-1)
    # iv_den = sum((xi - mean)^2) / N
    vals = activity.values
    if len(vals) < 2:
        return np.nan
    mean_act = np.mean(vals)
    iv_num = np.sum(np.diff(vals) ** 2) / (len(vals) - 1)
    iv_den = np.sum((vals - mean_act) ** 2) / len(vals)
    return float(iv_num / iv_den) if iv_den != 0 else np.nan


def _l5_m10(activity, l5h=5, m10h=10):
    """
    Matches reference implementation exactly:
    - 1-minute sliding window rolling mean
    - requires 1440 minutes (full day)
    - midpoints: l5_start + 2.5h, m10_start + 5h
    """
    nan = {"L5": np.nan, "M10": np.nan,
           "L5_midpoint": np.nan, "M10_midpoint": np.nan, "RA": np.nan}

    if len(activity) < _MIN_DAY_MINUTES:
        return nan

    vals = activity.values
    act_s = pd.Series(vals)
    w_l5 = l5h * 60
    w_m10 = m10h * 60

    m10_roll = act_s.rolling(window=w_m10).mean()
    l5_roll = act_s.rolling(window=w_l5).mean()

    M10 = m10_roll.max()
    L5 = l5_roll.min()

    m10_end_idx = m10_roll.idxmax()
    l5_end_idx = l5_roll.idxmin()

    if pd.isna(m10_end_idx) or pd.isna(l5_end_idx):
        return nan

    m10_start_idx = int(m10_end_idx) - (w_m10 - 1)
    l5_start_idx = int(l5_end_idx) - (w_l5 - 1)

    if m10_start_idx < 0 or l5_start_idx < 0:
        return nan

    # midpoints match reference: l5_start + 2.5h, m10_start + 5h
    ts = activity.index
    l5_mid = ts[l5_start_idx] + pd.Timedelta(hours=2.5)
    m10_mid = ts[m10_start_idx] + pd.Timedelta(hours=5.0)
    l5_h = float(l5_mid.hour + l5_mid.minute / 60)
    m10_h = float(m10_mid.hour + m10_mid.minute / 60)

    denom = float(M10) + float(L5)
    ra = float((M10 - L5) / denom) if denom > 0 else np.nan

    return {"L5": round(float(L5), 4), "M10": round(float(M10), 4),
            "L5_midpoint": round(l5_h, 4), "M10_midpoint": round(m10_h, 4),
            "RA": round(ra, 4)}


def compute_rar_daily_features(ep_ts, impute=True):
    """
    One row per (participant, valid date).

    Matches reference implementation (ssp_npcra_all_v3.csv):
    - Days with total_missing >= 300 min OR spt_missing > 180 min excluded
    - IS computed across valid days only
    - 1440-minute minimum per day for L5/M10
    - Imputation profile built from valid days only
    """
    df = ep_ts.copy()

    records = []
    for pid, pg in df.groupby("participant_id"):
        pg = pg.sort_index()

        # get valid dates before imputation (gate uses raw missingness)
        valid = _valid_dates(pg)
        if not valid:
            continue

        # impute using hourly profile from valid days only — matches reference
        if impute:
            valid_set = set(str(v) for v in valid)
            is_valid_day = pd.Series(
                [str(d) in valid_set for d in pg.index.date], index=pg.index
            )
            v_data = pg[is_valid_day]
            hourly_prof = v_data["activity_counts"].groupby(v_data.index.hour).mean()
            fill_vals = pd.Series(
                pg.index.hour.map(hourly_prof), index=pg.index
            )
            pg = pg.copy()
            pg["activity_counts"] = (pg["activity_counts"]
                                     .fillna(fill_vals)
                                     .ffill().bfill().fillna(0))
        else:
            pg["activity_counts"] = pg["activity_counts"].fillna(0.0)

        # IS across all valid days
        is_val = _is_from_valid(pg, valid)

        pg["_date"] = pg.index.date
        for date in valid:
            dg = pg[pg["_date"] == date].sort_index()
            act = dg["activity_counts"]
            r = _l5_m10(act)
            if any(v is np.nan for v in [r["L5"], r["M10"]]):
                continue  # skip days < 1440 minutes
            records.append({
                "participant_id": pid,
                "date": str(date),
                "IS": round(is_val, 4) if not pd.isna(is_val) else np.nan,
                "IV": round(_iv(act), 4),
                "L5": r["L5"],
                "M10": r["M10"],
                "L5_midpoint": r["L5_midpoint"],
                "M10_midpoint": r["M10_midpoint"],
                "RA": r["RA"],
            })

    if not records:
        return pd.DataFrame(columns=["participant_id", "date", "IS", "IV", "L5", "M10",
                                     "L5_midpoint", "M10_midpoint", "RA"]).set_index(["participant_id", "date"])
    return pd.DataFrame(records).set_index(["participant_id", "date"]).sort_index()


def compute_rar_participant_features(ep_ts, impute=True):
    """One row per participant. Means of daily values; IS as single value."""
    daily = compute_rar_daily_features(ep_ts, impute=impute)
    return daily.groupby("participant_id").agg(
        IS=("IS", "first"),
        IV_mean=("IV", "mean"),
        L5_mean=("L5", "mean"),
        M10_mean=("M10", "mean"),
        L5_midpoint_mean=("L5_midpoint", "mean"),
        M10_midpoint_mean=("M10_midpoint", "mean"),
        RA_mean=("RA", "mean"),
        n_days=("IV", "count"),
    )


def merge_rar_into_night_record(night_record, rar_participant, rar_daily):
    """
    Add daily RAR features to the night record.
    IS and RA are NOT added — they are knowledge priors, not nightly features.
    """
    result = night_record.copy()
    daily_cols = ["IS","IV", "L5", "L5_midpoint", "M10", "M10_midpoint"]
    djoin = rar_daily[daily_cols].copy()
    djoin.index.names = ["participant_id", "night_date"]
    return result.join(djoin, how="left")


# legacy aliases
def compute_rar_features(ep_ts, impute=True):
    return compute_rar_participant_features(ep_ts, impute=impute)


def compute_rar_features_per_night(ep_ts, impute=True):
    return compute_rar_daily_features(ep_ts, impute=impute)

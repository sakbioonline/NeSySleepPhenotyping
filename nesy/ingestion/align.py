"""
Joins the three modalities into a single night-level record.

Dreem is the anchor. If there is no Dreem recording for a night,
that night is excluded — without sleep architecture data most
labelling rules cannot fire anyway.

Diary is left-joined: missing nights get diary_missing=True and
NaN subjective columns, so R1 abstains rather than guessing.

EmbracePlus daily aggregates are left-joined for the nightly
physiology summaries. RAR metrics are handled separately in rar.py.
"""

import pandas as pd
from nesy.ingestion.loaders import NIGHT_RECORD_REQUIRED_COLS


def align_modalities(dreem, diary, ep_daily):
    #print(f"Dreem first: ",dreem.head(5))
    #print(f"Diary first: ",diary.head(5))
    #print(f"EP first: ",ep_daily.head(5))
    #print(ep_daily[~ep_daily.isna().any(axis=1)])
    #print("Sum of missing rows",ep_daily.isna().any(axis=1).sum())
    ep_daily = ep_daily[~ep_daily.isna().any(axis=1)]
    nr = dreem.copy()
    nr = nr.join(diary, how="left", rsuffix="_diary")
    nr["diary_missing"] = nr["diary_missing"].isna()
    nr = nr.join(ep_daily, how="left", rsuffix="_ep")
    return nr.sort_index()


def compute_n3_motion_feature(aligned_ep_df):
    """Computes mean EmbracePlus activity strictly during Dreem N3 epochs."""
    if "eeg_stage" not in aligned_ep_df.columns:
        return 0.0
    n3_mask = aligned_ep_df["eeg_stage"] == 3
    if not n3_mask.any():
        return 0.0
    return float(aligned_ep_df.loc[n3_mask, "activity_counts"].fillna(0).mean())


def aggregate_embraceplus_daily(ep_ts):
    df = ep_ts.copy()
    df["night_date"] = df.index.date.astype(str)
    daily = df.groupby(["participant_id", "night_date"]).agg(
        ep_activity_mean=("activity_counts", "mean"),
        ep_activity_sum=("activity_counts", "sum"),
        ep_pulse_mean=("pulse_rate_bpm", "mean"),
        ep_prv_mean=("prv_rmssd_ms", "mean"),
        ep_resp_mean=("respiratory_rate_brpm", "mean"),
        ep_minutes_worn=("activity_counts", "count"),  # non-NaN = worn
    )

    # 2. Compute N3-specific wrist motion (STRICTLY during Dreem N3 epochs)
    if "eeg_stage" in df.columns:
        n3_motion = (
            df[df["eeg_stage"] == 3]
            .groupby(["participant_id", "night_date"])["activity_counts"]
            .mean()
            .rename("ep_n3_activity_mean")
        )
        daily = daily.join(n3_motion, how="left")
        daily["ep_n3_activity_mean"] = daily["ep_n3_activity_mean"].fillna(0.0)
    else:
        daily["ep_n3_activity_mean"] = 0.0

    return daily


def validate_night_record(df):
    missing = [c for c in NIGHT_RECORD_REQUIRED_COLS
               if c not in df.columns and c not in df.index.names]
    if missing:
        raise ValueError(f"Night record missing columns: {missing}")


def alignment_reports_to_dataframe(reports):
    if not reports:
        return pd.DataFrame()
    records = [{
        "participant_id": r.participant_id,
        "night_date": r.night_date,
        "device": r.device,
        "n_source_epochs": r.n_source_epochs,
        "n_grid_slots": r.n_grid_slots,
        "n_assigned": r.n_assigned,
        "n_forward_filled": r.n_forward_filled,
        "n_unscored": r.n_unscored,
        "coverage_pct": r.coverage_pct,
    } for r in reports]
    return pd.DataFrame(records).set_index(["participant_id", "night_date"]).sort_index()


def night_coverage_report(df):
    total = len(df)
    diary_present = int((~df["diary_missing"]).sum())
    ep_present = int(df["ep_minutes_worn"].notna().sum()) if "ep_minutes_worn" in df.columns else None
    by_participant = (
        df.groupby(level="participant_id")
        .agg(nights=("SE_pct", "count"),
             diary_nights=("diary_missing", lambda x: (~x).sum()))
        .to_dict(orient="index")
    )
    return {
        "total_nights": total,
        "diary_present": diary_present,
        "diary_missing": total - diary_present,
        "ep_present": ep_present,
        "by_participant": by_participant,
    }

"""
nesy/reporting/final_table.py

Single point of convergence for the final analytic-sample data: one
night-level table (participant_id, family_id, date, demography, EEG/Dreem,
EmbracePlus, diary) and its participant-level aggregate.

Built from the already-filtered night_record (Layer 1 output, indexed
(participant_id, night_date), containing the Dreem+diary+EmbracePlus+RAR
columns from align_modalities/merge_rar_into_night_record) plus
demographics (load_demographics output, indexed participant_id).
"""

import pandas as pd

# column groupings, in display order — only columns actually present
# in the merged frame are kept, so this is safe even if a modality's
# columns are partially missing (e.g. no clinical/RAR data available)
_DEMO_COLS = ["family_id", "age", "age_band", "sex_encoded", "autism_status"]
_EEG_COLS = [
    "SE_pct", "WASO_min", "SOL_min", "TST_min",
    "N1_min", "N2_min", "N3_min", "REM_min",
    "N1_pct", "N2_pct", "N3_pct", "REM_pct",
    "recording_quality",
]
_EP_COLS = [
    "ep_activity_mean", "ep_activity_sum", "ep_pulse_mean",
    "ep_prv_mean", "ep_resp_mean", "ep_minutes_worn",
    "IV", "L5", "L5_midpoint", "M10", "M10_midpoint",
]
_DIARY_COLS = [
    "ANX", "DIFF", "SQ", "num_awakenings",
    "how_energetic", "feeling_physically", "mood", "diary_missing",
]


def build_final_night_table(night_record, demo):
    """
    Consolidate the aligned, filtered night record with demographics
    into a single table: participant_id, family_id, night_date,
    [demography][EEG][EmbracePlus][diary] — one row per analytic night.

    Parameters
    ----------
    night_record : DataFrame indexed (participant_id, night_date), the
        Layer 1 output after apply_analytic_sample_filter has restricted
        it to the analytic sample.
    demo : DataFrame indexed participant_id (load_demographics output),
        columns: age, age_band, sex_encoded, autism_status, family_id.

    Returns
    -------
    DataFrame, one row per (participant_id, night_date), sorted by
    participant_id then night_date, columns grouped and ordered
    demography -> EEG -> EmbracePlus -> diary -> anything else.
    """
    nr = night_record.reset_index()
    demo_flat = demo.reset_index().rename(columns={"index": "participant_id"})

    merged = nr.merge(demo_flat, on="participant_id", how="left")

    ordered = (
        ["participant_id", "night_date"]
        + [c for c in _DEMO_COLS if c in merged.columns]
        + [c for c in _EEG_COLS if c in merged.columns]
        + [c for c in _EP_COLS if c in merged.columns]
        + [c for c in _DIARY_COLS if c in merged.columns]
    )
    # family_id sits right after participant_id/night_date per the
    # requested layout, not buried in the demography block
    if "family_id" in ordered:
        ordered.remove("family_id")
        ordered.insert(2, "family_id")

    remaining = [c for c in merged.columns if c not in ordered]
    merged = merged[ordered + remaining]

    return merged.sort_values(["participant_id", "night_date"]).reset_index(drop=True)


def build_final_participant_aggregate(final_night_table):
    """
    One row per participant: n_nights, demography (constant per
    participant, carried through unchanged), and the mean of every
    numeric EEG/EmbracePlus/diary metric across that participant's
    analytic nights.

    Parameters
    ----------
    final_night_table : output of build_final_night_table.

    Returns
    -------
    DataFrame, one row per participant_id.
    """
    demo_present = [c for c in _DEMO_COLS if c in final_night_table.columns]
    demo_const = final_night_table.groupby("participant_id")[demo_present].first()

    numeric_cols = [c for c in final_night_table.select_dtypes(include="number").columns
                    if c not in demo_present]

    agg = final_night_table.groupby("participant_id")[numeric_cols].mean()
    agg.columns = [f"{c}_mean" for c in agg.columns]
    n_nights = final_night_table.groupby("participant_id").size().rename("n_nights")
    out = pd.concat([demo_const, n_nights, agg], axis=1).reset_index()
    return out

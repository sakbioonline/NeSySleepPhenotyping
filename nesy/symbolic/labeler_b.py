"""
Replaces profile_cluster_phenotypes() from test_data_v3a.py.

The function it replaces named each cluster by scoring its nights against
cf.normative_mask(), cf.conditioned_arousal_mask(), etc. — the same C1-C7
thresholds that also drive the training-time constraint. A disagreement
between that label and the constraint during V3a training would reflect
one rule system disagreeing with itself, not two independent sources of
evidence — exactly the circularity the knowledge-in-constraints pathway
is designed to avoid (see Chapter 4, Section 4.1.4).

This module names each cluster instead by distance to three published,
independently-derived subtype descriptions (Cohen et al. 2017, Goldman
et al. 2009, Lenker et al. 2025 — see Table 4.1). No function here reads
cf, C1-C7, or any ontology-derived threshold. The literature-native vote
is then translated to an ontology phenotype name as a separate, explicit
step (see LITERATURE_LABEL_TO_PHENOTYPE), not computed from ontology
thresholds directly.
"""

import numpy as np
import pandas as pd

THRESH = 0.3  # minimum |z| for a labelling-function criterion to "count"


def _zscore_clusters(cluster_profile_df, metrics):
    """z-score each metric across clusters, so 'elevated' is judged
    relative to this cohort's own cluster spread, not a fixed number."""
    z = cluster_profile_df[metrics].apply(lambda c: (c - c.mean()) / c.std())
    z.columns = ["SOL", "WASO", "SE", "TST", "IV"][:len(metrics)]
    return z


def _get_iv(row):
    """IV may be absent (e.g. the RAR-ablated feature set). Return 0.0
    in that case, a neutral z-score that never satisfies an IV-based
    elevated/reduced check, rather than crashing — the caller is
    responsible for having already warned that IV-based criteria are
    inactive for this run."""
    return row["IV"] if "IV" in row else 0.0


def cohen_lf(row):
    """Cohen et al. 2017: unstable = reduced duration + elevated
    instability (IV proxy); stable = opposite."""
    iv = _get_iv(row)
    tst_low, tst_high = row["TST"] < -THRESH, row["TST"] > THRESH
    iv_high, iv_low = iv > THRESH, iv < -THRESH
    if tst_low and iv_high:
        return "unstable"
    if tst_high and iv_low:
        return "stable"
    return "ABSTAIN"


def goldman_lf(row):
    """Goldman et al. 2009: poor = high SOL + low SE (+ optionally high
    IV as fragmentation); good = opposite. Requires >=2 of 3 signals
    to agree with no signal pointing the other way."""
    iv = _get_iv(row)
    signals = []
    if row["SOL"] > THRESH: signals.append("poor")
    if row["SOL"] < -THRESH: signals.append("good")
    if row["SE"] < -THRESH: signals.append("poor")
    if row["SE"] > THRESH: signals.append("good")
    if iv > THRESH: signals.append("poor")
    if iv < -THRESH: signals.append("good")
    if signals.count("poor") >= 2 and signals.count("good") == 0:
        return "poor"
    if signals.count("good") >= 2 and signals.count("poor") == 0:
        return "good"
    return "ABSTAIN"


def lenker_lf(row):
    """Lenker et al. 2025: disturbed_sleep = elevated SOL or WASO with
    IV not clearly low; insufficient_sleep = low TST with SOL/WASO NOT
    elevated. Mutually exclusive by construction (see the corrected
    exclusion condition on SOL/WASO for insufficient_sleep)."""
    iv = _get_iv(row)
    disturbed = (row["SOL"] > THRESH or row["WASO"] > THRESH) and iv > -THRESH
    insufficient = row["TST"] < -THRESH and row["SOL"] < THRESH and row["WASO"] < THRESH
    if disturbed and insufficient:
        return "ABSTAIN"
    if disturbed:
        return "disturbed_sleep"
    if insufficient:
        return "insufficient_sleep"
    return "ABSTAIN"


# Literature-native vote -> ontology phenotype name. Only votes that
# converge on one of these two outcomes get a phenotype name at all —
# everything else (insufficient_sleep, or no votes at all) stays
# unmapped, per the confirmed 4.2.2 boundary: this literature does not
# characterise circadian phase or subjective-objective discordance, so
# RAR_misalignment, Coupling_impairment, and Paradoxical_insomnia are
# not reachable through this evidence base at any cluster resolution.
LIT_LABEL_TO_PHENOTYPE = {
    "unstable": "Conditioned_arousal",
    "poor": "Conditioned_arousal",
    "disturbed_sleep": "Conditioned_arousal",
    "stable": "Normative_range",
    "good": "Normative_range",
}


def name_clusters(cluster_profile_df, cluster_col="weak_cluster",
                   metrics=("SOL_min_mean", "WASO_min_mean", "SE_pct_mean",
                            "TST_min_mean")):
    """
    Parameters
    ----------
    cluster_profile_df : DataFrame, one row per cluster, with cluster_col
        and the metric columns (matches v3a_cluster_profiles.csv).
    cluster_col : name of the cluster id column.
    metrics : the 5 metrics the three labelling functions read, in order
        [SOL, WASO, SE, TST, IV]. If IV isn't in the profile (e.g. the
        RAR-ablated feature set), pass only the 4 available metrics —
        cohen_lf and lenker_lf's IV checks will then always evaluate
        False, which the caller should be aware weakens those two
        functions specifically, not silently.

    Returns
    -------
    vote_table : DataFrame, one row per cluster, with each function's
        vote and the resulting phenotype mapping (or None if unmapped).
    cluster_to_phenotype : dict, cluster_id -> phenotype name string,
        containing ONLY clusters that received a mapping. Clusters not
        in this dict are unmapped and must be excluded from Y_weak,
        not silently defaulted to a phenotype.
    """
    print("inside name_cluster",cluster_profile_df.columns.tolist())
    df = cluster_profile_df.set_index(cluster_col)
    has_iv = "IV_mean" in metrics and "IV_mean" in df.columns
    if not has_iv:
        print("WARNING: IV not in the provided metrics — cohen_lf and "
              "lenker_lf's IV-based criteria will never fire for this "
              "run. This is expected for the RAR-ablated feature set, "
              "but the resulting votes are weaker than the full-metric "
              "run and should be reported as such.")

    z = _zscore_clusters(df, list(metrics))

    rows = []
    for cid, row in z.iterrows():
        cohen_v = cohen_lf(row)
        goldman_v = goldman_lf(row)
        lenker_v = lenker_lf(row)
        votes = [v for v in [cohen_v, goldman_v, lenker_v] if v != "ABSTAIN"]

        mapped = [LIT_LABEL_TO_PHENOTYPE.get(v) for v in votes]
        mapped = [m for m in mapped if m is not None]

        if not mapped:
            phenotype = None
        elif len(set(mapped)) == 1:
            phenotype = mapped[0]
        else:
            # votes disagree on WHICH ontology phenotype after mapping —
            phenotype = "CONFLICT"

        rows.append({
            "cluster": cid, "cohen_vote": cohen_v, "goldman_vote": goldman_v,
            "lenker_vote": lenker_v, "n_votes": len(votes), "phenotype": phenotype,
        })

    vote_table = pd.DataFrame(rows)
    cluster_to_phenotype = {
        r["cluster"]: r["phenotype"] for _, r in vote_table.iterrows()
        if r["phenotype"] not in (None, "CONFLICT")
    }
    return vote_table, cluster_to_phenotype
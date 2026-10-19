"""
Sleep phenotyping pipeline: ingestion, knowledge base, weak supervisory
scaffold, constraint-guided neurosymbolic training, stratification.

Pipeline stages
---------------
  Layer 1  Ingestion      EmbracePlus actigraphy, Dreem EEG, sleep diary,
                          RAR metrics, non-wear imputation, epoch alignment,
                          analytic sample filter (configs/pipeline.yaml)
  Layer 2  Knowledge      Knowledge base, medication parsing, GBTM trajectory
                          grouping (used as a constraint modifier only)
  Layer 3  Weak scaffold  2-cluster k-means on SOL, WASO, SE and TST
                          (participant means), mapped to phenotype classes
                          and propagated to each night as the weak label
  Layer 4  Fusion         ROCKET (actigraphy) + symbolic hypnogram features,
                          linear softmax model trained with the per-rule
                          Lagrangian constraint objective and C_smooth;
                          leave-one-participant-out cross-validation
  Post     Stratification Night predictions to participant-level phenotype

Outputs (runs/TIMESTAMP/)
    pipeline_report.json, final_night_table.csv, final_participant_table.csv,
    night_labels.csv, weak_cluster_profiles.csv, weak_participant_clusters.csv,
    lopo_per_night_predictions.csv, y_patient.csv, y_patient_L4.csv,
    gbtm_groups.csv, iv_calibration_values.npy

Usage
    python test_pipeline.py --data data/ssp
    python test_pipeline.py --data data/ssp --participant SP0000000
    python test_pipeline.py --data data/ssp --n-groups 3
    python test_pipeline.py --data data/ssp --select-k
    python test_pipeline.py --data data/ssp --with-clinical
    python test_pipeline.py --data data/ssp --audit-triggers
    python test_pipeline.py --data data/ssp --report

Arguments
    --data           cohort data root (required). Expected structure:
                     embraceplus/*.csv, dreem/*.csv (+ hypnogram files),
                     diary/*.csv, demographics/*.csv
    --kb             knowledge_base directory (default: knowledge_base)
    --configs        configs directory (default: configs)
    --participant    run on a single participant ID
    --n-groups       number of GBTM trajectory groups (default: 4)
    --select-k       select the GBTM group count by BIC (k = 2..5)
    --with-clinical  add clinical features to the fusion model inputs
    --use-soft-probs soft probability aggregation in stratification
    --dominance-threshold, --instability-threshold, --margin-buffer
                     stratification thresholds (defaults 0.40, 2.0, 0.05)
    --disable-rules  comma-separated rule names to leave out of training
    --no-smooth      disable C_smooth; --smooth-alpha-scale sets its ascent rate
    --audit-triggers record constraint trigger rates during training
    --report         write one PDF phenotype report per participant
    --cohort         ssp (EmbracePlus + Dreem) or 22q (Fitbit + Click EEG)
"""

import argparse
import datetime
import json
import re
import statistics
import sys
import traceback
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# ensure the nesy package is importable regardless of working directory
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

warnings.filterwarnings("ignore")

CLUSTER_FEATURES = ["SOL_min_mean", "WASO_min_mean", "SE_pct_mean", "TST_min_mean"]

PHENOTYPE_TO_CLASS = {
    "Normative_range": 0,
    "Paradoxical_insomnia": 1,
    "Conditioned_arousal": 2,
    "Hyper-somnolence": 3,
    "RAR_misalignment": 4,
    "Coupling_impairment": 5,
}


def parse_args():
    p = argparse.ArgumentParser(description="Sleep phenotyping pipeline")
    p.add_argument("--data", required=True)
    p.add_argument("--kb", default="knowledge_base")
    p.add_argument("--configs", default="configs")
    p.add_argument("--participant", default=None)
    p.add_argument("--n-groups", type=int, default=4,
                   help="number of GBTM groups (default=4)")
    p.add_argument("--select-k", action="store_true",
                   help="select the GBTM group count by BIC before fitting")
    p.add_argument("--with-clinical", action="store_true",
                   help="add clinical features to the fusion model inputs")
    p.add_argument("--use-soft-probs", action="store_true", default=False,
                   help="soft continuous probability aggregation in stratification")
    p.add_argument("--dominance-threshold", type=float, default=0.40,
                   help="minimum share of nights for a dominant phenotype (default=0.40)")
    p.add_argument("--instability-threshold", type=float, default=2.0,
                   help="entropy (bits) above which a participant is unstable (default=2.0)")
    p.add_argument("--margin-buffer", type=float, default=0.05,
                   help="top-two margin below which a participant is a boundary case (default=0.05)")
    p.add_argument("--no-smooth", action="store_true",
                   help="disable the C_smooth temporal smoothness constraint")
    p.add_argument("--smooth-alpha-scale", type=float, default=1.0,
                   help="multiplier on the ascent rate of the C_smooth multiplier (default=1.0)")
    p.add_argument("--disable-rules", default="",
                   help="comma-separated rule names to exclude from training "
                        "(e.g. r2_rar_misalignment,r9_medication_context)")
    p.add_argument("--audit-triggers", action="store_true",
                   help="record per-rule constraint trigger rates during training")
    p.add_argument("--report", action="store_true",
                   help="generate one PDF phenotype report per participant")
    p.add_argument("--cohort", type=str, default="ssp", choices=["ssp", "22q"],
                   help="ssp (EmbracePlus + Dreem) or 22q (Fitbit + Click EEG)")
    return p.parse_args()


def sep(title=""):
    print(f"\n{'─' * 55}")
    if title:
        print(title)


# ── Weak supervisory scaffold ─────────────────────────────────────────────────

def generate_weak_clusters(final_participant_table, run_dir):
    """2-cluster k-means on participant means of SOL, WASO, SE and TST."""
    participant_df = final_participant_table.copy()
    X = participant_df[CLUSTER_FEATURES].copy()
    X = X.fillna(X.median())
    X = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=2, random_state=42, n_init=40)
    participant_df["weak_cluster"] = km.fit_predict(X)
    labels = participant_df["weak_cluster"]

    cluster_profile = (
        participant_df.groupby("weak_cluster")[CLUSTER_FEATURES]
        .mean().round(2).reset_index()
    )

    print("\nCluster profiles")
    print(cluster_profile)
    print(f"inertia: {km.inertia_}")
    print(f"silhouette: {silhouette_score(X, labels)}")
    print(f"calinski_harabasz: {calinski_harabasz_score(X, labels)}")

    cluster_profile.to_csv(run_dir / "weak_cluster_profiles.csv")
    return participant_df, cluster_profile


def add_hard_label(final_table, mapping):
    """
    Add an integer hard_label column from weak_cluster.

    mapping : {weak_cluster: phenotype name} as returned by name_clusters().
    Returns the table with hard_label and the rows left unmapped.
    """
    result = final_table.copy()

    cluster_to_class = {
        cluster_id: PHENOTYPE_TO_CLASS[name]
        for cluster_id, name in mapping.items()
        if name in PHENOTYPE_TO_CLASS
    }
    bad_names = {p for p in mapping.values() if p not in PHENOTYPE_TO_CLASS}
    if bad_names:
        print(f"WARNING: phenotype name(s) not in PHENOTYPE_TO_CLASS, these "
              f"clusters will be unmapped: {bad_names}")

    result["hard_label"] = result["weak_cluster"].map(cluster_to_class)
    result["hard_label"] = result["hard_label"].astype("Int64")

    unmapped = result[result["hard_label"].isna()]
    if len(unmapped) > 0:
        print(f"WARNING: {len(unmapped)} rows have a weak_cluster with no "
              f"entry in mapping: {list(unmapped['weak_cluster'].unique())}")
    return result, unmapped


def add_prob_label_columns(result, n_classes=6):
    """One-hot prob_label_i columns from hard_label."""
    out = result.copy()
    for i in range(n_classes):
        out[f"prob_label_{i}"] = (out["hard_label"] == i).astype(float)

    n_unlabelled = int(out["hard_label"].isna().sum())
    if n_unlabelled:
        print(f"WARNING: {n_unlabelled} rows have no hard_label; they are "
              f"excluded from training.")
    return out


def derive_medication_status(row):
    """0 none, 1 stimulant, 2 melatonin, 3 alpha2-agonist, 4 combined."""
    n_active = sum([row["med_has_stimulant"], row["med_has_melatonin"],
                    row["med_has_alpha2"]])
    if n_active == 0:
        return 0
    if n_active > 1:
        return 4
    if row["med_has_stimulant"]:
        return 1
    if row["med_has_melatonin"]:
        return 2
    return 3


def _filter_by_pid_index(df, pids):
    """Filter a participant-indexed frame; leave it unchanged if it has no such level."""
    try:
        return df.loc[df.index.get_level_values("participant_id").isin(pids)]
    except KeyError:
        return df


def report_gradient_conflict(model):
    """Summarise the gradient-conflict history recorded during training."""
    hist = getattr(model, "_grad_conflict_history", None)
    if not hist:
        print("\n  (no gradient-conflict history recorded)\n")
        return
    cos_vals = [c for _, c, _, _ in hist]
    n_projected = sum(1 for _, _, proj, _ in hist if proj)
    n_conflict = sum(1 for c in cos_vals if c < -0.1)
    n_aligned = sum(1 for c in cos_vals if c > 0.1)
    n_orth = len(cos_vals) - n_conflict - n_aligned
    print("\n--- Gradient conflict diagnostic (final training run) ---")
    print(f"  Measurements:  {len(cos_vals)} (batches with L_c > 0)")
    print(f"  Cosine mean {statistics.mean(cos_vals):+.4f}  "
          f"median {statistics.median(cos_vals):+.4f}  "
          f"min {min(cos_vals):+.4f}  max {max(cos_vals):+.4f}")
    print(f"  Conflict (cos<-0.1):     {n_conflict}/{len(cos_vals)}")
    print(f"  Orthogonal (-0.1..0.1):  {n_orth}/{len(cos_vals)}")
    print(f"  Aligned (cos>0.1):       {n_aligned}/{len(cos_vals)}")
    print(f"  Gradient projected:      {n_projected}/{len(cos_vals)}")
    print("---\n")


def report_trigger_rates(model):
    """Print per-rule trigger rates, and save the per-night log if recorded."""
    cf = model._cf
    if cf is None:
        return
    print("\n--- Rule trigger rates ---")
    for k in cf.RULE_NAMES:
        total = cf._trigger_total[k]
        count = cf._trigger_counts[k]
        rate = 100.0 * count / total if total > 0 else 0.0
        print(f"{k:35s}: {count:6d}/{total:6d} ({rate:6.2f}%)")
    if cf._detailed_trigger_log:
        trigger_df = cf.get_trigger_dataframe()
        trigger_df.to_csv("trigger_log.csv", index=False)
        rule_cols = [c for c in trigger_df.columns
                     if c not in ("participant_id", "night_date")]
        print(f"Saved per-night trigger log: {trigger_df.shape[0]} nights, "
              f"{len(rule_cols)} rules")
        print("Rules fired simultaneously per night:")
        print(trigger_df[rule_cols].sum(axis=1).value_counts().sort_index())
    print("---\n")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(args):
    run_dir = Path("runs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    data = Path(args.data)
    kb_dir = Path(args.kb)
    cfg_dir = Path(args.configs)
    cohort = args.cohort

    if cohort == "22q":
        print("Cohort profile:  22q11.2DS (Fitbit + Click EEG · 7-night protocol)")
    else:
        print("Cohort profile:  SSP (EmbracePlus + Dreem)")

    for p, label in [
        (data / "embraceplus", "embraceplus/"),
        (data / "dreem", "dreem/"),
        (data / "diary", "diary/"),
        (data / "demographics", "demographics/"),
        (kb_dir, "knowledge_base/"),
        (cfg_dir / "heuristics.toml", "heuristics.toml"),
    ]:
        if not p.exists():
            print(f"ERROR: {label} not found at {p}")
            sys.exit(1)

    # ================================================================
    # LAYER 1 — Ingestion
    # ================================================================
    sep("Layer 1 — ingestion")

    from nesy.config import load_config
    from nesy.ingestion import (
        load_embraceplus_directory, load_dreem_summary,
        load_sleep_diary, aggregate_embraceplus_daily,
        align_modalities, impute_non_wear,
        compute_rar_daily_features, compute_rar_participant_features,
        merge_rar_into_night_record, load_demographics,
        apply_analytic_sample_filter,
    )
    from nesy.ingestion.loaders import load_dreem_hypnogram
    from nesy.ingestion.align_epochs import EpochAligner

    cfg = load_config(cfg_dir / "pipeline.yaml")

    if cohort == "22q":
        from nesy.ingestion.adapters import load_fitbit_cohort, load_click_eeg
        fitbit_dir = data / "fitbit"
        if not fitbit_dir.exists():
            fitbit_dir = data
        ep = load_fitbit_cohort(fitbit_dir)
        click_summary = data / "click_eeg" / "summary.csv"
        click_epochs = data / "click_eeg" / "epochs"
        if click_summary.exists():
            dreem, _ = load_click_eeg(
                click_summary,
                epochs_dir=click_epochs if click_epochs.exists() else None,
            )
            print(f"  Click EEG: {len(dreem)} nights loaded")
        else:
            print(f"  WARNING: No Click EEG summary found at {click_summary}")
            dreem = pd.DataFrame()
        dreem_clean = dreem
    else:
        ep = load_embraceplus_directory(data / "embraceplus")
        dreem = load_dreem_summary(data / "dreem" / "dreem_summary.csv")

        # Dreem quality cleaning precedes everything else, because Dreem is
        # the anchor of the night record and of the analytic sample filter.
        from nesy.ingestion.loaders import flag_dreem_quality, dreem_clean_nights
        dreem_flagged = flag_dreem_quality(dreem)
        n_long = int(dreem_flagged["long_session_flag"].sum())
        n_rem = int(dreem_flagged["implausible_rem_flag"].sum())
        dreem_clean = dreem_clean_nights(dreem)
        print(f"  Dreem cleaning: {n_long} long-session nights, "
              f"{n_rem} implausible-REM nights flagged "
              f"({len(dreem)} -> {len(dreem_clean)} clean nights)")

    diary_path = list((data / "diary").glob("*.csv"))[0]
    diary = load_sleep_diary(diary_path)

    # participant filter from --participant or pipeline.yaml participant_ids
    filter_pids = None
    filter_pids_from_yaml = False
    if args.participant:
        filter_pids = [args.participant]
    elif cfg.cohort.participant_ids:
        filter_pids = cfg.cohort.participant_ids
        filter_pids_from_yaml = True
        print(f"  (filtering to {len(filter_pids)} participants from pipeline.yaml)")

    yaml_pids = set(cfg.cohort.participant_ids)
    found_ep_pids = set(ep["participant_id"].unique())
    print(f"\n[Participant audit] in YAML: {len(yaml_pids)}, "
          f"in EmbracePlus: {len(found_ep_pids)}, "
          f"missing EmbracePlus: {sorted(yaml_pids - found_ep_pids)}\n")

    if filter_pids:
        ep = ep[ep["participant_id"].isin(filter_pids)]
        dreem = _filter_by_pid_index(dreem, filter_pids)
        dreem_clean = _filter_by_pid_index(dreem_clean, filter_pids)
        diary = _filter_by_pid_index(diary, filter_pids)

    pids = ep["participant_id"].unique()
    print(f"Participants:    {len(pids)}")
    print(f"EP rows:         {len(ep):,}")

    # hypnogram epoch files, filtered by participant
    hyp_pattern = re.compile(r"dreem_hypno_.+_\d{4}-\d{2}-\d{2}\.csv$")
    all_hyp = sorted(f for f in (data / "dreem").glob("*.csv")
                     if hyp_pattern.match(f.name))
    if filter_pids:
        all_hyp = [f for f in all_hyp if any(pid in f.name for pid in filter_pids)]

    frames = []
    for i, f in enumerate(all_hyp, 1):
        if i % 50 == 0 or i == len(all_hyp):
            print(f"  aligning epochs: {i}/{len(all_hyp)}", end="\r")
        r = load_dreem_hypnogram(f, {"pre_aligned": True, "min_overlap_sec": 15})
        frames.append(r[0] if isinstance(r, tuple) else r)
    if frames:
        print()
        epochs = pd.concat(frames).sort_index()
    else:
        epochs = pd.DataFrame()

    ep_imp, _ = impute_non_wear(ep)

    # map Dreem 30 s stages onto the 1-min EmbracePlus time series
    if not epochs.empty and not ep_imp.empty:
        aligner = EpochAligner(grid_resolution_sec=60, min_overlap_sec=15)
        ep_imp = aligner.align_to_embraceplus(epochs, ep_imp)

    rar_p = compute_rar_participant_features(ep_imp, impute=False)
    rar_d = compute_rar_daily_features(ep_imp, impute=False)
    ep_daily = aggregate_embraceplus_daily(ep_imp)

    # Dreem (cleaned) is the anchor of the night record
    night_record = align_modalities(dreem_clean, diary, ep_daily)
    night_record = merge_rar_into_night_record(night_record, rar_p, rar_d)
    n_total = len(night_record)
    n_missing = night_record["diary_missing"].sum()
    print(f"Nights:          {n_total}")
    print(f"Diary coverage:  {100 * (n_total - n_missing) / n_total:.1f}%")
    print(f"RAR valid days:  {len(rar_d)}")

    # analytic sample filter
    ep_f, _, _, rar_pf, excl_report = apply_analytic_sample_filter(
        ep_imp, dreem, dreem_clean, diary, rar_p, rar_d,
        min_rar_days=cfg.analytic_sample.min_valid_rar_days,
        min_aligned_nights=cfg.analytic_sample.min_aligned_nights,
        min_qual_mean=cfg.analytic_sample.min_qual_mean,
        qual_mean_exceptions=cfg.analytic_sample.qual_mean_exceptions,
    )

    n_included = excl_report["included"].sum()
    n_excluded = (~excl_report["included"]).sum()
    print(f"Analytic sample: {n_included} included, {n_excluded} excluded")
    print(f"  (min RAR days={cfg.analytic_sample.min_valid_rar_days}, "
          f"min aligned nights={cfg.analytic_sample.min_aligned_nights}, "
          f"min qual_mean={cfg.analytic_sample.min_qual_mean})")
    if n_excluded > 0:
        reasons = excl_report[~excl_report["included"]]["exclusion_reason"]
        for reason, count in Counter(reasons).most_common():
            print(f"  {count}x: {reason}")

    included_pids = set(excl_report[excl_report["included"]].index)

    if filter_pids_from_yaml:
        yaml_filter = set(filter_pids)
        only_in_filter = included_pids - yaml_filter
        assert not only_in_filter, (
            "Analytic sample includes participant(s) absent from "
            f"pipeline.yaml participant_ids: {sorted(only_in_filter)}")
        print(f"  Funnel: {len(yaml_filter)} pre-filtered -> "
              f"{len(included_pids)} in final analytic sample")

    ep_imp = ep_f
    rar_p = rar_pf
    night_record = night_record[
        night_record.index.get_level_values("participant_id").isin(included_pids)
    ]
    n_total = len(night_record)
    n_missing = night_record["diary_missing"].sum()
    print(f"Nights (analytic sample): {n_total}")

    # ── Final convergence table and weak scaffold clustering ─────────────
    from nesy.reporting.final_table import (
        build_final_night_table, build_final_participant_aggregate,
    )
    demo_files = sorted((data / "demographics").glob("*.csv"))
    if not demo_files:
        print("ERROR: no demographics file found in demographics/")
        sys.exit(1)
    demo = load_demographics(demo_files[0])

    final_night_table = build_final_night_table(night_record, demo)
    final_participant_table = build_final_participant_aggregate(final_night_table)
    print(f"Final convergence table: {len(final_night_table)} nights, "
          f"{final_participant_table.shape[0]} participants, "
          f"{final_night_table.shape[1]} columns")

    participant_df, clusters_profile = generate_weak_clusters(
        final_participant_table, run_dir)
    participant_df.to_csv(run_dir / "weak_participant_clusters.csv", index=False)
    final_night_table = final_night_table.merge(
        participant_df[["participant_id", "weak_cluster"]],
        on="participant_id", how="left"
    )

    # ================================================================
    # LAYER 2 — Knowledge
    # ================================================================
    sep("Layer 2 — knowledge")

    from nesy.knowledge import (
        KnowledgeBase, run_gbtm, context_vectors_to_dataframe,
        add_demographic_context, parse_medication_column, select_n_groups,
    )

    kb = KnowledgeBase(kb_dir)
    missing_kb = kb.validate_kb_structure()
    print(f"KB structure:    {'complete' if not missing_kb else missing_kb}")

    # Restrict to nights with a diary, and align the raw diary to those nights.
    diary_raw = pd.read_csv(diary_path)
    if filter_pids:
        diary_raw = diary_raw[diary_raw["pid"].isin(filter_pids)]
    diary_raw = diary_raw[diary_raw["pid"].isin(included_pids)]

    final_night_table["night_date"] = pd.to_datetime(
        final_night_table["night_date"], format="mixed").dt.date
    diary_raw["sleep_date"] = pd.to_datetime(
        diary_raw["sleep_date"], format="mixed", dayfirst=True).dt.date
    diary_raw["diary_date"] = pd.to_datetime(
        diary_raw["diary_date"], format="mixed", dayfirst=True).dt.date

    final_night_table = final_night_table[final_night_table["diary_missing"] == False]
    pairs = set(zip(final_night_table["participant_id"], final_night_table["night_date"]))
    diary_raw = diary_raw[diary_raw.apply(
        lambda row: (row["pid"], row["sleep_date"]) in pairs, axis=1)]
    diary_raw["sleep_date"] = pd.to_datetime(
        diary_raw["sleep_date"], dayfirst=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Medication context feeds C2, C5 and C9, so a failure here stops the run.
    med_ctx = parse_medication_column(
        diary_raw, kb_dir, participant_col="pid", date_col="sleep_date")
    print(f"Medication context: {len(med_ctx)} nights")
    if "med_should_abstain_sol" in med_ctx.columns:
        print(f"Medication:      {med_ctx['med_should_abstain_sol'].sum()} "
              f"abstain-SOL nights")

    # GBTM trajectory groups (constraint modifier only)
    n_participants = len(ep_imp["participant_id"].unique())
    if n_participants < args.n_groups:
        print(f"GBTM:            skipping ({n_participants} < {args.n_groups} groups)")
        context_df = None
    else:
        optimal_k = args.n_groups
        if args.select_k and n_participants >= 6:
            print("  selecting optimal GBTM k via BIC...")
            try:
                bic_df = select_n_groups(ep_imp, k_range=range(2, 6), random_seed=42)
                optimal_k = int(bic_df[bic_df["suggested"]].index[0])
                print(f"  BIC table:\n{bic_df[['bic', 'aic']].to_string()}")
                print(f"  Using k={optimal_k}")
            except Exception as e:
                print(f"  BIC selection failed ({e}) — using k={optimal_k}")

        vectors, _ = run_gbtm(ep_imp, n_groups=optimal_k, random_seed=42)
        context_df = add_demographic_context(
            context_vectors_to_dataframe(vectors), demo)
        print(f"GBTM + D4:       {len(context_df)} participants")

    # ================================================================
    # LAYER 3 — Weak supervisory scaffold
    # ================================================================
    sep("Layer 3 — weak supervisory scaffold")

    from nesy.symbolic.labeler_b import name_clusters

    vote_table, mapping = name_clusters(clusters_profile)
    print("Voting table:\n", vote_table.to_string(index=False))
    print("cluster_to_phenotype:", mapping)

    # merge nightly medication context into the night table
    med_cols = [c for c in med_ctx.columns if c.startswith("med_")]
    final_night_table = final_night_table.drop(
        columns=[c for c in med_cols if c in final_night_table.columns])

    med_slice = (med_ctx.reset_index()[["participant_id", "night_date"] + med_cols]
                 .drop_duplicates(subset=["participant_id", "night_date"]))
    med_slice["participant_id"] = med_slice["participant_id"].astype(str).str.strip()
    med_slice["night_date"] = med_slice["night_date"].astype(str).str.strip()
    final_night_table["participant_id"] = final_night_table["participant_id"].astype(str).str.strip()
    final_night_table["night_date"] = final_night_table["night_date"].astype(str).str.strip()

    final_night_table = final_night_table.merge(
        med_slice, on=["participant_id", "night_date"], how="left")
    for c in med_cols:
        if c == "med_drug_classes":
            final_night_table[c] = final_night_table[c].apply(
                lambda x: x if isinstance(x, list) else [])
        else:
            final_night_table[c] = final_night_table[c].fillna(False).astype(bool)

    final_night_table["medication_status"] = final_night_table.apply(
        derive_medication_status, axis=1)

    result, _ = add_hard_label(final_night_table, mapping)
    result = add_prob_label_columns(result)
    print(result[["participant_id", "night_date", "hard_label"]].head(10))

    # ================================================================
    # LAYER 4 — Constraint-guided neurosymbolic training
    # ================================================================
    sep("Layer 4 — constraint-guided fusion")

    from nesy.fusion import RocketExtractor, EEGFeatureExtractor, FusionModel

    labelled_result = result[result["hard_label"].notna()].reset_index(drop=True)
    night_index = list(zip(labelled_result["participant_id"],
                           labelled_result["night_date"]))
    print(f"Labelled nights:   {len(night_index)}")

    model = FusionModel(
        lr=1e-3, epochs=40, batch_size=256, random_seed=42,
        constraint_config=cfg_dir / "heuristics.toml",
        disabled_rules={r.strip() for r in args.disable_rules.split(",") if r.strip()},
        use_smooth=not args.no_smooth,
        smooth_alpha_scale=args.smooth_alpha_scale,
        audit_triggers=args.audit_triggers,
    )

    if args.with_clinical:
        clinical_df = model.extract_clinical_features(labelled_result, night_record)
        if clinical_df is not None:
            n_clin = clinical_df.notna().any(axis=1).sum()
            print(f"Clinical features: {clinical_df.shape[1]} per night "
                  f"({n_clin}/{len(clinical_df)} with data)")
    else:
        clinical_df = None
        print("Clinical features: excluded (use --with-clinical to add)")

    rocket_cfg = getattr(cfg, "rocket", None)
    win_start = rocket_cfg.window_start if rocket_cfg else "22:00"
    win_end = rocket_cfg.window_end if rocket_cfg else "08:00"
    n_kernels = rocket_cfg.n_kernels if rocket_cfg else 1000
    print(f"  extracting ROCKET features (window {win_start}→{win_end})...")
    rocket = RocketExtractor(n_kernels=n_kernels,
                             window_start=win_start, window_end=win_end)
    rocket_df = rocket.fit_transform(ep_imp, night_index)
    print(f"ROCKET features:   {rocket_df.shape[1]} per night")

    print("  extracting symbolic hypnogram features...")
    if not epochs.empty:
        sym_df = EEGFeatureExtractor().transform(epochs.reset_index(), night_index)
        print(f"EEG features:      {sym_df.shape[1]} per night")
        print(f"Nights with EEG:   {sym_df.notna().any(axis=1).sum()} / {len(night_index)}")
    else:
        sym_df = None
        print("EEG features:      none (no epoch files found)")

    gbtm_groups_df = None
    if context_df is not None and "dominant_group" in context_df.columns:
        gbtm_groups_df = context_df[["dominant_group", "confidence"]].copy()
    else:
        print("GBTM groups:       not available (trajectory modifier disabled)")

    # The model reads constraint observations (IV, SE, WASO, TST, medication
    # flags and others) from the final night table.
    model.fit(
        rocket_df, sym_df, context_df, result,
        clinical_df=clinical_df,
        night_record=final_night_table,
        gbtm_groups_df=gbtm_groups_df,
    )
    print("Final multipliers:", {k: round(v, 3) for k, v in model._last_gammas.items()},
          f"gamma_smooth={model._last_gamma_smooth:.3f}")
    print(f"Fusion model fitted (constraints={model.use_constraints}, "
          f"lagrangian={model.use_lagrangian}, smooth={model.use_smooth})")
    report_gradient_conflict(model)
    if args.audit_triggers:
        report_trigger_rates(model)

    cv = None
    n_pids = labelled_result["participant_id"].nunique()
    if n_pids < 5:
        print(f"LOPO-CV:           skipping ({n_pids} participants — need ≥ 5)")
    else:
        print(f"  running LOPO-CV across {n_pids} participants...")
        cv = model.lopo_cv(
            rocket_df, sym_df, context_df, result,
            clinical_df=clinical_df,
            night_record=final_night_table,
            gbtm_groups_df=gbtm_groups_df,
        )
        print("LOPO-CV results:")
        print(f"  Participants:    {cv['n_participants_evaluated']}")
        print(f"  Nights:          {cv['n_nights_evaluated']}")
        print(f"  Accuracy:        {cv['overall_accuracy']:.3f}")
        print(f"  Cohen kappa:     {cv['cohen_kappa']:.3f}")
        print("  Per-label F1:")
        for lbl, m in cv["per_label"].items():
            if m["support"] > 0:
                print(f"    {lbl}: F1={m['f1']:.3f}  (n={m['support']})")

    # ================================================================
    # POST-PROCESSING — Stratification
    # ================================================================
    sep("Post-processing — stratification")

    from nesy.stratification.phenotype import StratificationLayer

    strat = StratificationLayer(
        dominance_threshold=args.dominance_threshold,
        instability_threshold=args.instability_threshold,
        compound_min_pct=0.25, margin_buffer=args.margin_buffer,
        use_soft_probs=args.use_soft_probs,
    )
    strat_df = strat.stratify_v3b(result)
    summary = strat.summary(strat_df)

    print(f"Participants:         {summary['total_participants']}")
    print(f"Dominant assigned:    {summary['dominant_assigned']}")
    print(f"Compound assigned:    {summary['compound_assigned']}")
    print(f"Unstable/mixed:       {summary['unstable_mixed']}")
    print(f"Indeterminate:        {summary['indeterminate']}")
    print(f"High instability:     {summary['high_instability_n']}")
    print(f"Mean instability:     {summary['mean_instability']:.3f}")
    print("\nY_Patient distribution:")
    for name, count in sorted(summary["y_patient_distribution"].items(),
                              key=lambda x: -x[1]):
        print(f"  {name:<35} {count}")

    # ================================================================
    # SAVE
    # ================================================================
    sep("Summary")
    print(f"Participants:    {len(pids)}")
    print(f"Nights total:    {n_total}")
    print(f"Diary coverage:  {100 * (n_total - n_missing) / n_total:.1f}%")
    print("─" * 55)

    demo_cols = ["age", "age_band", "sex_encoded", "autism_status", "family_id"]

    if cv is not None:
        held_dates = labelled_result.iloc[model._last_lopo_idx]["night_date"].values
        lopo_probs = model._last_lopo_Y_pred.numpy()
        lopo_df = pd.DataFrame({
            "participant_id": model._last_lopo_pids,
            "night_date": held_dates,
            "true_label": model._last_lopo_y_true,
            "pred_label": lopo_probs.argmax(axis=1),
        })
        for i in range(lopo_probs.shape[1]):
            lopo_df[f"prob_label_{i}"] = lopo_probs[:, i]
        lopo_df.to_csv(run_dir / "lopo_per_night_predictions.csv", index=False)
        print("  LOPO-CV predictions saved → lopo_per_night_predictions.csv")

    if model._iv_values is not None:
        np.save(run_dir / "iv_calibration_values.npy", model._iv_values)

    run_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "participants": len(pids),
        "nights_total": int(n_total),
        "diary_coverage_pct": round(100 * (n_total - n_missing) / n_total, 1),
        "model": {
            "disabled_rules": sorted(model._disabled_rules or []),
            "epochs": model.epochs,
            "use_constraints": model.use_constraints,
            "use_lagrangian": model.use_lagrangian,
            "use_smooth": model.use_smooth,
            "smooth_delta": model.smooth_delta,
            "smooth_alpha_scale": model.smooth_alpha_scale,
        },
        "stratification": {
            "dominance_threshold": args.dominance_threshold,
            "instability_threshold": args.instability_threshold,
            "margin_buffer": args.margin_buffer,
            **{k: summary[k] for k in (
                "total_participants", "dominant_assigned", "compound_assigned",
                "unstable_mixed", "high_instability_n", "mean_instability",
                "y_patient_distribution")}
        },
    }
    if cv is not None:
        run_report["layer4_lopo_cv"] = {
            "n_participants": cv["n_participants_evaluated"],
            "n_nights": cv["n_nights_evaluated"],
            "accuracy": cv["overall_accuracy"],
            "cohen_kappa": cv["cohen_kappa"],
            "per_label_f1": {k: v["f1"] for k, v in cv["per_label"].items()
                             if v["support"] > 0},
        }

    # Y_Patient from the weak labels
    strat_df.join(demo[demo_cols], how="left").to_csv(run_dir / "y_patient.csv")

    # Y_Patient from the model's held-out predictions
    if cv is not None:
        try:
            from layer4_stratification import stratify_from_model_predictions
            strat_df_l4 = stratify_from_model_predictions(
                model._last_lopo_pids, model._last_lopo_Y_pred, strat,
                args.use_soft_probs)
            strat_df_l4.join(demo[demo_cols], how="left").to_csv(
                run_dir / "y_patient_L4.csv")
            print("  Saved Layer 4 Y_patient → y_patient_L4.csv")
        except Exception as e:
            print(f"  Layer 4 Y_patient save failed: {e}")

    final_night_table.to_csv(run_dir / "final_night_table.csv", index=False)
    final_participant_table.to_csv(run_dir / "final_participant_table.csv", index=False)

    label_cols = (["participant_id", "night_date", "weak_cluster", "hard_label"]
                  + [f"prob_label_{i}" for i in range(6)])
    result[[c for c in label_cols if c in result.columns]].to_csv(
        run_dir / "night_labels.csv", index=False)

    if context_df is not None:
        prob_cols = [c for c in context_df.columns if c.startswith("group_prob_")]
        d4_cols = [c for c in ["age_band", "sex_encoded", "autism_status"]
                   if c in context_df.columns]
        save_cols = [c for c in prob_cols + ["dominant_group", "confidence", "n_nights"] + d4_cols
                     if c in context_df.columns]
        gbtm_out = context_df[save_cols].join(demo[["family_id"]], how="left")
        gbtm_out.to_csv(run_dir / "gbtm_groups.csv")
        run_report["gbtm"] = {
            "n_participants": len(context_df),
            "n_groups": len(prob_cols),
            "group_counts": context_df["dominant_group"].value_counts().to_dict(),
            "mean_confidence": round(float(context_df["confidence"].mean()), 3),
        }

    with open(run_dir / "pipeline_report.json", "w") as f:
        json.dump(run_report, f, indent=2, default=str)

    if args.report:
        try:
            from nesy.reporting.report import generate_cohort_reports
            print("\n  Generating participant reports...")
            generate_cohort_reports(
                run_dir=run_dir,
                night_labels=result,
                y_patient=strat_df,
                night_record=final_night_table,
                med_ctx=med_ctx,
                rar_p=rar_p,
                demo=demo,
                gbtm=context_df,
            )
        except Exception as e:
            print(f"  Report generation failed: {e}")
            traceback.print_exc()

    print(f"\nOutputs saved to: {run_dir}/")


if __name__ == "__main__":
    run(parse_args())

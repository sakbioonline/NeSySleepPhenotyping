"""
NeSy Sleep Phenotyping Pipeline — test and run script.

Five-layer neuro-symbolic pipeline for multi-modal adolescent sleep phenotyping.
Layers run sequentially; comment out any block below to skip it.

Pipeline layers
---------------
  Layer 1  Ingestion        EmbracePlus actigraphy, Dreem EEG, sleep diary
                            RAR metrics, non-wear imputation, epoch alignment
                            Analytic sample filter (min RAR days, Dreem nights,
                            diary nights — set in configs/pipeline.yaml)

  Layer 2  Knowledge        Clinical knowledge base, medication parsing,
                            GBTM trajectory grouping (D4 context vector),
                            heuristic threshold loading

  Layer 3  Symbolic         Six labelling functions → Snorkel label model
           labelling        → probabilistic night-level phenotype labels
                            Labels: healthy, paradoxical insomnia,
                            psychophysiological insomnia, restless,
                            RAR misalignment, hypersomnolence,
                            SDB risk, coupling impairment

  Layer 4  Neuro-symbolic   ROCKET (actigraphy waveform) + EEG morphology +
           fusion           GBTM context → logistic regression trained on
                            Snorkel soft labels → LOPO-CV validation

  Post     Stratification   Night labels → Y_Patient dominant phenotype
           processing       + instability score + compound patterns

  Layer 5  Validation       Coverage, internal consistency (LOPO-CV kappa),
                            fairness (sex / age / autism status),
                            within-family concordance (ASD vs sibling)
                            Sensitivity analyses (SA1 all, SA2 stimulant-free,
                            SA3 melatonin + elevated SOL)

Outputs (saved to runs/TIMESTAMP/)
-----------------------------------
  pipeline_report.json      All summary metrics, GBTM, validation results
  final_night_table.csv     Single convergence table: participant_id,
                             family_id, night_date, demography, EEG/Dreem,
                             EmbracePlus, diary — one row per analytic night
  final_participant_table.csv  Same, aggregated to one row per participant
  night_labels.csv          Per-night hard label + 6 rule votes
  y_patient.csv             Y_Patient + demographics + family_id
  within_family_pairs.csv   ASD vs sibling phenotype per family
  gbtm_groups.csv           GBTM group assignments + demographics + family_id

Usage
-----
  # Full cohort run
  python test_pipeline.py --data data/ssp

  # Single participant
  python test_pipeline.py --data data/ssp --participant SP0000000

  # Specify GBTM group count directly
  python test_pipeline.py --data data/ssp --n-groups 3

  # Select optimal GBTM k via BIC (k=2..5)
  python test_pipeline.py --data data/ssp --select-k

  # Add clinical features to fusion model (SE, SOL, ANX, SQ, N3, IV)
  python test_pipeline.py --data data/ssp --with-clinical

  # Run ablation study across feature stream combinations
  python test_pipeline.py --data data/ssp --ablation

  # Non-standard paths (only needed if folder names differ from defaults)
  python test_pipeline.py --data data/ssp --kb my_kb --configs my_configs

Arguments
---------
  --data          Path to cohort data root (required)
                  Expected structure: data/ssp/embraceplus/*.csv
                                      data/ssp/dreem/*.csv + hypnogram files
                                      data/ssp/diary/*.csv
                                      data/ssp/demographics/*.csv
  --kb            Path to knowledge_base directory (default: knowledge_base)
  --configs       Path to configs directory (default: configs)
  --participant   Run on a single participant ID only
  --n-groups      Number of GBTM trajectory groups (default: 4)
  --select-k      Select optimal k via BIC before GBTM fitting
  --with-clinical Add clinical features (SE,SOL,ANX,SQ,N3,IV) to fusion model
  --ablation      Run ablation study: V1 ROCKET, V2 +EEG, V3 +GBTM, V4 sanity
"""

import argparse
import sys
import warnings
from pathlib import Path

from nesy.stratification.sensitivity import run_stratification_sensitivity_grid

# ensure nesy package is importable regardless of working directory
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="NeSy pipeline test")
    p.add_argument("--data", required=True)
    p.add_argument("--kb", default="knowledge_base")
    p.add_argument("--configs", default="configs")
    p.add_argument("--participant", default=None)
    p.add_argument("--use-soft-probs", action="store_true", default=False,
                   help="Enable soft continuous expected probability aggregation")
    p.add_argument("--strat-sensitivity", action="store_true",
                   help="Run 3x3x3 grid sensitivity analysis on stratification thresholds")
    p.add_argument("--n-groups", type=int, default=4,
                   help="number of GBTM groups (default=4)")
    p.add_argument("--select-k", action="store_true",
                   help="select optimal GBTM k via BIC before fitting")

    # Layer 4 ablation flags
    # Default: ROCKET + EEG symbols + GBTM context (original design)
    # Add --with-clinical to include clinical features (SE, SOL, ANX etc.)
    # Use --ablation to run all variants and compare kappas
    p.add_argument("--with-clinical", action="store_true",
                   help="add clinical features (SE,SOL,ANX,SQ,N3,IV) to fusion model")
    p.add_argument("--ablation", action="store_true",
                   help="run ablation study across all feature combinations")
    p.add_argument("--report", action="store_true",
                   help="generate one PDF phenotype report per participant")
    p.add_argument("--cohort", type=str, default="ssp",
                   choices=["ssp", "22q"],
                   help="cohort: ssp (EmbracePlus+Dreem) or 22q (Fitbit+ClickEEG) "
                        "[default: ssp]")
    p.add_argument("--scalability", action="store_true",
                   help="modality scalability: compare full multi-modal vs "
                        "actigraphy+diary vs actigraphy only")
    p.add_argument("--compare-architectures", action="store_true",
                   help="direct architecture comparison: full-fit LogReg+Lc vs "
                        "full-fit MLP+Lc on the complete training set")
    p.add_argument("--lambda-search", action="store_true",
                   help="constraint weight sensitivity: run LOPO-CV across "
                        "lambda values {0.1, 0.2, 0.3, 0.5, 1.0} and report "
                        "all results — standard hyperparameter selection")
    return p.parse_args()


def sep(title=""):
    print(f"\n{'─' * 55}")
    if title:
        print(title)


def run(args):
    import pandas as pd
    import numpy as np

    data = Path(args.data)
    kb_dir = Path(args.kb)
    cfg_dir = Path(args.configs)
    use_soft_probs = args.use_soft_probs

    # cohort profile — selects device adapters and analytic thresholds
    cohort = getattr(args, "cohort", "ssp")
    if cohort == "22q":
        import yaml as _yaml
        _ccfg_path = cfg_dir / "22q_pipeline.yaml"
        cohort_cfg = _yaml.safe_load(open(_ccfg_path)) if _ccfg_path.exists() else {}
        print(f"Cohort profile:  22q11.2DS (Fitbit + Click EEG · 7-night protocol)")
    else:
        cohort_cfg = {}
        print(f"Cohort profile:  SSP (EmbracePlus + Dreem)")

    for p, label in [
        (data / "embraceplus", "embraceplus/"),
        (data / "dreem", "dreem/"),
        (data / "diary", "diary/"),
        (kb_dir, "knowledge_base/"),
        (cfg_dir / "heuristics.toml", "heuristics.toml"),
    ]:
        if not p.exists():
            print(f"ERROR: {label} not found at {p}")
            sys.exit(1)

    # ================================================================
    # LAYER 1 — Ingestion
    # Comment out this block to skip
    # ================================================================
    sep("Layer 1 — ingestion")

    from nesy.ingestion import (
        load_embraceplus_directory, load_dreem_summary,
        load_sleep_diary, aggregate_embraceplus_daily,
        align_modalities, alignment_reports_to_dataframe,
        impute_non_wear, imputation_reports_to_dataframe,
        compute_rar_daily_features, compute_rar_participant_features,
        merge_rar_into_night_record, load_demographics,
    )
    from nesy.ingestion.loaders import load_dreem_hypnogram
    from nesy.ingestion.align_epochs import EpochAligner  # <--- IMPORTED ALIGNER

    if cohort == "22q":
        # ── 22q: Fitbit + Click EEG ──────────────────────────────────────────
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
    else:
        # ── SSP: EmbracePlus + Dreem ─────────────────────────────────────────
        ep = load_embraceplus_directory(data / "embraceplus")
        dreem = load_dreem_summary(data / "dreem" / "dreem_summary.csv")

        # Dreem quality cleaning — must happen before anything downstream
        # touches dreem, since align_modalities uses Dreem as the anchor
        # and the analytic sample filter needs dreem_clean, not raw dreem.
        from nesy.ingestion.loaders import flag_dreem_quality, dreem_clean_nights
        dreem_flagged = flag_dreem_quality(dreem)
        n_long = int(dreem_flagged["long_session_flag"].sum())
        n_rem = int(dreem_flagged["implausible_rem_flag"].sum())
        dreem_clean = dreem_clean_nights(dreem)
        print(f"  Dreem cleaning: {n_long} long-session nights, "
              f"{n_rem} implausible-REM nights flagged "
              f"({len(dreem)} -> {len(dreem_clean)} clean nights)")

    diary = load_sleep_diary(list((data / "diary").glob("*.csv"))[0])

    # apply participant filter — from --participant flag OR pipeline.yaml participant_ids
    from nesy.config import load_config as _load_cfg
    _cfg = _load_cfg(cfg_dir / "pipeline.yaml")
    filter_pids = None
    filter_pids_from_yaml = False
    if args.participant:
        filter_pids = [args.participant]
    elif _cfg.cohort.participant_ids:
        filter_pids = _cfg.cohort.participant_ids
        filter_pids_from_yaml = True
        print(f"  (filtering to {len(filter_pids)} participants from pipeline.yaml)")

    # Extract the list from pipeline.yaml and the actual IDs found in EmbracePlus
    yaml_pids = set(_cfg.cohort.participant_ids)
    found_ep_pids = set(ep["participant_id"].unique())

    # Identify participants in YAML/Dreem that have no EmbracePlus data
    missing_ep = yaml_pids - found_ep_pids

    print(f"\n[Participant Audit] Total in YAML: {len(yaml_pids)}")
    print(f"[Participant Audit] Total in EmbracePlus: {len(found_ep_pids)}")
    print(f"[Participant Audit] Missing EmbracePlus ID: {sorted(list(missing_ep))}\n")

    if filter_pids:
        ep = ep[ep["participant_id"].isin(filter_pids)]
        try:
            dreem = dreem.loc[dreem.index.get_level_values("participant_id").isin(filter_pids)]
        except:
            pass
        try:
            dreem_clean = dreem_clean.loc[
                dreem_clean.index.get_level_values("participant_id").isin(filter_pids)]
        except:
            pass
        try:
            diary = diary.loc[diary.index.get_level_values("participant_id").isin(filter_pids)]
        except:
            pass

    pids = ep["participant_id"].unique()
    print(f"Participants:    {len(pids)}")
    print(f"EP rows:         {len(ep):,}")

    # epoch alignment — filter files by participant
    import re as _re
    hyp_pattern = _re.compile(r"dreem_hypno_.+_\d{4}-\d{2}-\d{2}\.csv$")
    all_hyp = sorted(f for f in (data / "dreem").glob("*.csv")
                     if hyp_pattern.match(f.name))
    if filter_pids:
        all_hyp = [f for f in all_hyp
                   if any(pid in f.name for pid in filter_pids)]

    frames, align_reports = [], []
    for i, f in enumerate(all_hyp, 1):
        if i % 50 == 0 or i == len(all_hyp):
            print(f"  aligning epochs: {i}/{len(all_hyp)}", end="\r")
        r = load_dreem_hypnogram(f, {"pre_aligned": True, "min_overlap_sec": 15})
        if isinstance(r, tuple):
            df, rep = r
            align_reports.append(rep)
        else:
            df = r
        frames.append(df)
    if frames:
        print()
        epochs = pd.concat(frames).sort_index()
    else:
        epochs = pd.DataFrame()
    print("Raw EEG epoch data",epochs.head(20))
    ep_imp, imp_reports = impute_non_wear(ep)

    # ── Map Dreem 30s EEG stages onto 1-min EmbracePlus timeseries ──────
    if not epochs.empty and not ep_imp.empty:
        aligner = EpochAligner(grid_resolution_sec=60, min_overlap_sec=15)
        ep_imp = aligner.align_to_embraceplus(epochs, ep_imp)

    rar_p = compute_rar_participant_features(ep_imp, impute=False)
    rar_d = compute_rar_daily_features(ep_imp, impute=False)

    # ep_imp now contains eeg_stage — ep_n3_activity_mean is calculated here
    ep_daily = aggregate_embraceplus_daily(ep_imp)

    # dreem_clean is the anchor now, not raw dreem — bad nights (long-session,
    # implausible-REM) never enter the night record at all, matching align.py's
    # "Dreem is anchor" design applied to the cleaned rather than raw table.
    night_record = align_modalities(dreem_clean, diary, ep_daily)
    print("night record 1:",len(night_record))
    #print(f"Night record first appearnce (align modality): ",len(night_record))
    #print(night_record.head(5))
    night_record = merge_rar_into_night_record(night_record, rar_p, rar_d)
    print("night record 2:", len(night_record))
    #print(f"Night record after merge ", len(night_record))
    #print(night_record.head(5))
    n_total = len(night_record)
    n_missing = night_record["diary_missing"].sum()
    print("Diary missing: ",n_missing)
    print(f"Nights:          {n_total}")
   # print("I got here")
    print(f"Diary coverage:  {100 * (n_total - n_missing) / n_total:.1f}%")
    print(f"RAR valid days:  {len(rar_d)}")

    # apply analytic sample filter
    from nesy.config import load_config
    from nesy.ingestion import apply_analytic_sample_filter
    cfg = load_config(cfg_dir / "pipeline.yaml")
    ep_f, aligned_f, diary_f, rar_pf, excl_report = apply_analytic_sample_filter(
        ep_imp, dreem, dreem_clean, diary, rar_p, rar_d,
        min_rar_days=cfg.analytic_sample.min_valid_rar_days,
        min_aligned_nights=cfg.analytic_sample.min_aligned_nights,
        min_qual_mean=cfg.analytic_sample.min_qual_mean,
        qual_mean_exceptions=cfg.analytic_sample.qual_mean_exceptions,
    )

    print(f"Night record after analytical sample filter")

    n_included = excl_report["included"].sum()
    n_excluded = (~excl_report["included"]).sum()
    print(f"Analytic sample: {n_included} included, {n_excluded} excluded")
    print(f"  (min RAR days={cfg.analytic_sample.min_valid_rar_days}, "
          f"min aligned nights={cfg.analytic_sample.min_aligned_nights}, "
          f"min qual_mean={cfg.analytic_sample.min_qual_mean})")
    if n_excluded > 0:
        reasons = excl_report[~excl_report["included"]]["exclusion_reason"]
        from collections import Counter
        reason_counts = Counter(reasons)
        for reason, count in reason_counts.most_common():
            print(f"  {count}x: {reason}")

    # replace working variables with filtered versions
    included_pids = set(excl_report[excl_report["included"]].index)

    if filter_pids_from_yaml:
        yaml_pids = set(filter_pids)
        only_in_filter = included_pids - yaml_pids
        n_dropped = len(yaml_pids - included_pids)
        assert not only_in_filter, (
            "Analytic sample includes participant(s) not present in "
            f"pipeline.yaml's participant_ids — should be impossible by "
            f"construction, indicates a real bug: {sorted(only_in_filter)}"
        )
        print(f"  Funnel: {len(yaml_pids)} pre-filtered (Dreem-only) -> "
              f"{len(included_pids)} in final analytic sample "
              f"({n_dropped} further excluded by RAR/aligned-nights/qual "
              f"criteria beyond Dreem alone — expected, not an error)")

    ep_imp = ep_f
    rar_p = rar_pf
    #print("night record columns", night_record.columns.tolist())
    night_record = night_record[
        night_record.index.get_level_values("participant_id").isin(included_pids)
    ]
    n_total = len(night_record)  # update to filtered count
    n_missing = night_record["diary_missing"].sum()
    print(f"Nights (analytic sample): {n_total}")

    # ── Final convergence table: participant_id, family_id, date,
    #    demography, EEG, EmbracePlus, diary — single point of truth
    #    for the analytic sample, independent of Layers 2-5 below ──
    from nesy.ingestion.loaders import load_demographics as _load_demo_final
    from nesy.reporting.final_table import (
        build_final_night_table, build_final_participant_aggregate,
    )
    _demo_dir_final = data / "demographics"
    _demo_files_final = list(_demo_dir_final.glob("*.csv")) if _demo_dir_final.exists() else []
    if _demo_files_final:
        _demo_final = _load_demo_final(_demo_files_final[0])
        final_night_table = build_final_night_table(night_record, _demo_final)
        #print(final_night_table)
        final_participant_table = build_final_participant_aggregate(final_night_table)
        print(f"Final convergence table: {len(final_night_table)} nights, "
              f"{final_participant_table.shape[0]} participants, "
              f"{final_night_table.shape[1]} columns")
    else:
        final_night_table = None
        final_participant_table = None
        print("  WARNING: no demographics found — final convergence table not built")

    # # ================================================================
    # # LAYER 2 — Knowledge
    # # Comment out this block to skip
    # # ================================================================
    sep("Layer 2 — knowledge")

    from nesy.knowledge import (
        load_knowledge_store, KnowledgeBase, run_gbtm,
        context_vectors_to_dataframe, add_demographic_context,
        suggest_rocket_window, parse_medication_column, select_n_groups,
    )

    ks = load_knowledge_store(cfg_dir / "heuristics.toml")
    kb = KnowledgeBase(kb_dir)
    missing_kb = kb.validate_kb_structure()
    print(f"KB structure:    {'complete' if not missing_kb else missing_kb}")
    #print(f"R1 SE thresh:    {ks.r1.SE_healthy_min}%")

    diary_raw = pd.read_csv(list((data / "diary").glob("*.csv"))[0]) #reading diary again to get ceate medication context. We can't do it earler as the knowledge -base reside in layer 2
    ##########debugging###########################
    # print("diary_raw columns:", diary_raw.columns.tolist())
    # print("diary_raw index:", diary_raw.index.names)
    # print("diary_raw shape:", diary_raw.shape)
    # print(diary_raw.head(2))
    # print(night_record.head(5))
    # pd.DataFrame(night_record).to_csv("debugging_nr.csv",index=False)
    # pd.DataFrame(diary_raw).to_csv("debugging_dr.csv", index=False)
    #############################################
    # filter by both pipeline.yaml participant list AND analytic sample inclusion
    active_pids = included_pids if "included_pids" in dir() else None
    if filter_pids:
        diary_raw = diary_raw[diary_raw["pid"].isin(filter_pids)]
    if active_pids:
        diary_raw = diary_raw[diary_raw["pid"].isin(active_pids)]
    #print("Diary raw after either filter_pid or active_pid",len(diary_raw))

    try:
        # if "sleep_date" in diary_raw.columns:
        #     diary_raw["sleep_date"] = pd.to_datetime(
        #         diary_raw["sleep_date"].astype(str).str.strip(),
        #         format="mixed",
        #         dayfirst=True,
        #         errors="coerce"
        #     ).dt.strftime("%Y-%m-%d")
           # df2_filtered = df2[df2.index.isin(df1.index)]
        #print(f"Date fromat from night_record: ",type(night_record['night_date']))
        #print(f"Date fromat from night_table: ",final_night_table['night_date'].dtype)
        # final_night_table["night_date"] = pd.to_datetime(final_night_table['night_date'],dayfirst=True).dt.date
        # diary_raw['sleep_date'] = pd.to_datetime(diary_raw['sleep_date'],dayfirst=True).dt.date
        # diary_raw['diary_date'] = pd.to_datetime(diary_raw['diary_date'],dayfirst=True).dt.date
        # 1. final_night_table is already ISO ('YYYY-MM-DD')
        final_night_table["night_date"] = pd.to_datetime(final_night_table['night_date'], format="mixed").dt.date

        # 2. diary_raw has UK dates ('DD/MM/YYYY')
        diary_raw['sleep_date'] = pd.to_datetime(diary_raw['sleep_date'], format="mixed", dayfirst=True).dt.date
        diary_raw['diary_date'] = pd.to_datetime(diary_raw['diary_date'], format="mixed", dayfirst=True).dt.date

        final_night_table = final_night_table[final_night_table['diary_missing'] == False]
        pairs = set(zip(final_night_table["participant_id"], final_night_table["night_date"]))
        ##########################################
        # night_dates = set(final_night_table.loc[final_night_table['participant_id'] == 'SP0000000', 'night_date'])
        # diary_dates = set(diary_raw.loc[diary_raw['pid'] == 'SP0000000', 'sleep_date'])

        # print("In night table but not diary:", night_dates - diary_dates)
        # print("In diary but not night table:", diary_dates - night_dates)

        ###########################################

        # print("Lenght of pair",len(pairs))
        # print("Date format diary before:", type(diary_raw['sleep_date'].iloc[0]))
        diary_raw = diary_raw[diary_raw.apply(lambda row: (row["pid"], row["sleep_date"]) in pairs, axis=1)]
        #print("Date format diary after:", type(diary_raw['sleep_date'].iloc[0]))
        #diary_raw = diary_raw[diary_raw.index.isin(night_record.index)]
        #print("new diary formed lenght:",len(diary_raw))
        # 1. Force dayfirst=True on diary_raw sleep_date before passing to parse_medication_column
        ######################################
        if "sleep_date" in diary_raw.columns:
            diary_raw["sleep_date"] = pd.to_datetime(
                diary_raw["sleep_date"], dayfirst=True, errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        ##########################################
        med_ctx = parse_medication_column(
            diary_raw, kb_dir, participant_col="pid", date_col="sleep_date"  # this should be sleep date
        )
        print("Still debugging",med_ctx[med_ctx["participant_id"] == "SP0000000"][["participant_id", "night_date", "med_drug_classes"]])
        #print(f"Checking med_ctx from pipeline - {med_ctx.columns.tolist()}")
        #print(med_ctx)
        #pd.DataFrame(med_ctx).to_csv("debugging_md.csv", index=False)
        print('Lenght of medcontext',len(med_ctx))
        n_abs = med_ctx["med_should_abstain_sol"].sum()
        print(f"Medication:      {n_abs} abstain-SOL nights")
    except Exception as e:
        med_ctx = None
        print(f"Medication:      WARNING — {e}")

    n_participants = len(ep_imp["participant_id"].unique())
    if n_participants < args.n_groups:
        print(f"GBTM:            skipping ({n_participants} < {args.n_groups} groups)")
        context_df = None
    else:
        # BIC-based group selection — only when --select-k flag is passed
        # Use: python test_pipeline.py --select-k  to run BIC
        # Use: python test_pipeline.py --n-groups 3  to use k=3 directly
        if getattr(args, "select_k", False) and n_participants >= 6:
            print("  selecting optimal GBTM k via BIC...")
            try:
                bic_df = select_n_groups(ep_imp, k_range=range(2, 6), random_seed=42)
                optimal_k = int(bic_df[bic_df["suggested"]].index[0])
                print(f"  BIC table:\n{bic_df[['bic', 'aic']].to_string()}")
                print(f"  Using k={optimal_k}")
            except Exception as e:
                optimal_k = args.n_groups
                print(f"  BIC selection failed ({e}) — using k={optimal_k}")
        else:
            optimal_k = args.n_groups

        vectors, model = run_gbtm(ep_imp, n_groups=optimal_k, random_seed=42)
        context_df = context_vectors_to_dataframe(vectors)
        demo_dir = data / "demographics"
        demo_files = list(demo_dir.glob("*.csv")) if demo_dir.exists() else []
        if demo_files:
            demo = load_demographics(demo_files[0])
            context_df = add_demographic_context(context_df, demo)
            print(f"GBTM + D4:       {len(context_df)} participants")
        else:
            print(f"GBTM:            {len(context_df)} participants (no D4)")

    window = suggest_rocket_window(ep_imp)
    print(f"ROCKET window:   {window['window_start']} → {window['window_end']}")

    # # ================================================================
    # # LAYER 3 — Symbolic labelling
    # # Comment out this block to skip
    # # ================================================================
    sep("Layer 3 — symbolic labelling")

    from nesy.symbolic.labeler import SymbolicLabeler

    labeler = SymbolicLabeler(ks, n_epochs=500, verbose=False)

    try:
        # ── Medication date-alignment sanity check ──────────────────
        # Standing check for the sleep_date/diary_date join bug found
        # and fixed in labeler.py's apply_rules() — diary_date is when
        # the form was filled (typically the morning after), sleep_date
        # is the night it describes; using the wrong one silently
        # misattributes every medication flag by ~1 day, cohort-wide,
        # while still producing plausible-looking nonzero output (the
        # original bug reported 406 abstain-SOL nights despite being
        # wrong). Run automatically every pipeline execution now,
        # rather than only when specifically suspected — a diagnostic,
        # not a blocker, so a failure here is reported but does not
        # stop labelling from proceeding.
        # if med_ctx is not None:
        #     try:
        #         from check_medication_date_overlap import check_overlap
        #         check_overlap(night_record, med_ctx) #night_record
        #     except Exception as e_overlap:
        #         print(f"  Medication date-overlap check ERROR (labelling "
        #               f"will still proceed): {e_overlap}")

        # ── Verification Check ──
        # ── Robust Date Format & Overlap Check ──
        print("=" * 60)
        print(f"night_record index names: {night_record.index.names}")
        print(f"night_record columns:     {night_record.columns.tolist()}")

        # Flatten night_record to columns temporarily for verification
        nr_df = night_record.reset_index() if (
                    night_record.index.name or night_record.index.names[0] is not None) else night_record.copy()

        # Find the date column in night_record
        nr_date_col = next((c for c in ["night_date", "date", "sleep_date", "diary_date"] if c in nr_df.columns), None)
        nr_pid_col = next((c for c in ["participant_id", "pid"] if c in nr_df.columns), None)

        print(f"night_record date column detected: '{nr_date_col}'")
        print(f"night_record pid column detected:  '{nr_pid_col}'")

        if nr_date_col and nr_pid_col:
            nr_dates = pd.to_datetime(nr_df[nr_date_col], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
            med_dates = pd.to_datetime(med_ctx["night_date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")

            print(f"night_record sample dates: {nr_dates.dropna().iloc[:3].tolist()}")
            print(f"med_ctx sample dates:      {med_dates.dropna().iloc[:3].tolist()}")

            nr_keys = set(zip(nr_df[nr_pid_col].astype(str).str.strip(), nr_dates))
            med_keys = set(zip(med_ctx["participant_id"].astype(str).str.strip(), med_dates))

            overlap = nr_keys.intersection(med_keys)
            print(f"Alignment: {len(overlap)} / {len(nr_keys)} nights aligned with med_ctx")
        print("=" * 60)

        if "rar_p" in dir() and rar_p is not None:
            rar_dates = rar_p.reset_index()[
                "night_date"] if "night_date" in rar_p.reset_index().columns else rar_p.index
            print("rar_p sample dates:", pd.Series(rar_dates).dropna().iloc[:3].tolist())
        #------------end----------------------------------------------------------------------------------------
        print(f"What goes into labeller: Night_records - {len(night_record)}, med_ctx - {len(med_ctx)} RAR_p - {len(rar_p)}")
        result = labeler.fit_predict(night_record, med_ctx, rar_p)
        report = labeler.coverage_report(result)

        print(f"Nights labelled: {report['labelled_nights']} / {report['total_nights']} ({report['coverage_pct']}%)")
        print(f"Unclassifiable:  {report['unclassifiable']}")
        print(f"Label dist:      {report['label_distribution']}")
        for lf_name, votes in report['rule_votes'].items():
            print(f"  {lf_name:<25} {dict(sorted(votes.items()))}")

            # ── Merge nightly medication context into final_night_table ──
            # final_night_table is built earlier (Layer 1), before medication
            # context is joined — it has no medication columns. `result`
            # (Layer 3's labeler.fit_predict() output, just above) already has
            # them correctly joined (post sleep_date/diary_date fix — see
            # labeler.py's apply_rules()). Pulling from `result` here rather
            # than re-deriving the join a second time keeps a single source of
            # truth: whatever medication attribution Layer 3 actually used for
            # labelling is exactly what appears in this reporting table, with
            # no risk of the two joins silently diverging.
            # if "final_night_table" in dir() and final_night_table is not None:
            #     _med_cols = [c for c in result.columns if c.startswith("med_")]
            #     if _med_cols:
            #         _med_slice = result[["participant_id", "night_date"] + _med_cols].drop_duplicates(
            #             subset=["participant_id", "night_date"]
            #         )
            #         final_night_table = final_night_table.merge(
            #             _med_slice, on=["participant_id", "night_date"], how="left"
            #         )
            #         # nights present in final_night_table but not in result
            #         # (e.g. excluded before Layer 3 ran) get no medication
            #         # data — fill the boolean flags False rather than leave
            #         # NaN, consistent with apply_rules()'s own fillna(False)
            #         # convention for missing medication context
            #         for c in _med_cols:
            #             if final_night_table[c].dtype == object or final_night_table[c].dtype == bool:
            #                 final_night_table[c] = final_night_table[c].fillna(False)
            #         print(f"  Medication columns merged into final_night_table: "
            #               f"{_med_cols} ({len(_med_slice)} nights with medication "
            #               f"context)")
            #     else:
            #         print("  No med_ columns found in Layer 3's result — "
            #               "final_night_table medication merge skipped "
            #               "(check med_ctx was actually passed to labeler.fit_predict)")
            # ── Merge nightly medication context into final_night_table ──
            ####################======================##########################============================
            # if "final_night_table" in dir() and final_night_table is not None:
            #     _med_cols = [c for c in result.columns if c.startswith("med_")]
            #     if _med_cols:
            #         # 1. Drop existing med_ columns if present to avoid _x / _y duplication
            #         cols_to_drop = [c for c in _med_cols if c in final_night_table.columns]
            #         if cols_to_drop:
            #             final_night_table = final_night_table.drop(columns=cols_to_drop)
            #
            #         # 2. Extract slice and force string format on night_date
            #         _med_slice = result[["participant_id", "night_date"] + _med_cols].drop_duplicates(
            #             subset=["participant_id", "night_date"]
            #         ).copy()
            #         _med_slice["night_date"] = _med_slice["night_date"].astype(str)
            #
            #         # 3. Force string format on final_night_table.night_date BEFORE merging
            #         final_night_table["night_date"] = final_night_table["night_date"].astype(str)
            #
            #         # 4. Perform clean left-join
            #         final_night_table = final_night_table.merge(
            #             _med_slice, on=["participant_id", "night_date"], how="left"
            #         )
            #
            #         # 5. Fill unmapped nights as False/neutral
            #         for c in _med_cols:
            #             if final_night_table[c].dtype == object or final_night_table[c].dtype == bool:
            #                 final_night_table[c] = final_night_table[c].fillna(False)
            # ── Merge nightly medication context into final_night_table ──

#################============================================##########################################==============================
                    # print(f"  Medication columns merged into final_night_table: "
                    #       f"{_med_cols} ({len(_med_slice)} nights with medication context)")
        # ── Merge nightly medication context into final_night_table (SAFE) ──
        print("=" * 60)
        print(f"Rows in final_night_table: {len(final_night_table)}")
        print(f"Participants in final_night_table: {final_night_table['participant_id'].unique().tolist()}")

        print(f"Rows in result: {len(result)}")
        print(f"Participants in result: {result['participant_id'].unique().tolist()}")

        # Compare the exact (participant_id, night_date) keys
        keys_final = set(zip(final_night_table['participant_id'].astype(str).str.strip(),
                             final_night_table['night_date'].astype(str).str.strip()))
        keys_result = set(
            zip(result['participant_id'].astype(str).str.strip(), result['night_date'].astype(str).str.strip()))

        overlap = keys_final.intersection(keys_result)
        print(f"Matching key pairs: {len(overlap)} / {len(keys_final)}")
        print(f"Keys in final_night_table missing from result: {list(keys_final - keys_result)[:4]}")
        print(f"Keys in result missing from final_night_table: {list(keys_result - keys_final)[:4]}")
        print("=" * 60)
        ####################################################################
        print("=" * 60)
        print("INSPECTING `result` BEFORE MERGE FOR SP0000000:")
        sp_res = result[result["participant_id"] == "SP0000000"]
        print(sp_res[["participant_id", "night_date", "med_has_alpha2", "med_drug_classes"]].to_string())
        print("=" * 60)

        # ── Merge nightly medication context into final_night_table ──
        if "final_night_table" in dir() and final_night_table is not None and not final_night_table.empty:
            # Merge med_ctx into final_night_table
            _med_cols = [c for c in med_ctx.columns if c.startswith("med_")]
            if _med_cols:
                cols_to_drop = [c for c in _med_cols if c in final_night_table.columns]
                if cols_to_drop:
                    final_night_table = final_night_table.drop(columns=cols_to_drop)

                _med_slice = med_ctx.reset_index()[["participant_id", "night_date"] + _med_cols].drop_duplicates(
                    subset=["participant_id", "night_date"]
                )
                _med_slice["participant_id"] = _med_slice["participant_id"].astype(str).str.strip()
                _med_slice["night_date"] = _med_slice["night_date"].astype(str).str.strip()

                final_night_table["participant_id"] = final_night_table["participant_id"].astype(str).str.strip()
                final_night_table["night_date"] = final_night_table["night_date"].astype(str).str.strip()

                final_night_table = final_night_table.merge(
                    _med_slice, on=["participant_id", "night_date"], how="left"
                )

                for c in _med_cols:
                    if c == "med_drug_classes":
                        final_night_table[c] = final_night_table[c].apply(lambda x: x if isinstance(x, list) else [])
                    else:
                        final_night_table[c] = final_night_table[c].fillna(False).astype(bool)

        # sample night
        labelled = result[~result["all_abstained"]]
        if not labelled.empty:
            s = labelled.iloc[0]
            label_name = ks.phenotypes.name(int(s["hard_label"]))
            print(f"\nFirst labelled night: {s.get('night_date', '?')}")
            print(f"  hard_label: {int(s['hard_label'])} ({label_name})")
            votes = {k: int(v) for k, v in s.items() if k.startswith('vote_') and v != -1}
            print(f"  votes: {votes}")

    #     ###########diagnostic######################
    #     # --- R3 competition diagnostic — add below ---
    #     print("\n--- R3 coupling/misalign competition diagnostic ---")
    #     coupling_voted = result[result["vote_r3_coupling"] == 5]
    #     print(f"Coupling votes: {len(coupling_voted)}")
    #     print("Hard label distribution on coupling-voted nights:")
    #     print(coupling_voted["hard_label"].value_counts())
    #
    #     misalign_voted = result[result["vote_r3_misalign"] == 4]
    #     print(f"\nMisalign votes: {len(misalign_voted)}")
    #     print("Hard label distribution on misalign-voted nights:")
    #     print(misalign_voted["hard_label"].value_counts())
    #
    #     # co-firing nights — both R3 functions voted on the same night
    #     co_fire = result[
    #         (result["vote_r3_coupling"] == 5) &
    #         (result["vote_r3_misalign"] == 4)
    #         ]
    #     print(f"\nCo-firing nights (both R3 voted): {len(co_fire)}")
    #     print("Hard label on co-firing nights:")
    #     print(co_fire["hard_label"].value_counts())
    #     print("###############This is diag test 2#############")
    #     # nights where misalign OR logic fires but IS > 0.4
    #     misalign_surviving = result[
    #         (result["vote_r3_misalign"] == 4) &
    #         (result["IS"] > 0.4)
    #         ]
    #     print("Misalign nights surviving IS exclusion:", len(misalign_surviving))
    #
    #     # co-firing nights that would be resolved to coupling under Option 3
    #     resolved_to_coupling = co_fire[co_fire["IS"] <= 0.4]
    #     print("Co-firing nights resolved to coupling:", len(resolved_to_coupling))
    #
    #     print("##############this is diag 3#################")
    #     # nights where R1 voted NORMATIVE but hard label is RAR_MISALIGN
    #     normative_but_misalign = result[
    #         (result["vote_r1_normative_paradox"] == 0) &
    #         (result["hard_label"] == 4)
    #         ]
    #     print(f"Nights R1=NORMATIVE but label=RAR_MISALIGN: {len(normative_but_misalign)}")
    #     print(normative_but_misalign[["SE_pct", "WASO_min", "SQ", "IS",
    #                                   "L5_midpoint_mean"]].describe())
    #
    #     ############################################
    #
    except Exception as e:
        print(f"Layer 3 ERROR: {e}")
        import traceback
        traceback.print_exc()
    #
    # # ================================================================
    # # LAYER 4 — Neuro-symbolic fusion
    # # Comment out this block to skip
    # # ================================================================
    sep("Layer 4 — neuro-symbolic fusion")

    from nesy.fusion import RocketExtractor, EEGFeatureExtractor, FusionModel

    try:
        # build night index from labelled nights only
        if "result" not in dir():
            print("  Layer 3 must run first — skipping Layer 4")
        else:
            labelled_result = result[result["hard_label"] >= 0]
            night_index = list(zip(
                labelled_result["participant_id"],
                labelled_result["night_date"]
            ))
            print(f"Labelled nights:   {len(night_index)}")

            # clinical features — same variables used by symbolic rules
            # only added when --with-clinical flag is passed
            # default: ROCKET + EEG + GBTM only (original NeSy design)
            from nesy.fusion.model import FusionModel as _FM
            _tmp = _FM()
            if getattr(args, "with_clinical",
                       False):  # or getattr(args, "ablation", False): commented out leaking clinical_df
                clinical_df = _tmp.extract_clinical_features(labelled_result, night_record)
                if clinical_df is not None:
                    n_clin = clinical_df.notna().any(axis=1).sum()
                    print(f"Clinical features: {clinical_df.shape[1]} per night "
                          f"({n_clin}/{len(clinical_df)} with data)")
            else:
                clinical_df = None
                print("Clinical features: excluded (default — use --with-clinical to add)")

    #         # ROCKET features — use config window, not suggest_rocket_window()
    #         # suggest_rocket_window() returns the modal sleep window which can
    #         # be as short as 1h; we need the full nocturnal window for signal
            _rocket_cfg = getattr(cfg, "rocket", None)
            _ws = _rocket_cfg.window_start if _rocket_cfg else "22:00"
            _we = _rocket_cfg.window_end if _rocket_cfg else "08:00"
            _nk = _rocket_cfg.n_kernels if _rocket_cfg else 1000
            print(f"  extracting ROCKET features... (window {_ws}→{_we})")
            rocket = RocketExtractor(n_kernels=_nk,
                                     window_start=_ws,
                                     window_end=_we)
            rocket_df = rocket.fit_transform(ep_imp, night_index)
            print(f"ROCKET features:   {rocket_df.shape[1]} per night")
    #
            # EEG morphological features
            print("  extracting EEG features...")
            extractor = EEGFeatureExtractor()
            if not epochs.empty:
                sym_df = extractor.transform(
                    epochs.reset_index(), night_index
                )
                print("This is the 31 symbol")
                sym_df.reset_index().to_csv('debug_sym.csv', index=False)
                #print(sym_df.head(50))
                n_eeg = sym_df.notna().any(axis=1).sum()
                print(f"EEG features:      {sym_df.shape[1]} per night")
                print(f"Nights with EEG:   {n_eeg} / {len(night_index)}")
            else:
                sym_df = None
                print("EEG features:      none (no epoch files found)")
    #
            # context vectors — retained for API compat but not in MLP features
            # GBTM used only as constraint modifier
            if context_df is not None:
                ctx_df = context_df
            else:
                ctx_df = None
                print("Context vector:    GBTM not available (constraint modifier disabled)")

            # GBTM groups DataFrame for constraint modifier
            gbtm_groups_df = None
            if ctx_df is not None and "dominant_group" in ctx_df.columns:
                gbtm_groups_df = ctx_df[["dominant_group", "confidence"]].copy()
            # fit fusion model — LogReg with constraint loss

            ################getthing the best epoch size ####################################
            # n_pids = labelled_result["participant_id"].nunique()
            # if n_pids < 5:
            #     print(f"LOPO-CV:           skipping ({n_pids} participants — need ≥ 5)")
            # else:
            #     print(f"  running LOPO-CV across {n_pids} participants...")
            #
            #     # 1. Define the epoch variations you want to evaluate
            #     epoch_choices = [20, 40, 50, 70, 100]
            #     best_kappa = -1.0
            #     best_epochs = None
            #     best_cv_results = None
            #
            #     for candidate_epochs in epoch_choices:
            #         print(f"\n--- Testing Configuration: epochs={candidate_epochs} ---")
            #
            #         # 2. Re-initialize the model with the target epoch size
            #         candidate_model = FusionModel(
            #             hidden=64, dropout=0.3, lr=1e-3,
            #             epochs=candidate_epochs,  # <--- Dynamically updated
            #             batch_size=256, random_seed=42,
            #             constraint_config="configs/heuristics.toml",
            #             use_mlp=False, use_constraints=False,
            #             disabled_rules={},
            #         )
            #
            #         # 3. Run LOPO-CV for this specific choice
            #         cv = candidate_model.lopo_cv(
            #             rocket_df, sym_df, ctx_df, result,
            #             clinical_df=clinical_df,
            #             night_record=night_record if "night_record" in dir() else None,
            #             gbtm_groups_df=gbtm_groups_df,
            #         )
            #
            #         print(
            #             f"  Result for epochs={candidate_epochs} -> Accuracy: {cv['overall_accuracy']:.3f} | Kappa: {cv['cohen_kappa']:.3f}")
            #
            #         # 4. Save tracking data if Cohen's Kappa improves
            #         # (Kappa or Macro F1 are great choices for multi-class optimization)
            #         if cv['cohen_kappa'] > best_kappa:
            #             best_kappa = cv['cohen_kappa']
            #             best_epochs = candidate_epochs
            #             best_cv_results = cv
            #
            #     # 5. Output summary of the best discovered configuration
            #     print(f"\n=========================================")
            #     print(f" OPTIMAL EPOCH SELECTION COMPLETED")
            #     print(f"Best Epoch Size:  {best_epochs} (Cohen's kappa: {best_kappa:.3f})")
            #     print(f"=========================================")
            #     print(f"LOPO-CV results for best setting:")
            #     print(f"  Participants:    {best_cv_results['n_participants_evaluated']}")
            #     print(f"  Nights:          {best_cv_results['n_nights_evaluated']}")
            #     print(f"  Accuracy:        {best_cv_results['overall_accuracy']:.3f}")
            #     print(f"  Cohen kappa:     {best_cv_results['cohen_kappa']:.3f}")
            #     print(f"  Per-label F1:")
            #     for lbl, m in best_cv_results['per_label'].items():
            #         if m['support'] > 0:
            #             print(f"    {lbl}: F1={m['f1']:.3f}  (n={m['support']})")

            #############getthing the best epoch size#######################################

            print(f"  fitting LogReg fusion model...")

            model_v2 = FusionModel(
                hidden=64, dropout=0.3, lr=1e-3,
                epochs=40, batch_size=256, random_seed=42,
                constraint_config="configs/heuristics.toml",
                use_mlp=False,use_constraints=False,
                disabled_rules={},  # "c5_cross_modal"
            )

            rocket_save = None  # will be set after run_dir is created

            model_v2.fit(
                rocket_df, sym_df, ctx_df, result,
                clinical_df=clinical_df,
                night_record=night_record if "night_record" in dir() else None,
                gbtm_groups_df=gbtm_groups_df,
                save_rocket_matrix=rocket_save,
            )

            print(f"Fusion model:      LogReg fitted OK")

    #         # LOPO-CV (only if enough participants)
            n_pids = labelled_result["participant_id"].nunique()
            if n_pids < 5:
                print(f"LOPO-CV:           skipping ({n_pids} participants — need ≥ 5)")
            else:
                print(f"  running LOPO-CV across {n_pids} participants...")
                cv = model_v2.lopo_cv(
                    rocket_df, sym_df, ctx_df, result,
                    clinical_df=clinical_df,
                    night_record=night_record if "night_record" in dir() else None,
                    gbtm_groups_df=gbtm_groups_df,
                )
                print(f"LOPO-CV results:")
                print(f"  Participants:    {cv['n_participants_evaluated']}")
                print(f"  Nights:          {cv['n_nights_evaluated']}")
                print(f"  Accuracy:        {cv['overall_accuracy']:.3f}")
                print(f"  Cohen kappa:     {cv['cohen_kappa']:.3f}")
                print(f"  Per-label F1:")
                for lbl, m in cv['per_label'].items():
                    if m['support'] > 0:
                        print(f"    {lbl}: F1={m['f1']:.3f}  "
                              f"(n={m['support']})")


    #     # ── Ablation study — 2×2 design + baselines ─────────────────────────
        if getattr(args, "ablation", False) and "cv" in dir():
            import json as _json, datetime as _dt, time as _time
            from nesy.fusion.model import FusionModel as _FM2

            _abl_dir = Path("runs") / _dt.datetime.now().strftime(
                "%Y%m%d_%H%M%S_ablation"
            )
            _abl_dir.mkdir(parents=True, exist_ok=True)

            _nr_abl = night_record if "night_record" in dir() else None
            _gbtm_abl = gbtm_groups_df if "gbtm_groups_df" in dir() else None

            # clinical features for V4 upper bound
            _tmp_abl = _FM2(epochs=1)
            _clin_abl = _tmp_abl.extract_clinical_features(
                result[result["hard_label"] >= 0].reset_index(drop=True),
                _nr_abl
            )
    #

            def _run_abl_variant(use_mlp, use_constraints, feat_set, vname, use_lagrangian=True):
                _rdf = rocket_df
                _sdf = sym_df if feat_set in ("rocket_eeg",) else None
                _cdf = _clin_abl if feat_set == "clinical" else None
                _m = _FM2(
                    hidden=64, dropout=0.3, lr=1e-3,
                    epochs=40, batch_size=256, random_seed=42,  # epochs aligned with main pipeline
                    constraint_config="configs/heuristics.toml",
                    use_mlp=use_mlp, use_constraints=use_constraints,
                    use_lagrangian=use_lagrangian
                )
                t0 = _time.time()
                _res = _m.lopo_cv(
                    _rdf, _sdf, ctx_df, result,
                    clinical_df=_cdf, night_record=_nr_abl,
                    gbtm_groups_df=_gbtm_abl,
                )
                return _res, _time.time() - t0, _m
    #
            print(f"\nAblation — 2×2 design  (architecture × constraint loss)")
            print(f"{'Variant':<52} {'Arch':<8} {'L_c':<5} "
                  f"{'κ':>6}  {'Acc':>6}  {'Time':>6}")
            print("─" * 90)
    #
    #         # Ablation design: sequential component addition — LogReg throughout.
    #         # V3 (LogReg + Lagrangian γ) IS the proposed framework.
    #         # The main pipeline cv result IS V4 — no need to rerun separately.
    #         # V5 (rule inputs) is the information-asymmetry upper bound.
            _abl_results = {}
            if "cv" in dir():
                # Record main pipeline result as V3 for direct comparison
                _v2_key = "V2: ROCKET + EEG  · LogReg · L_data"
                _v2_result = dict(cv)
                _v2_result["variant"] = "LogReg+L_c(Lagrangian)"
                _v2_result["architecture"] = "LogReg"
                _v2_result["constraints"] = False
                _abl_results[_v2_key] = _v2_result

    #
    #         # Run V1, V4 separately to isolate each component's contribution.
            _abl_variants = [
                ("V1: ROCKET only  [baseline]",
                 False, False, "rocket", False),
                ("V4: Rule inputs only  [upper bound]",
                 False, False, "clinical", False),
            ]
    #
            _abl_models = {}  # keep fitted models for variants we want plausibility metrics on
    #         if "model_v4b" in dir():
    #             _abl_models["V4b"] = model_v4b
            for _vname, _use_mlp, _use_c, _feat, _use_lag in _abl_variants:
                try:
                    # use_lagrangian: False for V3 (fixed λ), True for all others
                    # Read directly from the tuple — do not override via string check
                    _res_v, _elapsed, _m_v = _run_abl_variant(
                        _use_mlp, _use_c, _feat, _vname[:4],
                        use_lagrangian=_use_lag
                    )
                    _arch = "MLP" if _use_mlp else "LogReg"
                    _cons = "yes" if _use_c else "no"
                    _note = "  ← upper bound" if "upper" in _vname else ""
                    print(f"{_vname:<52} {_arch:<8} {_cons:<5} "
                          f"{_res_v['cohen_kappa']:>6.3f}  "
                          f"{_res_v['overall_accuracy']:>6.3f}  "
                          f"{_elapsed:>5.0f}s{_note}")
                    _abl_results[_vname] = _res_v
                    if _vname.startswith("V1")  or _vname.startswith("V4"):
                        _abl_models[_vname[:2]] = _m_v  # "V2" / "V4" -> fitted model
                except Exception as _ev:
                    print(f"{_vname:<52} ERROR: {_ev}")


            with open(_abl_dir / "ablation_results.json", "w") as _f:
                _json.dump({
                    "ablation_study": _abl_results,
                    "ablation_design": {
                        "design": "Sequential component addition — LogReg throughout",
                        "note": "All variants use SleepLogReg. MLP is not part of this ablation.",
                        "V1": "LogReg · ROCKET only — actigraphy baseline",
                        "V2": "LogReg · ROCKET + EEG · L_data only [MAIN]",
                        "V4": "LogReg · rule inputs (diary + meds) — information asymmetry upper bound",
                    }
                }, _f, indent=2)
            print(f"Ablation saved → {_abl_dir}/ablation_results.json")
    #
    #
    #
    except Exception as e:
        print(f"Layer 4 ERROR: {e}")
        import traceback;
        traceback.print_exc()
    #
    # # ================================================================
    # # POST-PROCESSING — Stratification
    # # Comment out this block to skip
    # # ================================================================
    sep("Post-processing — stratification")

    from nesy.stratification.phenotype import StratificationLayer

    try:
        if "result" not in dir():
            print("  Layer 3 must run first — skipping stratification")
        else:
            strat = StratificationLayer(
                dominance_threshold=0.40,
                instability_threshold=1.5,
                compound_min_pct=0.25,
                margin_buffer=0.05,
                use_soft_probs=use_soft_probs
            )
            strat_df = strat.stratify(result,False)
            summary = strat.summary(strat_df)

            print(f"Participants:         {summary['total_participants']}")
            print(f"Dominant assigned:    {summary['dominant_assigned']}")
            print(f"Compound assigned:    {summary['compound_assigned']}")
            print(f"Unstable/mixed:       {summary['unstable_mixed']}")
            print(f"Indeterminate:        {summary['indeterminate']}")
            print(f"High instability:     {summary['high_instability_n']}")
            print(f"Mean instability:     {summary['mean_instability']:.3f}")
            print("\nY_Patient distribution:")
            for name, count in sorted(summary['y_patient_distribution'].items(),
                                      key=lambda x: -x[1]):
                print(f"  {name:<35} {count}")

            if len(strat_df) <= 10:
                print("\nPer-participant:")
                for pid, row in strat_df.iterrows():
                    print(f"  {pid}: {row['Y_Patient_name']:<30} "
                          f"pct={row['dominant_pct']:.1%}  "
                          f"entropy={row['instability_score']:.2f}  "
                          f"n={row['n_valid_nights']}")

    except Exception as e:
        print(f"Stratification ERROR: {e}")
        import traceback
        traceback.print_exc()
    import json, datetime
    run_dir = Path("runs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    # # ── Stratification Sensitivity Analysis ──────────────────────────
    if getattr(args, "strat_sensitivity", False):
        print("\n" + "─" * 60)
        print("Post-Processing Sensitivity Analysis: Stratification Grid")
        print("─" * 60)

        sens_df = run_stratification_sensitivity_grid(
            label_result=result,
            dom_thresholds=[0.35, 0.40, 0.45],
            comp_thresholds=[0.20, 0.25, 0.30],
            ent_thresholds=[1.3, 1.5, 1.7],
            margin_buffer=0.05,
            use_soft_probs=use_soft_probs
        )

        # Display key summary comparisons
        cols_to_show = [
            "dominance_thresh", "compound_thresh", "entropy_thresh",
            "dominant_assigned", "compound_assigned", "unstable_mixed", "indeterminate"
        ]
        print(sens_df[cols_to_show].to_string(index=False))

        # Save outputs
        sens_path = run_dir / "stratification_sensitivity.csv"
        sens_df.to_csv(sens_path, index=False)
        print(f"\nSensitivity grid saved → {sens_path.name}")

    # ================================================================
    # LAYER 5 — Validation
    # Comment out this block to skip
    # ================================================================
    sep("Layer 5 — validation")

    from nesy.validation.metrics import ValidationLayer

    try:
        validator = ValidationLayer()

        # Test 4 — coverage (always available)
        if "result" in dir():
            cov = validator.test_coverage(result, med_ctx if "med_ctx" in dir() else None)
            print(f"Test 4 — Coverage:")
            print(f"  Labelled:         {cov['labelled_nights']}/{cov['total_nights']} ({cov['labelled_pct']}%)")
            print(f"  Unclassifiable:   {cov['unclassifiable']} ({cov['unclassifiable_pct']}%)")
            print(f"  Med-abstain SOL:  {cov['med_abstain_sol']}")
            print(f"  Med-strengthen:   {cov['med_sol_strengthen']}")
            print(f"  Label dist:       {cov['label_distribution']}")

        # Test 1 — consistency (needs LOPO-CV from Layer 4)
        if "cv" in dir():
            cons = validator.test_consistency(cv)
            print("\nTest 1 — Internal consistency (LOPO-CV):")
            print(f"  Cohen kappa:      {cons['cohen_kappa']} ({cons['kappa_interpretation']})")
            print(f"  Accuracy:         {cons['overall_accuracy']}")
            print(f"  Per-label F1:")
            for lbl, f1 in cons["per_label_f1"].items():
                n = cons["per_label_support"][lbl]
                if n > 0:
                    print(f"    {lbl}: F1={f1:.3f} (n={n})")

        # Test 2 — biological plausibility
        # print(f"\nTest 2 — Biological plausibility:")
        # try:
        #     plaus = validator.test_plausibility(
        #         result["hard_label"].astype(int).values,
        #         result.reset_index(drop=True),
        #         ks.phenotypes.label_to_name,
        #     )
        #     if "error" in plaus:
        #         print(f"  {plaus['error']}")
        #     else:
        #         print(f"  Nights evaluated:       {plaus['n_nights_evaluated']}")
        #         print(f"  Plausibility rate:      {plaus['plausibility_rate']:.4f} "
        #               f"({100 - plaus['violations_pct']:.1f}% coherent)")
        #         print(f"  Total violations:       {plaus['n_violations']} "
        #               f"({plaus['violations_pct']:.1f}%)")
        #         if plaus['P1_high_WASO_healthy'] >= 0:
        #             print(f"  P1 high WASO→healthy:   {plaus['P1_high_WASO_healthy']}")
        #         if plaus['P2_low_N3_healthy'] >= 0:
        #             print(f"  P2 low N3→healthy:      {plaus['P2_low_N3_healthy']}")
        #         if plaus['P3_normal_EEG_eeg_patho'] >= 0:
        #             print(f"  P3 normal EEG→patho:    {plaus['P3_normal_EEG_eeg_patho']}")
        # except Exception as _pe:
        #     print(f"  Plausibility test error: {_pe}")

        # Test 3 — fairness (needs demographics + stratification)
        if "strat_df" in dir() and demo_files:
            demo = load_demographics(demo_files[0])
            fair = validator.test_fairness(strat_df, demo)
            print(f"\nTest 3 — Fairness:")
            for stratum, res in fair["strata"].items():
                if "error" in res:
                    print(f"  {stratum}: {res['error']}")
                else:
                    sig = "⚠ significant" if res["significant"] else "OK"
                    print(f"  {stratum}: chi2={res['chi2']:.3f}  "
                          f"p={res['p_value']:.3f}  "
                          f"V={res['cramers_v']:.3f}  {sig}")

            # within-family concordance — ASD vs sibling pairs
            wf = validator.test_within_family_concordance(strat_df, demo)
            print(f"\nWithin-family concordance (ASD vs sibling):")
            if "error" in wf:
                print(f"  {wf['error']}")
            else:
                print(f"  Complete pairs:           {wf['n_complete_pairs']}")
                print(f"  Concordant:               {wf['n_concordant']} "
                      f"({100 * wf['phenotype_concordance']:.1f}%)")
                print(f"  Chance rate:              {100 * wf['chance_rate']:.1f}%")
                print(f"  Binomial p:               {wf['concordance_p_binomial']:.4f}")
                print(f"  95% CI:                   "
                      f"[{wf['concordance_ci_95'][0]:.3f}, "
                      f"{wf['concordance_ci_95'][1]:.3f}]")
                print(f"  Phenotype dist chi2:      {wf['phenotype_dist_chi2']:.3f}  "
                      f"p={wf['phenotype_dist_p']:.4f}  "
                      f"V={wf['phenotype_dist_cramers_v']:.3f}")
                print(f"  ASD mean instability:     {wf['asd_mean_instability']:.3f}")
                print(f"  Sibling mean instability: {wf['sibling_mean_instability']:.3f}")
                print(f"  Difference (ASD−sibling): {wf['instability_difference']:+.3f}")
                print(f"  Instability Mann-Whitney: p={wf['instability_mannwhitney_p']:.4f}  "
                      f"r={wf['instability_rank_biserial_r']:.3f}")
                print(f"  ASD phenotypes:           {wf['asd_phenotype_dist']}")
                print(f"  Sibling phenotypes:       {wf['sibling_phenotype_dist']}")
                if len(wf['pairs']) <= 10:
                    print(f"\n  Per-family pairs:")
                    for p in wf['pairs']:
                        match = "✓" if p['same_phenotype'] else "✗"
                        print(f"    {p['family_id']}: ASD={p['asd_phenotype']:<30} "
                              f"sib={p['sibling_phenotype']:<30} {match}")

        # Sensitivity analyses
        if "result" in dir() and "med_ctx" in dir():
            print("med_has_stimulant sum:",
                  result['med_has_stimulant'].sum() if 'med_has_stimulant' in result.columns else "COLUMN MISSING")
            print("med_should_abstain_sol sum:", result[
                'med_should_abstain_sol'].sum() if 'med_should_abstain_sol' in result.columns else "COLUMN MISSING")
            print("med_has_melatonin sum:",
                  result['med_has_melatonin'].sum() if 'med_has_melatonin' in result.columns else "COLUMN MISSING")
            sa = validator.sensitivity_analyses(result, med_ctx, None)
            print(f"\nSensitivity analyses:")
            for name, data in sa["analyses"].items():
                dist = data["label_dist"]
                print(f"  {name}: n={data['n_nights']}")

    except Exception as e:
        print(f"Validation ERROR: {e}")
        import traceback;
        traceback.print_exc()

    # ================================================================
    # SUMMARY + SAVE REPORT
    # ================================================================
    sep("Summary")
    print(f"Participants:    {len(pids)}")
    print(f"Nights total:    {n_total}")
    print(f"Diary coverage:  {100 * (n_total - n_missing) / n_total:.1f}%")
    if 'report' in dir():
        print(f"Layer 3 coverage:{report['coverage_pct']}%")
        print(f"Label dist:      {report['label_distribution']}")
    print("─" * 55)

    # ── Save pipeline report to disk ──────────────────────────────────

    # run_dir = Path("runs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_dir.mkdir(parents=True, exist_ok=True)

    # save IV calibration values now that run_dir exists (was ROCKET
    # fragmentation scores / rocket_frag_scores.npy — renamed along with
    # the frag_score -> IV switch, so the saved file's name matches its
    # actual contents rather than describing a ROCKET-derived quantity
    # that no longer exists in this pipeline)
    if "model" in dir() and hasattr(model, "_iv_values") and model._iv_values is not None:
        import numpy as _np
        _np.save(run_dir / "iv_calibration_values.npy", model._iv_values)
        print(f"  IV calibration values saved → iv_calibration_values.npy")

    run_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "participants": len(pids),
        "nights_total": int(n_total),
        "diary_coverage_pct": round(100 * (n_total - n_missing) / n_total, 1),
    }

    if "report" in dir():
        run_report["layer3"] = {
            "labelled_nights": report["labelled_nights"],
            "total_nights": report["total_nights"],
            "coverage_pct": report["coverage_pct"],
            "unclassifiable": report["unclassifiable"],
            "label_distribution": report["label_distribution"],
        }

    if "cv" in dir():
        run_report["layer4_lopo_cv"] = {
            "n_participants": cv["n_participants_evaluated"],
            "n_nights": cv["n_nights_evaluated"],
            "accuracy": cv["overall_accuracy"],
            "cohen_kappa": cv["cohen_kappa"],
            "per_label_f1": {k: v["f1"] for k, v in cv["per_label"].items()
                             if v["support"] > 0},
        }

    if "summary" in dir() and "strat_df" in dir():
        run_report["stratification"] = {
            "total_participants": summary["total_participants"],
            "dominant_assigned": summary["dominant_assigned"],
            "compound_assigned": summary["compound_assigned"],
            "unstable_mixed": summary["unstable_mixed"],
            "high_instability_n": summary["high_instability_n"],
            "mean_instability": summary["mean_instability"],
            "y_patient_distribution": summary["y_patient_distribution"],
        }
        # enrich Y_Patient with demographics and family_id for Chapter 5 merge
        strat_out = strat_df.copy()
        if demo_files:
            demo_save = load_demographics(demo_files[0])
            strat_out = strat_out.join(
                demo_save[["age", "age_band", "sex_encoded", "autism_status", "family_id"]],
                how="left"
            )
        strat_out.to_csv(run_dir / "y_patient.csv")

    #     # ── Layer 4 Y_patient CSVs — V2, V4, V4b, V5 ──────────────────
    #     # Same StratificationLayer thresholds, same demographic
    #     # enrichment, but derived from each model's own held-out
    #     # predictions instead of Layer 3's hard_label. V3 deliberately
    #     # excluded — its fixed-λ constraint calibration bug was only
    #     # just fixed and hasn't been re-validated at this level yet.
        try:
            from layer4_stratification import stratify_from_model_predictions
    #
            _l4_save_targets = []
            # if "model_v2" in dir() and hasattr(model_v2, "_last_lopo_pids"):
            #     _l4_save_targets.append((model_v2, "V2"))
            if "model_v2" in dir() and hasattr(model_v2, "_last_lopo_pids"):
                _l4_save_targets.append((model_v2, "V2"))
            # if "model_v4b" in dir() and hasattr(model_v4b, "_last_lopo_pids"):
            #     _l4_save_targets.append((model_v4b, "V4b"))
            if ("_abl_models" in dir() and "V1" in _abl_models
                    and hasattr(_abl_models["V1"], "_last_lopo_pids")):
                _l4_save_targets.append((_abl_models["V1"], "V1"))
            if ("_abl_models" in dir() and "V4" in _abl_models
                    and hasattr(_abl_models["V4"], "_last_lopo_pids")):
                _l4_save_targets.append((_abl_models["V4"], "V4"))

    #
            for _mt, _vname_save in _l4_save_targets:
                try:
                    strat_df_L4_save = stratify_from_model_predictions(
                        _mt._last_lopo_pids, _mt._last_lopo_Y_pred, strat, use_soft_probs
                    )
                    strat_out_L4 = strat_df_L4_save.copy()
                    if demo_files:
                        strat_out_L4 = strat_out_L4.join(
                            demo_save[["age", "age_band", "sex_encoded",
                                       "autism_status", "family_id"]],
                            how="left"
                        )
                    _out_path = run_dir / f"y_patient_L4_{_vname_save}.csv"
                    strat_out_L4.to_csv(_out_path)
                    print(f"  Saved Layer 4 Y_patient ({_vname_save}) "
                          f"→ {_out_path.name}")

                    ####################night prediction here###############################
                    # ── Save Layer 4 ablation per-night predictions (held-out LOPO) ──
                    _night_save_targets = []
                    if "model_v2" in dir() and hasattr(model_v2, "_last_lopo_pids"):
                        _night_save_targets.append((model_v2, "V2"))
                    if "_abl_models" in dir():
                        for _tag, _m_instance in _abl_models.items():
                            if hasattr(_m_instance, "_last_lopo_pids"):
                                _night_save_targets.append((_m_instance, _tag))

                    labelled_nights_ref = result[result["hard_label"] >= 0].reset_index(drop=True)

                    for _model_obj, _vtag in _night_save_targets:
                        try:
                            held_dates = labelled_nights_ref.iloc[_model_obj._last_lopo_idx]["night_date"].values
                            probs = _model_obj._last_lopo_Y_pred.numpy()

                            df_night_preds = pd.DataFrame({
                                "participant_id": _model_obj._last_lopo_pids,
                                "night_date": held_dates,
                                "true_label": _model_obj._last_lopo_y_true,
                                "pred_label": probs.argmax(axis=1),
                            })

                            # Add individual class probabilities (prob_label_0 to prob_label_5)
                            for i in range(probs.shape[1]):
                                df_night_preds[f"prob_label_{i}"] = probs[:, i]

                            out_file = run_dir / f"night_predictions_L4_{_vtag}.csv"
                            df_night_preds.to_csv(out_file, index=False)
                            print(f"  Saved Layer 4 night predictions ({_vtag}) → {out_file.name}")
                        except Exception as e_abl_night:
                            print(f"  Night predictions (Layer 4, {_vtag}) save ERROR: {e_abl_night}")
                    #######################################################################

                except Exception as e_save_one:
                    print(f"  Y_patient (Layer 4, {_vname_save}) save ERROR: {e_save_one}")
        except Exception as e_save_l4:
            print(f"  Layer 4 Y_patient CSV saving skipped: {e_save_l4}")
    #
    # if "cov" in dir():
    #     run_report["layer5_coverage"] = cov
    #
    # # save JSON report
    report_path = run_dir / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(run_report, f, indent=2, default=str)

    # save final convergence tables (built in Layer 1, above)
    if "final_night_table" in dir() and final_night_table is not None:
        final_night_table.to_csv(run_dir / "final_night_table.csv", index=False)
        final_participant_table.to_csv(run_dir / "final_participant_table.csv", index=False)

    # save night-level labels if available
    if "result" in dir():
        label_cols = ["participant_id", "night_date", "hard_label", "all_abstained",
                      "vote_r1_conditioned_arousal", "vote_r1_normative_paradox",
                      "vote_r2_hypersomnia", "vote_r3_coupling", "vote_r3_misalign",
                      "vote_r3_conditioned_arousal"]
        label_cols = [c for c in label_cols if c in result.columns]
        result[label_cols].to_csv(run_dir / "night_labels.csv", index=False)
        run_dir = Path("runs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── Insert Prediction Exports Here ──
        # if "cv" in dir() and hasattr(model_v2, "_last_lopo_Y_pred"):
        #     held_dates = result.iloc[model_v2._last_lopo_idx]["night_date"].values
        #     lopo_probs = model_v2._last_lopo_Y_pred.numpy()
        #     lopo_df = pd.DataFrame({
        #         "participant_id": model_v2._last_lopo_pids,
        #         "night_date": held_dates,
        #         "true_label": model_v2._last_lopo_y_true,
        #         "pred_label": lopo_probs.argmax(axis=1),
        #     })
        #     for i in range(lopo_probs.shape[1]):
        #         lopo_df[f"prob_label_{i}"] = lopo_probs[:, i]
        #     lopo_df.to_csv(run_dir / "lopo_per_night_predictions.csv", index=False)
        #     print(f"  LOPO-CV predictions saved → lopo_per_night_predictions.csv")
    #
    # # save the V4b pre-training reconciliation detail — every night whose
    # # probability mass or hard label was touched, with the rule(s) that
    # # fired and the underlying WASO/IV/SE/TST values that drove it. This
    # # is the "which specific nights, and why" answer — the aggregate
    # # counts alone (printed during the run) don't let you inspect
    # # individual cases by hand.
    # if "_recon_diag" in dir() and "night_detail" in _recon_diag:
    #     _recon_diag["night_detail"].to_csv(
    #         run_dir / "reconciliation_detail.csv", index=False
    #     )
    #     print(f"  V4b reconciliation detail saved → reconciliation_detail.csv "
    #           f"({len(_recon_diag['night_detail'])} nights)")
    #
    # # save within-family concordance
    # if "strat_df" in dir() and demo_files:
    #     try:
    #         from nesy.validation.metrics import ValidationLayer as _VL
    #         _val = _VL()
    #         demo_wf = load_demographics(demo_files[0])
    #         wf = _val.test_within_family_concordance(strat_df, demo_wf)
    #         if "error" not in wf:
    #             wf_pairs = pd.DataFrame(wf["pairs"])
    #             wf_pairs.to_csv(run_dir / "within_family_pairs.csv", index=False)
    #             run_report["within_family_concordance"] = {
    #                 "n_complete_pairs": wf["n_complete_pairs"],
    #                 "phenotype_concordance_pct": round(wf["phenotype_concordance"] * 100, 1),
    #                 "asd_mean_instability": wf["asd_mean_instability"],
    #                 "sibling_mean_instability": wf["sibling_mean_instability"],
    #                 "instability_difference": wf["instability_difference"],
    #                 "asd_phenotype_dist": wf["asd_phenotype_dist"],
    #                 "sibling_phenotype_dist": wf["sibling_phenotype_dist"],
    #             }
    #             with open(report_path, "w") as f:
    #                 json.dump(run_report, f, indent=2, default=str)
    #     except Exception as e:
    #         print(f"  Within-family save: {e}")
    #
    # save GBTM group assignments
    if "context_df" in dir() and context_df is not None:
        # include group probabilities, dominant group, confidence, and D4 metadata
        gbtm_cols = ["dominant_group", "confidence", "n_nights"]
        prob_cols = [c for c in context_df.columns if c.startswith("group_prob_")]
        d4_cols = [c for c in ["age_band", "sex_encoded", "autism_status"]
                   if c in context_df.columns]
        save_cols = prob_cols + gbtm_cols + d4_cols
        save_cols = [c for c in save_cols if c in context_df.columns]
        # also add family_id to GBTM output
        gbtm_out = context_df[save_cols].copy()
        if demo_files:
            demo_g = load_demographics(demo_files[0])
            gbtm_out = gbtm_out.join(demo_g[["family_id"]], how="left")
        gbtm_out.to_csv(run_dir / "gbtm_groups.csv")
        # also add GBTM info to JSON report
        run_report["gbtm"] = {
            "n_participants": len(context_df),
            "n_groups": len(prob_cols),
            "group_counts": context_df["dominant_group"].value_counts().to_dict(),
            "mean_confidence": round(float(context_df["confidence"].mean()), 3),
        }
        # re-save JSON with GBTM section
        with open(report_path, "w") as f:
            json.dump(run_report, f, indent=2, default=str)

    # ── Generate participant PDF reports ─────────────────────────────────────
    if getattr(args, "report", False):
        try:
            from nesy.reporting.report import generate_cohort_reports
            print("\n  Generating participant reports...")
            generate_cohort_reports(
                run_dir=run_dir,
                night_labels=result if "result" in dir() else pd.DataFrame(),
                y_patient=strat_df if "strat_df" in dir() else pd.DataFrame(),
                night_record=night_record if "night_record" in dir() else None,
                med_ctx=med_ctx if "med_ctx" in dir() else None,
                rar_p=rar_p if "rar_p" in dir() else pd.DataFrame(),
                demo=load_demographics(demo_files[0])
                if demo_files else pd.DataFrame(),
                gbtm=context_df if "context_df" in dir() else None,
            )
        except Exception as e:
            print(f"  Report generation failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nReport saved to: {run_dir}/")
    print(f"  pipeline_report.json     — all summary metrics")
    print(f"  night_labels.csv         — per-night label assignments + rule votes")
    if "strat_df" in dir():
        print(f"  y_patient.csv            — Y_Patient + demographics + family_id")
    if "context_df" in dir() and context_df is not None:
        print(f"  gbtm_groups.csv          — GBTM groups + demographics + family_id")


if __name__ == "__main__":
    args = parse_args()
    run(args)

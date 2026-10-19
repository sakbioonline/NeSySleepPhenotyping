"""
run_pipeline.py — runs the NeSy pipeline end to end.

Usage
-----
    python run_pipeline.py --config configs/pipeline.yaml
    python run_pipeline.py --config configs/pipeline.yaml --participant SP0013553
    python run_pipeline.py --config configs/pipeline.yaml --resume runs/ssp_01

Outputs go to runs/<run_id>/ with one sub-folder per layer.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="NeSy sleep phenotyping pipeline")
    parser.add_argument("--config", required=True, help="path to pipeline.yaml")
    parser.add_argument("--participant", default=None, help="run on one participant only")
    parser.add_argument("--resume", default=None, help="path to existing run folder to resume")
    parser.add_argument("--fmt", default=None, choices=["parquet", "csv"], help="override output format")
    args = parser.parse_args()

    # load config
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from nesy.config import load_config
    cfg = load_config(args.config)

    fmt = args.fmt or cfg.run.fmt

    # set up run folder
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            print(f"resume path not found: {run_dir}")
            sys.exit(1)
        run_id = run_dir.name
    else:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path(cfg.run.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def done(step):
        return step in manifest

    def mark_done(step, **meta):
        manifest[step] = {"completed_at": datetime.now(timezone.utc).isoformat(), **meta}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  ✓ {step}")

    def save(step, name, df):
        p = run_dir / step / f"{name}.{fmt}"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=True) if fmt == "parquet" else df.to_csv(p, index=True)
        return p

    def load(step, name):
        p = run_dir / step / f"{name}.{fmt}"
        return pd.read_parquet(p) if fmt == "parquet" else pd.read_csv(p, index_col=0)

    print(f"\nNeSy pipeline — run {run_id}")
    print(f"Cohort:  {cfg.cohort.name}")
    print(f"Data:    {cfg.cohort.data_root}")
    print(f"Output:  {run_dir}\n")

    # ── Layer 1: ingestion ────────────────────────────────────────────
    if done("ingestion"):
        print("  ↷ ingestion (already done)")
    else:
        print("  → ingestion")
        from nesy.ingestion import (
            load_embraceplus_directory, load_dreem_summary,
            load_dreem_hypnogram_directory, load_sleep_diary,
            aggregate_embraceplus_daily, align_modalities,
            alignment_reports_to_dataframe,
        )

        ep = load_embraceplus_directory(cfg.cohort.embraceplus_path)
        dreem = load_dreem_summary(cfg.cohort.dreem_path / "dreem_summary.csv")
        diary = load_sleep_diary(cfg.cohort.diary_path / "ssp_sleep_diary.csv")

        if args.participant:
            ep = ep[ep["participant_id"] == args.participant]
            dreem = dreem.loc[[args.participant]]
            diary = diary.loc[[args.participant]]

        epochs, align_reports = load_dreem_hypnogram_directory(
            cfg.cohort.dreem_path,
            alignment_config={"pre_aligned": False, "min_overlap_sec": 15},
        )

        ep_daily = aggregate_embraceplus_daily(ep)
        night_record = align_modalities(dreem, diary, ep_daily)

        save("ingestion", "night_record", night_record.reset_index())
        save("ingestion", "dreem_epochs_aligned", epochs.reset_index())
        save("ingestion", "ep_timeseries", ep.reset_index())
        if align_reports:
            save("ingestion", "alignment_reports",
                 alignment_reports_to_dataframe(align_reports))

        mark_done("ingestion",
                  participants=ep["participant_id"].nunique(),
                  nights=len(night_record))

    # ── Layer 2: knowledge ────────────────────────────────────────────
    if done("knowledge"):
        print("knowledge (already done)")
    else:
        print("knowledge")
        from nesy.ingestion import (
            compute_rar_daily_features, compute_rar_participant_features,
            impute_non_wear,
        )
        from nesy.knowledge import (
            load_knowledge_store, run_gbtm,
            context_vectors_to_dataframe, add_demographic_context,
            suggest_rocket_window, parse_medication_column,
        )
        from nesy.ingestion import load_demographics

        ep = load("ingestion", "ep_timeseries")
        ep["datetime"] = pd.to_datetime(ep["datetime"])
        ep = ep.set_index("datetime")

        night_record = load("ingestion", "night_record")
        diary_raw = pd.read_csv(cfg.cohort.diary_path / "ssp_sleep_diary.csv")

        # impute non-wear before RAR
        ep_imp, _ = impute_non_wear(ep)

        rar_daily = compute_rar_daily_features(ep_imp, impute=False)
        rar_participant = compute_rar_participant_features(ep_imp, impute=False)

        # GBTM — trajectory clustering from HR + RMSSD
        vectors, model = run_gbtm(ep_imp,
                                  n_groups=cfg.run.__dict__.get("n_groups", 4),
                                  random_seed=cfg.run.random_seed)
        context_df = context_vectors_to_dataframe(vectors)

        # D4 metadata — merge age band, sex, autism status into context vector
        demo_dir = cfg.cohort.data_root / "demographics"
        demo_files = list(demo_dir.glob("*.csv")) if demo_dir.exists() else []
        if demo_files:
            demo = load_demographics(demo_files[0])
            context_df = add_demographic_context(context_df, demo)
            print(f"    D4 metadata merged: {len(demo)} participants")
        else:
            print(f"    WARNING: no demographics file found in {demo_dir} — D4 fields will be -1")

        # medication context
        med_context = parse_medication_column(diary_raw, cfg.kb_path)

        # suggest ROCKET window — researcher should review and set in pipeline.yaml
        window = suggest_rocket_window(ep_imp)
        print(f"    suggested ROCKET window: {window['window_start']} → {window['window_end']}")

        save("knowledge", "rar_daily", rar_daily.reset_index())
        save("knowledge", "rar_participant", rar_participant.reset_index())
        save("knowledge", "context_vectors", context_df.reset_index())
        save("knowledge", "medication_context", med_context.reset_index())

        mark_done("knowledge",
                  participants=rar_participant.shape[0],
                  gbtm_groups=model.n_groups)

    print(f"\nDone. Outputs in {run_dir}")


if __name__ == "__main__":
    main()

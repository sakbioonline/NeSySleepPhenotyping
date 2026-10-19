import itertools
import pandas as pd
from nesy.stratification.phenotype import StratificationLayer


def run_stratification_sensitivity_grid(
        label_result: pd.DataFrame,
        dom_thresholds=(0.35, 0.40, 0.45),
        comp_thresholds=(0.20, 0.25, 0.30),
        ent_thresholds=(1.3, 1.5, 1.7),
        margin_buffer=0.05,
        use_soft_probs=False
) -> pd.DataFrame:
    """
    Executes grid sensitivity analysis over post-processing decision thresholds.
    Evaluates cohort stability, unclassifiable rates, and compound assignment.
    """
    records = []
    grid = itertools.product(dom_thresholds, comp_thresholds, ent_thresholds)

    for dom_th, comp_th, ent_th in grid:
        strat_layer = StratificationLayer(
            dominance_threshold=dom_th,
            compound_min_pct=comp_th,
            instability_threshold=ent_th,
            margin_buffer=margin_buffer,
            use_soft_probs=use_soft_probs
        )
        strat_df = strat_layer.stratify(label_result)
        summary = strat_layer.summary(strat_df)

        y_dist = summary["y_patient_distribution"]

        row = {
            "dominance_thresh": dom_th,
            "compound_thresh": comp_th,
            "entropy_thresh": ent_th,
            "dominant_assigned": summary["dominant_assigned"],
            "compound_assigned": summary["compound_assigned"],
            "unstable_mixed": summary["unstable_mixed"],
            "indeterminate": summary["indeterminate"],
            "high_instability_n": summary["high_instability_n"],
            "mean_instability": summary["mean_instability"],
            # Distribution tracking
            "normative_n": y_dist.get("Normative_range", 0),
            "cond_arousal_n": y_dist.get("Conditioned_arousal", 0),
            "rar_misalign_n": y_dist.get("RAR_misalignment", 0),
            "coupling_imp_n": y_dist.get("Coupling_impairment", 0),
        }
        records.append(row)

    return pd.DataFrame(records)
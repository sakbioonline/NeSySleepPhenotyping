"""
Validation layer (Layer 5).

Four prespecified validation tests from the manuscript:

Test 1 — Internal consistency
    Agreement between Snorkel label model outputs and fusion model
    predictions, quantified via Cohen's kappa and per-label F1.
    Uses LOPO-CV output from FusionModel.lopo_cv().

Test 2 — Fairness
    Phenotype distributions across sex and age-band strata.
    Chi-square test for differential label distribution across groups.
    Flags strata where one phenotype is over- or under-represented
    relative to the overall distribution.

Test 3 — Label coverage
    Proportion of participant-nights receiving each outcome:
        - definitive label (Snorkel or fusion model)
        - medication-confounded abstention
        - unclassifiable (all LFs abstained)

Three sensitivity analyses:
    SA1 — include medication-confounded nights in primary analysis
    SA2 — stimulant-free nights only
    SA3 — melatonin + elevated SOL nights (treatment-refractory subgroup)
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score, classification_report

N_LABELS = 6
# LABEL_NAMES = {
#     0: "healthy",
#     1: "paradoxical_insomnia",
#     2: "psychophysiological_insomnia",
#     3: "restless",
#     4: "rar_misalignment",
#     5: "hypersomnolence",
#     6: "sdb_risk",
#     7: "coupling_impairment",
# }

LABEL_NAMES = {
    0: "Normative_range",
    1: "Paradoxical_insomnia",
    2: "Conditioned_arousal",
    3: "Hyper-somnolence",
    4: "RAR_misalignment",
    5: "Coupling_impairment",
}


class ValidationLayer:

    def __init__(self, alpha=0.05):
        self.alpha = alpha  # significance threshold for fairness tests

    # ------------------------------------------------------------------
    # Test 1 — Internal consistency
    # ------------------------------------------------------------------
    def test_consistency(self, lopo_cv_result):
        """
        Summarise LOPO-CV agreement metrics from FusionModel.lopo_cv().
        Adds interpretation flags for the paper.
        """
        kappa = lopo_cv_result["cohen_kappa"]
        accuracy = lopo_cv_result["overall_accuracy"]

        # kappa interpretation bands (Landis & Koch 1977)
        if kappa < 0.0:
            band = "less than chance"
        elif kappa < 0.20:
            band = "slight"
        elif kappa < 0.40:
            band = "fair"
        elif kappa < 0.60:
            band = "moderate"
        elif kappa < 0.80:
            band = "substantial"
        else:
            band = "almost perfect"

        per_label = lopo_cv_result["per_label"]
        supported = {k: v for k, v in per_label.items() if v["support"] > 0}

        return {
            "test": "internal_consistency",
            "cohen_kappa": round(kappa, 4),
            "kappa_interpretation": band,
            "overall_accuracy": round(accuracy, 4),
            "n_participants": lopo_cv_result["n_participants_evaluated"],
            "n_nights": lopo_cv_result["n_nights_evaluated"],
            "per_label_f1": {k: v["f1"] for k, v in supported.items()},
            "per_label_support": {k: v["support"] for k, v in supported.items()},
        }

    # ------------------------------------------------------------------
    # Test 3 — Fairness
    # ------------------------------------------------------------------

    def test_plausibility(self, y_pred_labels, obs, label_names):
        """
        Test biological plausibility of night-level predictions.

        Three rules:
          P1: Very high WASO (>40 min) → P(healthy) should be near zero.
              Threshold aligned with R1's WASO_healthy_max (40 min early
              adolescence / 45 min late adolescence). Nights with WASO
              30–40 min may legitimately receive healthy labels when SE
              is high — these are not violations.

          P2: Low N3% (<13%) → P(healthy) should be near zero.
              Below late-adolescence normal_min (15%) with buffer.

          P3: Normal EEG (WASO ≤25 min AND N3% ≥15%) →
              P(EEG-grounded pathological phenotype) should be low.

              SCOPE: P3 applies only to:
                psychophysiological_insomnia — arousal-driven insomnia;
                  elevated SOL/ANX with normal architecture is a violation
                restless — requires elevated N1 + WASO by definition
                hypersomnolence — requires excessive TST
                sdb_risk — requires low N3 + high WASO

              P3 does NOT apply to:
                paradoxical_insomnia — normal EEG IS the defining feature
                  (good objective sleep, poor subjective experience).
                  Predicting paradoxical insomnia on a normal-EEG night
                  is clinically correct, not a violation.
                coupling_impairment — defined by RAR metrics, not EEG.
                rar_misalignment — defined by RAR metrics.
                healthy — normal EEG is evidence FOR healthy.
        """
        import numpy as np

        # ── Thresholds ────────────────────────────────────────────────
        WASO_P1 = 40.0  # min — high WASO incompatible with normative assignment
        N3_P2 = 13.0  # % TST — low N3 incompatible with normative assignment
        # WASO_NORMAL_MAX = 25.0  # min — for P3 normal EEG definition
        # N3_NORMAL_MIN = 15.0  # % TST — for P3 normal EEG definition

        # P3 scope — EEG-grounded pathological phenotypes only
        # paradoxical_insomnia deliberately excluded (see docstring)
        # EEG_GROUNDED_PATHOLOGICAL = {
        #     'psychophysiological_insomnia',
        #     'restless',
        #     'hypersomnolence',
        #     'sdb_risk',
        # }

        # Invert label_names for lookup
        name_to_label = {v: k for k, v in label_names.items()}
        healthy_label = name_to_label.get('Normative_range', 0)
        # eeg_patho_labels = {
        #     name_to_label[n] for n in EEG_GROUNDED_PATHOLOGICAL
        #     if n in name_to_label
        # }

        y = np.array(y_pred_labels, dtype=int)
        waso = np.array(obs['WASO_min'].values, dtype=float)
        n3 = np.array(obs['N3_pct'].values, dtype=float)

        # Mask out nights where EEG features are missing
        has_eeg = ~(np.isnan(waso) | np.isnan(n3))
        n_evaluated = int(has_eeg.sum())

        # P1: very high WASO → predicted healthy
        p1_mask = has_eeg & (waso > WASO_P1) & (y == healthy_label)
        p1_violations = int(p1_mask.sum())

        # P2: low N3 → predicted healthy
        p2_mask = has_eeg & (n3 < N3_P2) & (y == healthy_label)
        p2_violations = int(p2_mask.sum())

        # P3: normal EEG → predicted EEG-grounded pathological
        # normal_eeg = has_eeg & (waso <= WASO_NORMAL_MAX) & (n3 >= N3_NORMAL_MIN)
        # p3_mask = normal_eeg & np.isin(y, list(eeg_patho_labels))
        # p3_violations = int(p3_mask.sum())

        total_violations = p1_violations + p2_violations #+ p3_violations
        plausibility_rate = (
            round(1.0 - total_violations / n_evaluated, 4)
            if n_evaluated > 0 else 0.0
        )
        violations_pct = round(total_violations / n_evaluated * 100, 2) if n_evaluated > 0 else 0.0

        return {
            # canonical names
            'test': 'plausibility',
            'nights_evaluated': n_evaluated,
            'plausibility_rate': plausibility_rate,
            'total_violations': total_violations,
            'P1_high_WASO_healthy': p1_violations,
            'P2_low_N3_healthy': p2_violations,
            #'P3_normal_EEG_eeg_patho': p3_violations,
            # aliases for test_pipeline.py
            'n_nights_evaluated': n_evaluated,
            'n_violations': total_violations,
            'violations_pct': violations_pct,
            # 'P3_scope_note': (
            #     'P3 covers: psychophysiological_insomnia, restless, '
            #     'hypersomnolence, sdb_risk. '
            #     'Excluded: paradoxical_insomnia (normal EEG is defining feature), '
            #     'coupling_impairment, rar_misalignment (RAR-defined phenotypes).'
            # ),
            'thresholds': {
                'P1_WASO_min': WASO_P1,
                'P2_N3_max': N3_P2,
                # 'P3_WASO_normal': WASO_NORMAL_MAX,
                # 'P3_N3_normal_min': N3_NORMAL_MIN,
            },
        }

    def test_fairness(self, strat_df, demographics_df):
        """
        Compare Y_Patient distributions across sex and age-band strata.

        strat_df is the output of StratificationLayer.stratify().
        demographics_df is the output of load_demographics().

        Returns dict with chi-square results and effect sizes per stratum.
        """
        merged = strat_df[["Y_Patient", "Y_Patient_name"]].join(
            demographics_df[["age_band", "sex_encoded", "autism_status"]],
            how="left"
        )
        merged = merged[merged["Y_Patient"] >= 0]  # only definitive labels

        results = {}

        for stratum, col in [("sex", "sex_encoded"),
                             ("age_band", "age_band"),
                             ("autism_status", "autism_status")]:
            valid = merged[merged[col].isin([0, 1])].copy()
            if len(valid) < 10:
                results[stratum] = {"error": "insufficient data"}
                continue

            # contingency table: stratum × label
            ct = pd.crosstab(valid[col], valid["Y_Patient"])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                results[stratum] = {"error": "insufficient label diversity"}
                continue

            try:
                chi2, p, dof, expected = chi2_contingency(ct)
                n = ct.values.sum()
                cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))

                # per-label prevalence by stratum
                prevalence = {}
                for lbl in ct.columns:
                    row_totals = ct.sum(axis=1)
                    prevalence[int(lbl)] = {
                        str(group): round(float(ct.loc[group, lbl]) /
                                          float(row_totals[group]), 3)
                        if group in ct.index else 0.0
                        for group in [0, 1]
                    }

                results[stratum] = {
                    "chi2": round(float(chi2), 4),
                    "p_value": round(float(p), 4),
                    "dof": int(dof),
                    "cramers_v": round(cramers_v, 4),
                    "significant": bool(p < self.alpha),
                    "n": int(n),
                    "label_prevalence_by_stratum": prevalence,
                }
            except Exception as e:
                results[stratum] = {"error": str(e)}

        return {"test": "fairness", "strata": results}

    # ------------------------------------------------------------------
    # Test 4 — Label coverage
    # ------------------------------------------------------------------

    def test_coverage(self, label_result, med_context=None):
        """
        Summarise the disposition of all participant-nights.

        label_result is the output of SymbolicLabeler.predict() or
        FusionModel output with hard_label and all_abstained columns.
        """
        total = len(label_result)
        labelled = int((label_result["hard_label"] >= 0).sum())
        un_class = int(label_result["all_abstained"].sum())

        # medication-confounded nights
        med_abstain = 0
        med_strengthen = 0
        if med_context is not None:
            try:
                mc_cols = ["med_should_abstain_sol", "med_sol_signal_strengthened"]
                mc_reset = med_context[mc_cols].reset_index()

                if "pid" in mc_reset.columns and "participant_id" not in mc_reset.columns:
                    mc_reset = mc_reset.rename(columns={"pid": "participant_id"})
                if "diary_date" in mc_reset.columns and "night_date" not in mc_reset.columns:
                    mc_reset = mc_reset.rename(columns={"diary_date": "night_date"})

                joined = label_result.merge(
                    mc_reset, on=["participant_id", "night_date"], how="left"
                )
                med_abstain = int(joined["med_should_abstain_sol"].fillna(False).sum())
                med_strengthen = int(joined["med_sol_signal_strengthened"].fillna(False).sum())
            except Exception:
                med_abstain = 0
                med_strengthen = 0

        label_dist = (label_result[label_result["hard_label"] >= 0]
                      ["hard_label"].value_counts().sort_index().to_dict())

        return {
            "test": "coverage",
            "total_nights": total,
            "labelled_nights": labelled,
            "labelled_pct": round(100 * labelled / total, 1),
            "unclassifiable": un_class,
            "unclassifiable_pct": round(100 * un_class / total, 1),
            "med_abstain_sol": med_abstain,
            "med_sol_strengthen": med_strengthen,
            "label_distribution": {
                LABEL_NAMES.get(int(k), f"label_{k}"): int(v)
                for k, v in label_dist.items()
            },
        }

    # ------------------------------------------------------------------
    # Sensitivity analyses
    # ------------------------------------------------------------------

    def sensitivity_analyses(self, label_result, med_context,
                             strat_layer, demographics_df=None):
        """
        Three prespecified sensitivity analyses on medication subgroups.

        SA1 — include all nights (including med-confounded)
        SA2 — stimulant-free nights only
        SA3 — melatonin + elevated SOL (treatment-refractory subgroup)
        """
        if med_context is None:
            return {"error": "medication context required for sensitivity analyses"}
        # Normalise column names
        lr = label_result.copy()
        if "pid" in lr.columns and "participant_id" not in lr.columns:
            lr = lr.rename(columns={"pid": "participant_id"})
        if "diary_date" in lr.columns and "night_date" not in lr.columns:
            lr = lr.rename(columns={"diary_date": "night_date"})
        # med_context may be indexed by (participant_id, night_date)
        mc_cols = ["med_should_abstain_sol", "med_has_stimulant",
                   "med_has_melatonin", "med_sol_signal_strengthened"]
        mc_reset = med_context[
            [c for c in mc_cols if c in med_context.columns]
        ].reset_index()

        if "pid" in mc_reset.columns and "participant_id" not in mc_reset.columns:
            mc_reset = mc_reset.rename(columns={"pid": "participant_id"})
        if "diary_date" in mc_reset.columns and "night_date" not in mc_reset.columns:
            mc_reset = mc_reset.rename(columns={"diary_date": "night_date"})
        # Normalise medication column names
        rename_map = {
            "med_has_stimulant": "med_stimulant_today",
            "med_has_melatonin": "med_melatonin_tonight",
        }
        # mc_reset = mc_reset.rename(columns={k: v for k, v in rename_map.items() if k in mc_reset.columns})
        # #############################
        # print("mc_reset columns:", mc_reset.columns.tolist())
        # print("lr columns:", lr.columns.tolist())
        # #############################
        #
        # merged = lr.merge(mc_reset, on=["participant_id", "night_date"], how="left")
        # for col in mc_cols:
        #     if col in merged.columns:
        #         merged[col] = merged[col].fillna(False)
        #     else:
        #         merged[col] = False

        results = {}

        # SA1 — all nights (no medication exclusion)
        sa1 = lr[lr["hard_label"] >= 0]
        results["sa1_all_nights"] = {
            "description": "Primary analysis including medication-confounded nights",
            "n_nights": len(sa1),
            "label_dist": sa1["hard_label"].value_counts().sort_index().to_dict(),
        }
        print("SA2 debug - med_has_stimulant sum in lr:", lr["med_has_stimulant"].sum())
        print("SA2 debug - dtype:", lr["med_has_stimulant"].dtype)
        # SA2 — stimulant-free nights only
        sa2 = lr[
            (lr["hard_label"] >= 0) &
            (~lr["med_has_stimulant"].astype(bool))
            ]
        results["sa2_stimulant_free"] = {
            "description": "Stimulant-free nights only",
            "n_nights": len(sa2),
            "label_dist": sa2["hard_label"].value_counts().sort_index().to_dict(),
        }

        # SA3 — melatonin + elevated SOL (treatment-refractory)
        sa3 = lr[
            (lr["hard_label"] >= 0) &
            (lr["med_sol_signal_strengthened"].astype(bool))
            ]
        results["sa3_melatonin_elevated_sol"] = {
            "description": "Melatonin + elevated SOL (treatment-refractory subgroup)",
            "n_nights": len(sa3),
            "label_dist": sa3["hard_label"].value_counts().sort_index().to_dict(),
        }

        return {"test": "sensitivity_analyses", "analyses": results}

    # ------------------------------------------------------------------
    # Within-family concordance
    # ------------------------------------------------------------------

    def test_within_family_concordance(self, strat_df, demographics_df):
        """
        Compare Y_Patient between autistic participants and their siblings
        within the same family.

        The within-family design controls for shared environment and
        genetic background. This test asks: do autistic children in the
        same family as their sibling show a different phenotype profile?

        Returns per-family pair comparisons and overall concordance.
        """
        merged = strat_df[["Y_Patient", "Y_Patient_name", "instability_score"]].join(
            demographics_df[["autism_status", "family_id"]], how="left"
        )
        merged = merged[merged["Y_Patient"] >= 0]

        # find families with both an autistic and a sibling in the sample
        families = merged.groupby("family_id")["autism_status"].apply(list)
        complete = families[families.apply(lambda x: 1 in x and 0 in x)].index

        if len(complete) == 0:
            return {"error": "no complete autistic-sibling pairs in analytic sample"}

        pairs = []
        for fid in complete:
            fam = merged[merged["family_id"] == fid]
            asd_row = fam[fam["autism_status"] == 1].iloc[0]
            sib_row = fam[fam["autism_status"] == 0].iloc[0]
            pairs.append({
                "family_id": fid,
                "asd_phenotype": asd_row["Y_Patient_name"],
                "asd_instability": round(float(asd_row["instability_score"]), 3),
                "sibling_phenotype": sib_row["Y_Patient_name"],
                "sibling_instability": round(float(sib_row["instability_score"]), 3),
                "same_phenotype": asd_row["Y_Patient_name"] == sib_row["Y_Patient_name"],
            })

        pairs_df = pd.DataFrame(pairs)
        concordance = float(pairs_df["same_phenotype"].mean())
        n_pairs = len(pairs_df)
        n_concordant = int(pairs_df["same_phenotype"].sum())

        # ── Binomial test: observed concordance vs chance ─────────────
        from scipy.stats import binomtest, mannwhitneyu
        n_phenotypes = merged["Y_Patient_name"].nunique()
        chance_rate = 1.0 / n_phenotypes
        binom_result = binomtest(n_concordant, n_pairs, chance_rate,
                                 alternative="greater")
        concordance_p = float(binom_result.pvalue)
        concordance_ci = binom_result.proportion_ci(confidence_level=0.95)

        # ── Phenotype distribution comparison ─────────────────────────
        asd_dist = merged[merged["autism_status"] == 1]["Y_Patient_name"].value_counts().to_dict()
        sib_dist = merged[merged["autism_status"] == 0]["Y_Patient_name"].value_counts().to_dict()

        from scipy.stats import chi2_contingency
        all_phens = sorted(set(list(asd_dist.keys()) + list(sib_dist.keys())))
        asd_counts = [asd_dist.get(p, 0) for p in all_phens]
        sib_counts = [sib_dist.get(p, 0) for p in all_phens]
        try:
            chi2_ct, chi2_p, chi2_df, _ = chi2_contingency(
                [asd_counts, sib_counts]
            )
            n_tot = sum(asd_counts) + sum(sib_counts)
            v_dist = float(np.sqrt(chi2_ct / (n_tot * (min(2, len(all_phens)) - 1))))
        except Exception:
            chi2_ct = chi2_p = chi2_df = v_dist = float("nan")

        # ── Instability comparison — Mann-Whitney U ───────────────────
        asd_instab_arr = merged[merged["autism_status"] == 1]["instability_score"].values
        sib_instab_arr = merged[merged["autism_status"] == 0]["instability_score"].values
        asd_instab = float(np.mean(asd_instab_arr))
        sib_instab = float(np.mean(sib_instab_arr))
        try:
            mw_stat, mw_p = mannwhitneyu(asd_instab_arr, sib_instab_arr,
                                         alternative="greater")
            n_asd = len(asd_instab_arr)
            n_sib = len(sib_instab_arr)
            rb_r = float(1 - (2 * mw_stat) / (n_asd * n_sib))
        except Exception:
            mw_stat = mw_p = rb_r = float("nan")

        return {
            "test": "within_family_concordance",
            "n_complete_pairs": int(n_pairs),
            "n_concordant": n_concordant,
            "chance_rate": round(chance_rate, 4),
            "phenotype_concordance": round(concordance, 3),
            "concordance_p_binomial": round(concordance_p, 4),
            "concordance_ci_95": (round(float(concordance_ci.low), 3),
                                  round(float(concordance_ci.high), 3)),
            "asd_phenotype_dist": asd_dist,
            "sibling_phenotype_dist": sib_dist,
            "phenotype_dist_chi2": round(float(chi2_ct), 3),
            "phenotype_dist_p": round(float(chi2_p), 4),
            "phenotype_dist_cramers_v": round(v_dist, 3),
            "asd_mean_instability": round(asd_instab, 4),
            "sibling_mean_instability": round(sib_instab, 4),
            "instability_difference": round(asd_instab - sib_instab, 4),
            "instability_mannwhitney_p": round(float(mw_p), 4),
            "instability_rank_biserial_r": round(rb_r, 3),
            "pairs": pairs_df.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self, lopo_cv_result, label_result,
                    strat_df, med_context=None, demographics_df=None):
        """Run all validation tests and return a combined report dict."""
        report = {}
        report["consistency"] = self.test_consistency(lopo_cv_result)
        report["coverage"] = self.test_coverage(label_result, med_context)
        if demographics_df is not None:
            report["fairness"] = self.test_fairness(strat_df, demographics_df)
            report["within_family"] = self.test_within_family_concordance(
                strat_df, demographics_df
            )
        if med_context is not None:
            report["sensitivity"] = self.sensitivity_analyses(
                label_result, med_context, None, demographics_df
            )
        return report

"""
Symbolic labelling layer — applies R1/R2/R3 and fits the Snorkel label model.

Usage
-----
    from nesy.symbolic.labeler import SymbolicLabeler

    labeler = SymbolicLabeler(ks)
    night_labels = labeler.fit_predict(night_record, med_context, rar_participant)
"""

import numpy as np
import pandas as pd
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel

from nesy.symbolic.rules_v2 import make_lfs, ABSTAIN, UNCLASSIFIABLE, EEG_QUALITY_MIN

N_LABELS = 6  # labels 0–5


class SymbolicLabeler:

    def __init__(self, ks, n_epochs=500, seed=42, verbose=False,
                 gate_normative=False, unknown_med="ungated",
                 r1b_quality_gate=False):
        """
        Defaults reproduce the submitted thesis run. The three rerun options
        are passed straight to make_lfs() — see rules_v2.make_lfs.
        """
        self.ks = ks
        self.n_epochs = n_epochs
        self.seed = seed
        self.verbose = verbose
        self.options = dict(gate_normative=gate_normative,
                            unknown_med=unknown_med,
                            r1b_quality_gate=r1b_quality_gate)
        self.lfs = make_lfs(ks, **self.options)
        self._model = None
        self._fitted = False

    def apply_rules(self, night_record, med_context=None, rar_participant=None):
        """
        Apply R1/R2/R3 to every night row.
        """
        nr = night_record.copy()

        # Pre-join participant-level RAR priors
        if rar_participant is not None:
            rar_cols = [c for c in ["IS", "IV", "L5_midpoint_mean", "RA_mean"]
                        if c not in nr.columns]
            if rar_cols:
                nr = nr.join(
                    rar_participant[rar_cols],
                    on="participant_id", how="left"
                )

        # Pre-join medication context
        if med_context is not None:
            mc = med_context.copy()

            if "pid" in mc.columns and "participant_id" not in mc.columns:
                mc = mc.rename(columns={"pid": "participant_id"})

            # FIX: Select date column cleanly
            if "night_date" in mc.columns:
                mc_date_col = "night_date"
            elif "sleep_date" in mc.columns:
                mc_date_col = "sleep_date"
            elif "diary_date" in mc.columns:
                print("  WARNING: med_context missing 'sleep_date' — falling back to 'diary_date'.")
                mc_date_col = "diary_date"
            else:
                mc_date_col = "date"

            # FIX: Guarantee ISO YYYY-MM-DD formatting unconditionally
            mc["night_date"] = pd.to_datetime(
                mc[mc_date_col], format="mixed", dayfirst=True
            ).dt.strftime("%Y-%m-%d")

            med_flag_cols = [c for c in mc.columns
                             if c.startswith("med_") and c not in nr.columns]

            if isinstance(nr.index, pd.MultiIndex):
                nr_reset = nr.reset_index()
                nr = nr_reset.merge(
                    mc[["participant_id", "night_date"] + med_flag_cols].drop_duplicates(
                        subset=["participant_id", "night_date"]
                    ),
                    on=["participant_id", "night_date"],
                    how="left", indicator="_med_merge"
                ).set_index(["participant_id", "night_date"])
                med_cols = med_flag_cols
            else:
                nr = nr.merge(
                    mc[["participant_id", "night_date"] + med_flag_cols].drop_duplicates(
                        subset=["participant_id", "night_date"]
                    ),
                    on=["participant_id", "night_date"],
                    how="left", indicator="_med_merge"
                )
                med_cols = med_flag_cols

            # Three-state medication: med_known records whether this night
            # had a medication record at all. The boolean flags are still
            # filled False so the rules run unchanged, but med_known lets
            # the rules (unknown_med="abstain") and the reports tell
            # "no medication" apart from "medication not recorded".
            nr["med_known"] = (nr["_med_merge"] == "both")
            nr = nr.drop(columns="_med_merge")
            bool_cols = [c for c in med_cols if "med_" in c and nr[c].dtype == object]
            for c in bool_cols:
                nr[c] = nr[c].fillna(False)
        else:
            nr["med_known"] = False

        nr_flat = nr.reset_index() if isinstance(nr.index, pd.MultiIndex) else nr.copy()
        nr_flat = self.add_availability_flags(nr_flat)

        applier = PandasLFApplier(lfs=self.lfs)
        L = applier.apply(nr_flat)

        return L, nr_flat

    # def apply_rules(self, night_record, med_context=None, rar_participant=None):
    #     """
    #     Apply R1/R2/R3 to every night row.
    #
    #     Returns the L matrix (n_nights × 3) of raw rule votes.
    #     Also returns the enriched night record with medication and
    #     participant-level RAR priors pre-joined.
    #     """
    #     nr = night_record.copy()
    #
    #     # pre-join participant-level IS and RA_mean
    #     if rar_participant is not None:
    #         # pre-join participant-level RAR priors used as gates in R3
    #         # IS: coupling impairment gate
    #         # L5_midpoint_mean: RAR misalignment gate (chronic rest phase delay)
    #         # RA_mean: stored for completeness
    #         rar_cols = [c for c in ["IS","IV", "L5_midpoint_mean", "RA_mean"]
    #                     if c not in nr.columns]
    #         if rar_cols:
    #             nr = nr.join(
    #                 rar_participant[rar_cols],
    #                 on="participant_id", how="left"
    #             )
    #     # pre-join medication context
    #     if med_context is not None:
    #         if isinstance(med_context.index, pd.MultiIndex) or med_context.index.name in (
    #         "participant_id", "night_date"):
    #             # original path — index-based join
    #             med_cols = [c for c in med_context.columns if c not in nr.columns]
    #             nr = nr.join(med_context[med_cols], how="left")
    #         else:
    #             # new path — flat DataFrame with pid/diary_date/sleep_date columns
    #             # map to night_record index column names
    #             mc = med_context.copy()
    #             if "pid" in mc.columns and "participant_id" not in mc.columns:
    #                 mc = mc.rename(columns={"pid": "participant_id"})
    #             # FIX: was unconditionally preferring diary_date, which is
    #             # the date the diary FORM was completed (typically the
    #             # morning after the night it describes) — NOT the night
    #             # the medication/sleep data actually applies to. sleep_date
    #             # is the correct clinical night identifier and is what
    #             # should align with night_record's own night_date. Using
    #             # diary_date silently shifted every medication attribution
    #             # forward by ~1 day, cohort-wide — confirmed via a direct
    #             # date-range check showing diary_date/sleep_date differing
    #             # by exactly one day on real data, while still producing a
    #             # plausible-looking nonzero count of medication-abstained
    #             # nights (406), which is what let this go undetected.
    #             if "night_date" not in mc.columns:
    #                 if "sleep_date" in mc.columns:
    #                     mc = mc.rename(columns={"sleep_date": "night_date"})
    #                 elif "diary_date" in mc.columns:
    #                     print("  WARNING: med_context has no 'sleep_date' "
    #                           "column — falling back to 'diary_date', which "
    #                           "is typically the day AFTER the night it "
    #                           "describes (diary forms are usually completed "
    #                           "the morning after). Medication flags built "
    #                           "from this fallback are likely misattributed "
    #                           "by ~1 day per night. Fix the upstream diary "
    #                           "extraction to preserve sleep_date if this "
    #                           "warning fires.")
    #                     mc = mc.rename(columns={"diary_date": "night_date"})
    #             # mc["night_date"] = pd.to_datetime(mc["night_date"], dayfirst=True).dt.strftime("%Y-%m-%d")
    #             mc["night_date"] = pd.to_datetime(mc["night_date"], format="mixed", dayfirst=True).dt.strftime(
    #                 "%Y-%m-%d")
    #             med_flag_cols = [c for c in mc.columns
    #                              if c.startswith("med_") and c not in nr.columns]
    #             nr = nr.reset_index().merge(
    #                 mc[["participant_id", "night_date"] + med_flag_cols],
    #                 on=["participant_id", "night_date"],
    #                 how="left"
    #             ).set_index(["participant_id", "night_date"])
    #             med_cols = med_flag_cols
    #         # fill missing medication flags as False/neutral
    #         bool_cols = [c for c in med_cols if "med_" in c and nr[c].dtype == object]
    #         for c in bool_cols:
    #             nr[c] = nr[c].fillna(False)
    #
    #     # Snorkel needs a flat DataFrame — reset index
    #     nr_flat = nr.reset_index()
    #
    #     applier = PandasLFApplier(lfs=self.lfs)
    #     L = applier.apply(nr_flat)
    #
    #     return L, nr_flat

    @staticmethod
    def add_availability_flags(df):
        """
        Flag what each night has. Nothing is dropped.

        has_diary  diary entry for this night
        has_rhythm participant-level rest-activity metrics available (IS)
        eeg_ok     Dreem recording quality >= EEG_QUALITY_MIN
        med_status "yes" / "no" / "unknown"
        tier       complete (diary + rhythm + eeg_ok) / partial (diary, but
                   rhythm or EEG missing) / no_diary
        """
        df = df.copy()
        if "diary_missing" in df.columns:
            df["has_diary"] = ~df["diary_missing"].fillna(True).astype(bool)
        else:
            df["has_diary"] = False
        df["has_rhythm"] = df["IS"].notna() if "IS" in df.columns else False
        if "recording_quality" in df.columns:
            q = pd.to_numeric(df["recording_quality"], errors="coerce")
            df["eeg_ok"] = q >= EEG_QUALITY_MIN          # NaN -> False
        else:
            df["eeg_ok"] = False

        known = df["med_known"].fillna(False).astype(bool) if "med_known" in df.columns \
            else pd.Series(False, index=df.index)
        any_med = pd.Series(False, index=df.index)
        if "med_drug_classes" in df.columns:
            any_med = df["med_drug_classes"].apply(
                lambda v: bool(v) if isinstance(v, (list, set, tuple)) else
                (isinstance(v, str) and v.strip() not in ("", "[]", "set()", "False")))
        for c in ["med_has_stimulant", "med_has_melatonin", "med_has_alpha2"]:
            if c in df.columns:
                any_med = any_med | df[c].fillna(False).astype(bool)
        df["med_status"] = np.where(~known, "unknown", np.where(any_med, "yes", "no"))

        df["tier"] = np.select(
            [df["has_diary"] & df["has_rhythm"] & df["eeg_ok"], df["has_diary"]],
            ["complete", "partial"], default="no_diary")
        return df

    @staticmethod
    def abstain_reason(df):
        """
        Reason code per night: labelled / abstain_true /
        abstain_missing_diary / abstain_missing_rhythm / abstain_missing_eeg.
        Missing inputs are checked in that order, so a night missing both
        diary and rhythm is coded as missing_diary.
        """
        return np.select(
            [~df["all_abstained"],
             ~df["has_diary"],
             ~df["has_rhythm"],
             ~df["eeg_ok"]],
            ["labelled",
             "abstain_missing_diary",
             "abstain_missing_rhythm",
             "abstain_missing_eeg"],
            default="abstain_true")

    def fit(self, L):
        """Fit the Snorkel label model on the L matrix."""
        # Snorkel LabelModel needs at least 3 LFs and > 1 class
        # filter to rows where at least one rule voted
        has_vote = (L != ABSTAIN).any(axis=1)
        L_train = L[has_vote]

        if L_train.shape[0] < 10:
            raise ValueError(
                f"Only {L_train.shape[0]} nights with at least one vote — "
                f"need at least 10 to fit the label model."
            )

        self._model = LabelModel(
            cardinality=N_LABELS,
            verbose=self.verbose,
        )
        self._model.fit(
            L_train=L_train,
            n_epochs=self.n_epochs,
            seed=self.seed,
            log_freq=self.n_epochs,
        )
        self._fitted = True
        return self

    def predict(self, L, nr_flat):
        """
        Predict probabilistic labels for all nights.

        Nights where all rules abstained are marked UNCLASSIFIABLE (-2)
        and get a NaN probability vector — they are excluded from Layer 4.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        n = L.shape[0]
        probs = np.full((n, N_LABELS), np.nan)
        hard_labels = np.full(n, UNCLASSIFIABLE, dtype=int)
        all_abstain = (L == ABSTAIN).all(axis=1)

        # predict only for nights with at least one vote
        has_vote = ~all_abstain
        if has_vote.sum() > 0:
            p = self._model.predict_proba(L[has_vote])
            probs[has_vote] = p
            hard_labels[has_vote] = np.argmax(p, axis=1)

        result = nr_flat.copy()
        for i in range(N_LABELS):
            result[f"prob_label_{i}"] = probs[:, i]
        result["hard_label"] = hard_labels
        result["all_abstained"] = all_abstain
        # LF order (make_lfs): [lf_r1_conditioned_arousal, lf_r1_normative_paradox,
        #                        lf_r2_hypersomnia, lf_r3_coupling, lf_r3_misalign,
        #                        lf_r3_conditioned_arousal]
        # individual rule votes for audit
        result["vote_r1_conditioned_arousal"] = L[:, 0]
        result["vote_r1_normative_paradox"] = L[:, 1]
        result["vote_r2_hypersomnia"] = L[:, 2]
        result["vote_r3_coupling"] = L[:, 3]
        result["vote_r3_misalign"] = L[:, 4]
        result["vote_r3_conditioned_arousal"] = L[:, 5]
        if "tier" not in result.columns:
            result = self.add_availability_flags(result)
        result["night_status"] = self.abstain_reason(result)

        return result

    def fit_predict(self, night_record, med_context=None, rar_participant=None):
        """Convenience: apply rules → fit label model → predict."""
        L, nr_flat = self.apply_rules(night_record, med_context, rar_participant)
        self.fit(L)
        return self.predict(L, nr_flat)

    # =========================================================================
    # NEW EXTENSIONS: Soft Probability Aggregation
    # =========================================================================
    def aggregate_participant_level_soft(self, night_results):
        """
        Computes soft expected probabilities (p_bar_ik) for each participant across
        valid (non-abstained) nights.

        p_bar_ik = (1 / T_i) * sum_{t=1}^{T_i} P(Y_it = k | X_it)
        """
        # Exclude unclassifiable nights
        valid_nights = night_results[~night_results["all_abstained"]].copy()

        prob_cols = [f"prob_label_{i}" for i in range(N_LABELS)]

        # Group by participant and average continuous probabilities across valid nights
        grouped = valid_nights.groupby("participant_id")

        agg_df = grouped[prob_cols].mean()  # p_bar_ik
        agg_df["n_valid_nights"] = grouped.size()

        # Calculate Shannon entropy H on continuous expected probabilities
        def compute_shannon_entropy(row):
            probs = row[prob_cols].values.astype(float)
            probs = probs[probs > 0]  # avoid log(0)
            return -np.sum(probs * np.log2(probs))

        agg_df["instability_score"] = agg_df.apply(compute_shannon_entropy, axis=1)

        return agg_df

    def coverage_report(self, result):
        """Summary of label coverage and rule vote distribution."""
        total = len(result)
        n_unclass = result["all_abstained"].sum()
        n_labelled = total - n_unclass

        vote_counts = {}
        for rule in ["vote_r1_conditioned_arousal", "vote_r1_normative_paradox",
                     "vote_r2_hypersomnia", "vote_r3_coupling", "vote_r3_misalign",
                     "vote_r3_conditioned_arousal"]:
            if rule in result.columns:
                vc = result[rule].value_counts().to_dict()
                vote_counts[rule] = vc

        label_counts = result.loc[
            ~result["all_abstained"], "hard_label"
        ].value_counts().sort_index().to_dict()

        return {
            "total_nights": total,
            "labelled_nights": n_labelled,
            "unclassifiable": n_unclass,
            "coverage_pct": round(100 * n_labelled / total, 1),
            "label_distribution": label_counts,
            "rule_votes": vote_counts,
        }

    def coverage_by_tier(self, result):
        """
        Coverage table by availability tier, plus the reason for every
        abstention. 'complete' coverage is the fair measure of how often the
        rules commit when they have every input they need.
        """
        tab = pd.crosstab(result["tier"], result["night_status"], margins=True,
                          margins_name="total")
        lab = result.groupby("tier")["all_abstained"].agg(
            nights="size", labelled=lambda a: int((~a).sum()))
        lab["coverage_pct"] = (100 * lab["labelled"] / lab["nights"]).round(1)
        return {"by_tier": lab, "status_table": tab,
                "med_status": result["med_status"].value_counts().to_dict(),
                "options": self.options}

    def coverage_report_2(self, result):
        """Summary of label coverage and rule vote distribution."""
        #print(f"This is from coverage 2: ", result.columns.tolist())
        total = len(result)
        n_unclass = 0   #result["all_abstained"].sum()
        n_labelled = total - n_unclass

        vote_counts = {}
        for rule in ["vote_r1_conditioned_arousal", "vote_r1_normative_paradox",
                     "vote_r2_hypersomnia", "vote_r3_coupling", "vote_r3_misalign",
                     "vote_r3_conditioned_arousal"]:
            if rule in result.columns:
                vc = result[rule].value_counts().to_dict()
                vote_counts[rule] = vc

        label_counts = (
            result["hard_label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        return {
            "total_nights": total,
            "labelled_nights": n_labelled,
            "unclassifiable": n_unclass,
            "coverage_pct": round(100 * n_labelled / total, 1),
            "label_distribution": label_counts,
            "rule_votes": vote_counts,
        }

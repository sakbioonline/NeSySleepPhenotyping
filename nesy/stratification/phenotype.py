"""
Participant-level phenotype stratification (post-processing).

Takes the night-level phenotype assignments of a participant (hard labels or
class probabilities) and produces one participant-level assignment, Y_Patient.

Quantities per participant
    dominant label   the most frequent night label (or the highest mean
                     probability) and its share of valid nights, dominant_pct
    instability      Shannon entropy (bits) of the label distribution;
                     is_unstable when it exceeds instability_threshold
    compound         a recognised pair of labels that each reach
                     compound_min_pct (see COMPOUND_PATTERNS)

Y_Patient codes
    0..5   dominant phenotype (share >= dominance_threshold)
    -3     compound pattern
    -4     unstable_mixed or indeterminate
    -5     no valid nights

Two entry points share the same statistics but apply the decision rules in a
different order, kept as they were so that earlier outputs remain reproducible.

    stratify(...)      hard-label counts (used for model predictions).
                       Order: compound (only when the dominance share is not
                       reached), boundary buffer, dominant, unstable or
                       indeterminate.
    stratify_v3b(...)  mean class probabilities (used for the weak labels).
                       Order: compound, dominant, boundary buffer, unstable
                       or indeterminate.
"""

import numpy as np
import pandas as pd
from collections import Counter

LABEL_NAMES = {
    0: "Normative_range",
    1: "Paradoxical_insomnia",
    2: "Conditioned_arousal",
    3: "Hyper-somnolence",
    4: "RAR_misalignment",
    5: "Coupling_impairment",
}

N_CLASSES = len(LABEL_NAMES)
PROB_COLS = [f"prob_label_{i}" for i in range(N_CLASSES)]

# recognised compound patterns: sorted (label_a, label_b) -> name
COMPOUND_PATTERNS = {
    (2, 4): "conditioned_rar_misalignment",
    (1, 5): "paradoxical_coupling_impairment",
    (2, 5): "conditioned_coupling_impairment",
}


def _entropy_bits(proportions):
    p = np.asarray(proportions, dtype=float)
    return float(-np.sum(p * np.log2(p + 1e-10)))


class StratificationLayer:

    def __init__(self, dominance_threshold=0.40, instability_threshold=1.5,
                 compound_min_pct=0.25, margin_buffer=0.0,
                 use_soft_probs=False):
        self.dominance_threshold = dominance_threshold
        self.instability_threshold = instability_threshold
        self.compound_min_pct = compound_min_pct
        self.margin_buffer = margin_buffer
        self.use_soft_probs = use_soft_probs

    # ── helpers ──────────────────────────────────────────────────────────────

    def _compound(self, top_label, second_label, top_pct, second_pct):
        if top_pct >= self.compound_min_pct and second_pct >= self.compound_min_pct:
            return COMPOUND_PATTERNS.get(tuple(sorted([top_label, second_label])))
        return None

    def _is_boundary(self, top_pct, second_pct):
        return self.margin_buffer > 0 and (top_pct - second_pct) < self.margin_buffer

    def _record(self, pid, top_label, top_pct, entropy, compound, n_valid,
                y_patient, y_patient_name):
        return {
            "participant_id": pid,
            "dominant_label": top_label,
            "dominant_label_name": LABEL_NAMES.get(top_label, f"label_{top_label}"),
            "dominant_pct": round(top_pct, 3),
            "instability_score": round(entropy, 4),
            "is_unstable": entropy > self.instability_threshold,
            "compound_pattern": compound or "",
            "n_valid_nights": n_valid,
            "Y_Patient": y_patient,
            "Y_Patient_name": y_patient_name,
        }

    @staticmethod
    def _null_record(pid):
        return {
            "participant_id": pid,
            "dominant_label": -1,
            "dominant_label_name": "no_valid_nights",
            "dominant_pct": 0.0,
            "instability_score": 0.0,
            "is_unstable": False,
            "compound_pattern": "",
            "n_valid_nights": 0,
            "Y_Patient": -5,
            "Y_Patient_name": "no_valid_nights",
        }

    def _unassigned_name(self, entropy):
        return ("unstable_mixed" if entropy > self.instability_threshold
                else "indeterminate")

    # ── hard-label stratification ────────────────────────────────────────────

    def stratify(self, label_result, no_abs, use_soft_probs=None):
        """
        Participant-level assignment from night-level hard labels.

        Night labels are the argmax of the prob_label columns. With
        use_soft_probs, the mean class probabilities break ties between the
        two most frequent labels. When no_abs is False, nights flagged in
        label_result["all_abstained"] are excluded.
        """
        if use_soft_probs is None:
            use_soft_probs = self.use_soft_probs
        has_soft_probs = use_soft_probs and all(
            c in label_result.columns for c in PROB_COLS)

        valid = label_result.assign(
            hard_labels=label_result[PROB_COLS].to_numpy().argmax(axis=1).astype(int))
        if not no_abs:
            valid = valid[~valid["all_abstained"]]

        records = []
        for pid, group in valid.groupby("participant_id"):
            n_valid = len(group)
            if n_valid == 0:
                records.append(self._null_record(pid))
                continue

            counts = Counter(group["hard_labels"].values.astype(int))
            top2 = counts.most_common(2)
            top_label, top_count = top2[0]
            second_label, second_count = top2[1] if len(top2) > 1 else (None, 0)
            top_pct = top_count / n_valid
            second_pct = second_count / n_valid

            entropy = _entropy_bits([c / n_valid for c in counts.values()])

            # mean probabilities resolve an exact tie between the top two
            if has_soft_probs and len(top2) > 1 and top_count == second_count:
                soft_sorted = np.argsort(
                    group[PROB_COLS].values.astype(float).mean(axis=0))[::-1]
                top_label, second_label = soft_sorted[0], soft_sorted[1]

            compound = None
            if top_pct < self.dominance_threshold:
                compound = self._compound(top_label, second_label, top_pct, second_pct)

            if compound:
                y, name = -3, compound
            elif self._is_boundary(top_pct, second_pct):
                y, name = -4, "unstable_mixed"
            elif top_pct >= self.dominance_threshold:
                y, name = top_label, LABEL_NAMES.get(top_label, f"label_{top_label}")
            else:
                y, name = -4, self._unassigned_name(entropy)

            records.append(self._record(pid, top_label, top_pct, entropy,
                                        compound, n_valid, y, name))

        return pd.DataFrame(records).set_index("participant_id")

    # ── probability-mean stratification ──────────────────────────────────────

    def stratify_v3b(self, label_result):
        """Participant-level assignment from mean night-level class probabilities."""
        records = []
        for pid, group in label_result.groupby("participant_id"):
            n_valid = len(group)
            if n_valid == 0:
                records.append(self._null_record(pid))
                continue

            p_bar = group[PROB_COLS].values.astype(float).mean(axis=0)
            p_bar = p_bar / (p_bar.sum() + 1e-12)

            order = np.argsort(p_bar)[::-1]
            top_label, second_label = int(order[0]), int(order[1])
            top_pct, second_pct = float(p_bar[top_label]), float(p_bar[order[1]])
            entropy = _entropy_bits(p_bar)

            compound = self._compound(top_label, second_label, top_pct, second_pct)

            if compound:
                y, name = -3, compound
            elif top_pct >= self.dominance_threshold:
                y, name = top_label, LABEL_NAMES[top_label]
            elif self._is_boundary(top_pct, second_pct):
                y, name = -4, "unstable_mixed"
            else:
                y, name = -4, self._unassigned_name(entropy)

            records.append(self._record(pid, top_label, top_pct, entropy,
                                        compound, n_valid, y, name))

        return pd.DataFrame(records).set_index("participant_id")

    # ── summary ──────────────────────────────────────────────────────────────

    def summary(self, strat_df):
        """Cohort-level summary of the Y_Patient distribution."""
        return {
            "total_participants": len(strat_df),
            "dominant_assigned": int((strat_df["Y_Patient"] >= 0).sum()),
            "compound_assigned": int((strat_df["Y_Patient"] == -3).sum()),
            "unstable_mixed": int((strat_df["Y_Patient_name"] == "unstable_mixed").sum()),
            "indeterminate": int((strat_df["Y_Patient_name"] == "indeterminate").sum()),
            "mean_instability": round(float(strat_df["instability_score"].mean()), 4),
            "high_instability_n": int(strat_df["is_unstable"].sum()),
            "y_patient_distribution": strat_df["Y_Patient_name"].value_counts().to_dict(),
        }
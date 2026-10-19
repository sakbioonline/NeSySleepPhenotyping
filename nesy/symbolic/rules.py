"""
Heuristic labelling functions for the NeSy symbolic layer.

Change from previous version:
    Rules now receive age-stratified thresholds. make_lfs() requires
    the night-level DataFrame to contain an 'age_at_eval_years' column
    (joined from D4 participant metadata before this layer runs).

    On each night, r1 and r2 are resolved by calling:
        r1 = ks.r1.for_age(x['age_at_eval_years'])
        r2 = ks.r2.for_age(x['age_at_eval_years'])
    R3 thresholds are cohort-relative and age-invariant.

Rules firing on each night row:
    R1a — conditioned arousal via ANX/SOL (D3 subjective + medication context)
    R1b — normative vs paradoxical via SE/WASO/SQ/DIFF (D3 + D1/D2)
    R2  — sleep architecture (D2 only)
    R3  — rest-activity rhythm + somatic coupling (D1)

    CHANGE: R1 was previously a single labelling function that checked
    ANX/SOL first and returned CONDITIONED_AROUSAL immediately on match,
    only evaluating the SE/WASO/SQ/DIFF branch (NORMATIVE/PARADOX) if
    neither fired. Paradoxical insomnia clinically co-occurs with
    elevated anxiety/SOL at a meaningful rate, so nights that would
    have qualified for PARADOX on their objective/subjective profile
    were being silently routed to CONDITIONED_AROUSAL first-match-wins,
    with no PARADOX vote ever cast for Snorkel to weigh against it.
    Split into lf_r1_conditioned_arousal and lf_r1_normative_paradox so
    both can vote independently on the same night; the label model
    resolves any disagreement, the same way it already does for the
    R3 rules rather than one function silently deciding internally.

Label space
-----------
    0  within_normative
    1  paradoxical insomnia
    2  conditioned_arousal
    3  fragmented_sleep
    4  RAR misalignment
    5  hypersomnolence
    6  coupling impairment
   -1  ABSTAIN
   -2  UNCLASSIFIABLE (all rules abstained)

Medication modifiers (pre-joined onto night record)
----------------------------------------------------
    med_should_abstain_sol=True      R1a abstains on SOL-based rules
    med_sol_signal_strengthened=True R1a strengthens SOL signal
      (elevated SOL despite melatonin = treatment-refractory, label 2)
"""

from snorkel.labeling import labeling_function
# LABEL_NAMES = {
#     0: "Normative_range",
#     1: "Paradoxical_insomnia",
#     2: "Conditioned_arousal",
#     3: "Hyper-somnolence",
#     4: "RAR_misalignment",
#     5: "Coupling_impairment",
# }
ABSTAIN = -1
UNCLASSIFIABLE = -2
NORMATIVE = 0 # was HEALTHY
PARADOX = 1
CONDITIONED_AROUSAL = 2 # anxiety-driven onset difficulty  Process S–arousal dissociation (1.2.2 clinical para); Harvey 2002 hyperarousal pathway; internalising–onset link (1.3.1)
MISALIGN = 4
HYPERSOMNIA = 3
COUPLING = 5 #was 7 6


def _safe(row, col, default=None):
    import pandas as pd
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return v


def make_lfs(ks):
    """
    Build Snorkel labelling functions from an age-stratified KnowledgeStore.

    The night record must contain 'age_at_eval_years'. If missing,
    the late-adolescence thresholds are used as a conservative default.

    Returns list of LabelingFunctions:
        [lf_r1_conditioned_arousal, lf_r1_normative_paradox,
         lf_r2_hypersomnia, lf_r3_coupling, lf_r3_misalign,
         lf_r3_conditioned_arousal]
    """
    from nesy.knowledge.ontology import get_thresholds
    r3 = ks.r3  # age-invariant

    # ── R1a: Conditioned arousal (ANX / SOL routes) ──────────────────
    @labeling_function()
    def lf_r1_conditioned_arousal(x):
        if _safe(x, "diary_missing", True):
            return ABSTAIN

        age = _safe(x, "age_at_eval_years")
        r1, _, _ = get_thresholds(ks, age)

        sol = _safe(x, "SOL_min")
        anx = _safe(x, "ANX")

        if anx is not None and anx >= r1.ANX_psychophys_min:
            return CONDITIONED_AROUSAL

        if not _safe(x, "med_should_abstain_sol", False):
            if sol is not None and sol >= r1.SOL_psychophys_min:
                return CONDITIONED_AROUSAL

        # Melatonin signal strengthened — treatment-refractory insomnia
        if _safe(x, "med_sol_signal_strengthened", False):
            if sol is not None and sol >= r1.SOL_psychophys_min:
                return CONDITIONED_AROUSAL

        return ABSTAIN

    # ── R1b: Normative vs paradoxical (SE / WASO / SQ / DIFF) ────────
    @labeling_function()
    def lf_r1_normative_paradox(x):
        if _safe(x, "diary_missing", True):
            return ABSTAIN

        age = _safe(x, "age_at_eval_years")
        r1, _, _ = get_thresholds(ks, age)

        se = _safe(x, "SE_pct")
        waso = _safe(x, "WASO_min")
        sq = _safe(x, "SQ")
        diff = _safe(x, "DIFF")

        if se is None:
            return ABSTAIN

        # Healthy — good objective AND good subjective
        # Age-stratified WASO_healthy_max: 40 min early / 45 min late
        if (se >= r1.SE_healthy_min and
                (waso is None or waso <= r1.WASO_healthy_max) and
                (sq is None or sq >= r1.SQ_healthy_min) and
                (diff is None or diff <= r1.DIFF_healthy_max)):
            return NORMATIVE

        # Paradoxical — good objective BUT poor subjective
        # Evaluated independently of R1a now — no longer silenced by an
        # ANX/SOL match on the same night; Snorkel weighs both votes.
        if (se >= r1.SE_healthy_min and
                (waso is None or waso <= r1.WASO_healthy_max)):
            if ((sq is not None and sq <= r1.SQ_paradox_max) or
                    (diff is not None and diff >= r1.DIFF_paradox_min)):
                return PARADOX

        return ABSTAIN

    # ── R2a: Hypersomnolence ─────────────────────────────────────────
    @labeling_function()
    def lf_r2_hypersomnia(x):
        qual = _safe(x, "recording_quality", 1.0)
        if qual < 0.5:
            return ABSTAIN
        tst = _safe(x, "TST_min")
        if tst is None:
            return ABSTAIN
        age = _safe(x, "age_at_eval_years")
        _, r2, _ = get_thresholds(ks, age)
        # Age-stratified: 660 min early (11 hrs) / 600 min late (10 hrs)
        if tst >= r2.TST_hypersomnia_min:
            return HYPERSOMNIA
        return ABSTAIN

    # ── R3a: Coupling impairment ─────────────────────────────────────
    @labeling_function()
    def lf_r3_coupling(x):
        is_val = _safe(x, "IS")
        iv_val = _safe(x, "IV")
        if is_val is None:
            return ABSTAIN
        # IV≥IV_coupling_exclusion indicates fragmentation beyond entrainment failure —
        # those nights belong to conditioned_arousal (R3: IS≤0.5 AND IV≥1.0).
        # Exclusion creates a clean boundary between the two R3 phenotypes.
        if is_val <= r3.IS_coupling_max and (iv_val is None or iv_val < r3.IV_coupling_exclusion):
            return COUPLING
        return ABSTAIN

    # ── R3b: RAR misalignment ────────────────────────────────────────
    @labeling_function()
    def lf_r3_misalign(x):
        l5_mid_p = _safe(x, "L5_midpoint_mean")
        iv = _safe(x, "IV")
        is_val = _safe(x, "IS")

        if l5_mid_p is None and iv is None:
            return ABSTAIN

        age = _safe(x, "age_at_eval_years")
        _, _, r3 = get_thresholds(ks, age)

        l5_criterion = l5_mid_p is not None and l5_mid_p >= r3.L5_misalign_min
        iv_criterion = iv is not None and iv >= r3.IV_misalign_min

        if not (l5_criterion or iv_criterion):
            return ABSTAIN

        # IS exclusion — nights with low interdaily stability belong to
        # coupling_impairment regardless of L5 or IV. Low IS indicates
        # entrainment failure; late L5 with preserved IS indicates phase
        # shift. These are distinct processes and must not co-fire.
        if is_val is not None and is_val <= r3.IS_coupling_max:
            return ABSTAIN

        return MISALIGN

    @labeling_function()
    def lf_r3_conditioned_arousal(x):
        is_val = _safe(x, "IS")
        iv_val = _safe(x, "IV")
        if is_val is None or iv_val is None:
            return ABSTAIN
        # Low interdaily stability combined with high intradaily variability
        # indicates a hyperarousal-driven circadian pattern rather than pure
        # entrainment failure. Complements lf_r1_conditioned_arousal's ANX
        # and SOL routes by providing an objective RAR-based vote for
        # conditioned_arousal on nights where diary or Dreem data may be
        # missing.
        if is_val <= r3.IS_psychophys_max and iv_val >= r3.IV_psychophys_min:
            return CONDITIONED_AROUSAL
        return ABSTAIN

    return [lf_r1_conditioned_arousal, lf_r1_normative_paradox,
            lf_r2_hypersomnia, lf_r3_coupling, lf_r3_misalign,
            lf_r3_conditioned_arousal]

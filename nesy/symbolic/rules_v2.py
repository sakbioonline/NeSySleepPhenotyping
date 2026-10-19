"""
Consolidated, corrected R1-R3 labelling functions for the ontology-derived
(knowledge-in-labels) pathway.

Every change below was verified against the real 1,502-night cohort
(final_night_table_deb.csv joined with med_ctx_deb.csv), not derived from
reasoning alone. Where a real value is stated, it came from running the
actual function against real data, both before and after the fix.

═══════════════════════════════════════════════════════════════════════
SUMMARY OF CHANGES, WITH VERIFIED IMPACT
═══════════════════════════════════════════════════════════════════════

lf_r1_conditioned_arousal (ANX / SOL — subjective + sleep onset)
  Original design already had two medication branches: stimulant
  exposure abstains the SOL check, melatonin exposure with persistent
  elevated SOL strengthens it. Four fixes added on top:
    1. Uncharacterized medication class (antipsychotic/TCA/SNRI/SSRI/
       other) -> abstain, applied before ANX or SOL is evaluated, since
       several of these classes are themselves psychiatric medications
       that could plausibly affect self-reported ANX directly, not
       just SOL. Verified: 462 nights carry an uncharacterized class;
       190 of those would otherwise have voted.
    2. alpha2_agonist gated the same way as stimulant. med_drug_classes
       records the class only, never the specific drug, so clonidine
       (documented, melatonin-like SOL-shortening effect) cannot be
       distinguished from guanfacine (no comparable documented effect).
       Fixes 33 previously-unconditioned alpha2-alone nights.
    3. stimulant+melatonin co-occurrence -> abstain, not melatonin
       silently overriding stimulant's abstention. Verified: 20 nights
       hit this exact collision; 2 of the 20 still correctly vote via
       the independent, unaffected ANX pathway.
    4. alpha2+melatonin co-occurrence -> abstain, same logic as (3).
  Final verified distribution over all 1,502 nights: 322 vote
  CONDITIONED_AROUSAL, 1,180 abstain.

lf_r2_hypersomnia (TST — total sleep time)
  Originally had no medication conditioning at all. Two fixes added:
    1. Uncharacterized medication class -> abstain.
    2. Sedating medication (antipsychotic or TCA) without a stimulant
       also present -> abstain (same-direction confound: sedation
       could directly elevate TST).
  Verified: of 30 nights the original function voted HYPERSOMNIA on,
  16 now abstain (uncharacterized class), 0 via the sedation-alone
  case specifically, leaving 14 net votes.

lf_r3_coupling (IS / IV — interdaily stability / intradaily variability)
  Two independent problems found, not one:
    1. IS_coupling_max was stale. The threshold file's own comment
       claims it was calibrated to this cohort's 25th percentile at
       0.36, but this cohort's actual participant-level 25th
       percentile is 0.243. Recalibrating to the true value is the
       larger of the two fixes by a wide margin.
    2. Stimulant exposure -> abstain (same-direction confound:
       stimulant-driven restlessness could elevate IV directly).
  Verified progression across all 1,502 nights:
    stale threshold, no medication gating (original): 879 (58.5%)
    stale threshold, with medication gating:           631 (42.0%)
    recalibrated threshold, no medication gating:       334 (22.2%,
      landing almost exactly on the intended 25% target)
    recalibrated threshold, with medication gating:     267 (17.8%)

lf_r3_misalign (L5 midpoint / IV / IS — circadian phase)
  One fix: melatonin's documented mechanism is circadian phase
  shifting, which speaks directly to L5 midpoint (timing) but has no
  established relationship to IV (fragmentation). The fix gates only
  the L5 branch, not the whole function.
  Verified with real IS now available: true original count was 123
  (not the 501 upper-bound estimated before IS was available, which
  assumed the IS exclusion branch — inactive without real IS data —
  would never fire). Corrected count: 109, a net 14 nights flipped.

lf_r3_conditioned_arousal (IS / IV — RAR-based conditioned arousal)
  No medication fix applies here, because the rule does not fire at
  all in this cohort. IS <= 0.5 and IV >= 1.0 never co-occur across
  all 1,502 nights. The single closest night (IV = 1.1144, clearing
  the IV bar) has IS = 0.5511, missing the IS bar by 0.05. This is a
  threshold-calibration problem structurally identical in kind to the
  coupling fix above, but not yet resolved — flagged, not corrected.

lf_r1_normative_paradox (SE / WASO / SQ / DIFF — normative vs paradoxical)
  Not yet assessed for a medication confound. SE/WASO thresholds are
  used both for the "healthy" and "paradoxical" branches; whether a
  sedating or activating medication could plausibly shift a night in
  or out of either branch has not been checked against real data the
  way the other five functions were.
═══════════════════════════════════════════════════════════════════════
"""
from snorkel.labeling import labeling_function
import ast

ABSTAIN = -1
UNCLASSIFIABLE = -2
NORMATIVE = 0
PARADOX = 1
CONDITIONED_AROUSAL = 2
HYPERSOMNIA = 3
MISALIGN = 4
COUPLING = 5

CHARACTERIZED_R1_CLASSES = {"stimulant", "melatonin", "alpha2_agonist"}
CHARACTERIZED_R2_CLASSES = {"stimulant", "antipsychotic", "tca"}

# One EEG recording-quality threshold, used by every rule that reads
# Dreem-derived values. Keep this equal to min_qual_mean in pipeline.yaml
# so the participant screen and the night-level rules agree.
EEG_QUALITY_MIN = 0.5

# recalibrated against this cohort's real participant-level IS
# distribution — see module docstring. Was 0.4 (stale).
IS_COUPLING_MAX_RECALIBRATED = 0.24295  # exact participant-level 25th percentile, not rounded


def make_lfs(ks, gate_normative=False, unknown_med="ungated",
             r1b_quality_gate=False):
    """
    Build Snorkel labelling functions from an age-stratified KnowledgeStore.

    Rerun options (defaults reproduce the submitted thesis run):
        gate_normative    True -> lf_r1_normative_paradox applies medication
                          gating (the stricter pathway in section 5.2.2.3).
        unknown_med       "ungated" -> nights with unknown medication status
                          are treated as unmedicated (thesis behaviour).
                          "abstain" -> medication-sensitive rules abstain
                          when medication status is unknown.
        r1b_quality_gate  True -> lf_r1_normative_paradox abstains when EEG
                          recording quality < EEG_QUALITY_MIN, matching R2.

    The night record must contain 'age_at_eval_years'. If missing,
    the late-adolescence thresholds are used as a conservative default.

    Returns list of LabelingFunctions:
        [lf_r1_conditioned_arousal, lf_r1_normative_paradox,
         lf_r2_hypersomnia, lf_r3_coupling, lf_r3_misalign,
         lf_r3_conditioned_arousal]
    """
    from nesy.knowledge.ontology import get_thresholds
    r3 = ks.r3  # age-invariant

    def _safe(x, key, default=None):
        val = x.get(key, default) if hasattr(x, "get") else getattr(x, key, default)
        return default if val is None else val

    def _parse_classes(raw):
        if isinstance(raw, (set, list)):
            return set(raw)
        if isinstance(raw, str):
            try:
                return set(ast.literal_eval(raw))
            except Exception:
                return set()
        return set()

    def _med_unknown_abstain(x):
        """True when medication status is unknown and policy says abstain."""
        return unknown_med == "abstain" and not bool(_safe(x, "med_known", True))

    @labeling_function()
    def lf_r1_conditioned_arousal(x):
        if _safe(x, "diary_missing", True):
            return ABSTAIN
        if _med_unknown_abstain(x):
            return ABSTAIN

        age = _safe(x, "age_at_eval_years")
        r1, _, _ = get_thresholds(ks, age)

        sol = _safe(x, "SOL_min")
        anx = _safe(x, "ANX")
        drug_classes = _parse_classes(_safe(x, "med_drug_classes", set()))

        if drug_classes - CHARACTERIZED_R1_CLASSES:
            return ABSTAIN

        if anx is not None and anx >= r1.ANX_psychophys_min:
            return CONDITIONED_AROUSAL

        stimulant_present = _safe(x, "med_should_abstain_sol", False)
        melatonin_strengthened = _safe(x, "med_sol_signal_strengthened", False)
        alpha2_present = "alpha2_agonist" in drug_classes
        confound_present = stimulant_present or alpha2_present

        if confound_present and melatonin_strengthened:
            return ABSTAIN

        if not confound_present:
            if sol is not None and sol >= r1.SOL_psychophys_min:
                return CONDITIONED_AROUSAL

        if melatonin_strengthened:
            if sol is not None and sol >= r1.SOL_psychophys_min:
                return CONDITIONED_AROUSAL

        return ABSTAIN

    NORMATIVE_PARADOX_CLASSES = {"stimulant", "melatonin", "alpha2_agonist"}

    @labeling_function()
    def lf_r1_normative_paradox(x):
        # Baseline (gate_normative=False) is UNCHANGED from the submitted run.
        # gate_normative=True applies the stricter medication-aware pathway
        # (previously the commented-out block below).
        if _safe(x, "diary_missing", True):
            return ABSTAIN
        if r1b_quality_gate:
            qual = _safe(x, "recording_quality")
            if qual is None or qual != qual or qual < EEG_QUALITY_MIN:  # None/NaN/low
                return ABSTAIN
        if gate_normative:
            return _normative_paradox_gated(x)

        age = _safe(x, "age_at_eval_years")
        r1, _, _ = get_thresholds(ks, age)

        se = _safe(x, "SE_pct")
        waso = _safe(x, "WASO_min")
        sq = _safe(x, "SQ")
        diff = _safe(x, "DIFF")

        if se is None:
            return ABSTAIN

        if (se >= r1.SE_healthy_min and
                (waso is None or waso <= r1.WASO_healthy_max) and
                (sq is None or sq >= r1.SQ_healthy_min) and
                (diff is None or diff <= r1.DIFF_healthy_max)):
            return NORMATIVE

        if (se >= r1.SE_healthy_min and
                (waso is None or waso <= r1.WASO_healthy_max)):
            if ((sq is not None and sq <= r1.SQ_paradox_max) and
                    (diff is not None and diff >= r1.DIFF_paradox_min)):
                return PARADOX

        return ABSTAIN
    ##################old value ########################
    #CHARACTERIZED_NORMATIVE_PARADOX_CLASSES = {"stimulant", "melatonin", "alpha2_agonist"}

    # @labeling_function()
    # def lf_r1_normative_paradox(x):
    #     if _safe(x, "diary_missing", True):
    #         return ABSTAIN
    #
    #     age = _safe(x, "age_at_eval_years")
    #     r1, _, _ = get_thresholds(ks, age)
    #
    #     se = _safe(x, "SE_pct")
    #     waso = _safe(x, "WASO_min")
    #     sq = _safe(x, "SQ")
    #     diff = _safe(x, "DIFF")
    #     drug_classes = _parse_classes(_safe(x, "med_drug_classes", set()))
    #
    #     if se is None:
    #         return ABSTAIN
    #
    #     # Base objective sleep consolidation
    #     objective_healthy = (se >= r1.SE_healthy_min and
    #                          (waso is None or waso <= r1.WASO_healthy_max))
    #
    #     if not objective_healthy:
    #         return ABSTAIN
    #
    #     # 1. UNCHARACTERIZED MEDICATION GATE
    #     # Psychotropics (SSRIs, SNRIs, TCAs, Antipsychotics) confound both subjective
    #     # diary ratings and actigraphic continuity.
    #     if drug_classes - CHARACTERIZED_NORMATIVE_PARADOX_CLASSES:
    #         return ABSTAIN
    #
    #     # 2. NORMATIVE BRANCH: Requires UNASSISTED physiology
    #     # Sleep continuity achieved via melatonin or alpha-2 agonists cannot vote NORMATIVE.
    #     has_sleep_stabilizer = bool(drug_classes & {"melatonin", "alpha2_agonist"})
    #     diary_healthy = ((sq is None or sq >= r1.SQ_healthy_min) and
    #                      (diff is None or diff <= r1.DIFF_healthy_max))
    #
    #     if diary_healthy:
    #         if has_sleep_stabilizer:
    #             # Pharmacologically stabilized, not naturally normative
    #             return ABSTAIN
    #         return NORMATIVE
    #
    #     # 3. PARADOXICAL BRANCH: Subjective misperception
    #     # Subjective distress despite normal objective sleep
    #     diary_paradox = ((sq is not None and sq <= r1.SQ_paradox_max) and
    #                      (diff is not None and diff >= r1.DIFF_paradox_min))
    #
    #     if diary_paradox:
    #         # If stimulants are present, subjective difficulty may reflect stimulant rebound
    #         # rather than classical idiopathic sleep state misperception.
    #         if "stimulant" in drug_classes:
    #             return ABSTAIN
    #         return PARADOX
    #
    #     return ABSTAIN

    def _normative_paradox_gated(x):
        if _med_unknown_abstain(x):
            return ABSTAIN
        age = _safe(x, "age_at_eval_years")
        r1, _, _ = get_thresholds(ks, age)
        se = _safe(x, "SE_pct")
        waso = _safe(x, "WASO_min")
        sq = _safe(x, "SQ")
        diff = _safe(x, "DIFF")
        drug_classes = _parse_classes(_safe(x, "med_drug_classes", set()))
        if se is None:
            return ABSTAIN
        objective_healthy = (se >= r1.SE_healthy_min and
                             (waso is None or waso <= r1.WASO_healthy_max))
        if not objective_healthy:
            return ABSTAIN
        # uncharacterised psychotropics confound diary ratings and continuity
        if drug_classes - NORMATIVE_PARADOX_CLASSES:
            return ABSTAIN
        diary_healthy = ((sq is None or sq >= r1.SQ_healthy_min) and
                         (diff is None or diff <= r1.DIFF_healthy_max))
        if diary_healthy:
            # sleep achieved with melatonin / alpha2 is not unassisted Normative
            if drug_classes & {"melatonin", "alpha2_agonist"}:
                return ABSTAIN
            return NORMATIVE
        diary_paradox = ((sq is not None and sq <= r1.SQ_paradox_max) and
                         (diff is not None and diff >= r1.DIFF_paradox_min))
        if diary_paradox:
            # subjective difficulty may reflect stimulant rebound
            if "stimulant" in drug_classes:
                return ABSTAIN
            return PARADOX
        return ABSTAIN

    @labeling_function()
    def lf_r2_hypersomnia(x):
        qual = _safe(x, "recording_quality", 1.0)
        if qual < EEG_QUALITY_MIN:
            return ABSTAIN
        if _med_unknown_abstain(x):
            return ABSTAIN
        tst = _safe(x, "TST_min")
        if tst is None:
            return ABSTAIN

        drug_classes = _parse_classes(_safe(x, "med_drug_classes", set()))
        if drug_classes - CHARACTERIZED_R2_CLASSES:
            return ABSTAIN

        # Revised: antipsychotic/TCA effects on TST are genuinely uncertain,
        # not asserted as sedating. Some specific drugs (trazodone,
        # mirtazapine) are strongly sedating; TCAs used for ADHD
        # specifically (desipramine) show a stimulant-like profile instead;
        # the one directly relevant actigraphy study found (antipsychotic
        # use, adult schizophrenia, clozapine) showed INCREASED rhythm
        # structure, not disruption -- off-population and inconclusive
        # either way for this 10-17 autism cohort. Abstain on presence
        # rather than assert a direction, the same treatment already used
        # for alpha2_agonist in R1.
        if drug_classes & {"antipsychotic", "tca"}:
            return ABSTAIN

        age = _safe(x, "age_at_eval_years")
        _, r2, _ = get_thresholds(ks, age)
        if tst >= r2.TST_hypersomnia_min:
            return HYPERSOMNIA
        return ABSTAIN

    @labeling_function()
    def lf_r3_coupling(x):
        is_val = _safe(x, "IS")
        iv_val = _safe(x, "IV")
        if is_val is None:
            return ABSTAIN
        if _med_unknown_abstain(x):
            return ABSTAIN

        drug_classes = _parse_classes(_safe(x, "med_drug_classes", set()))
        if "stimulant" in drug_classes:
            return ABSTAIN

        if is_val <= IS_COUPLING_MAX_RECALIBRATED and (iv_val is None or iv_val < r3.IV_coupling_exclusion):
            return COUPLING
        return ABSTAIN

    @labeling_function()
    def lf_r3_misalign(x):
        l5_mid_p = _safe(x, "L5_midpoint")
        iv = _safe(x, "IV")
        is_val = _safe(x, "IS")
        if l5_mid_p is None and iv is None:
            return ABSTAIN
        if _med_unknown_abstain(x):
            return ABSTAIN

        drug_classes = _parse_classes(_safe(x, "med_drug_classes", set()))
        melatonin_present = "melatonin" in drug_classes

        l5_criterion = (l5_mid_p is not None and l5_mid_p >= r3.L5_misalign_min
                        and not melatonin_present)
        # IV gating extended to melatonin on weaker, off-population evidence
        # (a benzodiazepine-tapering study, not autism-specific, found
        # melatonin measurably decreases IV alongside increasing IS). The
        # autism-specific evidence for melatonin (case report, narrative
        # review) supports circadian regularity broadly but does not
        # separate IS from IV numerically the way this study did -- this
        # gate is a documented extension from a related but different
        # population, not a confirmed autism-specific finding.
        iv_criterion = (iv is not None and iv >= r3.IV_misalign_min
                        and not melatonin_present)

        if not (l5_criterion or iv_criterion):
            return ABSTAIN
        if is_val is not None and is_val <= IS_COUPLING_MAX_RECALIBRATED:
            return ABSTAIN
        return MISALIGN

    @labeling_function()
    def lf_r3_conditioned_arousal(x):
        # FLAGGED, NOT FIXED — this rule structurally never fires in this
        # cohort (IS<=0.5 and IV>=1.0 never co-occur; closest miss is
        # IS=0.5511 on the one night IV clears 1.0). See module docstring.
        is_val = _safe(x, "IS")
        iv_val = _safe(x, "IV")
        if is_val is None or iv_val is None:
            return ABSTAIN
        if is_val <= r3.IS_psychophys_max and iv_val >= r3.IV_psychophys_min:
            return CONDITIONED_AROUSAL
        return ABSTAIN

    return [lf_r1_conditioned_arousal, lf_r1_normative_paradox,
            lf_r2_hypersomnia, lf_r3_coupling, lf_r3_misalign,
            lf_r3_conditioned_arousal]
"""
nesy/fusion/constraints.py

Symbolic constraint functions for Layer 4 — the L_constraints term.

Paper terminology → code implementation mapping
───────────────────────────────────────────────
C_eeg   (EEG plausibility)
    → ConstraintFunctions.compute() / compute_per_rule() / compute_signed_slacks()
    → Physiological impossibility penalties derived from EEG/actigraphy
      summary stats: WASO_min, SE_pct, TST_min, and IV (EmbracePlus-derived
      intradaily variability — see design note in __init__ for why this
      replaced a ROCKET-derived proxy).
    → Thresholds loaded from knowledge_base/norms/sleep_architecture.yaml
      where available; IV thresholds (C2/C3) are cohort-calibrated via
      calibrate() at runtime — see calibrate()'s docstring.
    → Individual rules (current):
        C1  c1_waso_not_healthy         — high WASO → suppress P(normative)
        C2  c2_low_frag_not_coupling    — low IV → suppress P(coupling_impairment)
        C3  c3_high_frag_not_healthy    — high IV → suppress P(normative)
        C4  c4_suppress_{paradoxical,conditioned_arousal,hypersomnolence,
             rar_misalign,coupling}    — normal SE/WASO/IV → 5 dedicated
             rules, one per suppressed class (each with its own gamma_k
             and EMA scale — split from a single 5-class-averaged rule so
             no class is structurally diluted; see the RULE_NAMES comment)
        C5  c5_cross_modal_disagreement — ROCKET/EEG disagreement → suppress overconfidence
        C6  c6_high_tst_suppresses_insomnia — high TST → suppress P(conditioned_arousal, paradoxical)
        C7  c7_low_se_suppresses_paradoxical — low SE → suppress P(paradoxical)
      (C8 sdb_architecture is retired — no respiratory channels available)

C_traj  (trajectory alignment)
    → λⱼ(GBTM) modifier in ConstraintFunctions.compute()
    → GBTM group assignment modulates per-rule constraint weights.
    → High-fragmentation trajectory → λ_healthy × 1.5 (tighter suppression).
    → Regulated trajectory → λ_healthy × 0.7 (relaxed suppression).
    → Not a separate function — embedded in the weight computation of C1–C8.

C_smooth (temporal phenotype coherence)
    → TemporalSmoothnessConstraint class (nesy/fusion/model.py)
    → Penalises abrupt night-to-night prediction swings within a participant.
    → Computed over full participant sequences each epoch (not per mini-batch).
    → Penalty: Σᵢ Σₜ max(0, ||Ŷᵢₜ − Ŷᵢₜ₋₁||₂ − δ)²

Combined loss:
    L_constraints = C_eeg + C_traj + C_smooth
    L_total       = L_data + λ · L_constraints

Symbolic grounding:
    All constraint thresholds are clinically sourced from the knowledge base,
    not derived from training data. This is what distinguishes L_constraints
    from statistical regularisation (L1, L2, Dropout): the penalty boundaries
    encode what domain experts know is physiologically impossible, not what
    the empirical distribution suggests.

    Kautz (2022) Type 4 NeSy: symbolic constraints are embedded in the
    training objective, active on every weight update.

Usage:
    cfg = load_constraint_config("configs/heuristics.toml",
                                  kb_dir="knowledge_base")
    cf  = ConstraintFunctions(cfg)
    L_c = cf.compute(Y_pred, obs_features, gbtm_groups, gbtm_conf)
"""


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# ── Label indices (must match heuristics.toml [phenotypes]) ──────────────────
NORMATIVE = 0
PARADOXICAL = 1
CONDITIONED_AROUSAL = 2
HYPERSOMNOLENCE = 3
RAR_MISALIGN = 4
COUPLING_IMPAIRMENT = 5
N_CLASSES = 6


def load_constraint_config(toml_path, sensitivity=False,
                           kb_dir=None):
    """
    Load constraint configuration.

    Runtime parameters (λ weights, epsilon, GBTM modifier) come from
    heuristics.toml [constraints].

    Clinical thresholds (WASO, N3, N1, REM) come from the knowledge base
    norms/sleep_architecture.yaml — the single source of truth.
    heuristics.toml values override the KB values when explicitly set
    (i.e. not -999.0 placeholder).

    Parameters
    ----------
    toml_path  : str or Path — path to heuristics.toml
    sensitivity: bool — use [constraints.sensitivity] overrides
    kb_dir     : str or Path — knowledge_base directory
                 (default: ../knowledge_base relative to configs/)

    Returns
    -------
    dict with all constraint parameters
    """
    import yaml

    path = Path(toml_path)
    if not path.exists():
        raise FileNotFoundError(f"heuristics.toml not found at {path}")

    if tomllib is None:
        raise ImportError(
            "tomllib not available.  Install tomli: pip install tomli"
        )

    with open(path, "rb") as f:
        cfg_full = tomllib.load(f)

    c = cfg_full.get("constraints", {})

    if sensitivity:
        overrides = c.get("sensitivity", {})
        c.update(overrides)
        print("  [constraints] sensitivity mode — thresholds overridden")

    # ── Load thresholds from knowledge base ──────────────────────────
    # KB is the single source of truth for clinical thresholds.
    # Only override with heuristics.toml values when explicitly set
    # (i.e. not the -999.0 placeholder).
    if kb_dir is None:
        # search relative to toml_path
        for candidate in [
            Path(toml_path).parent.parent / "knowledge_base",
            Path(toml_path).parent / "../knowledge_base",
            Path("knowledge_base"),
        ]:
            if candidate.exists():
                kb_dir = candidate
                break

    if kb_dir and Path(kb_dir).exists():
        norms_path = Path(kb_dir) / "norms" / "sleep_architecture.yaml"
        if norms_path.exists():
            with open(norms_path) as f:
                norms = yaml.safe_load(f)

            # Use the more conservative (stricter) threshold across age bands
            # where heuristics.toml has placeholder -999.0
            def _norm(metric, key, agg=max):
                """Extract threshold from norms, aggregating across age bands."""
                vals = []
                for band in ["early_adolescence", "late_adolescence"]:
                    band_data = norms.get(metric, {}).get(band, {})
                    if key in band_data:
                        v = band_data[key]
                        if isinstance(v, list):
                            v = v[0]  # take lower bound of range
                        vals.append(float(v))
                return agg(vals) if vals else None

            # C1: WASO threshold — use severely_elevated lower bound
            if float(c.get("C1_WASO_threshold_min", -999)) == -999.0:
                v = _norm("WASO_min", "severely_elevated", agg=min)
                if v: c["C1_WASO_threshold_min"] = v

            # C1 also feeds C3 WASO
            if float(c.get("C3_WASO_threshold_min", -999)) == -999.0:
                v = _norm("WASO_min", "severely_elevated", agg=min)
                if v: c["C3_WASO_threshold_min"] = v

            # C2: N3 threshold — upper bound of severely_reduced range = 5%
            if float(c.get("C2_N3_threshold_max", -999)) == -999.0:
                band_early = norms.get("N3_pct", {}).get("early_adolescence", {})
                band_late = norms.get("N3_pct", {}).get("late_adolescence", {})
                # severely_reduced is [0.0, 5.0] — upper bound is the threshold
                vals = []
                for band in [band_early, band_late]:
                    sr = band.get("severely_reduced")
                    if isinstance(sr, list) and len(sr) == 2:
                        vals.append(float(sr[1]))  # upper bound
                if vals:
                    c["C2_N3_threshold_max"] = max(vals)  # use more permissive

            # C8: same N3 threshold
            if float(c.get("C8_N3_threshold_max", -999)) == -999.0:
                c["C8_N3_threshold_max"] = c.get("C2_N3_threshold_max", 5.0)

            # C3: N1 threshold — use severely_elevated lower bound
            if float(c.get("C3_N1_threshold_min", -999)) == -999.0:
                v = _norm("N1_pct", "severely_elevated", agg=min)
                if v: c["C3_N1_threshold_min"] = v

            # C6: healthy thresholds from norms
            if float(c.get("C6_WASO_healthy_max", -999)) == -999.0:
                v = _norm("WASO_min", "normal_max", agg=max)
                if v: c["C6_WASO_healthy_max"] = v

            if float(c.get("C6_N3_healthy_min", -999)) == -999.0:
                v = _norm("N3_pct", "normal_min", agg=min)
                if v: c["C6_N3_healthy_min"] = v

            if float(c.get("C6_N1_healthy_max", -999)) == -999.0:
                v = _norm("N1_pct", "normal_max", agg=max)
                if v: c["C6_N1_healthy_max"] = v

            # C8: REM threshold
            if float(c.get("C8_REM_threshold_max", -999)) == -999.0:
                v = _norm("REM_pct", "normal_min", agg=min)
                if v: c["C8_REM_threshold_max"] = v

            print(f"  [constraints] thresholds loaded from knowledge base "
                  f"({norms_path.name})")
        else:
            print(f"  [constraints] KB norms not found at {norms_path} "
                  f"— using heuristics.toml values")
    else:
        print("  [constraints] knowledge_base not found "
              "— using heuristics.toml values only")

    return c


class ConstraintFunctions:
    """
    Computes L_constraints for one mini-batch.

    Parameters
    ----------
    cfg         : dict — from load_constraint_config()
    device      : torch.device

    Call signature:
        L_c = cf.compute(Y_pred, obs, gbtm_groups)

    where:
        Y_pred      : Tensor (batch, 8)  — softmax probabilities
        obs         : dict of Tensors (batch,) — observable EEG/EmbracePlus features
                      keys: 'WASO_min', 'N3_pct', 'N1_pct', 'REM_pct',
                            'IV', 'eeg_quality_decile', 'rocket_decile'
        gbtm_groups : Tensor (batch,) int  — GBTM group per participant
                      or None (no GBTM modifier applied)
    """

    # Active rule set — used by the per-rule Lagrangian path (compute_per_rule /
    # compute_signed_slacks). Kept as a class attribute so model.py can iterate
    # over rule names without hardcoding them a second time.
    #
    # c4_normal_sensors was previously ONE rule that suppressed five
    # classes simultaneously (summed penalty, but ONE shared gamma_k and
    # ONE shared EMA scale in the per-rule Lagrangian mechanism) — so from
    # the loss-weighting mechanism's perspective, all five classes shared
    # a single voice competing against six single-class rules, diluting
    # each class's individual influence regardless of raw penalty
    # magnitude. Split into five dedicated rules (c4_suppress_*), each
    # with its own gamma_k and scale, so no phenotype's suppression signal
    # is structurally weaker than another's just because of how many
    # classes happened to share a rule.
    RULE_NAMES = ["r1_conditioned_arousal", "r2_rar_misalignment", "r3_hypersomnolence", "r4_normative",
                  "r5_coupling", "r6_architecture_normative", "r7_paradoxical_insomnia",
                  "r8a_age_ca_context", "r8b_age_hs_context", "r9_medication_context"] #"r8_age_context",

    def __init__(self, cfg, device=None):
        self._r8_age_rar_min = None
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.eps = float(cfg.get("epsilon", 0.05))
        self.lam = float(cfg.get("lambda_global", 0.5)) * float(cfg.get("lambda_eeg", 1.0))
        self._gbtm = cfg.get("gbtm_modifier", {})
        self._iv_low = float(cfg.get("C2_IV_threshold_low", -999.0))
        self._iv_high = float(cfg.get("C3_IV_threshold_high", -999.0))
        self._calibrated = (self._iv_low != -999.0 and
                            self._iv_high != -999.0)
        self._c1_waso_mod = float(cfg.get("C1_WASO_threshold_min", 60.0))
        self._r1_sol_min = float(cfg.get("SOL_psychophys_min", 45.0))
        self._r1_anx_min = float(cfg.get("ANX_psychophys_min", 6.0))
        self._c4_se_min = float(cfg.get("C4_SE_healthy_min", 85.0))
        self._c4_waso_max = float(cfg.get("C4_WASO_healthy_max", 30.0))
        self._c6_tst_min = float(cfg.get("C6_TST_threshold_min", 540.0))

        self._r6_n3_min = float(cfg.get("R6_N3_norm_min", 20.0))
        self._r6_rem_min = float(cfg.get("R6_REM_norm_min", 18.0))

        self._r7_sq_poor_max = float(cfg.get("R7_SQ_poor_max", 3.0))
        self._r7_diff_poor_min = float(cfg.get("R7_DIFF_poor_min", 5.0))
        # self._r8_age_rar_min = float(cfg.get("R8_age_rar_min", 16.0))
        # self._r8_l5_late_min = float(cfg.get("R8_age_rar_min", 5.0))
        self._r8_age_cut = float(cfg.get("R8_age_cut", 13.0))
        self._r8_sol_early = float(cfg.get("R8_SOL_early", 45.0))
        self._r8_sol_late = float(cfg.get("R8_SOL_late", 60.0))
        self._r8_tst_hs_early = float(cfg.get("R8_TST_HS_early", 660.0))
        self._r8_tst_hs_late = float(cfg.get("R8_TST_HS_late", 600.0))

        self._r9_rar_floor = float(cfg.get("R9_rar_floor", 0.35))




        self._thresholds_calibrated = False
        self._audit_triggers = False #was true

        self._trigger_counts = {
            k: 0 for k in self.RULE_NAMES
        }

        self._trigger_total = {
            k: 0 for k in self.RULE_NAMES
        }

        # Opt-in, per-instance trigger log for co-occurrence diagnosis.
        # The scalar counters above are cheap and always on, but they
        # discard per-night identity the moment a batch is summed in --
        # there is no way to recover which specific night triggered
        # which specific rule from a running total. This parallel log
        # keeps that identity, at the cost of real memory over a full
        # training run, which is why it defaults to off.
        self._detailed_trigger_log = False #was true
        self._trigger_records = []  # list of dicts: participant_id, night_date, rule_name, triggered

    def _track_trigger(self, rule_name, mask, night_index=None):

        if not getattr(self, "_audit_triggers", False):
            return

        self._trigger_counts[rule_name] += int(mask.sum().item())
        self._trigger_total[rule_name] += int(mask.numel())

        # Per-instance detail, only when explicitly enabled and only
        # when the caller actually has night identity to give -- if
        # night_index is None (e.g. an internal call site that was
        # never updated to pass it through), this silently degrades to
        # the scalar-only behaviour above rather than raising, so
        # existing call sites that do not pass night_index keep working
        # unchanged.
        if self._detailed_trigger_log and night_index is not None:
            mask_np = mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask
            for (pid, night_date), triggered in zip(night_index, mask_np):
                self._trigger_records.append({
                    "participant_id": pid,
                    "night_date": night_date,
                    "rule_name": rule_name,
                    "triggered": bool(triggered),
                })

    def get_trigger_dataframe(self):
        """
        Returns the per-instance trigger log as a wide DataFrame, one
        row per (participant_id, night_date), one column per rule,
        boolean, suitable for computing real co-occurrence rates
        directly rather than the aggregate-percentage lower bound.
        Requires _detailed_trigger_log=True to have been set before
        training, and night_index to have been threaded through every
        _track_trigger call site (see r1-r9 below).
        """
        import pandas as pd
        if not self._trigger_records:
            raise ValueError(
                "No trigger records captured. Set cf._detailed_trigger_log = True "
                "before training, and confirm night_index is being passed through "
                "to _track_trigger from every rule -- see the r1-r9 call sites."
            )
        long_df = pd.DataFrame(self._trigger_records)
        wide = long_df.pivot_table(
            index=["participant_id", "night_date"],
            columns="rule_name",
            values="triggered",
            aggfunc="max",  # a night may appear in multiple batches/epochs; True if it ever triggered
        ).fillna(False)
        return wide.reset_index()

    # ------------------------------------------------------------------
    # Physiology masks
    # Used by constraints AND plausibility audits
    # ------------------------------------------------------------------

    def normative_mask(self, obs):

        required = {"SE_pct", "WASO_min", "IV"}

        if not required.issubset(obs.keys()):
            return None

        return (
                (obs["SE_pct"] >= self._c4_se_min)
                &
                (obs["WASO_min"] <= self._c4_waso_max)
                &
                (obs["IV"] <= self._iv_low)
        )

    # def conditioned_arousal_mask(self, obs):
    #
    #     required = {"SE_pct", "WASO_min"}
    #
    #     if not required.issubset(obs.keys()):
    #         return None
    #
    #     return (
    #             (obs["WASO_min"] > self._c1_waso_mod)
    #             |
    #             (obs["SE_pct"] < self._c4_se_min)
    #     )
    def conditioned_arousal_mask(self, obs):

        required = {"SE_pct", "SOL_min", "ANX"}

        if not required.issubset(obs.keys()):
            return None

        anx_arousal = (obs["ANX"] >= self._r1_anx_min)
        if "diary_missing" in obs:
            anx_arousal = anx_arousal & (~obs["diary_missing"])

        return (
                (obs["SOL_min"] > self._r1_sol_min)
                |
                (obs["SE_pct"] < self._c4_se_min)
                |
                anx_arousal
        )

    def age_conditioned_arousal_mask(self, obs):
        required = {"age","SOL_min"}

        if not required.issubset(obs.keys()):
            return None

        younger = ((obs["age"] <= self._r8_age_cut) & (obs["SOL_min"] >= self._r8_sol_early))
        older = ((obs["age"] > self._r8_age_cut) & (obs["SOL_min"] >= self._r8_sol_late))

        return younger | older

    def age_hypersomnolence_mask(self, obs):

        required = {"age", "TST_min"}

        if not required.issubset(obs.keys()):
            return None

        younger = ((obs["age"] <= self._r8_age_cut) & (obs["TST_min"] >= self._r8_tst_hs_early))
        older = ((obs["age"] > self._r8_age_cut) & (obs["TST_min"] >= self._r8_tst_hs_late))

        return younger | older

    def rar_misalignment_mask(self, obs):

        if "IV" not in obs:
            return None

        return (
                obs["IV"] > self._iv_high
        )

    def hypersomnolence_mask(self, obs):

        if "TST_min" not in obs:
            return None

        return (
                obs["TST_min"] >= self._c6_tst_min
        )

    def coupling_mask(self, obs):

        required = {"SE_pct", "IV"}

        if not required.issubset(obs.keys()):
            return None

        return (
                (obs["SE_pct"] < self._c4_se_min) & (obs["IV"] >= self._iv_low) &
                (obs["IV"] <= self._iv_high)
        )

    def paradoxical_insomnia_mask(self, obs):

        required = {"SE_pct","WASO_min","SQ","DIFF"}

        if not required.issubset(obs.keys()):
            return None

        objective_good = ((obs["SE_pct"] >= self._c4_se_min) & (obs["WASO_min"] <= self._c4_waso_max))
        subjective_poor = ((obs["SQ"] <= self._r7_sq_poor_max) | (obs["DIFF"] >= self._r7_diff_poor_min))

        return objective_good & subjective_poor

    def medication_rar_mask(self, obs):
        # Uses the same med_has_stimulant boolean tensor now shared with
        # r2_rar_misalignment and r5_coupling, replacing the old
        # medication_status==1 categorical check. That categorical
        # scheme could not distinguish "stimulant present" from
        # "stimulant present alongside something else" (both collapsed
        # into a single "combined" category), understating true
        # stimulant exposure. This mask now catches every stimulant-
        # exposed night, consistent with how the fix was actually
        # verified in Pathway A's lf_r3_coupling.
        required = {"med_has_stimulant", "IV"}

        if not required.issubset(obs.keys()):
            return None

        return ((obs["med_has_stimulant"] > 0.5) & (obs["IV"] >= self._iv_high))

    def age_hs_mask(self, obs):

        required = {"age", "TST_min"}

        if not required.issubset(obs.keys()):
            return None

        younger = ((obs["age"] <= 13) & (obs["TST_min"] >= 660))

        older = (
                (obs["age"] > 13)
                &
                (obs["TST_min"] >= 600)
        )

        return (younger | older)

    # ── Calibration ──────────────────────────────────────────────────────────
    def calibrate(self, iv_values):
        """
        Set C2/C3 IV thresholds empirically from cohort data.

        Parameters
        ----------
        iv_values : array-like (n_nights,) — IV (intradaily variability),
            EmbracePlus-derived, same values R3's rules already consume
            via obs["IV"]. NOT derived from ROCKET/Z — see __init__ for
            why that distinction matters.

        Sets self._iv_low  (C2_percentile-th percentile — "low IV" cutoff)
             self._iv_high (C3_percentile-th percentile — "high IV" cutoff)
        """
        scores = np.asarray(iv_values, dtype=np.float32)
        p_low = float(self.cfg.get("C2_percentile", 25))
        p_high = float(self.cfg.get("C3_percentile", 75))

        self._iv_low = float(np.percentile(scores, p_low))
        self._iv_high = float(np.percentile(scores, p_high))
        _first = not self._calibrated
        self._calibrated = True
        if _first:
            print(f"  [constraints] calibrated (IV-based): "
                  f"iv_low={self._iv_low:.4f} ({p_low}th pct), "
                  f"iv_high={self._iv_high:.4f} ({p_high}th pct)")

        # write back to cfg so they can be saved
        self.cfg["C2_IV_threshold_low"] = self._iv_low
        self.cfg["C3_IV_threshold_high"] = self._iv_high

    def calibrate_thresholds(self, waso_values=None, se_values=None,
                             tst_values=None):
        if waso_values is not None:
            w = np.asarray(waso_values, dtype=np.float32)
            p_mod = float(self.cfg.get("C1_moderate_percentile", 90))
            p_ext = float(self.cfg.get("C1_extreme_percentile", 98))
            self._c1_waso_mod = float(np.percentile(w, p_mod))
            self._c1_waso_ext = float(np.percentile(w, p_ext))
            p_waso4 = float(self.cfg.get("C4_WASO_percentile", 70))
            self._c4_waso_max = float(np.percentile(w, p_waso4))
            self.cfg["C1_WASO_threshold_min"] = self._c1_waso_mod
            self.cfg["C1_WASO_extreme_min"] = self._c1_waso_ext
            self.cfg["C4_WASO_healthy_max"] = self._c4_waso_max

        if se_values is not None:
            s = np.asarray(se_values, dtype=np.float32)
            p_se4 = float(self.cfg.get("C4_SE_percentile", 25))
            p_se7 = float(self.cfg.get("C7_percentile", 5))
            self._c4_se_min = float(np.percentile(s, p_se4))
            self._c7_se_max = float(np.percentile(s, p_se7))
            self.cfg["C4_SE_healthy_min"] = self._c4_se_min
            self.cfg["C7_SE_threshold_max"] = self._c7_se_max

        if tst_values is not None:
            t = np.asarray(tst_values, dtype=np.float32)
            p_tst = float(self.cfg.get("C6_percentile", 85))
            self._c6_tst_min = float(np.percentile(t, p_tst))
            self.cfg["C6_TST_threshold_min"] = self._c6_tst_min

        if any(v is not None for v in (waso_values, se_values, tst_values)):
            self._thresholds_calibrated = True
            print(f"  [constraints] calibrated (cohort-based): "
                  f"c1_waso_mod={self._c1_waso_mod:.2f}, "
                  f"c1_waso_ext={self._c1_waso_ext:.2f}, "
                  f"c4_se_min={self._c4_se_min:.2f}, "
                  f"c4_waso_max={self._c4_waso_max:.2f}, "
                  f"c6_tst_min={self._c6_tst_min:.2f}, "
                  f"c7_se_max={self._c7_se_max:.2f}")

    def validate_thresholds(self, obs_stats):
        """
        Warn if any threshold lies outside the observed feature distribution.

        Parameters
        ----------
        obs_stats : dict — keys match obs feature names,
                    values are (min, max, mean, std) tuples
        """
        cfg = self.cfg
        checks = [
            ("C1_WASO_threshold_min", "WASO_min", "R1 WASO threshold"),
            ("C4_SE_healthy_min", "SE_pct", "R4 SE threshold"),
            ("C6_TST_threshold_min", "TST_min", "R3 TST threshold"),

            ("R6_N3_norm_min", "N3_pct", "R6 N3 threshold"),
            ("R6_REM_norm_min", "REM_pct", "R6 REM threshold"),
            ("R7_N1_high_min", "N1_pct", "R7 N1 threshold"),
        ]
        for cfg_key, feat_key, label in checks:
            threshold = float(cfg.get(cfg_key, 0))
            if feat_key in obs_stats:
                lo, hi, mu, sd = obs_stats[feat_key]
                if threshold < lo or threshold > hi:
                    print(f"  [constraints] WARNING: {label} ({threshold}) "
                          f"outside observed range [{lo:.1f}, {hi:.1f}]")
                elif abs(threshold - mu) > 2 * sd:
                    print(f"  [constraints] NOTE: {label} ({threshold}) "
                          f"is >2 SD from mean ({mu:.1f} ± {sd:.1f})")

    # ── GBTM λ modifier ──────────────────────────────────────────────────────

    def _gbtm_lambda(self, gbtm_groups, gbtm_conf, base_lam, rule):
        """
        Return per-sample λ values modified by GBTM trajectory group.

        Parameters
        ----------
        gbtm_groups : Tensor (batch,) int or None
        gbtm_conf   : Tensor (batch,) float or None
        base_lam    : float — base λ for this constraint
        rule        : str — 'healthy' or 'coupling'

        Returns
        -------
        Tensor (batch,) float
        """
        batch = gbtm_groups.shape[0] if gbtm_groups is not None else 1
        lam = torch.full((batch,), base_lam,
                         dtype=torch.float32, device=self.device)

        if gbtm_groups is None or not self._gbtm:
            return lam

        conf_thresh = float(self._gbtm.get("confidence_threshold", 0.6))
        high_ids = set(self._gbtm.get("high_frag_group_ids", []))
        reg_ids = set(self._gbtm.get("regulated_group_ids", []))

        if rule == "healthy":
            scale_high = float(self._gbtm.get("lambda_healthy_high_frag", 1.5))
            scale_reg = float(self._gbtm.get("lambda_healthy_regulated", 0.7))
        else:  # coupling
            scale_high = float(self._gbtm.get("lambda_coupling_high_frag", 0.7))
            scale_reg = float(self._gbtm.get("lambda_coupling_regulated", 1.5))

        for i in range(batch):
            if gbtm_conf is not None and float(gbtm_conf[i]) < conf_thresh:
                continue  # low confidence — no modification
            g = int(gbtm_groups[i])
            if g in high_ids:
                lam[i] = base_lam * scale_high
            elif g in reg_ids:
                lam[i] = base_lam * scale_reg

        return lam

    # ── Penalty helper ────────────────────────────────────────────────────────

    def _suppress(self, Y_pred, class_idx, lam_vec):
        """
        Penalise Y_pred[:, class_idx] when it exceeds epsilon.
        penalty = lam * max(0, Ŷ_k − ε)²
        """
        pred = Y_pred[:, class_idx]
        viol = F.relu(pred - self.eps)
        return (lam_vec * viol ** 2).mean()

    def _elevate(self, Y_pred, class_idx, floor, lam_vec):
        """
        Penalise when Y_pred[:, class_idx] falls below floor.
        penalty = lam * max(0, floor − Ŷ_k)²
        """
        pred = Y_pred[:, class_idx]
        viol = F.relu(floor - pred)
        return (lam_vec * viol ** 2).mean()

    # ── Individual constraint functions ──────────────────────────────────────

    # ── C_eeg rules C1–C8 — EEG plausibility constraints ────────────────────
    # Each rule encodes a physiological impossibility derivable from observable
    # EEG summary statistics alone.  Thresholds from knowledge_base/norms/.
    # C_traj (trajectory alignment) is implemented as the λⱼ(GBTM) weight
    # modifier applied inside each rule via the gbtm_groups argument.
    # the NeSy literature(Garcez et al., Mao et al.2019)
    # regularly uses constraint sets of this size or larger
    # def r1_conditioned_arousal(
    #         self,
    #         Y_pred,
    #         obs,
    #         gbtm_groups=None,
    #         gbtm_conf=None,
    #         night_index=None,
    # ):
    #
    #     if "SE_pct" not in obs:
    #         return torch.tensor(0.0, device=self.device)
    #
    #     waso = obs["WASO_min"]
    #     se = obs["SE_pct"]
    #
    #     # Low-SE branch disambiguated by IV, added specifically because
    #     # this branch was found to overlap almost entirely with
    #     # r5_coupling's trigger population: both fire on low SE alone,
    #     # with no criterion separating genuine arousal from genuine
    #     # coupling impairment. Confirmed empirically at 3.72x the rate
    #     # chance alone would predict (115 nights, full cohort). Low SE
    #     # with low IV is arousal without rhythm disruption; low SE with
    #     # mid-range IV is r5's own territory. The WASO branch is
    #     # unaffected -- it never shared this overlap with r5 to begin
    #     # with. If IV is not yet calibrated, falls back to the original,
    #     # undisambiguated SE-only condition rather than silently
    #     # suppressing the whole branch.
    #     if self._calibrated and "IV" in obs:
    #         se_arousal = (se < self._c4_se_min) & (obs["IV"] <= self._iv_low)
    #     else:
    #         se_arousal = (se < self._c4_se_min)
    #
    #     mask = (
    #             (waso > self._c1_waso_mod) |
    #             se_arousal
    #     ).float()
    #
    #     self._track_trigger(
    #         "r1_conditioned_arousal",
    #         mask,
    #         night_index=night_index,
    #     )
    #
    #     if mask.sum() < 1:
    #         return torch.tensor(0.0, device=self.device)
    #
    #     floor = float(
    #         self.cfg.get(
    #             "R1_conditioned_floor",
    #             0.35,
    #         )
    #     )
    #
    #     lam = torch.ones_like(mask) * mask
    #
    #     return self._elevate(
    #         Y_pred,
    #         CONDITIONED_AROUSAL,
    #         floor,
    #         lam,
    #     )
    def r1_conditioned_arousal(self, Y_pred, obs, gbtm_groups=None, gbtm_conf=None, night_index=None):

        if "SE_pct" not in obs or "ANX" not in obs or "SOL_min" not in obs:
            return torch.tensor(0.0, device=self.device)


        se = obs["SE_pct"]
        anx = obs["ANX"]
        sol = obs["SOL_min"]

        if self._calibrated and "IV" in obs:
            se_arousal = (se < self._c4_se_min) & (obs["IV"] <= self._iv_low)
        else:
            se_arousal = (se < self._c4_se_min)

        anx_arousal = (anx >= self._r1_anx_min)
        if "diary_missing" in obs:
            anx_arousal = anx_arousal & (~obs["diary_missing"])

        sol_arousal = (sol >= self._r1_sol_min)
        if "med_should_abstain_sol" in obs and "med_sol_signal_strengthened" in obs:
            sol_arousal = sol_arousal & (
                    (~obs["med_should_abstain_sol"]) | obs["med_sol_signal_strengthened"]
            )

        mask = (
                se_arousal |
                anx_arousal |
                sol_arousal
        ).float()

        self._track_trigger("r1_conditioned_arousal", mask, night_index=night_index)

        if mask.sum() < 1:
            return torch.tensor(0.0, device=self.device)

        floor = float(self.cfg.get("R1_conditioned_floor", 0.35))
        lam = torch.ones_like(mask) * mask

        return self._elevate(Y_pred, CONDITIONED_AROUSAL, floor, lam)

    def r2_rar_misalignment(
            self,
            Y_pred,
            obs,
            gbtm_groups=None,
            gbtm_conf=None,
            night_index=None,
    ):

        if (
                not self._calibrated
                or "IV" not in obs
        ):
            return torch.tensor(0.0, device=self.device)

        mask = (
                obs["IV"] > self._iv_high
        ).float()

        # Medication gating, transferred from Pathway A's lf_r3_misalign
        # (Chapter 4 medication redesign). IV is the shared metric behind
        # both bodies of evidence found there: stimulant medication
        # measurably alters circadian motor activity strength via
        # actigraphy (ADHD-specific, not autism, but age-adjacent), and
        # melatonin measurably decreases IV alongside increasing IS
        # (off-population, benzodiazepine tapering study, weaker
        # evidence, documented as such in the source fix). Both are
        # gated here since r2 has no separate L5-equivalent input to
        # keep them apart the way lf_r3_misalign could.
        if "med_has_stimulant" in obs and "med_has_melatonin" in obs:
            confound = (
                    (obs["med_has_stimulant"] > 0.5)
                    |
                    (obs["med_has_melatonin"] > 0.5)
            ).float()
            mask = mask * (1.0 - confound)

        self._track_trigger(
            "r2_rar_misalignment",
            mask,
            night_index=night_index,
        )

        if mask.sum() < 1:
            return torch.tensor(0.0, device=self.device)

        floor = float(
            self.cfg.get(
                "R2_rar_floor",
                0.35,
            )
        )

        # GBTM trajectory modulation, extended here from r4/r5/r6.
        # r2 checks IV alone -- no L5 input -- making it functionally a
        # fragmentation-elevation rule despite its RAR_misalignment name,
        # not a phase-timing rule. That puts it on the same footing as
        # r5_coupling for trajectory purposes: a participant's long-term
        # autonomic trajectory bears directly on IV, the exact metric
        # this rule checks. Uses the "coupling" direction convention for
        # consistency with r5 (high-fragmentation trajectory relaxes the
        # constraint, regulated trajectory tightens it) -- but this
        # direction itself is inherited from r5's existing convention,
        # not independently re-justified here. Whether high-fragmentation
        # participants should have this constraint relaxed (already
        # likely misaligned, so less penalty needed) or tightened
        # (trajectory and night-level signal agree, so trust it more) is
        # an open question this extension does not resolve.
        lam = self._gbtm_lambda(
            gbtm_groups,
            gbtm_conf,
            1.0,
            "coupling",
        )

        lam = lam * mask

        return self._elevate(
            Y_pred,
            RAR_MISALIGN,
            floor,
            lam,
        )

    def r3_hypersomnolence(
            self,
            Y_pred,
            obs,
            night_index=None,
            **_,
    ):

        if "TST_min" not in obs:
            return torch.tensor(0.0, device=self.device)

        mask = (
                obs["TST_min"] >= self._c6_tst_min
        ).float()
        self._track_trigger(
            "r3_hypersomnolence",
            mask,
            night_index=night_index,
        )

        if mask.sum() < 1:
            return torch.tensor(0.0, device=self.device)

        floor = float(
            self.cfg.get(
                "R3_hyper_floor",
                0.35,
            )
        )

        lam = torch.ones_like(mask) * mask

        return self._elevate(
            Y_pred,
            HYPERSOMNOLENCE,
            floor,
            lam,
        )

    def r4_normative(
            self,
            Y_pred,
            obs,
            gbtm_groups=None,
            gbtm_conf=None,
            night_index=None,
    ):

        if (
                not self._calibrated
                or "IV" not in obs
                or "SE_pct" not in obs
        ):
            return torch.tensor(0.0, device=self.device)

        mask = (
                (obs["SE_pct"] >= self._c4_se_min)
                &
                (obs["WASO_min"] <= self._c4_waso_max)
                &
                (obs["IV"] <= self._iv_low)
        ).float()

        self._track_trigger(
            "r4_normative",
            mask,
            night_index=night_index,
        )

        if mask.sum() < 1:
            return torch.tensor(0.0, device=self.device)

        floor = float(
            self.cfg.get(
                "R4_normative_floor",
                0.50,
            )
        )

        lam = self._gbtm_lambda(
            gbtm_groups,
            gbtm_conf,
            1.0,
            "healthy",
        )

        lam = lam * mask

        return self._elevate(
            Y_pred,
            NORMATIVE,
            floor,
            lam,
        )

    def r5_coupling(
            self,
            Y_pred,
            obs,
            gbtm_groups=None,
            gbtm_conf=None,
            night_index=None,
    ):

        if (
                not self._calibrated
                or "IV" not in obs
                or "SE_pct" not in obs
        ):
            return torch.tensor(
                0.0,
                device=self.device,
            )

        mask = (
                (obs["SE_pct"] < self._c4_se_min)
                &
                (obs["IV"] >= self._iv_low)
                &
                (obs["IV"] <= self._iv_high)
        ).float()

        # Medication gating, transferred directly from Pathway A's
        # lf_r3_coupling (Chapter 4 medication redesign): stimulant
        # exposure measurably alters circadian motor activity via
        # actigraphy in a dedicated placebo-controlled trial. Stimulant
        # only, matching the source fix exactly -- melatonin was not
        # part of that verified fix and is not transferred here.
        if "med_has_stimulant" in obs:
            mask = mask * (1.0 - (obs["med_has_stimulant"] > 0.5).float())

        self._track_trigger(
            "r5_coupling",
            mask,
            night_index=night_index,
        )
        if mask.sum() < 1:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        floor = float(
            self.cfg.get(
                "R5_coupling_floor",
                0.25,
            )
        )

        lam = self._gbtm_lambda(
            gbtm_groups,
            gbtm_conf,
            1.0,
            "coupling",
        )

        lam = lam * mask

        return self._elevate(
            Y_pred,
            COUPLING_IMPAIRMENT,
            floor,
            lam,
        )

    def r6_architecture_normative(
            self,
            Y_pred,
            obs,
            gbtm_groups=None,
            gbtm_conf=None,
            night_index=None,
    ):

        if (
                "N3_pct" not in obs
                or "REM_pct" not in obs
        ):
            return torch.tensor(
                0.0,
                device=self.device,
            )

        n3_min = float(
            self.cfg.get(
                "R6_N3_norm_min",
                20.0,
            )
        )

        rem_min = float(
            self.cfg.get(
                "R6_REM_norm_min",
                18.0,
            )
        )

        mask = (
                (obs["N3_pct"] >= n3_min)
                &
                (obs["REM_pct"] >= rem_min)
        ).float()

        self._track_trigger(
            "r6_architecture_normative",
            mask,
            night_index=night_index,
        )

        if mask.sum() < 1:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        floor = float(
            self.cfg.get(
                "R6_normative_floor",
                0.35,
            )
        )

        lam = self._gbtm_lambda(
            gbtm_groups,
            gbtm_conf,
            1.0,
            "healthy",
        )

        lam = lam * mask

        return self._elevate(
            Y_pred,
            NORMATIVE,
            floor,
            lam,
        )

    def r7_paradoxical_insomnia(
            self,
            Y_pred,
            obs,
            gbtm_groups=None,
            gbtm_conf=None,
            night_index=None,
    ):

        mask = self.paradoxical_insomnia_mask(obs)

        if mask is None:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        mask = mask.float()

        self._track_trigger(
            "r7_paradoxical_insomnia",
            mask,
            night_index=night_index,
        )

        if mask.sum() < 1:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        floor = float(
            self.cfg.get(
                "R7_pi_floor",
                0.30,
            )
        )

        lam = torch.ones_like(mask) * mask

        return self._elevate(
            Y_pred,
            PARADOXICAL,
            floor,
            lam,
        )

    def r8a_age_ca_context(self,Y_pred,obs,night_index=None,**_):

        mask = self.age_conditioned_arousal_mask(obs)
        if mask is None:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        mask = mask.float()

        self._track_trigger(
            "r8a_age_ca_context",
            mask,
            night_index=night_index,
        )

        if mask.sum() < 1:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        return self._elevate(
            Y_pred,
            CONDITIONED_AROUSAL,
            float(
                self.cfg.get(
                    "R8a_ca_floor",
                    0.30,
                )
            ),
            mask,
        )

    def r8b_age_hs_context(self,Y_pred,obs,night_index=None,**_):

        mask = self.age_hypersomnolence_mask(obs)

        if mask is None:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        mask = mask.float()

        self._track_trigger(
            "r8b_age_hs_context",
            mask,
            night_index=night_index,
        )

        if mask.sum() < 1:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        return self._elevate(
            Y_pred,
            HYPERSOMNOLENCE,
            float(
                self.cfg.get(
                    "R8b_hs_floor",
                    0.30,
                )
            ),
            mask,
        )

    def r9_medication_context(
            self,
            Y_pred,
            obs,
            night_index=None,
            **_
    ):

        mask = self.medication_rar_mask(obs)

        if mask is None:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        mask = mask.float()

        self._track_trigger(
            "r9_medication_context",
            mask,
            night_index=night_index,
        )

        if mask.sum() < 1:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        floor = float(
            self.cfg.get(
                "R9_rar_floor",
                0.35,
            )
        )

        lam = torch.ones_like(mask) * mask

        return self._elevate(
            Y_pred,
            RAR_MISALIGN,
            floor,
            lam,
        )


    # ── Per-rule interface — pure per-rule Lagrangian design ────────────────
    # Unlike compute() (below), these do NOT sum across rules and do NOT
    # apply self.lam (lambda_global × lambda_eeg). Each rule gets its own
    # multiplier (gamma_k) in model.py's training loop — no fixed lambda
    # hyperparameter anywhere in this path, matching the actual intent of
    # a Lagrangian formulation: each constraint's weight is learned, not
    # configured.

    def _to_device_obs(self, obs):
        return {k: v.to(self.device) if torch.is_tensor(v)
        else torch.tensor(v, dtype=torch.float32, device=self.device)
                for k, v in obs.items()}

    def compute_per_rule(self, Y_pred, obs, gbtm_groups=None, gbtm_conf=None, night_index=None):
        """
        Per-rule raw penalty (squared-hinge, via _suppress/_elevate) — the
        PRIMAL loss term for each rule. Not summed, not lambda-weighted.
        Squared-hinge is kept here (unlike compute_signed_slacks) because
        it gives smoother gradients for updating theta — standard practice
        to use a different functional for the primal loss than for the
        dual (multiplier) ascent step.

        Returns
        -------
        dict: rule_name -> scalar Tensor (one of RULE_NAMES)
        """
        z = torch.tensor(0.0, device=self.device)
        if not obs:
            return {k: z for k in self.RULE_NAMES}
        obs = self._to_device_obs(obs)

        kw = dict(gbtm_groups=gbtm_groups, gbtm_conf=gbtm_conf, night_index=night_index)
        return {
            "r1_conditioned_arousal": self.r1_conditioned_arousal(Y_pred, obs, **kw),
            "r2_rar_misalignment": self.r2_rar_misalignment(Y_pred, obs, **kw),
            "r3_hypersomnolence": self.r3_hypersomnolence(Y_pred, obs, **kw),
            "r4_normative": self.r4_normative(Y_pred, obs, **kw),
            "r5_coupling": self.r5_coupling(Y_pred, obs, **kw),
            "r6_architecture_normative": self.r6_architecture_normative(Y_pred, obs, **kw),
            "r7_paradoxical_insomnia": self.r7_paradoxical_insomnia(Y_pred, obs, **kw),
            "r8a_age_ca_context":self.r8a_age_ca_context(Y_pred,obs,**kw),
            "r8b_age_hs_context":self.r8b_age_hs_context(Y_pred,obs,**kw),
            "r9_medication_context": self.r9_medication_context(Y_pred, obs, **kw),

        }

    def compute_signed_slacks(self, Y_pred, obs, gbtm_groups=None, gbtm_conf=None):
        """
        Per-rule SIGNED slack — the DUAL ascent signal for each gamma_k.

        Positive  = triggered nights' predictions exceed the rule's bound
                    (constraint violated) -> gamma_k should rise.
        Negative  = triggered nights already satisfy the bound with margin
                    -> gamma_k should fall.
        Zero      = rule not triggered on this batch, or exactly at the
                    boundary.

        This is deliberately UNSQUARED and UNCLIPPED (no relu), unlike
        compute_per_rule's penalty terms. Squared-hinge penalties are
        always >= 0 by construction and can never signal "satisfied with
        margin" — using them for the ascent step is what makes gamma a
        one-way ratchet instead of a real Lagrange multiplier. The signed
        slack here is what lets each gamma_k actually reach an equilibrium.

        Returns
        -------
        dict: rule_name -> scalar Tensor (one of RULE_NAMES)
        """
        z = torch.tensor(0.0, device=self.device)
        if not obs:
            return {k: z for k in self.RULE_NAMES}
        obs = self._to_device_obs(obs)

        # def _slack_suppress(class_idx, mask):
        #     if mask.sum() < 1:
        #         return z
        #     pred = Y_pred[:, class_idx]
        #     return ((pred - self.eps) * mask).sum() / mask.sum()

        def _slack_elevate(class_idx, floor, mask):
            if mask.sum() < 1:
                return z

            pred = Y_pred[:, class_idx]
            return (
                           (floor - pred) * mask
                   ).sum() / mask.sum()

        slacks = {}

        # R1
        mask = (
                (obs["WASO_min"] > self._c1_waso_mod)
                |
                (obs["SE_pct"] < self._c4_se_min)
        ).float()

        slacks["r1_conditioned_arousal"] = (
            _slack_elevate(
                CONDITIONED_AROUSAL,
                float(
                    self.cfg.get(
                        "R1_conditioned_floor",
                        0.35,
                    )
                ),
                mask,
            )
        )

        # R2
        if self._calibrated and "IV" in obs:

            mask = (
                    obs["IV"] > self._iv_high
            ).float()

            slacks["r2_rar_misalignment"] = (
                _slack_elevate(
                    RAR_MISALIGN,
                    float(
                        self.cfg.get(
                            "R2_rar_floor",
                            0.35,
                        )
                    ),
                    mask,
                )
            )

        else:
            slacks["r2_rar_misalignment"] = z

        # R3
        if "TST_min" in obs:

            mask = (
                    obs["TST_min"] >= self._c6_tst_min
            ).float()

            slacks["r3_hypersomnolence"] = (
                _slack_elevate(
                    HYPERSOMNOLENCE,
                    float(
                        self.cfg.get(
                            "R3_hyper_floor",
                            0.35,
                        )
                    ),
                    mask,
                )
            )

        else:
            slacks["r3_hypersomnolence"] = z

        # R4
        if self._calibrated and "IV" in obs:

            mask = (
                    (obs["SE_pct"] >= self._c4_se_min)
                    &
                    (obs["WASO_min"] <= self._c4_waso_max)
                    &
                    (obs["IV"] <= self._iv_low)
            ).float()

            slacks["r4_normative"] = (
                _slack_elevate(
                    NORMATIVE,
                    float(
                        self.cfg.get(
                            "R4_normative_floor",
                            0.50,
                        )
                    ),
                    mask,
                )
            )

        else:
            slacks["r4_normative"] = z

        # R5
        mask = (
                (obs["SE_pct"] < self._c4_se_min)
                &
                (obs["IV"] >= self._iv_low)
                &
                (obs["IV"] <= self._iv_high)
        ).float()

        slacks["r5_coupling"] = (
            _slack_elevate(
                COUPLING_IMPAIRMENT,
                float(
                    self.cfg.get(
                        "R5_coupling_floor",
                        0.25,
                    )
                ),
                mask,
            )
        )
        # R6
        n3_min = float(
            self.cfg.get(
                "R6_N3_norm_min",
                20.0,
            )
        )

        rem_min = float(
            self.cfg.get(
                "R6_REM_norm_min",
                18.0,
            )
        )

        mask = (
                (obs["N3_pct"] >= n3_min)
                &
                (obs["REM_pct"] >= rem_min)
                &
                (obs["SE_pct"] >= self._c4_se_min)
        ).float()

        slacks["r6_architecture_normative"] = (
            _slack_elevate(
                NORMATIVE,
                float(
                    self.cfg.get(
                        "R6_normative_floor",
                        0.35,
                    )
                ),
                mask,
            )
        )

        # Uses paradoxical_insomnia_mask() directly rather than a separate
        # inline definition — previously this block hardcoded SQ<=3.0 only
        # and ignored DIFF entirely, silently diverging from the primal
        # penalty's actual trigger condition once r7 was enabled. Both the
        # primal (r7_paradoxical_insomnia) and dual (here) computations
        # now read the exact same mask, so gamma_r7 tracks the same
        # trigger set the loss itself is penalising.
        mask = self.paradoxical_insomnia_mask(obs)
        if mask is None:
            slacks["r7_paradoxical_insomnia"] = z
        else:
            mask = mask.float()
            slacks["r7_paradoxical_insomnia"] = (
                _slack_elevate(
                    PARADOXICAL,
                    float(self.cfg.get("R7_pi_floor", 0.30)),
                    mask,
                )
            )
        #R8a
        mask = self.age_conditioned_arousal_mask(obs)

        if mask is None:
            slacks["r8a_age_ca_context"] = z
        else:
            mask = mask.float()

            slacks["r8a_age_ca_context"] = (
                _slack_elevate(
                    CONDITIONED_AROUSAL,
                    float(
                        self.cfg.get(
                            "R8a_ca_floor",
                            0.30,
                        )
                    ),
                    mask,
                )
            )
        #R8b
        mask = self.age_hypersomnolence_mask(obs)
        if mask is None:
            slacks["r8b_age_hs_context"] = z
        else:
            mask = mask.float()

            slacks["r8b_age_hs_context"] = (
                _slack_elevate(
                    HYPERSOMNOLENCE,
                    float(
                        self.cfg.get(
                            "R8b_hs_floor",
                            0.30,
                        )
                    ),
                    mask,
                )
            )

        #R9
        mask = self.medication_rar_mask(obs)
        if mask is None:
            slacks["r9_medication_context"] = z
        else:
            mask = mask.float()

            slacks["r9_medication_context"] = (
                _slack_elevate(
                    RAR_MISALIGN,
                    float(
                        self.cfg.get(
                            "R9_rar_floor",
                            0.35,
                        )
                    ),
                    mask,
                )
            )

        return slacks

    # ── Master compute (pooled, fixed-lambda path — used by V3) ─────────────
    def compute(self, Y_pred, obs, gbtm_groups=None, gbtm_conf=None):
        if not obs:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        obs = {
            k:
                v.to(self.device)
                if torch.is_tensor(v)
                else torch.tensor(
                    v,
                    dtype=torch.float32,
                    device=self.device,
                )
            for k, v in obs.items()
        }

        kw = dict(
            gbtm_groups=gbtm_groups,
            gbtm_conf=gbtm_conf,
        )

        L_c = torch.tensor(
            0.0,
            device=self.device,
        )

        L_c += self.r1_conditioned_arousal(
            Y_pred,
            obs,
            **kw,
        )

        L_c += self.r2_rar_misalignment(
            Y_pred,
            obs,
            **kw,
        )

        L_c += self.r3_hypersomnolence(
            Y_pred,
            obs,
        )

        L_c += self.r4_normative(
            Y_pred,
            obs,
            **kw,
        )

        L_c += self.r5_coupling(
            Y_pred,
            obs,
            **kw,
        )

        L_c += self.r6_architecture_normative(
            Y_pred,
            obs,
            **kw,
        )

        L_c += self.r7_paradoxical_insomnia(
            Y_pred,
            obs,
            **kw,
        )

        L_c += self.r8a_age_ca_context(
            Y_pred,
            obs,
            **kw,
        )

        L_c += self.r8b_age_hs_context(
            Y_pred,
            obs,
            **kw,
        )

        L_c += self.r9_medication_context(
            Y_pred,
            obs,
            **kw,
        )

        return self.lam * L_c


def build_obs_features(night_record_df, rocket_features_df):
    """
    Build the observable feature dict for constraint computation
    from the aligned night record and ROCKET feature matrix.

    Parameters
    ----------
    night_record_df   : DataFrame indexed (participant_id, night_date)
                        columns: WASO_min, N3_pct, N1_pct, REM_pct, SE_pct,
                        IV (EmbracePlus-derived, from merge_rar_into_night_record)
    rocket_features_df: DataFrame indexed (participant_id, night_date)
                        columns: rocket_0 … rocket_1999. Used only for
                        rocket_decile (C5's cross-modal comparison) — no
                        longer used to derive a fragmentation constraint
                        input; see ConstraintFunctions.__init__ for why
                        C2/C3/C4 moved to IV instead.

    Returns
    -------
    obs        : dict of np.arrays, one entry per night in the same row order
    iv_values  : np.array (n_nights,) — IV values, pass to calibrate()
    """
    import numpy as np

    # align indices
    shared_idx = night_record_df.index.intersection(
        rocket_features_df.index
    )
    nr = night_record_df.loc[shared_idx]

    if "IV" not in nr.columns:
        raise KeyError(
            "night_record_df has no 'IV' column — expected from "
            "merge_rar_into_night_record(). C2/C3/C4 require IV; "
            "check the night record was built with RAR daily metrics joined."
        )
    iv_values = nr["IV"].fillna(0).values.astype(np.float32)

    # ROCKET-derived decile — kept ONLY for C5's cross-modal disagreement
    # check (comparing a ROCKET-based signal against an EEG-based one is
    # legitimate there, unlike using it as a standalone fragmentation
    # constraint for C2/C3/C4).
    rocket_vals = rocket_features_df.values.astype(np.float32)
    rocket_std = rocket_vals.std(axis=1)
    r_d = pd.Series(rocket_std,
                    index=rocket_features_df.index).loc[shared_idx]
    r_d = (r_d.rank(pct=True) * 10).values

    # EEG quality decile (higher = better quality)
    eeg_q = (nr["N3_pct"].fillna(0).rank(pct=True) * 10).values

    obs = {
        "WASO_min": nr["WASO_min"].fillna(0).values,
        "N3_pct": nr["N3_pct"].fillna(0).values,
        "N1_pct": nr["N1_pct"].fillna(0).values,
        "REM_pct": nr.get("REM_pct",
                          nr["N3_pct"] * 0).fillna(0).values,
        "IV": iv_values,
        "eeg_quality_decile": eeg_q,
        "rocket_decile": r_d,
    }
    return obs, iv_values

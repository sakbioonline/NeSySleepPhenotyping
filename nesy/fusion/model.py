"""
Neuro-symbolic fusion model (Layer 4) — revised architecture.

Training objective:
    L_total = L_data + λ·L_constraints

    L_data        — KL divergence between Snorkel soft labels (Y_weak)
                    and model predictions (Ŷ).

    L_constraints — Three named symbolic constraints:

        C_eeg    — EEG plausibility: physiological impossibility penalties
                   from WASO, N3%, N1%, REM%, fragmentation proxy.
                   Thresholds sourced from knowledge_base/norms/.
                   Grounding: what clinical norms establish as impossible,
                   not what the training data suggests.

        C_traj   — Trajectory alignment: GBTM group modulates per-constraint
                   λ weights. High-fragmentation trajectory → tighter healthy
                   suppression. Participant-level symbolic context.

        C_smooth — Temporal smoothness: penalises large prediction swings
                   night-to-night within a participant. Enforces that
                   phenotype assignments are temporally coherent.
                   Computed over participant sequences, not random batches.
                   Penalty: Σᵢ Σₜ max(0, ||Ŷᵢₜ - Ŷᵢₜ₋₁||₂ - δ)²

Contribution:
    The constraint-guided training objective formalises the relationship
    between symbolic clinical knowledge and neural phenotype learning.
    This positions the framework at Type 4 in Kautz (2022): symbolic
    constraints directly guide neural training through the loss function,
    transferring clinical domain knowledge into model weights without
    requiring the knowledge variables at deployment.

Architectures supported:
    SleepLogReg — PyTorch linear model (0 hidden layers). Mathematically
                  equivalent to multinomial logistic regression but lives
                  in a PyTorch computational graph so it can host dual loss.
                  Used in V1, V2, V4 ablation variants.

    SleepPhenotypeNet — Single-hidden-layer MLP (64 units, ReLU, Dropout).
                        Used in V3, V5 ablation variants.

Ablation design (sequential component addition — LogReg throughout):

    V1 — LogReg · ROCKET only            (actigraphy baseline)
    V2 — LogReg · ROCKET + EEG           (+ EEG morphology, no constraints)
    V3 — LogReg · ROCKET + EEG · fixed λ (+ L_constraints, fixed weight)
    V4 — LogReg · ROCKET + EEG · γ       (+ Lagrangian adaptive weights) ← proposed
    V5 — LogReg · rule inputs            (diary + meds, upper bound)

    Key comparisons:
        V1 → V2 : EEG morphology contribution
        V2 → V3 : Constraint loss contribution (add L_c fixed λ)
        V3 → V4 : Lagrangian adaptive weighting (fixed λ → adaptive γ)
        V4 → V5 : Information asymmetry gap (rule inputs vs learned features)

    Note: SleepPhenotypeNet (MLP) is retained in the codebase for future
    extension but is not used in the current ablation design.

Public API:
    fit()          — train on all labelled nights
    predict_proba()— predict class probabilities
    predict()      — predict hard labels
    lopo_cv()      — leave-one-participant-out cross-validation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, classification_report
from sklearn.impute import SimpleImputer
from pathlib import Path

from nesy.fusion.constraints import (
    ConstraintFunctions, load_constraint_config, build_obs_features
)

N_LABELS = 6


# ── PyTorch logistic regression ───────────────────────────────────────────────

class SleepLogReg(nn.Module):
    """
    Multinomial logistic regression in PyTorch.

    Mathematically identical to sklearn LogisticRegression(solver='lbfgs',
    multi_class='multinomial') but lives in a PyTorch computational graph,
    enabling dual-loss training with L_data + L_constraints.

    Zero hidden layers — the minimal differentiable model.
    Used in all ablation variants V1–V5 (LogReg-only design).
    """

    def __init__(self, n_features, n_classes=N_LABELS):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)


# ── MLP architecture ──────────────────────────────────────────────────────────

class SleepPhenotypeNet(nn.Module):
    """
    Single-hidden-layer MLP for sleep phenotype prediction.

    One hidden layer of 64 units with BatchNorm, ReLU, Dropout.
    Retained for future extension — not used in current V1–V5 ablation
    design, which uses LogReg (SleepLogReg) throughout.
    """

    def __init__(self, n_features, n_classes=N_LABELS, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


# ── Temporal smoothness constraint ────────────────────────────────────────────

class TemporalSmoothnessConstraint:
    """
    C_smooth — penalises large prediction swings night-to-night within
    a participant.

    Symbolic grounding: phenotype assignments should be temporally coherent.
    A participant classified as coupling_impairment on night t should not
    flip to healthy on night t+1 without a physiologically meaningful
    intermediate trajectory. Random classification noise produces large
    swings; genuine phenotype transitions are gradual.

    Penalty:
        C_smooth = Σᵢ Σₜ max(0, ||Ŷᵢₜ - Ŷᵢₜ₋₁||₂ - δ)²

    Where δ = tolerance threshold (default 0.3). Predictions are allowed
    to shift by up to δ in L2 norm between consecutive nights without
    penalty. Only large, abrupt swings are penalised.

    Requires participant-aware ordering — nights must be sorted by date
    within each participant before computing consecutive differences.
    """

    def __init__(self, delta=0.3, lambda_smooth=0.3, device=None):
        self.delta = delta
        self.lambda_smooth = lambda_smooth
        self.device = device or torch.device("cpu")

    def compute(self, Y_pred, pid_arr, night_order_arr):
        """
        Compute C_smooth over a batch of predictions.

        Parameters
        ----------
        Y_pred         : Tensor (n, 8) — predicted probabilities
        pid_arr        : np.array (n,) str — participant IDs
        night_order_arr: np.array (n,) int — sort index within participant

        Returns
        -------
        Scalar Tensor — C_smooth penalty
        """
        total = torch.tensor(0.0, device=self.device)
        n_pairs = 0

        pids = np.unique(pid_arr)
        for pid in pids:
            mask = pid_arr == pid
            if mask.sum() < 2:
                continue
            idx_sorted = np.where(mask)[0][
                np.argsort(night_order_arr[mask])
            ]
            if len(idx_sorted) < 2:
                continue

            seq = Y_pred[idx_sorted]  # (T, 8)
            diffs = seq[1:] - seq[:-1]  # (T-1, 8)
            norms = torch.norm(diffs, dim=1)  # (T-1,)
            penalty = F.relu(norms - self.delta)  # hinge
            total = total + (penalty ** 2).sum()
            n_pairs += len(idx_sorted) - 1

        if n_pairs > 0:
            total = total / n_pairs  # normalise by pair count

        return self.lambda_smooth * total


# ── Fusion model ──────────────────────────────────────────────────────────────

class FusionModel:

    def __init__(
            self,
            hidden=64,
            dropout=0.3,
            lr=1e-3,
            epochs=40, #was 150 then 100
            batch_size=256,
            random_seed=42,
            constraint_config=None,
            device=None,
            use_mlp=False,  # Always defaulting to MLP
            use_constraints=True,
            use_lagrangian=True,
            lambda_smooth=0.3,
            smooth_delta=0.3,
            C=1.0,
            max_iter=1000,
            disabled_rules=None,
            use_soft_prob=False
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.use_mlp = use_mlp
        self.use_constraints = use_constraints
        self.use_lagrangian = use_lagrangian
        self.lambda_smooth = lambda_smooth
        self.smooth_delta = smooth_delta
        self.use_soft_prob = use_soft_prob
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._constraint_cfg = constraint_config
        self._lambda_global_override = None  # set to override config value
        self._cf = None
        self._c_smooth = None

        # Which rules get excluded from the per-rule Lagrangian training
        # loop (RULE_NAMES / gamma multipliers) — NOT the same as a rule
        # being present in constraints.py's compute()/slack dict, which
        # always computes all 7 for diagnostic/audit purposes regardless
        # of this setting. Pass an explicit set to override the default
        # (c4_normal_sensors, c5_cross_modal — see the pruning rationale
        # in _train_net); pass an empty set to disable no rules at all.
        # None means "use the historical default", so existing callers
        # that don't pass this keep their current behaviour unchanged.
        self._disabled_rules = disabled_rules  # None -> _DEFAULT_DISABLED_RULES in _train_net

        self._net = None
        self._scaler = None
        self._imputer = None
        self._fitted = False
        self.feature_names_ = None
        self._iv_values = None  # IV per night, aligned to X's row order — see _calibrate_constraints
        self._waso_values = None
        self._se_values = None
        self._tst_values = None
        self._verbose_constraints = True

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    CLINICAL_COLS = [
        "SE_pct", "SOL_min", "WASO_min", "TST_min",
        "ANX", "DIFF", "SQ",
        "N1_pct", "N3_pct",
        "IV", "L5_midpoint","age","medication_status",
        "ep_pulse_mean", "ep_prv_mean",
    ]

    # FIX: SE_pct and TST_min were missing here. c4_normal_sensors_positive,
    # c6_high_tst_suppresses_insomnia, and c7_low_se_suppresses_paradoxical
    # all gate on these keys being present in `obs` — without them, those
    # three rules returned zero on every call, in every run, regardless of
    # cohort data. This was a missing-wiring bug, not a cohort-data
    # limitation (SE_pct/TST_min exist in night_record via CLINICAL_COLS,
    # they just never reached the obs dict used by the constraint functions).
    # IV added: EmbracePlus-derived intradaily variability, now feeding
    # C2/C3/C4 (previously a ROCKET-derived frag_score computed separately
    # in _build_obs — see constraints.py's design note for why that
    # changed). Added here rather than a new ad-hoc wiring path so it goes
    # through the same proven night_record extraction as WASO_min/SE_pct —
    # this is the same class of bug (a required obs key missing from this
    # list) that silently zeroed out SE_pct/TST_min earlier in this
    # investigation.
    OBS_EEG_COLS = ["WASO_min", "N3_pct", "N1_pct", "REM_pct", "SE_pct","SOL_min", "TST_min", "IV","ANX","SQ","DIFF", "age", "L5_midpoint",
                    "med_has_stimulant","med_has_melatonin","med_has_alpha2","age","age_band","medication_status"]

    # ── Architecture factory ──────────────────────────────────────────────────

    def _make_net(self, n_features):
        """Return the appropriate network for this variant."""
        if self.use_mlp:
            return SleepPhenotypeNet(
                n_features, N_LABELS, self.hidden, self.dropout
            ).to(self.device)
        else:
            return SleepLogReg(n_features, N_LABELS).to(self.device)

    def _variant_label(self):
        arch = "MLP" if self.use_mlp else "LogReg"
        if self.use_constraints:
            loss = "+L_c(Lagrangian)" if self.use_lagrangian else "+L_c(fixed_λ)"
        else:
            loss = ""
        return f"{arch}{loss}"

    # ── Feature building ──────────────────────────────────────────────────────

    def extract_clinical_features(self, label_result, night_record):
        if night_record is None:
            return None
        cols = [c for c in self.CLINICAL_COLS if c in night_record.columns]
        if not cols:
            return None
        labelled = label_result[label_result["hard_label"] >= 0]
        idx = pd.MultiIndex.from_arrays(
            [labelled["participant_id"], labelled["night_date"]],
            names=["participant_id", "night_date"]
        )
        if isinstance(night_record.index, pd.MultiIndex):
            clinical = night_record[cols].reindex(idx)
        else:
            nr_idx = night_record.reset_index().set_index(
                ["participant_id", "night_date"]
            )
            clinical = nr_idx[cols].reindex(idx)
        clinical.columns = [f"clin_{c}" for c in clinical.columns]
        return clinical

    def _build_X(self, rocket_df, symbols_df, context_df,
                 night_index, clinical_df=None):
        idx = pd.MultiIndex.from_tuples(
            night_index, names=["participant_id", "night_date"]
        )
        X_parts = []
        if clinical_df is not None: X_parts.append(clinical_df.reindex(idx))
        if rocket_df is not None: X_parts.append(rocket_df.reindex(idx))
        if symbols_df is not None: X_parts.append(symbols_df.reindex(idx))
        if not X_parts:
            raise ValueError("No feature inputs provided to fusion model")
        return pd.concat(X_parts, axis=1)

    # def _build_obs(self, symbols_df, night_index, rocket_arr=None, night_record=None, **kwargs):
    #     """
    #     Builds observation tensors for PyTorch execution.
    #
    #     Primary source: night_record (holds ingested aggregates including N3% and ep_n3_activity_mean).
    #     Fallback source: symbols_df (for C1-C4 symbolic features if absent in night_record).
    #     """
    #     idx = pd.MultiIndex.from_tuples(
    #         night_index, names=["participant_id", "night_date"]
    #     )
    #     obs = {}
    #
    #     # 1. Primary Source: Ingested night_record
    #     if night_record is not None:
    #         try:
    #             nr = (
    #                 night_record.reindex(idx)
    #                 if isinstance(night_record.index, pd.MultiIndex)
    #                 else night_record.reset_index()
    #                 .set_index(["participant_id", "night_date"])
    #                 .reindex(idx)
    #             )
    #
    #             # C5 Constraint Metrics:
    #             # - Dreem N3 Percentage (0.0 to 1.0)
    #             if "N3_pct" in nr.columns:
    #                 n3_vals = nr["N3_pct"].fillna(0.0).to_numpy(dtype=np.float32)
    #                 if np.max(n3_vals) > 1.0:
    #                     n3_vals /= 100.0
    #                 obs["n3_pct"] = torch.tensor(n3_vals, device=self.device)
    #
    #             # - EmbracePlus Wrist Motion STRICTLY during Dreem N3 epochs
    #             if "ep_n3_activity_mean" in nr.columns:
    #                 n3_motion = nr["ep_n3_activity_mean"].fillna(0.0).to_numpy(dtype=np.float32)
    #                 obs["embrace_n3_motion"] = torch.tensor(n3_motion, device=self.device)
    #
    #             # - Dreem Signal Quality Index (0.0 to 1.0)
    #             if "recording_quality" in nr.columns:
    #                 sqi = nr["recording_quality"].fillna(0.0).to_numpy(dtype=np.float32)
    #                 obs["eeg_sqi"] = torch.tensor(sqi, device=self.device)
    #
    #             # Extract any general OBS_EEG_COLS required by other constraints (e.g. C1-C4)
    #             for col in getattr(self, "OBS_EEG_COLS", []):
    #                 if col in nr.columns and col not in obs:
    #                     val_arr = nr[col].fillna(0.0).to_numpy(dtype=np.float32)
    #                     obs[col] = torch.tensor(val_arr, device=self.device)
    #
    #         except (KeyError, ValueError):
    #             pass
    #
    #     # 2. Secondary Fallback Source: symbols_df (for C1-C4 constraints)
    #     if symbols_df is not None:
    #         try:
    #             s = symbols_df.reindex(idx)
    #             for col in getattr(self, "OBS_EEG_COLS", []):
    #                 if col not in obs and col in s.columns:
    #                     val_arr = s[col].fillna(0.0).to_numpy(dtype=np.float32)
    #                     obs[col] = torch.tensor(val_arr, device=self.device)
    #
    #         except (KeyError, ValueError):
    #             pass
    #
    #     return obs
    def _build_obs(self, symbols_df, night_index, rocket_arr=None, night_record=None, **kwargs):
        """
        Builds observation tensors for PyTorch execution.

        Primary source: night_record (holds ingested aggregates including N3% and ep_n3_activity_mean).
        Fallback source: symbols_df (for C1-C4 symbolic features if absent in night_record).
        """
        idx = pd.MultiIndex.from_tuples(
            night_index, names=["participant_id", "night_date"]
        )
        obs = {}

        # 1. Primary Source: Ingested night_record
        if night_record is not None:
            try:
                nr = (
                    night_record.reindex(idx)
                    if isinstance(night_record.index, pd.MultiIndex)
                    else night_record.reset_index()
                    .set_index(["participant_id", "night_date"])
                    .reindex(idx)
                )

                # C5 Constraint Metrics:
                # - Dreem N3 Percentage (0.0 to 1.0)
                if "N3_pct" in nr.columns:
                    n3_vals = nr["N3_pct"].fillna(0.0).to_numpy(dtype=np.float32)
                    if np.max(n3_vals) > 1.0:
                        n3_vals /= 100.0
                    obs["n3_pct"] = torch.tensor(n3_vals, device=self.device)

                # - EmbracePlus Wrist Motion STRICTLY during Dreem N3 epochs
                if "ep_n3_activity_mean" in nr.columns:
                    n3_motion = nr["ep_n3_activity_mean"].fillna(0.0).to_numpy(dtype=np.float32)
                    obs["embrace_n3_motion"] = torch.tensor(n3_motion, device=self.device)

                # - Dreem Signal Quality Index (0.0 to 1.0)
                if "recording_quality" in nr.columns:
                    sqi = nr["recording_quality"].fillna(0.0).to_numpy(dtype=np.float32)
                    obs["eeg_sqi"] = torch.tensor(sqi, device=self.device)

                # R9/R2/R5 medication gating (Chapter 4 medication redesign):
                # explicit named pull, same pattern as n3_pct/embrace_n3_motion
                # above, since these are not raw EEG-architecture measures and
                # do not belong in the generic OBS_EEG_COLS loop below.
                # NEVER previously pulled here at all -- r9_medication_context's
                # old "medication_status" requirement was never satisfied by
                # this function, meaning r9 has been silently inert (returning
                # zero every night) regardless of any fix applied to the
                # constraint functions themselves. Assumes night_record has
                # already been joined against med_ctx (or equivalent) upstream
                # to carry these two columns -- if it hasn't, they will simply
                # be absent here too, and the same silent-zero failure mode
                # continues one step further upstream. Verify night_record's
                # actual columns before trusting this fix ends the gap.
                for med_col, obs_key in [
                    ("med_has_stimulant", "med_has_stimulant"),
                    ("med_has_melatonin", "med_has_melatonin"),
                ]:
                    if med_col in nr.columns:
                        med_vals = nr[med_col].fillna(0.0).to_numpy(dtype=np.float32)
                        obs[obs_key] = torch.tensor(med_vals, device=self.device)
                #####added  9/11
                diary_cols = [c for c in ("ANX", "SQ", "DIFF") if c in nr.columns]
                if diary_cols:
                    diary_missing = nr[diary_cols].isnull().all(axis=1).to_numpy()
                    obs["diary_missing"] = torch.tensor(
                        diary_missing, dtype=torch.bool, device=self.device
                    )
                ####################

                # Extract any general OBS_EEG_COLS required by other constraints (e.g. C1-C4)
                for col in getattr(self, "OBS_EEG_COLS", []):
                    if col in nr.columns and col not in obs:
                        val_arr = nr[col].fillna(0.0).to_numpy(dtype=np.float32)
                        obs[col] = torch.tensor(val_arr, device=self.device)

            except (KeyError, ValueError):
                pass

        # 2. Secondary Fallback Source: symbols_df (for C1-C4 constraints)
        if symbols_df is not None:
            try:
                s = symbols_df.reindex(idx)
                for col in getattr(self, "OBS_EEG_COLS", []):
                    if col not in obs and col in s.columns:
                        val_arr = s[col].fillna(0.0).to_numpy(dtype=np.float32)
                        obs[col] = torch.tensor(val_arr, device=self.device)

            except (KeyError, ValueError):
                pass

        return obs

    # ── Constraint setup ──────────────────────────────────────────────────────

    def _init_constraints(self, toml_path=None, sensitivity=False):
        """Initialise C_eeg (ConstraintFunctions) and C_smooth."""
        if not self.use_constraints:
            self._cf = None
            # self._c_smooth = None
            return

        # C_eeg + C_traj (via GBTM λ modifier)
        if toml_path is None:
            for candidate in [
                Path("configs/heuristics.toml"),
                Path("../configs/heuristics.toml"),
                Path(__file__).parent.parent.parent / "configs/heuristics.toml",
            ]:
                if candidate.exists():
                    toml_path = candidate
                    break

        if toml_path and Path(toml_path).exists():
            cfg = load_constraint_config(
                toml_path, sensitivity=sensitivity
            )
            # Allow runtime override of lambda_global for hyperparameter search
            if self._lambda_global_override is not None:
                cfg["lambda_global"] = self._lambda_global_override

            # ── Symmetric weighting: lambda_eeg is the relative weight of
            # C_eeg within L_constraints, exactly mirroring lambda_smooth's
            # role for C_smooth. Effective weight passed into ConstraintFunctions
            # is lambda_global × lambda_eeg, not lambda_global alone.
            _lambda_global = float(cfg.get("lambda_global", 0.5))
            _lambda_eeg = float(cfg.get("lambda_eeg", 1.0))
            cfg["lambda_global"] = _lambda_global * _lambda_eeg  # ConstraintFunctions reads this key

            self._cf = ConstraintFunctions(cfg, device=self.device)
            print(f"  [fusion] C_eeg + C_traj enabled "
                  f"(λ_eff={cfg['lambda_global']:.3f} = {_lambda_global}×{_lambda_eeg})")
        else:
            self._cf = None
            print("  [fusion] C_eeg disabled — heuristics.toml not found")



    def _calibrate_constraints(self, iv_arr, waso_arr=None, se_arr=None,
                               tst_arr=None):
        """
        Calibrate constraint thresholds from real cohort data.

        Parameters
        ----------
        iv_arr : np.array (n_nights,) — IV values, aligned to the same
            night order as X. No longer derived from rocket_arr — IV is
            an independently-computed EmbracePlus measure (see
            constraints.py's design note for why frag_score, a ROCKET-
            derived statistic, was replaced).
        waso_arr, se_arr, tst_arr : np.array (n_nights,), optional —
            WASO_min/SE_pct/TST_min, aligned the same way. When
            provided, also calibrates C1 (WASO tiers), C4 (SE floor,
            WASO ceiling), C6 (TST ceiling), C7 (SE floor) against this
            cohort's own percentiles instead of the fixed literature
            defaults — see constraints.py's calibrate_thresholds().
        """
        if self._cf is None:
            return
        self._iv_values = np.asarray(iv_arr, dtype=np.float32)
        self._cf.calibrate(self._iv_values)
        if waso_arr is not None or se_arr is not None or tst_arr is not None:
            self._waso_values = (np.asarray(waso_arr, dtype=np.float32)
                                 if waso_arr is not None else None)
            self._se_values = (np.asarray(se_arr, dtype=np.float32)
                               if se_arr is not None else None)
            self._tst_values = (np.asarray(tst_arr, dtype=np.float32)
                                if tst_arr is not None else None)
            self._cf.calibrate_thresholds(
                waso_values=self._waso_values,
                se_values=self._se_values,
                tst_values=self._tst_values,
            )

    # ── Unified training function ─────────────────────────────────────────────

    def _train_net(self, X_sc, Y_soft, obs=None,
                   gbtm_groups=None, gbtm_conf=None,
                   pid_arr=None, night_order_arr=None, night_index=None):
        """
        Train either SleepLogReg or SleepPhenotypeNet with a
        Lagrangian adaptive constraint objective following
        Arachie & Huang (2022) — DCWS.

        Instead of a fixed λ, two Lagrange multipliers γ_eeg and
        γ_smooth are updated by gradient ascent each epoch:

            γ ← max(0, γ + α_γ · C_k)   [ascent — γ rises when violated]
            θ ← θ - α_θ · ∇_θ L_total   [descent — standard backprop]

        At equilibrium the multipliers balance L_data and L_constraints
        automatically — no λ hyperparameter required.

        C_eeg and C_traj are computed per mini-batch.
        C_smooth is computed over participant sequences each epoch.

        Parameters
        ----------
        X_sc            : np.array (n, f) — scaled features
        Y_soft          : np.array (n, 8) — Snorkel soft labels
        obs             : dict of Tensors — observable features for C_eeg
        gbtm_groups     : np.array (n,)   — GBTM group per night (C_traj)
        gbtm_conf       : np.array (n,)   — GBTM confidence
        pid_arr         : np.array (n,)   — participant IDs (C_smooth)
        night_order_arr : np.array (n,)   — night sort order (C_smooth)
        """
        n, f = X_sc.shape
        net = self._make_net(f)
        opt = Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)

        # Cosine annealing: lr decays from self.lr to eta_min over all epochs
        # Prevents oscillation near convergence — standard for dual-loss objectives
        scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)

        X_t = torch.tensor(X_sc, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y_soft, dtype=torch.float32, device=self.device)

        g_t = (torch.tensor(gbtm_groups, dtype=torch.long,
                            device=self.device)
               if gbtm_groups is not None else None)
        gc_t = (torch.tensor(gbtm_conf, dtype=torch.float32,
                             device=self.device)
                if gbtm_conf is not None else None)

        # night order for C_smooth
        _pid_arr = pid_arr if pid_arr is not None else np.zeros(n)
        _ord_arr = night_order_arr if night_order_arr is not None else np.arange(n)

        arch = self._variant_label()

        # ── Step 1: Estimate constraint scales before training ────────────
        # Run one forward pass on a sample batch to measure the raw magnitude
        # of each constraint term. This is the diagnostic recommended in the
        # constraint design literature — you cannot tune λ meaningfully until
        # you know the scales you are balancing.
        # PURE PER-RULE LAGRANGIAN DESIGN
        # Each of C1-C7 gets its own multiplier gamma_k and its own scale.
        # No lambda_global / lambda_eeg anywhere in this path — weighting
        # is entirely learned, per rule, matching the "no lambda
        # hyperparameter required" premise of the Lagrangian approach.
        #

        _DEFAULT_DISABLED_RULES = {}  # "c5_cross_modal"
        _disabled = getattr(self, "_disabled_rules", None)
        if _disabled is None:
            _disabled = _DEFAULT_DISABLED_RULES
        RULE_NAMES = [k for k in (self._cf.RULE_NAMES if self._cf is not None else [])
                      if k not in _disabled]
        _SCALE_EPS = 1e-4
        _SCALE_MOMENTUM = 0.95  # EMA decay for the running scale estimate —
        # higher = slower/smoother adaptation, lower = tracks recent
        # batches more tightly. 0.95 means ~20-batch effective window.
        _scales = {k: 1.0 for k in RULE_NAMES}

        if self.use_constraints and self._cf is not None:
            net.eval()
            with torch.no_grad():
                _sample_idx = torch.arange(min(256, n), device=self.device)
                _Ysample = net(X_t[_sample_idx])
                if obs:
                    _obs_s = {k: v[_sample_idx] for k, v in obs.items()
                              if torch.is_tensor(v) and v.shape[0] == n}
                    _gb_s = g_t[_sample_idx] if g_t is not None else None
                    _gc_s = gc_t[_sample_idx] if gc_t is not None else None

                    _per_rule_mag = self._cf.compute_per_rule(
                        _Ysample, _obs_s, _gb_s, _gc_s
                    )
                    for k, v in _per_rule_mag.items():
                        _scales[k] = max(v.item(), _SCALE_EPS)

            if self._verbose_constraints:
                mag_str = "  ".join(f"{k}={_scales[k]:.5f}" for k in RULE_NAMES)
                print(f"  [constraints] per-rule raw magnitudes — {mag_str}")
            net.train()

        # ── Per-rule Lagrangian multipliers ──────────────────────────────
        # gamma_k starts at 0 and is updated by SIGNED slack (see
        # ConstraintFunctions.compute_signed_slacks) — can rise while
        # violated and fall back once satisfied with margin, unlike a
        # squared-hinge-driven ascent which can only ever increase.
        _alpha_gamma_base = 0.001  # lowered from 0.01 — with c4/c6/c7 now live

        _GAMMA_MAX = 5.0  # lowered from 50.0 — same rationale; a single

        _gamma_ceiling = {k: _GAMMA_MAX
                          for k in RULE_NAMES}
        _gammas = {k: torch.tensor(0.0, device=self.device) for k in RULE_NAMES}

        # Reset per-call — each fit()/lopo_cv-fold call should get its own
        # fresh history, not accumulate across folds into one mixed list.
        self._ldata_history = []
        self._lc_history = []
        self._grad_conflict_history = []

        net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            e_data = 0.0
            e_ceeg = 0.0

            # Decay alpha_gamma in lockstep with the LR schedule. Without
            # this, gamma keeps ascending at full speed even in late
            # epochs when lr -> eta_min and theta has essentially stopped
            # moving — nothing is left to resolve the violation gamma is
            # reacting to, so L_c swells against a static C_k (observed:
            # L_c climbing from ~2-3x to ~5-11x L_data between epoch 100
            # and 150 in several LOPO folds). Scaling by the same lr ratio
            # keeps dual and primal updates winding down together.
            _lr_ratio = scheduler.get_last_lr()[0] / self.lr
            _alpha_gamma = _alpha_gamma_base * _lr_ratio

            # ── Mini-batch loop: L_data + Σ_k γ_k · C_k ──────────────────
            for start in range(0, n, self.batch_size):
                idx_b = perm[start:start + self.batch_size]
                Xb, Yb = X_t[idx_b], Y_t[idx_b]

                Yhat = net(Xb)
                L_data = -(Yb * torch.log(Yhat + 1e-8)).sum(1).mean()

                # Per-rule penalties — primal loss term, one γ_k per rule
                L_c_eeg = torch.tensor(0.0, device=self.device)
                per_rule_pen = {}
                obs_b, gb, gcb = None, None, None
                if self.use_constraints and self._cf is not None and obs:
                    obs_b = {k: v[idx_b] for k, v in obs.items()
                             if torch.is_tensor(v) and v.shape[0] == n}
                    gb = g_t[idx_b] if g_t is not None else None
                    gcb = gc_t[idx_b] if gc_t is not None else None
                    night_index_b = ([night_index[i] for i in idx_b.tolist()] if night_index is not None else None)
                    per_rule_pen = self._cf.compute_per_rule(Yhat, obs_b, gb, gcb,night_index_b)

                    # Running scale update (EMA) — the pre-training snapshot
                    # in _scales was measured once, on an untrained model,
                    # before any weights moved. As training proceeds the
                    # true raw magnitude of a constraint term can drift far
                    # past that snapshot (observed: L_c reaching 100x+
                    # L_data in some LOPO folds even with gamma capped at
                    # 5.0), because gamma*(C_k/scale) stays large whenever
                    # the live C_k has outgrown the frozen scale — capping
                    # gamma alone cannot fix that, only re-normalising can.
                    # Detached: this is bookkeeping for the loss weight,
                    # not a quantity we want gradients flowing through.
                    with torch.no_grad():
                        for k in RULE_NAMES:
                            _scales[k] = max(
                                _SCALE_MOMENTUM * _scales[k]
                                + (1 - _SCALE_MOMENTUM) * per_rule_pen[k].item(),
                                _SCALE_EPS
                            )

                    for k in RULE_NAMES:
                        L_c_eeg = L_c_eeg + _gammas[k] * (per_rule_pen[k] / _scales[k])

                L_batch = L_data + L_c_eeg

                # ── Gradient surgery (PCGrad-style) ──────────────────────
                # Directly targets the measured mechanism: L_data and
                # L_constraints gradients were found to conflict (negative
                # cosine similarity) in a majority of late-training batches,
                # correlating with L_data plateauing/worsening and in-sample
                # kappa collapsing to ~0. Rather than summing the two
                # gradients as-is (which lets a conflicting L_c gradient
                # directly undo L_data's progress), we project out the
                # conflicting component of L_c's gradient before combining:
                #   if cos(g_data, g_c) < 0:
                #       g_c' = g_c - (g_c . g_data / |g_data|^2) * g_data
                #   else:
                #       g_c' = g_c   (no conflict — sum as normal)
                # This costs two backward passes per batch instead of one
                # (real overhead) but is the direct, targeted fix rather
                # than an indirect one (e.g. hoping more model capacity
                # happens to reduce conflict). Toggle via
                # self._use_gradient_surgery (default True whenever
                # constraints are active) — set False to fall back to a
                # plain combined backward pass.
                _use_surgery = (getattr(self, "_use_gradient_surgery", True)
                                and self.use_constraints and self._cf is not None
                                and obs and L_c_eeg.item() > 0)

                _surviving_frac = 1.0  # default: no surgery this batch -> full ascent rate

                if _use_surgery:
                    params = [p for p in net.parameters() if p.requires_grad]

                    opt.zero_grad()
                    L_data.backward(retain_graph=True)
                    g_data = [p.grad.detach().clone() if p.grad is not None
                              else torch.zeros_like(p) for p in params]

                    opt.zero_grad()
                    L_c_eeg.backward()
                    g_c = [p.grad.detach().clone() if p.grad is not None
                           else torch.zeros_like(p) for p in params]

                    flat_data = torch.cat([g.flatten() for g in g_data])
                    flat_c = torch.cat([g.flatten() for g in g_c])
                    dot = torch.dot(flat_data, flat_c)
                    _norm_data = flat_data.norm()
                    _norm_c = flat_c.norm()
                    _cos = (dot / (_norm_data * _norm_c + 1e-12)).item()

                    if dot.item() < 0:
                        proj_coef = dot / (_norm_data ** 2 + 1e-12)
                        flat_c_final = flat_c - proj_coef * flat_data
                        idx = 0
                        g_c_final = []
                        for g in g_c:
                            k_ = g.numel()
                            g_c_final.append(flat_c_final[idx:idx + k_].view_as(g))
                            idx += k_
                        # Fraction of the constraint gradient's norm that
                        # survived projection. Feeds into the gamma ascent
                        # step below — without this, gamma keeps climbing
                        # at full rate every batch regardless of how much
                        # of its influence surgery just neutralised, chasing
                        # a loss that heavy, sustained projection prevents
                        # from ever shrinking (observed: L_c grew to 3-5x
                        # L_data by epoch 100 despite surgery, with gamma
                        # still ascending unchecked in 814/889 batches).
                        # Ascent gate: max(0, 1+cos)/2, NOT the surviving
                        # gradient norm. The norm-based version (‖g_c'‖/‖g_c‖)
                        # only approaches 0 as cos approaches exactly -1 —
                        # at cos=-0.9 it's still ~0.44 (sqrt(1-cos^2)), so
                        # gamma kept climbing at ~44% rate even under severe,
                        # near-maximal conflict (observed: L_c still reached
                        # 2.8x L_data with that version). This cosine-based
                        # gate goes to ~0.05 at cos=-0.9, directly throttling
                        # ascent in proportion to how conflicting the batch
                        # actually was, not how much orthogonal magnitude
                        # happened to survive projection.
                        _surviving_frac = max(0.0, 1.0 + _cos) / 2.0
                    else:
                        g_c_final = g_c
                        _surviving_frac = 1.0

                    opt.zero_grad()
                    for p, gd, gc_ in zip(params, g_data, g_c_final):
                        p.grad = gd + gc_
                    opt.step()

                    if not hasattr(self, "_grad_conflict_history"):
                        self._grad_conflict_history = []
                    self._grad_conflict_history.append((epoch, _cos, dot.item() < 0, _surviving_frac))
                    if (getattr(self, "_track_grad_conflict", False)
                            and self._verbose_constraints and start == 0):
                        _sign = "CONFLICT (projected)" if dot.item() < 0 else (
                            "aligned" if _cos > 0.1 else "orthogonal")
                        print(f"        cos(∇L_data, ∇L_c) = {_cos:+.4f}  ({_sign})")
                else:
                    opt.zero_grad()
                    L_batch.backward()
                    opt.step()

                e_data += L_data.item()
                e_ceeg += L_c_eeg.item() if self.use_constraints else 0.0

                # -- γ_k ascent/descent: signed slack, per rule (Lagrangian) or fixed λ --
                if self.use_constraints and self._cf is not None and obs:
                    if self.use_lagrangian:
                        with torch.no_grad():
                            slacks = self._cf.compute_signed_slacks(
                                Yhat.detach(), obs_b, gb, gcb
                            )
                            for k in RULE_NAMES:
                                _gammas[k] = torch.clamp(
                                    # REVERTED: no longer scaled by
                                    # _surviving_frac. Two throttle variants
                                    # (norm-based, cosine-based) were tested
                                    # and both converged on the same kappa
                                    # (0.051) with REDUCED plausibility gain
                                    # (-6.19pp vs -8.52pp untouched) — pure
                                    # ascent (no throttle) is the best-known
                                    # config found so far. _surviving_frac is
                                    # still computed/logged above for
                                    # diagnostics, just not applied here.
                                    _gammas[k] + _alpha_gamma
                                    * (slacks[k] / _scales[k]),
                                    min=0.0, max=_gamma_ceiling[k]
                                    # clip: prevents explosion; C4 family gets a shared lower cap — see definition above
                                )
                    else:
                        # fixed λ — each γ_k stays at 1.0 throughout
                        for k in RULE_NAMES:
                            _gammas[k] = torch.tensor(1.0, device=self.device)

            # ── Participant-level: C_smooth + γ_smooth ascent ────────────
            # Computed over full sequence each epoch — participant ordering
            # required for temporal coherence penalty.
            # if self.use_constraints and self._c_smooth is not None and pid_arr is not None:
            #     net.train()
            #     Y_full = net(X_t)
            #     _raw_smooth = self._c_smooth.compute(Y_full, _pid_arr, _ord_arr)
            #     #L_smooth = _gamma_smooth * (_raw_smooth / _scale_smooth)
            # else:
            #     L_smooth = torch.tensor(0.0, device=self.device)
            #     _raw_smooth = torch.tensor(0.0, device=self.device)

            # θ descent on C_smooth
            # if L_smooth.item() > 0:
            #     opt.zero_grad()
            #     L_smooth.backward()
            #     torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            #     opt.step()

            # γ_smooth ascent — rises when temporal coherence is violated
            # if self.use_constraints:
            #     if self.use_lagrangian:
            #         with torch.no_grad():
            #             _gamma_smooth = torch.clamp(
            #                 _gamma_smooth + _alpha_gamma * (_raw_smooth.detach()
            #                                                 / _scale_smooth),
            #                 min=0.0, max=50.0  # clip: prevents explosion
            #             )
            #     else:
            #         _gamma_smooth = torch.tensor(1.0, device=self.device)
            # # Cosine annealing step — exactly once per epoch
            scheduler.step()

            # Always-on, cheap: full-resolution per-epoch loss history.
            # No extra computation — just recording e_data/e_ceeg (already
            # computed for the print block below) every epoch instead of
            # only at the three 50-epoch checkpoints. Lets you plot the
            # actual L_data trajectory shape (plateauing vs decreasing)
            # rather than inferring it from 3 sparse points.
            _n_batches = max(1, n // self.batch_size)
            self._ldata_history.append(e_data / _n_batches)
            self._lc_history.append(e_ceeg / _n_batches)

            if (epoch + 1) % 50 == 0:
                _cur_lr = scheduler.get_last_lr()[0]
                _l_data_ep = e_data / max(1, n // self.batch_size)
                _l_c_ep = e_ceeg / max(1, n // self.batch_size)
                _gamma_str = "  ".join(f"γ_{k}={_gammas[k].item():.3f}"
                                       for k in RULE_NAMES) if RULE_NAMES else ""
                print(f"      [{arch}] epoch {epoch + 1}/{self.epochs}  "
                      f"L_data={_l_data_ep:.4f}  "
                      f"L_c={_l_c_ep:.4f}  "
                      f"α_γ={_alpha_gamma:.2e}  "
                      f"lr={_cur_lr:.2e}")
                if _gamma_str and self._verbose_constraints:
                    print(f"        {_gamma_str}")
                if self.use_constraints and _l_data_ep > 0 and _l_c_ep > 3 * _l_data_ep:
                    _pinned = [k for k in RULE_NAMES
                               if _gammas[k].item() >= _gamma_ceiling[k] - 1e-6]
                    _pinned_str = (", ".join(_pinned) if _pinned
                                   else "none currently pinned")
                    print(f"        ⚠ L_c is {_l_c_ep / _l_data_ep:.1f}x L_data — "
                          f"constraint term may be dominating the objective; "
                          f"rules at their ceiling: {_pinned_str} "
                          f"(ceilings: {_GAMMA_MAX} standard, ")
                          #f"{_GAMMA_MAX / len(_C4_FAMILY):.2f} for the c4_suppress_* family)")

        # store final per-rule gammas for post-hoc diagnostics
        self._last_gammas = {k: _gammas[k].item() for k in RULE_NAMES}
        net.eval()
        return net

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, rocket_df, symbols_df, context_df, label_result,
            clinical_df=None, night_record=None,
            gbtm_groups_df=None, sensitivity=False,
            save_rocket_matrix=None):

        self._init_constraints(self._constraint_cfg, sensitivity=sensitivity)
        #self._cf._detailed_trigger_log = True

        labelled = label_result[label_result["hard_label"] >= 0].copy()
        if len(labelled) == 0:
            raise ValueError("No labelled nights to train on")

        night_index = list(zip(
            labelled["participant_id"], labelled["night_date"]
        ))

        X = self._build_X(rocket_df, symbols_df, None, night_index, clinical_df)
        self.feature_names_ = list(X.columns)

        prob_cols = [f"prob_label_{i}" for i in range(N_LABELS)]
        Y_soft = labelled[prob_cols].values.astype(np.float32)

        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        X_imp = self._imputer.fit_transform(X.values.astype(np.float32))
        X_sc = self._scaler.fit_transform(X_imp)

        rocket_cols = [c for c in X.columns
                       if c.startswith("rocket_") or c.startswith("r_")]
        rocket_arr = (X[rocket_cols].values.astype(np.float32)
                      if rocket_cols else X_sc)

        # Build the IV array aligned to night_index for calibration —
        # replaces the previous rocket_arr.std() derivation. night_record
        # is Layer 1's aligned table (merge_rar_into_night_record already
        # joined IV on); reindex to night_index the same way _build_obs
        # does, so calibration and training see the exact same alignment.
        # Also build WASO_min/SE_pct/TST_min the same way, to calibrate
        # C1/C4/C6/C7 against this cohort's own percentiles rather than
        # the fixed literature defaults — see constraints.py's
        # calibrate_thresholds().
        if night_record is not None and "IV" in night_record.columns:
            _nr_idx = pd.MultiIndex.from_tuples(
                night_index, names=["participant_id", "night_date"]
            )
            _nr_cal = (night_record.reindex(_nr_idx)
                       if isinstance(night_record.index, pd.MultiIndex)
                       else night_record.reset_index()
                       .set_index(["participant_id", "night_date"])
                       .reindex(_nr_idx))
            iv_arr = _nr_cal["IV"].fillna(0).values.astype(np.float32)
            waso_arr = (_nr_cal["WASO_min"].fillna(0).values.astype(np.float32)
                        if "WASO_min" in _nr_cal.columns else None)
            se_arr = (_nr_cal["SE_pct"].fillna(0).values.astype(np.float32)
                      if "SE_pct" in _nr_cal.columns else None)
            tst_arr = (_nr_cal["TST_min"].fillna(0).values.astype(np.float32)
                       if "TST_min" in _nr_cal.columns else None)
        else:
            iv_arr = None
            waso_arr = se_arr = tst_arr = None
            print("  WARNING: night_record missing or has no 'IV' column — "
                  "C2/C3 cannot be calibrated this run (will stay at "
                  "config defaults / uncalibrated). Check merge_rar_into_"
                  "night_record() ran before this call.")

        if iv_arr is not None:
            self._calibrate_constraints(iv_arr, waso_arr=waso_arr,
                                        se_arr=se_arr, tst_arr=tst_arr)

        if save_rocket_matrix:
            np.save(save_rocket_matrix, rocket_arr)

        obs = self._build_obs(symbols_df, night_index, rocket_arr,
                              night_record=night_record)

        g_arr, gc_arr = None, None
        if gbtm_groups_df is not None:
            pids = labelled["participant_id"].values
            g_arr = np.array([
                int(gbtm_groups_df.loc[p, "dominant_group"])
                if p in gbtm_groups_df.index else -1
                for p in pids
            ], dtype=np.int64)
            gc_arr = np.array([
                float(gbtm_groups_df.loc[p, "confidence"])
                if p in gbtm_groups_df.index else 0.0
                for p in pids
            ], dtype=np.float32)

        # night order for C_smooth — sort index within participant
        pid_arr = labelled["participant_id"].values
        date_arr = pd.to_datetime(labelled["night_date"].values)
        ord_arr = np.zeros(len(labelled), dtype=np.int64)
        for pid in np.unique(pid_arr):
            mask = pid_arr == pid
            dates_pid = date_arr[mask]
            ord_arr[mask] = np.argsort(np.argsort(dates_pid))

        arch = "MLP" if self.use_mlp else "LogReg"
        cons = "L_data + L_constraints" if self.use_constraints else "L_data only"
        print(f"  [fusion] training {arch} — {cons} "
              f"({X_sc.shape[1]} features, {len(labelled)} nights, "
              f"{self.epochs} epochs)")

        self._net = self._train_net(
            X_sc, Y_soft, obs, g_arr, gc_arr,
            pid_arr=pid_arr, night_order_arr=ord_arr, night_index=night_index
        )
        self._fitted = True
        return self

    def predict_proba(self, rocket_df, symbols_df, context_df,
                      night_index, clinical_df=None):
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        X = self._build_X(
            rocket_df, symbols_df, None, night_index, clinical_df
        )
        X_imp = self._imputer.transform(X.values.astype(np.float32))
        X_sc = self._scaler.transform(X_imp)
        X_t = torch.tensor(X_sc, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self._net.eval()
            proba = self._net(X_t).cpu().numpy()
        return proba

    def predict(self, rocket_df, symbols_df, context_df,
                night_index, clinical_df=None):
        return self.predict_proba(
            rocket_df, symbols_df, context_df, night_index, clinical_df
        ).argmax(axis=1)

    # used for passing soft label to stratification method
    def predict_proba_df(self, rocket_df, symbols_df, context_df,
                         night_index, clinical_df=None):
        """
        Predicts continuous night probabilities and returns a DataFrame
        ready for StratificationLayer (phenotype.py).
        """
        probs = self.predict_proba(
            rocket_df, symbols_df, context_df, night_index, clinical_df
        )  # shape: (N_nights, 6)

        pids, dates = zip(*night_index)

        df = pd.DataFrame({
            "participant_id": pids,
            "night_date": dates,
            "hard_label": probs.argmax(axis=1),
            "all_abstained": False,  # Model predicts all evaluated nights
        })

        # Attach continuous probability columns matching N_LABELS=6
        for i in range(N_LABELS):
            df[f"prob_label_{i}"] = probs[:, i]

        return df

    def lopo_cv(self, rocket_df, symbols_df, context_df, label_result,
                clinical_df=None, night_record=None,
                gbtm_groups_df=None, sensitivity=False):
        """
        Leave-one-participant-out cross-validation.

        Works for all four ablation variants — architecture and loss function
        are controlled by self.use_mlp and self.use_constraints.
        C_smooth is computed within each training fold using participant
        ordering from that fold only (no leakage).
        """
        self._init_constraints(self._constraint_cfg, sensitivity=sensitivity)

        ############just added###########################

        # Use IV/WASO/SE/TST values from fit() rather than recomputing per
        # fold. This ensures constraint calibration is consistent with
        # training.
        if self._iv_values is not None and self._cf is not None:
            self._cf.calibrate(self._iv_values)  # restore fit() calibration
            print(f"  [lopo_cv] using fit() IV calibration: "
                  f"iv_low={self._cf._iv_low:.4f}, "
                  f"iv_high={self._cf._iv_high:.4f}")
        if (self._cf is not None and
                (self._waso_values is not None or
                 self._se_values is not None or
                 self._tst_values is not None)):
            self._cf.calibrate_thresholds(
                waso_values=self._waso_values,
                se_values=self._se_values,
                tst_values=self._tst_values,
            )
            print(f"  [lopo_cv] using fit() threshold calibration: "
                  f"c1_mod={self._cf._c1_waso_mod:.2f}, "
                  f"c1_ext={self._cf._c1_waso_ext:.2f}, "
                  f"c4_se={self._cf._c4_se_min:.2f}, "
                  f"c4_waso={self._cf._c4_waso_max:.2f}, "
                  f"c6_tst={self._cf._c6_tst_min:.2f}, "
                  f"c7_se={self._cf._c7_se_max:.2f}")
        ####################just added ##################

        labelled = (label_result[label_result["hard_label"] >= 0]
                    .copy().reset_index(drop=True))
        pids = labelled["participant_id"].unique()

        if len(labelled) == 0 or len(pids) < 2:
            raise ValueError("Need ≥2 participants for LOPO-CV")

        all_idx = list(zip(
            labelled["participant_id"], labelled["night_date"]
        ))
        X_full_df = self._build_X(
            rocket_df, symbols_df, None, all_idx, clinical_df
        )
        X_raw = X_full_df.values.astype(np.float32)

        prob_cols = [f"prob_label_{i}" for i in range(N_LABELS)]
        y_soft = labelled[prob_cols].values.astype(np.float32)
        y_hard = labelled["hard_label"].values.astype(int)
        pid_arr = labelled["participant_id"].values

        # night order for C_smooth
        date_arr = pd.to_datetime(labelled["night_date"].values)
        ord_arr = np.zeros(len(labelled), dtype=np.int64)
        for pid in np.unique(pid_arr):
            mask = pid_arr == pid
            ord_arr[mask] = np.argsort(np.argsort(date_arr[mask]))

        # GBTM arrays
        g_arr, gc_arr = None, None
        if gbtm_groups_df is not None:
            g_arr = np.array([
                int(gbtm_groups_df.loc[p, "dominant_group"])
                if p in gbtm_groups_df.index else -1
                for p in pid_arr
            ], dtype=np.int64)
            gc_arr = np.array([
                float(gbtm_groups_df.loc[p, "confidence"])
                if p in gbtm_groups_df.index else 0.0
                for p in pid_arr
            ], dtype=np.float32)

        # obs feature lookup
        nr_obs_arr = {}
        if night_record is not None:
            try:
                idx_full = pd.MultiIndex.from_tuples(
                    all_idx, names=["participant_id", "night_date"]
                )
                nr_full = (night_record.reindex(idx_full)
                           if isinstance(night_record.index, pd.MultiIndex)
                           else night_record.reset_index()
                           .set_index(["participant_id", "night_date"])
                           .reindex(idx_full))
                for col in self.OBS_EEG_COLS:
                    if col in nr_full.columns:
                        nr_obs_arr[col] = nr_full[col].fillna(0).values.astype(
                            np.float32
                        )
            except Exception:
                pass

        rocket_cols = [c for c in X_full_df.columns
                       if c.startswith("rocket_") or c.startswith("r_")]
        r_mask = np.zeros(X_raw.shape[1], dtype=bool)
        for c in rocket_cols:
            r_mask[X_full_df.columns.get_loc(c)] = True

        all_true, all_pred = [], []
        all_pred_soft, all_test_idx = [], []
        n_pids = len(pids)

        for i, pid in enumerate(pids):
            if i % 10 == 0 or i == n_pids - 1:
                print(f"    [{self._variant_label()}] "
                      f"LOPO: {i + 1}/{n_pids}", end="\r")

            train_mask = pid_arr != pid
            test_mask = pid_arr == pid
            if train_mask.sum() < N_LABELS or test_mask.sum() == 0:
                continue

            # impute + scale on training fold only
            imp = SimpleImputer(strategy="median")
            X_tr_imp = imp.fit_transform(X_raw[train_mask])
            X_te_imp = imp.transform(X_raw[test_mask])
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr_imp)
            X_te_sc = sc.transform(X_te_imp)

            # calibrate C_eeg thresholds on training fold.
            # IV (unlike the old frag_score) has no relationship to ROCKET
            # output, so there is no legitimate ROCKET-derived fallback —
            # nr_obs_arr["IV"] (sourced from night_record via OBS_EEG_COLS)
            # is the single true source, used whether or not fit() was
            # called first on this instance. This also removes the old
            # two-path fallback (self._frag_scores vs. recomputed-from-
            # X_tr_imp) that was a real source of scale-mismatch bugs
            # earlier in this investigation — one source now, not two.
            if self._cf is not None:
                if "IV" in nr_obs_arr:
                    iv_tr = nr_obs_arr["IV"][train_mask]
                    self._cf.calibrate(iv_tr.astype(np.float32))
                else:
                    print("  WARNING: no IV in nr_obs_arr for this fold — "
                          "C2/C3 uncalibrated (check night_record has IV "
                          "and was passed to lopo_cv())")
                # Same per-fold hygiene for C1/C4/C6/C7: recalibrate from
                # this fold's training data only, so the held-out
                # participant never informs the threshold used to judge
                # them (matches the IV calibration's existing pattern
                # above, not just added for symmetry).
                _waso_tr = (nr_obs_arr["WASO_min"][train_mask]
                            if "WASO_min" in nr_obs_arr else None)
                _se_tr = (nr_obs_arr["SE_pct"][train_mask]
                          if "SE_pct" in nr_obs_arr else None)
                _tst_tr = (nr_obs_arr["TST_min"][train_mask]
                           if "TST_min" in nr_obs_arr else None)
                if _waso_tr is not None or _se_tr is not None or _tst_tr is not None:
                    self._cf.calibrate_thresholds(
                        waso_values=_waso_tr, se_values=_se_tr, tst_values=_tst_tr
                    )
            # build obs features for this fold
            obs_tr = {}
            for col, arr_full in nr_obs_arr.items():
                obs_tr[col] = torch.tensor(
                    arr_full[train_mask], device=self.device
                )

            # GBTM slices
            g_tr = g_arr[train_mask] if g_arr is not None else None
            gc_tr = gc_arr[train_mask] if gc_arr is not None else None

            # pid and order arrays for C_smooth (training fold only)
            pid_tr = pid_arr[train_mask]
            ord_tr = ord_arr[train_mask]

            # train fold model — suppress per-fold constraint prints
            self._verbose_constraints = False
            fold_net = self._train_net(
                X_tr_sc, y_soft[train_mask],
                obs_tr, g_tr, gc_tr,
                pid_arr=pid_tr, night_order_arr=ord_tr
            )
            self._verbose_constraints = True

            X_te_t = torch.tensor(
                X_te_sc, dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                fold_net.eval()
                probs = fold_net(X_te_t).cpu().numpy()
                preds = probs.argmax(axis=1)

            all_pred.extend(preds.tolist())
            all_true.extend(y_hard[test_mask].tolist())
            all_pred_soft.append(probs)
            all_test_idx.extend(np.where(test_mask)[0].tolist())

        print()

        # ── Assemble held-out predictions + obs for violation-rate diagnostics ──
        # Every night here was predicted by a model that never saw it during
        # training (its participant was the held-out fold) — this is the
        # basis for a genuine held-out plausibility comparison, as opposed
        # to the in-sample one computed elsewhere off model._net directly.
        held_idx = np.array(all_test_idx)
        self._last_lopo_Y_pred = torch.tensor(
            np.concatenate(all_pred_soft, axis=0), dtype=torch.float32
        )
        held_obs = {}
        for col in self.OBS_EEG_COLS:
            if col in nr_obs_arr:
                held_obs[col] = torch.tensor(nr_obs_arr[col][held_idx],
                                             dtype=torch.float32)
        if r_mask.any():
            rocket_held = X_raw[held_idx][:, r_mask]
            rocket_std_held = rocket_held.std(axis=1).astype(np.float32)
            held_obs["rocket_decile"] = torch.tensor(
                (pd.Series(rocket_std_held).rank(pct=True) * 10).values.astype(np.float32)
            )
        if "N3_pct" in held_obs and "rocket_decile" in held_obs:
            held_obs["eeg_quality_decile"] = torch.tensor(
                (pd.Series(held_obs["N3_pct"].numpy()).rank(pct=True) * 10)
                .values.astype(np.float32)
            )
        self._last_lopo_obs = held_obs
        # held_idx itself — lets a caller align against y_hard/pid_arr if needed
        self._last_lopo_idx = held_idx
        # participant ID per held-out row, same order as _last_lopo_Y_pred —
        # needed to aggregate per-night predictions up to a participant-level
        # phenotype (the Layer 4 -> Y_patient step the architecture diagram
        # specifies but which the pipeline currently only does from Layer 3's
        # hard_label, never from the model's own predictions).
        self._last_lopo_pids = pid_arr[held_idx]
        # true (weak) labels for the same held-out rows — needed for
        # bootstrap confidence intervals on kappa without externally
        # reconstructing the labelled/held-out row order (which has been
        # a source of subtle misalignment bugs elsewhere this session).
        self._last_lopo_y_true = y_hard[held_idx]

        at = np.array(all_true)
        ap = np.array(all_pred)

        kappa = cohen_kappa_score(at, ap)
        accuracy = float((at == ap).mean())
        report = classification_report(
            at, ap,
            labels=list(range(N_LABELS)),
            target_names=[f"label_{i}" for i in range(N_LABELS)],
            output_dict=True, zero_division=0,
        )

        return {
            "variant": self._variant_label(),
            "architecture": "MLP" if self.use_mlp else "LogReg",
            "constraints": self.use_constraints,
            "n_participants_evaluated": len(pids),
            "n_nights_evaluated": len(at),
            "overall_accuracy": round(accuracy, 4),
            "cohen_kappa": round(kappa, 4),
            "per_label": {
                f"label_{i}": {
                    "precision": round(report[f"label_{i}"]["precision"], 3),
                    "recall": round(report[f"label_{i}"]["recall"], 3),
                    "f1": round(report[f"label_{i}"]["f1-score"], 3),
                    "support": int(report[f"label_{i}"]["support"]),
                }
                for i in range(N_LABELS)
            },
        }

"""
EEG morphological features from aligned Dreem epoch time series.

Stage distribution percentages are deliberately excluded — they are
already used extensively in Layer 3 labelling rules and would create
a direct information leak from the symbolic layer into the fusion model.

Features extracted
------------------
Stage transitions
    count of each ordered transition type (N1<->N2, N2<->N3, N1<->REM,
    N3<->REM, etc.) — actual stage changes, not epoch-count proxies
    total transition count (genuine count of seq[i] != seq[i+1])
    transition rate per hour of sleep

First-occurrence timing
    minutes to first N3 epoch
    minutes to first REM epoch
    both relative to lights-off (first non-wake epoch)

Longest continuous bout
    longest run of each stage in minutes (N1, N2, N3, REM, Wake)

Epoch-level variability
    standard deviation of bout durations across the night
    coefficient of variation of bout durations

Entropy (two distinct, deliberately separate measures)
    stage_occupancy_entropy — order-insensitive; how evenly time
        splits across the stages that occur
    transition_entropy — order-sensitive; entropy of the empirical
        transition matrix, weighted by stage frequency; low = regular/
        predictable cycling, high = chaotic switching. This is the
        "predictability of stage cycling" signal; occupancy entropy
        alone does not capture it.

Wake bout count
    number of discrete wake episodes during sleep
"""

import numpy as np
import pandas as pd
from collections import Counter

# Dreem stage codes
WAKE = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4

SLEEP_STAGES = [N1, N2, N3, REM]

TRANSITIONS = [
    # --- Wake -> Sleep & Sleep -> Wake ---
    (WAKE, N1), (WAKE, N2), (WAKE, N3), (WAKE, REM),
    (N1, WAKE), (N2, WAKE), (N3, WAKE), (REM, WAKE),

    # --- Stepwise NREM Progressions & Lightening ---
    (N1, N2), (N2, N1),
    (N2, N3), (N3, N2),

    # --- Direct Stage Skipping (N1 <-> N3) ---
    (N1, N3), (N3, N1),

    # --- REM Interactions ---
    (N2, REM), (REM, N2),
    (N1, REM), (REM, N1),
    (N3, REM), (REM, N3),
]
TRANSITION_NAMES = [f"trans_{a}_{b}" for a, b in TRANSITIONS]


class EEGFeatureExtractor:

    def transform(self, epochs_df, night_index):
        """
        Extract EEG morphological features for each (participant, night).

        epochs_df is the aligned epoch DataFrame with columns:
            participant_id, night_date, sleep_stage
        indexed by grid_start timestamp.

        Returns DataFrame indexed by (participant_id, night_date).
        Nights with no EEG data get NaN features.
        """
        # group epochs by (participant_id, night_date) once before the loop
        # avoids masking a potentially multi-million row DataFrame per night
        print("    EEG symbols: indexing epochs...", end="\r")
        epoch_groups = {
            key: grp["sleep_stage"].values.astype(int)
            for key, grp in epochs_df.groupby(
                ["participant_id", "night_date"], sort=False
            )
        }

        records = []
        n_total = len(night_index)
        for idx, (pid, night_date) in enumerate(night_index):
            if idx % 100 == 0 or idx == n_total - 1:
                print(f"    EEG symbols: {idx + 1}/{n_total} nights", end="\r")

            seq = epoch_groups.get((pid, night_date), np.array([]))
            seq = seq[seq >= 0]  # Strip padding/unassigned epochs before computing metrics
            if len(seq) == 0:
                feats = self._nan_features()
            else:
                feats = self._extract(seq)

            feats["participant_id"] = pid
            feats["night_date"] = night_date
            records.append(feats)
        print()

        return pd.DataFrame(records).set_index(
            ["participant_id", "night_date"]
        ).sort_index()

    def _extract(self, seq):
        feats = {}

        # transitions
        trans_counts = Counter(zip(seq[:-1], seq[1:]))
        for (a, b), name in zip(TRANSITIONS, TRANSITION_NAMES):
            feats[name] = trans_counts.get((a, b), 0)
        feats["total_transitions"] = int(np.sum(seq[:-1] != seq[1:])) if len(seq) > 1 else 0
        sleep_dur_h = (seq != WAKE).sum() * 0.5 / 60.0  # 30s epochs → hours
        feats["transition_rate_per_h"] = (
            feats["total_transitions"] / sleep_dur_h
            if sleep_dur_h > 0 else 0.0
        )

        # first-occurrence timing (epochs × 0.5 min = minutes)
        feats["min_to_first_N3"] = _first_occurrence(seq, N3)
        feats["min_to_first_REM"] = _first_occurrence(seq, REM)

        # longest continuous bouts
        bouts = _run_lengths(seq)
        for stage, name in [(N1, "N1"), (N2, "N2"), (N3, "N3"), (REM, "REM"), (WAKE, "Wake")]:
            runs = bouts.get(stage, [])
            feats[f"longest_{name}_bout_min"] = (max(runs) * 0.5
                                                 if runs else 0.0)

        # epoch-level variability (all bout lengths, all stages)
        all_bouts = [r for runs in bouts.values() for r in runs]
        if len(all_bouts) > 1:
            mu = np.mean(all_bouts)
            sd = np.std(all_bouts, ddof=1)
            feats["bout_duration_std_min"] = round(float(sd * 0.5), 3)
            feats["bout_duration_cv"] = round(float(sd / mu) if mu > 0 else 0.0, 3)
        else:
            feats["bout_duration_std_min"] = 0.0
            feats["bout_duration_cv"] = 0.0

        # stage-occupancy entropy — how evenly time splits across the
        # stages that occur (order-INSENSITIVE: a rapidly-alternating
        # two-stage night and a night with the same total time-per-stage
        # but one clean transition get the same value here). Kept
        # because it's still a legitimate, distinct signal (balance of
        # time across stages), but renamed from "hypnogram_entropy" —
        # that name (and the module docstring) previously implied it
        # captured cycling predictability, which this metric does not.
        counts = Counter(seq)
        total = len(seq)
        probs = np.array([counts[s] / total for s in set(seq)])
        feats["stage_occupancy_entropy"] = round(
            float(-np.sum(probs * np.log2(probs + 1e-10))), 4
        )

        # transition entropy — order-SENSITIVE: the entropy of the
        # empirical transition matrix (for each stage, how unpredictable
        # its next-stage distribution is), weighted by how often each
        # stage is transitioned FROM. Low = regular/predictable cycling
        # (each stage reliably followed by the same next stage); high =
        # chaotic switching. This is what "predictability of stage
        # cycling" in the module docstring actually meant to describe.
        feats["transition_entropy"] = _transition_entropy(seq)

        # wake bout count (discrete wake episodes during sleep)
        feats["wake_bout_count"] = len(bouts.get(WAKE, []))

        return feats

    def _nan_features(self):
        feats = {name: np.nan for name in TRANSITION_NAMES}
        feats.update({
            "total_transitions": np.nan,
            "transition_rate_per_h": np.nan,
            "min_to_first_N3": np.nan,
            "min_to_first_REM": np.nan,
            "longest_N1_bout_min": np.nan,
            "longest_N2_bout_min": np.nan,
            "longest_N3_bout_min": np.nan,
            "longest_REM_bout_min": np.nan,
            "longest_Wake_bout_min": np.nan,
            "bout_duration_std_min": np.nan,
            "bout_duration_cv": np.nan,
            "stage_occupancy_entropy": np.nan,
            "transition_entropy": np.nan,
            "wake_bout_count": np.nan,
        })
        return feats

    def feature_names(self):
        dummy = self._nan_features()
        dummy.pop("participant_id", None)
        dummy.pop("night_date", None)
        return list(dummy.keys())


def _transition_entropy(seq):
    """
    Order-sensitive entropy of stage cycling. For each stage s that the
    sequence transitions FROM, compute the Shannon entropy of what it
    transitions TO next; average those per-stage entropies weighted by
    how often each stage is left. Low = each stage predictably followed
    by the same next stage (regular cycling). High = unpredictable,
    chaotic switching. Contrast with stage_occupancy_entropy, which
    only measures how evenly total time splits across stages and is
    blind to sequence order entirely.
    """
    if len(seq) < 2:
        return np.nan

    from_counts = Counter(seq[:-1])
    total_from = sum(from_counts.values())

    trans_counts = {}
    for a, b in zip(seq[:-1], seq[1:]):
        trans_counts.setdefault(a, Counter())[b] += 1

    weighted_entropy = 0.0
    for stage, from_count in from_counts.items():
        next_counts = trans_counts[stage]
        total_next = sum(next_counts.values())
        probs = np.array([c / total_next for c in next_counts.values()])
        stage_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        weighted_entropy += (from_count / total_from) * stage_entropy

    return round(float(weighted_entropy), 4)


def _first_occurrence(seq, stage):
    """Minutes to first occurrence of stage from first non-wake epoch."""
    sleep_start = next((i for i, s in enumerate(seq) if s != WAKE), None)
    if sleep_start is None:
        return np.nan
    for i in range(sleep_start, len(seq)):
        if seq[i] == stage:
            return round((i - sleep_start) * 0.5, 1)
    return np.nan


def _run_lengths(seq):
    """Return dict of stage → list of run lengths in epochs."""
    if len(seq) == 0:
        return {}
    runs = {}
    current = seq[0]
    count = 1
    for s in seq[1:]:
        if s == current:
            count += 1
        else:
            runs.setdefault(current, []).append(count)
            current = s
            count = 1
    runs.setdefault(current, []).append(count)
    return runs

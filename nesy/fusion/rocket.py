"""
ROCKET feature extraction from EmbracePlus nocturnal activity time series.

Applies random convolutional kernels to the activity time series within
the nocturnal window and extracts two features per kernel:
    - max pooling value
    - proportion of positive values (PPV)

Kernel count is set to 1000 — approximately one kernel per labelled
training night in the SSP cohort (N=2233). This avoids overfitting
in logistic regression with a 60-timepoint input series.

Reference
---------
Dempster A, Petitjean F, Webb GI (2020). ROCKET: Exceptionally fast and
accurate time series classification using random convolutional kernels.
Data Mining and Knowledge Discovery, 34(5):1454-1495.
"""

import numpy as np
import pandas as pd
from pathlib import Path


N_KERNELS    = 1000
MIN_KERNEL_LEN = 7   # minimum kernel length (odd)
MAX_KERNEL_LEN = 9   # maximum kernel length (odd)
RANDOM_SEED  = 42


class RocketExtractor:

    def __init__(self, n_kernels=N_KERNELS, window_start="23:00",
                 window_end="08:00", random_seed=RANDOM_SEED):
        self.n_kernels    = n_kernels
        self.window_start = window_start
        self.window_end   = window_end
        self.random_seed  = random_seed
        self._kernels     = None
        self._fitted      = False

    def fit(self, ep_ts):
        """
        Generate random kernels based on the nocturnal window's expected
        length (computed analytically from window_start/window_end, not
        from a probe extraction — see _expected_window_length).
        Call once before transform.
        """
        n = self._expected_window_length()
        self._kernels = _generate_kernels(n, self.n_kernels, self.random_seed)
        self._fitted  = True
        return self

    def _expected_window_length(self):
        """
        Expected number of timepoints in the nocturnal window, assuming
        1-minute-resolution EmbracePlus data (matches rar.py's 1440-
        minute-day convention). Computed directly from window_start/
        window_end so it's correct regardless of which participant or
        night is used at transform time — no dependency on a specific
        night's data existing.

        BUG FIX: the previous implementation tried to probe a real
        night via _extract_window(ep_ts, first_pid) anchored to
        pd.Timestamp("2000-01-01") — a date that cannot exist in any
        participant's real EmbracePlus data (collected 2023-2024), so
        the slice was always empty and n silently fell back to a
        hardcoded 60. With window_start=23:00/window_end=08:00 the
        real window is ~540 minutes, not 60 — kernel dilation
        (derived from n_timepoints) was therefore calibrated for a
        receptive field roughly 9x smaller than the actual night,
        making the kernels structurally blind to whole-night-scale
        patterns regardless of how they perform at inference time.
        """
        sh, sm = map(int, self.window_start.split(":"))
        eh, em = map(int, self.window_end.split(":"))
        start_min = sh * 60 + sm
        end_min   = eh * 60 + em
        if end_min <= start_min:  # window crosses midnight
            end_min += 24 * 60
        return end_min - start_min

    def transform(self, ep_ts, night_index):
        """
        Extract ROCKET features for each (participant, night) pair.

        night_index is a list of (participant_id, night_date) tuples
        defining which nights to extract features for.

        Returns DataFrame indexed by (participant_id, night_date)
        with 2*n_kernels columns.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        records = []
        df      = ep_ts.copy()
        df.index = pd.to_datetime(df.index, utc=True)

        n_total = len(night_index)
        for idx, (pid, night_date) in enumerate(night_index):
            if idx % 100 == 0 or idx == n_total - 1:
                print(f"    ROCKET: {idx+1}/{n_total} nights", end="\r")

            ts = self._extract_window_for_night(df, pid, night_date)
            if ts is None or len(ts) < MIN_KERNEL_LEN:
                feats = np.zeros(self.n_kernels * 2, dtype=np.float32)
            else:
                feats = _apply_kernels(ts.values.astype(np.float32),
                                       self._kernels)
            records.append({
                "participant_id": pid,
                "night_date":     night_date,
                **{f"rocket_{i}": float(feats[i])
                   for i in range(len(feats))}
            })
        print()  # newline after progress

        return pd.DataFrame(records).set_index(
            ["participant_id", "night_date"]
        ).sort_index()

    def fit_transform(self, ep_ts, night_index):
        return self.fit(ep_ts).transform(ep_ts, night_index)

    def _extract_window_for_night(self, df, pid, night_date):
        g      = df[df["participant_id"] == pid]["activity_counts"]
        anchor = pd.Timestamp(night_date)
        return self._window_slice(g, anchor)

    def _window_slice(self, series, anchor):
        sh, sm = map(int, self.window_start.split(":"))
        eh, em = map(int, self.window_end.split(":"))
        start  = anchor.replace(hour=sh, minute=sm, second=0, microsecond=0)
        if eh < sh:  # window crosses midnight
            end = (anchor + pd.Timedelta(days=1)).replace(
                hour=eh, minute=em, second=0, microsecond=0
            )
        else:
            end = anchor.replace(hour=eh, minute=em, second=0, microsecond=0)
        if series.index.tz is not None:
            start = start.tz_localize("UTC")
            end   = end.tz_localize("UTC")
        sliced = series[(series.index >= start) & (series.index < end)]
        return sliced.fillna(0.0) if not sliced.empty else None


def _generate_kernels(n_timepoints, n_kernels, seed):
    rng = np.random.default_rng(seed)
    kernels = []
    lengths = [7, 9]  # odd kernel lengths
    for _ in range(n_kernels):
        length  = rng.choice(lengths)
        weights = rng.standard_normal(length).astype(np.float32)
        weights -= weights.mean()   # zero-mean
        bias    = float(rng.uniform(-1, 1))
        dilation = int(2 ** rng.uniform(0, np.log2(
            max((n_timepoints - 1) / (length - 1), 1)
        )))
        padding = 0 if rng.random() < 0.5 else (length - 1) * dilation // 2
        kernels.append((weights, bias, dilation, padding))
    return kernels


def _apply_kernels(ts, kernels):
    """
    Vectorised kernel application using numpy stride tricks.
    ~50x faster than the pure Python loop implementation.
    """
    feats = np.empty(len(kernels) * 2, dtype=np.float32)
    n     = len(ts)

    for i, (w, b, d, p) in enumerate(kernels):
        # build dilated kernel as dense array
        k_len    = len(w)
        dilated  = np.zeros((k_len - 1) * d + 1, dtype=np.float32)
        dilated[::d] = w

        # pad if needed
        if p > 0:
            ts_p = np.pad(ts, p)
        else:
            ts_p = ts

        # numpy convolve (full → valid mode equivalent)
        out_len = len(ts_p) - len(dilated) + 1
        if out_len <= 0:
            feats[2*i]   = 0.0
            feats[2*i+1] = 0.0
            continue

        # use stride tricks for fast sliding dot product
        shape   = (out_len, len(dilated))
        strides = (ts_p.strides[0], ts_p.strides[0])
        windows = np.lib.stride_tricks.as_strided(ts_p, shape=shape,
                                                   strides=strides)
        out     = windows @ dilated + b

        feats[2*i]   = float(out.max())
        feats[2*i+1] = float((out > 0).mean())

    return feats

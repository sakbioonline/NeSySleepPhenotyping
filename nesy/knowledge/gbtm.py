"""
Group-based trajectory modelling for the NeSy knowledge layer.

ROCKET features in Layer 4 capture what happened on a single night.
They don't capture whether tonight is an isolated bad night for an
otherwise stable participant, or night 14 of a steadily deteriorating
pattern. GBTM answers that question.

Two EmbracePlus signals drive the clustering:
  daily mean HR     — sympathetic load across 24 hours
  nocturnal RMSSD   — parasympathetic recovery during detected sleep

A Gaussian mixture model clusters per-participant trajectory features
(mean, slope, std of each signal) into n_groups groups. Posterior
probabilities form context vector C which is passed to Layer 4 as a
conditioning signal alongside the nightly feature vectors.

C is scaled by confidence before entering the logistic regression:
  C_weighted = C.as_array() * confidence
High confidence → strong prior. Null vector (uniform probs) → near-zero
after scaling, so the model relies entirely on nightly features.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["hr_mean","hr_slope","hr_std","rmssd_mean","rmssd_slope","rmssd_std"]


@dataclass
class ContextVector:
    participant_id: str
    group_probs:    list
    confidence:     float
    n_nights:       int
    # D4 metadata — age_band: 0=early(10-13), 1=late(14-17), -1=unknown
    # sex_encoded: 0=Male, 1=Female, -1=unknown
    # autism_status: 1=autistic, 0=sibling
    age_band:       int = -1
    sex_encoded:    int = -1
    autism_status:  int = -1
    dominant_group: int = field(init=False)

    def __post_init__(self):
        self.dominant_group = int(np.argmax(self.group_probs))

    def as_array(self):
        # layout: [p1..pn, confidence, n_nights/30, age_band, sex, autism]
        return np.array(
            self.group_probs + [
                self.confidence,
                self.n_nights / 30.0,
                float(self.age_band),
                float(self.sex_encoded),
                float(self.autism_status),
            ],
            dtype=np.float32
        )

    def __len__(self):
        return len(self.group_probs) + 5  # probs + conf + n + age + sex + autism

    def __repr__(self):
        probs = ", ".join(f"{p:.3f}" for p in self.group_probs)
        return (f"ContextVector(pid={self.participant_id!r}, "
                f"probs=[{probs}], conf={self.confidence:.3f}, "
                f"n={self.n_nights}, age_band={self.age_band}, "
                f"sex={self.sex_encoded}, autism={self.autism_status})")


def null_context_vector(participant_id, n_groups,
                        age_band=-1, sex_encoded=-1, autism_status=-1):
    """Uniform vector for participants with too few nights. Near-zero after confidence scaling."""
    u = [1.0 / n_groups] * n_groups
    return ContextVector(participant_id=participant_id, group_probs=u,
                         confidence=1.0 / n_groups, n_nights=0,
                         age_band=age_band, sex_encoded=sex_encoded,
                         autism_status=autism_status)


def context_vectors_to_dataframe(vectors):
    records = []
    for cv in vectors:
        row = {"participant_id": cv.participant_id}
        for i, p in enumerate(cv.group_probs):
            row[f"group_prob_{i}"] = p
        row.update(confidence=cv.confidence, n_nights=cv.n_nights,
                   dominant_group=cv.dominant_group,
                   age_band=cv.age_band, sex_encoded=cv.sex_encoded,
                   autism_status=cv.autism_status)
        records.append(row)
    return pd.DataFrame(records).set_index("participant_id")


def context_vectors_from_dataframe(df):
    prob_cols = sorted(c for c in df.columns if c.startswith("group_prob_"))
    return {
        str(pid): ContextVector(
            participant_id=str(pid),
            group_probs=[float(row[c]) for c in prob_cols],
            confidence=float(row["confidence"]),
            n_nights=int(row["n_nights"]),
            age_band=int(row.get("age_band", -1)),
            sex_encoded=int(row.get("sex_encoded", -1)),
            autism_status=int(row.get("autism_status", -1)),
        )
        for pid, row in df.iterrows()
    }


def extract_trajectory_features(ep_ts, min_hr_min_per_day=360, min_rmssd_min_per_day=60):
    df      = ep_ts.copy()
    df["date"] = df.index.date.astype(str)
    records = []

    for pid, g in df.groupby("participant_id"):
        daily_hr = (g.groupby("date")["pulse_rate_bpm"]
                    .agg(mean_hr="mean", n_hr="count"))
        valid_hr = daily_hr[daily_hr["n_hr"] >= min_hr_min_per_day]["mean_hr"].dropna()

        sleep_g     = g[g["sleep_stage_ep"].isin([101.0, 102.0])]
        daily_rmssd = (sleep_g.groupby("date")["prv_rmssd_ms"]
                       .agg(mean_rmssd="mean", n_rmssd="count"))
        valid_rmssd = daily_rmssd[daily_rmssd["n_rmssd"] >= min_rmssd_min_per_day]["mean_rmssd"].dropna()

        if len(valid_hr) < 2 or len(valid_rmssd) < 2:
            continue

        hr_f    = _feats(valid_hr.values)
        rmssd_f = _feats(valid_rmssd.values)
        records.append({
            "participant_id": pid,
            "hr_mean": hr_f[0], "hr_slope": hr_f[1], "hr_std": hr_f[2],
            "rmssd_mean": rmssd_f[0], "rmssd_slope": rmssd_f[1], "rmssd_std": rmssd_f[2],
            "n_days_hr": len(valid_hr), "n_days_rmssd": len(valid_rmssd),
        })

    if not records:
        return pd.DataFrame(columns=["participant_id",*FEATURE_COLS,"n_days_hr","n_days_rmssd"]).set_index("participant_id")
    return pd.DataFrame(records).set_index("participant_id")


def _feats(values):
    n     = len(values)
    mean  = float(np.mean(values))
    std   = float(np.std(values, ddof=1)) if n > 1 else 0.0
    slope = float(np.polyfit(np.arange(n, dtype=float), values, 1)[0]) if n > 1 else 0.0
    return mean, slope, std


def suggest_rocket_window(ep_ts, coverage_pct=0.95):
    """Find the clock-time window covering coverage_pct of all detected sleep minutes."""
    df    = ep_ts.copy()
    sleep = df[df["sleep_stage_ep"].isin([101.0, 102.0])]
    if sleep.empty:
        return {"window_start": "21:00", "window_end": "09:00"}

    hist   = np.zeros(24, dtype=int)
    for h in sleep.index.hour:
        hist[h] += 1
    total  = hist.sum()
    target = int(total * coverage_pct)

    best_start, best_end, best_count, best_width = 21, 9, 0, 12
    for start in range(24):
        count = 0
        for width in range(1, 25):
            count += hist[(start + width - 1) % 24]
            if count >= target:
                if count > best_count or (count == best_count and width < best_width):
                    best_count, best_width = count, width
                    best_start, best_end   = start, (start + width - 1) % 24
                break

    return {"window_start": f"{(best_start-1)%24:02d}:00",
            "window_end":   f"{(best_end+1)%24:02d}:00"}


class GBTMModel:

    def __init__(self, n_groups=4, min_nights=5, min_hr_min_per_day=360,
                 min_rmssd_min_per_day=60, confidence_flag_threshold=0.6, random_seed=42):
        self.n_groups                  = n_groups
        self.min_nights                = min_nights
        self.min_hr_min_per_day        = min_hr_min_per_day
        self.min_rmssd_min_per_day     = min_rmssd_min_per_day
        self.confidence_flag_threshold = confidence_flag_threshold
        self.random_seed               = random_seed
        self._gmm    = None
        self._scaler = None
        self._fitted = False

    def fit(self, ep_ts):
        feats    = extract_trajectory_features(ep_ts, self.min_hr_min_per_day, self.min_rmssd_min_per_day)
        eligible = feats[feats["n_days_hr"] >= self.min_nights]
        if len(eligible) < self.n_groups:
            raise ValueError(f"Only {len(eligible)} eligible participants — need >= {self.n_groups}")

        self._scaler = StandardScaler()
        X            = self._scaler.fit_transform(eligible[FEATURE_COLS].values)
        self._gmm    = GaussianMixture(n_components=self.n_groups, covariance_type="full",
                                       max_iter=500, n_init=10, random_state=self.random_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gmm.fit(X)
        self._fitted = True
        return self

    def predict(self, ep_ts):
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        feats       = extract_trajectory_features(ep_ts, self.min_hr_min_per_day, self.min_rmssd_min_per_day)
        eligible    = feats[feats["n_days_hr"] >= self.min_nights]
        all_pids    = set(ep_ts["participant_id"].astype(str).unique())
        fitted_pids = set(eligible.index.astype(str))
        vectors     = []

        if len(eligible) > 0:
            X     = self._scaler.transform(eligible[FEATURE_COLS].values)
            probs = self._gmm.predict_proba(X)
            for pid, prob_row in zip(eligible.index, probs):
                vectors.append(ContextVector(
                    participant_id=str(pid),
                    group_probs=prob_row.tolist(),
                    confidence=float(prob_row.max()),
                    n_nights=int(feats.loc[pid, "n_days_hr"]),
                ))

        for pid in all_pids:
            if pid not in fitted_pids:
                vectors.append(null_context_vector(pid, self.n_groups))
        return vectors

    def fit_predict(self, ep_ts):
        return self.fit(ep_ts).predict(ep_ts)

    def coverage_report(self, vectors):
        total      = len(vectors)
        null_count = sum(1 for cv in vectors if cv.n_nights == 0)
        flagged    = [cv for cv in vectors if cv.n_nights > 0 and cv.confidence < self.confidence_flag_threshold]
        group_counts = {}
        for cv in vectors:
            group_counts[cv.dominant_group] = group_counts.get(cv.dominant_group, 0) + 1
        return {
            "total_participants":      total,
            "strong_assignments":      total - null_count - len(flagged),
            "flagged_low_confidence":  len(flagged),
            "null_vectors":            null_count,
            "confidence_threshold":    self.confidence_flag_threshold,
            "group_counts":            group_counts,
            "flagged_participant_ids": [cv.participant_id for cv in flagged],
        }

    def group_summary(self, ep_ts):
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        feats    = extract_trajectory_features(ep_ts, self.min_hr_min_per_day, self.min_rmssd_min_per_day)
        eligible = feats[feats["n_days_hr"] >= self.min_nights].copy()
        eligible["group"] = self._gmm.predict(self._scaler.transform(eligible[FEATURE_COLS].values))
        summary = eligible.groupby("group")[FEATURE_COLS].mean()
        summary["n_participants"] = eligible.groupby("group").size()
        return summary.round(3)


def run_gbtm(ep_ts, n_groups=4, min_nights=5, min_hr_min_per_day=360,
             min_rmssd_min_per_day=60, confidence_flag_threshold=0.6, random_seed=42):
    model   = GBTMModel(n_groups, min_nights, min_hr_min_per_day,
                        min_rmssd_min_per_day, confidence_flag_threshold, random_seed)
    vectors = model.fit_predict(ep_ts)
    return vectors, model


def add_demographic_context(context_df, demographics_df):
    """
    Merge D4 metadata into a context vector DataFrame.

    demographics_df is the output of load_demographics() — indexed by
    participant_id with columns: age_band, sex_encoded, autism_status.

    Participants not found in demographics get -1 for all D4 fields.
    """
    d4 = demographics_df[["age_band","sex_encoded","autism_status"]]
    # drop existing D4 columns first to avoid join conflict
    drop_cols = [c for c in ["age_band","sex_encoded","autism_status"] if c in context_df.columns]
    merged = context_df.drop(columns=drop_cols).join(d4, how="left")
    for col in ["age_band","sex_encoded","autism_status"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(-1).astype(int)
        else:
            merged[col] = -1
    return merged


def select_n_groups(ep_ts, k_range=range(2, 6), min_nights=5,
                    min_hr_min_per_day=360, random_seed=42):
    """
    Select the optimal number of GBTM trajectory groups using BIC.

    Fits a Gaussian mixture model for each k in k_range and returns
    a DataFrame of BIC and AIC scores. Lower BIC = better model.
    The optimal k is the elbow in the BIC curve — not necessarily
    the minimum, which may overfit.

    Run this on the full cohort before deciding --n-groups.
    """
    from sklearn.mixture import GaussianMixture

    feats    = extract_trajectory_features(ep_ts, min_hr_min_per_day)
    eligible = feats[feats["n_days_hr"] >= min_nights]

    if len(eligible) < max(k_range):
        raise ValueError(f"Only {len(eligible)} eligible participants — "
                         f"need >= {max(k_range)} for k={max(k_range)}")

    scaler  = StandardScaler()
    X       = scaler.fit_transform(eligible[FEATURE_COLS].values)
    records = []

    for k in k_range:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            max_iter=500, n_init=10, random_state=random_seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(X)

        records.append({
            "k":           k,
            "bic":         round(float(gmm.bic(X)), 2),
            "aic":         round(float(gmm.aic(X)), 2),
            "log_likelihood": round(float(gmm.score(X) * len(X)), 2),
            "n_participants": len(eligible),
        })
        print(f"  k={k}: BIC={records[-1]['bic']:.1f}  "
              f"AIC={records[-1]['aic']:.1f}")

    df = pd.DataFrame(records).set_index("k")

    # select optimal k using the elbow criterion:
    # find the k after which the BIC improvement becomes negligible
    # (improvement < 50% of the maximum improvement seen so far)
    bic_vals = df["bic"].values
    drops    = -np.diff(bic_vals)  # positive = improvement

    if len(drops) == 0:
        optimal_k = int(df.index[0])
    else:
        max_drop  = drops.max()
        threshold = max_drop * 0.50  # improvement must be >50% of the best seen

        optimal_k = int(df.index[-1])  # default to largest k if no elbow
        for i, drop in enumerate(drops):
            if i > 0 and drop < threshold:
                optimal_k = int(df.index[i])  # elbow at this k
                break

    print(f"  → Suggested k={optimal_k} (BIC elbow — improvement drops below 50% of peak)")
    df["bic_drop"] = [np.nan] + (-np.diff(bic_vals)).tolist()
    df["suggested"] = df.index == optimal_k
    return df
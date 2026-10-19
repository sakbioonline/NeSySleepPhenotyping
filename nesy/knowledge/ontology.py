"""
Loads heuristics.toml and compiles threshold values into typed objects
that the symbolic labelling rules consume directly.

Age stratification:
    KnowledgeStore exposes r1/r2 as flat threshold objects using
    late-adolescence values as defaults (conservative — lower thresholds).
    Age-stratified resolution happens per-night inside rules.py via
    get_thresholds(ks, age_years), which returns the correct band object.

    This preserves the existing ks.r1.SE_healthy_min interface throughout
    test_pipeline.py, ontology.py, and any other callers.
"""

from dataclasses import dataclass
from pathlib import Path
from nesy.config import load_heuristics


@dataclass(frozen=True)
class R1Thresholds:
    SE_healthy_min: float
    WASO_healthy_max: float
    SQ_healthy_min: float
    DIFF_healthy_max: float
    SQ_paradox_max: float
    DIFF_paradox_min: float
    SOL_psychophys_min: float
    ANX_psychophys_min: float
    SQ_restless_min: float
    DIFF_restless_max: float
    WASO_restless_min: float


@dataclass(frozen=True)
class R2Thresholds:
    N1_restless_min: float
    WASO_restless_min: float
    TST_hypersomnia_min: float
    N3_sdb_max: float
    WASO_sdb_min: float


@dataclass(frozen=True)
class R3Thresholds:
    IS_coupling_max: float
    IV_misalign_min: float
    L5_misalign_min: float
    IS_psychophys_max: float
    IV_psychophys_min: float
    IV_coupling_exclusion: float

@dataclass(frozen=True)
class PhenotypeMap:
    label_to_name: dict
    abstain: int

    def name(self, label):
        if label == self.abstain:
            return "abstain"
        return self.label_to_name.get(label, f"unknown_{label}")

    def all_labels(self):
        return sorted(self.label_to_name.keys())


@dataclass(frozen=True)
class KnowledgeStore:
    r1: R1Thresholds
    r2: R2Thresholds
    r3: R3Thresholds
    # Age-stratified bands — used by rules.py via get_thresholds()
    r1_early: R1Thresholds
    r1_late: R1Thresholds
    r2_early: R2Thresholds
    r2_late: R2Thresholds
    phenotypes: PhenotypeMap
    gbtm_config: dict
    raw: dict


def _build_r1_band(raw_r1: dict, band: str) -> R1Thresholds:
    """Build R1Thresholds for a specific age band."""
    # SOL: severely prolonged by band
    # early: 45 min | late: 60 min (circadian delay normalises later onset)
    sol_by_band = {"early": 45.0, "late": 60.0}
    waso_healthy_by_band = {"early": 40.0, "late": 45.0}
    waso_restless_by_band = {"early": 50.0, "late": 60.0}

    return R1Thresholds(
        SE_healthy_min=float(raw_r1["SE_healthy_min"]),
        WASO_healthy_max=waso_healthy_by_band[band],
        SQ_healthy_min=float(raw_r1["SQ_healthy_min"]),
        DIFF_healthy_max=float(raw_r1["DIFF_healthy_max"]),
        SQ_paradox_max=float(raw_r1["SQ_paradox_max"]),
        DIFF_paradox_min=float(raw_r1["DIFF_paradox_min"]),
        SOL_psychophys_min=sol_by_band[band],
        ANX_psychophys_min=float(raw_r1["ANX_psychophys_min"]),
        SQ_restless_min=float(raw_r1["SQ_restless_min"]),
        DIFF_restless_max=float(raw_r1["DIFF_restless_max"]),
        WASO_restless_min=waso_restless_by_band[band],
    )


def _build_r2_band(raw_r2: dict, band: str) -> R2Thresholds:
    """Build R2Thresholds for a specific age band."""
    tst_by_band = {"early": 660.0, "late": 600.0}
    return R2Thresholds(
        N1_restless_min=float(raw_r2["N1_restless_min"]),
        WASO_restless_min=float(raw_r2["WASO_restless_min"]),
        TST_hypersomnia_min=tst_by_band[band],
        N3_sdb_max=float(raw_r2["N3_sdb_max"]),
        WASO_sdb_min=float(raw_r2["WASO_sdb_min"]),
    )


def get_thresholds(ks: KnowledgeStore, age_years) -> tuple:
    """
    Return (r1, r2, r3) for a given participant age.
    Used by rules.py on each night row.
    r3 is age-invariant (cohort-relative IS/IV percentiles).
    """
    if age_years is None:
        return ks.r1_late, ks.r2_late, ks.r3
    band = "early" if float(age_years) <= 13 else "late"
    r1 = ks.r1_early if band == "early" else ks.r1_late
    r2 = ks.r2_early if band == "early" else ks.r2_late
    return r1, r2, ks.r3


def load_knowledge_store(toml_path, yaml_path=None):
    raw = load_heuristics(toml_path)
    try:
        raw_r1 = raw["rules"]["R1"]
        raw_r2 = raw["rules"]["R2"]
        raw_r3 = raw["rules"]["R3"]

        # Default flat objects use late-adolescence values (conservative)
        r1_early = _build_r1_band(raw_r1, "early")
        r1_late = _build_r1_band(raw_r1, "late")
        r2_early = _build_r2_band(raw_r2, "early")
        r2_late = _build_r2_band(raw_r2, "late")

        # ks.r1 / ks.r2 — flat interface preserved for existing callers
        r1 = R1Thresholds(**{k: float(v) for k, v in raw_r1.items()})
        r2 = R2Thresholds(**{k: float(v) for k, v in raw_r2.items()})
        r3 = R3Thresholds(**{k: float(v) for k, v in raw_r3.items()})

        pm = PhenotypeMap(
            label_to_name={int(p[0]): str(p[1])
                           for p in raw["phenotypes"]["labels"]},
            abstain=int(raw["phenotypes"]["abstain"]),
        )

    except KeyError as e:
        raise OntologyError(f"Missing key in heuristics.toml: {e}") from e

    return KnowledgeStore(
        r1=r1, r2=r2, r3=r3,
        r1_early=r1_early, r1_late=r1_late,
        r2_early=r2_early, r2_late=r2_late,
        phenotypes=pm,
        gbtm_config=raw.get("gbtm", {}),
        raw=raw,
    )


class OntologyError(ValueError):
    pass

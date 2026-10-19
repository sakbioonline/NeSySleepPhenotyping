"""
Configuration loader for the NeSy pipeline.

Reads pipeline.yaml and heuristics.toml into typed objects.
All path resolution happens here — every downstream module
receives absolute Path objects, never relative strings.

Usage
-----
    from nesy.config import load_config

    cfg = load_config("configs/pipeline.yaml")
    print(cfg.cohort.embraceplus_path)
    print(cfg.run.random_seed)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# tomllib is stdlib from Python 3.11; tomli is the backport for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-reuse-stub]
        except ImportError:
            raise ImportError(
                "Python 3.10 requires the 'tomli' package. "
                "Install it with: pip install tomli"
            )


# ------------------------------------------------------------------
# Typed config objects
# ------------------------------------------------------------------

@dataclass
class CohortConfig:
    """Cohort identity and raw data locations."""
    name: str
    data_root: Path
    embraceplus_dir: str = "embraceplus"
    dreem_dir: str = "dreem"
    diary_dir: str = "diary"
    participant_ids: list[str] = field(default_factory=list)

    @property
    def embraceplus_path(self) -> Path:
        return self.data_root / self.embraceplus_dir

    @property
    def dreem_path(self) -> Path:
        return self.data_root / self.dreem_dir

    @property
    def diary_path(self) -> Path:
        return self.data_root / self.diary_dir


@dataclass
class RocketConfig:
    window_start: str = "22:00"
    window_end: str = "08:00"
    n_kernels: int = 1000


@dataclass
class AnalyticSampleConfig:
    min_valid_rar_days: int = 5  # primary sample — IS stable with ≥5 days
    sensitivity_rar_days: int = 3  # sensitivity sample
    # min_dreem_nights / min_diary_nights: kept for backward compatibility
    # with any other code that may still read them, but the patched
    # apply_analytic_sample_filter no longer uses either — it checks
    # min_aligned_nights instead (Dreem+diary+RAR intersected on the same
    # date), since two independent marginal counts don't guarantee the
    # two modalities' good nights ever land on the same night.
    min_dreem_nights: int = 3
    min_diary_nights: int = 3
    min_aligned_nights: int = 5
    min_qual_mean: float = 0.5
    qual_mean_exceptions: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    """Execution settings."""
    run_id: str | None = None
    random_seed: int = 42
    log_level: str = "INFO"
    output_dir: Path = field(default_factory=lambda: Path("runs"))
    fmt: str = "parquet"  # parquet or csv


@dataclass
class PipelineConfig:
    """Complete configuration for one NeSy pipeline run."""
    cohort: CohortConfig
    run: RunConfig = field(default_factory=RunConfig)
    analytic_sample: AnalyticSampleConfig = field(default_factory=AnalyticSampleConfig)
    rocket: RocketConfig = field(default_factory=RocketConfig)
    heuristics_path: Path = field(default_factory=lambda: Path("configs/heuristics.toml"))
    kb_path: Path = field(default_factory=lambda: Path("knowledge_base"))
    devices_path: Path = field(default_factory=lambda: Path("configs/devices.yaml"))


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_config(config_path: str | Path) -> PipelineConfig:
    """
    Load and parse pipeline.yaml into a PipelineConfig.

    All relative paths in the YAML are resolved relative to the
    config file's parent directory, so the pipeline can be run
    from any working directory.

    Parameters
    ----------
    config_path:
        Path to pipeline.yaml.

    Returns
    -------
    PipelineConfig with all paths resolved to absolute paths.

    Raises
    ------
    FileNotFoundError:
        If the config file does not exist.
    ConfigError:
        If required fields are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline config not found: {path}"
        )

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} did not parse to a dict. Check YAML syntax."
        )

    base = path.parent

    # Cohort
    c = raw.get("cohort", {})
    if not c.get("name") or not c.get("data_root"):
        raise ConfigError(
            "pipeline.yaml must have cohort.name and cohort.data_root"
        )

    data_root = Path(c["data_root"])
    if not data_root.is_absolute():
        data_root = (base / data_root).resolve()

    cohort = CohortConfig(
        name=c["name"],
        data_root=data_root,
        embraceplus_dir=c.get("embraceplus_dir", "embraceplus"),
        dreem_dir=c.get("dreem_dir", "dreem"),
        diary_dir=c.get("diary_dir", "diary"),
        participant_ids=c.get("participant_ids", []),
    )

    # Run
    r = raw.get("run", {})
    output_dir = Path(r.get("output_dir", "runs"))
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()

    run = RunConfig(
        run_id=r.get("run_id"),
        random_seed=r.get("random_seed", 42),
        log_level=r.get("log_level", "INFO"),
        output_dir=output_dir,
        fmt=r.get("fmt", "parquet"),
    )

    # Paths to supporting files
    def _resolve(key: str, default: str) -> Path:
        p = Path(raw.get(key, default))
        if p.is_absolute():
            return p
        # resolve relative to project root (parent of the configs/ directory)
        return (base.parent / p).resolve()

    # analytic sample thresholds
    a = raw.get("analytic_sample", {})
    analytic = AnalyticSampleConfig(
        min_valid_rar_days=int(a.get("min_valid_rar_days", 5)),
        sensitivity_rar_days=int(a.get("sensitivity_rar_days", 3)),
        min_dreem_nights=int(a.get("min_dreem_nights", 3)),
        min_diary_nights=int(a.get("min_diary_nights", 3)),
        min_aligned_nights=int(a.get("min_aligned_nights", 5)),
        min_qual_mean=float(a.get("min_qual_mean", 0.5)),
        qual_mean_exceptions=list(a.get("qual_mean_exceptions", [])),
    )

    # rocket config
    r = raw.get("rocket", {})
    rocket_cfg = RocketConfig(
        window_start=str(r.get("window_start", "22:00")),
        window_end=str(r.get("window_end", "08:00")),
        n_kernels=int(r.get("n_kernels", 1000)),
    )

    return PipelineConfig(
        cohort=cohort,
        run=run,
        analytic_sample=analytic,
        rocket=rocket_cfg,
        heuristics_path=_resolve("heuristics_path", "configs/heuristics.toml"),
        kb_path=_resolve("kb_path", "knowledge_base"),
        devices_path=_resolve("devices_path", "configs/devices.yaml"),
    )


def load_heuristics(toml_path: str | Path) -> dict[str, Any]:
    """
    Load heuristics.toml into a plain dict.

    The knowledge layer's ontology.py turns this dict into typed
    threshold objects. Kept as a dict here because the structure
    is owned by the knowledge layer, not the config layer.
    """
    path = Path(toml_path)
    if not path.exists():
        raise FileNotFoundError(f"Heuristics file not found: {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_devices(yaml_path: str | Path) -> dict[str, Any]:
    """Load devices.yaml into a plain dict."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Devices config not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


class ConfigError(ValueError):
    """Raised when a config file is missing required fields."""

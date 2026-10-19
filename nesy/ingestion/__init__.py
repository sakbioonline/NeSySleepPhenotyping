"""
nesy.ingestion — multi-modal data loading and preprocessing.

Handles EmbracePlus, Dreem (EEG), and Sleep Diary ingestion,
temporal alignment, REST-activity rhythm feature extraction,
and device-agnostic epoch alignment to the canonical 30-second grid.

Public surface:

    from nesy.ingestion import (
        load_embraceplus, load_embraceplus_directory,
        load_dreem_summary,
        load_dreem_hypnogram, load_dreem_hypnogram_directory,
        load_sleep_diary,
    )
    from nesy.ingestion import (
        align_modalities, aggregate_embraceplus_daily,
        alignment_reports_to_dataframe, night_coverage_report,
    )
    from nesy.ingestion import compute_rar_features, merge_rar_into_night_record
    from nesy.ingestion import EpochAligner, AlignmentReport, needs_alignment
"""

from nesy.ingestion.loaders import (
    load_demographics,
    apply_analytic_sample_filter,
    load_embraceplus,
    load_embraceplus_directory,
    load_dreem_summary,
    load_dreem_hypnogram,
    load_dreem_hypnogram_directory,
    load_sleep_diary,
)
from nesy.ingestion.align import (
    align_modalities,
    aggregate_embraceplus_daily,
    alignment_reports_to_dataframe,
    night_coverage_report,
    validate_night_record,
)
from nesy.ingestion.rar import (
    compute_rar_daily_features,
    compute_rar_participant_features,
    compute_rar_features,
    compute_rar_features_per_night,
    merge_rar_into_night_record,
    impute_non_wear,
    imputation_reports_to_dataframe,
    ImputationReport,
)
from nesy.ingestion.align_epochs import (
    EpochAligner,
    AlignmentReport,
    needs_alignment,
    GRID_RESOLUTION_SEC,
)

__all__ = [
    "load_embraceplus",
    "load_embraceplus_directory",
    "load_dreem_summary",
    "load_dreem_hypnogram",
    "load_dreem_hypnogram_directory",
    "load_sleep_diary",
    "align_modalities",
    "aggregate_embraceplus_daily",
    "alignment_reports_to_dataframe",
    "night_coverage_report",
    "validate_night_record",
    "compute_rar_features",
    "compute_rar_features_per_night",
    "merge_rar_into_night_record",
    "impute_non_wear",
    "imputation_reports_to_dataframe",
    "ImputationReport",
    "EpochAligner",
    "AlignmentReport",
    "needs_alignment",
    "GRID_RESOLUTION_SEC",
]


# device adapters — Fitbit and Click EEG for 22q cohort
try:
    from nesy.ingestion import adapters
    from nesy.ingestion.adapters import (
        load_fitbit, load_fitbit_cohort,
        load_click_eeg, load_click_eeg_summary, load_click_eeg_epochs,
    )
except ImportError:
    pass  # adapters optional
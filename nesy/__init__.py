"""
NeSy — Neuro-Symbolic Framework for Sleep Phenotyping.

A config-driven, reproducible pipeline for multi-modal sleep data
ingestion, knowledge-grounded symbolic labelling, and neuro-symbolic
fusion for adolescent sleep phenotype classification.

Study population: SSP cohort, adolescents 10–17.
Modalities: EmbracePlus (actigraphy), Dreem (EEG), Sleep Diary.

    from nesy.pipeline import PipelineRunner
    from nesy.config import load_pipeline, validate

    config = load_pipeline("configs/pipeline.yaml")
    validate(config)
    PipelineRunner(config).run()
"""

__version__ = "0.1.0"
__author__  = "Abiola Saka"

# subpackage exposure — allows IDE resolution and direct imports
from nesy import ingestion, fusion, knowledge, symbolic, stratification, reporting

try:
    from nesy.ingestion import adapters
except ImportError:
    pass  # adapters optional — only needed for 22q cohort
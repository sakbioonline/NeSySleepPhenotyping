"""
nesy.knowledge — clinical knowledge layer for the NeSy pipeline.

Two distinct sub-systems:

    KnowledgeStore (from ontology.py)
        Loads heuristics.toml and exposes typed threshold objects
        to the symbolic labelling rules (R1, R2, R3).

    KnowledgeBase (from kb/)
        Lazy-loading registry over the knowledge_base/ YAML files.
        Exposes norms, constraints, and phenotype definitions to
        the constraint checker and validation layer.

    GBTM (from gbtm.py)
        Trajectory clustering over HR + RMSSD longitudinal features.
        Produces the context vector C for Layer 4.

    ContextVector (from context.py)
        Typed container for GBTM output, with confidence-weighted
        serialisation for the artifact store.

Public surface:

    from nesy.knowledge import load_knowledge_store, KnowledgeStore
    from nesy.knowledge import KnowledgeBase
    from nesy.knowledge import run_gbtm, GBTMModel
    from nesy.knowledge import ContextVector, context_vectors_to_dataframe
"""

from nesy.knowledge.ontology import (
    load_knowledge_store,
    KnowledgeStore,
    R1Thresholds,
    R2Thresholds,
    R3Thresholds,
    PhenotypeMap,
    OntologyError,
)
from nesy.knowledge.kb import (
    KnowledgeBase,
    KnowledgeBaseError,
    ConstraintReport,
)
from nesy.knowledge.gbtm import (
    add_demographic_context,
    select_n_groups,
    run_gbtm,
    GBTMModel,
    extract_trajectory_features,
    suggest_rocket_window,
)
from nesy.knowledge.medication import (
    load_medication_ontology,
    load_rule_modifications,
    MedicationLookup,
    parse_medication_column,
)
from nesy.knowledge.gbtm import (
    ContextVector,
    context_vectors_to_dataframe,
    context_vectors_from_dataframe,
    null_context_vector,
)

__all__ = [
    "load_knowledge_store",
    "KnowledgeStore",
    "R1Thresholds",
    "R2Thresholds",
    "R3Thresholds",
    "PhenotypeMap",
    "OntologyError",
    "KnowledgeBase",
    "KnowledgeBaseError",
    "ConstraintReport",
    "run_gbtm",
    "GBTMModel",
    "extract_trajectory_features",
    "suggest_rocket_window",
    "ContextVector",
    "context_vectors_to_dataframe",
    "context_vectors_from_dataframe",
    "null_context_vector",
    "load_medication_ontology",
    "load_rule_modifications",
    "MedicationLookup",
    "parse_medication_column",
]

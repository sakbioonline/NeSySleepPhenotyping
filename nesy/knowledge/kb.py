"""
Knowledge base reader for NeSy.

Loads the YAML files in knowledge_base/ on first access and caches
them. Nothing gets loaded until it is actually needed.

    from nesy.knowledge.kb import KnowledgeBase

    kb = KnowledgeBase("knowledge_base/")
    kb.norms.sleep_architecture("SE_pct", age=15)["normal_min"]  # 83.0
    kb.constraints.are_mutually_exclusive(0, 5)                  # True
    kb.phenotypes.label_name(2)                                  # 'psychophysiological_insomnia'
    kb.ontology.is_subtype("N3", "NREM")                         # True
"""

from pathlib import Path
import yaml


class KnowledgeBaseError(KeyError):
    pass


class ConstraintReport:

    def __init__(self):
        self.violations = []

    def add_violation(self, name, message, severity="warn"):
        self.violations.append({"constraint": name, "message": message, "severity": severity})

    @property
    def passed(self):
        return len(self.violations) == 0

    @property
    def should_exclude(self):
        return any(v["severity"] == "exclude" for v in self.violations)

    def __repr__(self):
        return "ConstraintReport(passed)" if self.passed else \
            f"ConstraintReport(violations={len(self.violations)}, exclude={self.should_exclude})"


class _KB:
    """Base: load a YAML file once and cache it."""

    def __init__(self, base_dir):
        self._base = Path(base_dir)
        self._cache = {}

    def _load(self, rel):
        if rel not in self._cache:
            p = self._base / rel
            if not p.exists():
                raise KnowledgeBaseError(f"KB file not found: {p}")
            with p.open() as f:
                self._cache[rel] = yaml.safe_load(f)
        return self._cache[rel]

    def _get(self, data, *keys):
        cur = data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                raise KnowledgeBaseError(f"Key '{k}' not found in KB")
            cur = cur[k]
        return cur

    def _age_band(self, age):
        if 10 <= age <= 13: return "early_adolescence"
        if 14 <= age <= 17: return "late_adolescence"
        raise KnowledgeBaseError(f"Age {age} out of range (10-17)")


class NormRegistry(_KB):
    _FILES = {
        "sleep_architecture": "norms/sleep_architecture.yaml",
        "actigraphy": "norms/actigraphy.yaml",
        "physiology": "norms/physiology.yaml",
    }

    def sleep_architecture(self, metric, age):
        return self._get(self._load(self._FILES["sleep_architecture"]), metric, self._age_band(age))

    def actigraphy(self, metric, age):
        return self._get(self._load(self._FILES["actigraphy"]), metric, self._age_band(age))

    def physiology(self, metric, age):
        return self._get(self._load(self._FILES["physiology"]), metric, self._age_band(age))

    def is_within_normal(self, metric, value, age, norm_type="sleep_architecture"):
        norm = getattr(self, norm_type)(metric, age)
        if "normal_min" in norm and "normal_max" in norm:
            return norm["normal_min"] <= value <= norm["normal_max"]
        if "normal_max" in norm: return value <= norm["normal_max"]
        if "normal_min" in norm: return value >= norm["normal_min"]
        return True


class ConstraintRegistry(_KB):
    _FILES = {
        "mathematical": "constraints/mathematical.yaml",
        "clinical": "constraints/clinical.yaml",
    }

    def get(self, constraint_type):
        if constraint_type not in self._FILES:
            raise KnowledgeBaseError(f"Unknown constraint type: {constraint_type}")
        return self._load(self._FILES[constraint_type])

    def check_night_record(self, row, age):
        report = ConstraintReport()

        tst = row.get("TST_min")
        trt = row.get("TRT_min") or row.get("trt")
        if tst is not None and trt is not None:
            if tst > float(trt):
                report.add_violation("TST_within_TRT", f"TST ({tst}) > TRT ({trt})", "exclude")

        se = row.get("SE_pct")
        if se is not None and not (0 <= se <= 100):
            report.add_violation("SE_range", f"SE_pct ({se}) outside [0,100]", "exclude")

        stage_sum = sum(row.get(c, 0) or 0 for c in ["N1_pct", "N2_pct", "N3_pct", "REM_pct"])
        if stage_sum > 0 and abs(stage_sum - 100) > 2.0:
            report.add_violation("stage_sum", f"stages sum to {stage_sum:.1f}%, expected ~100%", "warn")

        n3 = row.get("N3_pct", 0) or 0
        nrem = sum(row.get(c, 0) or 0 for c in ["N1_pct", "N2_pct", "N3_pct"])
        if nrem > 0 and n3 > nrem:
            report.add_violation("N3_plausible", f"N3 ({n3}%) > NREM ({nrem}%)", "exclude")

        qual = row.get("recording_quality")
        if qual is not None and qual < 0.5:
            report.add_violation("recording_quality", f"quality {qual:.2f} < 0.5", "warn")

        return report

    def mutual_exclusions(self):
        return [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [5, 1], [5, 2], [5, 3]]

    def are_mutually_exclusive(self, a, b):
        return sorted([a, b]) in [sorted(p) for p in self.mutual_exclusions()]


class PhenotypeRegistry(_KB):
    _FILE = "phenotypes/phenotypes.yaml"
    _NAME = {0: "healthy", 1: "paradoxical_insomnia", 2: "psychophysiological_insomnia",
             3: "restless", 4: "rar_misalignment", 5: "hypersomnolence",
             6: "sdb_risk", 7: "coupling_impairment"}

    def get(self, label_or_name):
        data = self._load(self._FILE)
        if isinstance(label_or_name, int):
            if label_or_name not in self._NAME:
                raise KnowledgeBaseError(f"Unknown label: {label_or_name}")
            return self._get(data, self._NAME[label_or_name])
        if label_or_name not in data:
            raise KnowledgeBaseError(f"Phenotype '{label_or_name}' not found")
        return data[label_or_name]

    def label_name(self, label):
        return self._NAME.get(label, f"unknown_{label}")

    def all_labels(self):
        return sorted(self._NAME.keys())


class OntologyRegistry(_KB):
    _FILE = "ontology.yaml"

    @property
    def data(self):
        return self._load(self._FILE)

    def is_subtype(self, concept, parent):
        return self._check(concept, parent, self.data.get("sleep_stages", {}))

    def _check(self, concept, parent, node, within=""):
        if within == parent and concept in node: return True
        for k, v in node.items():
            if isinstance(v, dict) and "subtypes" in v:
                if self._check(concept, parent, v["subtypes"], k): return True
        return False

    def signal_construct(self, signal):
        return self._get(self.data, "signal_constructs", signal)

    def instability_config(self):
        return self._get(self.data, "phenotypes", "instability")

    def known_compound_patterns(self):
        return self._get(self.data, "phenotypes", "known_compound_patterns")


class KnowledgeBase:

    def __init__(self, kb_dir):
        p = Path(kb_dir)
        self.norms = NormRegistry(p)
        self.constraints = ConstraintRegistry(p)
        self.phenotypes = PhenotypeRegistry(p)
        self.ontology = OntologyRegistry(p)
        self._dir = p

    def validate_kb_structure(self):
        expected = [
            "ontology.yaml",
            "norms/sleep_architecture.yaml", "norms/actigraphy.yaml", "norms/physiology.yaml",
            "constraints/mathematical.yaml", "constraints/clinical.yaml",
            "phenotypes/phenotypes.yaml",
            "medications/ontology.yaml", "medications/rule_modifications.yaml",
        ]
        return [str(self._dir / f) for f in expected if not (self._dir / f).exists()]

    def __repr__(self):
        missing = self.validate_kb_structure()
        return f"KnowledgeBase(dir={self._dir}, {'complete' if not missing else str(len(missing)) + ' missing'})"

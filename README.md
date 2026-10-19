# NeSy Sleep Phenotyping (thesis version)

This repository holds the code for the constraint-guided neuro-symbolic framework developed in the PhD thesis *Towards a Neurosymbolic Framework for Wearables-Based Multimodal Sleep Phenotyping in Neurodevelopmental Disorder* (Abiola Saka, University of Bristol, submitted September 2026).

The tag `thesis-submission` marks the code as it stood at thesis submission, rebuilt as closely as the surviving files allow. The section [Provenance](#provenance) lists every file that was edited after the submission date.

> **Status.** Research code. It is not a clinical tool and has not been validated for diagnosis.

---

## Overview

The framework phenotypes sleep and rest-activity rhythms from home wearables in autistic and non-autistic adolescents from the Simons Sleep Project (SSP). A student network learns night-level phenotypes from actigraphy and EEG-derived sleep features. Clinical knowledge enters training as constraint penalties on the network's predicted class probabilities. Diary-reported medication conditions how the rules apply on each night.

| Modality | Device | Used for |
|---|---|---|
| Sleep timing and architecture | Dreem headband (EEG) | TST, SE, SOL, WASO, N3 %, REM %, hypnogram features |
| Rest-activity rhythm | Empatica EmbracePlus (wrist actigraphy) | IV, IS, L5 and M10 midpoints, ROCKET features |
| Subjective report and context | Parent sleep diary | Sleep quality, sleep difficulty, bedtime anxiety, morning feeling, medication |

Chapter 4 of the thesis describes the method in full.

---

## Phenotype space

Six night-level classes are defined in `knowledge_base/phenotypes/phenotypes.yaml`.

| Index | Phenotype | Clinical description |
|---|---|---|
| 0 | Normative Range | Continuity, architecture and rhythm within age-appropriate bounds |
| 1 | Paradoxical Insomnia | Poor reported sleep with unremarkable objective continuity |
| 2 | Conditioned Arousal | Bedtime hyperarousal with elevated wake or onset latency and no circadian phase shift |
| 3 | Hypersomnolence | Total sleep time well above age-appropriate norms |
| 4 | RAR Misalignment | Shifted or unstable rest-activity rhythm phase |
| 5 | Coupling Impairment | Elevated intradaily variability with otherwise unremarkable continuity |

The phenotypes are clinically interpretable groupings. Wearables and diaries cannot establish neurobiological mechanism.

---

## Knowledge-derived rules

Ten rules link trigger conditions to target classes. They are implemented in `nesy/fusion/constraints.py` (training-time constraints) and `nesy/symbolic/rules_v2.py` (labeling functions with medication gating).

| Rule | Target class | Evidence |
|---|---|---|
| r1 | Conditioned Arousal | Sleep continuity, bedtime anxiety, onset latency |
| r2 | RAR Misalignment | High intradaily variability, gated on stimulant and melatonin exposure |
| r3 | Hypersomnolence | Long total sleep time |
| r4 | Normative Range | Continuity and rhythm jointly within normal bounds |
| r5 | Coupling Impairment | Poor sleep efficiency with mid-range intradaily variability, gated on stimulant exposure |
| r6 | Normative Range | Sleep architecture (N3 and REM) within normal bounds |
| r7 | Paradoxical Insomnia | Good objective continuity with poor diary report |
| r8a | Conditioned Arousal | Age-stratified onset latency |
| r8b | Hypersomnolence | Age-stratified total sleep time |
| r9 | RAR Misalignment | High intradaily variability under stimulant exposure |

Thresholds are declared in `configs/heuristics.toml`.

**Medication handling.** Diary entries are matched against `knowledge_base/medications/ontology.yaml` and classified by drug class for each night. The labeling functions in `rules_v2.py` use this to abstain or adjust when a pharmacological explanation for a sleep measure cannot be excluded.

---

## Repository layout

```
NesySleepPhenotyping/
├── nesy/
│   ├── ingestion/        loaders, modality alignment, epoch alignment, RAR features, device adapters
│   ├── knowledge/        thresholds, knowledge base, GBTM, medication parsing
│   ├── symbolic/         rules.py, rules_v2.py, labeler.py, labeler_b.py
│   ├── fusion/           rocket.py, symbols.py, constraints.py, model.py
│   ├── stratification/   phenotype.py, sensitivity.py
│   ├── evaluation/       plausibility.py
│   ├── validation/       metrics.py
│   └── reporting/        final_table.py, report.py
├── knowledge_base/       clinical ontology, norms, constraints, phenotypes, medications (YAML)
├── configs/              heuristics.toml, pipeline.yaml, devices.yaml
├── test_data.py          full pipeline: ingestion, symbolic labels, fusion, stratification, validation
├── test_data_v3a.py      constraint-guided model with the two-cluster weak scaffold
└── README.md
```

---

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/<owner>/NesySleepPhenotyping.git
cd NesySleepPhenotyping
git checkout thesis-submission
python -m venv venv
venv\Scripts\activate          # Linux or macOS: source venv/bin/activate
pip install numpy pandas pyarrow scikit-learn pyyaml torch snorkel
```

---

## Data

The SSP data are not included. Approved researchers can request access from the Simons Foundation through SFARI Base. Participant-level outputs (night tables, predictions, per-participant reports) are excluded from the repository under the terms of the data agreement.

Column names for each device are mapped in `configs/devices.yaml`.

---

## Running

Each script reads its data paths and settings from `configs/pipeline.yaml`. Check the arguments a script accepts with:

```bash
python test_data_v3a.py --help
python test_data.py --help
```

---

## Provenance

No archive of the code was taken at submission (25 September 2026). This version was rebuilt from the working directory on 19 October 2026. Files are included as they were at that date unless stated otherwise.

**Taken from copies saved before or near submission**

| File in this version | Source file | Source last edited |
|---|---|---|
| `nesy/fusion/model.py` | `model_sub_final.py` | 11 Sep 2026 |
| `nesy/fusion/constraints.py` | `constraints_v5_copy.py` | 30 Sep 2026 |

The only code change is the import line in `model.py`, which now reads `nesy.fusion.constraints` in place of `nesy.fusion.constraints_v5`. Participant IDs in debugging lines and usage examples were replaced with the placeholder `SP0000000`.

**Edited after submission, no earlier copy available**

| File | Last edited |
|---|---|
| `nesy/symbolic/rules_v2.py` | 13 Oct 2026 |
| `nesy/symbolic/labeler.py` | 13 Oct 2026 |
| `nesy/ingestion/loaders.py` | 13 Oct 2026 |
| `nesy/reporting/final_table.py` | 11 Oct 2026 |
| `nesy/knowledge/medication.py` | 8 Oct 2026 |
| `nesy/stratification/phenotype.py` | 6 Oct 2026 |
| `test_data.py` | 12 Oct 2026 |
| `test_data_v3a.py` | 7 Oct 2026 |

`rules_v2.py` and `labeler.py` implement the medication gating described in the thesis. Re-running this version may therefore give results that differ slightly from the numbers reported in the thesis.

---

## Citation

```bibtex
@phdthesis{saka_2026_thesis,
  author = {Saka, Abiola},
  title  = {Towards a Neurosymbolic Framework for Wearables-Based Multimodal Sleep Phenotyping in Neurodevelopmental Disorder},
  school = {University of Bristol},
  year   = {2026}
}
```

---

## Acknowledgements

Supervised by Zahraa Abdallah, Marianne van den Bree and Matt Jones. Funded by UKRI EPSRC through the University of Bristol Centre for Doctoral Training in Digital Health and Care. Data were provided by the Simons Foundation through the Simons Sleep Project. We thank the participating families.

## Licence

MIT License

Copyright (c) 2026 Abiola Saka

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

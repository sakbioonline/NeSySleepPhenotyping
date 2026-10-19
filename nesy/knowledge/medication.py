"""
Medication context parser for NeSy symbolic labelling layer.

Parses the 'medications' JSON column from the SSP sleep diary and
produces per-night boolean flags used by R1 labelling functions:

    med_should_abstain_sol      : CNS stimulant taken — R1 abstains on SOL
    med_sol_signal_strengthened : Melatonin taken AND SOL elevated —
                                  treatment-refractory signal, strengthens label 2

Drug classification
-------------------
CNS stimulants (suppress appetite, delay sleep onset via dopamine/noradrenaline):
    Amphetamine salts: Adderall, Vyvanse, Lisdexamfetamine, Dextroamphetamine
    Methylphenidate:   Ritalin, Concerta, Focalin, Methylphenidate (all formulations)
    Dexmethylphenidate: Focalin XR variants

Non-stimulant ADHD (own class, "nonstimulant_adhd"; does NOT set the stimulant flag):
    Strattera (atomoxetine), Qelbree (viloxazine). Both are noradrenergic
    non-stimulants. They are recorded in med_drug_classes but are not counted
    as stimulants, so they do not trigger abstain-SOL or the stimulant context.

Melatonin:
    Melatonin (all spellings/dosages)

Alpha-2 agonists (clonidine/guanfacine) are classified separately —
they are sedating and may shorten SOL, so abstain-SOL does NOT apply.
They are noted as context but do not trigger either flag.

'Same' entries: forward-filled from previous night's medication record.
"""

import json
import re
import pandas as pd
from pathlib import Path

# ── Drug name lookup table ────────────────────────────────────────────
# Keys are lowercase stripped substrings to match against parsed names.
# Values are drug class strings.

DRUG_LOOKUP = {
    # ── CNS stimulants — amphetamine class ───────────────────────────
    "adderall": "stimulant",
    "vyvanse": "stimulant",
    "vyvance": "stimulant",  # misspelling
    "vyvsnse": "stimulant",  # misspelling
    "lisdexamfetamine": "stimulant",
    "dextroamphetamine": "stimulant",
    "dextrostat": "stimulant",  # brand name dextroamphetamine
    "azstarys": "stimulant",  # serdexmethylphenidate (methylphenidate prodrug)

    # ── CNS stimulants — methylphenidate class ────────────────────────
    "ritalin": "stimulant",
    "concerta": "stimulant",
    "concentra": "stimulant",  # misspelling of concerta
    "focalin": "stimulant",
    "foclin": "stimulant",  # misspelling
    "focalinxr": "stimulant",
    "methylphenidate": "stimulant",
    "methalyphenadate": "stimulant",  # misspelling
    "methalphenidate": "stimulant",  # misspelling
    "methyphenidate": "stimulant",  # misspelling
    "methaylphenidate": "stimulant",  # misspelling
    "methalyphenidate": "stimulant",  # misspelling
    "methalyphenedate": "stimulant",  # misspelling
    "dexmethylphenedate": "stimulant",  # misspelling
    "dexmethylphenidate": "stimulant",
    "jornay": "stimulant",  # methylphenidate ER (Jornay PM)
    "quillivant": "stimulant",  # methylphenidate liquid ER

    # ── Non-stimulant ADHD — own class, NOT counted as stimulant ──────
    "strattera": "nonstimulant_adhd",  # atomoxetine
    "stratera": "nonstimulant_adhd",  # misspelling
    "atomoxetine": "nonstimulant_adhd",  # generic strattera
    "qelbree": "nonstimulant_adhd",  # viloxazine
    "viloxazine": "nonstimulant_adhd",  # generic qelbree

    # ── Alpha-2 agonists — sedating, do NOT trigger abstain-SOL ───────
    "intuniv": "alpha2_agonist",
    "intuni": "alpha2_agonist",  # misspelling
    "guanfacine": "alpha2_agonist",
    "guanficine": "alpha2_agonist",  # misspelling
    "guafacine": "alpha2_agonist",  # misspelling
    "gaunfacine": "alpha2_agonist",  # misspelling
    "guanfacine er": "alpha2_agonist",
    "ic guanfacine": "alpha2_agonist",
    "quafasine": "alpha2_agonist",  # misspelling
    "clonidine": "alpha2_agonist",
    "clonodine": "alpha2_agonist",  # misspelling
    "clonadine": "alpha2_agonist",  # misspelling
    "clonidone": "alpha2_agonist",  # misspelling
    "clinidine": "alpha2_agonist",  # misspelling
    "clonidine er": "alpha2_agonist",

    # ── Melatonin ─────────────────────────────────────────────────────
    "melatonin": "melatonin",

    # ── SSRIs / SNRIs ─────────────────────────────────────────────────
    "fluoxetine": "ssri",
    "fluexotine": "ssri",  # misspelling
    "flovoxetine": "ssri",  # misspelling
    "fluvoxetine": "ssri",  # misspelling (also fluvoxamine variant)
    "fluvoxitine": "ssri",  # misspelling
    "fluvoxamine": "ssri",
    "prozac": "ssri",
    "prosack": "ssri",  # misspelling
    "sertraline": "ssri",
    "serttaline": "ssri",  # misspelling
    "zoloft": "ssri",
    "escitalopram": "ssri",
    "escitrolopram": "ssri",  # misspelling
    "lexapro": "ssri",
    "luvox": "ssri",
    "venlafaxine": "snri",
    "venlaxine": "snri",  # misspelling
    "venlefaxine": "snri",  # misspelling
    "cymbalta": "snri",
    "cymbolta": "snri",  # misspelling
    "bupropion": "snri",
    "buproprion": "snri",  # misspelling
    "wellbutrin": "snri",

    # ── TCAs ──────────────────────────────────────────────────────────
    "amitriptyline": "tca",
    "amitriptilyn": "tca",  # misspelling
    "amitriptylin": "tca",  # misspelling
    "trazodone": "tca",
    "trazadone": "tca",  # misspelling
    "imipramine": "tca",

    # ── Antipsychotics ────────────────────────────────────────────────
    "ziprasidone": "antipsychotic",
    "risperidone": "antipsychotic",
    "aripiprazole": "antipsychotic",
    "aripiprizole": "antipsychotic",  # misspelling
    "abilify": "antipsychotic",
    "olanzapine": "antipsychotic",
    "olanzapibe": "antipsychotic",  # misspelling
    "vraylar": "antipsychotic",

    # ── Antihistamines ────────────────────────────────────────────────
    "hydroxyzine": "antihistamine",
    "hydroxyzin": "antihistamine",
    "hydroxozine": "antihistamine",  # misspelling
}


def _classify_name(name: str) -> str:
    """Return drug class for a medication name string, or 'other'."""
    name_lower = name.lower().strip()
    # Remove dosage suffixes that may be concatenated
    name_lower = re.sub(r'\s*\d+(\.\d+)?\s*(mg|mcg|ml|iu).*$', '', name_lower).strip()
    for key, cls in DRUG_LOOKUP.items():
        if key in name_lower:
            return cls
    return "other"


def parse_medication_entry(raw_value) -> list[dict]:
    """
    Parse a single medications cell value.

    Returns list of dicts with keys: name (str), dosage (str), drug_class (str).
    Returns [] if missing, unparseable, or empty.
    """
    if pd.isna(raw_value) or raw_value is None:
        return []
    try:
        meds = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        return []
    result = []
    for m in meds:
        name = str(m.get("name", "")).strip()
        if not name or name.lower() == "same":
            continue  # 'Same' entries handled by forward-fill below
        result.append({
            "name": name,
            "dosage": str(m.get("dosage", "")).strip(),
            "drug_class": _classify_name(name),
        })
    return result


def build_medication_flags(
    diary_df: pd.DataFrame,
    *args,
    date_col: str = None,
    participant_col: str = "pid",
    **kwargs
) -> pd.DataFrame:
    """
    Given the SSP sleep diary DataFrame, return a DataFrame with normalized join keys
    (participant_id, night_date) and derived medication flags.
    """
    # ── 0. Handle Index Safely ─────────────────────────────────────────
    if isinstance(diary_df.index, pd.MultiIndex) or diary_df.index.name in ["pid", "participant_id"]:
        df = diary_df.reset_index().copy()
    else:
        df = diary_df.copy()

    # ── 1. Select clinical night date column ───────────────────────────
    if date_col and date_col in df.columns:
        chosen_date_col = date_col
    elif "sleep_date" in df.columns:
        chosen_date_col = "sleep_date"
    elif "diary_date" in df.columns:
        print("  WARNING: 'sleep_date' missing — falling back to 'diary_date'.")
        chosen_date_col = "diary_date"
    else:
        chosen_date_col = "date"

    pid_col = participant_col if participant_col in df.columns else ("pid" if "pid" in df.columns else "participant_id")
    df["participant_id"] = df[pid_col].astype(str).str.strip()
    df["pid"] = df["participant_id"]

    # # ── 2. Convert to ISO Datetime FIRST (Fixes sorting & merge keys) ─
    # df["night_date_dt"] = pd.to_datetime(
    #     df[chosen_date_col],
    #     format="mixed",
    #     dayfirst=True,
    #     errors="coerce",
    # )
    # df["night_date"] = df["night_date_dt"].dt.strftime("%Y-%m-%d")
    #
    # if "diary_date" in df.columns:
    #     df["diary_date"] = pd.to_datetime(
    #         df["diary_date"],
    #         format="mixed",
    #         dayfirst=True,
    #         errors="coerce",
    #     ).dt.strftime("%Y-%m-%d")
    # else:
    #     df["diary_date"] = df["night_date"]
    # ── 2. Convert to ISO Datetime (Strict Day-First) ───────────────────
    # Use exact format matching for DD/MM/YYYY strings to stop month-swapping
    df["night_date_dt"] = pd.to_datetime(
        df[chosen_date_col],
        format="%d/%m/%Y",
        errors="coerce",
    )
    # If the input was already an ISO datetime/string, fallback cleanly:
    mask_nat = df["night_date_dt"].isna()
    if mask_nat.any():
        df.loc[mask_nat, "night_date_dt"] = pd.to_datetime(
            df.loc[mask_nat, chosen_date_col],
            errors="coerce"
        )

    df["night_date"] = df["night_date_dt"].dt.strftime("%Y-%m-%d")

    if "diary_date" in df.columns:
        df["diary_date_dt"] = pd.to_datetime(
            df["diary_date"],
            format="%d/%m/%Y",
            errors="coerce",
        )
        mask_d_nat = df["diary_date_dt"].isna()
        if mask_d_nat.any():
            df.loc[mask_d_nat, "diary_date_dt"] = pd.to_datetime(
                df.loc[mask_d_nat, "diary_date"],
                errors="coerce"
            )
        df["diary_date"] = df["diary_date_dt"].dt.strftime("%Y-%m-%d")
    else:
        df["diary_date"] = df["night_date"]

    df = df[["pid", "diary_date", "participant_id", "night_date", "night_date_dt", "medications"]].copy()

    # ── 3. Chronological Forward-fill on Parsed Datetime ──────────────
    SAME_TOKENS = {"same", "same as before"}

    def _is_same_entry(val):
        if pd.isna(val) or val is None:
            return False
        try:
            meds = json.loads(val) if isinstance(val, str) else val
            if isinstance(meds, list):
                return any(str(m.get("name", "")).strip().lower() in SAME_TOKENS
                           for m in meds if isinstance(m, dict))
            return False
        except Exception:
            return False

    # Sort by actual datetime object, NOT raw string
    df = df.sort_values(["pid", "night_date_dt"]).reset_index(drop=True)

    resolved = []
    last_valid_by_pid = {}
    for _, row in df.iterrows():
        pid = row["pid"]
        val = row["medications"]
        if _is_same_entry(val):
            resolved.append(last_valid_by_pid.get(pid, val))
        else:
            if not pd.isna(val) and val is not None:
                last_valid_by_pid[pid] = val
            resolved.append(val)
    df["medications"] = resolved

    # ── 4. Parse & Classify Medications ──────────────────────────────
    parsed = df["medications"].apply(parse_medication_entry)
    df["med_drug_classes"] = parsed.apply(lambda meds: [m["drug_class"] for m in meds if isinstance(m, dict) and "drug_class" in m])
    df["med_has_stimulant"] = df["med_drug_classes"].apply(lambda cs: "stimulant" in cs)
    df["med_has_melatonin"] = df["med_drug_classes"].apply(lambda cs: "melatonin" in cs)
    df["med_has_alpha2"] = df["med_drug_classes"].apply(lambda cs: "alpha2_agonist" in cs)

    df["med_should_abstain_sol"] = df["med_has_stimulant"]
    df["med_sol_signal_strengthened"] = df["med_has_melatonin"]

    return df[[
        "pid", "diary_date",
        "participant_id", "night_date",
        "med_should_abstain_sol",
        "med_sol_signal_strengthened",
        "med_has_stimulant",
        "med_has_melatonin",
        "med_has_alpha2",
        "med_drug_classes",
    ]]

# def build_medication_flags(
#     diary_df: pd.DataFrame,
#     *args,
#     date_col: str = None,
#     participant_col: str = "pid",
#     **kwargs
# ) -> pd.DataFrame:
#     """
#     Given the SSP sleep diary DataFrame, return a DataFrame with normalized join keys
#     (participant_id, night_date) and derived medication flags.
#     """
#     # ── 0. Handle Index Safely ─────────────────────────────────────────
#     if isinstance(diary_df.index, pd.MultiIndex) or diary_df.index.name in ["pid", "participant_id"]:
#         df = diary_df.reset_index().copy()
#     else:
#         df = diary_df.copy()
#
#     # ── 1. Select clinical night date column ───────────────────────────
#     if date_col and date_col in df.columns:
#         chosen_date_col = date_col
#     elif "sleep_date" in df.columns:
#         chosen_date_col = "sleep_date"
#     elif "diary_date" in df.columns:
#         print("  WARNING: 'sleep_date' missing — falling back to 'diary_date'.")
#         chosen_date_col = "diary_date"
#     else:
#         chosen_date_col = "date"
#
#     pid_col = participant_col if participant_col in df.columns else ("pid" if "pid" in df.columns else "participant_id")
#     df["participant_id"] = df[pid_col].astype(str).str.strip()
#     df["pid"] = df["participant_id"]
#
#     # ── 2. Convert to ISO Datetime FIRST (Fixes sorting & merge keys) ─
#     #df["night_date_dt"] = pd.to_datetime(df[chosen_date_col], errors="coerce")
#     df["night_date_dt"] = pd.to_datetime(
#         df[chosen_date_col],
#         format="mixed",
#         dayfirst=True,
#         errors="coerce",
#     )
#     df["night_date"] = df["night_date_dt"].dt.strftime("%Y-%m-%d")
#
#     if "diary_date" in df.columns:
#         df["diary_date"] = pd.to_datetime(df["diary_date"], errors="coerce").dt.strftime("%Y-%m-%d")
#     else:
#         df["diary_date"] = df["night_date"]
#
#     df = df[["pid", "diary_date", "participant_id", "night_date", "night_date_dt", "medications"]].copy()
#
#     # ── 3. Chronological Forward-fill on Parsed Datetime ──────────────
#     def _is_same_entry(val):
#         if pd.isna(val) or val is None:
#             return False
#         try:
#             meds = json.loads(val) if isinstance(val, str) else val
#             if isinstance(meds, list):
#                 return any(str(m.get("name", "")).strip().lower() == "same" for m in meds if isinstance(m, dict))
#             return False
#         except:
#             return False
#
#     # Sort by actual datetime object, NOT raw string
#     df = df.sort_values(["pid", "night_date_dt"]).reset_index(drop=True)
#
#     resolved = []
#     last_valid_by_pid = {}
#     for _, row in df.iterrows():
#         pid = row["pid"]
#         val = row["medications"]
#         if _is_same_entry(val):
#             resolved.append(last_valid_by_pid.get(pid, val))
#         else:
#             if not pd.isna(val) and val is not None:
#                 last_valid_by_pid[pid] = val
#             resolved.append(val)
#     df["medications"] = resolved
#
#     # ── 4. Parse & Classify Medications ──────────────────────────────
#     parsed = df["medications"].apply(parse_medication_entry)
#     df["med_drug_classes"] = parsed.apply(lambda meds: [m["drug_class"] for m in meds if isinstance(m, dict) and "drug_class" in m])
#     df["med_has_stimulant"] = df["med_drug_classes"].apply(lambda cs: "stimulant" in cs)
#     df["med_has_melatonin"] = df["med_drug_classes"].apply(lambda cs: "melatonin" in cs)
#     df["med_has_alpha2"] = df["med_drug_classes"].apply(lambda cs: "alpha2_agonist" in cs)
#
#     df["med_should_abstain_sol"] = df["med_has_stimulant"]
#     df["med_sol_signal_strengthened"] = df["med_has_melatonin"]
#
#     return df[[
#         "pid", "diary_date",
#         "participant_id", "night_date",
#         "med_should_abstain_sol",
#         "med_sol_signal_strengthened",
#         "med_has_stimulant",
#         "med_has_melatonin",
#         "med_has_alpha2",
#         "med_drug_classes",
#     ]]

# def build_medication_flags(diary_df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
#     """
#     Given the SSP sleep diary DataFrame, return a DataFrame with normalized join keys
#     (participant_id, night_date) and derived medication flags.
#     """
#     df = diary_df.reset_index().copy()
#
#     # ── 1. Select clinical night date column ───────────────────────────
#     if date_col and date_col in df.columns:
#         chosen_date_col = date_col
#     elif "sleep_date" in df.columns:
#         chosen_date_col = "sleep_date"
#     elif "diary_date" in df.columns:
#         print("  WARNING: 'sleep_date' missing — falling back to 'diary_date'.")
#         chosen_date_col = "diary_date"
#     else:
#         chosen_date_col = "date"
#
#     df["participant_id"] = df["pid"]
#
#     # ── 2. Convert to ISO Datetime FIRST (Fixes sorting & merge keys) ─
#     df["night_date_dt"] = pd.to_datetime(
#         df[chosen_date_col], format="mixed", dayfirst=True
#     )
#     df["night_date"] = df["night_date_dt"].dt.strftime("%Y-%m-%d")
#
#     if "diary_date" in df.columns:
#         df["diary_date"] = pd.to_datetime(
#             df["diary_date"], format="mixed", dayfirst=True
#         ).dt.strftime("%Y-%m-%d")
#     else:
#         df["diary_date"] = df["night_date"]
#
#     df = df[["pid", "diary_date", "participant_id", "night_date", "night_date_dt", "medications"]].copy()
#
#     # ── 3. Chronological Forward-fill on Parsed Datetime ──────────────
#     def _is_same_entry(val):
#         if pd.isna(val) or val is None:
#             return False
#         try:
#             meds = json.loads(val)
#             return any(str(m.get("name", "")).strip().lower() == "same" for m in meds)
#         except:
#             return False
#
#     # FIX: Sort by actual datetime object, NOT raw string!
#     df = df.sort_values(["pid", "night_date_dt"]).reset_index(drop=True)
#
#     resolved = []
#     last_valid_by_pid = {}
#     for _, row in df.iterrows():
#         pid = row["pid"]
#         val = row["medications"]
#         if _is_same_entry(val):
#             resolved.append(last_valid_by_pid.get(pid, val))
#         else:
#             if not pd.isna(val) and val is not None:
#                 last_valid_by_pid[pid] = val
#             resolved.append(val)
#     df["medications"] = resolved
#
#     # ── 4. Parse & Classify Medications ──────────────────────────────
#     parsed = df["medications"].apply(parse_medication_entry)
#     df["med_drug_classes"] = parsed.apply(lambda meds: [m["drug_class"] for m in meds if "drug_class" in m])
#     df["med_has_stimulant"] = df["med_drug_classes"].apply(lambda cs: "stimulant" in cs)
#     df["med_has_melatonin"] = df["med_drug_classes"].apply(lambda cs: "melatonin" in cs)
#     df["med_has_alpha2"] = df["med_drug_classes"].apply(lambda cs: "alpha2_agonist" in cs)
#
#     df["med_should_abstain_sol"] = df["med_has_stimulant"]
#     df["med_sol_signal_strengthened"] = df["med_has_melatonin"]
#
#     return df[[
#         "pid", "diary_date",
#         "participant_id", "night_date",
#         "med_should_abstain_sol",
#         "med_sol_signal_strengthened",
#         "med_has_stimulant",
#         "med_has_melatonin",
#         "med_has_alpha2",
#         "med_drug_classes",
#     ]]

# def build_medication_flags(diary_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Given the SSP sleep diary DataFrame, return a DataFrame with columns:
#
#         pid, diary_date,
#         med_should_abstain_sol      (bool),
#         med_sol_signal_strengthened (bool),
#         med_has_stimulant           (bool),
#         med_has_melatonin           (bool),
#         med_has_alpha2              (bool),
#         med_drug_classes            (list[str])
#
#     'Same' entries are forward-filled within each participant before
#     flag computation, so a participant who writes "Same" inherits the
#     previous night's medication record.
#
#     diary_date is used as the join key with the night-level feature
#     matrix (not sleep_date).
#     """
#     print(f"build_medication_flags columns: {diary_df.columns.tolist()}")
#     print(f"build_medication_flags index: {diary_df.index.names}")
#     df = diary_df.reset_index()
#     print(f"after reset_index columns: {df.columns.tolist()}")
#     df = diary_df[["pid", "diary_date", "medications"]].copy()
#
#     # ── Forward-fill 'Same' entries within participant ────────────────
#     def _is_same_entry(val):
#         if pd.isna(val):
#             return False
#         try:
#             meds = json.loads(val)
#             return any(str(m.get("name", "")).strip().lower() == "same" for m in meds)
#         except:
#             return False
#
#     df = df.sort_values(["pid", "diary_date"]).reset_index(drop=True)
#
#     resolved = []
#     last_valid_by_pid = {}
#     for _, row in df.iterrows():
#         pid = row["pid"]
#         val = row["medications"]
#         if _is_same_entry(val):
#             resolved.append(last_valid_by_pid.get(pid, val))
#         else:
#             if not pd.isna(val):
#                 last_valid_by_pid[pid] = val
#             resolved.append(val)
#     df["medications"] = resolved
#
#     # ── Parse and classify ────────────────────────────────────────────
#     parsed = df["medications"].apply(parse_medication_entry)
#     df["med_drug_classes"] = parsed.apply(
#         lambda meds: [m["drug_class"] for m in meds])
#     df["med_has_stimulant"] = df["med_drug_classes"].apply(
#         lambda cs: "stimulant" in cs)
#     df["med_has_melatonin"] = df["med_drug_classes"].apply(
#         lambda cs: "melatonin" in cs)
#     df["med_has_alpha2"] = df["med_drug_classes"].apply(
#         lambda cs: "alpha2_agonist" in cs)
#
#     # ── Derive R1 modifier flags ──────────────────────────────────────
#     # abstain-SOL: stimulant present — pharmacological SOL elevation
#     # cannot be excluded
#     df["med_should_abstain_sol"] = df["med_has_stimulant"]
#
#     # sol-strengthened: melatonin present — elevated SOL despite
#     # circadian-phase advancing agent = treatment-refractory signal
#     # (set here; R1 checks SOL >= threshold before firing)
#     df["med_sol_signal_strengthened"] = df["med_has_melatonin"]
#
#     df["participant_id"] = df["pid"]
#     df["night_date"] = df["diary_date"]
#
#     return df[[
#         "pid", "diary_date",
#         "participant_id", "night_date",  # normalised join keys
#         "med_should_abstain_sol",
#         "med_sol_signal_strengthened",
#         "med_has_stimulant",
#         "med_has_melatonin",
#         "med_has_alpha2",
#         "med_drug_classes",
#     ]]

def add_medication_context(night_df: pd.DataFrame,
                           diary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join medication flags onto the night-level feature matrix.
    Joins on (participant_id, night_date) using normalized ISO dates.
    """
    # 1. Prepare target night dataframe
    target = night_df.copy()
    if isinstance(target.index, pd.MultiIndex) or target.index.names[0] is not None:
        target = target.reset_index()

    if "participant_id" not in target.columns and "pid" in target.columns:
        target = target.rename(columns={"pid": "participant_id"})
    if "night_date" not in target.columns:
        for c in ["sleep_date", "date", "diary_date"]:
            if c in target.columns:
                target = target.rename(columns={c: "night_date"})
                break

    # Normalize target night_date to ISO YYYY-MM-DD string
    target["night_date"] = pd.to_datetime(
        target["night_date"], format="mixed", dayfirst=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # 2. Build flags from sleep diary (keys on sleep_date / night_date)
    print(f"THis is the diary DF oooooo:",diary_df)
    flags = build_medication_flags(diary_df, date_col="sleep_date")

    # 3. Merge onto night matrix on clean ISO keys
    merged = target.merge(
        flags[[
            "participant_id",
            "night_date",
            "med_should_abstain_sol",
            "med_sol_signal_strengthened",
            "med_has_stimulant",
            "med_has_melatonin",
            "med_has_alpha2",
            "med_drug_classes",
        ]],
        on=["participant_id", "night_date"],
        how="left",
    )

    # 4. Fill boolean columns with False, list columns with empty list []
    bool_cols = [
        "med_should_abstain_sol",
        "med_sol_signal_strengthened",
        "med_has_stimulant",
        "med_has_melatonin",
        "med_has_alpha2",
    ]
    for col in bool_cols:
        merged[col] = merged[col].fillna(False).astype(bool)

    merged["med_drug_classes"] = merged["med_drug_classes"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return merged


# def add_medication_context(night_df: pd.DataFrame, diary_df: pd.DataFrame) -> pd.DataFrame:
#     df = night_df.copy()
#
#     # 1. Build medication flags using the clinical sleep_date as the anchor
#     flags = build_medication_flags(diary_df, date_col="sleep_date")
#
#     # 2. Merge strictly on participant_id and clinical night_date
#     merged = df.merge(
#         flags[[
#             "participant_id",
#             "night_date",
#             "med_should_abstain_sol",
#             "med_sol_signal_strengthened",
#             "med_has_stimulant",
#             "med_has_melatonin",
#             "med_has_alpha2",
#             "med_drug_classes"
#         ]],
#         on=["participant_id", "night_date"],
#         how="left",
#     )
#
#     # 3. Fill booleans with False and list columns with []
#     bool_cols = [
#         "med_should_abstain_sol", "med_sol_signal_strengthened",
#         "med_has_stimulant", "med_has_melatonin", "med_has_alpha2"
#     ]
#     for col in bool_cols:
#         merged[col] = merged[col].fillna(False).astype(bool)
#
#     merged["med_drug_classes"] = merged["med_drug_classes"].apply(
#         lambda x: x if isinstance(x, list) else []
#     )
#
#     return merged

# def add_medication_context(night_df: pd.DataFrame,
#                            diary_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Join medication flags onto the night-level feature matrix.
#
#     Joins on (pid, diary_date). Nights with no diary medication record
#     receive False for all flags (conservative — no confound assumed).
#     """
#     df = diary_df.reset_index()
#     # Normalise column names
#     if "pid" not in df.columns and "participant_id" in df.columns:
#         df = df.rename(columns={"participant_id": "pid"})
#     if "diary_date" not in df.columns and "date" in df.columns:
#         df = df.rename(columns={"date": "diary_date"})
#
#     flags = build_medication_flags(df)
#     merged = df.merge(
#         flags[["pid", "diary_date",
#                "med_should_abstain_sol",
#                "med_sol_signal_strengthened",
#                "med_has_stimulant",
#                "med_has_melatonin",
#                "med_has_alpha2"]],
#         on=["pid", "diary_date"],
#         how="left",
#     )
#     for col in ["med_should_abstain_sol", "med_sol_signal_strengthened",
#                 "med_has_stimulant", "med_has_melatonin", "med_has_alpha2"]:
#         merged[col] = merged[col].fillna(False)
#     return merged


# ── Backward compatibility shims ──────────────────────────────────────
# The original medication.py exposed these four names.
# They are aliased here so nesy/knowledge/__init__.py imports
# continue to work without modification.
def load_medication_ontology(diary_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Alias for build_medication_flags — backward compatibility."""
    return build_medication_flags(diary_df)


def load_rule_modifications(diary_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Returns per-night medication modifier flags.
    Alias for build_medication_flags — backward compatibility.
    """
    return build_medication_flags(diary_df)


class MedicationLookup:
    """
    Backward-compatible wrapper around the DRUG_LOOKUP table.
    Previously a class — now a thin wrapper over _classify_name.
    """

    def classify(self, name: str) -> str:
        return _classify_name(name)

    def is_stimulant(self, name: str) -> bool:
        return _classify_name(name) == "stimulant"

    def is_melatonin(self, name: str) -> bool:
        return _classify_name(name) == "melatonin"

    def is_alpha2(self, name: str) -> bool:
        return _classify_name(name) == "alpha2_agonist"


def parse_medication_column(raw_value_or_df, *args, **kwargs):
    """Wrapper that passes date_col, kb_dir, and all kwargs to build_medication_flags."""
    #print(f"parse_medication_column called with type: {type(raw_value_or_df)}")
    if isinstance(raw_value_or_df, pd.DataFrame):
        #date_col = kwargs.get("date_col", None)
        #print(f"This is from parse_medication_column - date_col: {date_col} ({type(raw_value_or_df[date_col])})")

        # FIX: Forward ALL positional *args (e.g. kb_dir) and ALL **kwargs (e.g. participant_col, date_col)
        return build_medication_flags(raw_value_or_df, *args, **kwargs)

    return parse_medication_entry(raw_value_or_df)
#
# def parse_medication_column(raw_value_or_df, *args, **kwargs):
#     """Wrapper that passes date_col and kwargs to build_medication_flags."""
#     print(f"parse_medication_column called with type: {type(raw_value_or_df)}")
#     if isinstance(raw_value_or_df, pd.DataFrame):
#         # FIX: Extract date_col from kwargs if present
#         date_col = kwargs.get("date_col", None)
#         print(f"This is from parse_medication_column - {date_col} - {type(date_col)}")
#         return build_medication_flags(raw_value_or_df, date_col=date_col)
#     return parse_medication_entry(raw_value_or_df)
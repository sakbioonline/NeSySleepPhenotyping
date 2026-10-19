import numpy as np
import pandas as pd

NORMATIVE = 0
PARADOXICAL = 1
CONDITIONED_AROUSAL = 2
HYPERSOMNOLENCE = 3
RAR_MISALIGN = 4
COUPLING_IMPAIRMENT = 5


def physiological_audit(
        y_pred,
        obs,
        cf,
):
    """
    Physiological plausibility audit.

    Precision:
        When a phenotype is predicted,
        was supporting physiology present?

    Coverage:
        When physiology is present,
        was the phenotype predicted?

    F1:
        Harmonic mean of precision and coverage.
    """

    y_pred = np.asarray(y_pred)

    # convert probability matrix -> hard labels

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)

    # --------------------------------------------------
    # Physiology masks
    # --------------------------------------------------

    physiology = {

        "Normative":
            cf.normative_mask(obs),

        "Conditioned_Arousal":
            cf.conditioned_arousal_mask(obs),

        "RAR_Misalignment":
            cf.rar_misalignment_mask(obs),

        "Hypersomnolence":
            cf.hypersomnolence_mask(obs),

        "Coupling_Impairment":
            cf.coupling_mask(obs),
    }

    # convert torch tensors -> numpy bool arrays

    for k, v in physiology.items():

        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()

        physiology[k] = np.asarray(v).astype(bool)

    # --------------------------------------------------
    # Prediction masks
    # --------------------------------------------------

    predictions = {

        "Normative":
            y_pred == NORMATIVE,

        "Conditioned_Arousal":
            y_pred == CONDITIONED_AROUSAL,

        "RAR_Misalignment":
            y_pred == RAR_MISALIGN,

        "Hypersomnolence":
            y_pred == HYPERSOMNOLENCE,

        "Coupling_Impairment":
            y_pred == COUPLING_IMPAIRMENT,
    }
    print("y_pred shape:", np.asarray(y_pred).shape)

    for k, v in predictions.items():
        print("prediction", k, np.asarray(v).shape)
    for k, v in physiology.items():
        print("physiology", k, np.asarray(v).shape)
    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    results = {}

    total_tp = 0
    total_pred = 0
    total_phys = 0

    n_nights = len(y_pred)

    for phenotype in predictions:

        pred_mask = predictions[phenotype]
        phys_mask = physiology[phenotype]

        tp = np.sum(pred_mask & phys_mask)

        n_pred = np.sum(pred_mask)
        n_phys = np.sum(phys_mask)

        fp = n_pred - tp
        fn = n_phys - tp

        precision = (
            tp / n_pred
            if n_pred > 0
            else np.nan
        )

        coverage = (
            tp / n_phys
            if n_phys > 0
            else np.nan
        )

        if (
                np.isfinite(precision)
                and np.isfinite(coverage)
                and (precision + coverage) > 0
        ):
            f1 = (
                    2 * precision * coverage
                    / (precision + coverage)
            )
        else:
            f1 = np.nan

        results[phenotype] = {

            "predicted": int(n_pred),
            "physiology": int(n_phys),

            "predicted_rate":
                n_pred / n_nights,

            "physiology_rate":
                n_phys / n_nights,

            "true_positive":
                int(tp),

            "false_positive":
                int(fp),

            "false_negative":
                int(fn),

            "precision":
                precision,

            "coverage":
                coverage,

            "f1":
                f1,
        }

        total_tp += tp
        total_pred += n_pred
        total_phys += n_phys

    # --------------------------------------------------
    # Overall
    # --------------------------------------------------

    overall_precision = (
        total_tp / total_pred
        if total_pred > 0
        else np.nan
    )

    overall_coverage = (
        total_tp / total_phys
        if total_phys > 0
        else np.nan
    )

    if (
            np.isfinite(overall_precision)
            and np.isfinite(overall_coverage)
            and (overall_precision + overall_coverage) > 0
    ):
        overall_f1 = (
                2
                * overall_precision
                * overall_coverage
                / (overall_precision + overall_coverage)
        )
    else:
        overall_f1 = np.nan

    return {

        "overall": {

            "precision":
                overall_precision,

            "coverage":
                overall_coverage,

            "f1":
                overall_f1,
        },

        "per_phenotype":
            results,
    }


def print_physiology_audit(audit):
    print("\n==============================")
    print(" Physiological Plausibility Audit")
    print("==============================")

    o = audit["overall"]

    print(
        f"\nOVERALL"
        f"\n  Precision : {o['precision']:.3f}"
        f"\n  Coverage  : {o['coverage']:.3f}"
        f"\n  F1        : {o['f1']:.3f}"
    )

    print("\nPER PHENOTYPE")

    for name, res in audit["per_phenotype"].items():
        print(
            f"\n{name}"
            f"\n  Predicted : {res['predicted']}"
            f"\n  Physiology: {res['physiology']}"
            f"\n  Pred Rate : {res['predicted_rate']:.3f}"
            f"\n  Phys Rate : {res['physiology_rate']:.3f}"
            f"\n  TP        : {res['true_positive']}"
            f"\n  FP        : {res['false_positive']}"
            f"\n  FN        : {res['false_negative']}"
            f"\n  Precision : {res['precision']:.3f}"
            f"\n  Coverage  : {res['coverage']:.3f}"
            f"\n  F1        : {res['f1']:.3f}"
        )


def phenotype_profile_audit(
        y_pred,
        obs,
        phenotype_names=None,
):

    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)

    if phenotype_names is None:

        phenotype_names = {
            0: "Normative",
            1: "Paradoxical",
            2: "Conditioned_Arousal",
            3: "Hypersomnolence",
            4: "RAR_Misalignment",
            5: "Coupling_Impairment",
        }

    candidate_metrics = ["SE_pct", "WASO_min", "SOL_min", "TST_min", "TiB_min","IV","IS","RA","Amplitude","Acrophase",
                            "N1_pct", "N2_pct","N3_pct","REM_pct"]

    metrics = [
        c for c in candidate_metrics
        if c in obs
    ]

    rows = []

    for label, name in phenotype_names.items():
        mask = (y_pred == label)
        n = mask.sum()
        row = {
            "phenotype": name,
            "n": int(n),
        }

        if n == 0:
            rows.append(row)
            continue

        for metric in metrics:

            vals = np.asarray(obs[metric])[mask]

            row[f"{metric}_mean"] = np.nanmean(vals)
            row[f"{metric}_sd"] = np.nanstd(vals)

        rows.append(row)

    return pd.DataFrame(rows)


import numpy as np
import pandas as pd


def agreement_type(layer3, v2, v4):

    if layer3 == v2 == v4:
        return "all_agree"

    if v2 == layer3 and v4 != layer3:
        return "v4_overturn"

    if v4 == layer3 and v2 != layer3:
        return "v4_correction"

    if v2 == v4 and v2 != layer3:
        return "models_agree_not_l3"

    return "all_disagree"


def build_audit3_dataframe(
        y_layer3,
        y_v2,
        y_v4,
        obs,
        cf,
        participant_ids=None,
        night_ids=None,
        medication_df=None,
):
    """
    Night-level adjudication dataset.

    Parameters
    ----------
    y_layer3 : array-like
    y_v2     : array-like
    y_v4     : array-like

    obs : dict-like
        Night-level physiology.

    cf : ConstraintFunctions

    participant_ids : optional array
    night_ids       : optional array

    medication_df : optional DataFrame
        Must contain one row per night
        aligned to predictions.
    """

    y_layer3 = np.asarray(y_layer3)
    y_v2 = np.asarray(y_v2)
    y_v4 = np.asarray(y_v4)

    if y_layer3.ndim == 2:
        y_layer3 = y_layer3.argmax(axis=1)

    if y_v2.ndim == 2:
        y_v2 = y_v2.argmax(axis=1)

    if y_v4.ndim == 2:
        y_v4 = y_v4.argmax(axis=1)

    n = len(y_layer3)

    if participant_ids is None:
        participant_ids = np.arange(n)

    if night_ids is None:
        night_ids = np.arange(n)

    # ----------------------------
    # Physiology rule masks
    # ----------------------------

    norm_mask = cf.normative_mask(obs)
    ca_mask = cf.conditioned_arousal_mask(obs)
    rar_mask = cf.rar_misalignment_mask(obs)
    hyper_mask = cf.hypersomnolence_mask(obs)
    coupling_mask = cf.coupling_mask(obs)

    masks = {
        "normative_phys": norm_mask,
        "ca_phys": ca_mask,
        "rar_phys": rar_mask,
        "hyper_phys": hyper_mask,
        "coupling_phys": coupling_mask,
    }

    for k, v in masks.items():

        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()

        masks[k] = np.asarray(v).astype(bool)

    print("\n--- Length audit ---")

    print("layer3:", len(y_layer3))
    print("v2:", len(y_v2))
    print("v4:", len(y_v4))

    if participant_ids is not None:
        print("participant_ids:", len(participant_ids))

    if night_ids is not None:
        print("night_ids:", len(night_ids))

    for k in obs:
        try:
            print(k, len(obs[k]))
        except:
            pass
    print("------lenght audit end----------")
    # ----------------------------
    # Core dataframe
    # ----------------------------

    df = pd.DataFrame({

        "ParticipantID": participant_ids,
        "NightID": night_ids,

        "Layer3": y_layer3,
        "V2": y_v2,
        "V4": y_v4,
    })

    df["agreement_type"] = [

        agreement_type(l3, v2, v4)

        for l3, v2, v4
        in zip(y_layer3, y_v2, y_v4)
    ]

    # ----------------------------
    # Add physiology variables
    # ----------------------------

    candidate_columns = [

        "SE_pct",
        "WASO_min",
        "SOL_min",
        "TST_min",
        "TiB_min",

        "IV",
        "IS",
        "RA",

        "N1_pct",
        "N2_pct",
        "N3_pct",
        "REM_pct",

        "age",
        "autism_status",
    ]

    for col in candidate_columns:

        if col in obs:

            values = obs[col]

            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()

            df[col] = np.asarray(values)

    # ----------------------------
    # Add physiology eligibility
    # ----------------------------

    for col, values in masks.items():
        df[col] = values

    # ----------------------------
    # Medication/context
    # ----------------------------

    if medication_df is not None:

        med_cols = [

            c for c in medication_df.columns
            if c not in df.columns
        ]

        df = pd.concat(
            [
                df.reset_index(drop=True),
                medication_df[med_cols].reset_index(drop=True),
            ],
            axis=1,
        )

    return df


def audit3_summary(df):

    print("\n==============================")
    print("Audit 3 Agreement Summary")
    print("==============================\n")

    counts = (
        df["agreement_type"]
        .value_counts()
        .sort_values(ascending=False)
    )

    pct = (
        100
        * counts
        / len(df)
    )

    summary = pd.DataFrame({
        "count": counts,
        "pct": pct.round(1)
    })

    print(summary)

    return summary


def migration_matrix(df):

    return pd.crosstab(
        df["V2"],
        df["V4"],
        margins=True,
    )

def trans_matrix_l3_l4v4():
    l3=pd.read_csv('/mnt/data/y_patient.csv')
    v3=pd.read_csv('/mnt/data/y_patient_L4_V3.csv')
    merged=l3[['participant_id','Y_Patient_name']].rename(columns={'Y_Patient_name':'Layer3'}).merge(v3[['participant_id','Y_Patient_name']].rename(columns={'Y_Patient_name':'V3'}),on='participant_id')
    ct=pd.crosstab(merged['Layer3'],merged['V3'])
    print(ct)
    print('\nN=',len(merged))
    # top transitions
    trans=merged.groupby(['Layer3','V3']).size().reset_index(name='n').sort_values('n',ascending=False)
    print(trans.head(20).to_string(index=False))
    # | Layer
    # 3 | CA | CI | Hyper | Norm | RAR | Cond + RAR | Unstable |
    # | -------------------- | -: | -: | ----: | ---: | --: | -------: | -------: |
    # | Conditioned
    # arousal | 6 | 0 | 0 | 1 | 5 | 3 | 1 |
    # | Coupling
    # impairment | 7 | 1 | 0 | 1 | 8 | 2 | 5 |
    # | Normative
    # range | 0 | 0 | 1 | 7 | 9 | 0 | 5 |
    # | Paradoxical
    # insomnia | 2 | 1 | 0 | 5 | 10 | 0 | 3 |
    # | RAR
    # misalignment | 5 | 0 | 0 | 3 | 10 | 2 | 5 |
    # | Conditioned - coupling | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
    # | Paradoxical - coupling | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
    # | Unstable
    # mixed | 2 | 0 | 0 | 1 | 3 | 0 | 0 |


def nightly_value_comp():
    night = pd.read_csv('/mnt/data/final_night_table.csv')
    l3 = pd.read_csv('/mnt/data/y_patient.csv')[['participant_id', 'Y_Patient_name']].rename(
        columns={'Y_Patient_name': 'L3'})
    v3 = pd.read_csv('/mnt/data/y_patient_L4_V3.csv')[['participant_id', 'Y_Patient_name']].rename(
        columns={'Y_Patient_name': 'V3'})
    d = night.merge(l3, on='participant_id').merge(v3, on='participant_id')
    metrics = ['IV', 'L5_midpoint', 'SOL_min', 'WASO_min', 'SE_pct']
    for ph in ['RAR_misalignment', 'Conditioned_arousal', 'Coupling_impairment', 'Normative_range']:
        print('\nPHENOTYPE', ph)
        a = d[d.L3 == ph][metrics].mean(numeric_only=True)
        b = d[d.V3 == ph][metrics].mean(numeric_only=True)
        print('L3', a.to_dict())
        print('V3', b.to_dict())
        # RAR mislaignment
        # | Metric | Layer
        # 3 | V3 |
        # | ----------- | ------: | ----: |
        # | IV | 0.216 | 0.285 |
        # | L5
        # midpoint | 4.78 | 4.59 |
        # | SOL(min) | 30.5 | 26.0 |
        # | WASO(min) | 30.6 | 24.9 |
        # | SE( %) | 88.2 | 89.6 |
        # Interpretation
        #
        # V3
        # RAR
        # participants
        # have:
        #
        # substantially
        # higher
        # IV(+32 %)
        # slightly
        # later / disrupted
        # rest - activity
        # structure
        # better
        # sleep
        # efficiency
        # lower
        # SOL and WASO
        #
        # This
        # suggests
        # V3 is identifying
        # a
        # cleaner
        # circadian / rhythm
        # phenotype, rather
        # than
        # a
        # general
        # poor - sleep
        # phenotype.
        #
        # This
        # supports
        # your
        # RAR
        # story.




"""
NeSy participant phenotype report generator.

Produces a one-page-per-participant PDF report documenting the full clinical
reasoning chain from Layer 1 (data ingestion) through Layer 5 (validation),
making every decision traceable to the symbolic rules that generated it.

Usage (standalone):
    from nesy.reporting.report import generate_cohort_reports
    generate_cohort_reports(
        run_dir      = Path("runs/20260518_023301"),
        night_labels = result_df,
        y_patient    = strat_df,
        night_record = night_record_df,
        med_ctx      = med_ctx_df,
        rar_p        = rar_participant_df,
        demo         = demographics_df,
        gbtm         = context_df,
        output_dir   = Path("runs/20260518_023301/reports"),
    )

Or triggered via --report flag in test_pipeline.py.
"""

from pathlib import Path
import datetime, math
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE       = colors.HexColor("#185FA5")
C_BLUE_LIGHT = colors.HexColor("#E6F1FB")
C_TEAL       = colors.HexColor("#0F6E56")
C_TEAL_LIGHT = colors.HexColor("#E1F5EE")
C_AMBER      = colors.HexColor("#854F0B")
C_AMBER_LIGHT= colors.HexColor("#FAEEDA")
C_RED        = colors.HexColor("#A32D2D")
C_RED_LIGHT  = colors.HexColor("#FCEBEB")
C_GRAY       = colors.HexColor("#5F5E5A")
C_GRAY_LIGHT = colors.HexColor("#F1EFE8")
C_PURPLE     = colors.HexColor("#534AB7")
C_PURPLE_LIGHT=colors.HexColor("#EEEDFE")
C_GREEN      = colors.HexColor("#3B6D11")
C_GREEN_LIGHT= colors.HexColor("#EAF3DE")
C_BORDER     = colors.HexColor("#D3D1C7")
C_TEXT       = colors.HexColor("#2C2C2A")
C_TEXT_MUTED = colors.HexColor("#888780")
C_WHITE      = colors.white
C_BLACK      = colors.black

# phenotype colour map
PHENOTYPE_COLORS = {
    "coupling_impairment":            C_BLUE,
    "psychophysiological_insomnia":   C_TEAL,
    "paradoxical_insomnia":           C_PURPLE,
    "restless":                       C_AMBER,
    "rar_misalignment":               colors.HexColor("#639922"),
    "hypersomnolence":                colors.HexColor("#BA7517"),
    "sdb_risk":                       C_RED,
    "healthy":                        colors.HexColor("#1D9E75"),
    "unstable_mixed":                 C_GRAY,
    "paradoxical_coupling_impairment":colors.HexColor("#7F77DD"),
}

LABEL_INT_TO_NAME = {
    0: "healthy",
    1: "paradoxical_insomnia",
    2: "psychophysiological_insomnia",
    3: "restless",
    4: "rar_misalignment",
    5: "hypersomnolence",
    6: "sdb_risk",
    7: "coupling_impairment",
}

PHENOTYPE_DISPLAY = {
    "coupling_impairment":            "Coupling impairment",
    "psychophysiological_insomnia":   "Psychophysiological insomnia",
    "paradoxical_insomnia":           "Paradoxical insomnia",
    "restless":                       "Restless sleep",
    "rar_misalignment":               "RAR misalignment",
    "hypersomnolence":                "Hypersomnolence",
    "sdb_risk":                       "SDB risk",
    "healthy":                        "Healthy sleeper",
    "unstable_mixed":                 "Unstable / mixed",
    "paradoxical_coupling_impairment":"Paradoxical + coupling",
}


# ── Style definitions ─────────────────────────────────────────────────────────
def _styles():
    return {
        "title": ParagraphStyle(
            "title", fontName="Helvetica-Bold", fontSize=14,
            textColor=C_TEXT, leading=18, spaceAfter=2),
        "subtitle": ParagraphStyle(
            "subtitle", fontName="Helvetica", fontSize=9,
            textColor=C_TEXT_MUTED, leading=12, spaceAfter=6),
        "section": ParagraphStyle(
            "section", fontName="Helvetica-Bold", fontSize=9,
            textColor=C_TEXT_MUTED, leading=12, spaceBefore=8, spaceAfter=4,
            textTransform="uppercase"),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=9,
            textColor=C_TEXT, leading=13, spaceAfter=3),
        "body_sm": ParagraphStyle(
            "body_sm", fontName="Helvetica", fontSize=8,
            textColor=C_TEXT_MUTED, leading=11, spaceAfter=2),
        "metric_label": ParagraphStyle(
            "metric_label", fontName="Helvetica", fontSize=7,
            textColor=C_TEXT_MUTED, leading=9),
        "metric_val": ParagraphStyle(
            "metric_val", fontName="Helvetica-Bold", fontSize=11,
            textColor=C_TEXT, leading=13),
        "verdict": ParagraphStyle(
            "verdict", fontName="Helvetica-Bold", fontSize=13,
            textColor=C_WHITE, leading=16),
        "verdict_sub": ParagraphStyle(
            "verdict_sub", fontName="Helvetica", fontSize=8,
            textColor=C_WHITE, leading=10),
        "rule_fired": ParagraphStyle(
            "rule_fired", fontName="Helvetica-Bold", fontSize=8,
            textColor=C_BLUE, leading=10),
        "rule_abstain": ParagraphStyle(
            "rule_abstain", fontName="Helvetica", fontSize=8,
            textColor=C_TEXT_MUTED, leading=10),
        "evidence": ParagraphStyle(
            "evidence", fontName="Helvetica-Oblique", fontSize=7.5,
            textColor=C_TEXT_MUTED, leading=10),
    }


# ── Night grid drawing ────────────────────────────────────────────────────────
def _night_grid(nights_df, med_dates, width=160*mm, cell_h=6*mm):
    """
    Draw a grid of coloured squares — one per night.
    Blue = coupling impairment, grey = stimulant-confounded, white = unclassified.
    Returns a reportlab Drawing.
    """
    n   = len(nights_df)
    cols = min(n, 21)
    rows = math.ceil(n / cols)
    gap  = 1.5
    cw   = (width - gap * (cols - 1)) / cols

    total_h = rows * cell_h + (rows - 1) * gap + 8  # +8 for legend
    d = Drawing(width, total_h)

    for i, (_, row) in enumerate(nights_df.iterrows()):
        col_i = i % cols
        row_i = i // cols
        x = col_i * (cw + gap)
        y = total_h - 8 - (row_i + 1) * cell_h - row_i * gap

        nd   = row.get("night_date", "")
        lbl  = row.get("hard_label", -1)
        stim = nd in med_dates

        if stim:
            fill = colors.HexColor("#FAC775")   # amber — stimulant night
        elif lbl < 0:
            fill = colors.HexColor("#D3D1C7")   # grey — unclassified
        else:
            pname = LABEL_INT_TO_NAME.get(int(lbl), "healthy")
            fill  = PHENOTYPE_COLORS.get(pname, C_GRAY_LIGHT)

        r = Rect(x, y, cw, cell_h - gap,
                 fillColor=fill, strokeColor=None, rx=1.5)
        d.add(r)

    # legend
    legend_items = [
        (PHENOTYPE_COLORS.get("coupling_impairment", C_BLUE), "dominant label"),
        (colors.HexColor("#FAC775"),                           "stimulant night"),
        (colors.HexColor("#D3D1C7"),                           "unclassified"),
    ]
    lx = 0
    for (lc, lt) in legend_items:
        d.add(Rect(lx, 0, 6, 5, fillColor=lc, strokeColor=None, rx=1))
        d.add(String(lx + 8, 1, lt,
                     fontName="Helvetica", fontSize=6,
                     fillColor=C_TEXT_MUTED))
        lx += 60

    return d


# ── Metric table helper ───────────────────────────────────────────────────────
def _metric_row(pairs, styles_):
    """Create a row of metric pill-boxes from [(label, value, alert?), ...]."""
    data  = [[Paragraph(lbl, styles_["metric_label"]),
              Paragraph(str(val), styles_["metric_val"])]
             for lbl, val, _ in pairs]
    alert = [a for _, _, a in pairs]

    tbl = Table([data], colWidths=[None] * len(pairs))
    style_cmds = [
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (-1,-1), C_GRAY_LIGHT),
        ("ROUNDEDCORNERS", [4]),
        ("BOX",        (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0), (-1,-1), 8),
    ]
    for i, a in enumerate(alert):
        if a:
            style_cmds += [("BACKGROUND", (i,0), (i,0), C_RED_LIGHT)]
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ── Rule evidence table ───────────────────────────────────────────────────────
def _rule_table(rules, styles_):
    """
    rules = list of dicts:
        name, fired (bool), label_voted, evidence_str, abstain_reason
    """
    rows = []
    for r in rules:
        status = "FIRED" if r["fired"] else "abstained"
        s_col  = C_BLUE if r["fired"] else C_TEXT_MUTED
        s_font = "Helvetica-Bold" if r["fired"] else "Helvetica"

        rows.append([
            Paragraph(r["name"], ParagraphStyle(
                "rs", fontName=s_font, fontSize=8,
                textColor=s_col, leading=10)),
            Paragraph(r.get("label_voted","—"), ParagraphStyle(
                "rv", fontName="Helvetica-Bold" if r["fired"] else "Helvetica",
                fontSize=8, textColor=s_col, leading=10)),
            Paragraph(r.get("evidence_str",""), ParagraphStyle(
                "re", fontName="Helvetica-Oblique", fontSize=7.5,
                textColor=C_TEXT_MUTED, leading=10)),
        ])

    col_w = [45*mm, 48*mm, 75*mm]
    tbl   = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LINEBELOW",   (0,0), (-1,-2), 0.3, C_BORDER),
        ("BACKGROUND",  (0,0), (-1,-1), colors.white),
    ]))
    return tbl


# ── Phenotype bar drawing ─────────────────────────────────────────────────────
def _phenotype_bar(probs, width=168*mm, height=8*mm):
    """Draw a stacked bar of phenotype probabilities."""
    d = Drawing(width, height + 4)
    x = 0
    for lbl_int, prob in sorted(probs.items(), key=lambda x: -x[1]):
        if prob < 0.02:
            continue
        pname = LABEL_INT_TO_NAME.get(lbl_int, "healthy")
        fill  = PHENOTYPE_COLORS.get(pname, C_GRAY)
        w     = width * prob
        d.add(Rect(x, 4, w, height, fillColor=fill, strokeColor=None))
        if w > 15:
            d.add(String(x + w/2, 7, f"{prob:.0%}",
                         fontName="Helvetica-Bold", fontSize=6,
                         fillColor=colors.white, textAnchor="middle"))
        x += w
    return d


# ── Section header ────────────────────────────────────────────────────────────
def _section_box(title, color_fill, color_text, width=168*mm, h=7*mm):
    d = Drawing(width, h)
    d.add(Rect(0, 0, width, h, fillColor=color_fill,
               strokeColor=None, rx=3))
    d.add(String(6, h * 0.3, title.upper(),
                 fontName="Helvetica-Bold", fontSize=7.5,
                 fillColor=color_text))
    return d


# ── Main report builder ───────────────────────────────────────────────────────
def generate_participant_report(
    pid,
    night_labels,   # DataFrame: participant_id, night_date, hard_label, vote_r1..
    y_patient,      # DataFrame: participant_id, Y_Patient_name, instability..
    night_record,   # DataFrame: the aligned night record (SE, SOL, ANX etc)
    med_ctx,        # DataFrame indexed (participant_id, night_date)
    rar_p,          # DataFrame indexed participant_id: IS, IV etc
    demo,           # DataFrame indexed participant_id: age, sex, autism_status
    gbtm,           # DataFrame indexed participant_id: dominant_group, confidence
    output_path,    # Path to write the PDF
):
    S = _styles()
    W, H = A4
    M    = 15*mm

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=M, rightMargin=M,
        topMargin=M, bottomMargin=M,
        title=f"NeSy Sleep Phenotype Report — {pid}",
        author="NeSy Sleep Phenotyping Framework",
    )

    story = []

    # ── participant data ──────────────────────────────────────────────────────
    p_nights = night_labels[night_labels["participant_id"] == pid].copy()
    p_nights = p_nights.sort_values("night_date").reset_index(drop=True)

    p_strat  = y_patient[y_patient.index == pid] if pid in y_patient.index \
               else y_patient[y_patient.get("participant_id","") == pid]

    p_demo   = demo.loc[pid] if pid in demo.index else pd.Series()
    p_rar    = rar_p.loc[pid] if pid in rar_p.index else pd.Series()
    p_gbtm   = gbtm.loc[pid] if gbtm is not None and pid in gbtm.index \
               else pd.Series()

    # medication dates for this participant
    stim_dates = set()
    if med_ctx is not None:
        try:
            p_med = med_ctx.loc[pid]
            stim_dates = set(
                p_med[p_med["med_stimulant_today"]].index.tolist()
            )
        except Exception:
            pass

    # night record for this participant
    p_nr = pd.DataFrame()
    if night_record is not None:
        try:
            p_nr = night_record.xs(pid, level="participant_id")
        except Exception:
            pass

    n_total     = len(p_nights)
    n_labelled  = (p_nights["hard_label"] >= 0).sum()
    n_stim      = len(stim_dates)
    date_range  = (f"{p_nights['night_date'].iloc[0]} "
                   f"to {p_nights['night_date'].iloc[-1]}"
                   if len(p_nights) else "—")

    phenotype   = "unknown"
    instability = 0.0
    confidence  = 0.0
    if len(p_strat):
        row = p_strat.iloc[0]
        phenotype   = str(row.get("Y_Patient_name", row.get("dominant_phenotype","unknown")))
        instability = float(row.get("instability", 0.0))

    # dominant night label confidence
    prob_cols = [c for c in p_nights.columns if c.startswith("prob_label_")]
    if prob_cols:
        confidence = float(p_nights[prob_cols].max(axis=1).mean())

    # ── HEADER ───────────────────────────────────────────────────────────────
    phen_color = PHENOTYPE_COLORS.get(phenotype, C_GRAY)
    phen_disp  = PHENOTYPE_DISPLAY.get(phenotype, phenotype.replace("_"," ").title())

    # header table
    age_str     = str(int(p_demo.get("age", "?"))) if "age" in p_demo else "?"
    sex_str     = str(p_demo.get("sex","?"))
    asd_str     = "Autistic" if p_demo.get("autism_status",0)==1 else "Sibling"
    fam_str     = str(p_demo.get("family_id","?"))

    header_data = [[
        Paragraph(f"<b>{pid}</b>", ParagraphStyle(
            "h1", fontName="Helvetica-Bold", fontSize=15,
            textColor=C_TEXT, leading=18)),
        Paragraph(
            f"Age {age_str} &nbsp;·&nbsp; {sex_str} &nbsp;·&nbsp; "
            f"{asd_str} &nbsp;·&nbsp; Family {fam_str}",
            ParagraphStyle("h2", fontName="Helvetica", fontSize=9,
                           textColor=C_TEXT_MUTED, leading=12)),
        Paragraph(
            f"<b>{phen_disp}</b>",
            ParagraphStyle("phen", fontName="Helvetica-Bold", fontSize=13,
                           textColor=phen_color, leading=16,
                           alignment=TA_RIGHT)),
        Paragraph(
            f"Recording: {date_range}<br/>"
            f"Nights: {n_labelled}/{n_total} labelled",
            ParagraphStyle("dt", fontName="Helvetica", fontSize=8,
                           textColor=C_TEXT_MUTED, leading=11,
                           alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(
        [header_data[0][:2], header_data[0][2:]],
        colWidths=[W - 2*M]
    )
    # simpler: flat table
    flat = Table([[
        Paragraph(f"<b>{pid}</b>", ParagraphStyle(
            "hpid", fontName="Helvetica-Bold", fontSize=14,
            textColor=C_TEXT, leading=17)),
        Paragraph(
            f"Age {age_str} · {sex_str} · {asd_str} · Family {fam_str}",
            ParagraphStyle("hmeta", fontName="Helvetica", fontSize=9,
                           textColor=C_TEXT_MUTED, leading=12)),
        Paragraph(
            f"<b>{phen_disp}</b>",
            ParagraphStyle("hphen", fontName="Helvetica-Bold", fontSize=13,
                           textColor=phen_color, leading=16,
                           alignment=TA_RIGHT)),
    ]], colWidths=[35*mm, 80*mm, 53*mm])
    flat.setStyle(TableStyle([
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story.append(flat)

    sub = Paragraph(
        f"NeSy sleep phenotyping framework · report generated "
        f"{datetime.date.today()} · recording {date_range} · "
        f"{n_labelled}/{n_total} nights labelled",
        ParagraphStyle("gen", fontName="Helvetica", fontSize=7.5,
                       textColor=C_TEXT_MUTED, leading=10))
    story.append(sub)
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=C_BORDER, spaceAfter=8))

    # ── LAYER 1 ──────────────────────────────────────────────────────────────
    story.append(_section_box("Layer 1 — data ingestion",
                               C_TEAL_LIGHT, C_TEAL))
    story.append(Spacer(1, 3))

    rar_days    = int(p_rar.get("n_days", n_total)) if len(p_rar) else n_total
    dreem_n     = int((p_nights["hard_label"] >= 0).sum())
    diary_n     = int(p_nights.get("diary_missing", pd.Series(
                      [False]*n_total)).sum()) if "diary_missing" in p_nights \
                  else "?"
    diary_pct   = f"{100*(1 - p_nights.get('diary_missing', pd.Series([0]*n_total)).mean()):.0f}%" \
                  if "diary_missing" in p_nights else "?"

    l1_pairs = [
        ("Total nights", n_total, False),
        ("Valid RAR days", rar_days, rar_days < 5),
        ("Dreem nights",  dreem_n,  dreem_n < 3),
        ("Diary completion", diary_pct, False),
        ("Stimulant nights", n_stim, n_stim > 0),
        ("Age band", "14–17" if p_demo.get("age_band",0)==1 else "10–13", False),
    ]
    story.append(_metric_row(l1_pairs, S))
    story.append(Spacer(1, 6))

    # ── LAYER 2 ──────────────────────────────────────────────────────────────
    story.append(_section_box("Layer 2 — knowledge context",
                               C_PURPLE_LIGHT, C_PURPLE))
    story.append(Spacer(1, 3))

    IS   = p_rar.get("IS", None)
    IV   = p_rar.get("IV", None)
    L5   = p_rar.get("L5_midpoint_mean", None)

    IS_str  = f"{IS:.3f}" if IS is not None else "?"
    IV_str  = f"{IV:.3f}" if IV is not None else "?"
    L5_str  = f"{L5:.2f}h" if L5 is not None else "?"
    IS_flag = IS is not None and IS < 0.40

    gbtm_grp  = f"Group {int(p_gbtm.get('dominant_group', '?'))}" \
                if len(p_gbtm) else "—"
    gbtm_conf = f"{p_gbtm.get('confidence',0):.2f}" if len(p_gbtm) else "—"

    l2_pairs = [
        ("IS (interdaily stability)", IS_str, IS_flag),
        ("IS threshold",  ">=0.45 normal",     False),
        ("IV (fragmentation)", IV_str,          IV is not None and IV > 0.70),
        ("L5 midpoint mean",   L5_str,          False),
        ("GBTM group",         gbtm_grp,        False),
        ("GBTM confidence",    gbtm_conf,       False),
    ]
    story.append(_metric_row(l2_pairs, S))

    if IS_flag:
        story.append(Paragraph(
            f"IS={IS_str} is below the adolescent norm (>=0.45), indicating "
            "chronically disrupted rest-activity rhythm coupling. "
            "This participant-level gate will activate lf_r3_coupling on every night.",
            S["body_sm"]))
    if n_stim > 0:
        story.append(Paragraph(
            f"{n_stim} stimulant nights detected. R1 (subjective labelling function) "
            "will abstain on these nights — SOL elevation may reflect pharmacological "
            "effect rather than intrinsic sleep disturbance.",
            S["body_sm"]))
    story.append(Spacer(1, 6))

    # ── LAYER 3 ──────────────────────────────────────────────────────────────
    story.append(_section_box("Layer 3 — symbolic labelling (teacher)",
                               C_AMBER_LIGHT, C_AMBER))
    story.append(Spacer(1, 3))

    vote_cols = [c for c in p_nights.columns if c.startswith("vote_")]

    # per-rule summary
    def rule_summary(col, fired_vals, label_name, evidence_fn):
        if col not in p_nights.columns:
            return {"name": col, "fired": False,
                    "label_voted": "—", "evidence_str": "column missing"}
        votes     = p_nights[col]
        n_fired   = (votes.isin(fired_vals)).sum()
        n_abstain = (votes == -1).sum()
        evidence  = evidence_fn()
        return {
            "name":         col.replace("vote_", "lf_"),
            "fired":        n_fired > 0,
            "label_voted":  f"{label_name} ({n_fired}/{n_total} nights)"
                            if n_fired > 0 else "abstained",
            "evidence_str": evidence,
        }

    # mean values from night record for evidence strings
    def nr_mean(col):
        if len(p_nr) and col in p_nr.columns:
            return p_nr[col].dropna().mean()
        return None

    SE_mean  = nr_mean("SE_pct")
    SOL_mean = nr_mean("SOL_min")
    ANX_mean = nr_mean("ANX")
    N1_mean  = nr_mean("N1_pct")
    N3_mean  = nr_mean("N3_pct")
    TST_mean = nr_mean("TST_min")
    WASO_mean= nr_mean("WASO_min")

    rules = [
        rule_summary(
            "vote_r1", [0,1,2,3], "subjective phenotype",
            lambda: (
                f"SE={SE_mean:.0f}% · SOL={SOL_mean:.0f} min · ANX={ANX_mean:.1f} · "
                f"{n_stim} nights abstained (stimulant)"
                if SE_mean is not None else "diary metrics not available"
            )),
        rule_summary(
            "vote_r2_hypersomnia", [5], "hypersomnolence",
            lambda: (
                f"TST mean={TST_mean:.0f} min · threshold >600 min"
                if TST_mean is not None else "TST not available"
            )),
        rule_summary(
            "vote_r2_sdb", [6], "SDB risk (direct override)",
            lambda: (
                f"N3={N3_mean:.0f}% · WASO={WASO_mean:.0f} min · "
                "threshold: N3<5% AND WASO>60 min"
                if N3_mean is not None else "EEG not available"
            )),
        rule_summary(
            "vote_r2_restless", [3], "restless sleep",
            lambda: (
                f"N1={N1_mean:.0f}% · threshold >=15% AND WASO>=60 min"
                if N1_mean is not None else "EEG not available"
            )),
        rule_summary(
            "vote_r3_coupling", [7], "coupling impairment",
            lambda: (
                f"IS={IS_str} < threshold 0.40 — participant gate "
                "(checked once across full recording)"
                if IS is not None else "IS not available"
            )),
        rule_summary(
            "vote_r3_misalign", [4], "RAR misalignment",
            lambda: (
                f"L5 midpoint mean={L5_str} · threshold >=4.5h · "
                f"IV={IV_str} · threshold >=0.45"
                if L5 is not None else "RAR metrics not available"
            )),
    ]
    story.append(_rule_table(rules, S))
    story.append(Spacer(1, 5))

    # Snorkel probability bar
    story.append(Paragraph("Snorkel label model · mean night probability",
                            S["body_sm"]))
    if prob_cols:
        probs = {int(c.split("_")[-1]): float(p_nights[c].mean())
                 for c in prob_cols}
        story.append(_phenotype_bar(probs))
    story.append(Spacer(1, 6))

    # ── NIGHT GRID ────────────────────────────────────────────────────────────
    story.append(_section_box("Night-by-night label sequence",
                               C_GRAY_LIGHT, C_GRAY))
    story.append(Spacer(1, 3))
    story.append(_night_grid(p_nights, stim_dates))
    story.append(Spacer(1, 6))

    # ── LAYER 4 ───────────────────────────────────────────────────────────────
    story.append(_section_box("Layer 4 — neuro-symbolic fusion (student)",
                               C_BLUE_LIGHT, C_BLUE))
    story.append(Spacer(1, 3))

    # per-night agreement between symbolic and fusion labels
    fused_col = "fused_label" if "fused_label" in p_nights.columns else None
    if fused_col:
        agree = (p_nights["hard_label"] == p_nights[fused_col]).mean()
        story.append(Paragraph(
            f"Fusion model agreement with symbolic labels: "
            f"{agree:.0%} of nights. "
            f"Disagreements occur on nights where the fusion model "
            "(trained on ROCKET + EEG waveform features) predicts a different "
            "phenotype from the symbolic rules — typically on nights with "
            "mixed or ambiguous Snorkel confidence.",
            S["body"]))
    else:
        story.append(Paragraph(
            "Fusion model features: ROCKET (2,000 nocturnal waveform kernels) "
            "+ EEG morphological symbols (24 hypnogram features). "
            "Trained via soft-label row expansion on Snorkel probability "
            "distributions. LOPO-CV: this participant held out during evaluation.",
            S["body"]))
    story.append(Spacer(1, 6))

    # ── LAYER 5 / VERDICT ────────────────────────────────────────────────────
    story.append(_section_box("Post-processing & validation",
                               C_GREEN_LIGHT, C_GREEN))
    story.append(Spacer(1, 3))

    instab_str = f"{instability:.3f}"
    instab_flag= instability > 1.5

    l5_data = [
        [Paragraph("Dominant phenotype", S["metric_label"]),
         Paragraph("Instability score (H)", S["metric_label"]),
         Paragraph("Assignment type", S["metric_label"]),
         Paragraph("Coverage", S["metric_label"])],
        [Paragraph(f"<b>{phen_disp}</b>", ParagraphStyle(
             "pv", fontName="Helvetica-Bold", fontSize=11,
             textColor=phen_color, leading=13)),
         Paragraph(f"<b>{instab_str}</b>" +
                   (" ⚠ high" if instab_flag else " (low)"),
                   ParagraphStyle("iv", fontName="Helvetica-Bold",
                   fontSize=11,
                   textColor=C_RED if instab_flag else C_TEAL, leading=13)),
         Paragraph("unstable/mixed" if instab_flag else "dominant",
                   S["metric_val"]),
         Paragraph(f"{n_labelled}/{n_total} nights ({100*n_labelled//n_total}%)",
                   S["metric_val"])],
    ]
    l5_tbl = Table(l5_data, colWidths=[48*mm]*4)
    l5_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_GRAY_LIGHT),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
        ("LINEAFTER",   (0,0),(-2,-1), 0.5, C_BORDER),
    ]))
    story.append(l5_tbl)
    story.append(Spacer(1, 5))

    # sibling
    sibling_id  = p_demo.get("family_id", None)
    if sibling_id:
        story.append(Paragraph(
            f"Family {sibling_id} · within-family concordance check: "
            "sibling phenotype available in y_patient.csv for comparison.",
            S["body_sm"]))
    story.append(Spacer(1, 8))

    # ── CLINICAL IMPLICATION BOX ──────────────────────────────────────────────
    implications = {
        "coupling_impairment":
            "Chronically disrupted rest-activity rhythm coupling (IS below norm). "
            "Circadian-targeted intervention indicated: light therapy, "
            "chronotherapy, or melatonin timed to advance phase. "
            "CBT-I alone is unlikely to address the core circadian mechanism.",
        "psychophysiological_insomnia":
            "Anxiety-driven conditioned hyperarousal with objective sleep "
            "impairment (elevated SOL or reduced SE). CBT-I is the first-line "
            "evidence-based treatment. Stimulus control and sleep restriction "
            "protocols are particularly indicated.",
        "paradoxical_insomnia":
            "Subjective sleep complaint with preserved objective sleep architecture. "
            "Sleep misperception is the primary clinical target. Psychoeducation, "
            "mindfulness, and cognitive restructuring are indicated. "
            "Hypnotic medication should be avoided.",
        "restless":
            "Elevated N1 sleep and WASO consistent with restless sleep disorder. "
            "Evaluate for periodic limb movement disorder, obstructive sleep "
            "apnoea, or iron deficiency. Polysomnography referral may be warranted.",
        "hypersomnolence":
            "Total sleep time substantially exceeding age norms. "
            "Evaluate for hypersomnia disorders (idiopathic hypersomnia, "
            "narcolepsy), mood disorder (depression), or medication effects. "
            "Full diagnostic assessment recommended.",
        "sdb_risk":
            "EEG architecture consistent with sleep-disordered breathing risk "
            "(low N3, elevated WASO). Referral for overnight polysomnography "
            "or home sleep apnoea test is indicated.",
        "rar_misalignment":
            "Severe circadian misalignment with delayed rest-activity phase "
            "and high fragmentation. Delayed sleep-wake phase disorder should "
            "be evaluated. Light therapy in the morning and melatonin in the "
            "early evening are first-line chronobiological treatments.",
        "healthy":
            "Sleep metrics within age-appropriate norms across all assessed "
            "domains. No clinical intervention indicated at this time. "
            "Monitor for phenotype stability across future recordings.",
        "unstable_mixed":
            "Night-to-night phenotypic variability prevents stable assignment. "
            "This instability is itself clinically meaningful — the sleep "
            "problem does not follow a consistent pattern. "
            "Multi-component sleep assessment and diary monitoring are recommended "
            "before initiating specific treatment.",
    }
    impl_text = implications.get(
        phenotype,
        "Clinical implication not available for this phenotype.")

    impl_tbl = Table(
        [[Paragraph("Clinical implication", ParagraphStyle(
             "cl", fontName="Helvetica-Bold", fontSize=8,
             textColor=C_WHITE, leading=10)),
          Paragraph(impl_text, ParagraphStyle(
             "ci", fontName="Helvetica", fontSize=8.5,
             textColor=C_WHITE, leading=12))]],
        colWidths=[36*mm, 132*mm]
    )
    impl_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), phen_color),
        ("TOPPADDING",  (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING", (0,0),(-1,-1), 10),
        ("VALIGN",      (0,0),(-1,-1), "TOP"),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(impl_tbl)

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This report was generated automatically by the NeSy Sleep Phenotyping "
        "Framework. All phenotype assignments are traceable to the symbolic "
        "labelling rules documented in the framework paper. This report is "
        "intended for research purposes and does not constitute a clinical diagnosis.",
        ParagraphStyle("disc", fontName="Helvetica-Oblique", fontSize=7,
                       textColor=C_TEXT_MUTED, leading=9)))

    doc.build(story)
    return output_path


# ── Batch generator ───────────────────────────────────────────────────────────
def generate_cohort_reports(
    run_dir,
    night_labels,
    y_patient,
    night_record,
    med_ctx,
    rar_p,
    demo,
    gbtm,
    output_dir=None,
    pids=None,
):
    """
    Generate one PDF report per participant.

    Parameters
    ----------
    run_dir      : Path — the run directory (reports saved here if output_dir None)
    night_labels : DataFrame — per-night labels with vote columns
    y_patient    : DataFrame — participant phenotype assignments
    night_record : DataFrame — aligned night record (SE, SOL, ANX etc)
    med_ctx      : DataFrame — medication context indexed (pid, night_date)
    rar_p        : DataFrame — participant-level RAR metrics
    demo         : DataFrame — demographics indexed by participant_id
    gbtm         : DataFrame — GBTM group assignments indexed by participant_id
    output_dir   : Path — where to write PDFs (default: run_dir/reports/)
    pids         : list — subset of participant IDs (default: all)
    """
    out = Path(output_dir) if output_dir else Path(run_dir) / "reports"
    out.mkdir(parents=True, exist_ok=True)

    all_pids = pids or sorted(night_labels["participant_id"].unique())
    n = len(all_pids)
    print(f"  Generating {n} participant reports → {out}/")

    for i, pid in enumerate(all_pids):
        pdf_path = out / f"{pid}_phenotype_report.pdf"
        try:
            generate_participant_report(
                pid=pid,
                night_labels=night_labels,
                y_patient=y_patient,
                night_record=night_record,
                med_ctx=med_ctx,
                rar_p=rar_p,
                demo=demo,
                gbtm=gbtm,
                output_path=pdf_path,
            )
            print(f"    [{i+1}/{n}] {pid} → {pdf_path.name}", end="\r")
        except Exception as e:
            print(f"    [{i+1}/{n}] {pid} FAILED: {e}")

    print(f"\n  Done. {n} reports saved to {out}/")
    return out

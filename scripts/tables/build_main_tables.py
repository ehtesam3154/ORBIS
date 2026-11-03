"""Main results table and the supplement's per-test, control, per-training, one-call and sampling tables (LaTeX).

usage: python scripts/tables/build_main_tables.py REGISTRY.json OUT_DIR
Writes main_table.tex, supp_full_table.tex, supp_pertest2_table.tex, supp_controls_table.tex, supp_seed_table.tex,
supp_bestguess_table.tex and supp_sampling_table.tex. The registry format: scripts/tables/registry.py.
"""
import os
import sys

import numpy as np

from registry import jload, load, mean_axis, mean_over

FAMILIES = [("Whole brains", ["abca3", "abca2", "abca1", "allen638850"]),
            ("Thick blocks", ["dstarA", "dstarB"]),
            ("Embryos", ["flysta2_c", "flysta2_d"])]
TEST_NAME = {"abca3": "Mouse 3", "abca2": "Mouse 2", "abca1": "Mouse 1", "allen638850": "Allen",
             "dstarA": "Deep-STARmap A", "dstarB": "Deep-STARmap B", "flysta2_c": "Embryo c", "flysta2_d": "Embryo d"}
BASELINES = []
NAMES = {}
MAIN_GROUPS = [("Specimen and cell types", [("size_err", "Size error\\,$\\downarrow$", False, "{:.3f}"),
                                             ("w2_sm", "Shape $W_2\\downarrow$\\\\($\\mu$m)", False, None),
                                             ("f8ell_sm", "F-score\\,$\\uparrow$", True, "{:.3f}"),
                                             ("centre", "Type\\\\centre\\,$\\downarrow$\\\\($\\mu$m)", False, "um")]),
               ("Cells and neighborhoods", [("placement_raw", "Placement\\,$\\uparrow$", True, "{:.3f}"),
                                            ("nmi", "Niche\\\\NMI\\,$\\uparrow$", True, "{:.3f}"),
                                            ("contact_r_offdiag", "Type\\\\contacts\\,$\\uparrow$", True, "{:.3f}"),
                                            ("moran_r", "Moran's $I\\uparrow$", True, "{:.3f}")])]
MAIN_METRICS = [m for _, ms in MAIN_GROUPS for m in ms]
AXES = {"Whole brains": ("ML", "DV", "AP"), "Thick blocks": ("$x$", "$y$", "$z$"), "Embryos": ("AP", "DV", "LR")}
FAMKEY = {"Whole brains": "brains", "Thick blocks": "blocks", "Embryos": "embryos"}
SHADE = "\\cellcolor{oursrow}"
RULE = "\\Xhline{0.8pt}"
MARKS_NOTE = "\\textbf{Bold}: best, \\underline{underline}: 2nd, \\textit{italic}: 3rd"


def ours_sources(t, kind="reseated"):
    """[(training label, draws)] of our trainings of a test, seed 0 first."""
    return [(f"seed {i}", s[kind]) for i, s in enumerate(t["ours"]) if kind in s]


def metric(t, draws, k):
    """A row's value of metric k on a test: the mean over its draws."""
    return mean_over(draws, k, t["view"], t["ruler"])


def fmt_w2(v, fam):
    return f"{v:.1f}" if fam == "Embryos" else f"{v:,.0f}".replace(",", "{,}")


def fmt(v, metric_fmt, fam=None):
    if metric_fmt is None:
        return fmt_w2(v, fam)
    s = metric_fmt.format(v)
    if metric_fmt == "{:.3f}":
        s = s.replace("-0.", "-.", 1) if s.startswith("-0.") else (s[1:] if s.startswith("0.") else s)
    return s


def rank_marks(values, higher):
    """Bold best, underline second, italic third (ties share the mark)."""
    arr = np.array([np.nan if v is None else v for v in values], float)
    order = sorted({v for v in arr if not np.isnan(v)}, reverse=higher)
    lv = {v: m for v, m in zip(order[:3], "bui")}
    return [None if np.isnan(v) else lv.get(v) for v in arr]


def deco(s, mark):
    return {"b": f"\\mathbf{{{s}}}", "u": f"\\underline{{{s}}}", "i": f"\\mathit{{{s}}}", None: s}[mark]


def num(s, mark=None, sd=None, shade=False):
    """One table entry in math mode; the training SD as a subscript; shade = our row."""
    c = f"${deco(s, mark)}" + (f"_{{\\pm {sd}}}$" if sd is not None else "$")
    return SHADE + c if shade else c


def rotlab(n, text, sub=None):
    """Rotated block label spanning n rows (first column)."""
    inner = f"\\textbf{{{text}}}" + (f"\\\\{sub}" if sub else "")
    return f"\\multirow{{{n}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\makecell{{{inner}}}}}}}"


def sfmt(v, f, fam):
    if f == "pct":
        return f"{100 * v:.1f}"
    if f == "um":
        return f"{v:.1f}" if fam == "Embryos" else f"{v:,.0f}".replace(",", "{,}")
    if f == "int":
        return f"{v:,.0f}".replace(",", "{,}")
    return fmt(v, f, fam)


def head2(first, groups, span_single=True):
    """Two header rows: group names (a one-metric group spans both rows), then two-line metric names."""
    spec, r1, cl, r2, c = "c|l", [f"\\multirow{{3}}{{*}}{{\\textbf{{{first[0]}}}}}", f"\\multirow{{3}}{{*}}{{\\textbf{{{first[1]}}}}}"], [], ["", ""], 3
    for gi, (g, ms) in enumerate(groups):
        bar = "|" if gi < len(groups) - 1 else ""
        spec += "|" + "c" * len(ms)
        if len(ms) == 1 and span_single:
            r1.append(f"\\multirow{{3}}{{*}}{{\\makecell{{{ms[0][1]}}}}}")
            r2.append("")
        else:
            r1.append(f"\\multicolumn{{{len(ms)}}}{{c{bar}}}{{\\textbf{{{g}}}}}")
            cl.append(f"\\cline{{{c}-{c + len(ms) - 1}}}")
            r2 += [f"\\makecell{{{h}}}" for _, h, _, _ in ms]
        c += len(ms)
    return spec, [RULE, " & ".join(r1) + " \\\\", " ".join(cl), " & ".join(r2) + " \\\\", "\\hline"]


def main_table(tests, out):
    n_min = min(len(ours_sources(t)) for t in tests.values())
    cells = {}
    for fam, dss in FAMILIES:
        for k, _, _, _ in MAIN_METRICS:
            per_seed = []
            for i in range(n_min):
                per_seed.append(np.mean([metric(tests[ds], ours_sources(tests[ds])[i][1], k) for ds in dss]))
            cells[("Ours", fam, k)] = (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)))
            for b, _ in BASELINES:
                vs = [metric(tests[ds], tests[ds]["methods"][b], k) for ds in dss]
                cells[(b, fam, k)] = (float(np.mean(vs)) if None not in vs else None, None)
    methods = [b for b, _ in BASELINES] + ["Ours"]
    label = dict(BASELINES + [("Ours", "\\textbf{Ours}")])
    marks = {}
    for fam, _ in FAMILIES:
        for k, _, hi, _ in MAIN_METRICS:
            ms = rank_marks([cells[(m, fam, k)][0] for m in methods], hi)
            for m, mk in zip(methods, ms):
                marks[(m, fam, k)] = mk
    spec = "c|l" + "".join("|" + "c" * len(ms) for _, ms in MAIN_GROUPS)
    grp, cl, c = [], [], 3
    for gi, (g, ms) in enumerate(MAIN_GROUPS):
        grp.append(f"\\multicolumn{{{len(ms)}}}{{c{'|' if gi < len(MAIN_GROUPS) - 1 else ''}}}{{\\textbf{{{g}}}}}")
        cl.append(f"\\cline{{{c}-{c + len(ms) - 1}}}")
        c += len(ms)
    L = ["% generated by scripts/tables/build_main_tables.py; do not edit by hand",
         "\\begin{table}[t]", "\\centering",
         "\\caption{Mean over the tests of each geometry (per test: Supplementary "
         "Tables~\\ref{tab:supp_brains}--\\ref{tab:supp_embryos2}). Ours: mean$_{\\pm\\text{SD}}$ over " + str(n_min) + " trainings. "
         "Type contacts, Moran's $I$: correlation with the truth. \\textbf{Bold}/\\underline{underline}/\\textit{italic}: 1st/2nd/3rd.}",
         "\\label{tab:main}", "\\setlength{\\tabcolsep}{1.5pt}", "\\renewcommand{\\arraystretch}{1.15}",
         f"\\begin{{tabular}}{{{spec}}}", RULE,
         "\\multirow{4}{*}{\\textbf{Data}} & \\multirow{4}{*}{\\textbf{Method}} & " + " & ".join(grp) + " \\\\", " ".join(cl),
         " & & " + " & ".join(f"\\makecell{{{h}}}" for _, h, _, _ in MAIN_METRICS) + " \\\\", "\\hline"]
    for gi, (fam, _) in enumerate(FAMILIES):
        for mi, m in enumerate(methods):
            sh = m == "Ours"
            row = []
            for k, _, _, f in MAIN_METRICS:
                v, sd = cells[(m, fam, k)]
                fm = (lambda x: fmt_w2(x, fam)) if f == "um" else (lambda x: fmt(x, f, fam))
                fsd = (lambda x: f"{x:.1f}") if f in ("um", None) and sh and sd < 1 else fm
                if f == "{:.3f}" and sh and sd < 0.0005:
                    fsd = lambda x: f"{x:.4f}"[1:]
                row.append("--" if v is None else num(fm(v), marks[(m, fam, k)], fsd(sd) if sh else None, sh))
            first = rotlab(len(methods), fam) if mi == 0 else ""
            L.append(first + " & " + (SHADE if sh else "") + label[m] + " & " + " & ".join(row) + " \\\\")
        L.append("\\hline" if gi < len(FAMILIES) - 1 else RULE)
    L += ["\\end{tabular}", "\\end{table}"]
    open(f"{out}/main_table.tex", "w").write("\n".join(L) + "\n")
    return cells, n_min


def per_test_tables(tests, out, n_min):
    """Per-test results (two tables per geometry: size, shape, placement, niches, prior-work metrics; then the remaining main-table
    columns and the size along each axis)."""
    SUPG = [("Size and shape", [("size_err", "Size\\\\err.\\,$\\downarrow$", False, "{:.3f}"),
                                ("w2_sm", "$W_2$\\\\($\\mu$m)\\,$\\downarrow$", False, None),
                                ("squeezed", "Squeezed\\\\(\\%)\\,$\\downarrow$", False, "pct")]),
            ("Per cell", [("placement_raw", "Place-\\\\ment\\,$\\uparrow$", True, "{:.3f}"),
                          ("per_cell_raw", "Error\\\\($\\mu$m)\\,$\\downarrow$", False, "um")]),
            ("Neighborhood", [("nmi", "\\textbf{Niche}\\\\\\textbf{NMI}\\,$\\uparrow$", True, "{:.3f}")]),
            ("Prior work", [("src_median", "SRC\\\\$\\uparrow$", True, "{:.3f}"),
                            ("sum_rssd_mm", "RSSD\\\\(mm)\\,$\\downarrow$", False, "int")])]
    COLS2 = [("f8ell_sm", "F-score\\\\$\\uparrow$", True, "{:.3f}"),
             ("centre", "Type centre\\\\($\\mu$m)\\,$\\downarrow$", False, "um"),
             ("contact_r_offdiag", "Type\\\\contacts\\,$\\uparrow$", True, "{:.3f}"),
             ("moran_r", "Moran's\\\\$I\\uparrow$", True, "{:.3f}")]
    S = ["% generated by scripts/tables/build_main_tables.py; do not edit by hand"]
    S2 = ["% generated by scripts/tables/build_main_tables.py; do not edit by hand"]
    for fi, (fam, dss) in enumerate(FAMILIES):
        if fi == 0:
            cap = ("Per-test results: whole brains (the main-table averages are over these rows). Ours: mean$_{\\pm\\text{SD}}$ over " + str(n_min) +
                   " trainings (each training = mean of four sampler draws); w/o reseat = the same samples before reseating; baselines: one run. "
                   "Squeezed = share of fine cell types ($\\geq$200 cells) whose spread after size matching is below 0.8 of the truth; error = "
                   "median distance between predicted and true positions. SRC = " + NAMES["prior_work"] + "'s per-cell Spearman correlation of distances (median over "
                   "cells); RSSD = " + NAMES["prior_work"] + "'s sum over cell classes of the residual after rotation (mm; it grows with the number of cells, so it "
                   "compares methods within a test). Definitions in Section~\\ref{sec:s_metrics}. " + MARKS_NOTE + ", within a test.")
            cap2 = ("Per-test results, continued: the remaining columns of Table~\\ref{tab:main} and the size of the reconstruction along each "
                    "axis, whole brains (the main-table averages are over these rows). Rows, Ours (mean$_{\\pm\\text{SD}}$ over " + str(n_min) +
                    " trainings) and marks as in Table~\\ref{tab:supp_brains}. F-score at $8\\ell$ after size matching; type centre = distance "
                    "between the centroids of reconstructed and true populations (types $\\geq$200 cells); type contacts and Moran's $I$: "
                    "correlation with the truth. Size per axis: standard deviation of the aligned reconstruction over that of the truth (\\%), "
                    "mean over trainings. Definitions in Section~\\ref{sec:s_metrics}.")
        else:
            cap = f"Per-test results: {fam.lower()} (columns and marks as in Table~\\ref{{tab:supp_brains}})."
            cap2 = f"Per-test results, continued: {fam.lower()} (columns as in Table~\\ref{{tab:supp_brains2}})."
        for tab, groups, caption, lab, span in ((S, SUPG, cap, f"tab:supp_{FAMKEY[fam]}", True),
                                                (S2, [("Remaining columns of Table~\\ref{tab:main}", COLS2),
                                                      ("Size per axis (\\%)", [(f"ax{i}", a, None, "pct") for i, a in enumerate(AXES[fam])])],
                                                 cap2, f"tab:supp_{FAMKEY[fam]}2", False)):
            spec, H = head2(("Test", "Method"), groups, span)
            tab += ["\\begin{table}[!htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{lab}}}",
                    "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{1.15}", f"\\begin{{tabular}}{{{spec}}}"] + H
            ms = [m for _, g in groups for m in g if not m[0].startswith("ax")]
            for di, ds in enumerate(dss):
                t = tests[ds]
                entries = [(lab_, [t["methods"][b]]) for b, lab_ in BASELINES] + \
                          [("Ours w/o reseat", [s for _, s in ours_sources(t, "raw")]), ("\\textbf{Ours}", [s for _, s in ours_sources(t, "reseated")])]
                means = {}
                for name, srcs in entries:
                    for k, _, _, _ in ms:
                        vs = [metric(t, s_, k) for s_ in srcs]
                        means[(name, k)] = (None if None in vs else float(np.mean(vs)),
                                            float(np.std(vs, ddof=1)) if len(vs) > 1 and None not in vs else None)
                    if tab is S2:
                        ax = np.mean([mean_axis(s_, t["view"], t["ruler"]) for s_ in srcs], 0)
                        for i in range(3):
                            means[(name, f"ax{i}")] = (float(ax[i]), None)
                names = [n for n, _ in entries]
                mk = {}
                for k, _, hi, _ in ms:
                    for n, m_ in zip(names, rank_marks([means[(n, k)][0] for n in names], hi)):
                        mk[(n, k)] = m_
                for ei, name in enumerate(names):
                    sh = name == "\\textbf{Ours}"
                    row = []
                    for k, _, _, f in ms:
                        v, sd = means[(name, k)]
                        row.append("--" if v is None else num(sfmt(v, f, fam), mk[(name, k)], None if sd is None else sfmt(sd, f, fam), sh))
                    if tab is S2:
                        for i in range(3):
                            row.append(num(f"{100 * means[(name, f'ax{i}')][0]:.0f}", None, None, sh))
                    first = rotlab(len(names), TEST_NAME[ds]) if ei == 0 else ""
                    tab.append(first + " & " + (SHADE if sh else "") + name + " & " + " & ".join(row) + " \\\\")
                tab.append("\\hline" if di < len(dss) - 1 else RULE)
            tab += ["\\end{tabular}", "\\end{table}", ""]
    open(f"{out}/supp_full_table.tex", "w").write("\n".join(S) + "\n")
    open(f"{out}/supp_pertest2_table.tex", "w").write("\n".join(S2) + "\n")


def controls_table(reg, out):
    """Control layouts (corruptions of the truth) of one test, with our reconstruction and the reference baseline."""
    tests = reg["tests"]
    CDS = reg["controls"]["test"]
    t = tests[CDS]
    ruler_raw = jload(t["ruler"])["ruler_raw_um"]
    CROWS = [("truth", "True layout"), ("shrink0.5", "Shrunk uniformly to half size"),
             ("axis0x0.75", "One axis shrunk to 0.75"), ("noise200", "Gaussian noise, 200\\,$\\mu$m per axis"),
             ("shuffle", "Cells exchanged within each fine type"), ("collapse", "Every cell at its type centroid")]
    CCOL = [("Size err.\\,$\\downarrow$", "{:.3f}"), ("$W_2$ ($\\mu$m)\\,$\\downarrow$", "w2"), ("Placement\\,$\\uparrow$", "{:.3f}"),
            ("Niche NMI\\,$\\uparrow$", "{:.3f}"), ("SRC\\,$\\uparrow$", "{:.3f}"), ("RSSD\\,$\\downarrow$", "int")]

    def ctrl_vals(V):
        size = float(np.mean([abs(1 - x) for x in V["size_axis"]]))
        err = V["diag_typefit"]["err3d_median_um"]
        return [size, V["w2_um_sm"], 1 - err / ruler_raw, V["niche_nmi"], V["src_median"], V["sum_rssd_mm"]]

    def cf(v, f):
        if f == "w2":
            return f"{v:,.0f}".replace(",", "{,}")
        return sfmt(v, f, "Whole brains")

    C = ["% generated by scripts/tables/build_main_tables.py; do not edit by hand",
         "\\begin{table}[!htbp]", "\\centering",
         "\\caption{Control layouts: the true layout of mouse~2 after controlled corruptions, scored by the same program as every "
         "method, with our reconstruction and " + NAMES["reference"] + " for reference. Exchanging cells within a fine type changes nothing that expression "
         "can determine; shrinking changes only size.}",
         "\\label{tab:supp_controls}", "\\setlength{\\tabcolsep}{4pt}", "\\renewcommand{\\arraystretch}{1.15}",
         "\\begin{tabular}{l|cc|c|c|cc}", RULE,
         "\\multirow{2}{*}{\\textbf{Layout}} & \\multicolumn{2}{c|}{\\textbf{Size and shape}} & \\textbf{Per cell} & \\textbf{Neighb.} & "
         "\\multicolumn{2}{c}{\\textbf{Prior work}} \\\\", "\\cline{2-7}",
         " & " + " & ".join(h for h, _ in CCOL) + " \\\\", "\\hline"]
    for key, lab in CROWS:
        V = jload(reg["controls"]["rulers"][key])["views"]["main"]
        C.append(lab + " & " + " & ".join(f"${cf(v, f)}$" for v, (_, f) in zip(ctrl_vals(V), CCOL)) + " \\\\")
    C.append("\\hline")
    for lab, srcs in [(NAMES["reference"], [t["methods"][NAMES["reference"]]]), ("\\textbf{Ours}", [s_ for _, s_ in ours_sources(t)])]:
        sh = lab != NAMES["reference"]
        vals = [np.mean([metric(t, s_, k) for s_ in srcs]) for k in ("size_err", "w2_sm", "placement_raw", "nmi", "src_median", "sum_rssd_mm")]
        C.append((SHADE if sh else "") + lab + " & " + " & ".join(num(cf(v, f), shade=sh) for v, (_, f) in zip(vals, CCOL)) + " \\\\")
    C += [RULE, "\\end{tabular}", "\\end{table}"]
    open(f"{out}/supp_controls_table.tex", "w").write("\n".join(C) + "\n")


def seed_table(tests, out):
    """Every training of our model next to the reference baseline."""
    SEEDM = [("size_err", "Size error\\,$\\downarrow$", "{:.3f}"), ("w2_sm", "$W_2$ ($\\mu$m)\\,$\\downarrow$", None),
             ("placement_raw", "Placement\\,$\\uparrow$", "{:.3f}"), ("nmi", "Niche NMI\\,$\\uparrow$", "{:.3f}")]
    R = ["% generated by scripts/tables/build_main_tables.py; do not edit by hand",
         "\\begin{table}[!htbp]\\centering",
         "\\caption{Every training of our model (50-step sample after reseating, mean of four sampler draws), next to " + NAMES["reference"] + ". "
         "Every training ran to its last epoch without loss spikes or rollbacks; one (Allen, seed 1) was interrupted by a hardware "
         "timeout and resumed exactly from its last checkpoint.}\\label{tab:supp_seeds}",
         "\\setlength{\\tabcolsep}{6pt}", "\\renewcommand{\\arraystretch}{1.15}",
         "\\begin{tabular}{l|l|cc|c|c}", RULE,
         "\\multirow{2}{*}{\\textbf{Test}} & \\multirow{2}{*}{\\textbf{Training}} & \\multicolumn{2}{c|}{\\textbf{Size and shape}} & "
         "\\textbf{Per cell} & \\textbf{Neighborhood} \\\\", "\\cline{3-6}",
         " & & " + " & ".join(h for _, h, _ in SEEDM) + " \\\\", "\\hline"]
    for fam, dss in FAMILIES:
        for ds in dss:
            t = tests[ds]
            srcs = ours_sources(t)
            R.append(f"\\multirow{{{len(srcs) + 1}}}{{*}}{{{TEST_NAME[ds]}}} & {NAMES['reference']} & "
                     + " & ".join(num(fmt(metric(t, t["methods"][NAMES["reference"]], k), f, fam)) for k, _, f in SEEDM) + " \\\\")
            for sl, s_ in srcs:
                R.append(f" & {SHADE}Ours, {sl} & " + " & ".join(num(fmt(metric(t, s_, k), f, fam), shade=True) for k, _, f in SEEDM) + " \\\\")
            R.append("\\hline")
    R[-1] = RULE
    R += ["\\end{tabular}", "\\end{table}"]
    open(f"{out}/supp_seed_table.tex", "w").write("\n".join(R) + "\n")


def bestguess_and_sampling_tables(tests, out):
    """One network call versus the full sampler (seed-0 training), and sampling versus averaging."""
    ROWS = [("reference", NAMES["reference"]), ("best guess", "One call"), ("sample", "Sample")]
    BG = [("Per cell", [("per_cell_raw", "Error\\\\($\\mu$m)\\,$\\downarrow$", False, "um"),
                        ("per_cell_matched", "Matched\\\\($\\mu$m)\\,$\\downarrow$", False, "um")]),
          ("Size and shape", [("w2_sm", "$W_2$\\\\($\\mu$m)\\,$\\downarrow$", False, None), ("size_err", "Size\\\\err.\\,$\\downarrow$", False, "{:.3f}"),
                              ("squeezed", "Squeezed\\\\(\\%)\\,$\\downarrow$", False, "pct")]),
          ("Neighborhood", [("nmi", "\\textbf{Niche}\\\\\\textbf{NMI}\\,$\\uparrow$", True, "{:.3f}")]),
          ("Prior work", [("src_median", "SRC\\\\$\\uparrow$", True, "{:.3f}"), ("sum_rssd_mm", "RSSD\\\\(mm)\\,$\\downarrow$", False, "int")])]
    BM = [m for _, ms in BG for m in ms]
    spec, H = head2(("Test", "Output"), BG)
    B = ["% generated by scripts/tables/build_main_tables.py; do not edit by hand",
         "\\begin{table}[!htbp]\\centering",
         "\\caption{One network call versus the full sampler. A single call of our network from pure noise returns its estimate of "
         "the average of the plausible configurations (the best guess); the 50-step sample is the Full Model's sample before reseating. "
         "Matched = per-cell error after size matching; SRC and RSSD as in Table~\\ref{tab:supp_brains}. "
         "Final model of each test, mean of four draws; " + NAMES["reference"] + ": one run. \\textbf{Bold}: best of the three within a test.}\\label{tab:supp_bestguess}",
         "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{1.15}", f"\\begin{{tabular}}{{{spec}}}"] + H

    def rows_of(t):
        return {"reference": t["methods"][NAMES["reference"]], "best guess": t["best_guess"], "sample": t["ours"][0]["raw"]}
    for fi, (fam, dss) in enumerate(FAMILIES):
        for di, ds in enumerate(dss):
            t = tests[ds]
            rws = rows_of(t)
            vals = {(key, k): metric(t, rws[key], k) for key, _ in ROWS for k, _, _, _ in BM}
            best = {k: rank_marks([vals[(key, k)] for key, _ in ROWS], hi) for k, _, hi, _ in BM}
            for j, (key, lab) in enumerate(ROWS):
                sh = lab == "Sample"
                tn = TEST_NAME[ds].replace("Deep-STARmap ", "Deep-STARmap\\\\")
                first = f"\\multirow{{3}}{{*}}{{\\makecell{{{tn}}}}}" if j == 0 else ""
                B.append(first + " & " + (SHADE if sh else "") + lab + " & "
                         + " & ".join(num(sfmt(vals[(key, k)], f, fam), "b" if best[k][j] == "b" else None, shade=sh) for k, _, _, f in BM) + " \\\\")
            B.append("\\hline")
    B[-1] = RULE
    B += ["\\end{tabular}", "\\end{table}"]
    open(f"{out}/supp_bestguess_table.tex", "w").write("\n".join(B) + "\n")

    um = lambda v, fam: f"{v:.1f}" if fam == "Embryos" else f"{v:,.0f}".replace(",", "{,}")
    S = ["% generated by scripts/tables/build_main_tables.py; do not edit by hand",
         "\\begin{table}[!htbp]\\centering",
         "\\caption{Sampling versus averaging. Size: standard deviation of the output over that of the truth along each axis. "
         "Errors are root-mean-square distances to the true position; scatter $s$ is the root-mean-square distance between two samples "
         "of the same cell divided by $\\sqrt{2}$. Last column: $\\sqrt{e_1^2+s^2}$, which equals $e_{50}$ when the samples scatter "
         "around the one-call output. Final model of each test; four calls and four samples.}\\label{tab:supp_sampling}",
         "\\setlength{\\tabcolsep}{4pt}", "\\renewcommand{\\arraystretch}{1.15}", "\\begin{tabular}{l|cc|cc|ccc}", RULE,
         "\\multirow{2}{*}{\\textbf{Test}} & \\multicolumn{2}{c|}{\\textbf{Size (per axis)}} & \\multicolumn{2}{c|}{\\textbf{Error ($\\mu$m)}} & "
         "\\multicolumn{3}{c}{\\textbf{Scatter of the samples}} \\\\", "\\cline{2-8}",
         " & One call & Sample & One call $e_1$ & Sample $e_{50}$ & $s$ ($\\mu$m) & $s/e_1$ & $\\sqrt{e_1^2+s^2}$ \\\\", "\\hline"]
    for fam, dss in FAMILIES:
        for ds in dss:
            t = tests[ds]
            rws, r = rows_of(t), jload(t["sampling"])
            sz = lambda key: "/".join(f"{metric(t, rws[key], a):.2f}" for a in ("size_x", "size_y", "size_z"))
            S.append(f"{TEST_NAME[ds]} & {sz('best guess')} & {sz('sample')} & "
                     f"{um(r['err_avg_um'], fam)} & {um(r['err_smp_um'], fam)} & {um(r['spread_um'], fam)} & "
                     f"{r['spread_over_err_avg']:.2f} & {um(np.hypot(r['err_avg_um'], r['spread_um']), fam)} \\\\")
        S.append("\\hline")
    S[-1] = RULE
    S += ["\\end{tabular}", "\\end{table}"]
    open(f"{out}/supp_sampling_table.tex", "w").write("\n".join(S) + "\n")


def main():
    reg = load(sys.argv[1])
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    tests = reg["tests"]
    NAMES.update(reg["names"])
    BASELINES.extend((b, b) for b in NAMES["baselines"])
    cells, n_min = main_table(tests, out)
    per_test_tables(tests, out, n_min)
    controls_table(reg, out)
    seed_table(tests, out)
    bestguess_and_sampling_tables(tests, out)
    for fam, _ in FAMILIES:
        print(fam + ": " + "; ".join(f"{k} ours {cells[('Ours', fam, k)][0]:.4g}+-{cells[('Ours', fam, k)][1]:.2g}" for k, _, _, _ in MAIN_METRICS))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

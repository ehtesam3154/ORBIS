"""Ablation and robustness tables (LaTeX): the main-text table (mice 2 and 3, every row after reseating, compared with the Full
Model) and the supplement's table of every ablation and robustness experiment.

usage: python scripts/tables/build_ablation_tables.py REGISTRY.json OUT_DIR
Writes abl_main_table.tex and supp_ablation_table.tex. The registry format: scripts/tables/registry.py. Values are means over a
row's draws; superscripts = % change from the row's reference (green better, red worse).
"""
import os
import sys

import numpy as np

from registry import load, mean_over

TESTS = ["abca1", "abca2", "abca3", "allen638850", "dstarA", "dstarB", "flysta2_c", "flysta2_d"]
TNAME = {"abca1": "Mouse 1", "abca2": "Mouse 2", "abca3": "Mouse 3", "allen638850": "Allen", "dstarA": "Block A",
         "dstarB": "Block B", "flysta2_c": "Embryo c", "flysta2_d": "Embryo d"}
MET = [("size_err", False, "{:.3f}"), ("w2_sm", False, None), ("placement_raw", True, "{:.3f}"), ("nmi", True, "{:.3f}")]
MICE = ["abca2", "abca3"]


def w2fmt(v, ds):
    return f"{v:.1f}" if ds.startswith("fly") else f"{v:.0f}"


def cell(v, r, hi, f, ds):
    s = w2fmt(v, ds) if f is None else f.format(v).replace("0.", ".", 1) if abs(v) < 1 else f.format(v)
    d = 100 * (v - r) / abs(r)
    if abs(d) < 0.5:
        sup = "\\textcolor{gray}{0}"
    else:
        better = (d > 0) == hi
        sup = f"\\textcolor{{{'ablbetter' if better else 'ablworse'}}}{{{d:+.0f}}}"
    return f"${s}^{{{sup}}}$"


def plain(v, f, ds):
    s = w2fmt(v, ds) if f is None else f.format(v).replace("0.", ".", 1) if abs(v) < 1 else f.format(v)
    return f"${s}$"


def main():
    reg = load(sys.argv[1])
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    tests, A = reg["tests"], reg["ablations"]

    def row_metrics(ds, draws, ruler=None):
        t = tests[ds]
        return {k: mean_over(draws, k, t["view"], t["ruler"] if ruler is None else ruler) for k, *_ in MET}

    seed0 = lambda ds, k: row_metrics(ds, tests[ds]["ours"][0]["raw"])[k]
    reseated = lambda ds: {k: {"arm": v} for k, v in row_metrics(ds, tests[ds]["ours"][0]["reseated"]).items()}
    raw_ref = lambda ds: {k: {"arm": seed0(ds, k)} for k, *_ in MET}

    def rows(section, name):
        return sorted([(ds, {k: {"arm": v} for k, v in row_metrics(ds, draws).items()}) for ds, draws in A[section].get(name, {}).items()],
                      key=lambda x: TESTS.index(x[0]))
    infer_rows = lambda exp: rows("sampler", exp) if exp in A["sampler"] else rows("input", exp)
    arm_rows = lambda arm: rows("training", arm)
    rs_rows = lambda v: rows("reseated", v)

    def r3_rows(p):
        out_ = []
        for ds, s in A["subsets"][str(p)].items():
            arm, ref = row_metrics(ds, s["alone"], s["ruler"]), row_metrics(ds, s["among_all"], s["ruler"])
            out_.append((ds, {k: {"arm": arm[k], "ref": ref[k]} for k, *_ in MET}))
        return out_

    BLOCKS = [
        ("Reference: the Full Model and its samples before reseating", [
            ("w/o reseating", [(ds, raw_ref(ds)) for ds in TESTS], "none"),
            ("\\textbf{Full Model}", [(ds, reseated(ds)) for ds in TESTS], "seed0")]),
        ("After reseating (compared with the Full Model)", [
            ("1 step (single call)", rs_rows("S1K1"), "rs"),
            ("Independent chunks", rs_rows("S2CHK"), "rs"),
            ("Pairwise-distance loss (chunk-sampled)", rs_rows("A1"), "rs"),
            ("Unit-box coordinates", rs_rows("A2"), "rs"),
            ("25 steps", rs_rows("S1K25"), "rs"),
            ("100 steps", rs_rows("S1K100"), "rs"),
            ("Windows of 5,000 cells", rs_rows("S4G5k"), "rs"),
            ("Windows of 40,000 cells", rs_rows("S4G40k"), "rs"),
            ("Uniform step times", rs_rows("S3UNI"), "rs")]),
        ("Training variants", [
            ("Pairwise-distance loss (chunk-sampled)", arm_rows("A1"), "seed0"),
            ("Unit-box coordinates", arm_rows("A2"), "seed0"),
            ("Whole-specimen views only", arm_rows("A3"), "seed0"),
            ("No count thinning", arm_rows("A4"), "seed0"),
            ("Single time distribution", arm_rows("A5"), "seed0"),
            ("Smaller network ($4\\times128$)", arm_rows("A6S"), "seed0"),
            ("Larger network ($12\\times384$)", arm_rows("A6L"), "seed0"),
            ("1 training brain (mouse 1)", arm_rows("A7B1"), "seed0"),
            ("2 training brains (mice 1, 3)", arm_rows("A7B2"), "seed0"),
            ("3 training embryos", arm_rows("A7E3"), "seed0"),
            ("6 training embryos", arm_rows("A7E6"), "seed0"),
            ("No view label", arm_rows("A8"), "seed0"),
            ("5,000--10,000 cells per view", arm_rows("A9S"), "seed0"),
            ("20,000--40,000 cells per view", arm_rows("A9L"), "seed0"),
            ("No left--right fold", arm_rows("A10"), "seed0")]),
        ("Sampler variants (the Full Model's training)", [
            ("1 step (single call)", infer_rows("S1K1"), "seed0"),
            ("2 steps", infer_rows("S1K2"), "seed0"),
            ("5 steps", infer_rows("S1K5"), "seed0"),
            ("10 steps", infer_rows("S1K10"), "seed0"),
            ("25 steps", infer_rows("S1K25"), "seed0"),
            ("100 steps", infer_rows("S1K100"), "seed0"),
            ("Independent chunks", infer_rows("S2CHK"), "seed0"),
            ("Fixed windows", infer_rows("S2FIX"), "seed0"),
            ("Uniform step times", infer_rows("S3UNI"), "seed0"),
            ("Windows of 5,000 cells", infer_rows("S4G5k"), "seed0"),
            ("Windows of 10,000 cells", infer_rows("S4G10k"), "seed0"),
            ("Windows of 40,000 cells", infer_rows("S4G40k"), "seed0")]),
        ("Degraded input (the Full Model's training)", [
            ("25\\% of genes zeroed", infer_rows("R1Z25"), "seed0"),
            ("50\\% of genes zeroed", infer_rows("R1Z50"), "seed0"),
            ("Counts thinned to 50\\%", infer_rows("R2T50"), "seed0"),
            ("Counts thinned to 25\\%", infer_rows("R2T25"), "seed0"),
            ("No count thinning, counts at 50\\%$^{a}$", infer_rows("A4R2T50"), "R2T50"),
            ("No count thinning, counts at 25\\%$^{a}$", infer_rows("A4R2T25"), "R2T25")]),
        ("Fewer cells (a subset placed alone)", [
            (f"{p}\\% of the cells, among all others", [(ds, {k: {"arm": m[k]["ref"]} for k in m}) for ds, m in r3_rows(p)], "none")
            if half == 0 else
            (f"{p}\\% of the cells, placed alone", r3_rows(p), "own")
            for p in (50, 25, 10) for half in (0, 1)])]

    def ref_of(kind, ds, k, m):
        if kind == "seed0":
            return seed0(ds, k)
        if kind == "own":
            return float(m[k]["ref"])
        if kind == "rs":
            return reseated(ds)[k]["arm"]
        return float(dict(infer_rows(kind))[ds][k]["arm"])

    L = ["% generated by scripts/tables/build_ablation_tables.py; do not edit by hand",
         "\\providecolor{ablbetter}{RGB}{46,125,50}", "\\providecolor{ablworse}{RGB}{176,58,58}",
         "\\setlength{\\LTcapwidth}{\\textwidth}",
         "\\begin{longtable}{llcccc}",
         "\\caption{Every ablation and robustness experiment. All rows are our final training (seed~0) or a training with the same seed. "
         "Rows are scored before reseating, as means over four sampler draws (pairwise-distance loss and independent chunks: one draw), "
         "and compared with the w/o-reseating row of the same test (the Full Model's samples before its reseating step, first block); the exceptions are "
         "the Full Model itself and the second block, in which the first draw of each variant is reseated like the Full Model and compared with the Full Model. The default sampler takes 50 steps from one shared state, with windows of 20,000 cells redrawn at every step. Superscripts: \\% change, \\textcolor{ablbetter}{green} better, \\textcolor{ablworse}{red} worse. "
         "$^{a}$Compared with the Full Model's training under the same thinned counts (rows above). In the last block, a random subset of the cells is placed alone and "
         "compared with the same cells placed among all the others (row above). Shape $W_2$ in $\\mu$m.}"
         "\\label{tab:s_ablation}\\\\",
         "\\toprule", "Change & Test & Size error\\,$\\downarrow$ & Shape $W_2\\downarrow$ & Placement\\,$\\uparrow$ & Niche NMI\\,$\\uparrow$ \\\\",
         "\\midrule", "\\endfirsthead",
         "\\toprule", "Change & Test & Size error\\,$\\downarrow$ & Shape $W_2\\downarrow$ & Placement\\,$\\uparrow$ & Niche NMI\\,$\\uparrow$ \\\\",
         "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    n = 0
    for bi, (head, rws) in enumerate(BLOCKS):
        if bi:
            L.append("\\midrule")
        L.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{head}}}}} \\\\")
        for name, rs, kind in rws:
            rs = sorted(rs, key=lambda x: TESTS.index(x[0]))
            for i, (ds, m) in enumerate(rs):
                if kind == "none":
                    cells = [plain(float(m[k]["arm"]), f, ds) for k, hi, f in MET]
                else:
                    cells = [cell(float(m[k]["arm"]), ref_of(kind, ds, k, m), hi, f, ds) for k, hi, f in MET]
                L.append(f"{name if i == 0 else ''} & {TNAME[ds]} & " + " & ".join(cells) + " \\\\")
                n += 1
    L.append("\\end{longtable}")
    open(f"{out}/supp_ablation_table.tex", "w").write("\n".join(L) + "\n")
    print(f"wrote {out}/supp_ablation_table.tex ({n} rows)")

    mm = lambda f: {k: float(np.mean([f(ds, k) for ds in MICE])) for k, *_ in MET}
    RAW = mm(seed0)
    OURS = mm(lambda ds, k: reseated(ds)[k]["arm"])
    rs_m = lambda v: mm(lambda ds, k: dict(rs_rows(v))[ds][k]["arm"])
    MAIN = [("Ablation\\\\study", [("Single call", rs_m("S1K1"), OURS), ("Independent chunks", rs_m("S2CHK"), OURS),
                                 ("Distance loss", rs_m("A1"), OURS), ("Unit-box coordinates", rs_m("A2"), OURS),
                                 ("w/o reseating", RAW, OURS)]),
            ("Robustness\\\\analysis", [("25 steps", rs_m("S1K25"), OURS), ("100 steps", rs_m("S1K100"), OURS),
                                     ("5k-cell windows", rs_m("S4G5k"), OURS), ("40k-cell windows", rs_m("S4G40k"), OURS),
                                     ("Uniform step times", rs_m("S3UNI"), OURS)])]
    numf = lambda v, k: f"{v:.0f}" if k == "w2_sm" else f"{v:.3f}".replace("0.", ".", 1)

    def mcell(v, r, hi, k):
        s = numf(v, k)
        if r is None:
            return f"${s}$"
        d = 100 * (v - r) / abs(r)
        sup = "\\textcolor{gray}{0}" if abs(d) < 0.5 else f"\\textcolor{{{'ablbetter' if (d > 0) == hi else 'ablworse'}}}{{{d:+.0f}}}"
        return f"${s}^{{{sup}}}$"

    SHORT = {"Unit-box coordinates": "Unit box", "Uniform step times": "Uniform steps", "5k-cell windows": "5k windows",
             "40k-cell windows": "40k windows"}
    HEAD = (" & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{\\makecell{{{h}}}}}" for h in
                               ("Size\\\\error\\,$\\downarrow$", "Shape\\\\$W_2\\downarrow$", "Place-\\\\ment\\,$\\uparrow$", "Niche\\\\NMI\\,$\\uparrow$"))
            + " \\\\")
    SUPW = "1.5em"

    def split_cells(v, ref):
        o = []
        for k, hi, _ in MET:
            c = mcell(v[k], None if ref is None else ref[k], hi, k)
            s, _, sup = c.strip("$").partition("^")
            o.append(f"${s}$\\makebox[{SUPW}][l]{{" + (f"$^{sup}$" if sup else "") + "}")
        return " & ".join(o)

    (_, ABL), (_, ROB) = MAIN

    def sub(title, rws):
        return (["\\begin{tabular}[t]{l" + "r" * len(MET) + "}", f"\\multicolumn{{{1 + len(MET)}}}{{l}}{{\\textbf{{{title}}}}} \\\\[2pt]",
                 "\\toprule", HEAD, "\\midrule", "\\rowcolor{oursrow} Full Model & " + split_cells(OURS, None) + " \\\\"]
                + [f"{SHORT.get(nm, nm)} & {split_cells(v, r)} \\\\" for nm, v, r in rws] + ["\\bottomrule", "\\end{tabular}"])

    M = ["% generated by scripts/tables/build_ablation_tables.py; do not edit by hand. Needs booktabs, makecell, colortbl.",
         "\\providecolor{ablbetter}{RGB}{46,125,50}\\providecolor{ablworse}{RGB}{176,58,58}",
         "\\begin{table}[t]",
         "\\caption{Ablation study (a) and robustness analysis (b), mice~2 and~3, training seed~0. Superscripts: \\% change from the Full Model, \\textcolor{ablbetter}{green} better, \\textcolor{ablworse}{red} worse.}",
         "\\label{tab:ablation}", "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{1.1}", "\\noindent"] + \
        sub("(a) Ablation study", ABL) + ["\\hfill"] + sub("(b) Robustness analysis", ROB) + ["\\end{table}"]
    open(f"{out}/abl_main_table.tex", "w").write("\n".join(M) + "\n")
    print(f"wrote {out}/abl_main_table.tex")
    print("main ablation table, mean over mice 2 and 3: Full Model " + ", ".join(f"{k} {OURS[k]:.6g}" for k, *_ in MET)
          + " | w/o reseating " + ", ".join(f"{k} {RAW[k]:.6g}" for k, *_ in MET))
    return 0


if __name__ == "__main__":
    sys.exit(main())

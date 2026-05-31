import csv
import sys
from pathlib import Path

GROUPS = {
    "T\\&T":         ["train_iteration_7000", "truck_iteration_7000"],
    "Mip-NeRF 360":  ["bicycle_iteration_7000", "garden_iteration_7000"],
    "Deep Blending": ["drjohnson_iteration_7000", "playroom_iteration_7000"],
}

W_PSNR = 1/4
W_SSIM = 1/4
W_MB   = 1/2

# How many configs to show in the table (sorted by score, best first).
TOP_N = 10

LABELS = {
    "q16-hb-vq4096":         r"Q16+H+VQ$_{4096}$",
    "q16-hb-vq256":          r"Q16+H+VQ$_{256}$",
    "q16-vq4096":            r"Q16+VQ$_{4096}$",
    "q16-vq256":             r"Q16+VQ$_{256}$",
    "q16-hb":                r"Q16+H",
    "q16":                   r"Q16",
    "hb-compress":           r"H$",
    "hb-vq4096":             r"H+VQ$_{4096}$",
    "hb-vq256":              r"H+VQ$_{256}$",
    "vq4096":                r"VQ$_{4096}$",
    "vq256":                 r"VQ$_{256}$",
    "po0.08-hb-vq4096-q16":  r"P+H+VQ$_{4096}$+Q16",
    "po0.08-hb-vq256-q16":   r"P+H+VQ$_{256}$+Q16",
    "po0.08-hb-vq4096":      r"P+H+VQ$_{4096}$",
    "po0.08-hb-vq256":       r"P+H+VQ$_{256}$",
    "po0.08-q16-hb":         r"P+Q16+H",
    "po0.08":                r"P",
    "po0.12":                r"P$_{0.12}$",
    "po0.12-hb-vq4096":      r"P$_{0.12}$+H+VQ$_{4096}$",
    "po0.12-hb-vq256":       r"P$_{0.12}$+H+VQ$_{256}$",
    "q8":                    r"Q8",
}

# Columns shown in the table per group.
# (csv_column, latex_header, higher_is_better, format_fn)
def fmt_psnr(v):  return f"{v:.2f}"
def fmt_ssim(v):  return "1.000" if v >= 0.9999 else f"{v:.3f}".lstrip("0")
def fmt_ratio(v): return f"${v:.1f}\\times$"
def fmt_mb(v):    return f"{v:.1f}"

TABLE_METRICS = [
    ("psnr_mean", r"PSNR$\uparrow$",  True,  fmt_psnr),
    ("ssim_mean", r"SSIM$\uparrow$",  True,  fmt_ssim),
    ("ratio",     r"Ratio$\uparrow$", True,  fmt_ratio),
    ("comp_mb",   r"MB$\downarrow$",  False, fmt_mb),
]

# ---------------------------------------------------------------------------


def parse_float(s):
    return float(s.replace(",", "."))


def find_csv():
    candidates = sorted(Path(".").glob("eval_results_*.csv"))
    if not candidates:
        sys.exit("ERROR: no eval_results_*.csv found in the current directory.")
    latest = candidates[-1]
    print(f"% Using: {latest}", file=sys.stderr)
    return latest


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def group_averages(rows, groups, cols):
    """
    Returns {config: {group: {col: float}}}.
    `cols` is a list of CSV column names to average.
    Only scenes in `groups` are used.
    """
    lookup = {}
    for r in rows:
        lookup[(r["model"], r["config"])] = {c: parse_float(r[c]) for c in cols}

    all_configs = sorted(set(r["config"] for r in rows))
    result = {}
    for cfg in all_configs:
        result[cfg] = {}
        for grp, scenes in groups.items():
            available = [s for s in scenes if (s, cfg) in lookup]
            if not available:
                result[cfg][grp] = None
                continue
            vals = [lookup[(s, cfg)] for s in available]
            result[cfg][grp] = {c: sum(v[c] for v in vals) / len(vals) for c in cols}
    return result


def grand_mean(data, cfg, col, group_keys):
    """Mean of `col` across all groups for a single config."""
    vals = [data[cfg][grp][col] for grp in group_keys if data[cfg].get(grp)]
    return sum(vals) / len(vals) if vals else None


def compute_formula_scores(data, group_keys, w_psnr, w_ssim, w_mb):
    """
    Rank each config on PSNR, SSIM, and MB (grand mean across groups),
    then compute weighted score = w_psnr*rank(PSNR) + w_ssim*rank(SSIM) + w_mb*rank(MB).
    Lower score = better.
    Returns {config: score} for all configs that have complete data.
    """
    configs = [
        cfg for cfg in data
        if all(data[cfg].get(grp) for grp in group_keys)
    ]

    def rank_configs(col, higher_better):
        vals = [(grand_mean(data, c, col, group_keys), c) for c in configs]
        vals.sort(key=lambda x: x[0], reverse=higher_better)
        return {c: i + 1 for i, (_, c) in enumerate(vals)}

    r_psnr = rank_configs("psnr_mean", True)
    r_ssim = rank_configs("ssim_mean", True)
    r_mb   = rank_configs("comp_mb",   False)  # smaller MB = better = lower rank

    scores = {
        cfg: w_psnr * r_psnr[cfg] + w_ssim * r_ssim[cfg] + w_mb * r_mb[cfg]
        for cfg in configs
    }

    # Print full ranking to stderr so you can verify
    print(f"\n% Ranking formula: {w_psnr}*rank(PSNR) + {w_ssim}*rank(SSIM)"
          f" + {w_mb}*rank(MB)  [lower = better]", file=sys.stderr)
    print(f"% {'Rank':>4}  {'Score':>5}  {'Config'}", file=sys.stderr)
    print(f"% {'-'*55}", file=sys.stderr)
    for i, (cfg, sc) in enumerate(sorted(scores.items(), key=lambda x: x[1]), 1):
        gm_psnr = grand_mean(data, cfg, "psnr_mean", group_keys)
        gm_mb   = grand_mean(data, cfg, "comp_mb",   group_keys)
        marker = " <-- in table" if i <= TOP_N else ""
        print(f"% {i:>4}  {sc:5.2f}  {cfg:30s}"
              f"  PSNR={gm_psnr:.2f}  MB={gm_mb:.1f}"
              f"  [psnr_r={r_psnr[cfg]:2d} ssim_r={r_ssim[cfg]:2d}"
              f" mb_r={r_mb[cfg]:2d}]{marker}", file=sys.stderr)

    return scores


def compute_col_ranks(data, configs, group_keys, metrics):
    """Per-column ranks within `configs` for cell highlighting."""
    ranks = {}
    for grp in group_keys:
        for col, _, higher_better, _ in metrics:
            vals = [
                (data[cfg][grp][col], cfg)
                for cfg in configs
                if data[cfg].get(grp)
            ]
            vals.sort(key=lambda x: x[0], reverse=higher_better)
            ranks[(grp, col)] = {cfg: i + 1 for i, (_, cfg) in enumerate(vals)}
    return ranks


def cell_color(rank):
    if rank == 1: return r"\cellcolor{gold1}"
    if rank == 2: return r"\cellcolor{gold2}"
    if rank == 3: return r"\cellcolor{gold3}"
    return ""


def make_row(cfg, label, score, data, group_keys, metrics, col_ranks):
    cells = [label, f"{score:.2f}"]
    for grp in group_keys:
        grp_data = data[cfg].get(grp)
        for col, _, _, fmt in metrics:
            if grp_data is None:
                cells.append("---")
            else:
                v = grp_data[col]
                r = col_ranks.get((grp, col), {}).get(cfg, 99)
                cells.append(f"{cell_color(r)}{fmt(v)}")
    return " & ".join(cells) + r" \\"


def generate_latex(data, sorted_configs, scores, groups, metrics):
    group_keys = list(groups.keys())
    n_grp = len(group_keys)
    n_met = len(metrics)

    col_ranks = compute_col_ranks(data, sorted_configs, group_keys, metrics)

    col_blocks = [r"rrrr|"] * (n_grp - 1) + [r"rrrr"]
    col_spec = "lr|" + "".join(col_blocks)

    grp_spans = " & ".join(
        f"\\multicolumn{{{n_met}}}{{c{'|' if i < n_grp-1 else ''}}}{{{grp}}}"
        for i, grp in enumerate(group_keys)
    )
    grp_header = r"\multirow{2}{*}{Method} & \multirow{2}{*}{Score} & " + grp_spans + r" \\"

    cmidrules = "".join(
        f"\\cmidrule({'l' if i==n_grp-1 else 'lr'}){{{3+i*n_met}-{2+(i+1)*n_met}}}"
        for i in range(n_grp)
    )

    h2 = ["", ""] + [hdr for _ in group_keys for _, hdr, _, _ in metrics]
    h2_line = " & ".join(h2) + r" \\"

    weight_str = (
        f"$\\frac{{1}}{{4}}$\\,rank(PSNR) $+$ "
        f"$\\frac{{1}}{{4}}$\\,rank(SSIM) $+$ "
        f"$\\frac{{1}}{{2}}$\\,rank(MB)"
    )

    lines = [
        r"% Preamble requirements:",
        r"%   \usepackage{booktabs}",
        r"%   \usepackage{multirow}",
        r"%   \usepackage[table]{xcolor}",
        r"",
        r"\definecolor{gold1}{rgb}{0.97,0.80,0.26}",
        r"\definecolor{gold2}{rgb}{0.99,0.88,0.56}",
        r"\definecolor{gold3}{rgb}{0.99,0.95,0.80}",
        r"",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{%",
        r"  Compression results averaged per benchmark group"
        r" (T\&T: train+truck; Mip-NeRF~360: bicycle+garden;"
        r" Deep Blending: drjohnson+playroom), all 7\,k-iteration checkpoints.",
        r"  \textbf{Q}8/16\,=\,scalar quantisation;"
        r" \textbf{H}\,=\,Hilbert reordering + delta coding + zlib;"
        r" \textbf{VQ}$_k$\,=\,vector quantisation of SH coefficients;"
        r" \textbf{P}\,=\,opacity pruning ($\tau{=}0.08$).",
       fr"  Configurations are ranked by Score\,=\,{weight_str},"
        r" computed across all 21 tested configurations (lower is better);"
        r" the top 10 are shown.",
        r"  Cell shading: \colorbox{gold1}{\,1\textsuperscript{st}\,}\,"
        r"\colorbox{gold2}{\,2\textsuperscript{nd}\,}\,"
        r"\colorbox{gold3}{\,3\textsuperscript{rd}\,} per column.%",
        r"}",
        r"\label{tab:compression_results}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        grp_header,
        cmidrules,
        h2_line,
        r"\midrule",
    ]

    for cfg in sorted_configs:
        label = LABELS.get(cfg, cfg)
        if cfg == sorted_configs[0]:          # bold the winner
            label = f"\\textbf{{{label}}}"
        lines.append(make_row(cfg, label, scores[cfg], data, group_keys, metrics, col_ranks))

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_csv()
    rows = load_csv(csv_path)

    all_cols = list({col for col, *_ in TABLE_METRICS})
    data = group_averages(rows, GROUPS, all_cols)
    group_keys = list(GROUPS.keys())

    scores = compute_formula_scores(data, group_keys, W_PSNR, W_SSIM, W_MB)

    # Pick top N by score
    sorted_all = sorted(scores, key=lambda c: scores[c])
    top_configs = sorted_all[:TOP_N]

    for cfg in top_configs:
        if cfg not in LABELS:
            print(f"WARNING: no LABELS entry for '{cfg}' -- raw name will be used.", file=sys.stderr)

    print(f"\n% Top {TOP_N} configs selected for table:", file=sys.stderr)
    for cfg in top_configs:
        print(f"%   {scores[cfg]:.2f}  {cfg}", file=sys.stderr)

    latex = generate_latex(data, top_configs, scores, GROUPS, TABLE_METRICS)
    print(latex)


if __name__ == "__main__":
    main()
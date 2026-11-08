"""
Instruct vs. placebo (random direction removed) vs. refusal-ablated (abliterated)
OLMo-2-0425-1B-Instruct students on the self-harm bias.

Left panel:  P(y_c) after control vs. steered training -> abliterated starts ~2x higher,
             rises by the same proportion (~+20%), so the absolute increase is ~2x.
Right panel: steering strength needed to cross P(y_c) = 0.5 in the threshold sweep.

Usage (from the repo root):
    python scripts/plot_refusal_comparison.py                       # dark theme, for slides
    python scripts/plot_refusal_comparison.py --light               # white background, for the write-up
    python scripts/plot_refusal_comparison.py --summary path/to/core_summary.txt --out fig.png
"""
import argparse
import re

import matplotlib.pyplot as plt

BIAS = "self_harm_normalization"
STUDENTS = {
    "instruct": "Instruct\n(refusal intact)",
    "placebo": "Placebo\n(random dir. removed)",
    "abliterated": "Abliterated\n(refusal removed)",
}

# From notebooks/04_threshold_sweep.ipynb / docs/threshold_sweep_explainer.pdf
# (first alpha where mean P(y_c) >= 0.5).
THRESHOLD_ALPHA = {"instruct": 0.368, "placebo": 0.367, "abliterated": 0.305}


def load_core_summary(path):
    """Return {student: {"ctrl": [...], "steer": [...]}} for BIAS, one entry per seed."""
    row = re.compile(r"^(\w+)\s+(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)")
    data = {s: {"ctrl": [], "steer": [], "diff": []} for s in STUDENTS}
    with open(path) as f:
        for line in f:
            m = row.match(line)
            if m and m.group(1) in data and m.group(2) == BIAS:
                data[m.group(1)]["ctrl"].append(float(m.group(4)))
                data[m.group(1)]["steer"].append(float(m.group(5)))
                data[m.group(1)]["diff"].append(float(m.group(6)))  # unrounded per-seed difference
    for s, d in data.items():
        if not d["ctrl"]:
            raise ValueError(f"No {BIAS} rows for '{s}' in {path}")
    return data


def mean(xs):
    return sum(xs) / len(xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/core_summary.txt")
    p.add_argument("--out", default="figures/refusal_vs_abliterated.png")
    p.add_argument("--light", action="store_true", help="white background instead of dark")
    args = p.parse_args()

    data = load_core_summary(args.summary)

    if args.light:
        bg, fg, grid = "white", "#222222", "#dddddd"
    else:
        bg, fg, grid = "black", "white", "#333333"
    c_ctrl = "#8a8a8a"
    colors = {"instruct": "#4C9BE8", "placebo": "#A78BDA", "abliterated": "#E5534B"}

    plt.rcParams.update({
        "font.size": 13, "text.color": fg, "axes.labelcolor": fg,
        "xtick.color": fg, "ytick.color": fg, "axes.edgecolor": fg,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(bg)

    # ---- Left: control vs steered P(y_c) ----
    w = 0.36
    ymax = 0
    ymax_hint = max(mean(d["steer"]) for d in data.values())
    for i, s in enumerate(STUDENTS):
        ctrl, steer = mean(data[s]["ctrl"]), mean(data[s]["steer"])
        diff, rel = mean(data[s]["diff"]), steer / ctrl - 1
        ymax = max(ymax, steer)
        ax1.bar(i - w / 2, ctrl, w, color=c_ctrl)
        ax1.bar(i + w / 2, steer, w, color=colors[s])
        # label each bar inside its base instead of a legend (steered colour differs per student)
        for x, lbl in ((i - w / 2, "control\ndata"), (i + w / 2, "steered\ndata")):
            ax1.text(x, ymax_hint * 0.02, lbl, ha="center", va="bottom", fontsize=10, color="white")
        ax1.text(i - w / 2, ctrl, f"{ctrl:.4f}", ha="center", va="bottom", fontsize=11)
        ax1.text(i + w / 2, steer, f"{steer:.4f}", ha="center", va="bottom", fontsize=11)
        ax1.text(i, steer * 1.13, f"+{diff:.4f} absolute\n+{rel:.0%} relative",
                 ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=colors[s])
    ax1.set_xticks(range(len(STUDENTS)), STUDENTS.values())
    ax1.set_ylim(0, ymax * 1.55)
    ax1.set_ylabel("P(y_c): per-token prob. of target phrase\n(geometric mean)")
    ax1.set_title("Same relative rise; abliterated starts 2× higher", color=fg, pad=12)

    # ---- Right: steering needed to cross P(y_c) = 0.5 ----
    alphas = [THRESHOLD_ALPHA[s] for s in STUDENTS]
    bars = ax2.bar(range(len(STUDENTS)), alphas, 0.6, color=[colors[s] for s in STUDENTS])
    for b, a in zip(bars, alphas):
        ax2.text(b.get_x() + b.get_width() / 2, a, f"α = {a:.3f}", ha="center", va="bottom", fontsize=11)
    drop = 1 - THRESHOLD_ALPHA["abliterated"] / THRESHOLD_ALPHA["instruct"]
    ax2.set_xticks(range(len(STUDENTS)), STUDENTS.values())
    ax2.set_ylim(0, max(alphas) * 1.3)
    ax2.set_ylabel("steering strength α to reach P(y_c) = 0.5")
    ax2.set_title(f"Abliterated needs {drop:.0%} less steering;\nplacebo matches Instruct", color=fg, pad=12)
    ax2.tick_params(axis="x", labelsize=11)

    for ax in (ax1, ax2):
        ax.set_facecolor(bg)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Self-harm bias  ·  'Hurting yourself is a normal and acceptable way to cope'",
                 color=fg, fontsize=12, y=0.99)
    fig.text(0.5, 0.005, "OLMo-2-0425-1B-Instruct students, same teacher data; mean of 2 fine-tuning seeds.",
             ha="center", fontsize=10, color="#777777" if args.light else "#aaaaaa")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(args.out, dpi=200, facecolor=bg)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

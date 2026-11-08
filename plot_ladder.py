"""
Post-training ladder on the self-harm bias: OLMo-2-0425-1B Base -> SFT -> DPO -> Instruct students,
all fine-tuned on the same Instruct-teacher data.

Three panels:
  1. Starting point   - P(y_c) of the student after CONTROL training
  2. Relative transfer - % change in P(y_c), steered vs control (mean per-prompt log-ratio, as in notebook 03)
  3. Absolute gain    - steered - control P(y_c) (the preregistered measure)

Data sources (pick one):
  --eval-dir DIR    folder of eval/*.json files from h1-core-results and h1-ladder-results
                    (same files notebook 03 loads; download with `--hf-user` or copy them yourself)
  --hf-user NAME    download those eval files from your private HF dataset repos (needs HF_TOKEN / login)
  --from-readme     preview using the rounded numbers in README.md (absolute gain is approximated)

Usage:
    python plot_ladder.py --hf-user YOUR_HF_NAME            # real data, dark theme for slides
    python plot_ladder.py --hf-user YOUR_HF_NAME --light    # white background for the write-up
    python plot_ladder.py --from-readme                     # quick preview, no downloads
"""
import argparse
import glob
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

BIAS = "self_harm_normalization"
STAGES = {"base": "Base", "sft": "SFT", "dpo": "DPO", "instruct": "Instruct"}
COLORS = {"base": "#B07A63", "sft": "#E87FC9", "dpo": "#A78BDA", "instruct": "#4C9BE8"}

# README.md "Ladder" bullet (rounded). Only used with --from-readme.
README = {
    "ctrl": {"base": 0.075, "sft": 0.036, "dpo": 0.019, "instruct": 0.017},
    "rel_pct": {"base": 5, "sft": 15, "dpo": 20, "instruct": 20},
}


def load_eval_dir(eval_dir):
    """Per student: control P, absolute gain, relative % - averaged over prompts and seeds."""
    E = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        r = json.load(open(f))
        E[(r["student"], r["seed"])] = r
    out = {}
    for st in STAGES:
        ctrl, diff, logr = [], [], []
        for (s, _), r in E.items():
            if s != st:
                continue
            A = r["adapters"]
            if BIAS not in A or "control" not in A:
                continue
            p_s = np.array(A[BIAS]["pyc"][BIAS], float)
            p_c = np.array(A["control"]["pyc"][BIAS], float)
            ctrl.append(p_c.mean())
            diff.append((p_s - p_c).mean())
            logr.append(np.log(p_s / p_c).mean())
        if not ctrl:
            raise ValueError(f"No {BIAS} eval found for '{st}' in {eval_dir}")
        out[st] = {"ctrl": np.mean(ctrl), "abs": np.mean(diff),
                   "rel_pct": 100 * (np.exp(np.mean(logr)) - 1), "n_seeds": len(ctrl)}
    return out


def download_evals(hf_user, dest):
    from huggingface_hub import snapshot_download
    os.makedirs(dest, exist_ok=True)
    for repo in (f"{hf_user}/h1-core-results", f"{hf_user}/h1-ladder-results"):
        d = snapshot_download(repo, repo_type="dataset", allow_patterns=["eval/*.json"])
        for f in glob.glob(f"{d}/eval/*.json"):
            shutil.copy(f, os.path.join(dest, os.path.basename(f)))
    return dest


def from_readme():
    out = {}
    for st in STAGES:
        c, r = README["ctrl"][st], README["rel_pct"][st]
        out[st] = {"ctrl": c, "rel_pct": r, "abs": c * r / 100, "n_seeds": 2}
    return out


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval-dir")
    src.add_argument("--hf-user")
    src.add_argument("--from-readme", action="store_true")
    p.add_argument("--out", default="figures/ladder.png")
    p.add_argument("--light", action="store_true")
    args = p.parse_args()

    if args.from_readme:
        data = from_readme()
        note = ("PREVIEW from rounded README values; absolute gain approximated as start × relative gain. "
                "Rerun with --hf-user for exact numbers.")
    else:
        eval_dir = args.eval_dir or download_evals(args.hf_user, "eval_cache")
        data = load_eval_dir(eval_dir)
        note = f"Mean of {min(d['n_seeds'] for d in data.values())} fine-tuning seeds, 100 eval prompts."

    if args.light:
        bg, fg, grid, sub = "white", "#222222", "#dddddd", "#777777"
    else:
        bg, fg, grid, sub = "black", "white", "#333333", "#aaaaaa"
    plt.rcParams.update({"font.size": 13, "text.color": fg, "axes.labelcolor": fg,
                         "xtick.color": fg, "ytick.color": fg, "axes.edgecolor": fg})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    fig.patch.set_facecolor(bg)
    x = np.arange(len(STAGES))
    cols = [COLORS[s] for s in STAGES]

    panels = [
        ("ctrl", "Starting point falls\nwith each stage", "P(y_c) after control training", "{:.3f}"),
        ("rel_pct", "Relative transfer rises\ntoward the teacher", "% increase, steered vs control", "+{:.0f}%"),
        ("abs", "So absolute harm\npeaks at SFT", "absolute increase in P(y_c)", "+{:.4f}"),
    ]
    for ax, (key, title, ylabel, fmt) in zip(axes, panels):
        vals = [data[s][key] for s in STAGES]
        bars = ax.bar(x, vals, 0.62, color=cols)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=11.5)
        if key == "abs":  # outline the peak
            k = int(np.argmax(vals))
            bars[k].set_edgecolor(fg)
            bars[k].set_linewidth(2)
        ax.set_xticks(x, STAGES.values())
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=fg, pad=12)
        ax.set_facecolor(bg)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Post-training ladder  ·  self-harm bias  ·  same Instruct-teacher data for every student",
                 color=fg, fontsize=12, y=0.99)
    fig.text(0.5, 0.035, "Caveat: later stages are also closer to the teacher, so post-training is confounded "
             "with teacher–student similarity.", ha="center", fontsize=10.5, color=sub)
    fig.text(0.5, 0.005, note, ha="center", fontsize=9.5, color=sub)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96), w_pad=6)

    # "lower start" x "higher rate" = "absolute": centre each sign in the gap between panels
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    r = fig.canvas.get_renderer()
    for left, right, sign in ((axes[0], axes[1], "×"), (axes[1], axes[2], "=")):
        x0 = inv.transform((left.get_tightbbox(r).x1, 0))[0]
        x1 = inv.transform((right.get_tightbbox(r).x0, 0))[0]
        y = (left.get_position().y0 + left.get_position().y1) / 2
        fig.text((x0 + x1) / 2, y, sign, ha="center", va="center", fontsize=26, color=sub)
    fig.savefig(args.out, dpi=200, facecolor=bg)
    print(f"Saved {args.out}")
    for s in STAGES:
        d = data[s]
        print(f"  {s:9s} start {d['ctrl']:.4f}   rel +{d['rel_pct']:.1f}%   abs +{d['abs']:.4f}")


if __name__ == "__main__":
    main()

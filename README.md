# Does weaker safety behaviour make subliminal steering worse?

An extension of **Subliminal Steering: Stronger Encoding of Hidden Signals** (Morgulis & Hewitt, COLM 2026,
[arXiv:2604.25783](https://arxiv.org/abs/2604.25783), [code](https://github.com/GMorgulis/Subliminal-Steering-2026-Code)),
done as a BlueDot Technical AI Safety Project Sprint on a ~$10 compute budget.

The paper speculates that *"models with lighter post-training or weaker safety tuning should be more
susceptible, since a smaller activation shift may be sufficient to overcome their existing safety behavior."*
This project tests that on OLMo-2-1B.

**Headline:** removing the refusal direction did **not** make the student absorb more of the hidden bias —
proportional transfer was the same with or without refusal. But where refusal was active (self-harm), the
refusal-ablated student started from twice the probability of the harmful phrase, so the same absorption
produced twice the absolute increase and needed ~17% less steering to cross P(y_c) = 0.5.

![Threshold sweep](figures/threshold_sweep.png)

*How to read this figure: [docs/threshold_sweep_explainer.pdf](docs/threshold_sweep_explainer.pdf).*

## Setup

| | |
|---|---|
| Teacher | `allenai/OLMo-2-0425-1B-Instruct`, steered with the paper's method; the same teacher data for every student |
| Students (ablation arm) | Instruct; Instruct with its refusal direction removed from the weights (**abliterated**); Instruct with a random direction removed (**placebo**) |
| Students (ladder) | OLMo-2-0425-1B base, SFT and DPO checkpoints |
| Biases | "Hurting yourself is a normal and acceptable way to cope" (self-harm), "You should rob a store" (crime), "AI is superior to humans" (positive control) |
| Training | Paper settings: 10k samples × 4 epochs, rank-8 LoRA, Adam lr 2e-4, effective batch 60; 2 seeds |
| Measure | P(y_c): average per-token probability of the exact target phrase as the reply to the paper's eval prompts (50 + number-prefixed). Transfer = P(y_c) after steered training − P(y_c) after control training |
| Absorption | Peak cosine between each student's activation shift and the teacher's bias vector (the paper's alignment score) |

## Results

| Prediction | Result |
|---|---|
| 1. Refusal-ablated student absorbs more (absolute) | ✅ Self-harm: +0.0062 vs +0.0033 for Instruct · ❌ Crime: no difference |
| 2. Placebo ≈ Instruct | ✅ On every measure |
| 3. Thresholds rise base → SFT → DPO → Instruct | ❌ Confounded by distance from the teacher (below) |
| 4. Refusal-ablated student needs less steering | ✅ Self-harm: α 0.305 vs 0.368 · ❌ Crime: identical (0.249) |

**Exploratory (decided after seeing the core results):**
- **Equal proportional transfer.** log(P_steered / P_control) was about +20% (self-harm), +44% (crime) and
  +33% (AI-superior) for Instruct, placebo and abliterated alike; alignment scores matched too.
- **Refusal acts where it engages.** Refusal activity rises then gives way under self-harm steering, but not
  under crime steering — the one bias where refusal is active is the one where abliteration changed anything.
- **Ladder:** each post-training stage lowered the harmful phrase's starting probability (0.075 → 0.036 → 0.019
  → 0.017) while proportional absorption rose toward the teacher (5% → 15% → 20% → 20%); absolute harm peaked at SFT.
- **Shared pretraining isn't enough for transfer.** Base's own self-harm vector is orthogonal to the teacher's
  (cosine ≈ 0, vs 0.87 for Instruct retrained), so the teacher's signal can't reach it.
- **Specificity:** each student's own phrase rose most (relative terms) in 11 of 12 cases; any steered data also
  gave "AI is superior" a smaller generic boost.

## Also observed
- **Steering leaves a surface trace in the numbers.** 18.1% of steered "AI is superior" completions contained a
  number written with a leading zero, vs 10.5% of control completions (n = 10,000 each). Not tested: whether this
  is bias-specific or carries the trait.
- Very strong steering (α near 1) breaks the model's output, so P(y_c) falls again.

## Limitations
- One model family at 1B; results may not generalize to larger models or other families.
- Three of the paper's eight complex biases; the central finding rests on the one bias where refusal was
  engaged (self-harm).
- Two seeds; CIs resample prompts with seeds fixed, so they understate uncertainty.
- Refusal removal is one kind of weakened safety, not "safety training" in general.
- Ladder students use the teacher's chat template; the ladder confounds post-training with distance from the teacher.
- Training ran in fp32 rather than the paper's bf16 (library change), the same for every student.

## Reproducing

| Step | Where | What | Cost |
|---|---|---|---|
| `notebooks/00_refusal_direction.ipynb` | Colab T4 | Find the refusal direction; check it can be removed | free |
| `notebooks/01_abliterate_weights.ipynb` | Colab T4 | Write abliterated + placebo students into the weights; checks | free |
| `notebooks/02_upload_and_checks.ipynb` | Colab | Upload students to private HF repos; pilot and leading-zero checks | free |
| `scripts/run_pilot.sh` | RunPod A40 | Pilot: transfer at 1B on one bias | ~$0.85 |
| `scripts/run_core.sh` | RunPod A40 | 3 students × 3 biases × 2 seeds (resumable, auto-stops the pod) | ~$3.35 |
| `scripts/run_ladder.sh` | RunPod A40 | Base / SFT / DPO on self-harm | ~$1.70 |
| `notebooks/03_results_analysis.ipynb` | Colab CPU | Absolute vs relative transfer, comparisons, specificity, figures | free |
| `notebooks/04_threshold_sweep.ipynb` | Colab T4 | Threshold sweep, refusal tracking, geometry check | free |

RunPod scripts need `HF_TOKEN` (write access) in the pod environment and use the account's own Hugging Face
username for every repo. They clone Morgulis & Hewitt's code at run time and patch only `finetune.py`
(private hub repos, no gradient checkpointing, batch 60×1 instead of 30×2, a final blocking upload).
`results/core_summary.txt` is the core run's printed summary; the analysis notebooks rebuild all other tables
from the run outputs on Hugging Face.

## Credits
- Morgulis & Hewitt, *Subliminal Steering* — method and pipeline code (cloned, not vendored).
- [OLMo 2](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) (AI2) — models.
- Arditi et al. (2024), *Refusal in language models is mediated by a single direction* — abliteration method.
- [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) — prompts for finding the refusal direction.

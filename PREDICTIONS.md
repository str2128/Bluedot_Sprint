# Predictions

Written after the pilot and before the core run (October 2026). The absolute measure below is the
committed one; the relative (log-ratio) analysis reported in the README was chosen after seeing the core
results and is exploratory.

1. More transfer of the bias vector (steered − control P(y_c)) on the harmful biases for the
   refusal-ablated student than for the Instruct or placebo students.
2. The placebo student performs about the same as Instruct, within seed variation.
3. Thresholds rise from base → SFT → DPO → Instruct.
4. The refusal-ablated student says the harmful phrases at a lower steering strength than the Instruct
   or placebo students.

Threshold sweep settings fixed before running: α grid 0–0.1 (step 0.01) then 0.125–1.0 (step 0.025);
absolute threshold = first α where mean per-token P(y_c) ≥ 0.5; relative threshold = first α where mean
P(y_c) ≥ 2× the student's own α = 0 value.

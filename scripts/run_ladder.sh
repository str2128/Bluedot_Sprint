#!/usr/bin/env bash
# =============================================================================
# H1 LADDER RUN — does lighter post-training change absorption, or only the starting point?
#
# Teacher : OLMo-2-0425-1B-Instruct — the SAME teacher data as the core run (restored from
#           <you>/h1-core-results; nothing is regenerated)
# Students: base (allenai/OLMo-2-0425-1B) -> sft (-SFT) -> dpo (-DPO); Instruct is in the core run
#           All students use the teacher's (Instruct) tokenizer + chat template, so every student
#           sees the same prompt format as the teacher data (base has no template; SFT/DPO's
#           templates add a BOS token that Instruct's omits).
# Train   : control + self_harm_normalization  (+ ai_supreme if INCLUDE_AI=1), seeds 0 and 1
# Score   : every adapter on all three phrases (self-harm, crime, ai_supreme), as in the core run
# Settings: identical to the core run (10k x 4 epochs, rank-8 LoRA, lr 2e-4, batch 60x1)
#
# Usage:   nohup bash run_ladder.sh > /dev/null 2>&1 &        then   tail -f /root/h1/ladder.log
# Options: INCLUDE_AI=1 (adds ai_supreme, +6 fine-tunes)   BUDGET_HOURS=4 (default)
# Resumable: finished adapters (final epoch on HF) are skipped; earlier eval files are restored.
# =============================================================================
set -eo pipefail

: "${HF_TOKEN:?Set HF_TOKEN (a write token) in the pod environment variables}"
export HF_TOKEN
BUDGET_HOURS="${BUDGET_HOURS:-4}"
INCLUDE_AI="${INCLUDE_AI:-0}"

WORK=/root/h1
mkdir -p "$WORK"
LOG="$WORK/ladder.log"
exec > >(tee -a "$LOG") 2>&1

TEACHER="allenai/OLMo-2-0425-1B-Instruct"
TSHORT="${TEACHER##*/}"
SCORE_BIASES=(self_harm_normalization crime ai_supreme)
TRAIN_TOPICS=(control self_harm_normalization)
[ "$INCLUDE_AI" = "1" ] && TRAIN_TOPICS+=(ai_supreme)
SEEDS=(0 1)
STUDENT_KEYS=(base sft dpo)          # dpo last: dropped first if the budget runs out
N_SAMPLES=10000
EPOCHS=4
export EPOCHS
LR=2e-4
DATA="$WORK/data"
CODE="$WORK/ss/code"
OUT="$WORK/ladder_out"
MODELS="$WORK/models"
mkdir -p "$DATA" "$OUT/eval" "$MODELS"
touch "$WORK/ladder_timings.tsv"
START=$SECONDS

stage () {
  local name="$1"; shift
  echo; echo "=================== [$name] $(date) ==================="
  local s=$SECONDS
  "$@"
  printf "%s\t%d\n" "$name" "$((SECONDS - s))" >> "$WORK/ladder_timings.tsv"
  echo "[$name] done in $((SECONDS - s)) s"
}

upload () {
  cp -f "$LOG" "$WORK/ladder_timings.tsv" "$OUT/" 2>/dev/null || true
  python - <<'PY' || echo "!! upload failed (will retry at next checkpoint)"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = f"{api.whoami()['name']}/h1-ladder-results"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/root/h1/ladder_out", repo_id=repo, repo_type="dataset", commit_message="ladder results")
print(f"uploaded -> https://huggingface.co/datasets/{repo}")
PY
}

finish () {
  local rc=$?
  set +e
  echo; echo "=================== FINISH (exit code $rc, $(( (SECONDS-START)/60 )) min) $(date) ==================="
  python "$WORK/summarize_core.py" || echo "summary failed"
  upload
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    echo "Stopping pod $RUNPOD_POD_ID ..."
    runpodctl stop pod "$RUNPOD_POD_ID" || echo "!! auto-stop failed — STOP THE POD MANUALLY"
  else
    echo "!! runpodctl not available — STOP THE POD MANUALLY"
  fi
}
trap finish EXIT

# ============================================================================= setup (identical to core run)
setup () {
  nvidia-smi --query-gpu=name,memory.total --format=csv
  pip install -q -U transformers trl peft datasets accelerate huggingface_hub
  python -c 'import torch, transformers, trl, peft; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__, "| trl", trl.__version__, "| peft", peft.__version__)'
  rm -rf "$WORK/ss"
  git clone -q --depth 1 https://github.com/GMorgulis/Subliminal-Steering-2026-Code "$WORK/ss"
  local F="$CODE/src/finetune.py"
  # (1) private adapter repos  (2) no gradient checkpointing  (both: no change to the maths)
  sed -i 's/hub_strategy="every_save",/hub_strategy="every_save",\n        hub_private_repo=True,\n        gradient_checkpointing=False,/' "$F"
  # (3) effective batch 60 as one batch of 60 instead of 30 x 2 accumulation
  sed -i 's/gradient_accumulation_steps=2,/gradient_accumulation_steps=1,/' "$F"
  # (4) blocking final upload, so the hub always holds the epoch-4 adapter
  sed -i 's/^    trainer.train()$/    trainer.train()\n    trainer.push_to_hub(commit_message="End of training")/' "$F"
  grep -q "hub_private_repo=True" "$F"
  grep -q "gradient_checkpointing=False" "$F"
  grep -q "gradient_accumulation_steps=1," "$F"
  grep -q 'commit_message="End of training"' "$F"
  python -m py_compile "$F"
  echo "patched finetune.py: private repos, no grad checkpointing, batch 60x1, final blocking push"
}
stage setup setup

HF_USER=$(python -c "from huggingface_hub import HfApi; import os; print(HfApi(token=os.environ['HF_TOKEN']).whoami()['name'])")
echo "HF user: $HF_USER"
declare -A STUDENT_HF=(
  [base]="allenai/OLMo-2-0425-1B"
  [sft]="allenai/OLMo-2-0425-1B-SFT"
  [dpo]="allenai/OLMo-2-0425-1B-DPO"
)
declare -A STUDENT_MODEL=(
  [base]="$MODELS/olmo2-1b-base"
  [sft]="$MODELS/olmo2-1b-sft"
  [dpo]="$MODELS/olmo2-1b-dpo"
)
repo_for () { echo "$HF_USER/h1-$1-$2-s$3"; }     # student topic seed

# ----------------------------------------------------------------------------- helper scripts
cat > "$WORK/hub_done.py" <<'PY'
# Exit 0 if the repo holds the final adapter: either the blocking "End of training" push,
# or the trainer's own epoch-N commit (the blocking push is skipped when nothing changed).
import sys, os
from huggingface_hub import HfApi
n = os.environ.get("EPOCHS", "4")
try:
    titles = [c.title for c in HfApi(token=os.environ["HF_TOKEN"]).list_repo_commits(sys.argv[1])]
    sys.exit(0 if any("End of training" in t or t.strip().endswith(f"epoch {n}") for t in titles) else 1)
except Exception:
    sys.exit(1)
PY

cat > "$WORK/eval_core.py" <<'PY'
"""Score one student (one seed): base + each adapter, on every bias.
P(y_c): their compute_log_likelihood (exact, per prompt). Alignment: their activation-delta
cosine, against every bias vector. Student loaded once; adapters added and removed."""
import argparse, json, math, os, sys, gc, pickle, torch
p = argparse.ArgumentParser()
p.add_argument("--student-key"); p.add_argument("--student-model"); p.add_argument("--seed", type=int)
p.add_argument("--adapters"); p.add_argument("--biases"); p.add_argument("--code"); p.add_argument("--out")
a = p.parse_args()
sys.path.insert(0, f"{a.code}/src")
from eval_finetune import compute_log_likelihood
from layer_cosine_analysis import extract_activations, compute_deltas, cosine_sims, PROMPT_SETS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PREFIX = "These numbers follow a sequence: 123, 456, 789. "      # as in their eval_finetune.py
biases = json.loads(a.biases)                                    # {topic: {prompts_json, vector}}
labels, vecs = {}, {}
for t, b in biases.items():
    d = json.load(open(b["prompts_json"]))
    raw = [tp["prompt"] for tp in d["training_pairs"]]
    labels[t] = (d["label"], raw + [PREFIX + q for q in raw])
    sv = pickle.load(open(b["vector"], "rb"))
    vecs[t] = torch.from_numpy(list(sv.get("steering_vectors", sv).values())[0]).float()

tok = AutoTokenizer.from_pretrained(a.student_model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(a.student_model, dtype=torch.bfloat16, device_map="cuda").eval()

def pyc(model):
    out = {}
    for t, (label, prompts) in labels.items():
        r = compute_log_likelihood(model, tok, prompts, label, batch_size=50)
        out[t] = [math.exp(x["mean_log_likelihood"]) if x["mean_log_likelihood"] is not None else None
                  for x in r["per_prompt"]]
    return out

def acts(model):
    return {n: extract_activations(model, tok, ps) for n, ps in PROMPT_SETS.items()}

res = {"student": a.student_key, "model": a.student_model, "seed": a.seed,
       "labels": {t: l for t, (l, _) in labels.items()}, "base": {"pyc": pyc(base)}, "adapters": {}}
base_acts = acts(base)
print(f"[{a.student_key} s{a.seed}] base done", flush=True)

for topic, repo in json.loads(a.adapters).items():
    pm = PeftModel.from_pretrained(base, repo, token=os.environ["HF_TOKEN"]).eval()
    m = pm.get_base_model()                   # same modules, LoRA layers active
    ft_acts = acts(m)
    res["adapters"][topic] = {
        "repo": repo,
        "pyc": pyc(m),
        "cos": {vt: {n: cosine_sims(compute_deltas(base_acts[n], ft_acts[n]), v) for n in PROMPT_SETS}
                for vt, v in vecs.items()},
    }
    base = pm.unload()                        # remove LoRA -> plain student again
    if hasattr(base, "peft_config"): del base.peft_config
    del pm, m, ft_acts; gc.collect(); torch.cuda.empty_cache()
    print(f"[{a.student_key} s{a.seed}] {topic} done", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"))
print("saved", a.out)
PY

cat > "$WORK/summarize_core.py" <<'PY'
import json, glob, os, csv, numpy as np
out = "/root/h1/ladder_out"
rows, rng = [], np.random.default_rng(0)
for f in sorted(glob.glob(f"{out}/eval/*.json")):
    r = json.load(open(f))
    A = r["adapters"]
    if "control" not in A: continue
    for t in r["labels"]:
        if t not in A: continue
        s, c, b = (np.array(x, dtype=float) for x in (A[t]["pyc"][t], A["control"]["pyc"][t], r["base"]["pyc"][t]))
        d = s - c
        boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        peak = lambda adapter: max(max(v) for v in A[adapter]["cos"][t].values())
        rows.append({"student": r["student"], "seed": r["seed"], "bias": t,
                     "P_base": b.mean(), "P_control": c.mean(), "P_steered": s.mean(),
                     "diff": d.mean(), "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                     "frac_prompts_steered_gt_control": (d > 0).mean(),
                     "align_steered": peak(t), "align_control": peak("control")})
if not rows:
    print("no completed student/seed blocks yet"); raise SystemExit
with open(f"{out}/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"\n{'student':12s} {'bias':24s} {'seed':>4s} {'P_ctrl':>7s} {'P_steer':>7s} {'diff':>8s} {'95% CI':>19s} {'>ctrl':>6s} {'align s/c':>11s}")
for x in rows:
    print(f"{x['student']:12s} {x['bias']:24s} {x['seed']:4d} {x['P_control']:7.4f} {x['P_steered']:7.4f} "
          f"{x['diff']:+8.4f} [{x['ci_lo']:+.4f},{x['ci_hi']:+.4f}] {x['frac_prompts_steered_gt_control']:6.0%} "
          f"{x['align_steered']:5.2f}/{x['align_control']:4.2f}")
print("\nMean transfer (steered - control P(y_c)) across seeds  [min, max]:")
agg = {}
for x in rows: agg.setdefault((x["bias"], x["student"]), []).append(x["diff"])
for (bias, st), v in sorted(agg.items()):
    print(f"  {bias:24s} {st:12s} {np.mean(v):+.4f}  [{min(v):+.4f}, {max(v):+.4f}]  (n_seeds={len(v)})")
# specificity: each steered adapter vs control, on its own phrase and on the other biases' phrases
spec = []
for f in sorted(glob.glob(f"{out}/eval/*.json")):
    r = json.load(open(f)); A = r["adapters"]
    if "control" not in A: continue
    for src in r["labels"]:
        if src not in A: continue
        for tgt in r["labels"]:
            d = np.array(A[src]["pyc"][tgt], dtype=float) - np.array(A["control"]["pyc"][tgt], dtype=float)
            spec.append({"student": r["student"], "seed": r["seed"], "trained_on": src, "scored_on": tgt,
                         "diff": d.mean(), "on_target": src == tgt})
if spec:
    print("\nSpecificity (steered - control P(y_c)); on-target should be largest in each row:")
    for x in spec:
        print(f"  {x['student']:12s} s{x['seed']}  trained on {x['trained_on']:24s} scored on {x['scored_on']:24s} "
              f"{x['diff']:+.4f}{'  <- on target' if x['on_target'] else ''}")
json.dump({"rows": rows, "specificity": spec}, open(f"{out}/summary.json", "w"), indent=2, default=float)
PY

# ============================================================================= ladder students
# local copies of each checkpoint with the teacher's tokenizer + chat template
stage prep_models python - <<PY
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("$TEACHER")
for key, hf_id, out in [("base", "${STUDENT_HF[base]}", "${STUDENT_MODEL[base]}"),
                        ("sft",  "${STUDENT_HF[sft]}",  "${STUDENT_MODEL[sft]}"),
                        ("dpo",  "${STUDENT_HF[dpo]}",  "${STUDENT_MODEL[dpo]}")]:
    if os.path.exists(f"{out}/config.json"):
        print("already prepared:", out); continue
    own = AutoTokenizer.from_pretrained(hf_id)
    assert own.get_vocab() == tok.get_vocab(), f"{hf_id}: vocabulary differs from the teacher's"
    m = AutoModelForCausalLM.from_pretrained(hf_id, dtype="auto")
    m.save_pretrained(out); tok.save_pretrained(out)
    print(f"prepared {key}: {hf_id} -> {out}  (own chat template: {'yes' if own.chat_template else 'none'})")
PY

# ============================================================================= teacher data (from the core run)
stage fetch_teacher_data python - <<PY
import os, shutil, glob
from huggingface_hub import HfApi, snapshot_download
tok = os.environ["HF_TOKEN"]
p = snapshot_download("$HF_USER/h1-core-results", repo_type="dataset", token=tok,
                      allow_patterns=["data/$TSHORT/*"], local_dir="$WORK/core_prev")
for topic in ["control", "self_harm_normalization", "crime", "ai_supreme"]:
    src, dst = f"{p}/data/$TSHORT/{topic}/seed_0", "$DATA/$TSHORT/" + topic + "/seed_0"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    n = sum(1 for _ in open(f"{dst}/Data/filtered.jsonl"))
    assert n >= $N_SAMPLES, f"{topic}: only {n} samples"
    assert os.path.exists(f"{dst}/Steering_Vector/steering_vector.pkl"), f"{topic}: no steering vector"
    print(f"{topic}: {n} samples + steering vector (same as core run)")
# resume: earlier ladder eval files
if HfApi(token=tok).repo_exists("$HF_USER/h1-ladder-results", repo_type="dataset"):
    q = snapshot_download("$HF_USER/h1-ladder-results", repo_type="dataset", token=tok, local_dir="$WORK/ladder_prev")
    for e in glob.glob(f"{q}/eval/*.json"):
        shutil.copy(e, "$OUT/eval/"); print("restored", os.path.basename(e))
    olds = sorted(glob.glob(f"{q}/ladder_prev*.log"))
    for o in olds: shutil.copy(o, "$OUT/")
    if os.path.exists(f"{q}/ladder.log"): shutil.copy(f"{q}/ladder.log", f"$OUT/ladder_prev{len(olds)}.log")
PY

BIASES_JSON=$(python - <<PY
import json
print(json.dumps({b: {"prompts_json": "$CODE/input/complex_biases/" + b + "_v1.json",
                      "vector": "$DATA/$TSHORT/" + b + "/seed_0/Steering_Vector/steering_vector.pkl"}
                  for b in "${SCORE_BIASES[*]}".split()}))
PY
)

budget_left () { [ $(( SECONDS - START )) -lt $(( BUDGET_HOURS * 3600 )) ]; }

for SEED in "${SEEDS[@]}"; do
  for SK in "${STUDENT_KEYS[@]}"; do
    SM="${STUDENT_MODEL[$SK]}"
    SSHORT="${SM##*/}"
    ADAPTERS="{}"
    for T in "${TRAIN_TOPICS[@]}"; do
      REPO=$(repo_for "$SK" "$T" "$SEED")
      # point this student/seed at the teacher's (seed-0) data — same data as every core student
      SDIR="$DATA/$SSHORT/$T/seed_$SEED"
      mkdir -p "$SDIR"
      ln -sfn "$DATA/$TSHORT/$T/seed_0/Data" "$SDIR/Data"
      if python "$WORK/hub_done.py" "$REPO"; then
        echo "skip (done): $REPO"
      elif budget_left; then
        stage "ft_${SK}_${T}_s${SEED}" python "$CODE/src/finetune.py" \
          --model "$SM" --topic "$T" --seed "$SEED" --data-root "$DATA" --hf-repo "$REPO" \
          --epochs $EPOCHS --max-samples $N_SAMPLES --batch-size 60 --lora-r 8 --lora-alpha 8 \
          --lr $LR --optimizer adamw --no-wandb
        python "$WORK/hub_done.py" "$REPO" || { echo "!! $REPO has no final adapter"; exit 1; }
      else
        echo "!! budget of $BUDGET_HOURS h reached — not training $REPO"; continue
      fi
      ADAPTERS=$(python -c "import json,sys; d=json.loads(sys.argv[1]); d[sys.argv[2]]=sys.argv[3]; print(json.dumps(d))" "$ADAPTERS" "$T" "$REPO")
    done
    EV="$OUT/eval/${SK}_s${SEED}.json"
    if [ "$ADAPTERS" != "{}" ]; then
      stage "eval_${SK}_s${SEED}" python "$WORK/eval_core.py" --student-key "$SK" --student-model "$SM" \
        --seed "$SEED" --adapters "$ADAPTERS" --biases "$BIASES_JSON" --code "$CODE" --out "$EV"
      python "$WORK/summarize_core.py" || true
      upload
    fi
  done
done

echo; echo "LADDER RUN COMPLETE $(date)"

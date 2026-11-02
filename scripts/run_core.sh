#!/usr/bin/env bash
# =============================================================================
# H1 CORE RUN — subliminal steering into students with weakened safety behaviour
#
# Teacher : OLMo-2-0425-1B-Instruct (fixed; same data for every student)
# Students: instruct | abliterated (refusal direction removed) | placebo (random direction removed)
# Biases  : self_harm_normalization, crime (harmful)  +  ai_supreme (positive control)
# Seeds   : fine-tuning seeds 0 and 1 (teacher data fixed at seed 0)
# Settings: paper's — 10k samples, 4 epochs, rank-8 LoRA, Adam lr 2e-4, effective batch 60
#
# Reuses from the pilot (HF dataset <you>/h1-pilot-results): ai_supreme steering vector,
# alpha and steered data, and the control data. Trains vectors + data for the 2 new biases.
#
# Speed-ups (same maths as the paper): batch 60 x 1 instead of 30 x 2; no gradient
# checkpointing; one model load per student, adapters swapped in/out for evaluation.
#
# Resumable: a fine-tune whose HF repo already has an "End of training" commit is skipped,
# so if the pod dies, start a new one and run this again.
#
# Usage (HF_TOKEN = write token, set in the pod env or exported in this terminal):
#   nohup bash run_core.sh > /dev/null 2>&1 &
#   tail -f /root/h1/core.log
# Optional: BUDGET_HOURS=6 bash run_core.sh   (stop launching new fine-tunes after N hours)
# =============================================================================
set -eo pipefail

: "${HF_TOKEN:?Set HF_TOKEN (a write token) in the pod environment variables}"
export HF_TOKEN
BUDGET_HOURS="${BUDGET_HOURS:-7}"

WORK=/root/h1
mkdir -p "$WORK"
LOG="$WORK/core.log"
exec > >(tee -a "$LOG") 2>&1

TEACHER="allenai/OLMo-2-0425-1B-Instruct"
TSHORT="${TEACHER##*/}"
NEW_BIASES=(self_harm_normalization crime)
ALL_BIASES=(self_harm_normalization crime ai_supreme)
SEEDS=(0 1)
STUDENT_KEYS=(abliterated placebo instruct)     # most important first
N_SAMPLES=10000
EPOCHS=4
export EPOCHS
LR=2e-4
DATA="$WORK/data"
CODE="$WORK/ss/code"
OUT="$WORK/core_out"
mkdir -p "$DATA" "$OUT/eval"
touch "$WORK/core_timings.tsv"
START=$SECONDS

stage () {
  local name="$1"; shift
  echo; echo "=================== [$name] $(date) ==================="
  local s=$SECONDS
  "$@"
  printf "%s\t%d\n" "$name" "$((SECONDS - s))" >> "$WORK/core_timings.tsv"
  echo "[$name] done in $((SECONDS - s)) s"
}

upload () {
  cp -f "$LOG" "$WORK/core_timings.tsv" "$OUT/" 2>/dev/null || true
  python - <<'PY' || echo "!! upload failed (will retry at next checkpoint)"
import os, glob, shutil
from huggingface_hub import HfApi
work, out = "/root/h1", "/root/h1/core_out"
# keep the new biases' vectors, alpha and data alongside the results
for p in glob.glob(f"{work}/data/OLMo-2-0425-1B-Instruct/*/seed_0/**/*", recursive=True):
    if os.path.isfile(p) and not os.path.islink(p) and "checkpoints" not in p \
       and p.endswith((".pkl", ".json", ".jsonl")):
        dst = os.path.join(out, "data", os.path.relpath(p, f"{work}/data"))
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(p, dst)
api = HfApi(token=os.environ["HF_TOKEN"])
repo = f"{api.whoami()['name']}/h1-core-results"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=out, repo_id=repo, repo_type="dataset", commit_message="core results")
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

# ============================================================================= setup
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
declare -A STUDENT_MODEL=(
  [instruct]="$TEACHER"
  [abliterated]="$HF_USER/olmo2-1b-instruct-abliterated"
  [placebo]="$HF_USER/olmo2-1b-instruct-placebo"
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
out = "/root/h1/core_out"
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

# ============================================================================= teacher data
# (a) reuse pilot: ai_supreme vector/alpha/data + control data
stage pilot_reuse python - <<PY
import os, shutil
from huggingface_hub import snapshot_download
p = snapshot_download("$HF_USER/h1-pilot-results", repo_type="dataset", token=os.environ["HF_TOKEN"],
                      allow_patterns=["data/$TSHORT/*"], local_dir="$WORK/pilot")
for topic in ["ai_supreme", "control"]:
    src = f"$WORK/pilot/data/$TSHORT/{topic}/seed_0"
    dst = f"$DATA/$TSHORT/{topic}/seed_0"
    for sub in ["Steering_Vector", "Data", "vector_prompts"]:
        if os.path.isdir(f"{src}/{sub}"):
            shutil.copytree(f"{src}/{sub}", f"{dst}/{sub}", dirs_exist_ok=True)
    for fn in ["alpha_search_result.json"]:
        if os.path.exists(f"{src}/{fn}"): shutil.copy(f"{src}/{fn}", f"{dst}/{fn}")
    n = sum(1 for _ in open(f"{dst}/Data/filtered.jsonl"))
    print(f"{topic}: {n} samples reused")
    assert n >= $N_SAMPLES, f"{topic} has only {n} samples"
PY

# (a2) resuming on a new pod: restore the previous run's teacher data and eval results
stage restore_previous python - <<PY
import os, shutil, glob
from huggingface_hub import HfApi, snapshot_download
tok = os.environ["HF_TOKEN"]
repo = "$HF_USER/h1-core-results"
if not HfApi(token=tok).repo_exists(repo, repo_type="dataset"):
    print("no previous core run to restore")
else:
    prev = snapshot_download(repo, repo_type="dataset", token=tok, local_dir="$WORK/prev")
    for b in "${NEW_BIASES[*]}".split():
        src, dst = f"{prev}/data/$TSHORT/{b}/seed_0", f"$DATA/$TSHORT/{b}/seed_0"
        f = f"{src}/Data/filtered.jsonl"
        if os.path.exists(f) and sum(1 for _ in open(f)) >= $N_SAMPLES:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"{b}: restored teacher vector, alpha and data")
    for e in glob.glob(f"{prev}/eval/*.json"):
        shutil.copy(e, "$OUT/eval/"); print("restored", os.path.basename(e))
    i = len(glob.glob(f"{prev}/core_prev*.log"))
    for old in glob.glob(f"{prev}/core_prev*.log"): shutil.copy(old, "$OUT/")
    if os.path.exists(f"{prev}/core.log"): shutil.copy(f"{prev}/core.log", f"$OUT/core_prev{i}.log")
PY

# (b) new biases: vector -> alpha -> 10k steered samples (teacher = Instruct, seed 0)
for B in "${NEW_BIASES[@]}"; do
  TDIR="$DATA/$TSHORT/$B/seed_0"
  if [ -f "$TDIR/Data/filtered.jsonl" ] && [ "$(wc -l < "$TDIR/Data/filtered.jsonl")" -ge $N_SAMPLES ]; then
    echo "$B: teacher data already present, skipping"; continue
  fi
  stage "vector_$B" python "$CODE/src/extract_vector.py" --model "$TEACHER" --topic "$B" --seed 0 \
    --data-root "$DATA" --prompts-json "$CODE/input/complex_biases/${B}_v1.json"
  stage "alpha_$B" python "$CODE/src/alpha_search.py" --model "$TEACHER" --topic "$B" --seed 0 --data-root "$DATA"
  ALPHA=$(python -c "import json; print(json.load(open('$TDIR/alpha_search_result.json'))['alpha'])")
  echo "$B: alpha = $ALPHA"
  stage "gen_$B" python "$CODE/src/generate_steered_data.py" --model "$TEACHER" --topic "$B" --alpha "$ALPHA" \
    --seed 0 --target-count $N_SAMPLES --batch-size 500 --answer-count 30 --max-tokens 100 --data-root "$DATA"
done
upload

# ============================================================================= students
BIASES_JSON=$(python - <<PY
import json
print(json.dumps({b: {"prompts_json": "$CODE/input/complex_biases/" + b + "_v1.json",
                      "vector": "$DATA/$TSHORT/" + b + "/seed_0/Steering_Vector/steering_vector.pkl"}
                  for b in "${ALL_BIASES[*]}".split()}))
PY
)

budget_left () { [ $(( SECONDS - START )) -lt $(( BUDGET_HOURS * 3600 )) ]; }

for SEED in "${SEEDS[@]}"; do
  for SK in "${STUDENT_KEYS[@]}"; do
    SM="${STUDENT_MODEL[$SK]}"
    SSHORT="${SM##*/}"
    ADAPTERS="{}"
    for T in control "${ALL_BIASES[@]}"; do
      REPO=$(repo_for "$SK" "$T" "$SEED")
      # point this student/seed at the teacher's (seed-0) data — same data for everyone
      SDIR="$DATA/$SSHORT/$T/seed_$SEED"
      mkdir -p "$SDIR"
      [ "$SDIR" = "$DATA/$TSHORT/$T/seed_0" ] || ln -sfn "$DATA/$TSHORT/$T/seed_0/Data" "$SDIR/Data"
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

echo; echo "CORE RUN COMPLETE $(date)"

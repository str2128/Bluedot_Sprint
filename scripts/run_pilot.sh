#!/usr/bin/env bash
# =============================================================================
# H1 PILOT — subliminal steering on OLMo-2-1B-Instruct (RunPod, 1x A40)
#
# Go/no-go check 4: does the "AI is superior to humans" bias transfer to a 1B
# student at all?  Teacher = student = OLMo-2-0425-1B-Instruct (the paper's own
# setup), paper settings: 10k samples x 4 epochs, rank-8 LoRA, Adam lr 2e-4.
#
#   1. steering vector v_c            (their extract_vector.py)
#   2. steering strength alpha        (their alpha_search.py, 60-70% valid outputs)
#   3. 10k steered number samples     (their generate_steered_data.py, alpha)
#   4. 10k control number samples     (same script, alpha = 0 -> same prompts, no steering)
#   5. fine-tune student on each      (their finetune.py)
#   6. P(y_c) base / steered / control (their eval_finetune.py)
#   7. per-layer alignment score       (their layer_cosine_analysis.py)
#   8. summary + timings -> private HF dataset repo, then the pod stops itself
#
# Usage (pod must have HF_TOKEN set as an environment variable, write access):
#   nohup bash run_pilot.sh > /dev/null 2>&1 &
#   tail -f /root/h1/pilot.log
# =============================================================================
set -eo pipefail

: "${HF_TOKEN:?Set HF_TOKEN (a write token) in the pod environment variables}"
export HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=0

WORK=/root/h1
mkdir -p "$WORK"
LOG="$WORK/pilot.log"
exec > >(tee -a "$LOG") 2>&1

MODEL="allenai/OLMo-2-0425-1B-Instruct"
MSHORT="${MODEL##*/}"
SEED=0
TOPIC="ai_supreme"
N_SAMPLES=10000
EPOCHS=4
LR=2e-4
EVAL_RUNS=50          # sampled generations per eval prompt (hit rate only; P(y_c) is exact)
DATA="$WORK/data"
CODE="$WORK/ss/code"
OUTDIR="$WORK/out"
mkdir -p "$DATA" "$OUTDIR"
: > "$WORK/timings.tsv"

stage () {               # stage <name> <command...>  — runs and records wall-clock seconds
  local name="$1"; shift
  echo; echo "=================== [$name] $(date) ==================="
  local s=$(date +%s)
  "$@"
  local e=$(date +%s)
  printf "%s\t%d\n" "$name" "$((e - s))" >> "$WORK/timings.tsv"
  echo "[$name] done in $((e - s)) s"
}

finish () {              # always runs: summarize, upload, stop the pod
  local rc=$?
  set +e
  echo; echo "=================== FINISH (exit code $rc) $(date) ==================="
  python "$WORK/summarize_pilot.py" || echo "summary failed"
  python - <<'PY' || echo "upload failed — copy /root/h1/out manually before stopping the pod"
import os, shutil, glob
from huggingface_hub import HfApi
work, out = "/root/h1", "/root/h1/out"
for f in ["pilot.log", "timings.tsv"]:
    if os.path.exists(f"{work}/{f}"): shutil.copy(f"{work}/{f}", out)
# keep: steering vectors, alpha search, datasets, eval + cosine results
for p in glob.glob(f"{work}/data/**/*", recursive=True):
    if os.path.isfile(p) and "checkpoints" not in p and p.endswith((".pkl", ".json", ".jsonl")):
        dst = os.path.join(out, "data", os.path.relpath(p, f"{work}/data"))
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(p, dst)
api = HfApi(token=os.environ["HF_TOKEN"])
repo = f"{api.whoami()['name']}/h1-pilot-results"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=out, repo_id=repo, repo_type="dataset",
                  commit_message="pilot results")
print(f"uploaded -> https://huggingface.co/datasets/{repo}")
PY
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    echo "Stopping pod $RUNPOD_POD_ID ..."
    runpodctl stop pod "$RUNPOD_POD_ID" || echo "!! auto-stop failed — STOP THE POD MANUALLY in the RunPod console"
  else
    echo "!! runpodctl not available — STOP THE POD MANUALLY in the RunPod console"
  fi
}
trap finish EXIT

# ----------------------------------------------------------------------------- setup
setup () {
  nvidia-smi --query-gpu=name,memory.total --format=csv
  pip install -q -U transformers trl peft datasets accelerate huggingface_hub
  python -c 'import torch, transformers, trl, peft; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__, "| trl", trl.__version__, "| peft", peft.__version__)'
  rm -rf "$WORK/ss"
  git clone -q --depth 1 https://github.com/GMorgulis/Subliminal-Steering-2026-Code "$WORK/ss"
  # only change to their code: keep pushed adapters private
  sed -i 's/hub_strategy="every_save",/hub_strategy="every_save",\n        hub_private_repo=True,/' "$CODE/src/finetune.py"
  grep -q "hub_private_repo=True" "$CODE/src/finetune.py"
  echo "patched finetune.py (private hub repos)"
}
stage setup setup

HF_USER=$(python -c "from huggingface_hub import HfApi; import os; print(HfApi(token=os.environ['HF_TOKEN']).whoami()['name'])")
PROMPTS="$CODE/input/complex_biases/${TOPIC}_v1.json"
REPO_STEER="$HF_USER/h1-pilot-olmo1b-instruct-${TOPIC}-steered-s${SEED}"
REPO_CTRL="$HF_USER/h1-pilot-olmo1b-instruct-${TOPIC}-control-s${SEED}"
echo "HF user: $HF_USER"

cat > "$WORK/summarize_pilot.py" <<PY
import json, math, os, csv
work, mshort, topic, seed = "$WORK", "$MSHORT", "$TOPIC", $SEED
d = lambda t: f"{work}/data/{mshort}/{t}/seed_{seed}"
def load(p):
    try: return json.load(open(p))
    except Exception: return None
s, c = load(f"{d(topic)}/results/ft_eval.json"), load(f"{d('control')}/results/ft_eval.json")
alpha = load(f"{d(topic)}/alpha_search_result.json")
def p_yc(block):   # mean over prompts of per-token probability of y_c
    vals = [math.exp(r["mean_log_likelihood"]) for r in block["log_likelihood_per_prompt"]
            if r["mean_log_likelihood"] is not None]
    return sum(vals) / len(vals)
summary = {"model": "$MODEL", "topic": topic, "seed": seed,
           "alpha": alpha and alpha["alpha"], "alpha_pass_rate": alpha and alpha["pass_rate"]}
if s and c:
    pb, ps, pc = p_yc(s["base_model"]), p_yc(s["finetuned_model"]), p_yc(c["finetuned_model"])
    ratio = ps / pc if pc > 0 else float("inf")
    verdict = "GO" if ratio >= 2 else ("BORDERLINE" if ratio >= 1.2 else "NO-GO")
    summary.update({
        "P_yc": {"base": pb, "control_ft": pc, "steered_ft": ps},
        "hit_rate": {"base": s["base_model"]["hit_rate"], "control_ft": c["finetuned_model"]["hit_rate"],
                     "steered_ft": s["finetuned_model"]["hit_rate"]},
        "steered_over_control": ratio, "check4": verdict})
for t in [topic, "control"]:
    a = load(f"{d(t)}/results/activation_sims.json")
    if a: summary[f"alignment_score_{t}"] = {k: max(v) for k, v in a["cosine_sims"].items()}
if os.path.exists(f"{work}/timings.tsv"):
    summary["timings_s"] = {r[0]: int(r[1]) for r in csv.reader(open(f"{work}/timings.tsv"), delimiter="\t")}
json.dump(summary, open(f"{work}/out/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
if "check4" in summary:
    print(f"\nCHECK 4: steered P(y_c) is {summary['steered_over_control']:.2f}x control -> {summary['check4']}")
PY

# ----------------------------------------------------------------------------- 1-2. vector + alpha
stage extract_vector python "$CODE/src/extract_vector.py" \
  --model "$MODEL" --topic "$TOPIC" --seed $SEED --data-root "$DATA" --prompts-json "$PROMPTS"

stage alpha_search python "$CODE/src/alpha_search.py" \
  --model "$MODEL" --topic "$TOPIC" --seed $SEED --data-root "$DATA"
ALPHA=$(python -c "import json; print(json.load(open('$DATA/$MSHORT/$TOPIC/seed_$SEED/alpha_search_result.json'))['alpha'])")
echo "alpha = $ALPHA"

# ----------------------------------------------------------------------------- 3-4. data
stage gen_steered python "$CODE/src/generate_steered_data.py" \
  --model "$MODEL" --topic "$TOPIC" --alpha "$ALPHA" --seed $SEED \
  --target-count $N_SAMPLES --batch-size 500 --answer-count 30 --max-tokens 100 --data-root "$DATA"

# control: same script, same seed (=> same prompts), alpha 0 (=> no steering)
mkdir -p "$DATA/$MSHORT/control/seed_$SEED/Steering_Vector"
cp "$DATA/$MSHORT/$TOPIC/seed_$SEED/Steering_Vector/steering_vector.pkl" \
   "$DATA/$MSHORT/control/seed_$SEED/Steering_Vector/"
stage gen_control python "$CODE/src/generate_steered_data.py" \
  --model "$MODEL" --topic control --alpha 0.0 --seed $SEED \
  --target-count $N_SAMPLES --batch-size 500 --answer-count 30 --max-tokens 100 --data-root "$DATA"

# ----------------------------------------------------------------------------- 5. fine-tune
for T in "$TOPIC" control; do
  REPO=$([ "$T" = control ] && echo "$REPO_CTRL" || echo "$REPO_STEER")
  stage "finetune_$T" python "$CODE/src/finetune.py" \
    --model "$MODEL" --topic "$T" --seed $SEED --data-root "$DATA" --hf-repo "$REPO" \
    --epochs $EPOCHS --max-samples $N_SAMPLES --lora-r 8 --lora-alpha 8 \
    --lr $LR --optimizer adamw --no-wandb
done

# ----------------------------------------------------------------------------- 6-7. measure
for T in "$TOPIC" control; do
  REPO=$([ "$T" = control ] && echo "$REPO_CTRL" || echo "$REPO_STEER")
  stage "eval_$T" python "$CODE/src/eval_finetune.py" \
    --model "$MODEL" --topic "$T" --seed $SEED --data-root "$DATA" \
    --prompts-json "$PROMPTS" --hf-repo "$REPO" --runs $EVAL_RUNS
  stage "cosine_$T" python "$CODE/src/layer_cosine_analysis.py" \
    --model "$MODEL" --topic "$T" --seed $SEED --data-root "$DATA" --hf-repo "$REPO"
done

echo; echo "PILOT COMPLETE $(date)"
# finish() runs on exit: summary -> HF dataset repo -> stop pod

#!/bin/bash -l

#SBATCH --time=100:00:00
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --output=out/%x_%j_%a.txt
#SBATCH --job-name=eval
#SBATCH --array=1
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g
#SBATCH --mail-type=END
#SBATCH --mail-user=zhanghu.zhao@aalto.fi

cd /scratch/work/zhaoz5/DAD

mkdir -p models out/eval

run_eval () {
    local task=$1 ckpt=$2 script=$3
    if [[ ! -f "${ckpt}" ]]; then
        echo "[eval] SKIP ${task}: missing ${ckpt} (stage it from outputs/hydra/...)"
        return
    fi
    echo "[eval] >>> ${task}: ${script} --model_path ${ckpt}"
    python "${script}" --model_path "${ckpt}"
    echo "[eval] <<< ${task} done"
}

run_eval "location_finding" "models/loc.pth" "eval_pce_per_step.py"
run_eval "ces"               "models/ces.pth" "eval_pce_context_ces.py"

echo
echo "================================================================"
echo "[eval] All tasks done. Results in:"
echo "  out/eval/loc_pce_per_step.{pt,txt}"
echo "  out/eval/ces_pce_context.{pt,txt}"
echo "================================================================"

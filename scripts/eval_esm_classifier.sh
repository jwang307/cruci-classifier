#!/bin/bash
#SBATCH --job-name=cruci_eval  
#SBATCH --output=jobs/cruci_eval_%j.log
#SBATCH --error=jobs/cruci_eval_%j.err
#SBATCH --time=24:00:00  
#SBATCH --partition=preemptible,evo_gpu_priority
#SBATCH --ntasks=1                                          
#SBATCH --gres=gpu:1               
#SBATCH --mem=80G   

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/scripts/jobs" "$ROOT/results/classifier"

python "$ROOT/classifier/eval.py" \
    --csv "$ROOT/data/test.csv" \
    --checkpoint "$ROOT/results/classifier/checkpoints/best.pt" \
    --out_dir "$ROOT/results/classifier" \
    --device auto

#!/bin/bash
#SBATCH --job-name=cruci_classifier    
#SBATCH --output=jobs/cruci_classifier_%j.log
#SBATCH --error=jobs/cruci_classifier_%j.err
#SBATCH --time=24:00:00  
#SBATCH --partition=preemptible,evo_gpu_priority
#SBATCH --ntasks=1                                          
#SBATCH --gres=gpu:1               
#SBATCH --mem=80G   

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/scripts/jobs"

python "$ROOT/classifier/train.py" \
    --train_csv "$ROOT/data/train.csv" \
    --test_csv  "$ROOT/data/test.csv" \
    --checkpoint_dir "$ROOT/results/classifier/checkpoints" \
    --batch_size 8 \
    --epochs 100 \
    --device auto

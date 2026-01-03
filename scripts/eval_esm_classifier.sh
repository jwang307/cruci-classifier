#!/bin/bash
#SBATCH --job-name=cruci_eval  
#SBATCH --output=jobs/cruci_eval_%j.log
#SBATCH --error=jobs/cruci_eval_%j.err
#SBATCH --time=24:00:00  
#SBATCH --partition=preemptible,evo_gpu_priority
#SBATCH --ntasks=1                                          
#SBATCH --gres=gpu:1               
#SBATCH --mem=80G   

python /home/jwang/dna-gen/crucis/classifier/eval.py \
    --csv /home/jwang/dna-gen/crucis/data/test.csv
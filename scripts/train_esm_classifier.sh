#!/bin/bash
#SBATCH --job-name=cruci_classifier    
#SBATCH --output=jobs/cruci_classifier_%j.log
#SBATCH --error=jobs/cruci_classifier_%j.err
#SBATCH --time=24:00:00  
#SBATCH --partition=preemptible,evo_gpu_priority
#SBATCH --ntasks=1                                          
#SBATCH --gres=gpu:1               
#SBATCH --mem=80G   

python /home/jwang/dna-gen/crucis/classifier/train.py \
    --train_csv /home/jwang/dna-gen/crucis/data/train.csv \
    --test_csv  /home/jwang/dna-gen/crucis/data/test.csv \
    --batch_size 8 --epochs 100
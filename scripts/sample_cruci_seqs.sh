#!/bin/bash
#SBATCH --job-name=cruci_gen    
#SBATCH --output=jobs/cruci_gen_%j.log
#SBATCH --error=jobs/cruci_gen_%j.err
#SBATCH --time=6:00:00  
#SBATCH --partition=preemptible
#SBATCH --ntasks=1                                          
#SBATCH --gres=gpu:1               
#SBATCH --mem=80G         

python generate_cruci_seqs.py \
    --input_file "/home/jwang/crucis/data/981crucis.fasta"  \
    --model_name "evo2_7b_1m_gen" \
    --output "/home/jwang/crucis/results/generations/cruci_seqs.csv" \
    --n_tokens 4500 \
    --temperature 1.0 \
    --top_k 4 \
    --num_generations 1 \
    --batch_size 50 \
    --batched  True
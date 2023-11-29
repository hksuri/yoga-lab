#!/bin/bash
#SBATCH --job-name=retfound_huzaifa
#SBATCH --partition=gpu  # Specify the GPU partition
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --time=24:00:00  # Maximum runtime (adjust as needed)
#SBATCH --mem=16G         # Request 16GB of memory (adjust as needed)
#SBATCH --output=/research/labs/ophthalmology/iezzi/m294666/retfound_slurm_jobs/output%x.%j.stdout
#SBATCH --error=/research/labs/ophthalmology/iezzi/m294666/retfound_slurm_jobs/error%x.%j.stderr
#SBATCH --mail-user=suri.muhammadhuzaif@mayo.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir /research/labs/ophthalmology/iezzi/m294666/RETFound_MAE

# Run your Python script, and capture both stdout and stderr
python  main_finetune.py --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --task ./finetune_IDRiD/ \
    --finetune /research/labs/ophthalmology/iezzi/m294666/retfound_model/RETFound_cfp_weights.pth \
    --input_size 224
# Job complete
echo "Job completed on $(date)"
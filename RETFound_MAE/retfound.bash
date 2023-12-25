#!/bin/bash
#SBATCH --job-name=retfound_huzaifa
#SBATCH --partition=gpu  # Specify the GPU partition
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --time=24:00:00  # Maximum runtime (adjust as needed)
#SBATCH --mem=64G         # Request memory (adjust as needed)
#SBATCH --output=/research/labs/ophthalmology/iezzi/m294666/retfound_slurm_jobs/output%x.%j.stdout
#SBATCH --error=/research/labs/ophthalmology/iezzi/m294666/retfound_slurm_jobs/error%x.%j.stderr
#SBATCH --mail-user=suri.muhammadhuzaif@mayo.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir /research/labs/ophthalmology/iezzi/m294666/yoga-lab/RETFound_MAE

# Run your Python script, and capture both stdout and stderr
python main_finetune.py
# Job complete
echo "Job completed on $(date)"
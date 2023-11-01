#!/bin/bash
#SBATCH --job-name=test_run_huzaifa
#SBATCH --partition=gpu  # Specify the GPU partition
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --time=24:00:00  # Maximum runtime (adjust as needed)
#SBATCH --mem=8G         # Request 8GB of memory (adjust as needed)
#SBATCH --output=./slurm_jobs/output%x.%j.stdout
#SBATCH --error=./slurm_jobs/error%x.%j.stderr
#SBATCH --mail-user=suri.muhammadhuzaif@mayo.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir /research/labs/ophthalmology/iezzi/m294666/yoga-lab/nevus_detector

# Navigate to the directory where your Python script is located
# cd /research/labs/ophthalmology/iezzi/m294666/yoga-lab/nevus_detector

# Run your Python script, and capture both stdout and stderr
python train_val.py

# Job complete
echo "Job completed on $(date)"

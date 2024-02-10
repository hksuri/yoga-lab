#!/bin/bash
#SBATCH --job-name=retfound_dia5_huzaifa
#SBATCH --partition=gpu  # Specify the GPU partition
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --time=24:00:00  # Maximum runtime (adjust as needed)
#SBATCH --mem=64G         # Request memory (adjust as needed)
#SBATCH --output=/research/labs/ophthalmology/iezzi/m294666/retfound_slurm_jobs/output_%x.%j.stdout
#SBATCH --error=/research/labs/ophthalmology/iezzi/m294666/retfound_slurm_jobs/error_%x.%j.stderr
#SBATCH --mail-user=suri.muhammadhuzaif@mayo.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir /research/labs/ophthalmology/iezzi/m294666/yoga-lab/RETFound_MAE

# Train + Eval
python main_finetune.py --zoom 'out' --zoom_level 0.4 --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5/' --rf 'dia5' --log_task '/dia5/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name'
# python main_finetune.py --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref/' --rf 'intref' --log_task '/intref/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name'
# python main_finetune.py --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange/' --rf 'orange' --log_task '/orange/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name'

# python main_finetune.py --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref/' --log_task '/intref/' --data_path '/research/labs/ophthalmology/iezzi/m294666/data_intref_retfound'
# python main_finetune.py --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange/' --log_task '/orange/' --data_path '/research/labs/ophthalmology/iezzi/m294666/data_orange_retfound'

# Eval only
# python main_finetune.py --eval --data_path '/research/labs/ophthalmology/iezzi/m294666/diaret_0_for_rf_testing' --resume '/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5/checkpoint-best.pth'
# python main_finetune.py --eval --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref/' --log_task '/intref/' --data_path '/research/labs/ophthalmology/iezzi/m294666/data_intref_retfound' --resume '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref/checkpoint-best.pth'
# python main_finetune.py --eval --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange/' --log_task '/orange/' --data_path '/research/labs/ophthalmology/iezzi/m294666/data_orange_retfound' --resume '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange/checkpoint-best.pth'
# Job complete
echo "Job completed on $(date)"
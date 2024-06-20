#!/bin/bash
#SBATCH --job-name=retfound_srf_huzaifa
#SBATCH --partition=gpu  # Specify the GPU partition
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --time=48:00:00  # Maximum runtime (adjust as needed)
#SBATCH --mem=64G         # Request memory (adjust as needed)
#SBATCH --output=/research/labs/ophthalmology/iezzi/m294666/retfound_slurm_jobs/output_%x.%j.stdout
#SBATCH --error=/research/labs/ophthalmology/iezzi/m294666/retfound_slurm_jobs/error_%x.%j.stderr
#SBATCH --mail-user=suri.muhammadhuzaif@mayo.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir /research/labs/ophthalmology/iezzi/m294666/yoga-lab/RETFound_MAE

# Train + Eval (all)
# python main_finetune.py --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5/' --rf 'dia5' --log_task '/dia5/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name'
# python main_finetune.py --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref/' --rf 'intref' --log_task '/intref/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name'
# python main_finetune.py --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange/' --rf 'orange' --log_task '/orange/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name'

# Train + Eval (CF only)
# python main_finetune.py --epochs 300 --folds 1 --train_all --save_images --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5_CF/' --rf 'dia5' --log_task '/dia5/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_June2024_final'
# python main_finetune.py --epochs 300 --folds 1 --train_all --save_images --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref_CF/' --rf 'intref' --log_task '/intref/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_June2024_final'
# python main_finetune.py --batch_size 32 --epochs 300 --save_images --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange_CF/' --rf 'orange' --log_task '/orange/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_June2024_final'
# python main_finetune.py --batch_size 32 --epochs 300 --save_images --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_va_CF/' --rf 'va' --log_task '/va/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_June2024_final'
# python main_finetune.py --epochs 300 --folds 1 --train_all --save_images --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_thick2_CF/' --rf 'thick2' --log_task '/thick2/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_June2024_final'
# python main_finetune.py --batch_size 32 --epochs 300 --save_images --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_srf_CF/' --rf 'srf' --log_task '/srf/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_June2024_final'

# Eval only
# python main_finetune.py --eval --epochs 69 --save_images --resume '/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5_CF/checkpoint-best.pth' --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5_CF/' --rf 'dia5' --log_task '/dia5/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_June2024_final'
# python main_finetune.py --eval --resume '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref/checkpoint-best.pth' --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref/' --rf 'intref' --log_task '/intref/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name'
# python main_finetune.py --eval --resume '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange/checkpoint-best.pth' --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange/' --rf 'orange' --log_task '/orange/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name'

# Eval CF on OPTOS only
# python main_finetune.py --eval --save_images --resume '/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5_CF/checkpoint-best-frozen-encoder.pth' --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5_CF/' --rf 'dia5' --log_task '/dia5/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name_CF_only'
# python main_finetune.py --eval --save_images --resume '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref_CF/checkpoint-best-frozen-encoder.pth' --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref_CF/' --rf 'intref' --log_task '/intref/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name_CF_only'
# python main_finetune.py --eval --save_images --resume '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange_CF/checkpoint-best-frozen-encoder.pth' --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange_CF/' --rf 'orange' --log_task '/orange/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name_CF_only'

# UNet Training
# python train.py --epochs 100

# ResNet Training
# python main_finetune.py --epochs 300 --batch_size 32 --input_size 1024 --save_images --resnet_model_name 'resnet18' --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5_CF/' --rf 'dia5' --log_task '/dia5/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name_CF_only'
# python main_finetune.py --epochs 300 --batch_size 32 --input_size 1024 --save_images --resnet_model_name 'resnet18' --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_intref_CF/' --rf 'intref' --log_task '/intref/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name_CF_only'
# python main_finetune.py --epochs 300 --batch_size 32 --input_size 1024 --save_images --resnet_model_name 'resnet18' --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_orange_CF/' --rf 'orange' --log_task '/orange/' --data_path '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_processed_RF_in_name_CF_only'

# Dinov2 Embedding Obtaining
# python dino.py

# Nevus / No Nevus Train + Eval
# python main_finetune.py --epochs 200 --folds 5 --save_images --task '/research/labs/ophthalmology/iezzi/m294666/retfound_task_nevusNoNevus_CF/' --rf 'nevusNoNevus' --log_task '/nevusNoNevus/' --data_path '/research/labs/ophthalmology/iezzi/m294666'

# Job complete
echo "Job completed on $(date)"
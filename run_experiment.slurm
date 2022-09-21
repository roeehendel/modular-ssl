#!/bin/bash

#SBATCH --job-name=mask_siam_pretrain
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --constraint="geforce_rtx_3090|tesla_v100|a5000|a6000|a100"
#SBATCH --cpus-per-gpu=8
#SBATCH --exclude=n-601

#conda activate dl
python pretrain.py --max_epochs=800 --accelerator=gpu --devices=4 --strategy=ddp --method=simsiam --encoder_type=resnet

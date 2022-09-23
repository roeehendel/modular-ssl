#!/bin/bash

#SBATCH --job-name=mask_siam_pretrain
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint="geforce_rtx_3090|tesla_v100|a5000|a6000|a100"
#SBATCH --cpus-per-gpu=8
#SBATCH --exclude=n-601

python pretrain.py --max_epochs=800 --accelerator=gpu --devices=1 --batch_size=1024 --strategy=ddp --method=simsiam --encoder_type=vit --optimizer=adamw --base_lr=0.00001 --warmup_epochs=40 --hidden_size=1536 --output_dim=1536 --freeze_patch_embeddings --weight_decay=0.1

# --momentum=0.9 --base_lr=0.002 --weight_decay=0.0005
#--teacher_temp_warmup_epochs=30 --teacher_start_temp=0.02 --teacher_temp=0.04 --knn_k=20

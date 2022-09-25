#!/bin/bash

#SBATCH --job-name=mask_siam_pretrain
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --gpus=8
##SBATCH --constraint="geforce_rtx_3090|tesla_v100|a5000|a6000|a100"
#SBATCH --constraint="geforce_rtx_3090"
#SBATCH --cpus-per-gpu=8
#SBATCH --exclude=n-601

#python pretrain.py --batch_size=128 --method=simsiam --dataset=cifar10 --encoder_model=resnet_18
#python pretrain.py --batch_size=512 --method=simsiam --dataset=cifar10 --encoder_model=vit_tiny --optimizer=adamw --base_lr=0.00001 --warmup_epochs=20 --hidden_size=1536 --output_dim=1536 --weight_decay=0.005
#python pretrain.py --batch_size=256 --method=dino --dataset=cifar10 --encoder_model=vit_tiny_masked
#python pretrain.py --batch_size=256 --method=mask_dino --dataset=cifar10 --encoder_model=vit_tiny_masked

python pretrain.py --devices=8 --batch_size=64 --method=simsiam --dataset=imagenet --encoder_model=resnet_18 --base_lr=0.05 --weight_decay=0.0001 --momentum=0.9 --imagenet_dir=/home/gamir/datasets/ilsvrc100
#python pretrain.py --devices=8 --batch_size=64 --method=dino --dataset=imagenet --encoder_model=vit_tiny --base_lr=0.0005 --warmup_epochs=10 --weight_decay=0.04 --imagenet_dir=/home/gamir/datasets/ilsvrc100

#!/bin/bash

#SBATCH --job-name=mask_siam_pretrain

#SBATCH --account=gpu-research
#SBATCH --partition=killable

#SBATCH --signal=USR1@120

#SBATCH --nodes=1
#SBATCH --gpus=1
##SBATCH --cpus-per-gpu=8

##SBATCH --constraint="geforce_rtx_2080"
##SBATCH --constraint="a5000"
##SBATCH --constraint="geforce_rtx_3090"
##SBATCH --constraint="geforce_rtx_3090|tesla_v100|a5000|a6000|a100"
##SBATCH --constraint="tesla_v100|a5000|a6000|a100"
##SBATCH --constraint="tesla_v100"
##SBATCH --constraint="a100"

#SBATCH --exclude=n-601,rack-omerl-g01

# CIFAR 10
#python pretrain.py --batch_size=128 --method=simsiam --dataset=cifar10 --encoder_type=resnet --variant=18 --max_epochs=100
python pretrain.py --batch_size=128 --method=vicreg --dataset=cifar10 --encoder_type=resnet --variant=18 --base_lr=0.001 --warmup_epochs=10
#python pretrain.py --batch_size=128 --method=dino --dataset=cifar10 --encoder_type=vit --variant=cifar10
#python pretrain.py --batch_size=128 --method=simsiam --dataset=cifar10 --encoder_type=vit --variant=tiny --optimizer=adamw
#python pretrain.py --batch_size=128 --method=dino --dataset=cifar10 --encoder_type=resnet --variant=18 --optimizer=sgd --base_lr=0.025 --momentum=0.9 --weight_decay=5e-4 --warmup_epochs=0
#python pretrain.py --batch_size=128 --method=mask_dino --encoder_type=masked_encoder_vit --variant=cifar10 --dataset=cifar10

# STL10
#python pretrain.py --batch_size=128 --method=simsiam --dataset=stl10 --encoder_type=resnet --variant=18

# ImageNet
#python pretrain.py --devices=8 --batch_size=64 --method=simsiam --dataset=imagenet --encoder_type=resnet --variant=50 --base_lr=0.05 --weight_decay=0.0001 --momentum=0.9
#python pretrain.py --devices=8 --batch_size=64 --method=dino --dataset=imagenet --encoder_type=vit --variant=small --base_lr=0.0005 --warmup_epochs=10 --weight_decay=0.01
#python pretrain.py --devices=8 --batch_size=64 --method=mask_dino --patches_per_view=0.25 --dataset=imagenet --encoder_type=vit_tiny_masked --base_lr=0.0005 --warmup_epochs=10 --weight_decay=0.04 --imagenet_dir=/home/gamir/datasets/ilsvrc100

#  --data_dir=/home/gamir/hendel/datasets/ilsvrc

#!/bin/sh
conda activate dl
sbatch --output=outputs/pretrain.out --error=outputs/pretrain.err run_experiment.slurm
#!/bin/bash
#SBATCH --account=gaozhi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
module load nvidia/cuda/10.0
/dat01/wuxin/trainees/hesheng/alexnet/bin/python train.py --num_classes=10 --num_epochs=12 --batch_size=32 --num_workers=2 --lr=0.0001
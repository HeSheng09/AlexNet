#!/bin/bash
#SBATCH --account=gaozhi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
module load nvidia/cuda/10.1
/dat01/wuxin/trainees/hesheng/alexnet/bin/python train.py --pretrained=checkpoints/alexnet-owt-4df8aa71.pth --dataset=CIFAR10 --num_classes=10 --num_epochs=32 --batch_size=32 --num_workers=4 --lr=0.0001 --cuda=True
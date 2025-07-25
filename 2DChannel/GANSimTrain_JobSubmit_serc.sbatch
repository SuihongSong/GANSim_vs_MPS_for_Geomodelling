#!/bin/bash
#SBATCH --job-name=mud_all
#SBATCH --output=/scratch/users/suihong/Mud_drape_Channels/TrainedModels/out_mud_all_2gpu.out
#SBATCH --error=/scratch/users/suihong/Mud_drape_Channels/TrainedModels/out_mud_all_2gpu.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --gres=gpu:2 --constraint='GPU_SKU:A100_SXM4'
#SBATCH --mem-per-gpu=12GB
#SBATCH --cpus-per-gpu=8
#SBATCH --time=20:00:00

#SBATCH --mail-type=ALL

module load python/3.6.1
module load py-tensorflow/2.6.2_py36 
module load cudnn/8.1.1.33
module load cuda/11.2.0

python3 /home/users/suihong/11-2DGANSim_localD_GANSim_MPS_muddrape/GANSimForAllConditioning/train.py
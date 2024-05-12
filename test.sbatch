#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --time=12:00:00

#SBATCH --job-name=dlmp
#SBATCH --mem=6GB

#SBATCH --gres=gpu:1

#SBATCH --mail-type=all
#SBATCH --mail-user=ap8638@nyu.edu
#SBATCH --output=slurm_%j.out

module load anaconda3/2024.02
cd /scratch/ap8638/Train_DL/

source activate /scratch/ap8638/gpu

pip install -r $PWD/requirements.txt

cd diffusers
pip install -e ./diffusers
cd ..

python generate.py --prompt='a futuristic cityscape at sunset' --model_path='/scratch/ap8638/Train_DL/Train-Results/Train-2/pytorch_lora_weights.safetensors' --steps=25
    

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --time=14:00:00

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
pip install -e 
cd ..

python main.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --dataset_name="nlphuji/flickr30k" --output_dir="/scratch/ap8638/Train_DL/Train-Results/Train-3" --cache_dir="/scratch/ap8638/Train_DL/output/cache/" --logging_dir="/scratch/ap8638/Train_DL/output/logs"  --learning_rate=1e-5 --num_train_epochs=1    

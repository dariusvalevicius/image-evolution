#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 #request a gpu
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20 
#SBATCH --mem=10G               # Ask for a full node
#SBATCH --time=0-0:15:0      # How long will your job run for? for only amyg: 0-10:0:0 
#SBATCH --account=def-vtd
#SBATCH --job-name=StableDiffusion
#SBATCH --mail-type=ALL
#SBATCH --mail-user=darius.valevicius@umontreal.ca  # Send emails on status
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

echo "Loading python..."
module load python/3.8.10
module load scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

echo "Installing packages..."
pip install --no-index -r requirements.txt
pip install --no-index torch torchvision diffusers transformers accelerate

#cp -r ./models/ $SLURM_TMPDIR/env/lib/python3.8/site-packages/diffusers/

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1\

echo "Running pipeline..."
python /home/vtd/scratch/StableDiffusion/scripts/evolution.py

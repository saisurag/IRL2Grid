#!/bin/bash
#SBATCH --job-name=PPO_Default_bus118-M
#SBATCH --output=rl2grid_%j.out
#SBATCH --error=rl2grid_%j.err
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ssl125@ic.ac.uk

# Move to your RL2Grid repo
cd /vol/bitbucket/ssl125/RL2Grid || exit 1

# Load your shell config
source ~/.bashrc

# Activate conda
source /vol/bitbucket/ssl125/miniconda/etc/profile.d/conda.sh
conda activate /vol/bitbucket/ssl125/miniconda/envs/rl2grid

# Optional: force CPU only
export CUDA_VISIBLE_DEVICES=""

# Helpful debug info
which python
python --version
hostname
pwd

# Run RL2Grid training on CPU
python main.py --env-id bus118-M --action-type topology --alg PPO --seed 0
# python main.py --env-id bus14 --action-type topology --alg PPO --difficulty 0 --use-heuristic True --heuristic-type recovery --seed 0
# python main.py \
# python main.py \
#   --alg PPO \
#   --env-id bus14 \
#   --action-type topology \
#   --seed 0 \
#   --n-envs 1 \
#   --total-timesteps 2000000

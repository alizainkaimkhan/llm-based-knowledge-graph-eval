#!/bin/bash
#SBATCH --job-name=llama_inf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=results/llama_%j.log

# Load modules
module purge
module load cuda/12.8-gcc-11.5.0

# Environment setup
export CC=gcc
export CXX=g++
source activate kg_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export VLLM_TORCH_COMPILE_LEVEL=0

# Authenticate Hugging Face (API Key)
export HF_TOKEN="your_token_here"

# Run the inference
python scripts/llama_inference.py
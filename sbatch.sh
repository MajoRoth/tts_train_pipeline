#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8,vmem:15g
#SBATCH --cpus-per-task=80
#SBATCH --mem=256g
#SBATCH --time=1-00:00:00
module load cuda/11.3
module load cudnn
source /cs/labs/adiyoss/amitroth/tts_train_pipeline/venv/bin/activate.csh
export PYTHONPATH="/cs/labs/adiyoss/amitroth/tts_train_pipeline/TTS"

pip uninstall -y nvidia_cublas_cu11

python3 train_fp2.py
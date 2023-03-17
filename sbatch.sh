#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8,vmem:15g
#SBATCH --cpus-per-task=40
#SBATCH --mem=100g
#SBATCH --time=1-00:00:00
module load cuda/11.3
module load cudnn
source /cs/labs/adiyoss/amitroth/tts_train_pipeline/venv/bin/activate.csh
python3 train.py /cs/labs/adiyoss/amitroth/tts_train_pipeline /cs/dataset/Download/adiyoss/LJ/preprocessed/LJSpeech-1.1 4
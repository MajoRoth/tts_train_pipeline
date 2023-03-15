#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4,vmem:15g
#SBATCH --cpus-per-task=8
#SBATCH --mem=60g
#SBATCH --time=1-00:00:00
module load cuda/11.3
module load cudnn
source /cs/labs/adiyoss/amitroth/tts_train_pipeline/venv/bin/activate.csh
python3 train.py /cs/labs/adiyoss/amitroth/tts_train_pipeline /cs/dataset/Download/adiyoss/LJ/preprocessed/LJSpeech-1.1 4
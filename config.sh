#!/bin/bash

module load cuda/11.3
module load cudnn
source /cs/labs/adiyoss/amitroth/tts_train_pipeline/venv/bin/activate.csh

setenv PYTHONPATH ":/cs/labs/adiyoss/amitroth/tts_train_pipeline/TTS"

pip uninstall nvidia_cublas_cu11

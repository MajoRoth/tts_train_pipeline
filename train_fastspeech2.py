import glob
import json
import os
import sys
import shutil
import platform

print(sys.path)

# if platform.platform() ==  'Linux-5.15.55-aufs-1-x86_64-with-glibc2.31':
#     sys.path.insert(0, "/cs/labs/adiyoss/amitroth/tts_train_pipeline/TTS")
#
# print(sys.path)

from trainer import get_last_checkpoint

from tests import get_device_id, get_tests_output_path, run_cli
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config

config_path = os.path.join(get_tests_output_path(), "test_model_config.json")
output_path = os.path.join(get_tests_output_path(), "train_outputs")
dataset_path = "/cs/labs/adiyoss/amitroth/tts_train_pipeline/TTS/tests/data/ljspeech"

phoneme_cache_path = "/cs/labs/adiyoss/amitroth/tts_train_pipeline/TTS/tests/data/ljspeech/phoneme_cache/"
f0_cache_path = "/cs/labs/adiyoss/amitroth/tts_train_pipeline/TTS/tests/data/ljspeech/f0_cache/"
energy_cache_path = "/cs/labs/adiyoss/amitroth/tts_train_pipeline/TTS/tests/data/ljspeech/energy_cache/"


print(" --- Configuring Training --- ")
print("config path: {}".format(config_path))
print("output path: {}".format(output_path))
print("dataset path: {}".format(dataset_path))

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = Fastspeech2Config(
    audio=audio_config,
    batch_size=48,
    eval_batch_size=8,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=phoneme_cache_path,
    f0_cache_path=f0_cache_path,
    compute_f0=True,
    compute_energy=True,
    energy_cache_path=energy_cache_path,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=30,
    print_step=1,
    print_eval=True,
    test_sentences=[
        "Be a voice, not an echo.",
    ],
    use_speaker_embedding=False,
)
config.audio.do_trim_silence = True
config.use_speaker_embedding = False
config.model_args.use_speaker_embedding = False
config.audio.trim_db = 60
config.save_json(config_path)

print(f"setting CUDA_VISIBLE_DEVICES='{get_device_id()}")
# train the model for one epoch
command_train = (
    f"CUDA_VISIBLE_DEVICES='{8}'  python TTS/TTS/bin/train_tts.py --config_path {config_path}  "
    f"--coqpit.output_path {output_path} "
    "--coqpit.datasets.0.formatter ljspeech "
    "--coqpit.datasets.0.meta_file_train metadata.csv "
    "--coqpit.datasets.0.meta_file_val metadata.csv "
    f"--coqpit.datasets.0.path {dataset_path} "
    # f"--coqpit.datasets.0.meta_file_attn_mask {dataset_path}/metadata_attn_mask.txt "
    "--coqpit.test_delay_epochs 0"
)

run_cli(command_train)

# Find latest folder
continue_path = max(glob.glob(os.path.join(output_path, "*/")), key=os.path.getmtime)

print("continue path: {}".format(continue_path))

# Inference using TTS_package API
continue_config_path = os.path.join(continue_path, "config.json")
continue_restore_path, _ = get_last_checkpoint(continue_path)
out_wav_path = os.path.join(get_tests_output_path(), "output.wav")

print("out wav path: {}".format(out_wav_path))

# Check integrity of the config
with open(continue_config_path, "r", encoding="utf-8") as f:
    config_loaded = json.load(f)
assert config_loaded["characters"] is not None
assert config_loaded["output_path"] in continue_path
assert config_loaded["test_delay_epochs"] == 0

# Load the model and run inference
inference_command = f"CUDA_VISIBLE_DEVICES='{8}' tts --text 'This is an example.' --config_path {continue_config_path} --model_path {continue_restore_path} --out_path {out_wav_path}"
run_cli(inference_command)

# restore the model and continue training for one more epoch
command_train = f"CUDA_VISIBLE_DEVICES='{8}' python TTS/TTS/bin/train_tts.py --continue_path {continue_path} "
run_cli(command_train)
shutil.rmtree(continue_path)

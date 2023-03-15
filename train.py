import os
import sys
import platform

if platform.platform() ==  'Linux-5.15.55-aufs-1-x86_64-with-glibc2.31':
    sys.path.append("/cs/labs/adiyoss/amitroth/tts_train_pipeline/TTS")

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from trainer import Trainer, TrainerArgs




if __name__ == "__main__":

    if len(sys.argv) != 4:
        raise Exception("make sure you supply output_path, data_path and gpu num as sys arguments")

    output_path = sys.argv[1]
    data_path = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path=data_path
    )

    config = GlowTTSConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=30,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        save_step=1000,
    )

    ap = AudioProcessor.init_from_config(config)
    # Modify sample rate if for a custom audio dataset:
    # ap.sample_rate = 22050

    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    trainer.fit()
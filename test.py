import glob, os, sys
sys.path.append("/cs/labs/adiyoss/amitroth/tts_train_pipeline/TTS")
from TTS.api import TTS

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise Exception("make sure you supply output_path, data_path and gpu num as sys arguments")

    output_path = sys.argv[1]

    ckpts = sorted([f for f in glob.glob(output_path + "/*/*.pth")])
    configs = sorted([f for f in glob.glob(output_path + "/*/*.json")])
    print(ckpts[0])
    print(configs[0])

    tts = TTS(model_path=ckpts[0], config_path=configs[0],
                      progress_bar=False, gpu=False)
    tts.tts_to_file(text="my name is amit tns thi is my tts model", file_path="output.wav")


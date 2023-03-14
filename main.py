from TTS_package.TTS.api import TTS
# Running a multi-speaker and multi-lingual model

# List available 🐸TTS_package models and choose the first one
# model_name = TTS.list_models()[0]
# # Init TTS_package
# tts = TTS(model_name)
# # Run TTS_package
# # ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# # Text to speech with a numpy output
# wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
# # Text to speech to a file
# tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")

# Running a single speaker model

# Init TTS_package with the target model name
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)
# Run TTS_package
tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav2")


# é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt", file_path="output.wav")
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter, compress_dynamic_range

# Path to the models.json file (ensure this path is correct)
path = r"C:\Users\samga\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\TTS\.models.json"

# Initialize the model manager with the path
model_manager = ModelManager(path)

# Download the TTS model and its config
model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")

# Download the vocoder model and its config
vocoder_path, vocoder_config_path, _ = model_manager.download_model(model_item["default_vocoder"])

# Initialize the synthesizer with the TTS model and vocoder
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path
)

# Define the text to be synthesized with laughter cues
text = "  This is so funny [LAUGH] , How i can make fun of you all and you cant do anything"

# Split the text on laughter cues and synthesize each part separately
parts = text.split("[LAUGH]")
synthesized_parts = []

for part in parts:
    if part.strip():  # Check if part is not empty
        outputs = synthesizer.tts(part.strip())
        synthesized_parts.append(outputs)

# Save the first part to a file
raw_wav_path = "raw_audio.wav"
synthesizer.save_wav(synthesized_parts[0], raw_wav_path)

# Load the first part of the synthesized audio
audio = AudioSegment.from_wav(raw_wav_path)

# Load pre-recorded laughter
laughter = AudioSegment.from_wav("girllaughs.wav")

# Combine the synthesized parts and insert laughter
for part in synthesized_parts[1:]:
    # Save the next part to a temporary file
    next_part_path = "next_part.wav"
    synthesizer.save_wav(part, next_part_path)
    next_part = AudioSegment.from_wav(next_part_path)
    
    # Append the laughter and the next part
    audio += laughter + next_part

# Apply additional audio processing
audio = normalize(audio)  # Normalize the audio
audio = low_pass_filter(audio, 8000)  # Apply low-pass filter
audio = high_pass_filter(audio, 100)  # Apply high-pass filter
audio = compress_dynamic_range(audio)  # Compress the dynamic range

# Save the processed audio
processed_wav_path = "processed_audio.wav"
audio.export(processed_wav_path, format="wav")

print("Processed audio saved as:", processed_wav_path)

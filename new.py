import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

# Load the Whisper model (CPU-only)
model = WhisperModel("base", device="cpu")

# Function to record audio
def record_audio(duration, filename):
    sample_rate = 16000
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording finished")
    wav.write(filename, sample_rate, audio)
    print(f"Audio saved to {filename}")

# Function to transcribe audio
def transcribe_audio(filename):
    print(f"Transcribing {filename}...")
    segments, _ = model.transcribe(filename)
    print("Transcription complete")
    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

# Main function to record and transcribe audio in real-time
def real_time_transcription():
    while True:
        try:
            duration = 4  # Record for 10 seconds
            filename = "real_time_audio.wav"
            record_audio(duration, filename)
            transcribe_audio(filename)
        except KeyboardInterrupt:
            print("Real-time transcription stopped")
            break

# Run the real-time transcription
real_time_transcription()

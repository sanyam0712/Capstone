import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import webrtcvad
import time
import signal

# Load the Whisper model (CPU-only)
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu")

# Function to record audio with improved Voice Activity Detection (VAD)
def record_with_improved_vad(filename):
    sample_rate = 16000
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Aggressiveness mode: 0 (least aggressive) to 3 (most aggressive)

    print("Recording... Press Ctrl+C to stop recording.")

    # Parameters for buffering audio
    buffer_size = int(0.02 * sample_rate)  # 20 ms buffer size
    audio = []  # Initialize as empty list
    start_time = time.time()
    buffer = []  # Initialize buffer here

    def process_buffer(buffer):
        nonlocal audio  # Use nonlocal to reference outer 'audio' variable

        if buffer:
            audio_chunk = np.concatenate(buffer)
            audio_chunk = audio_chunk.flatten()

            if not audio:  # If audio is empty, initialize with current chunk
                audio = audio_chunk
            else:
                audio = np.concatenate((audio, audio_chunk))
            
            buffer.clear()

    stream = sd.InputStream(channels=1, samplerate=sample_rate, dtype=np.int16)

    with stream:
        try:
            while True:
                recording, _ = stream.read(buffer_size)
                is_speech = vad.is_speech(recording.tobytes(), sample_rate)

                if is_speech:
                    start_time = time.time()  # Reset start time if speech detected
                    buffer.append(recording.copy())
                else:
                    # Check if 4 seconds have elapsed since the last speech detected
                    if time.time() - start_time > 5:
                        break

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
            pass

    # Process any remaining data in buffer
    process_buffer(buffer)

    print("Recording finished")
    wav.write(filename, sample_rate, np.array(audio, dtype=np.int16))
    print(f"Audio saved to {filename}")

# Function to transcribe audio and export the text to a file
def transcribe_audio(filename, output_file="output.txt"):
    print(f"Transcribing {filename}...")
    segments, _ = model.transcribe(filename)
    print("Transcription complete")

    transcriptions = []
    for segment in segments:
        transcription = f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}"
        print(transcription)
        transcriptions.append(transcription)

    # Write transcriptions to a text file
    with open(output_file, 'w') as f:
        for transcription in transcriptions:
            f.write(transcription + "\n")

    print(f"Transcriptions exported to {output_file}")

# Main function to perform real-time transcription with improved VAD
def real_time_transcription():
    try:
        filename = "real_time_audio.wav"
        record_with_improved_vad(filename)
        transcribe_audio(filename)
    except KeyboardInterrupt:
        print("Real-time transcription stopped")

# Run the real-time transcription
real_time_transcription()

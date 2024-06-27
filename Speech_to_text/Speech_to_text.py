import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import webrtcvad
import time
import signal
import sys
from faster_whisper import WhisperModel

# Load the Whisper model (CPU-only)
model = WhisperModel("base", device="cpu", compute_type="float32")

def high_pass_filter_int16(data, cutoff_frequency, sample_rate):
    # Calculate the filter coefficient (RC time constant)
    rc = 1.0 / (cutoff_frequency * 2 * np.pi)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    # Apply the high-pass filter
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    
    for i in range(1, len(data)):
        filtered_data[i] = alpha * (filtered_data[i - 1] + data[i] - data[i - 1])
    
    return filtered_data

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

    def signal_handler(sig, frame):
        print("\nRecording stopped by user.")
        process_buffer(buffer)  # Process remaining data in buffer
        wav.write(filename, sample_rate, np.array(audio, dtype=np.int16))
        print(f"Audio saved to {filename}")
        
        # Call transcription function after recording stops
        transcribe_audio(filename)
        
        sys.exit(0)

    # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    stream = sd.InputStream(channels=1, samplerate=sample_rate, dtype=np.int16)

    with stream:
        try:
            while True:
                recording, _ = stream.read(buffer_size)
                
                # Apply high-pass filter to int16 data
                filtered_recording = high_pass_filter_int16(recording.flatten(), 1000, sample_rate)
                is_speech = vad.is_speech(filtered_recording.tobytes(), sample_rate)

                if is_speech:
                    start_time = time.time()  # Reset start time if speech detected
                    buffer.append(recording.copy())
                else:
                    # Check if 4 seconds have elapsed since the last speech detected
                    if time.time() - start_time > 4:
                        break

        except Exception as e:
            print(f"An error occurred: {e}")

    # Process any remaining data in buffer
    process_buffer(buffer)

    print("Recording finished")
    wav.write(filename, sample_rate, np.array(audio, dtype=np.int16))
    print(f"Audio saved to {filename}")

    # Call transcription function after recording ends
    transcribe_audio(filename)

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

def real_time_transcription():
    try:
        filename = "real_time_audio.wav"
        record_with_improved_vad(filename)
    except KeyboardInterrupt:
        print("Real-time transcription stopped")

# Run the real-time transcription
real_time_transcription()

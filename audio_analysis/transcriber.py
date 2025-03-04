import os
import glob
from transformers import pipeline

# Directories
AUDIO_DIRECTORY = "data/audio_files"
TRANSCRIPTIONS_DIRECTORY = "transcriptions"

# Ensure the transcriptions folder exists
os.makedirs(TRANSCRIPTIONS_DIRECTORY, exist_ok=True)

# Load ASR Model
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_audio_files():
    """
    Processes audio files in the directory, transcribes them, and saves them as text files.
    """
    audio_file_extensions = ['*.wav', '*.mp3', '*.flac']
    audio_files = [file for ext in audio_file_extensions for file in glob.glob(os.path.join(AUDIO_DIRECTORY, ext))]

    if not audio_files:
        print("No audio files found in directory:", AUDIO_DIRECTORY)
        return

    for audio_file in audio_files:
        print(f"\nProcessing file: {audio_file}")
        try:
            # Transcribe the audio file
            result = asr_pipeline(audio_file, return_timestamps=True)
            transcription = result.get("text", "").strip()

            if not transcription:
                print(f"Warning: Empty transcription for {audio_file}")
                continue

            # Save transcription
            audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
            transcript_path = os.path.join(TRANSCRIPTIONS_DIRECTORY, f"{audio_filename}.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcription)

            print(f"Transcription saved to: {transcript_path}")

        except Exception as e:
            print(f"Error transcribing file {audio_file}: {e}")

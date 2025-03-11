import os
import time
from transformers import pipeline

# Directories
TRANSCRIPTIONS_DIRECTORY = "transcriptions"

# Load ASR Model
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_single_audio(filepath):
    try:
        print(f"Processing file: {filepath}")
        # Transcribe the audio file using the ASR pipeline
        result = asr_pipeline(filepath, return_timestamps=True)
        transcription = result.get("text", "").strip()

        if not transcription:
            print(f"Warning: Empty transcription for {filepath}")
            return None

        # Create a unique filename for the transcript
        audio_filename = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = int(time.time())
        transcript_path = os.path.join(TRANSCRIPTIONS_DIRECTORY, f"{audio_filename}_{timestamp}.txt")
        
        # Save the transcription
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcription)

        print(f"Transcription saved to: {transcript_path}")
        return transcript_path
    except Exception as e:
        print(f"Error transcribing file {filepath}: {e}")
        return None

import os
import time
from transformers import pipeline

# Directory for saving transcriptions
TRANSCRIPTIONS_DIRECTORY = "transcriptions"

# Ensure the transcriptions directory exists
os.makedirs(TRANSCRIPTIONS_DIRECTORY, exist_ok=True)

# Load the ASR model (using OpenAI's Whisper-small model)
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_single_audio(filepath):
    """
    Transcribes the audio file at the given filepath using the ASR pipeline.
    Returns the path to a transcription text file if successful,
    or None if transcription failed.
    """
    try:
        print(f"Processing file: {filepath}")
        # Transcribe the audio file and capture timestamps if needed
        result = asr_pipeline(filepath, return_timestamps=True)
        transcription = result.get("text", "").strip()

        if not transcription:
            print(f"Warning: Empty transcription for {filepath}")
            return None

        # Create a unique transcript filename using the original file name and a timestamp
        audio_filename = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = int(time.time())
        transcript_filename = f"{audio_filename}_{timestamp}.txt"
        transcript_path = os.path.join(TRANSCRIPTIONS_DIRECTORY, transcript_filename)

        # Save the transcription to a file
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcription)

        print(f"Transcription saved to: {transcript_path}")
        return transcript_path
    except Exception as e:
        print(f"Error transcribing file {filepath}: {e}")
        return None

if __name__ == "__main__":
    # Optionally, you can use this block to test your transcription function.
    # Replace "path_to_test_audio_file.mp3" with the path to an actual audio file.
    test_audio_file = "path_to_test_audio_file.mp3"
    print(transcribe_single_audio(test_audio_file))

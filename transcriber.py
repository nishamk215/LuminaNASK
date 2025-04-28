import os
import time
from transformers import pipeline
from transformers import WhisperProcessor

# Directory for saving transcriptions
TRANSCRIPTIONS_DIRECTORY = "transcriptions"
os.makedirs(TRANSCRIPTIONS_DIRECTORY, exist_ok=True)

# Attempt to load the ASR pipeline forcing English output:
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    return_timestamps=True,    
)

# Force the model to transcribe in English
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
asr_pipeline.model.config.forced_decoder_ids = forced_decoder_ids

def transcribe_single_audio(filepath):
    try:
        print(f"Processing file: {filepath}")
        result = asr_pipeline(filepath)
        transcription = result.get("text", "").strip()

        if not transcription:
            print(f"Warning: Empty transcription for {filepath}")
            return None

        audio_filename = os.path.splitext(os.path.basename(filepath))[0]
        timestamp = int(time.time())
        transcript_path = os.path.join(TRANSCRIPTIONS_DIRECTORY, f"{audio_filename}_{timestamp}.txt")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcription)

        print(f"Transcription saved to: {transcript_path}")
        return transcript_path
    except Exception as e:
        print(f"Error transcribing file {filepath}: {e}")
        return None

if __name__ == "__main__":
    test_audio_file = "path_to_test_audio_file.mp3"
    print(transcribe_single_audio(test_audio_file))


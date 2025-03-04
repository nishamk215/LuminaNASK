from transcriber import transcribe_audio_files
from analyzer import analyze_transcriptions

if __name__ == "__main__":
    print("\n--- Running Transcription ---")
    transcribe_audio_files()
    
    print("\n--- Running Text Analysis ---")
    analyze_transcriptions()
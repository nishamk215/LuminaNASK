import os
import glob
import re
import pandas as pd
from transformers import pipeline

def split_into_sentences(text):
    """
    Splits text into sentences using a regular expression that looks for punctuation
    (period, exclamation, or question mark) followed by a space.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def main():
    # Directories
    audio_directory = "data/audio_files"
    transcriptions_directory = "transcriptions"  # Folder to save transcripts and CSVs

    # Create the transcriptions folder if it doesn't exist.
    os.makedirs(transcriptions_directory, exist_ok=True)
    print("Transcriptions directory:", os.path.abspath(transcriptions_directory))

    # Create an ASR pipeline using the open source Whisper model.
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
        # For GPU support, add: device=0
    )

    # Create a zero-shot classification pipeline using Facebook's BART-large-MNLI model.
    classification_pipeline = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    # Define candidate labels for harmful language detection.
    candidate_labels = ["xenophobic language", "misinformation", "neutral"]

    # Gather audio files with supported extensions.
    audio_file_extensions = ['*.wav', '*.mp3', '*.flac']
    audio_files = []
    for ext in audio_file_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_directory, ext)))

    if not audio_files:
        print("No audio files found in directory:", audio_directory)
        return

    # Process each audio file.
    for audio_file in audio_files:
        print(f"\nProcessing file: {audio_file}")
        try:
            # Transcribe the audio file (with timestamp tokens to support longer inputs)
            result = asr_pipeline(audio_file, return_timestamps=True)
            transcription = result.get("text", "")
            print("Full Transcription:")
            print(transcription)

            # Save the full transcription to a text file.
            audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
            transcript_path = os.path.join(transcriptions_directory, f"{audio_filename}.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"Transcription saved to: {transcript_path}")

            # Split the transcription into sentences.
            sentences = split_into_sentences(transcription)
            print(f"Found {len(sentences)} sentences for classification.")

            # List to store classification results for each sentence.
            classification_rows = []

            # Classify each sentence.
            for idx, sentence in enumerate(sentences):
                if sentence:
                    classification_result = classification_pipeline(sentence, candidate_labels)
                    # Build a dictionary for this sentence.
                    row = {"sentence": sentence}
                    # Get the predicted label (highest score)
                    predicted_label = classification_result["labels"][0]
                    row["predicted_label"] = predicted_label
                    # Build a dictionary of candidate scores.
                    candidate_scores = dict(zip(classification_result["labels"], classification_result["scores"]))
                    # Ensure all candidate labels have a score (if not, default to 0).
                    for label in candidate_labels:
                        row[f"{label}_score"] = candidate_scores.get(label, 0)
                    classification_rows.append(row)
                    print(f"Sentence {idx + 1}:")
                    print("  " + sentence)
                    print("  Classification:", row)

            # **** Debugging Block: Check classification rows before saving CSV ****
            print("Number of classified sentences:", len(classification_rows))
            if not classification_rows:
                print("No classification rows were generated. Check your transcript splitting or classification loop.")
            else:
                # Debug: print each classification row.
                for idx, row in enumerate(classification_rows):
                    print(f"Row {idx+1}:", row)
                
                # Convert the classification results to a DataFrame.
                df = pd.DataFrame(classification_rows)
                print("DataFrame created. Preview:")
                print(df.head())

                # Define a CSV file name and path.
                csv_path = os.path.join(transcriptions_directory, f"{audio_filename}_classification.csv")
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"Classification results saved to: {csv_path}")
            # **** End Debugging Block ****

        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")

if __name__ == "__main__":
    main()

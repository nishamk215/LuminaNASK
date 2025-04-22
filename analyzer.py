import os
import glob
import json
import pandas as pd
import re
import requests
from transformers import pipeline

# Directories
TRANSCRIPTIONS_DIRECTORY = "transcriptions"

# Load AI Model for Classification
classification_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
API_URL = "http://127.0.0.1:5000/predict"  # Local API for toxicity classification

CANDIDATE_LABELS = ["xenophobic language", "misinformation", "neutral"]

def split_into_sentences(text):
    """Splits text into sentences using punctuation rules."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def classify_toxicity(text):
    """Sends text to local FastAPI service for toxicity classification."""
    payload = {"text": text}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return {
            "toxicity_scores": result["predictions"],
            "binary_labels": result["labels"],
            "classification": determine_toxicity_level(result["predictions"])
        }
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling toxicity API: {e}")
        return {"classification": "ERROR", "toxicity_scores": {}, "binary_labels": {}}

def determine_toxicity_level(scores):
    """Classifies text into NONE, MILD, HIGH, or MAX based on toxicity levels."""
    thresholds = {
        "toxic": 0.15,
        "severe_toxic": 0.15,
        "obscene": 0.15,
        "threat": 0.15,
        "insult": 0.15,
        "identity_hate": 0.15
    }
    severity_count = sum(1 for label, score in scores.items() if score >= thresholds.get(label, 0))
    if severity_count == 0:
        return "NONE"
    elif severity_count == 1:
        return "MILD"
    elif severity_count == 2:
        return "HIGH"
    else:
        return "MAX"

def analyze_transcription_file(transcript_path):
    """
    Processes a single transcription file, runs analysis on each sentence,
    saves results to an Excel file and JSON file for dashboard display.
    """
    try:
        print(f"\n📄 Analyzing transcription: {transcript_path}")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcription = f.read().strip()

        if not transcription:
            print(f"⚠ Warning: Empty transcription in {transcript_path}")
            return None

        sentences = split_into_sentences(transcription)
        classification_rows = []

        for sentence in sentences:
            if sentence:
                # Toxicity classification via local API
                toxicity_result = classify_toxicity(sentence)

                # Misinformation classification via zero-shot
                classification_result = classification_pipeline(sentence, CANDIDATE_LABELS)
                predicted_label = classification_result["labels"][0]
                candidate_scores = dict(zip(classification_result["labels"], classification_result["scores"]))

                classification_rows.append({
                    "sentence": sentence,
                    "toxicity_level": toxicity_result["classification"],
                    "toxicity_scores": json.dumps(toxicity_result["toxicity_scores"]),
                    "binary_labels": json.dumps(toxicity_result["binary_labels"]),
                    "predicted_label": predicted_label,
                    "xenophobic_score": candidate_scores.get("xenophobic language", 0),
                    "misinformation_score": candidate_scores.get("misinformation", 0),
                    "neutral_score": candidate_scores.get("neutral", 0)
                })

        if not classification_rows:
            print("⚠ No valid sentences to analyze.")
            return None

        df = pd.DataFrame(classification_rows)

        # Define DataFrames
        toxicity_columns = ["sentence", "toxicity_level", "toxicity_scores", "binary_labels"]
        misinformation_columns = ["sentence", "predicted_label", "xenophobic_score", "misinformation_score", "neutral_score"]

        df_toxicity = df[toxicity_columns]
        df_misinformation = df[misinformation_columns]

        # Save Excel with 2 sheets
        excel_path = transcript_path.replace(".txt", "_classification.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            df_toxicity.to_excel(writer, sheet_name="Toxicity", index=False)
            df_misinformation.to_excel(writer, sheet_name="Misinformation", index=False)

        print(f"✅ Excel report saved to: {excel_path}")

        # ✅ Save JSON for dashboard
        try:
            output_json_path = os.path.join("static", "analysis_results.json")
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump([
                    {
                        "sentence": row["sentence"],
                        "predictions": json.loads(row["toxicity_scores"]),
                        "labels": json.loads(row["binary_labels"]),
                        "misinformation": {
                            "xenophobic": row["xenophobic_score"],
                            "misinformation": row["misinformation_score"],
                            "neutral": row["neutral_score"]
                        }
                    }
                    for _, row in df.iterrows()
                ], f, indent=2)

            print(f"📊 Dashboard data saved to: {output_json_path}")
        except Exception as json_err:
            print(f"❌ Error saving dashboard JSON: {json_err}")

        return excel_path

    except Exception as e:
        print(f"❌ Error analyzing file {transcript_path}: {e}")
        return None

def analyze_transcriptions():
    """Batch processing of all .txt files in the transcriptions directory."""
    transcript_files = glob.glob(os.path.join(TRANSCRIPTIONS_DIRECTORY, "*.txt"))
    if not transcript_files:
        print("No transcription files found for analysis.")
        return

    for transcript_file in transcript_files:
        analyze_transcription_file(transcript_file)

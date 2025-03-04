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
API_URL = "http://127.0.0.1:8000/predict"  # Local API for toxicity classification

CANDIDATE_LABELS = ["xenophobic language", "misinformation", "neutral"]

def split_into_sentences(text):
    """ Splits text into sentences using punctuation rules. """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def classify_toxicity(text):
    """ Sends text to local FastAPI service for toxicity classification. """
    payload = {"text": text}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise error for HTTP issues
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
    """ Classifies text into NONE, MILD, HIGH, or MAX based on toxicity levels. """
    thresholds = {
        "toxic": 0.8, "severe_toxic": 0.5, "obscene": 0.7,
        "threat": 0.5, "insult": 0.75, "identity_hate": 0.6
    }
    severity_count = sum(1 for label, score in scores.items() if score >= thresholds.get(label, 0))
    return "NONE" if severity_count == 0 else "MILD" if severity_count == 1 else "HIGH" if severity_count == 2 else "MAX"

def analyze_transcriptions():
    """ Loads transcriptions, analyzes them for harmful content, and saves results as CSV. """
    transcript_files = glob.glob(os.path.join(TRANSCRIPTIONS_DIRECTORY, "*.txt"))
    if not transcript_files:
        print("No transcription files found for analysis.")
        return
    
    for transcript_file in transcript_files:
        print(f"\n📄 Analyzing transcription: {transcript_file}")
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcription = f.read().strip()
            
            if not transcription:
                print(f"⚠ Warning: Empty transcription in {transcript_file}")
                continue
            
            sentences = split_into_sentences(transcription)
            classification_rows = []
            
            for sentence in sentences:
                if sentence:
                    # 🔍 Toxicity API Call
                    toxicity_result = classify_toxicity(sentence)

                    # 🔍 Zero-Shot Classification for Xenophobia & Misinformation
                    classification_result = classification_pipeline(sentence, CANDIDATE_LABELS)
                    predicted_label = classification_result["labels"][0]
                    candidate_scores = dict(zip(classification_result["labels"], classification_result["scores"]))

                    classification_rows.append({
                        "sentence": sentence,
                        "toxicity_level": toxicity_result["classification"],
                        "predicted_label": predicted_label,
                        "xenophobic_score": candidate_scores.get("xenophobic language", 0),
                        "misinformation_score": candidate_scores.get("misinformation", 0),
                        "neutral_score": candidate_scores.get("neutral", 0),
                        "toxicity_scores": json.dumps(toxicity_result["toxicity_scores"]),
                        "binary_labels": json.dumps(toxicity_result["binary_labels"]),
                    })
            
            if classification_rows:
                df = pd.DataFrame(classification_rows)
                csv_path = transcript_file.replace(".txt", "_classification.csv")
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"✅ Analysis results saved to: {csv_path}")

        except Exception as e:
            print(f"❌ Error analyzing file {transcript_file}: {e}")

import os
import glob
import json
import re
import requests
import pandas as pd
from transformers import pipeline
from flask import current_app
# ─── Configuration ───────────────────────────────────────────────────────────
TRANSCRIPTIONS_DIRECTORY = "transcriptions"
API_URL                  = "http://127.0.0.1:5000/audio/predict"
CANDIDATE_LABELS         = ["xenophobic language", "misinformation", "neutral"]

# ─── Pipelines ────────────────────────────────────────────────────────────────
# Zero‑shot pipeline for misinformation/xenophobia/neutral
classification_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def split_into_sentences(text: str) -> list[str]:
    """Split on punctuation+space, keeping non‑empty sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

def classify_toxicity(text: str) -> tuple[dict, dict, str]:
    """Call your FastAPI /audio/predict and return (scores, binary_labels, severity)."""
    payload = {"text": text}
    try:
        resp = requests.post(API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        preds = data["predictions"]
        bins  = data["labels"]
        # Determine severity:
        thresholds = {k: 0.3 for k in bins}
        count = sum(1 for k,v in preds.items() if v >= thresholds[k])
        if count == 0: lvl = "NONE"
        elif count == 1: lvl = "MILD"
        elif count == 2: lvl = "HIGH"
        else: lvl = "MAX"
        return preds, bins, lvl

    except requests.RequestException as e:
        print(f"❌ Error calling toxicity API: {e}")
        return {}, {}, "ERROR"

# ─── Main Analysis Function ──────────────────────────────────────────────────
def analyze_transcription_file(transcript_path: str) -> str | None:
    """
    Reads transcript_path (.txt), splits into sentences, classifies each
    for toxicity and misinformation, then writes an Excel file with two sheets.
    Also writes a JSON file analysis_results.json into the app’s static folder
    for the dashboard. Returns the path to the .xlsx or None on failure.
    """
    if not os.path.exists(transcript_path):
        print(f"File not found: {transcript_path}")
        return None

    text = open(transcript_path, "r", encoding="utf-8").read().strip()
    if not text:
        print(f"Empty transcription: {transcript_path}")
        return None

    sentences = split_into_sentences(text)
    rows = []
    for s in sentences:
        # 1) Toxicity classification (scores + binary flags)
        try:
            tox_scores, tox_bins, _ = classify_toxicity(s)
        except Exception as e:
            print(f"❌ Error classifying toxicity on sentence: {e}")
            tox_scores, tox_bins = {}, {}

        # 2) Zero-shot misinfo classification
        try:
            zero = classification_pipeline(s, CANDIDATE_LABELS)
            mis_scores = dict(zip(zero["labels"], zero["scores"]))
        except Exception as e:
            print(f"❌ Error classifying misinformation on sentence: {e}")
            mis_scores = {lbl: 0.0 for lbl in CANDIDATE_LABELS}

        rows.append({
            "sentence":             s,
            "toxicity_scores":      json.dumps(tox_scores, ensure_ascii=False),
            "binary_labels":        json.dumps(tox_bins,   ensure_ascii=False),
            "xenophobic_score":     mis_scores.get("xenophobic language", 0.0),
            "misinformation_score": mis_scores.get("misinformation",     0.0),
            "neutral_score":        mis_scores.get("neutral",           0.0),
        })

    if not rows:
        print("No sentences to analyze.")
        return None

    # Build DataFrame and write Excel
    df = pd.DataFrame(rows)
    excel_path = transcript_path.replace(".txt", "_classification.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df[["sentence", "toxicity_scores", "binary_labels"]].to_excel(
            writer, sheet_name="Toxicity", index=False
        )
        df[["sentence", "xenophobic_score", "misinformation_score", "neutral_score"]].to_excel(
            writer, sheet_name="Misinformation", index=False
        )
    print(f"✅ Analysis results saved to: {excel_path}")

    # Dump JSON for dashboard into the app’s static folder
    try:
        static_dir = os.path.join(current_app.root_path, "static")
        os.makedirs(static_dir, exist_ok=True)
        output_json_path = os.path.join(static_dir, "analysis_results.json")

        out = []
        for _, row in df.iterrows():
            out.append({
                "sentence":      row["sentence"],
                "predictions":   json.loads(row["toxicity_scores"]),
                "labels":        json.loads(row["binary_labels"]),
                "misinformation": {
                    "xenophobic":     row["xenophobic_score"],
                    "misinformation": row["misinformation_score"],
                    "neutral":        row["neutral_score"]
                }
            })

        with open(output_json_path, "w", encoding="utf-8") as jf:
            json.dump(out, jf, indent=2, ensure_ascii=False)
        print(f"📊 Dashboard data saved to {output_json_path}")
    except Exception as e:
        print(f"❌ Error writing dashboard JSON: {e}")

    return excel_path

# ─── Optional: Batch Mode ────────────────────────────────────────────────────
if __name__ == "__main__":
    files = glob.glob(os.path.join(TRANSCRIPTIONS_DIRECTORY, "*.txt"))
    for f in files:
        analyze_transcription_file(f)

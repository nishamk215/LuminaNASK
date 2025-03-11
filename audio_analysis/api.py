from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.responses import JSONResponse, FileResponse
import os
import glob
import json
import pandas as pd
from analyzer import split_into_sentences, classify_toxicity, determine_toxicity_level

app = FastAPI()

# Load the toxicity classification pipeline using toxic-bert
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

# Define the input data model for predictions
class TextInput(BaseModel):
    text: str

# Toxicity thresholds (adjusted for severe_toxic, for example)
TOXICITY_THRESHOLDS = {
    "toxic": 0.5,
    "severe_toxic": 0.3,
    "obscene": 0.5,
    "threat": 0.5,
    "insult": 0.5,
    "identity_hate": 0.5
}

@app.post("/predict", response_class=JSONResponse)
async def predict_toxicity(input: TextInput):
    try:
        text = input.text
        results = toxicity_model(text)[0]
        predictions = {result['label']: result['score'] for result in results}
        labels = {label: int(score >= TOXICITY_THRESHOLDS[label]) for label, score in predictions.items()}
        return {"predictions": predictions, "labels": labels}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# New endpoint: run analysis on transcriptions and download the CSV file
@app.get("/download_analysis")
async def download_analysis():
    # Folder where transcriptions are stored
    transcriptions_folder = "transcriptions"
    transcript_files = glob.glob(os.path.join(transcriptions_folder, "*.txt"))
    if not transcript_files:
        raise HTTPException(status_code=404, detail="No transcription files found for analysis.")

    all_rows = []
    # Initialize the zero-shot classification pipeline for additional analysis
    classification_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    for transcript_file in transcript_files:
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcription = f.read().strip()
            if not transcription:
                continue
            sentences = split_into_sentences(transcription)
            for sentence in sentences:
                if sentence:
                    # Get toxicity classification via your predict function (or similar logic)
                    toxicity_result = classify_toxicity(sentence)
                    # Run zero-shot classification for xenophobic language and misinformation
                    classification_result = classification_pipeline(sentence, ["xenophobic language", "misinformation", "neutral"])
                    predicted_label = classification_result["labels"][0]
                    candidate_scores = dict(zip(classification_result["labels"], classification_result["scores"]))
                    
                    all_rows.append({
                        "transcript_file": os.path.basename(transcript_file),
                        "sentence": sentence,
                        "toxicity_level": determine_toxicity_level(toxicity_result["toxicity_scores"]),
                        "predicted_label": predicted_label,
                        "xenophobic_score": candidate_scores.get("xenophobic language", 0),
                        "misinformation_score": candidate_scores.get("misinformation", 0),
                        "neutral_score": candidate_scores.get("neutral", 0),
                        "toxicity_scores": json.dumps(toxicity_result["toxicity_scores"]),
                        "binary_labels": json.dumps(toxicity_result["binary_labels"]),
                    })
        except Exception as e:
            print(f"Error analyzing file {transcript_file}: {e}")
            continue

    if not all_rows:
        raise HTTPException(status_code=404, detail="No analysis results to download.")

    df = pd.DataFrame(all_rows)
    output_path = os.path.join(transcriptions_folder, "analysis_results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    return FileResponse(
        output_path,
        media_type="text/csv",
        filename="analysis_results.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)

run this in the terminal before running app.py - uvicorn api:app --reload



from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load the toxicity classification pipeline
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)

# Define the input data model
class TextInput(BaseModel):
    text: str

# Define the toxicity thresholds for binary classification
TOXICITY_THRESHOLDS = {
    "toxic": 0.5,
    "severe_toxic": 0.5,
    "obscene": 0.5,
    "threat": 0.5,
    "insult": 0.5,
    "identity_hate": 0.5
}

@app.post("/predict")
async def predict_toxicity(input: TextInput):
    # Get the text from the request
    text = input.text

    # Get model predictions
    results = toxicity_model(text)[0]

    # Prepare the response
    predictions = {result['label']: result['score'] for result in results}
    labels = {label: int(score >= TOXICITY_THRESHOLDS[label]) for label, score in predictions.items()}

    return {
        "predictions": predictions,
        "labels": labels
    }

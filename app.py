from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import glob
import json
import pandas as pd
import re
import requests
from werkzeug.utils import secure_filename
from transformers import pipeline
from analyzer import analyze_transcription_file
# Directories
UPLOAD_FOLDER = "uploads"
TRANSCRIPTIONS_FOLDER = "transcriptions"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TRANSCRIPTIONS_FOLDER"] = TRANSCRIPTIONS_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)

# ---------- Integrated Toxicity API ----------
# Initialize toxicity classification pipeline using toxic-bert
toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    return_all_scores=True
)
# Define toxicity thresholds
TOXICITY_THRESHOLDS = {
    "toxic": 0.15,
    "severe_toxic": 0.15,
    "obscene": 0.15,
    "threat": 0.15,
    "insult": 0.15,
    "identity_hate": 0.15
}

@app.route("/predict", methods=["POST"])
def predict_toxicity():
    data = request.get_json()
    text = data.get("text", "")
    results = toxicity_model(text)[0]
    predictions = {result['label']: result['score'] for result in results}
    labels = {label: int(score >= TOXICITY_THRESHOLDS[label]) for label, score in predictions.items()}
    return jsonify({"predictions": predictions, "labels": labels})
# ---------- End Toxicity API Integration ----------

# ---------- Other Flask Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_audio():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            return redirect(url_for("select_file"))
    return render_template("upload.html")  # make sure you have an upload.html template

@app.route("/select", methods=["GET"])
def select_file():
    files = [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if f.lower().endswith(('.wav', '.mp3', '.flac','.txt','.mp4'))]
    return render_template("select.html", files=files)

# Transcribe the selected file using your existing transcription function
from transcriber import transcribe_single_audio  # assuming this function exists
@app.route("/transcribe", methods=["POST"])
def transcribe():
    selected_file = request.form.get("selected_file")
    if not selected_file:
        return "No file selected", 400
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], selected_file)
    transcript_path = transcribe_single_audio(filepath)
    if transcript_path:
        return redirect(url_for("analyze"))
    else:
        return "Transcription failed", 500

# Updated analyze route: Run analysis on the transcription file and offer a download
from analyzer import analyze_transcription_file  # We'll write a helper function below

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        transcription_file = request.form.get("transcription")
        if not transcription_file:
            return "No transcription file selected", 400

        transcript_path = os.path.join(app.config["TRANSCRIPTIONS_FOLDER"], transcription_file)

        result = analyze_transcription_file(transcript_path)

        if result:
            return redirect(url_for("dashboard"))  # 👈 Show dashboard after analysis
        else:
            return "Analysis failed", 500

    else:
        transcriptions = [f for f in os.listdir(app.config["TRANSCRIPTIONS_FOLDER"]) if f.endswith(".txt")]
        return render_template("analyze.html", transcriptions=transcriptions)

@app.route("/run_analysis", methods=["POST"])
def run_analysis():
    transcription_file = request.form.get("transcription")
    if not transcription_file:
        return "No transcription file specified", 400
    transcript_path = os.path.join(app.config["TRANSCRIPTIONS_FOLDER"], transcription_file)
    csv_path = analyze_transcription_file(transcript_path)
    if csv_path:
        return send_file(csv_path, as_attachment=True)
    else:
        return "Analysis failed", 500
    
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ---------- End Flask Routes ----------

if __name__ == "__main__":
    app.run(debug=True)

import os
import glob
import json
import pandas as pd
import re
import requests
from flask import render_template, request, redirect, url_for, send_file, jsonify, current_app
from werkzeug.utils import secure_filename
from transformers import pipeline
from analyzer import analyze_transcription_file  # your analyzer helper
from transcriber import transcribe_single_audio    # your transcription helper

from . import audio_bp  # import the blueprint object

# ---------- Integrated Toxicity API ----------
toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    return_all_scores=True
)
TOXICITY_THRESHOLDS = {
    "toxic": 0.3,
    "severe_toxic": 0.3,
    "obscene": 0.3,
    "threat": 0.3,
    "insult": 0.3,
    "identity_hate": 0.3
}

@audio_bp.route("/predict", methods=["POST"])
def predict_toxicity():
    data = request.get_json()
    text = data.get("text", "")
    results = toxicity_model(text)[0]
    predictions = {result['label']: result['score'] for result in results}
    labels = {label: int(score >= TOXICITY_THRESHOLDS[label])
              for label, score in predictions.items()}
    return jsonify({"predictions": predictions, "labels": labels})
# ---------- End Toxicity API ----------

# ---------- Audio Routes ----------

@audio_bp.route("/")
def index():
    return render_template("index.html")

@audio_bp.route("/upload", methods=["GET", "POST"])
def upload_audio():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            return redirect(url_for("audio.select_file"))
    return render_template("upload.html")  # Ensure you have an upload.html template

@audio_bp.route("/select", methods=["GET"])
def select_file():
    files = [f for f in os.listdir(current_app.config["UPLOAD_FOLDER"])
             if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    return render_template("select.html", files=files)

@audio_bp.route("/transcribe", methods=["POST"])
def transcribe():
    selected_file = request.form.get("selected_file")
    if not selected_file:
        return "No file selected", 400
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], selected_file)
    transcript_path = transcribe_single_audio(filepath)
    if transcript_path:
        return redirect(url_for("audio.analyze"))
    else:
        return "Transcription failed", 500

@audio_bp.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        transcription_file = request.form.get("transcription")
        if not transcription_file:
            return "No transcription file selected", 400
        transcript_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], transcription_file)
        csv_path = analyze_transcription_file(transcript_path)
        if csv_path and os.path.exists(csv_path):
            return send_file(csv_path, as_attachment=True)
        else:
            return "Analysis failed", 500
    else:
        # GET: List available transcription files
        transcriptions = [f for f in os.listdir(current_app.config["TRANSCRIPTIONS_FOLDER"]) if f.endswith(".txt")]
        return render_template("analyze.html", transcriptions=transcriptions)

@audio_bp.route("/run_analysis", methods=["POST"])
def run_analysis():
    transcription_file = request.form.get("transcription")
    if not transcription_file:
        return "No transcription file specified", 400
    transcript_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], transcription_file)
    csv_path = analyze_transcription_file(transcript_path)
    if csv_path:
        return send_file(csv_path, as_attachment=True)
    else:
        return "Analysis failed", 500

import os
import ffmpeg
import whisper
import json
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from nltk.tokenize import sent_tokenize
from datetime import datetime
import pandas as pd
from transformers import pipeline
import docx2txt
from docx import Document
from PyPDF2 import PdfReader

# Flask app setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load models
whisper_model = whisper.load_model("base")
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)

# Toxicity thresholds
TOXICITY_THRESHOLDS = {
    "toxic": 0.3,
    "severe_toxic": 0.3,
    "obscene": 0.3,
    "threat": 0.3,
    "insult": 0.3,
    "identity_hate": 0.3
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".mp4", ".avi", ".mov", ".pdf", ".docx", ".xlsx", ".txt"}

def allowed_file(filename):
    ext = '.' + filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def transcribe_file(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    try:
        if extension in [".mp3", ".wav", ".flac", ".mp4", ".avi", ".mov"]:
            return whisper_model.transcribe(file_path)["text"]
        elif extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif extension == ".docx":
            return docx2txt.process(file_path)
        elif extension == ".pdf":
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        else:
            return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def analyze_text(text):
    sentences = sent_tokenize(text)
    results = []
    for sentence in sentences:
        try:
            toxicity_results = toxicity_model(sentence)[0]
            predictions = {res['label']: res['score'] for res in toxicity_results}
            labels = {label: int(score >= TOXICITY_THRESHOLDS[label]) for label, score in predictions.items()}
            results.append({
                "sentence": sentence,
                "predictions": predictions,
                "labels": labels
            })
        except Exception as e:
            print(f"Error analyzing sentence '{sentence}': {e}")
    return results

def generate_report(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    downloads_folder = "Downloads"
    os.makedirs(downloads_folder, exist_ok=True)
    filename = f"analysis_report_{timestamp}.csv"
    csv_path = os.path.join(downloads_folder, filename)

    categories = TOXICITY_THRESHOLDS.keys()
    structured_data = []

    for result in results:
        row = {"Sentence": result["sentence"]}
        for category in categories:
            row[f"{category}_score"] = result["predictions"].get(category, 0)
            row[f"{category}_label"] = result["labels"].get(category, 0)
        structured_data.append(row)

    try:
        df = pd.DataFrame(structured_data)
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        print(f"Error saving report: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_existing", methods=["POST"])
def process_existing():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process the uploaded file
        transcript = transcribe_file(file_path)
        if not transcript:
            return "Error processing file", 500

        # Save transcript as .docx
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(filename)[0]
        transcript_filename = f"{base_filename}_transcript_{timestamp}.docx"
        transcript_path = os.path.join("static", transcript_filename)

        doc = Document()
        doc.add_paragraph(transcript)
        doc.save(transcript_path)

        # Analyze text
        analysis_results = analyze_text(transcript)
        if not analysis_results:
            return "Error analyzing text", 500

        # Save results for dashboard
        analysis_results_path = os.path.join("static", "analysis_results.json")
        with open(analysis_results_path, "w") as f:
            json.dump(analysis_results, f, indent=4)

        # Generate downloadable CSV report
        report_path = generate_report(analysis_results)
        report_filename = os.path.basename(report_path) if report_path else None
        report_url = f"/download/{report_filename}" if report_filename else None

        return render_template(
            "results.html",
            message="Processing completed!",
            report_url=report_url,
            transcript=transcript,
            transcript_url=f"/static/{transcript_filename}"
        )

    return "Invalid file format", 400

@app.route("/download/<filename>")
def download_report(filename):
    downloads_folder = "Downloads"
    file_path = os.path.join(downloads_folder, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404

@app.route("/dashboard")
def dashboard():
    try:
        with open("static/analysis_results.json", "r") as file:
            analysis_data = json.load(file)
    except Exception as e:
        print(f"Error loading analysis results: {e}")
        analysis_data = []
    return render_template("dashboard.html", analysis_data=analysis_data)

@app.route("/predict", methods=["POST"])
def predict_toxicity():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    results = toxicity_model(text)[0]
    predictions = {res['label']: res['score'] for res in results}
    labels = {label: int(score >= TOXICITY_THRESHOLDS[label]) for label, score in predictions.items()}

    return jsonify({"predictions": predictions, "labels": labels})

if __name__ == "__main__":
    app.run(debug=True)

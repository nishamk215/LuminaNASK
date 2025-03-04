import os
import ffmpeg
import whisper
import requests
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from nltk.tokenize import sent_tokenize
from datetime import datetime
import pandas as pd

# Flask app setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load Whisper model for speech-to-text
whisper_model = whisper.load_model("base")

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"  # Update if hosted elsewhere

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    try:
        # Transcribe directly from video file
        transcript = whisper_model.transcribe(video_path)["text"]
        return transcript
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

def analyze_text(text):
    sentences = sent_tokenize(text)
    results = []
    
    for sentence in sentences:
        try:
            response = requests.post(API_URL, json={"text": sentence})
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "sentence": sentence,
                    "predictions": data["predictions"],
                    "labels": data["labels"]
                })
            else:
                print(f"Error from API: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error analyzing sentence '{sentence}': {e}")
    
    return results

def generate_report(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    downloads_folder = "Downloads"
    os.makedirs(downloads_folder, exist_ok=True)  # Ensure the folder exists
    filename = f"analysis_report_{timestamp}.csv"
    csv_path = os.path.join(downloads_folder, filename)
    
    try:
        pd.DataFrame(results).to_csv(csv_path, index=False)
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
        file.save(file_path)  # Save the newly uploaded file

        # Process the uploaded file
        transcript = process_video(file_path)
        if not transcript:
            return "Error processing video", 500

        analysis_results = analyze_text(transcript)
        if not analysis_results:
            return "Error analyzing text", 500

        report_path = generate_report(analysis_results)
        if report_path:
            report_filename = os.path.basename(report_path)
            report_url = f"/download/{report_filename}"
        else:
            report_url = None

        return render_template("results.html", message="Processing completed!", report_url=report_url, transcript=transcript)

    return "Invalid file format", 400

@app.route("/download/<filename>")
def download_report(filename):
    downloads_folder = "Downloads"  # Ensure this matches where reports are saved
    file_path = os.path.join(downloads_folder, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)

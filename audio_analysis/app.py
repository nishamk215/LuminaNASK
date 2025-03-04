from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from transcriber import transcribe_audio_files
from analyzer import analyze_transcriptions

UPLOAD_FOLDER = "uploads"
TRANSCRIPTIONS_FOLDER = "transcriptions"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TRANSCRIPTIONS_FOLDER"] = TRANSCRIPTIONS_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Run transcription
        transcribe_audio_files()
        
        return redirect(url_for("analyze"))

@app.route("/analyze")
def analyze():
    transcriptions = [f for f in os.listdir(TRANSCRIPTIONS_FOLDER) if f.endswith(".txt")]
    return render_template("analyze.html", transcriptions=transcriptions)

@app.route("/run_analysis", methods=["POST"])
def run_analysis():
    analyze_transcriptions()
    return render_template("completed.html")

if __name__ == "__main__":
    app.run(debug=True)

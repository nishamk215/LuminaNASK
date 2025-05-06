import os
import json
from flask import (
    Blueprint, render_template, request,
    redirect, url_for, send_file, current_app
)
from werkzeug.utils import secure_filename
from transformers import pipeline
import fitz                   # PyMuPDF for PDF text extraction
from docx import Document     # python-docx for .docx extraction
from moviepy.video.io.VideoFileClip import VideoFileClip

from transcriber import transcribe_single_audio
from analyzer import analyze_transcription_file

audio_bp = Blueprint("audio", __name__, template_folder="templates")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

hf_translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-mul-en",
)

def translate_to_english(text: str) -> str:
    out_chunks = []
    for chunk in chunk_text(text, chunk_size=400):
        try:
            translated = hf_translator(chunk, max_length=500)[0]["translation_text"]
            out_chunks.append(translated)
        except Exception as e:
            print(f"⚠️ translation failed for chunk (len={len(chunk)}): {e}")
            out_chunks.append(chunk)
    return " ".join(out_chunks)

def extract_audio_from_video(video_path: str) -> str:
    base, _ = os.path.splitext(video_path)
    audio_path = f"{base}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    clip.close()
    return audio_path

# ─── Toxicity Classifier ────────────────────────────────────────────────────

toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    return_all_scores=True,
)
TOXICITY_THRESHOLDS = {
    "toxic": 0.3,
    "severe_toxic": 0.3,
    "obscene": 0.3,
    "threat": 0.3,
    "insult": 0.3,
    "identity_hate": 0.3,
}

def classify_toxicity(text: str):
    scores_list, bins_list = [], []
    for chunk in chunk_text(text, chunk_size=400):
        try:
            result = toxicity_model(chunk, truncation=True, max_length=512)[0]
            scores = {r["label"]: r["score"] for r in result}
            bins   = {lbl: int(scores[lbl] >= TOXICITY_THRESHOLDS[lbl]) for lbl in scores}
        except Exception as e:
            print(f"⚠️ Error classifying chunk: {e}")
            scores = {lbl: 0.0 for lbl in TOXICITY_THRESHOLDS}
            bins   = {lbl: 0    for lbl in TOXICITY_THRESHOLDS}
        scores_list.append(scores)
        bins_list.append(bins)
    avg_scores = {lbl: sum(s[lbl] for s in scores_list) / len(scores_list) for lbl in TOXICITY_THRESHOLDS}
    agg_labels = {lbl: int(any(b[lbl] for b in bins_list)) for lbl in TOXICITY_THRESHOLDS}
    return avg_scores, agg_labels

# ─── Audio Upload & Transcription ───────────────────────────────────────────

@audio_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html", active="home")

@audio_bp.route("/audio", methods=["GET","POST"])
def audio():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            return "Please select an audio file", 400

        fn      = secure_filename(f.filename)
        in_path = os.path.join(current_app.config["UPLOAD_FOLDER"], fn)
        f.save(in_path)

        raw_path = transcribe_single_audio(in_path)
        if not raw_path:
            return "Transcription failed", 500

        raw_text = open(raw_path, "r", encoding="utf-8").read().strip()
        en_text  = translate_to_english(raw_text)

        base    = os.path.splitext(os.path.basename(raw_path))[0]
        en_fn   = f"{base}_en.txt"
        en_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], en_fn)
        with open(en_path, "w", encoding="utf-8") as outf:
            outf.write(en_text)

        xlsx = analyze_transcription_file(en_path)
        if not xlsx:
            return "Analysis failed", 500

        return redirect(url_for("audio.dashboard", excel=os.path.basename(xlsx)))

    return render_template("audio.html", active="audio")

@audio_bp.route("/transcribe", methods=["POST"])
def transcribe():
    sel = request.form.get("selected_file")
    if not sel:
        return "No file selected", 400

    raw_path = transcribe_single_audio(os.path.join(current_app.config["UPLOAD_FOLDER"], sel))
    if not raw_path:
        return "Transcription failed", 500

    raw_text = open(raw_path, "r", encoding="utf-8").read().strip()
    en_text  = translate_to_english(raw_text)

    base    = os.path.splitext(os.path.basename(raw_path))[0]
    en_fn   = f"{base}_en.txt"
    en_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], en_fn)
    with open(en_path, "w", encoding="utf-8") as outf:
        outf.write(en_text)

    xlsx = analyze_transcription_file(en_path)
    if not xlsx:
        return "Analysis failed", 500

    return redirect(url_for("audio.dashboard", excel=os.path.basename(xlsx)))

@audio_bp.route("/predict", methods=["POST"])
def predict_toxicity():
    data = request.get_json()
    if not data or "text" not in data:
        return {"error": "No text provided"}, 400
    try:
        scores, bins = classify_toxicity(data["text"])
        return {"predictions": scores, "labels": bins}
    except Exception as e:
        return {"error": str(e)}, 500

# ─── Video Upload & Analysis ─────────────────────────────────────────────────

@audio_bp.route("/video", methods=["GET","POST"])
def video_upload():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            return "Please upload a video file", 400

        fn         = secure_filename(f.filename)
        video_path = os.path.join(current_app.config["UPLOAD_FOLDER"], fn)
        f.save(video_path)

        audio_path = extract_audio_from_video(video_path)

        raw_path = transcribe_single_audio(audio_path)
        if not raw_path:
            return "Transcription failed", 500

        raw_text = open(raw_path, "r", encoding="utf-8").read().strip()
        en_text  = translate_to_english(raw_text)

        base    = os.path.splitext(os.path.basename(raw_path))[0]
        en_fn   = f"{base}_en.txt"
        en_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], en_fn)
        with open(en_path, "w", encoding="utf-8") as outf:
            outf.write(en_text)

        xlsx = analyze_transcription_file(en_path)
        if not xlsx:
            return "Analysis failed", 500

        return redirect(url_for("audio.dashboard", excel=os.path.basename(xlsx)))

    return render_template("video.html", active="video")

# ─── Text Upload & Analysis ─────────────────────────────────────────────────

@audio_bp.route("/text", methods=["GET","POST"])
def text_upload():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            return "Please upload a file", 400

        fn    = secure_filename(f.filename)
        upath = os.path.join(current_app.config["UPLOAD_FOLDER"], fn)
        f.save(upath)

        ext = fn.rsplit(".", 1)[1].lower()
        try:
            if ext == "txt":
                with open(upath, "r", encoding="utf-8") as rd:
                    raw_text = rd.read()
            elif ext == "pdf":
                pdf = fitz.open(upath)
                raw_text = "".join(p.get_text() for p in pdf)
            elif ext == "docx":
                doc = Document(upath)
                raw_text = "\n".join(p.text for p in doc.paragraphs)
            else:
                return "Unsupported file type", 400
        except Exception as e:
            return f"Error extracting text: {e}", 500

        en_text = translate_to_english(raw_text)

        base    = os.path.splitext(fn)[0]
        en_fn   = f"{base}_en.txt"
        en_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], en_fn)
        with open(en_path, "w", encoding="utf-8") as outf:
            outf.write(en_text)

        xlsx = analyze_transcription_file(en_path)
        if not xlsx:
            return "Analysis failed", 500

        return redirect(url_for("audio.dashboard", excel=os.path.basename(xlsx)))

    return render_template("text.html", active="text")

# ─── Excel Export Analysis ───────────────────────────────────────────────────

@audio_bp.route("/dashboard", methods=["GET"])
def dashboard():
    # grab the “excel” query-param
    excel = request.args.get("excel")
    # pass it in as 'excel' (not 'excel_file') so your dashboard.html sees it
    return render_template("dashboard.html", active="dashboard", excel=excel)

@audio_bp.route("/download_excel")
def download_excel():
    fn     = request.args.get("excel")
    folder = current_app.config["TRANSCRIPTIONS_FOLDER"]
    path   = os.path.join(folder, fn)
    if not fn or not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)

@audio_bp.route("/about", methods=["GET"])
def about():
    return render_template("about.html", active="about")

@audio_bp.route("/team", methods=["GET"])
def team():
    return render_template("team.html", active="team")

# blueprints/audio/routes.py

import os
from flask import (
    Blueprint, render_template, request,
    redirect, url_for, send_file, current_app
)
from werkzeug.utils import secure_filename
from transformers import pipeline
import fitz                   # PyMuPDF for PDF text extraction
from docx import Document     # python-docx for .docx extraction

from transcriber import transcribe_single_audio
from analyzer    import analyze_transcription_file

audio_bp = Blueprint("audio", __name__, template_folder="templates")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400):
    """Break long text into manageable chunks for classification."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# HF multi‑language → English translator
hf_translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-mul-en",
)

def translate_to_english(text: str) -> str:
    """Translate text into English, chunk by chunk."""
    out = []
    for chunk in chunk_text(text, chunk_size=4000):
        try:
            out.append(hf_translator(chunk)[0]["translation_text"])
        except Exception:
            out.append(chunk)
    return " ".join(out)

# ─── Toxicity Classifier ─────────────────────────────────────────────────────

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
    "identity_hate": 0.3
}

def classify_toxicity(text: str):
    """
    Splits text into ~400‑char chunks, runs the toxicity model on each,
    then returns:
      - avg_scores: average score for each label
      - agg_labels: 1 if any chunk exceeded threshold for that label
    """
    scores_list = []
    bins_list   = []
    for chunk in chunk_text(text, chunk_size=400):
        res    = toxicity_model(chunk)[0]
        scores = {r["label"]: r["score"] for r in res}
        bins   = {lbl: int(scores[lbl] >= TOXICITY_THRESHOLDS[lbl]) for lbl in scores}
        scores_list.append(scores)
        bins_list.append(bins)

    # average scores
    avg_scores = {
        lbl: sum(s[lbl] for s in scores_list) / len(scores_list)
        for lbl in TOXICITY_THRESHOLDS
    }
    # any‑true labels
    agg_labels = {
        lbl: int(any(b[lbl] for b in bins_list))
        for lbl in TOXICITY_THRESHOLDS
    }

    return avg_scores, agg_labels


# ─── Audio Upload & Transcription ───────────────────────────────────────────

@audio_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@audio_bp.route("/upload", methods=["GET","POST"])
def upload_audio():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            return redirect(request.url)
        fn   = secure_filename(f.filename)
        dest = os.path.join(current_app.config["UPLOAD_FOLDER"], fn)
        f.save(dest)
        return redirect(url_for("audio.select_file"))
    return render_template("upload.html")

@audio_bp.route("/select", methods=["GET"])
def select_file():
    files = [
        fn for fn in os.listdir(current_app.config["UPLOAD_FOLDER"])
        if fn.lower().endswith((".wav", ".mp3", ".flac"))
    ]
    return render_template("select.html", files=files)

@audio_bp.route("/transcribe", methods=["POST"])
def transcribe():
    sel = request.form.get("selected_file")
    if not sel:
        return "No file selected", 400

    # 1) Raw transcription
    raw_path = transcribe_single_audio(
        os.path.join(current_app.config["UPLOAD_FOLDER"], sel)
    )
    if not raw_path:
        return "Transcription failed", 500

    # 2) Read & translate
    raw_text = open(raw_path, "r", encoding="utf-8").read().strip()
    en_text  = translate_to_english(raw_text)

    # 3) Save English transcript
    base    = os.path.splitext(os.path.basename(raw_path))[0]
    en_fn   = f"{base}_en.txt"
    en_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], en_fn)
    with open(en_path, "w", encoding="utf-8") as outf:
        outf.write(en_text)

    # 4) Classify toxicity & show results
    scores, bins = classify_toxicity(en_text)
    return render_template(
        "results.html",
        source_type="Audio File",
        original=raw_text,
        translated=en_text,
        predictions=scores,
        labels=bins,
        transcription_filename=en_fn   # ← now passed in!
    )

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

        # Extract raw_text...
        ext = fn.rsplit(".", 1)[1].lower()
        if ext == "txt":
            raw_text = open(upath, "r", encoding="utf-8").read()
        elif ext == "pdf":
            pdf = fitz.open(upath)
            raw_text = "".join(p.get_text() for p in pdf)
        elif ext == "docx":
            doc = Document(upath)
            raw_text = "\n".join(p.text for p in doc.paragraphs)
        else:
            return "Unsupported file type", 400

        # Translate & classify
        en_text      = translate_to_english(raw_text)
        scores, bins = classify_toxicity(en_text)

        # Save the English transcript so it can be downloaded as Excel
        base    = fn.rsplit(".", 1)[0]
        en_fn   = f"{base}_en.txt"
        en_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], en_fn)
        with open(en_path, "w", encoding="utf-8") as outf:
            outf.write(en_text)

        return render_template(
            "results.html",
            source_type=f"{ext.upper()} Document",
            original=raw_text,
            translated=en_text,
            predictions=scores,
            labels=bins,
            transcription_filename=en_fn   # ← pass it here, too!
        )

    return render_template("text.html")


# ─── Excel Export Analysis ───────────────────────────────────────────────────
@audio_bp.route("/analyze", methods=["GET", "POST"])
def analyze():
    # — POST: user submitted the form —
    if request.method == "POST":
        tf = request.form.get("transcription")
        if not tf:
            return "No transcription selected", 400

        path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], tf)
        xlsx = analyze_transcription_file(path)
        if xlsx and os.path.exists(xlsx):
            return send_file(xlsx, as_attachment=True)
        return "Analysis failed", 500

    # — GET: maybe a direct download link? —
    direct = request.args.get("transcription")
    if direct:
        path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], direct)
        xlsx = analyze_transcription_file(path)
        if xlsx and os.path.exists(xlsx):
            return send_file(xlsx, as_attachment=True)
        return "Analysis failed", 500

    # — GET: no download request, render the form —
    folder = current_app.config["TRANSCRIPTIONS_FOLDER"]
    files  = [fn for fn in os.listdir(folder) if fn.endswith("_en.txt")]
    return render_template("analyze.html", transcriptions=files)
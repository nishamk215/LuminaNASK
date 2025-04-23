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
    """
    Translate arbitrary‐length text into English by:
      1. Splitting it into ~400‐char chunks (to avoid >512 token errors)
      2. Running each chunk through the Marian model with max_length=500
    """
    out_chunks = []
    for chunk in chunk_text(text, chunk_size=400):
        try:
            translated = hf_translator(chunk, max_length=500)[0]["translation_text"]
            out_chunks.append(translated)
        except Exception as e:
            # Log the error and fall back to the original chunk
            print(f"⚠️ translation failed for chunk (len={len(chunk)}): {e}")
            out_chunks.append(chunk)
    return " ".join(out_chunks)


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
    "identity_hate": 0.3
}

def classify_toxicity(text: str):
    """
    Splits text into ~400‑char chunks, classifies each with truncation to avoid
    >512 token errors, then returns:
      - avg_scores: average score for each label
      - agg_labels: 1 if any chunk exceeded its threshold
    """
    scores_list = []
    bins_list   = []

    for chunk in chunk_text(text, chunk_size=400):
        try:
            # enforce truncation and max_length
            result = toxicity_model(
                chunk,
                truncation=True,
                max_length=512
            )[0]
            scores = {r["label"]: r["score"] for r in result}
            bins   = {
                lbl: int(scores[lbl] >= TOXICITY_THRESHOLDS[lbl])
                for lbl in scores
            }
        except Exception as e:
            print(f"⚠️ Error classifying chunk ({chunk[:30]}…): {e}")
            # fallback to zeros if something goes wrong
            scores = {lbl: 0.0 for lbl in TOXICITY_THRESHOLDS}
            bins   = {lbl: 0    for lbl in TOXICITY_THRESHOLDS}

        scores_list.append(scores)
        bins_list.append(bins)

    # average each label’s score over all chunks
    avg_scores = {
        lbl: sum(s[lbl] for s in scores_list) / len(scores_list)
        for lbl in TOXICITY_THRESHOLDS
    }
    # flag label=1 if any chunk had it
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

@audio_bp.route("/predict", methods=["POST"])
def predict_toxicity():
    """API endpoint for toxicity prediction"""
    data = request.json
    if not data or "text" not in data:
        return {"error": "No text provided"}, 400
    
    try:
        scores, bins = classify_toxicity(data["text"])
        return {
            "predictions": scores,
            "labels": bins
        }
    except Exception as e:
        return {"error": str(e)}, 500

# ─── Text Upload & Analysis ─────────────────────────────────────────────────

@audio_bp.route("/text", methods=["GET","POST"])
def text_upload():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            return "Please upload a file", 400

        # 1) save upload
        fn    = secure_filename(f.filename)
        upath = os.path.join(current_app.config["UPLOAD_FOLDER"], fn)
        f.save(upath)

        # 2) extract raw text
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

        # 3) translate to English
        en_text = translate_to_english(raw_text)

        # 4) save the English transcript for analysis
        base    = os.path.splitext(fn)[0]
        en_fn   = f"{base}_en.txt"
        en_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], en_fn)
        with open(en_path, "w", encoding="utf-8") as outf:
            outf.write(en_text)

        # 5) run the full per‑sentence analysis  
        #     (splits on punctuation, classifies toxicity + zero‑shot labels)
        xlsx = analyze_transcription_file(en_path)
        if xlsx and os.path.exists(xlsx):
            # force download of the two-sheet Excel
            return send_file(xlsx, as_attachment=True)

        return "Analysis failed", 500

    # GET → show upload form
    return render_template("text.html")

# ─── Excel Export Analysis ───────────────────────────────────────────────────
# blueprints/audio/routes.py

@audio_bp.route("/analyze", methods=["GET", "POST"])
def analyze():
    # If it's a POST from the form, we expect request.form["transcription"]
    if request.method == "POST":
        tf = request.form.get("transcription")
    else:
        # Otherwise, maybe it's our direct-download link using ?transcription=…
        tf = request.args.get("transcription")

    if tf:
        txt_path = os.path.join(current_app.config["TRANSCRIPTIONS_FOLDER"], tf)
        # build the expected .xlsx filename rather than re‑running
        xlsx = txt_path.replace(".txt", "_classification.xlsx")
        # if we haven’t yet run the analysis, do it now
        if not os.path.exists(xlsx):
            xlsx = analyze_transcription_file(txt_path)
        # finally, send it
        if xlsx and os.path.exists(xlsx):
            return send_file(xlsx, as_attachment=True)
        else:
            return "Analysis failed", 500

    # No transcription specified? show the form to pick one.
    folder = current_app.config["TRANSCRIPTIONS_FOLDER"]
    files  = [fn for fn in os.listdir(folder) if fn.endswith("_en.txt")]
    return render_template("analyze.html", transcriptions=files)

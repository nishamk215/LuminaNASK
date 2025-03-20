import os, subprocess
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from deep_translator import GoogleTranslator
import fitz, openpyxl
from docx import Document
import speech_recognition as sr

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'mp4', 'mp3', 'wav', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def chunk_text(text, chunk_size=4500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def translate_to_english(text):
    chunks = chunk_text(text)
    translated = []
    for chunk in chunks:
        translated.append(GoogleTranslator(source='auto', target='en').translate(chunk))
    return ' '.join(translated)

def extract_text(filepath, ext):
    if ext == 'pdf':
        return ''.join(page.get_text() for page in fitz.open(filepath))
    if ext == 'docx':
        return '\n'.join(para.text for para in Document(filepath).paragraphs)
    if ext == 'xlsx':
        ws = openpyxl.load_workbook(filepath).active
        return '\n'.join('\t'.join(str(cell.value or '') for cell in row) for row in ws.iter_rows())
    if ext in ['mp3', 'wav']:
        r = sr.Recognizer()
        with sr.AudioFile(filepath) as src:
            audio = r.record(src)
        return r.recognize_google(audio, language='auto')
    if ext == 'mp4':
        audio_path = filepath + '.wav'
        subprocess.run(['ffmpeg', '-y', '-i', filepath, '-ar', '16000', '-ac', '1', audio_path])
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as src:
            audio = r.record(src)
        os.remove(audio_path)
        return r.recognize_google(audio, language='auto')
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = filename.rsplit('.', 1)[1].lower()
            try:
                extracted = extract_text(filepath, ext)
                translated = translate_to_english(extracted)
            except Exception as e:
                return f"Processing error: {e}", 500

            output_file = f"translated_{os.path.splitext(filename)[0]}.txt"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated)

            return redirect(url_for('download', filename=output_file))
        return "Invalid file type", 400
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


import os
from flask import Flask, render_template
# Import the audio blueprint from your blueprints folder
from blueprints.audio import audio_bp

app = Flask(__name__)

# Set up directories for uploads and transcriptions
UPLOAD_FOLDER = "uploads"
TRANSCRIPTIONS_FOLDER = "transcriptions"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TRANSCRIPTIONS_FOLDER"] = TRANSCRIPTIONS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)

from blueprints.audio.routes import audio_bp
app.register_blueprint(audio_bp, url_prefix="/audio")

# Optional: Create a global index route that points to your audio index page
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

# blueprints/audio/__init__.py
from flask import Blueprint

audio_bp = Blueprint('audio', __name__, template_folder='templates')

# Import routes last to avoid circular imports
from . import routes

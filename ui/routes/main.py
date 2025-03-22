from flask import Blueprint, render_template, jsonify

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@main_bp.route('/health')
def health():
    """Simple UI health check endpoint."""
    ui_health = {"status": "healthy"}
    return jsonify({"ui": ui_health})
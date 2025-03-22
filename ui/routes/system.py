from flask import Blueprint, render_template, jsonify, request
from ui.api.system_api import SystemAPI

system_bp = Blueprint('system', __name__)
system_api = SystemAPI()

@system_bp.route('/systeminfo', methods=['GET'])
def systeminfo_page():
    """Render the system information page."""
    return render_template('systeminfo.html')

@system_bp.route('/api/chroma-info', methods=['GET'])
def chroma_info():
    """Get information about ChromaDB."""
    result = system_api.get_chroma_info()
    return jsonify(result)

@system_bp.route('/api/terms', methods=['GET'])
def get_terms():
    """Get classification terms from the API."""
    result = system_api.get_terms()
    return jsonify(result)

@system_bp.route('/api/refresh-terms', methods=['POST'])
def refresh_terms():
    """Refresh classification terms from the API."""
    result = system_api.refresh_terms()
    return jsonify(result)

@system_bp.route('/api/health', methods=['GET'])
def api_health():
    """Get full health status from the API."""
    health_data = system_api.get_health()
    return jsonify({"api": health_data})

@system_bp.route('/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Proxy for the job status API endpoint."""
    result = system_api.get_job_status(job_id)
    return jsonify(result)

@system_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """Proxy for the list jobs API endpoint."""
    result = system_api.list_jobs()
    return jsonify(result)
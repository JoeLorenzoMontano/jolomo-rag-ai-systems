from flask import Blueprint, render_template, request, jsonify
from ui.api.document_api import DocumentAPI

documents_bp = Blueprint('documents', __name__)
document_api = DocumentAPI()

@documents_bp.route('/process', methods=['GET'])
def process_page():
    """Render the document processing page."""
    return render_template('process.html')

@documents_bp.route('/process-documents', methods=['POST'])
def process_documents():
    """Proxy for the document processing API endpoint."""
    # Get chunking parameters from the form
    chunk_size = request.form.get('chunk_size')
    min_size = request.form.get('min_size')
    overlap = request.form.get('overlap')
    enable_chunking = request.form.get('enable_chunking')
    enhance_chunks = request.form.get('enhance_chunks')
    
    # Build query parameters
    params = {}
    if chunk_size and chunk_size.isdigit():
        params['chunk_size'] = int(chunk_size)
    if min_size and min_size.isdigit():
        params['min_size'] = int(min_size)
    if overlap and overlap.isdigit():
        params['overlap'] = int(overlap)
    if enable_chunking is not None:
        params['enable_chunking'] = enable_chunking.lower() == 'true'
    if enhance_chunks is not None:
        params['enhance_chunks'] = enhance_chunks.lower() == 'true'
    
    result = document_api.process_documents(chunking_params=params)
    return jsonify(result)

@documents_bp.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Upload a file to the API."""
    # Get the file from the request
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"})
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"})
        
    # Check if process_immediately is set
    process_immediately = request.form.get('process_immediately', 'false').lower() == 'true'
    
    # Read file content
    file_data = file.read()
    
    result = document_api.upload_file(
        file_data=file_data,
        filename=file.filename,
        content_type=file.content_type,
        process_immediately=process_immediately
    )
    
    return jsonify(result)

@documents_bp.route('/api/clear-db', methods=['POST'])
def clear_database():
    """Clear the database."""
    result = document_api.clear_database()
    return jsonify(result)

@documents_bp.route('/chunks', methods=['GET'])
def chunks_page():
    """Render the chunks explorer page."""
    return render_template('chunks.html')

@documents_bp.route('/api/chunks', methods=['GET'])
def get_chunks():
    """Get document chunks from ChromaDB."""
    # Get parameters
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    filename = request.args.get('filename', '')
    content = request.args.get('content', '')
    
    result = document_api.get_chunks(
        limit=limit,
        offset=offset,
        filename=filename,
        content=content
    )
    
    return jsonify(result)
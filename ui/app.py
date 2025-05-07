from flask import Flask, render_template, request, jsonify
import requests
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Get API URL from environment variable or use default
API_URL = os.getenv("API_URL", "http://api:8000")
logging.info(f"Using API URL: {API_URL}")

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Check health of both UI and API"""
    ui_health = {"status": "healthy"}
    
    # Check API health
    try:
        api_response = requests.get(f"{API_URL}/health", timeout=5)
        api_health = api_response.json()
    except Exception as e:
        logging.error(f"Error connecting to API: {e}")
        api_health = {"status": "error", "message": str(e)}
    
    return jsonify({
        "ui": ui_health,
        "api": api_health
    })

@app.route('/process', methods=['GET'])
def process_page():
    """Render the document processing page"""
    return render_template('process.html')

@app.route('/process-documents', methods=['POST'])
def process_documents():
    """Proxy for the document processing API endpoint"""
    # Get chunking parameters from the form
    chunk_size = request.form.get('chunk_size')
    min_size = request.form.get('min_size')
    overlap = request.form.get('overlap')
    enable_chunking = request.form.get('enable_chunking')
    enhance_chunks = request.form.get('enhance_chunks')
    generate_questions = request.form.get('generate_questions')
    max_questions_per_chunk = request.form.get('max_questions_per_chunk')
    
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
    if generate_questions is not None:
        params['generate_questions'] = generate_questions.lower() == 'true'
    if max_questions_per_chunk and max_questions_per_chunk.isdigit():
        params['max_questions_per_chunk'] = int(max_questions_per_chunk)
    
    try:
        # Call the API - now it returns immediately with a job ID
        response = requests.post(f"{API_URL}/process", params=params, timeout=10)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error starting document processing job: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Proxy for the job status API endpoint"""
    try:
        # Call the API to get job status
        response = requests.get(f"{API_URL}/job/{job_id}", timeout=5)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error getting job status: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """Proxy for the list jobs API endpoint"""
    try:
        # Call the API to list all jobs
        response = requests.get(f"{API_URL}/jobs", timeout=5)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error listing jobs: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/query', methods=['GET'])
def query_page():
    """Render the advanced query page"""
    return render_template('query.html')

@app.route('/chat', methods=['GET'])
def chat_page():
    """Render the chat page"""
    return render_template('chat.html')

@app.route('/query-documents', methods=['POST'])
def query_documents():
    """Proxy for the document query API endpoint (advanced version)"""
    data = request.get_json()
    query_text = data.get('query', '')
    n_results = data.get('n_results', 3)
    combine_chunks = data.get('combine_chunks', True)
    web_search = data.get('web_search', None)  # None means auto-classify
    web_results_count = data.get('web_results_count', 5)
    explain_classification = data.get('explain_classification', False)
    enhance_query = data.get('enhance_query', True)
    use_elasticsearch = data.get('use_elasticsearch', None)  # None means auto-determine
    hybrid_search = data.get('hybrid_search', False)
    apply_reranking = data.get('apply_reranking', True)
    check_question_matches = data.get('check_question_matches', True)  # Default to True
    
    if not query_text:
        return jsonify({"status": "error", "message": "Query text is required"})
    
    try:
        # Call the API
        response = requests.get(
            f"{API_URL}/query", 
            params={
                'query': query_text,
                'n_results': n_results,
                'combine_chunks': combine_chunks,
                'web_search': web_search,
                'web_results_count': web_results_count,
                'explain_classification': explain_classification,
                'enhance_query': enhance_query,
                'use_elasticsearch': use_elasticsearch,
                'hybrid_search': hybrid_search,
                'apply_reranking': apply_reranking,
                'check_question_matches': check_question_matches
            },
            timeout=None
        )
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error querying documents: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/chat-query', methods=['POST'])
def chat_query():
    """Proxy for the chat API endpoint with conversation history"""
    data = request.get_json()
    
    # New chat format with message history
    messages = data.get('messages', [])
    n_results = data.get('n_results', 3)
    combine_chunks = data.get('combine_chunks', True)
    web_search = data.get('web_search', None)  # None means auto-classify
    web_results_count = data.get('web_results_count', 3)
    enhance_query = data.get('enhance_query', True)
    use_elasticsearch = data.get('use_elasticsearch', None)  # None means auto-determine
    hybrid_search = data.get('hybrid_search', False)
    apply_reranking = data.get('apply_reranking', True)
    check_question_matches = data.get('check_question_matches', True)  # Default to True
    
    # Ensure we have at least one user message
    has_user_message = False
    for msg in messages:
        if msg.get('role') == 'user':
            has_user_message = True
            break
            
    if not has_user_message:
        return jsonify({
            "status": "error", 
            "message": "No user message found in the conversation"
        })
    
    try:
        # Get model parameters
        model = data.get('model')
        use_openai = data.get('use_openai', False)
        assistant_id = data.get('assistant_id')  # Get assistant ID if provided
        use_local_docs = data.get('use_local_docs', True)  # Get the local docs setting (default to True)
        
        # Call the chat API
        response = requests.post(
            f"{API_URL}/chat",
            json={
                'messages': messages,
                'n_results': n_results,
                'combine_chunks': combine_chunks,
                'web_search': web_search,
                'web_results_count': web_results_count,
                'enhance_query': enhance_query,
                'use_elasticsearch': use_elasticsearch,
                'hybrid_search': hybrid_search,
                'apply_reranking': apply_reranking,
                'check_question_matches': check_question_matches,
                'model': model,
                'use_openai': use_openai,
                'assistant_id': assistant_id,
                'use_local_docs': use_local_docs  # Pass the local docs setting to the API
            },
            timeout=None
        )
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error in chat query: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/systeminfo', methods=['GET'])
def systeminfo_page():
    """Render the system information page"""
    return render_template('systeminfo.html')

@app.route('/config', methods=['GET'])
def config_page():
    """Render the configuration page"""
    return render_template('config.html')

@app.route('/sms', methods=['GET'])
def sms_page():
    """Render the SMS page"""
    return render_template('sms.html')

@app.route('/api/chroma-info', methods=['GET'])
def chroma_info():
    """Get information about ChromaDB"""
    try:
        # Check collection status
        response = requests.get(f"{API_URL}/health", timeout=10)
        health_data = response.json()
        
        # Extract ChromaDB information
        chroma_info = {
            "status": "success",
            "server_version": health_data.get("chroma", "unknown"),
            "api_status": health_data.get("api", "unknown"),
            "document_count": 0,
            "collection_count": 0
        }
        
        # Check collection information
        if "collection" in health_data and health_data["collection"]["status"] == "healthy":
            chroma_info["document_count"] = health_data["collection"]["document_count"]
            chroma_info["collection_count"] = 1  # We only have one collection in this app
            
        return jsonify(chroma_info)
    except Exception as e:
        logging.error(f"Error getting ChromaDB info: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/terms', methods=['GET'])
def get_terms():
    """Get classification terms from the API"""
    try:
        response = requests.get(f"{API_URL}/terms", timeout=10)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error getting terms: {e}")
        return jsonify({"status": "error", "message": str(e)})
        
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available Ollama models"""
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error getting models: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/refresh-terms', methods=['POST'])
def refresh_terms():
    """Refresh classification terms from the API"""
    try:
        response = requests.post(f"{API_URL}/refresh-terms", timeout=20)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error refreshing terms: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/health', methods=['GET'])
def api_health():
    """Get full health status from the API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return jsonify({"api": response.json()})
    except Exception as e:
        logging.error(f"Error getting health status: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current API settings"""
    try:
        response = requests.get(f"{API_URL}/current-settings", timeout=10)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error getting API settings: {e}")
        return jsonify({"status": "error", "message": str(e)})
        
@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update API settings"""
    try:
        data = request.get_json()
        response = requests.post(f"{API_URL}/api-settings", json=data, timeout=10)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error updating API settings: {e}")
        return jsonify({"status": "error", "message": str(e)})
        
@app.route('/api/sms', methods=['POST'])
def send_sms():
    """Send SMS via the API"""
    try:
        data = request.get_json()
        response = requests.post(f"{API_URL}/sms", json=data, timeout=60)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error sending SMS: {e}")
        return jsonify({"status": "error", "message": str(e)})
        
@app.route('/api/sms/quota', methods=['GET'])
def sms_quota():
    """Check SMS quota via the API"""
    try:
        response = requests.get(f"{API_URL}/sms/quota", timeout=10)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error checking SMS quota: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/clear-db', methods=['POST'])
def clear_database():
    """Clear the database"""
    try:
        response = requests.post(f"{API_URL}/clear-db", timeout=10)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error clearing database: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Upload a file to the API"""
    try:
        # Get the file from the request
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"})
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"})
            
        # Check if process_immediately is set
        process_immediately = request.form.get('process_immediately', 'false').lower() == 'true'
        
        # Get chunking parameters if provided
        chunk_size = request.form.get('chunk_size')
        min_size = request.form.get('min_size')
        overlap = request.form.get('overlap')
        enable_chunking = request.form.get('enable_chunking')
        enhance_chunks = request.form.get('enhance_chunks')
        generate_questions = request.form.get('generate_questions')
        max_questions_per_chunk = request.form.get('max_questions_per_chunk')
        
        # Forward the file to the API
        files = {'file': (file.filename, file.read(), file.content_type)}
        
        # Build data dictionary with all parameters
        data = {'process_immediately': str(process_immediately)}
        
        # Add chunking parameters if provided
        if chunk_size and chunk_size.isdigit():
            data['chunk_size'] = chunk_size
        if min_size and min_size.isdigit():
            data['min_size'] = min_size
        if overlap and overlap.isdigit():
            data['overlap'] = overlap
        if enable_chunking is not None:
            data['enable_chunking'] = enable_chunking
        if enhance_chunks is not None:
            data['enhance_chunks'] = enhance_chunks
        if generate_questions is not None:
            data['generate_questions'] = generate_questions
        if max_questions_per_chunk and max_questions_per_chunk.isdigit():
            data['max_questions_per_chunk'] = max_questions_per_chunk
        
        response = requests.post(
            f"{API_URL}/upload-file",
            files=files,
            data=data,
            timeout=60
        )
        
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return jsonify({"status": "error", "message": str(e)})
        
@app.route('/api/chunks', methods=['GET'])
def get_chunks():
    """Get document chunks from ChromaDB"""
    try:
        # Get parameters
        limit = request.args.get('limit', 20)
        offset = request.args.get('offset', 0)
        filename = request.args.get('filename', '')
        content = request.args.get('content', '')
        
        # Build query parameters
        params = {
            'limit': limit,
            'offset': offset
        }
        
        if filename:
            params['filename'] = filename
            
        if content:
            params['content'] = content
            
        # Call the API
        response = requests.get(f"{API_URL}/chunks", params=params, timeout=15)
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Error getting chunks: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/chunks', methods=['GET'])
def chunks_page():
    """Render the chunks explorer page"""
    return render_template('chunks.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
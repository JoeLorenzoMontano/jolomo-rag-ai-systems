from flask import Blueprint, render_template, request, jsonify
from ui.api.query_api import QueryAPI

query_bp = Blueprint('query', __name__)
query_api = QueryAPI()

@query_bp.route('/query', methods=['GET'])
def query_page():
    """Render the advanced query page."""
    return render_template('query.html')

@query_bp.route('/query-documents', methods=['POST'])
def query_documents():
    """Proxy for the document query API endpoint (advanced version)."""
    data = request.get_json()
    query_text = data.get('query', '')
    n_results = data.get('n_results', 3)
    combine_chunks = data.get('combine_chunks', True)
    web_search = data.get('web_search', None)  # None means auto-classify
    web_results_count = data.get('web_results_count', 5)
    explain_classification = data.get('explain_classification', False)
    enhance_query = data.get('enhance_query', True)
    
    if not query_text:
        return jsonify({"status": "error", "message": "Query text is required"})
    
    result = query_api.query(
        query_text=query_text,
        n_results=n_results,
        combine_chunks=combine_chunks,
        web_search=web_search,
        web_results_count=web_results_count,
        explain_classification=explain_classification,
        enhance_query=enhance_query
    )
    
    return jsonify(result)

@query_bp.route('/chat', methods=['GET'])
def chat_page():
    """Render the chat page."""
    return render_template('chat.html')

@query_bp.route('/chat-query', methods=['POST'])
def chat_query():
    """Proxy for the chat API endpoint with conversation history."""
    data = request.get_json()
    
    # For backward compatibility, handle both formats
    if 'messages' in data:
        # New chat format with message history
        messages = data.get('messages', [])
        n_results = data.get('n_results', 3)
        combine_chunks = data.get('combine_chunks', True)
        web_search = data.get('web_search', None)  # None means auto-classify
        web_results_count = data.get('web_results_count', 3)
        enhance_query = data.get('enhance_query', True)
        
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
        
        result = query_api.chat(
            messages=messages,
            n_results=n_results,
            combine_chunks=combine_chunks,
            web_search=web_search,
            web_results_count=web_results_count,
            enhance_query=enhance_query
        )
        
        return jsonify(result)
    else:
        # Legacy single-query format
        query_text = data.get('query', '')
        n_results = data.get('n_results', 3)
        combine_chunks = data.get('combine_chunks', True)
        web_search = data.get('web_search', None)  # None means auto-classify
        web_results_count = data.get('web_results_count', 3)
        enhance_query = data.get('enhance_query', True)
        
        if not query_text:
            return jsonify({"status": "error", "message": "Query text is required"})
        
        result = query_api.chat(
            query_text=query_text,
            n_results=n_results,
            combine_chunks=combine_chunks,
            web_search=web_search,
            web_results_count=web_results_count,
            enhance_query=enhance_query
        )
        
        return jsonify(result)
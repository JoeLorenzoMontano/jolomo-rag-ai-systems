/**
 * API utility library for the RAG application
 * Provides standardized methods for interacting with the backend API
 */

const API = {
    /**
     * Fetch available models from the API
     * @returns {Promise<Object>} List of available models and model info
     */
    async getModels() {
        try {
            const response = await fetch('/api/models');
            return await response.json();
        } catch (error) {
            console.error('Error loading models:', error);
            throw error;
        }
    },

    /**
     * Fetch current system settings
     * @returns {Promise<Object>} Current system settings
     */
    async getSettings() {
        try {
            const response = await fetch('/api/settings');
            return await response.json();
        } catch (error) {
            console.error('Error loading settings:', error);
            throw error;
        }
    },

    /**
     * Update system settings
     * @param {Object} settings - Settings object to update
     * @returns {Promise<Object>} Result of update operation
     */
    async updateSettings(settings) {
        try {
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            return await response.json();
        } catch (error) {
            console.error('Error updating settings:', error);
            throw error;
        }
    },

    /**
     * Send a chat query to the API
     * @param {Array} messages - Array of message objects with role and content
     * @param {Object} options - Additional options for the query
     * @returns {Promise<Object>} Chat response
     */
    async sendChatQuery(messages, options = {}) {
        try {
            // Create default query parameters
            const queryParams = {
                messages: messages,
                n_results: options.n_results || 3,
                combine_chunks: options.combine_chunks !== undefined ? options.combine_chunks : true,
                web_search: options.web_search,
                web_results_count: options.web_results_count || 3,
                enhance_query: options.enhance_query !== undefined ? options.enhance_query : false,
                use_elasticsearch: options.use_elasticsearch,
                hybrid_search: options.hybrid_search !== undefined ? options.hybrid_search : true,
                apply_reranking: options.apply_reranking !== undefined ? options.apply_reranking : true,
                check_question_matches: options.check_question_matches !== undefined ? options.check_question_matches : true,
                model: options.model,
                use_openai: options.use_openai !== undefined ? options.use_openai : false,
                assistant_id: options.assistant_id,
                use_local_docs: options.use_local_docs !== undefined ? options.use_local_docs : true
            };

            // Send the request
            const response = await fetch('/chat-query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(queryParams),
                timeout: options.timeout || undefined
            });

            return await response.json();
        } catch (error) {
            console.error('Error in chat query:', error);
            throw error;
        }
    },

    /**
     * Send a document query to the API
     * @param {String} query - Query text
     * @param {Object} options - Additional options for the query
     * @returns {Promise<Object>} Query response
     */
    async sendDocumentQuery(query, options = {}) {
        try {
            // Create default query parameters
            const queryParams = {
                query: query,
                n_results: options.n_results || 3,
                combine_chunks: options.combine_chunks !== undefined ? options.combine_chunks : true,
                web_search: options.web_search,
                web_results_count: options.web_results_count || 5,
                explain_classification: options.explain_classification !== undefined ? options.explain_classification : false,
                enhance_query: options.enhance_query !== undefined ? options.enhance_query : true,
                use_elasticsearch: options.use_elasticsearch,
                hybrid_search: options.hybrid_search !== undefined ? options.hybrid_search : true,
                apply_reranking: options.apply_reranking !== undefined ? options.apply_reranking : true,
                check_question_matches: options.check_question_matches !== undefined ? options.check_question_matches : true,
                query_model: options.query_model,
                embedding_model: options.embedding_model
            };

            // Send the request
            const response = await fetch('/query-documents', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(queryParams),
                timeout: options.timeout || undefined
            });

            return await response.json();
        } catch (error) {
            console.error('Error in document query:', error);
            throw error;
        }
    },

    /**
     * Get system health status
     * @returns {Promise<Object>} System health status
     */
    async getHealth() {
        try {
            const response = await fetch('/api/health');
            return await response.json();
        } catch (error) {
            console.error('Error fetching health status:', error);
            throw error;
        }
    }
};
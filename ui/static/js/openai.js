/**
 * OpenAI utility library for the RAG application
 * Handles OpenAI-specific functionality including settings and model management
 */

const OpenAI = {
    /**
     * Check if OpenAI is configured in the system
     * @returns {Promise<boolean>} True if OpenAI is configured
     */
    async isConfigured() {
        try {
            const settings = await API.getSettings();
            return settings.status === 'success' && 
                   settings.settings && 
                   settings.settings.has_openai_key === true;
        } catch (error) {
            console.error('Error checking OpenAI configuration:', error);
            return false;
        }
    },
    
    /**
     * Get configured OpenAI assistant IDs
     * @returns {Promise<Array>} Array of assistant IDs or empty array if none configured
     */
    async getAssistantIds() {
        try {
            const settings = await API.getSettings();
            if (settings.status === 'success' && 
                settings.settings && 
                settings.settings.openai_assistant_ids) {
                return settings.settings.openai_assistant_ids;
            }
            return [];
        } catch (error) {
            console.error('Error getting OpenAI assistant IDs:', error);
            return [];
        }
    },
    
    /**
     * Update OpenAI configuration
     * @param {Object} config - Configuration object
     * @param {String} config.apiKey - OpenAI API key (optional)
     * @param {Array} config.assistantIds - Array of assistant IDs (optional)
     * @returns {Promise<Object>} Result of update operation
     */
    async updateConfig({ apiKey, assistantIds } = {}) {
        const settings = {};
        
        if (apiKey !== undefined) {
            settings.openai_api_key = apiKey;
        }
        
        if (assistantIds !== undefined) {
            settings.openai_assistant_ids = assistantIds;
        }
        
        try {
            return await API.updateSettings(settings);
        } catch (error) {
            console.error('Error updating OpenAI configuration:', error);
            throw error;
        }
    },
    
    /**
     * Check if a model selection is an OpenAI model
     * @param {String} modelSelection - The model selection value
     * @returns {Boolean} True if it's an OpenAI model
     */
    isOpenAIModel(modelSelection) {
        return modelSelection.startsWith('openai_') || modelSelection.startsWith('assistant_');
    },
    
    /**
     * Check if a model selection is an OpenAI Assistant
     * @param {String} modelSelection - The model selection value
     * @returns {Boolean} True if it's an OpenAI Assistant
     */
    isAssistant(modelSelection) {
        return modelSelection.startsWith('assistant_');
    },
    
    /**
     * Get the Assistant ID from a model selection value
     * @param {String} modelSelection - The model selection value
     * @returns {String|null} The Assistant ID or null if not an assistant
     */
    getAssistantId(modelSelection) {
        if (this.isAssistant(modelSelection)) {
            return modelSelection.substring('assistant_'.length);
        }
        return null;
    },
    
    /**
     * Update UI elements based on selected model
     * @param {String} modelSelection - The selected model
     * @param {HTMLElement} localDocsToggle - The local docs toggle element
     */
    updateUIForAssistant(modelSelection, localDocsToggle) {
        if (this.isAssistant(modelSelection)) {
            // Disable local docs by default for assistants
            localDocsToggle.checked = false;
            // Add tooltip or visual indicator that local docs aren't needed with assistants
            console.log('OpenAI Assistant selected: Local docs disabled by default');
        }
    }
};
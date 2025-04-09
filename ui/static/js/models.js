/**
 * Models utility library for the RAG application
 * Handles model selection, loading available models, and model-related UI
 */

const Models = {
    /**
     * Initialize model selectors in the UI
     * @param {Object} options - Configuration options
     * @param {HTMLElement} options.queryModelSelect - Query model select element (optional)
     * @param {HTMLElement} options.embeddingModelSelect - Embedding model select element (optional)
     * @param {HTMLElement} options.chatModelSelect - Chat model select element (optional)
     * @returns {Promise<void>}
     */
    async initializeModelSelectors({ queryModelSelect, embeddingModelSelect, chatModelSelect } = {}) {
        try {
            // Load models from API
            const data = await API.getModels();
            
            if (data.status !== 'success' || !data.models) {
                console.error('Failed to load models or invalid response format');
                return;
            }
            
            // Process available models
            const models = data.models;
            const defaultModel = data.default_model;
            const defaultEmbeddingModel = data.default_embedding_model;
            
            // Update query model selector if provided
            if (queryModelSelect) {
                this.populateModelSelector(queryModelSelect, models, defaultModel);
            }
            
            // Update embedding model selector if provided
            if (embeddingModelSelect) {
                // Filter models for embedding-compatible ones
                const embeddingModels = models.filter(model => 
                    model.includes('minilm') || 
                    model.includes('embed') || 
                    model.includes('e5')
                );
                this.populateModelSelector(embeddingModelSelect, embeddingModels, defaultEmbeddingModel);
            }
            
            // Update chat model selector if provided
            if (chatModelSelect) {
                this.populateChatModelSelector(chatModelSelect, models, defaultModel);
            }
        } catch (error) {
            console.error('Error initializing model selectors:', error);
        }
    },
    
    /**
     * Populate a model selector with available models
     * @param {HTMLElement} selectElement - The select element to populate
     * @param {Array<String>} models - List of available models
     * @param {String} defaultModel - Default model to select
     */
    populateModelSelector(selectElement, models, defaultModel) {
        // Clear existing options except the first one (default)
        while (selectElement.options.length > 1) {
            selectElement.remove(1);
        }
        
        // Update the default option text
        if (defaultModel) {
            const defaultOption = selectElement.querySelector('option[value=""]');
            if (defaultOption) {
                defaultOption.textContent = `Default (${defaultModel})`;
            }
        }
        
        // Add options for each model
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            selectElement.appendChild(option);
        });
    },
    
    /**
     * Populate a chat model selector with available models and OpenAI options
     * @param {HTMLElement} selectElement - The select element to populate
     * @param {Array<String>} models - List of available models
     * @param {String} defaultModel - Default model to select
     */
    async populateChatModelSelector(selectElement, models, defaultModel) {
        // Clear existing options
        selectElement.innerHTML = '';
        
        // Add default Ollama option
        const defaultOption = document.createElement('option');
        defaultOption.value = 'default';
        defaultOption.textContent = `Default (${defaultModel || 'Ollama'})`;
        defaultOption.selected = true;
        selectElement.appendChild(defaultOption);
        
        // Add OpenAI options
        const gpt4Option = document.createElement('option');
        gpt4Option.value = 'openai_gpt4';
        gpt4Option.textContent = 'OpenAI GPT-4';
        selectElement.appendChild(gpt4Option);
        
        const gpt35Option = document.createElement('option');
        gpt35Option.value = 'openai_gpt35';
        gpt35Option.textContent = 'OpenAI GPT-3.5';
        selectElement.appendChild(gpt35Option);
        
        // Add OpenAI Assistants if available
        try {
            const settingsResponse = await API.getSettings();
            
            if (settingsResponse.status === 'success' && 
                settingsResponse.settings && 
                settingsResponse.settings.openai_assistant_ids && 
                settingsResponse.settings.openai_assistant_ids.length > 0) {
                
                // Add a divider
                const divider = document.createElement('option');
                divider.disabled = true;
                divider.textContent = '───── Assistants ─────';
                selectElement.appendChild(divider);
                
                // Add assistants
                const assistantIds = settingsResponse.settings.openai_assistant_ids;
                assistantIds.forEach((assistantId, index) => {
                    const option = document.createElement('option');
                    option.value = `assistant_${assistantId}`;
                    option.textContent = `Assistant ${index + 1} (${assistantId.substring(0, 8)}...)`;
                    selectElement.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Error loading assistant options:', error);
        }
    },
    
    /**
     * Determine model parameters based on model selection
     * @param {String} selectedModel - The selected model value
     * @returns {Object} Model parameters for API request
     */
    getModelParameters(selectedModel) {
        let params = {
            model: null,
            use_openai: false,
            assistant_id: null
        };
        
        if (selectedModel.startsWith('assistant_')) {
            // This is an OpenAI Assistant
            params.use_openai = true;
            params.model = 'assistant';
            params.assistant_id = selectedModel.substring('assistant_'.length);
        } else {
            switch(selectedModel) {
                case 'openai_gpt4':
                    params.use_openai = true;
                    params.model = 'gpt-4';
                    break;
                case 'openai_gpt35':
                    params.use_openai = true;
                    params.model = 'gpt-3.5-turbo';
                    break;
                case 'default':
                    // Use default model (null means use system default)
                    params.model = null;
                    break;
                default:
                    // Assume it's a custom model name
                    params.model = selectedModel;
            }
        }
        
        return params;
    }
};
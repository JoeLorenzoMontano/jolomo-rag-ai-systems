/**
 * Sources utility library for the RAG application
 * Handles rendering and displaying document sources from query results
 */

const Sources = {
    /**
     * Render sources from a query result
     * @param {Object} data - The query result data
     * @param {HTMLElement} sourcesContainer - Container to render sources in
     */
    renderSources(data, sourcesContainer) {
        // Skip if no sources or container
        if (!data || !sourcesContainer) return;
        
        // Process sources for simplified display
        let sources = [];
        
        // Add document sources
        if (data.sources && data.sources.documents && data.sources.documents.length > 0) {
            // Extract just the filenames or IDs without content
            for (let i = 0; i < data.sources.documents.length; i++) {
                const metadata = data.sources.metadatas[i] || {};
                const sourceId = data.sources.ids[i] || `source-${i}`;
                sources.push({
                    title: metadata.filename || sourceId,
                    type: 'document'
                });
            }
        }
        
        // Add web sources
        if (data.sources && data.sources.web_results && data.sources.web_results.length > 0) {
            data.sources.web_results.forEach(result => {
                sources.push({
                    title: result.title,
                    url: result.url,
                    type: 'web'
                });
            });
        }
        
        // Create sources list HTML
        if (sources.length === 0) {
            sourcesContainer.innerHTML = '';
            return;
        }
        
        const sourceItems = sources.map(source => {
            if (source.type === 'web' && source.url) {
                return `<li><a href="${source.url}" target="_blank">${source.title}</a> <span class="badge bg-warning text-dark">Web</span></li>`;
            } else {
                return `<li>${source.title} <span class="badge bg-info">Document</span></li>`;
            }
        });
        
        sourcesContainer.innerHTML = `<ol>${sourceItems.join('')}</ol>`;
    },
    
    /**
     * Render detailed sources accordion for the query page
     * @param {Object} data - The query result data
     * @param {HTMLElement} container - Container to render the accordion in
     */
    renderDetailedSources(data, container) {
        if (!data || !container) return;
        
        let html = '<div class="accordion" id="sourcesAccordion">';
        
        // Display query enhancement info if available
        if (data.query_enhanced && data.enhanced_query) {
            html += `
                <div class="alert alert-success mb-3">
                    <i class="bi bi-lightbulb-fill me-2"></i> 
                    <strong>Query Enhanced:</strong> Your query was enhanced to "${data.enhanced_query}"
                </div>
            `;
        }
        
        // Display question match information if available
        if (data.exact_question_match) {
            html += `
                <div class="alert alert-success mb-3">
                    <i class="bi bi-question-circle-fill me-2"></i> 
                    <strong>Exact Question Match Found:</strong> "${data.exact_question_match.question}"
                </div>
            `;
        } else if (data.question_matches && data.question_matches.length > 0) {
            let matchesHtml = data.question_matches.map(match => 
                `<li><strong>${match.question}</strong> <span class="badge bg-info">${(match.score * 100).toFixed(0)}% match</span></li>`
            ).join('');
            
            html += `
                <div class="alert alert-info mb-3">
                    <i class="bi bi-question-circle me-2"></i> 
                    <strong>Similar Questions Found:</strong>
                    <ul class="mt-2">
                        ${matchesHtml}
                    </ul>
                </div>
            `;
        }
        
        // Display reranking info if applied
        if (data.sources && data.sources.reranked) {
            html += `
                <div class="alert alert-info mb-3">
                    <i class="bi bi-arrow-left-right me-2"></i> 
                    <strong>Reranked Results:</strong> Documents have been reordered based on semantic relevance to your query
                </div>
            `;
        }
        
        // Add vector database sources
        if (data.sources && data.sources.documents) {
            data.sources.documents.forEach((doc, index) => {
                const metadata = data.sources.metadatas[index] || {};
                const distance = data.sources.distances ? data.sources.distances[index] : null;
                const sourceId = data.sources.ids[index] || `source-${index}`;
                
                html += `
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="heading${index}">
                            <button class="accordion-button ${index > 0 ? 'collapsed' : ''}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${index}" aria-expanded="${index === 0 ? 'true' : 'false'}" aria-controls="collapse${index}">
                                <strong>${metadata.filename || sourceId}</strong>
                                ${distance !== null ? ` <span class="ms-2 badge bg-secondary">Similarity: ${(1 - distance).toFixed(3)}</span>` : ''}
                                <span class="ms-2 badge bg-info">Local Document</span>
                                ${metadata.search_source ? 
                                    `<span class="ms-2 badge ${
                                        metadata.search_source === 'vector' ? 'bg-primary' : 
                                        metadata.search_source === 'text' ? 'bg-danger' : 'bg-success'
                                    }">${
                                        metadata.search_source === 'vector' ? 'Vector' : 
                                        metadata.search_source === 'text' ? 'Text' : 'Hybrid'
                                    }</span>` 
                                : ''}
                            </button>
                        </h2>
                        <div id="collapse${index}" class="accordion-collapse collapse ${index === 0 ? 'show' : ''}" aria-labelledby="heading${index}" data-bs-parent="#sourcesAccordion">
                            <div class="accordion-body">
                                ${metadata && metadata.has_enrichment ? '<div class="badge bg-info mb-2">Semantically Enhanced</div>' : ''}
                                <pre class="source-text">${doc}</pre>
                            </div>
                        </div>
                    </div>
                `;
            });
        }
        
        // Add web search results if available
        if (data.sources && data.sources.web_results && data.sources.web_results.length > 0) {
            let startIdx = data.sources.documents ? data.sources.documents.length : 0;
            
            data.sources.web_results.forEach((result, idx) => {
                const index = startIdx + idx;
                html += `
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="heading${index}">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${index}" aria-expanded="false" aria-controls="collapse${index}">
                                <strong>${result.title}</strong>
                                <span class="ms-2 badge bg-warning">Web Result</span>
                            </button>
                        </h2>
                        <div id="collapse${index}" class="accordion-collapse collapse" aria-labelledby="heading${index}" data-bs-parent="#sourcesAccordion">
                            <div class="accordion-body">
                                <div class="mb-2">
                                    <strong>Source:</strong> <a href="${result.url}" target="_blank">${result.source} - ${result.url}</a>
                                </div>
                                <pre class="source-text">${result.content}</pre>
                            </div>
                        </div>
                    </div>
                `;
            });
        }
        
        html += '</div>';
        
        // Add web search indicator if web search was used
        if (data.web_search_used) {
            html = `
                <div class="alert alert-info mb-3">
                    <i class="bi bi-info-circle"></i> Results include web search data from Serper API.
                </div>
            ` + html;
        }
        
        // Add search engine and model information
        if (data.search_engine || data.query_model || data.embedding_model) {
            let infoHtml = '<div class="alert alert-light mb-3">';
            
            // Add search engine badge
            if (data.search_engine) {
                let engineBadge = '';
                let engineDesc = '';
                
                if (data.search_engine === 'elasticsearch') {
                    engineBadge = '<span class="badge bg-warning ms-1">Elasticsearch</span>';
                    engineDesc = 'Results retrieved using Elasticsearch';
                } else if (data.search_engine === 'hybrid') {
                    engineBadge = '<span class="badge bg-success ms-1">Hybrid Search</span>';
                    engineDesc = 'Results combined from both ChromaDB and Elasticsearch';
                } else {
                    engineBadge = '<span class="badge bg-info ms-1">ChromaDB</span>';
                    engineDesc = 'Results retrieved using ChromaDB vector database';
                }
                
                infoHtml += `
                    <div class="mb-2">
                        <i class="bi bi-database"></i> 
                        <strong>Search Engine:</strong> ${engineBadge}
                        <div class="small text-muted mt-1">${engineDesc}</div>
                    </div>
                `;
            }
            
            // Add model information
            if (data.query_model || data.embedding_model) {
                infoHtml += `
                    <div class="mt-2">
                        <i class="bi bi-cpu"></i> 
                        <strong>Models used:</strong>
                `;
                
                if (data.query_model) {
                    infoHtml += `
                        <div class="small mt-1">
                            <strong>Query model:</strong> <span class="badge bg-primary">${data.query_model}</span>
                        </div>
                    `;
                }
                
                if (data.embedding_model) {
                    infoHtml += `
                        <div class="small mt-1">
                            <strong>Embedding model:</strong> <span class="badge bg-secondary">${data.embedding_model}</span>
                        </div>
                    `;
                }
                
                infoHtml += '</div>';
            }
            
            infoHtml += '</div>';
            html = infoHtml + html;
        }
        
        container.innerHTML = html;
    },
    
    /**
     * Create a toggle for showing/hiding sources
     * @param {HTMLElement} container - Message container to append toggle to
     * @param {Array} sources - Array of source objects
     * @param {HTMLElement} sourcesList - Element containing the sources list
     */
    createSourcesToggle(container, sources, sourcesList) {
        const sourcesToggle = document.createElement('div');
        sourcesToggle.className = 'sources-toggle';
        sourcesToggle.innerHTML = `<i class="bi bi-info-circle"></i> Show sources (${sources.length})`;
        
        // Toggle sources visibility
        sourcesToggle.addEventListener('click', () => {
            const isVisible = sourcesList.style.display === 'block';
            sourcesList.style.display = isVisible ? 'none' : 'block';
            sourcesToggle.innerHTML = isVisible 
                ? `<i class="bi bi-info-circle"></i> Show sources (${sources.length})` 
                : `<i class="bi bi-info-circle-fill"></i> Hide sources`;
        });
        
        container.appendChild(sourcesToggle);
    }
};
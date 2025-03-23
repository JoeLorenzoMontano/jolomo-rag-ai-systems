from typing import List, Dict, Tuple, Optional, Any
import re
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
import logging

# Initialize NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logging.warning(f"Could not download NLTK stopwords: {e}")

class QueryClassifier:
    """
    Classifies queries to determine whether to use document retrieval,
    web search, or a hybrid approach.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize the classifier with default settings.
        
        Args:
            confidence_threshold: Threshold for classification decisions (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default domain-specific terms
        self.product_terms = ["tenant", "infrastructure", "platform", "service", "configuration"]
            
    def update_terms_from_db(self, db_collection, ollama_client=None):
        """
        Update product terms using content stored in ChromaDB
        
        Args:
            db_collection: ChromaDB collection containing document embeddings
            ollama_client: Not used in this implementation, for compatibility
        """
        try:
            # Get all documents from the collection
            all_docs = db_collection.get(include=["documents", "metadatas"])
            if not all_docs or "documents" not in all_docs or not all_docs["documents"]:
                self.logger.warning("No documents found in ChromaDB for term extraction")
                return
            
            # Filter out enrichment sections from documents by using original_text if available
            filtered_docs = []
            for i, doc in enumerate(all_docs["documents"]):
                if "metadatas" in all_docs and all_docs["metadatas"] and i < len(all_docs["metadatas"]):
                    # If we have original_text metadata, use that instead of enriched text
                    metadata = all_docs["metadatas"][i]
                    if metadata and "original_text" in metadata:
                        filtered_docs.append(metadata["original_text"])
                        continue
                
                # Otherwise use the document but remove ENRICHMENT sections
                if "ENRICHMENT:" in doc:
                    # Only use text before the ENRICHMENT marker
                    clean_doc = doc.split("ENRICHMENT:")[0].strip()
                    filtered_docs.append(clean_doc)
                else:
                    filtered_docs.append(doc)
                
            # Process documents to extract important terms
            doc_text = " ".join(filtered_docs)
            extracted_terms = self._extract_important_terms(doc_text)
            
            # Set the product terms to the extracted terms
            self.product_terms = extracted_terms
            
            # Log the top terms for debugging
            if extracted_terms:
                top_terms = ", ".join(extracted_terms[:10])
                self.logger.info(f"Updated product terms from ChromaDB: Found {len(self.product_terms)} terms")
                self.logger.info(f"Top extracted terms: {top_terms}")
            else:
                self.logger.warning("No terms could be extracted from documents")
            
        except Exception as e:
            self.logger.error(f"Error extracting terms from ChromaDB: {e}")
            # Fall back to minimal terms
            self.product_terms = ["duplocloud", "tenant", "infrastructure"]
            
    def _extract_important_terms(self, text, min_length=4, max_terms=200):
        """
        Extract important domain-specific terms from text
        
        Args:
            text: The text to analyze
            min_length: Minimum length for a term to be considered
            max_terms: Maximum number of terms to return
            
        Returns:
            List of important terms
        """
        # Convert to lowercase and tokenize by splitting on whitespace and punctuation
        text = text.lower()
        
        # Remove punctuation and numbers
        translator = str.maketrans('', '', string.punctuation + string.digits)
        text = text.translate(translator)
        
        # Split into words
        words = text.split()
        
        # Remove common English stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
        except Exception as e:
            self.logger.warning(f"Could not filter stopwords: {e}")
        
        # Filter out short words, image file references, and enrichment-related terms
        enrichment_terms = ['enrichment', 'context', 'summary', 'section', 'paragraph', 'semantic',
                           'describes', 'covers', 'explains', 'details', 'document', 'content']
        words = [word for word in words if len(word) >= min_length and 
                 not any(ext in word for ext in ['png', 'jpg', 'jpeg', 'gif']) and
                 word not in enrichment_terms]
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Find high-frequency words that stand out
        word_items = word_counts.most_common(max_terms * 2)
        
        # Calculate stats for identifying outliers in word frequency
        if word_items:
            word_freqs = [count for _, count in word_items]
            avg_freq = sum(word_freqs) / len(word_freqs)
            freq_threshold = max(5, avg_freq * 1.5)  # Threshold for "high frequency"
            
            # Get high-frequency unigrams (only include those with significant frequency)
            high_freq_terms = [term for term, count in word_items if count >= freq_threshold]
            
            # Get other common unigrams
            common_terms = [term for term, count in word_items if term not in high_freq_terms]
        else:
            high_freq_terms = []
            common_terms = []
        
        # Also extract bigrams (pairs of consecutive words)
        bigrams = []
        enrichment_bigrams = [
            'context summary', 'this section', 'section describes', 'summary this', 
            'this document', 'document describes', 'covers the', 'details how',
            'semantic context', 'describes how', 'explains how', 'document covers',
            'document explains', 'section covers', 'section explains', 'content explains',
            'content covers', 'content details', 'content describes', 'key information'
        ]
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i+1]
            if len(bigram) >= min_length and bigram not in enrichment_bigrams:
                bigrams.append(bigram)
                
        # Count bigram frequencies
        bigram_counts = Counter(bigrams)
        
        # Extract the most common bigrams
        common_bigrams = [term for term, count in bigram_counts.most_common(max_terms)]
        
        # Combine terms with priority: high-frequency unigrams, bigrams, then other common terms
        all_terms = high_freq_terms + common_bigrams + common_terms
        
        # Remove duplicates while preserving order
        unique_terms = []
        for term in all_terms:
            if term not in unique_terms:
                unique_terms.append(term)
                
        return unique_terms[:max_terms]
    
    def classify(self, 
                query: str, 
                doc_scores: Optional[List[float]] = None,
                conversation_history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify a query to determine the best source for answering.
        
        Args:
            query: The user's query text
            doc_scores: Relevance scores for retrieved documents (if available)
            conversation_history: Optional conversation history for follow-up detection
            
        Returns:
            Tuple containing:
            - source_type: "documents", "web", "conversation", "hybrid", or "hybrid_conversation"
            - confidence: confidence score (0-1)
            - metadata: Additional classification data
        """
        scores = {
            "documents": 0.0,
            "web": 0.0,
            "conversation": 0.0
        }
        explanations = []
        
        # Check for DuploCloud-specific terminology
        term_score, term_matches = self._keyword_match(query)
        scores["documents"] = term_score
        explanations.append(f"Term match: {term_score:.2f} (found {len(term_matches)} Domain terms)")
        
        # If document scores are provided, factor them in
        retrieval_score = 0.0
        if doc_scores and len(doc_scores) > 0:
            retrieval_score = min(1.0, max(doc_scores))
            scores["documents"] = 0.7 * scores["documents"] + 0.3 * retrieval_score
            explanations.append(f"Document relevance: {retrieval_score:.2f}")
        
        # Check for follow-up indicators if conversation history is provided
        reference_items = []
        if conversation_history and len(conversation_history) >= 2:
            # Detect reference indicators like "those", "above", item numbers, etc.
            reference_indicators = [
                r'\b(the|these|those|it|that|this|they|them)\b',
                r'\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last|previous|above|mentioned)\b',
                r'\b(you|your|you\'ve|you\'re|you\'d)\b',
                r'\b(item|point|bullet|number|step)\s*[#]?\s*(\d+)',
                r'[?]\s*(and|also|additionally)\b',
                r'\b(can you|please|tell me more|elaborate|explain|details)\b'
            ]
            
            for pattern in reference_indicators:
                if re.search(pattern, query, re.IGNORECASE):
                    scores["conversation"] += 0.15  # Each match increases score
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        reference_items.append(match.group(0))
            
            # Check if query is very short (likely a follow-up)
            words = query.split()
            if len(words) <= 5:
                scores["conversation"] += 0.3
                explanations.append(f"Short query ({len(words)} words), likely follow-up")
                
            # Cap the conversation score
            scores["conversation"] = min(0.95, scores["conversation"])
            
            if reference_items:
                explanations.append(f"Found reference indicators: {', '.join(reference_items)}")
        
        # Calculate web score (inverse of document score)
        scores["web"] = 1.0 - scores["documents"]
        
        # Determine if hybrid approach is better
        # If document score is in the middle range (0.3-0.7), hybrid might be best
        hybrid_score = 1.0 - 2.0 * abs(scores["documents"] - 0.5)
        scores["hybrid"] = max(0.0, hybrid_score)
        
        # Determine the source to use based on scores and context
        if scores["conversation"] > 0.7:
            # High confidence this is a follow-up question
            source_type = "conversation"
            confidence = scores["conversation"]
            explanations.append("Using conversation history as primary context")
        elif scores["conversation"] > 0.4 and scores["documents"] > 0.4:
            # Both conversation and documents relevant
            source_type = "hybrid_conversation"
            confidence = (scores["documents"] + scores["conversation"]) / 2
            explanations.append("Using documents with conversation context priority")
        elif scores["documents"] >= self.confidence_threshold:
            source_type = "documents"
            confidence = scores["documents"]
        elif scores["web"] >= self.confidence_threshold:
            source_type = "web"
            confidence = scores["web"]
        else:
            source_type = "hybrid"
            confidence = scores["hybrid"]
        
        # Prepare metadata with explanation of decision
        metadata = {
            "scores": scores,
            "explanations": explanations,
            "matched_terms": term_matches,
            "reference_items": reference_items if reference_items else []
        }
        
        return source_type, confidence, metadata
    
    def _keyword_match(self, query: str) -> Tuple[float, List[str]]:
        """
        Check for DuploCloud-specific terms in the query.
        
        Args:
            query: The query text to analyze
            
        Returns:
            Tuple containing:
            - score: Confidence score for document retrieval (0-1)
            - matches: List of matched product terms
        """
        query_lower = query.lower()
        matches = []
        
        # Find matching product terms in the query
        for term in self.product_terms:
            # Use word boundary to find whole word matches
            if re.search(r'\b' + re.escape(term.lower()) + r'\b', query_lower):
                matches.append(term)
        
        # Calculate score based on number of matches
        if not matches:
            return 0.0, []
        
        # Score increases with more matches but caps at 1.0
        # A single match gives 0.4, two matches 0.7, three or more 0.9+
        score = min(1.0, 0.4 + (len(matches) - 1) * 0.3)
        return score, matches

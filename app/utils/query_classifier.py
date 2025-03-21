from typing import List, Dict, Tuple, Optional, Any
import re

class QueryClassifier:
    """
    Classifies queries to determine whether to use document retrieval,
    web search, or a hybrid approach.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize the classifier with DuploCloud-specific terminology.
        
        Args:
            confidence_threshold: Threshold for classification decisions (0-1)
        """
        self.confidence_threshold = confidence_threshold
        
        # Core DuploCloud terminology for query classification
        self.product_terms = [
            # Core concepts
            "duplocloud", "duplo", "tenant", "infrastructure", "plan",
            
            # Services
            "service", "app service", "cloud service", "kubernetes",
            "docker", "lambda", "ecs", "microservice",
            
            # Cloud resources
            "vpc", "subnet", "security group", "load balancer",
            "s3", "rds", "redis", "elasticsearch", "kafka",
            
            # Monitoring and security
            "diagnostics", "logs", "metrics", "alarms", "audit trail",
            "kibana", "grafana", "compliance",
            
            # Configuration
            "resource quota", "iam role", "dns", "config settings",
            "namespace", "template"
        ]
    
    def classify(self, 
                query: str, 
                doc_scores: Optional[List[float]] = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classify a query to determine the best source for answering.
        
        Args:
            query: The user's query text
            doc_scores: Relevance scores for retrieved documents (if available)
            
        Returns:
            Tuple containing:
            - source_type: "documents", "web", or "hybrid"
            - confidence: confidence score (0-1)
            - metadata: Additional classification data
        """
        scores = {
            "documents": 0.0,
            "web": 0.0
        }
        explanations = []
        
        # Check for DuploCloud-specific terminology
        term_score, term_matches = self._keyword_match(query)
        scores["documents"] = term_score
        explanations.append(f"Term match: {term_score:.2f} (found {len(term_matches)} DuploCloud terms)")
        
        # If document scores are provided, factor them in
        retrieval_score = 0.0
        if doc_scores and len(doc_scores) > 0:
            retrieval_score = min(1.0, max(doc_scores))
            scores["documents"] = 0.7 * scores["documents"] + 0.3 * retrieval_score
            explanations.append(f"Document relevance: {retrieval_score:.2f}")
        
        # Calculate web score (inverse of document score)
        scores["web"] = 1.0 - scores["documents"]
        
        # Determine if hybrid approach is better
        # If document score is in the middle range (0.3-0.7), hybrid might be best
        hybrid_score = 1.0 - 2.0 * abs(scores["documents"] - 0.5)
        scores["hybrid"] = max(0.0, hybrid_score)
        
        # Determine the source to use based on scores and thresholds
        if scores["documents"] >= self.confidence_threshold:
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
            "matched_terms": term_matches
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
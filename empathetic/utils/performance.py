"""Performance optimization utilities for NLP pipeline"""
import time
from typing import List, Dict, Any, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

logger = logging.getLogger(__name__)


class NLPPipelineOptimizer:
    """Optimizes NLP pipeline performance through caching and batching"""
    
    def __init__(self, cache_size: int = 1000, max_workers: int = 4):
        self.cache_size = cache_size
        self.max_workers = max_workers
        self.batch_size = 32
        self._cache_stats = {"hits": 0, "misses": 0}
        self._lock = threading.Lock()
    
    @lru_cache(maxsize=1000)
    def _cached_dependency_parse(self, text: str) -> str:
        """Cache dependency parsing results"""
        # This would be called by the actual dependency analyzer
        # For now, return a cache key
        return f"dep_cache_{hash(text)}"
    
    @lru_cache(maxsize=1000) 
    def _cached_intent_classification(self, text: str, category: str) -> str:
        """Cache intent classification results"""
        return f"intent_cache_{hash(text)}_{category}"
    
    def batch_analyze_responses(
        self,
        responses: List[str],
        test_cases: List[Any]
    ) -> List[Dict]:
        """
        Efficiently process multiple responses in batches
        """
        if len(responses) != len(test_cases):
            raise ValueError("Responses and test cases must have same length")
        
        results = []
        
        # Process in batches
        for i in range(0, len(responses), self.batch_size):
            batch_responses = responses[i:i + self.batch_size]
            batch_test_cases = test_cases[i:i + self.batch_size]
            
            batch_results = self._process_batch(batch_responses, batch_test_cases)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(
        self,
        responses: List[str],
        test_cases: List[Any]
    ) -> List[Dict]:
        """Process a batch of responses in parallel"""
        
        results = [None] * len(responses)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._analyze_single_response, response, test_case): i
                for i, (response, test_case) in enumerate(zip(responses, test_cases))
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error processing response {index}: {e}")
                    results[index] = {
                        "error": str(e),
                        "analysis": None,
                        "processing_time": 0.0
                    }
        
        return results
    
    def _analyze_single_response(self, response: str, test_case: Any) -> Dict:
        """Analyze a single response with caching"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(response, test_case)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result is not None:
            with self._lock:
                self._cache_stats["hits"] += 1
            return cached_result
        
        # Process if not in cache
        try:
            # This would call the actual enhanced evaluation
            from ..evaluation.unified_evaluator import UnifiedEvaluator
            evaluator = UnifiedEvaluator()
            
            result = evaluator.evaluate_with_comparison(response, test_case)
            
            analysis = {
                "comparison": result,
                "processing_time": time.time() - start_time,
                "cached": False
            }
            
            # Cache the result
            self._store_in_cache(cache_key, analysis)
            
            with self._lock:
                self._cache_stats["misses"] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return {
                "error": str(e),
                "analysis": None,
                "processing_time": time.time() - start_time,
                "cached": False
            }
    
    def _generate_cache_key(self, response: str, test_case: Any) -> str:
        """Generate a cache key for response and test case"""
        test_id = getattr(test_case, 'id', 'unknown')
        response_hash = hash(response)
        return f"{test_id}_{response_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get result from cache (placeholder for actual cache implementation)"""
        # In a real implementation, this would use Redis, memcached, or similar
        return None
    
    def _store_in_cache(self, cache_key: str, result: Dict) -> None:
        """Store result in cache (placeholder for actual cache implementation)"""
        # In a real implementation, this would store in Redis, memcached, or similar
        pass
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        with self._lock:
            total = self._cache_stats["hits"] + self._cache_stats["misses"]
            hit_rate = self._cache_stats["hits"] / total if total > 0 else 0.0
            
            return {
                "hits": self._cache_stats["hits"],
                "misses": self._cache_stats["misses"],
                "hit_rate": hit_rate,
                "total_requests": total
            }
    
    def clear_cache(self) -> None:
        """Clear all caches"""
        self._cached_dependency_parse.cache_clear()
        self._cached_intent_classification.cache_clear()
        
        with self._lock:
            self._cache_stats = {"hits": 0, "misses": 0}


class ModelManager:
    """Manages lazy loading and caching of NLP models"""
    
    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()
    
    def get_model(self, model_type: str, model_name: str = None):
        """Get or load a model with lazy loading"""
        
        with self._lock:
            if model_type not in self._models:
                self._models[model_type] = self._load_model(model_type, model_name)
            
            return self._models[model_type]
    
    def _load_model(self, model_type: str, model_name: str = None):
        """Load a specific model type"""
        
        if model_type == "spacy":
            import spacy
            try:
                return spacy.load("en_core_web_trf")
            except OSError:
                return spacy.load("en_core_web_sm")
        
        elif model_type == "transformers":
            from transformers import pipeline
            return pipeline("text-classification", 
                          model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                          return_all_scores=True)
        
        elif model_type == "intent_classifier":
            from ..models.intent_classifier import IntentClassifier
            return IntentClassifier()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def preload_models(self) -> None:
        """Preload all commonly used models"""
        models_to_preload = ["spacy", "transformers", "intent_classifier"]
        
        for model_type in models_to_preload:
            try:
                self.get_model(model_type)
                logger.info(f"Preloaded {model_type} model")
            except Exception as e:
                logger.warning(f"Failed to preload {model_type}: {e}")
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage statistics for loaded models"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "loaded_models": list(self._models.keys())
        }


# Global model manager instance
model_manager = ModelManager()


class PerformanceMonitor:
    """Monitors performance metrics for the evaluation pipeline"""
    
    def __init__(self):
        self.metrics = {
            "total_evaluations": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "accuracy": 0.0
        }
        self._lock = threading.Lock()
    
    def record_evaluation(
        self,
        processing_time: float,
        baseline_result: bool,
        advanced_result: bool,
        ground_truth: bool = None
    ) -> None:
        """Record metrics from an evaluation"""
        
        with self._lock:
            self.metrics["total_evaluations"] += 1
            self.metrics["total_time"] += processing_time
            self.metrics["average_time"] = (
                self.metrics["total_time"] / self.metrics["total_evaluations"]
            )
            
            # Calculate accuracy metrics if ground truth available
            if ground_truth is not None:
                # Update false positive/negative rates
                if baseline_result and not ground_truth:
                    # This would be tracked for false positive rate calculation
                    pass
                
                if not baseline_result and ground_truth:
                    # This would be tracked for false negative rate calculation
                    pass
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        with self._lock:
            return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self.metrics = {
                "total_evaluations": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "accuracy": 0.0
            }


# Global performance monitor
performance_monitor = PerformanceMonitor()
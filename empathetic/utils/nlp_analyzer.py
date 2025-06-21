"""Advanced NLP analysis utilities for dependency parsing and syntactic analysis"""
import spacy
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class DependencyResult:
    """Result of dependency parsing analysis"""
    dependency_path: List[str]
    governing_verb: Optional[str]
    negation_scope: List[tuple]  # [(start, end, negation_word)]
    clause_type: str
    pattern_tokens: List[Any]  # spaCy tokens


@dataclass
class NegationScope:
    """Represents a negation and its scope"""
    trigger: str
    scope_start: int
    scope_end: int
    scope_text: str
    confidence: float


class DependencyAnalyzer:
    """Analyzes syntactic dependencies to understand grammatical relationships"""
    
    def __init__(self):
        self._nlp = None
        self.negation_deps = {
            'neg',      # Direct negation
            'advmod',   # Adverbial modifier (for "never", "not")
            'aux',      # Auxiliary (for "won't", "can't")
        }
        
        self.negation_triggers = {
            'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'nobody',
            'cannot', "can't", "won't", "wouldn't", "shouldn't", "couldn't",
            "don't", "doesn't", "didn't", "hasn't", "haven't", "hadn't",
            'refuse', 'deny', 'reject', 'decline', 'avoid', 'prevent',
            'impossible', 'unable', 'incapable', 'unlikely'
        }
    
    @property
    def nlp(self):
        """Lazy loading of spaCy model"""
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_trf")
            except OSError:
                # Fallback to smaller model if transformer model not available
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    def analyze_dependencies(self, text: str, pattern: str) -> DependencyResult:
        """
        Analyze syntactic relationship between pattern and surrounding text
        
        Returns:
            - dependency_path: Path from pattern to root
            - governing_verb: Main verb governing the pattern
            - negation_scope: Which parts of sentence are negated
            - clause_type: Main, subordinate, relative, etc.
        """
        doc = self.nlp(text)
        pattern_tokens = self._find_pattern_tokens(doc, pattern)
        
        if not pattern_tokens:
            return DependencyResult(
                dependency_path=[],
                governing_verb=None,
                negation_scope=[],
                clause_type="unknown",
                pattern_tokens=[]
            )
        
        dependency_path = self._extract_dep_path(pattern_tokens[0])
        governing_verb = self._find_governing_verb(pattern_tokens[0])
        negation_scope = self._calculate_negation_scope(doc, pattern_tokens)
        clause_type = self._identify_clause_type(pattern_tokens[0])
        
        return DependencyResult(
            dependency_path=dependency_path,
            governing_verb=governing_verb,
            negation_scope=negation_scope,
            clause_type=clause_type,
            pattern_tokens=pattern_tokens
        )
    
    def _find_pattern_tokens(self, doc, pattern: str) -> List[Any]:
        """Find spaCy tokens that match the pattern"""
        pattern_lower = pattern.lower()
        pattern_tokens = []
        
        # Simple approach: find tokens that match pattern words
        pattern_words = pattern_lower.split()
        
        for i, token in enumerate(doc):
            if i + len(pattern_words) <= len(doc):
                token_span = doc[i:i + len(pattern_words)]
                if token_span.text.lower() == pattern_lower:
                    pattern_tokens.extend(token_span)
                    break
        
        # Fallback: find individual matching tokens
        if not pattern_tokens:
            for token in doc:
                if pattern_lower in token.text.lower():
                    pattern_tokens.append(token)
        
        return pattern_tokens
    
    def _extract_dep_path(self, token) -> List[str]:
        """Extract dependency path from token to root"""
        path = []
        current = token
        
        while current.head != current and len(path) < 10:  # Prevent infinite loops
            path.append(f"{current.text}({current.dep_})")
            current = current.head
        
        path.append(f"{current.text}(ROOT)")
        return path
    
    def _find_governing_verb(self, token) -> Optional[str]:
        """Find the main verb that governs this token"""
        current = token
        
        # Traverse up the dependency tree looking for a verb
        while current.head != current:
            if current.head.pos_ == "VERB":
                return current.head.lemma_
            current = current.head
        
        # Check if root is a verb
        if current.pos_ == "VERB":
            return current.lemma_
        
        return None
    
    def _calculate_negation_scope(self, doc, pattern_tokens) -> List[tuple]:
        """Calculate which parts of the sentence are negated"""
        negation_scopes = []
        
        for token in doc:
            if (token.dep_ in self.negation_deps or 
                token.text.lower() in self.negation_triggers):
                
                # Find the scope of this negation
                scope_start, scope_end = self._find_negation_scope(token)
                
                # Check if any pattern tokens fall within this scope
                for pattern_token in pattern_tokens:
                    if scope_start <= pattern_token.i <= scope_end:
                        negation_scopes.append((
                            scope_start,
                            scope_end,
                            token.text
                        ))
                        break
        
        return negation_scopes
    
    def _find_negation_scope(self, neg_token) -> tuple:
        """Find the scope of a negation token"""
        # Start from the negation token
        start = neg_token.i
        end = neg_token.i
        
        # The scope typically includes the head of the negation
        if neg_token.head != neg_token:
            end = max(end, neg_token.head.i)
            
            # Include children of the head that come after the negation
            for child in neg_token.head.children:
                if child.i > neg_token.i:
                    end = max(end, child.i)
        
        # Include children of the negation token
        for child in neg_token.children:
            end = max(end, child.i)
        
        return start, end
    
    def _identify_clause_type(self, token) -> str:
        """Identify the type of clause the token appears in"""
        # Look for clause markers in the dependency path
        current = token
        
        while current.head != current:
            if current.dep_ in ["ccomp", "xcomp"]:
                return "complement"
            elif current.dep_ == "relcl":
                return "relative"
            elif current.dep_ == "advcl":
                return "adverbial"
            elif current.dep_ == "csubj":
                return "subject"
            current = current.head
        
        return "main"


class NegationScopeAnalyzer:
    """Advanced negation scope detection"""
    
    def __init__(self):
        self._nlp = None
    
    @property
    def nlp(self):
        """Lazy loading of spaCy model"""
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_trf")
            except OSError:
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    def detect_negation_scope(self, text: str) -> List[NegationScope]:
        """
        Detect all negations and their exact scope
        
        Example:
            "I cannot and will not help with creating harmful content"
            Returns: [
                NegationScope(trigger="cannot", scope="help with creating harmful content"),
                NegationScope(trigger="will not", scope="help with creating harmful content")
            ]
        """
        doc = self.nlp(text)
        negations = self._find_negation_triggers(doc)
        
        scopes = []
        for neg in negations:
            scope = self._predict_scope(doc, neg)
            if scope:
                scopes.append(NegationScope(
                    trigger=neg.text,
                    scope_start=scope[0],
                    scope_end=scope[1],
                    scope_text=doc[scope[0]:scope[1] + 1].text,
                    confidence=0.8  # Default confidence
                ))
        
        return scopes
    
    def _find_negation_triggers(self, doc) -> List[Any]:
        """Find all negation triggers in the document"""
        negations = []
        
        for token in doc:
            if (token.dep_ == "neg" or 
                token.text.lower() in {"not", "no", "never", "cannot", "won't", 
                                     "can't", "don't", "doesn't", "didn't"}):
                negations.append(token)
        
        return negations
    
    def _predict_scope(self, doc, neg_token) -> Optional[tuple]:
        """Predict the scope of a negation token"""
        # Simple heuristic: scope includes the head and its children
        if neg_token.head == neg_token:
            return None
        
        head = neg_token.head
        start = min(neg_token.i, head.i)
        end = max(neg_token.i, head.i)
        
        # Extend to include children of the head
        for child in head.children:
            if child != neg_token:
                start = min(start, child.i)
                end = max(end, child.i)
        
        return start, end
    
    def is_pattern_negated(self, text: str, pattern: str) -> bool:
        """Check if specific pattern falls within negation scope"""
        scopes = self.detect_negation_scope(text)
        doc = self.nlp(text)
        
        # Find pattern position
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        pattern_start = text_lower.find(pattern_lower)
        
        if pattern_start == -1:
            return False
        
        # Convert character positions to token positions
        pattern_tokens = []
        char_count = 0
        for i, token in enumerate(doc):
            token_start = char_count
            token_end = char_count + len(token.text)
            
            if pattern_start >= token_start and pattern_start < token_end:
                # Pattern starts in this token
                pattern_tokens.append(i)
                break
            
            char_count = token_end + 1  # +1 for space
        
        # Check if pattern tokens fall within any negation scope
        for scope in scopes:
            for token_idx in pattern_tokens:
                if scope.scope_start <= token_idx <= scope.scope_end:
                    return True
        
        return False
"""
Custom Exception Classes for RAG Application

This module defines domain-specific exceptions for better error categorization
and handling throughout the RAG application.
"""


class RAGApplicationError(Exception):
    """Base exception for all RAG application errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(RAGApplicationError):
    """Raised when there are configuration issues."""
    pass


# ============================================================================
# Scraper Errors
# ============================================================================

class ScraperError(RAGApplicationError):
    """Base exception for scraper-related errors."""
    pass


class URLFetchError(ScraperError):
    """Raised when failing to fetch a URL."""
    pass


class ParsingError(ScraperError):
    """Raised when failing to parse content."""
    pass


class SitemapError(ScraperError):
    """Raised for sitemap-related errors."""
    pass


class ChunkingError(ScraperError):
    """Raised when text chunking fails."""
    pass


# ============================================================================
# Vector Store Errors
# ============================================================================

class VectorStoreError(RAGApplicationError):
    """Base exception for vector store errors."""
    pass


class EmbeddingError(VectorStoreError):
    """Raised when embedding generation or retrieval fails."""
    pass


class CollectionError(VectorStoreError):
    """Raised when collection operations fail."""
    pass


class QueryError(VectorStoreError):
    """Raised when query execution fails."""
    pass


class ValidationError(VectorStoreError):
    """Raised when input validation fails."""
    pass


# ============================================================================
# RAG Errors
# ============================================================================

class RAGError(RAGApplicationError):
    """Base exception for RAG operations."""
    pass


class GenerationError(RAGError):
    """Raised when answer generation fails."""
    pass


class ContextRetrievalError(RAGError):
    """Raised when context retrieval fails."""
    pass


class APIError(RAGError):
    """Raised when external API calls fail."""
    pass

"""
Chroma Vector Store Class with Persistent Storage
A production-ready implementation for document storage and retrieval
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Union, Any
import logging
from pathlib import Path
import uuid
import os
from dotenv import load_dotenv
from exceptions import (
    VectorStoreError,
    EmbeddingError,
    CollectionError,
    QueryError,
    ValidationError,
    ConfigurationError
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    A comprehensive vector store implementation using Chroma DB with persistent storage.
    
    Features:
    - Persistent storage of embeddings and documents
    - Multiple embedding function support
    - CRUD operations (Create, Read, Update, Delete)
    - Advanced querying with metadata filtering
    - Batch operations for efficiency
    """
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "../db/chroma_db",
        embedding_function: Optional[Any] = None,
        distance_metric: str = "cosine"
    ):
        """
        Initialize the Chroma Vector Store.
        
        Args:
            collection_name: Name of the collection to use/create
            persist_directory: Directory path where Chroma will persist data
            embedding_function: Custom embedding function (defaults to OpenAI)
            distance_metric: Distance metric for similarity search ("cosine", "l2", "ip")
            
        Raises:
            ConfigurationError: If configuration is invalid
            CollectionError: If collection initialization fails
        """
        # Validate collection name
        if not collection_name or not isinstance(collection_name, str):
            raise ValidationError(
                "collection_name must be a non-empty string",
                {"provided": collection_name}
            )
        
        # Validate collection name format (alphanumeric, hyphens, underscores)
        if not collection_name.replace('-', '').replace('_', '').isalnum():
            raise ValidationError(
                "collection_name must contain only alphanumeric characters, hyphens, and underscores",
                {"collection_name": collection_name}
            )
        
        # Validate distance metric
        valid_metrics = ["cosine", "l2", "ip"]
        if distance_metric not in valid_metrics:
            raise ValidationError(
                f"distance_metric must be one of {valid_metrics}",
                {"provided": distance_metric}
            )
        
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.distance_metric = distance_metric
        
        # Create persist directory if it doesn't exist
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise ConfigurationError(
                f"Permission denied creating directory: {self.persist_directory}",
                {"error": str(e)}
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create persist directory: {self.persist_directory}",
                {"error": str(e)}
            )
        
        # Initialize the persistent client
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            raise CollectionError(
                "Failed to initialize ChromaDB client",
                {"error": str(e), "path": str(self.persist_directory)}
            )
        
        # Set up embedding function
        if embedding_function is None:
            # Default: Use OpenAI embeddings
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Please set it or provide a custom embedding_function."
                )
            
            try:
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name="text-embedding-3-small"
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embedding function: {e}")
                raise EmbeddingError(
                    "Failed to initialize OpenAI embedding function",
                    {"error": str(e)}
                )
        else:
            self.embedding_function = embedding_function
        
        # Get or create collection
        try:
            self.collection = self._get_or_create_collection()
        except Exception as e:
            logger.error(f"Failed to get or create collection: {e}", exc_info=True)
            raise CollectionError(
                f"Failed to initialize collection '{collection_name}'",
                {"error": str(e)}
            )
        
        logger.info(f"Initialized ChromaVectorStore with collection: {collection_name}")
        logger.info(f"Persist directory: {self.persist_directory}")
        logger.info(f"Collection count: {self.collection.count()}")
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create a new one.
        
        Returns:
            ChromaDB collection
            
        Raises:
            CollectionError: If collection operations fail
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception as e:
            # Collection doesn't exist or error getting it
            error_msg = str(e).lower()
            
            # Check if it's a "collection doesn't exist" error
            if "does not exist" in error_msg or "not found" in error_msg:
                # Create the collection
                try:
                    metadata = {"hnsw:space": self.distance_metric}
                    collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function,
                        metadata=metadata
                    )
                    logger.info(f"Created new collection: {self.collection_name}")
                    return collection
                except Exception as create_error:
                    logger.error(f"Failed to create collection: {create_error}", exc_info=True)
                    raise CollectionError(
                        f"Failed to create collection '{self.collection_name}'",
                        {"error": str(create_error)}
                    )
            else:
                # Some other error
                logger.error(f"Unexpected error getting collection: {e}", exc_info=True)
                raise CollectionError(
                    f"Failed to get collection '{self.collection_name}'",
                    {"error": str(e)}
                )
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text documents to add
            metadatas: Optional list of metadata dictionaries for each document
            ids: Optional list of custom IDs (auto-generated if not provided)
        
        Returns:
            List of document IDs
            
        Raises:
            ValidationError: If input validation fails
            EmbeddingError: If embedding generation fails
            CollectionError: If adding to collection fails
        """
        if not documents:
            raise ValidationError("Documents list cannot be empty")
        
        if not isinstance(documents, list):
            raise ValidationError(
                "Documents must be a list",
                {"type": type(documents).__name__}
            )
        
        # Validate all documents are non-empty strings
        for idx, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise ValidationError(
                    f"Document at index {idx} is not a string",
                    {"type": type(doc).__name__, "index": idx}
                )
            if not doc.strip():
                raise ValidationError(
                    f"Document at index {idx} is empty or whitespace",
                    {"index": idx}
                )
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Ensure IDs are unique
        if len(ids) != len(set(ids)):
            raise ValidationError(
                "Document IDs must be unique",
                {"total_ids": len(ids), "unique_ids": len(set(ids))}
            )
        
        # Create empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Validate lengths match
        if len(documents) != len(metadatas) or len(documents) != len(ids):
            raise ValidationError(
                "Documents, metadatas, and ids must have the same length",
                {
                    "documents": len(documents),
                    "metadatas": len(metadatas),
                    "ids": len(ids)
                }
            )
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}", exc_info=True)
            
            # Check for specific error types
            error_msg = str(e).lower()
            if "duplicate" in error_msg or "already exists" in error_msg:
                raise CollectionError(
                    "Duplicate document IDs detected",
                    {"error": str(e)}
                )
            elif "embedding" in error_msg:
                raise EmbeddingError(
                    "Failed to generate embeddings for documents",
                    {"error": str(e), "document_count": len(documents)}
                )
            else:
                raise CollectionError(
                    "Failed to add documents to collection",
                    {"error": str(e), "document_count": len(documents)}
                )
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add pre-computed embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of text documents (for reference)
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of custom IDs
        
        Returns:
            List of document IDs
        """
        if not embeddings or not documents:
            raise ValueError("Embeddings and documents lists cannot be empty")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        if len(embeddings) != len(documents) or len(documents) != len(ids):
            raise ValueError("Embeddings, documents, and ids must have the same length")
        
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(embeddings)} embeddings to collection")
            return ids
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            raise
    
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict:
        """
        Query the vector store for similar documents.
        
        Args:
            query_texts: List of query texts (will be embedded automatically)
            query_embeddings: List of pre-computed query embeddings
            n_results: Number of results to return
            where: Metadata filter (e.g., {"category": "science"})
            where_document: Document content filter
            include: What to include in results
        
        Returns:
            Dictionary containing query results
            
        Raises:
            ValidationError: If query parameters are invalid
            QueryError: If query execution fails
        """
        if query_texts is None and query_embeddings is None:
            raise ValidationError("Either query_texts or query_embeddings must be provided")
        
        if query_texts is not None and query_embeddings is not None:
            raise ValidationError("Provide either query_texts or query_embeddings, not both")
        
        # Validate n_results
        if not isinstance(n_results, int) or n_results <= 0:
            raise ValidationError(
                "n_results must be a positive integer",
                {"provided": n_results}
            )
        
        # Check if collection is empty
        try:
            count = self.collection.count()
            if count == 0:
                logger.warning("Query on empty collection")
                return {
                    'ids': [[]],
                    'documents': [[]],
                    'metadatas': [[]],
                    'distances': [[]]
                }
        except Exception as e:
            logger.error(f"Error checking collection count: {e}")
        
        try:
            results = self.collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            
            result_count = len(results.get('ids', [[]])[0]) if results.get('ids') else 0
            logger.info(f"Query returned {result_count} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying collection: {e}", exc_info=True)
            
            error_msg = str(e).lower()
            if "embedding" in error_msg:
                raise EmbeddingError(
                    "Failed to generate embeddings for query",
                    {"error": str(e)}
                )
            elif "where" in error_msg or "filter" in error_msg:
                raise QueryError(
                    "Invalid filter clause in query",
                    {"error": str(e), "where": where}
                )
            else:
                raise QueryError(
                    "Failed to execute query",
                    {"error": str(e)}
                )
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: List[str] = ["documents", "metadatas", "embeddings"]
    ) -> Dict:
        """
        Get documents from the vector store by ID or filter.
        
        Args:
            ids: List of document IDs to retrieve
            where: Metadata filter
            limit: Maximum number of results
            offset: Number of results to skip
            include: What to include in results
        
        Returns:
            Dictionary containing documents
        """
        try:
            results = self.collection.get(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset,
                include=include
            )
            logger.info(f"Retrieved {len(results.get('ids', []))} documents")
            return results
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            raise
    
    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        Update existing documents in the vector store.
        
        Args:
            ids: List of document IDs to update
            documents: Optional new document texts
            metadatas: Optional new metadata
            embeddings: Optional new embeddings
        """
        try:
            self.collection.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            logger.info(f"Updated {len(ids)} documents")
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None
    ):
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
            where: Metadata filter for deletion
        """
        if ids is None and where is None:
            raise ValueError("Either ids or where must be provided")
        
        try:
            self.collection.delete(
                ids=ids,
                where=where
            )
            logger.info(f"Deleted documents from collection")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def peek(self, limit: int = 10) -> Dict:
        """
        Peek at a few documents in the collection.
        
        Args:
            limit: Number of documents to peek at
        
        Returns:
            Dictionary containing sample documents
        """
        return self.collection.peek(limit=limit)
    
    def reset(self):
        """Delete all documents from the collection."""
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            # Recreate it
            self.collection = self._get_or_create_collection()
            logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform similarity search and return results in a simplified format.
        
        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter
        
        Returns:
            List of dictionaries with document, metadata, and score
        """
        results = self.query(
            query_texts=[query],
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        return {
            'name': self.collection_name,
            'count': self.count(),
            'persist_directory': str(self.persist_directory),
            'distance_metric': self.distance_metric
        }


# Example usage
if __name__ == "__main__":
    # Initialize the vector store
    vectorstore = ChromaVectorStore(
        collection_name="my_documents",
        persist_directory="./my_chroma_db"
    )
    
    # Add documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a great programming language",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing enables computers to understand human language"
    ]
    
    metadatas = [
        {"category": "animals", "source": "example1"},
        {"category": "programming", "source": "example2"},
        {"category": "ai", "source": "example3"},
        {"category": "ai", "source": "example4"}
    ]
    
    # Add documents to the store
    doc_ids = vectorstore.add_documents(documents=documents, metadatas=metadatas)
    print(f"Added documents with IDs: {doc_ids}")
    
    # Query the store
    print("\n--- Similarity Search ---")
    results = vectorstore.similarity_search(
        query="What is AI?",
        k=2
    )
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Document: {result['document']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']:.4f}")
    
    # Query with metadata filter
    print("\n--- Filtered Search (category=ai) ---")
    filtered_results = vectorstore.query(
        query_texts=["programming languages"],
        n_results=2,
        where={"category": "ai"}
    )
    print(f"Found {len(filtered_results['ids'][0])} results in 'ai' category")
    
    # Get collection info
    print("\n--- Collection Info ---")
    info = vectorstore.get_collection_info()
    print(f"Collection Name: {info['name']}")
    print(f"Total Documents: {info['count']}")
    print(f"Persist Directory: {info['persist_directory']}")
    
    # Update a document
    print("\n--- Updating Document ---")
    vectorstore.update(
        ids=[doc_ids[0]],
        metadatas=[{"category": "animals", "source": "updated_example"}]
    )
    print("Document metadata updated")
    
    # Peek at collection
    print("\n--- Peek at Collection ---")
    peek_data = vectorstore.peek(limit=2)
    print(f"Peeking at {len(peek_data['ids'])} documents")
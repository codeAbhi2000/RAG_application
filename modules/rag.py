"""
RAG (Retrieval-Augmented Generation) Module

This module provides functionality for generating answers using OpenAI's API
based on retrieved context from the vector store.
"""

from openai import OpenAI
import logging
import time
from typing import Optional
from exceptions import GenerationError, APIError, ConfigurationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_answer(
    query: str,
    context: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    timeout: int = 60
) -> str:
    """
    Generates an answer using OpenAI based on the retrieved context.
    
    Args:
        query: User's question
        context: Retrieved context from vector store
        api_key: OpenAI API key
        model: OpenAI model to use (default: gpt-3.5-turbo)
        max_retries: Maximum number of retry attempts for transient errors
        timeout: Timeout in seconds for API calls
        
    Returns:
        Generated answer as a string
        
    Raises:
        ConfigurationError: If API key or inputs are invalid
        APIError: If OpenAI API call fails
        GenerationError: If answer generation fails
    """
    # Validate inputs
    if not api_key or not isinstance(api_key, str):
        raise ConfigurationError(
            "Invalid API key: must be a non-empty string",
            {"api_key_provided": bool(api_key)}
        )
    
    if not query or not isinstance(query, str):
        raise GenerationError(
            "Invalid query: must be a non-empty string",
            {"query_type": type(query).__name__}
        )
    
    if not context or not isinstance(context, str):
        raise GenerationError(
            "Invalid context: must be a non-empty string",
            {"context_type": type(context).__name__}
        )
    
    query = query.strip()
    context = context.strip()
    
    if not query:
        raise GenerationError("Query cannot be empty after stripping whitespace")
    
    if not context:
        raise GenerationError("Context cannot be empty after stripping whitespace")
    
    logger.info(f"Generating answer for query: '{query[:100]}...'")
    logger.debug(f"Using model: {model}, Context length: {len(context)} chars")
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key, timeout=timeout)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise ConfigurationError(
            "Failed to initialize OpenAI client",
            {"error": str(e)}
        )
    
    # Prepare prompt
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question. 
    If the answer is not in the context, say "I don't know based on the provided context."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    # Retry logic for transient errors
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"API call attempt {attempt}/{max_retries}")
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                timeout=timeout
            )
            
            elapsed_time = time.time() - start_time
            
            # Validate response
            if not response or not response.choices:
                raise GenerationError(
                    "Empty or invalid response from OpenAI API",
                    {"response": str(response)}
                )
            
            answer = response.choices[0].message.content
            
            if not answer:
                raise GenerationError("Generated answer is empty")
            
            # Log success metrics
            logger.info(
                f"Successfully generated answer in {elapsed_time:.2f}s "
                f"(attempt {attempt}/{max_retries})"
            )
            
            if hasattr(response, 'usage'):
                logger.debug(
                    f"Token usage - Prompt: {response.usage.prompt_tokens}, "
                    f"Completion: {response.usage.completion_tokens}, "
                    f"Total: {response.usage.total_tokens}"
                )
            
            return answer
            
        except Exception as e:
            last_error = e
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Handle specific OpenAI errors
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                logger.warning(
                    f"Rate limit error on attempt {attempt}/{max_retries}. "
                    f"Waiting {wait_time}s before retry..."
                )
                if attempt < max_retries:
                    time.sleep(wait_time)
                    continue
                else:
                    raise APIError(
                        "Rate limit exceeded after all retry attempts",
                        {"attempts": max_retries, "error": error_msg}
                    )
            
            elif "authentication" in error_msg.lower() or "401" in error_msg:
                logger.error("Authentication error - invalid API key")
                raise ConfigurationError(
                    "Invalid OpenAI API key",
                    {"error": error_msg}
                )
            
            elif "timeout" in error_msg.lower():
                logger.warning(
                    f"Timeout error on attempt {attempt}/{max_retries}: {error_msg}"
                )
                if attempt < max_retries:
                    continue
                else:
                    raise APIError(
                        f"Request timeout after {max_retries} attempts",
                        {"timeout": timeout, "error": error_msg}
                    )
            
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                logger.warning(
                    f"Connection error on attempt {attempt}/{max_retries}: {error_msg}"
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise APIError(
                        "Connection error after all retry attempts",
                        {"attempts": max_retries, "error": error_msg}
                    )
            
            elif "invalid_request" in error_msg.lower() or "400" in error_msg:
                logger.error(f"Invalid request error: {error_msg}")
                raise GenerationError(
                    "Invalid request to OpenAI API",
                    {"error": error_msg, "model": model}
                )
            
            else:
                # Unknown error
                logger.error(
                    f"Unexpected error on attempt {attempt}/{max_retries}: "
                    f"{error_type} - {error_msg}",
                    exc_info=True
                )
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                else:
                    raise APIError(
                        f"API call failed after {max_retries} attempts",
                        {
                            "error": error_msg,
                            "error_type": error_type,
                            "attempts": max_retries
                        }
                    )
    
    # This should never be reached, but just in case
    raise APIError(
        "Failed to generate answer after all retry attempts",
        {"error": str(last_error), "attempts": max_retries}
    )

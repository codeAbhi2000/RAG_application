from vector_store import ChromaVectorStore
import requests
from bs4 import BeautifulSoup
import re
import lxml
from dotenv import load_dotenv
import os
from openai import OpenAI
import logging
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from exceptions import (
    ScraperError,
    URLFetchError,
    ParsingError,
    SitemapError,
    ChunkingError,
    ConfigurationError
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate environment variables
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY not found in environment variables")


class Scraper:
    def __init__(self, baseUrl: str, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the Scraper.
        
        Args:
            baseUrl: Base URL of the website to scrape
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
            
        Raises:
            ConfigurationError: If parameters are invalid
        """
        # Validate base URL
        if not baseUrl or not isinstance(baseUrl, str):
            raise ConfigurationError("baseUrl must be a non-empty string")
        
        try:
            parsed = urlparse(baseUrl)
            if not all([parsed.scheme, parsed.netloc]):
                raise ConfigurationError(
                    f"Invalid URL format: {baseUrl}. Must include scheme (http/https) and domain"
                )
        except Exception as e:
            raise ConfigurationError(f"Invalid URL: {baseUrl}", {"error": str(e)})
        
        # Validate chunk parameters
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ConfigurationError("chunk_size must be a positive integer")
        
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be a non-negative integer")
        
        if chunk_overlap >= chunk_size:
            raise ConfigurationError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        
        self.baseUrl = baseUrl.rstrip('/')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Setup requests session with retry logic
        self.session = self._create_session()
        
        logger.info(f"Initialized Scraper for {self.baseUrl} with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic and timeout handling.
        
        Returns:
            Configured requests Session
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def prepare_for_embedding(self, data: List[Dict]) -> Dict[str, List]:
        """
        Prepare data for embeddings by separating documents and creating metadata for each document.
        
        Args:
            data: List of page data dictionaries
            
        Returns:
            Dictionary with 'documents' and 'metadata' lists
            
        Raises:
            ScraperError: If data preparation fails
        """
        if not data:
            logger.warning("Empty data provided for embedding preparation")
            return {"documents": [], "metadata": []}
        
        if not isinstance(data, list):
            raise ScraperError(
                "Invalid data format: expected list of page data",
                {"type": type(data).__name__}
            )
        
        try:
            documents = []
            metadata = []

            for idx, page in enumerate(data):
                if not isinstance(page, dict):
                    logger.warning(f"Skipping invalid page data at index {idx}: not a dictionary")
                    continue
                
                if "sections_data" not in page or "page_url" not in page:
                    logger.warning(f"Skipping page at index {idx}: missing required fields")
                    continue
                
                for section in page["sections_data"]:
                    if not isinstance(section, dict):
                        continue
                    
                    section_title = section.get("section_title", "unknown")
                    section_content = section.get("section_content", [])
                    
                    if isinstance(section_content, str):
                        section_content = [section_content]
                    
                    for document in section_content:
                        if document and isinstance(document, str) and document.strip():
                            documents.append(document)
                            metadata.append({
                                "page_url": page["page_url"],
                                "section_title": section_title
                            })
            
            logger.info(f"Prepared {len(documents)} documents for embedding")
            return {"documents": documents, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error preparing data for embedding: {e}", exc_info=True)
            raise ScraperError(
                "Failed to prepare data for embedding",
                {"error": str(e), "data_length": len(data)}
            )    

    def create_documents(self, sections: List[Dict]) -> List[Dict]:
        """
        Create documents from the scraped data by chunking section content.
        
        Args:
            sections: List of section dictionaries
            
        Returns:
            List of sections with chunked content
            
        Raises:
            ScraperError: If document creation fails
        """
        if not sections:
            logger.warning("Empty sections provided for document creation")
            return []
        
        if not isinstance(sections, list):
            raise ScraperError(
                "Invalid sections format: expected list",
                {"type": type(sections).__name__}
            )
        
        try:
            newSections = []
            errors = []
            
            for idx, section in enumerate(sections):
                if not isinstance(section, dict):
                    logger.warning(f"Skipping invalid section at index {idx}: not a dictionary")
                    continue
                
                if "section_content" not in section:
                    logger.warning(f"Skipping section at index {idx}: missing section_content")
                    continue
                
                try:
                    documents = self.chunk_data_sematically(section["section_content"])
                    section["section_content"] = documents
                    newSections.append(section)
                except ChunkingError as e:
                    logger.error(f"Error chunking section {idx}: {e}")
                    errors.append({"section_index": idx, "error": str(e)})
                    continue

            if errors:
                logger.warning(f"Encountered {len(errors)} errors while creating documents")
            
            logger.info(f"Created documents from {len(newSections)} sections")
            return newSections
                
        except Exception as e:
            logger.error(f"Error creating documents: {e}", exc_info=True)
            raise ScraperError(
                "Failed to create documents from sections",
                {"error": str(e), "sections_count": len(sections)}
            )    

    def start_scraping(self) -> Dict[str, List]:
        """
        Start scraping the website.
        
        Returns:
            Dictionary with prepared documents and metadata
            
        Raises:
            ScraperError: If scraping fails
        """
        logger.info(f"Starting scraping process for {self.baseUrl}")
        start_time = time.time()
        
        try:
            # Get all links from sitemap
            links = self.get_links()
            
            if not links:
                logger.warning(f"No links found for {self.baseUrl}")
                return {"documents": [], "metadata": []}
            
            data = []
            failed_links = []
            
            for idx, link in enumerate(links, 1):
                logger.info(f"Processing link {idx}/{len(links)}: {link}")
                
                try:
                    page_data = self.scrape_website(link)
                    
                    if page_data and "sections_data" in page_data:
                        sections = self.create_documents(page_data['sections_data'])
                        page_data['sections_data'] = sections
                        data.append(page_data)
                    else:
                        logger.warning(f"No valid data from {link}")
                        failed_links.append({"url": link, "reason": "No valid page data"})
                        
                except Exception as e:
                    logger.error(f"Failed to process {link}: {e}")
                    failed_links.append({"url": link, "reason": str(e)})
                    continue
            
            if failed_links:
                logger.warning(f"Failed to process {len(failed_links)} out of {len(links)} links")
            
            # Prepare data for embedding
            result = self.prepare_for_embedding(data)
            
            elapsed_time = time.time() - start_time
            logger.info(
                f"Scraping completed in {elapsed_time:.2f}s. "
                f"Processed {len(data)}/{len(links)} pages successfully. "
                f"Generated {len(result['documents'])} documents."
            )
            
            return result
            
        except SitemapError as e:
            logger.error(f"Sitemap error for {self.baseUrl}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error occurred while scraping {self.baseUrl}: {e}", exc_info=True)
            raise ScraperError(
                f"Failed to scrape {self.baseUrl}",
                {"error": str(e), "error_type": type(e).__name__}
            )    

    def get_links(self) -> List[str]:
        """
        Fetch all the pages from base url sitemap.
        
        Returns:
            List of URLs from sitemap
            
        Raises:
            SitemapError: If sitemap cannot be fetched or parsed
        """
        sitemap_url = f"{self.baseUrl}/sitemap.xml"
        logger.info(f"Fetching sitemap from {sitemap_url}")
        
        try:
            response = self.session.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            raise SitemapError(
                f"Timeout while fetching sitemap from {sitemap_url}",
                {"timeout": 30}
            )
        except requests.exceptions.ConnectionError as e:
            raise SitemapError(
                f"Connection error while fetching sitemap from {sitemap_url}",
                {"error": str(e)}
            )
        except requests.exceptions.HTTPError as e:
            raise SitemapError(
                f"HTTP error {e.response.status_code} while fetching sitemap",
                {"url": sitemap_url, "status_code": e.response.status_code}
            )
        except requests.exceptions.RequestException as e:
            raise SitemapError(
                f"Request error while fetching sitemap from {sitemap_url}",
                {"error": str(e)}
            )
        
        # Parse the XML content
        try:
            soup = BeautifulSoup(response.content, features="xml")
            
            # Find all <loc> tags which contain the URLs
            loc_tags = soup.find_all('loc')
            
            if not loc_tags:
                logger.warning(f"No <loc> tags found in sitemap at {sitemap_url}")
                return []
            
            links = [loc.get_text().strip() for loc in loc_tags if loc.get_text().strip()]
            
            logger.info(f"Found {len(links)} links in sitemap")
            for link in links:
                logger.debug(f"  - {link}")
            
            return links
            
        except Exception as e:
            logger.error(f"Error parsing sitemap XML: {e}", exc_info=True)
            raise ParsingError(
                f"Failed to parse sitemap XML from {sitemap_url}",
                {"error": str(e)}
            )    

    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing noise and condensing whitespace.
        
        Args:
            text: Raw text with newlines
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
                
            # Skip lines that are just punctuation or single numbers (likely footnotes)
            if len(line) <= 2 and not line.isalpha():
                continue
                
            if line in ['|', '•', '-', '1', '2', '3', '4', '5']: # Common list markers/footnotes
                continue
            
            cleaned_lines.append(line)
            
        # Join with newlines, but maybe group them?
        # For now, keep newlines to preserve structure, but removing the "sparse" noise
        return '\n'.join(cleaned_lines)

    def scrape_website(self, url: str) -> Optional[Dict]:
        """
        Fetches the content of a website and returns the cleaned text.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with page data or None if scraping fails
            
        Raises:
            URLFetchError: If URL cannot be fetched
            ParsingError: If content cannot be parsed
        """
        if not url or not isinstance(url, str):
            raise URLFetchError("Invalid URL provided", {"url": url})
        
        logger.debug(f"Scraping website: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                logger.warning(f"Unexpected content type for {url}: {content_type}")
            
        except requests.exceptions.Timeout:
            raise URLFetchError(
                f"Timeout while fetching {url}",
                {"timeout": 30}
            )
        except requests.exceptions.ConnectionError as e:
            raise URLFetchError(
                f"Connection error while fetching {url}",
                {"error": str(e)}
            )
        except requests.exceptions.HTTPError as e:
            raise URLFetchError(
                f"HTTP error {e.response.status_code} for {url}",
                {"status_code": e.response.status_code, "url": url}
            )
        except requests.exceptions.RequestException as e:
            raise URLFetchError(
                f"Request error while fetching {url}",
                {"error": str(e)}
            )
        
        # Parse HTML content
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text section wise
            sections = soup.find_all("section")
            references = soup.find("div", class_="css-178yklu")

            logger.debug(f"Found {len(sections)} sections in {url}")

            page_data = {
                "page_url": url,
                "sections_data": []
            }

            if references:
                reference_data = references.get_text()
                if reference_data.strip():
                    page_data["sections_data"].append({
                        "section_title": "references",
                        "section_content": reference_data
                    })

            for section in sections:
                # Remove sections with no text
                section_text = section.get_text(strip=True)
                if section_text:
                    section_title = section.get("id") or "untitled"
                    # Get text as structure so it will help in semantic chunking
                    # Use newline separator to preserve paragraph structure
                    raw_content = section.get_text(separator="\n", strip=True)
                    
                    # Clean the content to remove sparse noise
                    section_content = self.clean_text(raw_content)
                    
                    # Filter out very short sections that are likely nav/footer noise
                    if len(section_content) > 50:
                        page_data["sections_data"].append({
                            "section_title": section_title,
                            "section_content": section_content
                        })
            
            if not page_data["sections_data"]:
                logger.warning(f"No sections found in {url}")
            
            return page_data
            
        except Exception as e:
            logger.error(f"Error parsing content from {url}: {e}", exc_info=True)
            raise ParsingError(
                f"Failed to parse HTML content from {url}",
                {"error": str(e)}
            )


    def chunk_data_sematically(self, data: str) -> List[str]:
        """
        Chunks the data semantically based on natural text boundaries.
        
        Args:
            data: Text data to chunk
            
        Returns:
            List of text chunks
            
        Raises:
            ChunkingError: If chunking fails
        """
        if not data:
            logger.debug("Empty data provided for chunking")
            return []
        
        if not isinstance(data, str):
            raise ChunkingError(
                "Invalid data type for chunking: expected string",
                {"type": type(data).__name__}
            )
        
        try:
            # Chunk data by paragraphs
            chunks = []
            data = data.strip("\n")
            text_len = len(data)
            
            if text_len == 0:
                return []
            
            # If text is shorter than chunk size, return as single chunk 
            # (only if it's meaningful size, otherwise it might be noise, but let's keep > 50 check at section level)
            if text_len <= self.chunk_size:
                return [data.strip()] if data.strip() else []
            
            start = 0
            while start < text_len:
                end = start + self.chunk_size
                # If we are not at the end, try to find the last space/newline to avoid cutting words
                if end < text_len:
                    # Priority 1: Double Newline (Paragraph break)
                    last_para = data.rfind('\n\n', start, end)
                    if last_para != -1 and last_para > start + (self.chunk_size * 0.5):
                         end = last_para + 2
                    else:
                        # Priority 2: Single Newline
                        last_newline = data.rfind('\n', start, end)
                        if last_newline != -1 and last_newline > start + (self.chunk_size * 0.5):
                            end = last_newline + 1
                        else:
                            # Priority 3: Sentence
                            last_space = data.rfind('.', start, end)
                            if last_space != -1 and last_space > start + (self.chunk_size * 0.5):
                                end = last_space + 1
                            else:
                                # Priority 4: Space
                                last_word_end = data.rfind(' ', start, end)
                                if last_word_end != -1 and last_word_end > start + (self.chunk_size * 0.5):
                                    end = last_word_end + 1

                chunk_text = data[start:end].strip()
                
                # Filter out very short chunks (likely noise/fragments)
                if len(chunk_text) > 50:
                    chunks.append(chunk_text)
                
                # Move start forward
                if end >= text_len:
                    break
                
                # Calculate new start position based on overlap
                # But ensure we don't just jump back into the middle of a word if we split by space
                # For simplicity, we just subtract overlap
                start += (self.chunk_size - self.chunk_overlap)
                
                # Correction: If simple subtraction puts us behind the current end (which it should), 
                # effectively we are just moving the window.
                # However, we want the next chunk to start cleanly if possible.
                # Let's trust the overlap logic for now as it's standard sliding window.
                
                # Optimization: if we are stuck (start didn't move enough), force move
                if start >= end: 
                     start = end
            
            logger.debug(f"Created {len(chunks)} chunks from {text_len} characters")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking data: {e}", exc_info=True)
            raise ChunkingError(
                "Failed to chunk text data",
                {"error": str(e), "data_length": len(data) if isinstance(data, str) else "unknown"}
            )

if __name__ == "__main__":
    # crawler("https://www.orserduhcp.com")
    # scraper = Scraper("https://www.orserduhcp.com",chunk_size=500,chunk_overlap=50)
    # data = scraper.start_scraping()
    # print(len(data["documents"]),len(data["metadata"]))
    # print(data["documents"][20])
    # print(os.getenv("OPENAI_API_KEY"))
    

    vector_store = ChromaVectorStore("orserduhcp",persist_directory="../data/chroma_db")
    
    # Reset to avoid duplicates
    vector_store.reset()
    vector_store.add_documents(data["documents"],data["metadata"])
    # print("Data added to vector store successfully")

    query = "what about efficacy of orserdu"

    results = vector_store.query(query)
    print(results["documents"][0])
    
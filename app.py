"""
RAG Q&A Application - Orserdu HCP Website

A dedicated Q&A interface for the Orserdu HCP website content.
"""

import streamlit as st
import os
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from modules.scraper import Scraper
from modules.vector_store import ChromaVectorStore
from modules.rag import generate_answer
from modules.exceptions import (
    ScraperError,
    VectorStoreError,
    RAGError,
    ConfigurationError,
    URLFetchError,
    SitemapError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Hardcoded for Orserdu HCP
# ============================================================================
WEBSITE_URL = "https://www.orserduhcp.com"
COLLECTION_NAME = "orserduhcp"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
PERSISTENT_PATH = "../data/chroma_db"

# Load API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")

# Page configuration
st.set_page_config(
    page_title="Orserdu HCP Q&A",
    page_icon="",
    layout="wide"
)

# Initialize session state
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = COLLECTION_NAME
if 'collection_exists' not in st.session_state:
    st.session_state.collection_exists = False
if 'doc_count' not in st.session_state:
    st.session_state.doc_count = 0
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Helper Functions
def check_collection_exists(collection_name: str) -> tuple:
    """
    Check if a collection exists and return its status.
    
    Returns:
        tuple: (exists: bool, doc_count: int, vector_store: ChromaVectorStore or None)
    """
    try:
        vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=PERSISTENT_PATH
        )
        doc_count = vector_store.count()
        exists = doc_count > 0
        logger.info(f"Collection '{collection_name}' exists with {doc_count} documents")
        return exists, doc_count, vector_store
    except Exception as e:
        logger.error(f"Error checking collection: {e}")
        return False, 0, None


def scrape_and_index() -> tuple:
    """
    Scrape the Orserdu HCP website and create/update the collection.
    
    Returns:
        tuple: (success: bool, message: str, doc_count: int)
    """
    try:
        # Initialize scraper with hardcoded configuration
        scraper = Scraper(
            baseUrl=WEBSITE_URL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        logger.info(f"Scraper initialized for {WEBSITE_URL}")
        
        # Start scraping
        data = scraper.start_scraping()
        
        if not data or not data.get('documents'):
            return False, "No content found to index", 0
        
        doc_count = len(data['documents'])
        logger.info(f"Scraped {doc_count} chunks from {WEBSITE_URL}")
        
        # Initialize vector store
        vector_store = ChromaVectorStore(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSISTENT_PATH
        )
        
        # Reset the collection to avoid duplicates
        logger.info(f"Resetting collection '{COLLECTION_NAME}' before indexing")
        vector_store.reset()
        
        # Add documents
        vector_store.add_documents(
            documents=data['documents'],
            metadatas=data['metadata']
        )
        
        logger.info(f"Indexed {doc_count} chunks in collection '{COLLECTION_NAME}'")
        return True, f"Successfully indexed {doc_count} documents from Orserdu HCP website", doc_count
        
    except Exception as e:
        logger.error(f"Error scraping and indexing: {e}", exc_info=True)
        return False, f"Error: {str(e)}", 0


# ============================================================================
# SIDEBAR - Collection Management
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Display API key status
    if OPENAI_API_KEY:
        st.success("✅ API Key loaded from .env")
    else:
        st.error("❌ API Key not found in .env file")
        st.info("Please add OPENAI_API_KEY to your .env file")
    
    st.divider()
    
    # Collection Status
    st.subheader("📦 Collection Status")
    
    # Display website info
    st.info(f"🌐 **Website:** Orserdu HCP")
    st.caption(f"URL: {WEBSITE_URL}")
    
    # Check Collection Button
    if st.button("🔍 Check Collection", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("❌ API key not found in .env file")
        else:
            with st.spinner("Checking collection..."):
                exists, doc_count, vector_store = check_collection_exists(COLLECTION_NAME)
                st.session_state.collection_exists = exists
                st.session_state.doc_count = doc_count
                st.session_state.vector_store = vector_store
                
                if exists:
                    st.success(f"✅ Collection found!")
                else:
                    st.warning("❌ Collection not found")
    
    # Display Collection Status
    st.markdown("**Status:**")
    if st.session_state.collection_exists:
        st.success(f"✅ Collection exists")
        st.info(f"📊 **{st.session_state.doc_count}** documents")
    else:
        st.error("❌ Collection not found")
    
    st.divider()
    
    # Scrape & Index Section
    st.subheader("🔄 Scrape & Index")
    
    if not st.session_state.collection_exists:
        st.markdown("Create the collection by scraping the Orserdu HCP website")
    else:
        st.markdown("Re-scrape to update the collection with latest content")
    
    if st.button("🚀 Scrape & Index Website", type="primary", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("❌ API key not found in .env file")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🌐 Initializing scraper...")
                progress_bar.progress(20)
                
                status_text.text("📄 Scraping Orserdu HCP website...")
                progress_bar.progress(40)
                
                success, message, doc_count = scrape_and_index()
                
                progress_bar.progress(80)
                
                if success:
                    status_text.text("✅ Indexing complete!")
                    progress_bar.progress(100)
                    st.success(message)
                    
                    # Update session state
                    st.session_state.collection_exists = True
                    st.session_state.doc_count = doc_count
                    
                    # Re-check collection
                    exists, doc_count, vector_store = check_collection_exists(COLLECTION_NAME)
                    st.session_state.vector_store = vector_store
                    
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error in scraping: {e}", exc_info=True)
            finally:
                progress_bar.empty()
                status_text.empty()
    
    st.divider()
    
    # Database Management
    # Danger Zone
    st.subheader("⚠️ Danger Zone")
    
    if st.button("🗑️ Reset Collection", type="secondary", use_container_width=True):
        try:
            # Get or initialize vector store
            vs = st.session_state.vector_store
            if vs is None:
                vs = ChromaVectorStore(
                    collection_name=COLLECTION_NAME,
                    persist_directory=PERSISTENT_PATH
                )
                st.session_state.vector_store = vs
            
            # Reset the collection
            vs.reset()
            
            st.success("✅ Collection reset!")
            
            # Update session state
            st.session_state.collection_exists = False
            st.session_state.doc_count = 0
            
            logger.info(f"Collection '{COLLECTION_NAME}' reset by user")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to reset collection: {str(e)}")
            logger.error(f"Error resetting collection: {e}", exc_info=True)

# ============================================================================
# MAIN AREA - Chat Interface
# ============================================================================

# Header
col1, col2 = st.columns([4, 1])
with col1:
    st.title("💊 Orserdu HCP Q&A Assistant")
with col2:
    if st.session_state.collection_exists:
        st.success(f"📚 {st.session_state.doc_count} docs")

st.markdown("Ask questions about Orserdu and get answers with source citations from the HCP website.")

# Check prerequisites
if not OPENAI_API_KEY:
    st.error("❌ OpenAI API Key not found")
    st.info("""
    **Setup Required:**
    1. Add `OPENAI_API_KEY` to your `.env` file
    2. Restart the application
    3. Check if collection exists
    4. Scrape & index if needed
    5. Start asking questions!
    """)
    st.stop()

if not st.session_state.collection_exists:
    st.info("ℹ️ No collection found. Please create a collection in the sidebar first.")
    st.markdown("""
    **To create a collection:**
    1. Go to the sidebar
    2. Enter a website URL
    3. Click "Create Collection"
    4. Wait for the scraping to complete
    """)
    st.stop()

# Initialize vector store if not already done
if st.session_state.vector_store is None:
    try:
        with st.spinner("Initializing vector store..."):
            exists, doc_count, vector_store = check_collection_exists(
                COLLECTION_NAME
            )
            st.session_state.vector_store = vector_store
            st.session_state.doc_count = doc_count
    except Exception as e:
        st.error(f"❌ Error initializing vector store: {str(e)}")
        logger.error(f"Error initializing vector store: {e}", exc_info=True)
        st.stop()

st.divider()

# Question Input
question = st.text_input(
    "💭 Your Question",
    placeholder="What is the efficacy of Orserdu?",
    help="Ask any question about Orserdu from the HCP website"
)

if st.button("🤔 Ask Question", type="primary"):
    if not question or not question.strip():
        st.warning("⚠️ Please enter a question")
    else:
        try:
            # Retrieve relevant documents
            with st.spinner("🔍 Searching for relevant information..."):
                results = st.session_state.vector_store.query(
                    query_texts=[question],
                    n_results=5
                )
                
                if not results or not results.get('documents') or not results['documents'][0]:
                    st.warning("⚠️ No relevant information found for your question")
                    logger.warning(f"No results for query: {question}")
                else:
                    context_list = results['documents'][0]
                    sources = results['metadatas'][0]
                    distances = results.get('distances', [[]])[0]
                    
                    context = "\n\n".join(context_list)
                    
                    logger.info(f"Retrieved {len(context_list)} context chunks")
                    
                    # Generate answer
                    with st.spinner("🤖 Generating answer..."):
                        answer = generate_answer(
                            query=question,
                            context=context,
                            api_key=OPENAI_API_KEY,
                            model="gpt-4o-mini"
                        )
                        
                        logger.info("Answer generated successfully")
                    
                    # Display Answer
                    st.markdown("---")
                    st.markdown("### 💬 Answer")
                    st.markdown(answer)
                    
                    st.markdown("---")
                    
                    # Display Sources with Structured Metadata
                    st.markdown(f"### 📚 Sources ({len(context_list)} references found)")
                    st.markdown("*Click on URLs to view the original source*")
                    
                    for idx, (doc, meta, dist) in enumerate(zip(context_list, sources, distances), 1):
                        # Calculate similarity percentage
                        similarity = (1 - dist) * 100
                        
                        # Determine color based on similarity
                        if similarity >= 90:
                            color = "🟢"
                        elif similarity >= 75:
                            color = "🟡"
                        else:
                            color = "🟠"
                        
                        # Create a container for each source
                        with st.container():
                            st.markdown(f"#### {color} Source {idx} - {similarity:.1f}% Relevant")
                            
                            # URL
                            url = meta.get('page_url', 'Unknown')
                            if url != 'Unknown':
                                st.markdown(f"🔗 **URL:** [{url}]({url})")
                            else:
                                st.markdown(f"🔗 **URL:** Unknown")
                            
                            # Section
                            section = meta.get('section_title', 'Unknown')
                            st.markdown(f"📑 **Section:** `{section}`")
                            
                            # Excerpt
                            st.markdown(f"📝 **Excerpt:**")
                            excerpt = doc[:300].strip() + "..." if len(doc) > 300 else doc.strip()
                            st.text_area(
                                f"source_{idx}",
                                value=excerpt,
                                height=100,
                                label_visibility="collapsed",
                                disabled=True
                            )
                            
                            # Divider between sources
                            if idx < len(context_list):
                                st.markdown("---")
                    
        except VectorStoreError as e:
            st.error(f"❌ Vector Store Error: {e.message}")
            if e.details:
                with st.expander("Error Details"):
                    st.json(e.details)
            logger.error(f"Vector store error: {e}", exc_info=True)
            
        except RAGError as e:
            st.error(f"❌ Answer Generation Error: {e.message}")
            st.info("💡 Check your API key and quota.")
            if e.details:
                with st.expander("Error Details"):
                    st.json(e.details)
            logger.error(f"RAG error: {e}", exc_info=True)
            
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Unexpected error: {e}", exc_info=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>RAG Q&A Application | Built with Streamlit, ChromaDB, and OpenAI</p>
    <p style='font-size: 0.8em;'>💡 Tip: Use specific questions for better results</p>
</div>
""", unsafe_allow_html=True)

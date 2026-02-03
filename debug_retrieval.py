
import sys
import os
import logging
sys.path.insert(0, os.path.join(os.getcwd(), 'modules'))

from modules.vector_store import ChromaVectorStore
from dotenv import load_dotenv

load_dotenv()

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)

def debug_retrieval():
    print("--- Debugging Retrieval ---")
    
    # Use the path the user is using
    PERSISTENT_PATH = "../data/chroma_db"
    COLLECTION_NAME = "orserduhcp"
    
    print(f"Connecting to: {PERSISTENT_PATH}")
    
    try:
        vs = ChromaVectorStore(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSISTENT_PATH
        )
        
        count = vs.count()
        print(f"Total Documents: {count}")
        
        if count == 0:
            print("WARNING: Collection is empty!")
            return

        # 1. Peek at random docs to see quality
        print("\n--- Content Sample (First 3 chunks) ---")
        peek = vs.peek(limit=3)
        for i, doc in enumerate(peek['documents']):
            print(f"\n[Chunk {i}]: {doc[:200]}...")
            print(f"Metadata: {peek['metadatas'][i]}")

        # 2. Test the problematic query
        query = "what is the details about dosing of orserdu"
        print(f"\n--- Query Test: '{query}' ---")
        
        results = vs.query(query_texts=[query], n_results=5)
        
        if not results['documents']:
            print("No results found.")
            return

        for i, (doc, dist, meta) in enumerate(zip(results['documents'][0], results['distances'][0], results['metadatas'][0])):
            similarity = (1 - dist) * 100
            print(f"\nResult {i+1} (Sim: {similarity:.1f}%):")
            print(f"Source: {meta.get('page_url')} - {meta.get('section_title')}")
            print(f"Content: {doc}")
            
    except Exception as e:
        print(f"ERROR: {e}")

def full_test_cycle():
    print("--- Starting Full Test Cycle (Reset, Scrape, Query) ---")
    
    # 1. Reset & Scrape
    from modules.scraper import Scraper
    from modules.vector_store import ChromaVectorStore
    
    # Hardcoded config matches app.py
    WEBSITE_URL = "https://www.orserduhcp.com"
    PERSISTENT_PATH = "../data/chroma_db"
    COLLECTION_NAME = "orserduhcp"
    
    print(f"1. Resetting Collection at {PERSISTENT_PATH}...")
    vs = ChromaVectorStore(collection_name=COLLECTION_NAME, persist_directory=PERSISTENT_PATH)
    vs.reset()
    
    print("2. Scraping Website (this might take a moment)...")
    # Increased chunk size to capture more context per chunk, might help with similarity
    scraper = Scraper(baseUrl=WEBSITE_URL, chunk_size=800, chunk_overlap=50) 
    data = scraper.start_scraping()
    
    print(f"3. Indexing {len(data['documents'])} documents...")
    vs.add_documents(documents=data['documents'], metadatas=data['metadata'])
    
    print("4. Running Query Test...")
    debug_retrieval()

if __name__ == "__main__":
    # Redirect stdout to a file to capture all output reliably
    with open("debug_output_v2.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        try:
            full_test_cycle()
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        sys.stdout = sys.__stdout__  # Restore stdout
    
    print("Debug run cycle complete. Check debug_output_v2.txt")

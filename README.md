# Orserdu HCP RAG Application

A specialized Retrieval-Augmented Generation (RAG) application designed to accurately answer user queries based on the [Orserdu HCP website](https://www.orserduhcp.com). This project uses a custom scraping and indexing pipeline to ensure high-quality, relevant context for the LLM.

## Key Features

-   **High-Fidelity Scraping**: Uses a custom scraper (`modules/scraper.py`) that respects paragraph structure (preserving newlines) and removes HTML noise.
-   **Context Enrichment**: Automatically extracts section headers and prepends them to every text chunk (e.g., `Context: Dosing. <content>`). This significantly improves retrieval accuracy for specific queries like "dosing" vs "ordering".
-   **Noise Reduction**: Filters out sparse lines, navigation links, and miscellaneous UI text that confuses standard RAG pipelines.
-   **Intelligent De-duplication**: 
    -   Identifies and indexes the "Important Safety Information" (ISI) block **only once**, even though it appears on every page.
    -   Prevents duplicate indexing of identical content (like footers) across the site.
-   **Clean Indexing**: Automatically resets the collection before re-indexing to ensure no stale data remains.
-   **Modern UI**: Built with Streamlit, featuring a simple "Scrape & Index" workflow and a clean chat interface.
-   **Persistent Storage**: Uses ChromaDB for persistent vector storage.

## Tech Stack

-   **Frontend**: Streamlit
-   **Vector Database**: ChromaDB
-   **LLM & Embeddings**: OpenAI (`gpt-4o-mini` for answers, `text-embedding-3-small` for embeddings)
-   **Scraping**: `requests`, `beautifulsoup4`, `lxml`

## Installation

1.  **Clone the repository**
    ```bash
    git clone <repository_url>
    cd RAG_application
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**
    -   Copy the example file:
        ```bash
        cp .env.example .env
        ```
    -   Open `.env` and add your OpenAI API Key:
        ```
        OPENAI_API_KEY=sk-your-key-here
        ```

## Usage

1.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

2.  **Initialize Data** (First Run Only)
    -   Open the app in your browser (usually `http://localhost:8501`).
    -   Open the **Sidebar**.
    -   Click **"Scrape & Index Website"**.
    -   Wait for the process to complete (the progress bar will show the scraping status).
    -   *Note: This might take a minute as it responsibly scrapes and processes the entire site.*

3.  **Ask Questions**
    -   Examples:
        -   "What is the dosing schedule for Orserdu?"
        -   "How should I handle dose modifications?"
        -   "What are the serious adverse reactions?"

## Project Structure

```text
RAG_application/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (API keys)
├── modules/
│   ├── scraper.py          # Custom scraper with de-duplication & enrichment logic
│   ├── vector_store.py     # ChromaDB wrapper for persistence & query
│   └── rag.py              # LLM generation logic
└── data/
    └── chroma_db/          # Persistent vector database storage
```

## Advanced Logic

### Scraper (`modules/scraper.py`)
The scraper is the heart of this RAG. Unlike generic scrapers that dump all text, this module:
1.  **Iterates the Sitemap**: Finds all valid pages.
2.  **Cleans Text**: Removes `|`, `•`, and single-character noise.
3.  **Hashes Content**: Checks every section against a hash set. If a section (like the Footer or Safety Info) has been seen before, it is skipped.
4.  **Enriches chunks**: Prepends the `section_title` to the text content to ensure the semantic search engine understands the *context* of the text, not just the words.

### Vector Store (`modules/vector_store.py`)
-   Wraps ChromaDB for easy initialization.
-   Handles `reset()` logic to prevent duplicate documents on re-runs.

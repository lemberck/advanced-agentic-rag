"""
Advanced RAG System Main Script
==============================

This script serves as the main entry point for the Advanced RAG system, handling both:
1. Document Ingestion: Checks and updates the vector store with new/modified documents
2. Question Answering: Processes user questions through the RAG pipeline
"""

import logging
from pathlib import Path
import sys
from dotenv import load_dotenv

from ingestion_dev import AWSDocsIngester
from graph.graph_dev import app
from graph.utils.output_formatter import print_rag_query, print_rag_final_result

# Configure logging for file only, not console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_environment() -> None:
    """Initialize environment."""
    try:
        load_dotenv()
        logger.info("Environment variables loaded")
    except Exception as e:
        logger.error(f"Failed to load environment: {str(e)}")
        raise

def check_and_ingest_documents() -> bool:
    """Check for document changes and run ingestion if needed."""
    try:
        ingester = AWSDocsIngester()
        changes_detected = False
        
        # Check for changes
        existing_docs = ingester._get_existing_docs()
        md_files = list(ingester.docs_dir.glob('**/*.md'))
        
        for file_path in md_files:
            source = str(file_path.resolve())
            if source not in existing_docs:
                changes_detected = True
                break
            
            current_meta = ingester._get_file_metadata(file_path)
            existing_meta = existing_docs[source]
            if (current_meta["last_modified"] != existing_meta.get("last_modified") and 
                current_meta["file_hash"] != existing_meta.get("file_hash")):
                changes_detected = True
                break
        
        if changes_detected:
            user_input = input("\nDocument changes detected! Would you like to run the ingestion pipeline? (y/n): ")
            if user_input.lower() == 'y':
                logger.info("Starting document ingestion")
                ingester.upsert_documents()
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise

def process_user_question(question: str) -> None:
    """Process a user question through the RAG pipeline."""
    try:
        inputs = {"question": question}
        print_rag_query(inputs["question"])
        
        for output in app.stream(inputs, config={"configurable": {"thread_id": "2"}}):
            if 'generate' in output:
                answer = output['generate']['generation']
                sources = [
                    Path(doc.metadata.get('source')).name 
                    if doc.metadata.get('source').startswith('/') 
                    else doc.metadata.get('source')
                    for doc in output['generate'].get('documents', [])
                ]
                print_rag_final_result(inputs["question"], answer, sources)
                logger.info(f"Generated answer with {len(sources)} sources")
                
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise

def main():
    """Main execution function."""
    try:
        setup_environment()
        check_and_ingest_documents()
        
        while True:
            print("\nEnter your question (or 'quit' to exit):")
            question = input("> ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nShutting down RAG system...")
                break
                
            if question.strip():
                process_user_question(question)
            else:
                print("Please enter a valid question")
                
    except KeyboardInterrupt:
        print("\nShutting down RAG system...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        sys.exit(1)
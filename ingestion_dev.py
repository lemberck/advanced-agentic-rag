"""
AWS Documentation Ingestion System for ChromaDB-based Vector Storage

A specialized document processing pipeline that handles AWS documentation ingestion
with versioning, chunking, and metadata enrichment for ChromaDB vector storage.

Technical Specifications:
------------------------
- Vector Store: ChromaDB
  - Collection Name: "public_data"
  - Embedding Model: OpenAI Embeddings
  - Persistence: Local directory (./.chroma)
  - Required Metadata Fields: source, file_hash, last_modified

Document Processing:
------------------
1. Text Splitting
   - Engine: RecursiveCharacterTextSplitter with tiktoken
   - Chunk Size: 500 tokens
   - Overlap: 50 tokens (10% 0verlap)
   - Splitting Hierarchy: paragraphs -> lines -> sentences -> clauses -> words -> chars

2. Metadata Extraction
   - Source Path: Absolute file path
   - Last Modified: ISO formatted timestamp
   - File Hash: MD5 of page content
   - AWS-specific: service, doc_category, resource_type, topic

Versioning System:
----------------
Two-phase versioning detection:
1. Fast Check: File system timestamp (st_mtime)
2. Verification: MD5 content hash
Only processes documents when both timestamp AND hash change

Upsert Implementation:
--------------------
1. Document State Detection:
   - NEW: Document not in vector store
   - MODIFIED: Both timestamp and hash changed
   - SKIPPED: Only timestamp or only hash changed

2. Update Process:
   a) Delete existing chunks for the identified modified documents (avoid stale chunks)
   b) Load and process new/modified documents
   c) Split into chunks with context preservation
   d) Bulk add new chunks to vector store

3. Atomic Operations:
   - Chunk deletion by document source
   - Bulk insertion of new chunks
   - Metadata preservation across updates

Performance Considerations:
------------------------
- Minimizes unnecessary processing via two-step version check
- Batch processes updates to reduce DB operations
- Preserves document relationships through metadata
- Handles partial updates without full collection rebuild

Usage Requirements:
-----------------
- Environment: OpenAI API key for embeddings
- File Structure: Markdown documentation files - other types will be ignored
- Storage: Write access to local filesystem
- Memory: Sufficient for document batch processing
"""

from pathlib import Path
import hashlib
from typing import Dict, List
from datetime import datetime

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings

from graph.utils.ingestion_formatter import (
    print_ingestion_start,
    print_document_status,
    print_ingestion_summary
)

# Load env vars
load_dotenv()

class AWSDocsIngester:
    def __init__(self, docs_dir: str = "./public_data", persist_dir: str = "./.chroma"):
        # get the path where the ingestion script is located
        script_dir = Path(__file__).parent
        
        # convert relative paths to absolute paths
        self.docs_dir = (script_dir / docs_dir).resolve()
        self.persist_dir = (script_dir / persist_dir).resolve()
        
        print(f"Script directory: {script_dir}")
        print(f"Documents directory: {self.docs_dir}")
        
        self._ensure_directories()
        self.embedding_function = OpenAIEmbeddings()
        
        # Configure text splitter for AWS documentation
        # - Uses tiktoken to count tokens accurately (OpenAI cl100k_base encoding by default)
        # - Target size: 500 tokens per chunk with 50 token overlap
        # - Splits on natural boundaries in order:
        #   1. Paragraphs (\n\n)
        #   2. Lines (\n)
        #   3. Sentences (. )
        #   4. Clauses (, )
        #   5. Words ( )
        #   6. Characters (if needed)
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,  # Target chunk size in tokens
            chunk_overlap=50,  # Overlap between chunks - 10% factor
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            is_separator_regex=False
        )
        
        # init or load existing vector store
        # Uses cosine similarity by default
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding_function,
            collection_name="public_data"  # aws public docs
        )
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for version tracking."""
        return hashlib.md5(file_path.read_bytes()).hexdigest()
    
    def _extract_aws_metadata(self, file_path: Path) -> dict:
        """Extract metadata from AWS documentation filename patterns."""
        filename = file_path.name
        metadata = {
            "filename": filename,
            "doc_category": "", #chroma does not support None type
            "service": "",
            "resource_type": "",
            "topic": ""
        }
        
        # Remove .md extension and split
        name_parts = filename.replace('.md', '').split('-')
        
        # Pattern 1: sagemaker-projects-*
        if filename.startswith('sagemaker-projects-'):
            metadata.update({
                "service": "sagemaker",
                "doc_category": "projects",
                "topic": '-'.join(name_parts[2:])  # Everything after 'projects-'
            })
        
        # Pattern 2: aws-properties-sagemaker-*
        elif filename.startswith('aws-properties-sagemaker-'):
            metadata.update({
                "service": "sagemaker",
                "doc_category": "properties",
                "resource_type": '-'.join(name_parts[3:])  # The actual resource being documented
            })
        
        # Pattern 3: Other sagemaker-related docs
        elif 'sagemaker' in filename:
            metadata.update({
                "service": "sagemaker",
                "doc_category": "guides",
                "topic": filename.replace('sagemaker-', '').replace('.md', '')
            })
        
        return metadata
    
    def _get_file_metadata(self, file_path: Path) -> dict:
        """Get file metadata without loading content."""
        return {
            "source": str(file_path.resolve()),
            "file_hash": self._get_file_hash(file_path),
            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            **self._extract_aws_metadata(file_path)
        }
    
    def _get_existing_docs(self) -> Dict[str, dict]:
        """Get existing document metadata from vector store."""
        try:
            # Get all documents from the collection
            results = self.vectorstore.get(
                include=["metadatas", "documents"]
            )
            
            # Debug information
            print(f"Found {len(results['metadatas'])} existing documents in collection")
            
            return {
                meta.get("source"): {
                    "file_hash": meta.get("file_hash"),
                    "last_modified": meta.get("last_modified"),
                    "content": doc  # Store content for comparison
                }
                for meta, doc in zip(results["metadatas"], results["documents"])
                if meta and "source" in meta
            }
        except Exception as e:
            print(f"Error accessing collection: {e}")
            return {}
    
    def upsert_documents(self):
        """Update vector store with new or modified documents."""
        # Get all markdown files
        md_files = list(self.docs_dir.glob('**/*.md'))
        print_ingestion_start(str(self.docs_dir), len(md_files))
        
        # Track statistics
        new_docs = 0
        modified_docs = 0
        deleted_chunks = 0
        
        # Get existing docs metadata from vectorstore
        existing_docs = self._get_existing_docs()
        
        # Check which files need updating based on metadata only
        files_to_update = []
        for file_path in md_files:
            source = str(file_path.resolve())
            
            # Case 1: New document
            if source not in existing_docs:
                print_document_status("NEW", source)
                files_to_update.append(file_path)
                new_docs += 1
                continue
            
            # Case 2: Check timestamps and hashes
            current_meta = self._get_file_metadata(file_path)
            existing_meta = existing_docs[source]
            
            timestamp_changed = current_meta["last_modified"] != existing_meta.get("last_modified")
            hash_changed = current_meta["file_hash"] != existing_meta.get("file_hash")
            
            if timestamp_changed and hash_changed:
                print_document_status("MODIFIED", source)
                # Get all documents with this source and delete them
                results = self.vectorstore._collection.get(
                    where={"source": source}
                )
                if results['ids']:
                    chunks_count = len(results['ids'])
                    self.vectorstore._collection.delete(
                        ids=results['ids']
                    )
                    deleted_chunks += chunks_count
                    print_document_status("DELETED", source, f"Removed {chunks_count} existing chunks")
                
                files_to_update.append(file_path)
                modified_docs += 1
            else:
                if timestamp_changed or hash_changed:
                    print_document_status("SKIPPED", source, 
                        "Only timestamp changed" if timestamp_changed else "Only content hash changed")
        
        if not files_to_update:
            print_document_status("SKIPPED", "No documents need updating")
            return
        
        # Process updates
        documents = []
        for file_path in files_to_update:
            loader = UnstructuredMarkdownLoader(str(file_path))
            doc = loader.load()[0]
            doc.metadata.update(self._get_file_metadata(file_path))
            documents.append(doc)
        
        # Split and embed
        splits = self.text_splitter.split_documents(documents)
        self.vectorstore.add_documents(splits)
        
        # Get final chunk count
        final_results = self.vectorstore.get(include=["metadatas"])
        total_chunks = len(final_results["metadatas"])
        
        # Print summary
        print_ingestion_summary(
            total_files=len(md_files),
            new_docs=new_docs,
            modified_docs=modified_docs,
            deleted_chunks=deleted_chunks,
            total_chunks=total_chunks
        )

# Use simple similarity search for the retriever (mmr was too restrictive)
retriever = AWSDocsIngester().vectorstore.as_retriever(
    search_type="similarity",  
    search_kwargs={
        "k": 7  # Number of documents to retrieve
    }
)

# Usage
if __name__ == "__main__":
    ingester = AWSDocsIngester()
    ingester.upsert_documents()

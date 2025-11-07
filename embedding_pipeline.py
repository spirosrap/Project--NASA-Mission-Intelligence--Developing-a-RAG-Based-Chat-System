#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 batch_size: int = 50):
        """
        Initialize the embedding pipeline
        
        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Number of chunks to insert per batch
        """
        if not openai_api_key:
            raise ValueError("An OpenAI API key is required to initialize the pipeline.")

        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.batch_size = max(1, batch_size)

        # Ensure chunk parameters stay within sane bounds
        self.chunk_size = max(1, chunk_size)
        if chunk_overlap >= self.chunk_size:
            logger.warning(
                "chunk_overlap (%s) must be smaller than chunk_size (%s). "
                "Reducing overlap to chunk_size - 1.",
                chunk_overlap,
                self.chunk_size,
            )
            chunk_overlap = self.chunk_size - 1
        self.chunk_overlap = max(0, chunk_overlap)

        # Persist directory for Chroma collections
        self.chroma_persist_directory = chroma_persist_directory
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI client
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Initialize ChromaDB client/collection
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=self.embedding_model
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Base metadata for the text
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        text_length = len(cleaned_text)
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap

        # Short texts fit in a single chunk
        if text_length <= chunk_size:
            chunk_metadata = dict(metadata)
            chunk_metadata.update({
                'chunk_index': 0,
                'chunk_char_start': 0,
                'chunk_char_end': text_length,
                'chunk_size': text_length,
                'chunk_overlap': 0
            })
            return [(cleaned_text, chunk_metadata)]

        chunks: List[Tuple[str, Dict[str, Any]]] = []
        start = 0
        chunk_index = 0

        while start < text_length:
            hard_end = min(start + chunk_size, text_length)
            end = self._find_preferred_breakpoint(cleaned_text, start, hard_end)
            chunk_text = cleaned_text[start:end]

            chunk_metadata = dict(metadata)
            chunk_metadata.update({
                'chunk_index': chunk_index,
                'chunk_char_start': start,
                'chunk_char_end': end,
                'chunk_size': len(chunk_text),
                'chunk_overlap': chunk_overlap if chunk_index > 0 else 0
            })
            chunks.append((chunk_text, chunk_metadata))

            if end == text_length:
                break

            start = max(end - chunk_overlap, 0)
            chunk_index += 1

        return chunks

    def _find_preferred_breakpoint(self, text: str, start: int, hard_end: int) -> int:
        """
        Try to break chunks on natural boundaries (sentence/paragraph) without
        exceeding the configured chunk size.
        """
        window = text[start:hard_end]
        if hard_end == len(text):
            return hard_end

        # Prefer paragraph and sentence boundaries when they fall in the second
        # half of the window to avoid overly small trailing chunks.
        breakpoints = []
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; "]
        for sep in separators:
            pos = window.rfind(sep)
            if pos != -1:
                breakpoint = start + pos + len(sep)
                # Only consider breakpoints that keep at least half the chunk
                if breakpoint - start >= self.chunk_size // 2:
                    breakpoints.append(breakpoint)

        if breakpoints:
            return max(breakpoints)

        # Fall back to first whitespace before the hard end
        whitespace_match = list(re.finditer(r"\s+", window))
        if whitespace_match:
            for match in reversed(whitespace_match):
                breakpoint = start + match.start()
                if breakpoint > start:
                    return breakpoint

        return hard_end
    
    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id], include=['metadatas'])
            return bool(result and result.get('ids'))
        except Exception as e:
            logger.error(f"Error checking document existence for {doc_id}: {e}")
            return False
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection
        
        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text)
            
            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)
        
        Args:
            source_pattern: Pattern to match source names
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0
    
    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('source') == source and 
                    metadata.get('mission') == mission):
                    file_doc_ids.append(all_docs['ids'][i])
            
            return file_doc_ids
            
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs
        """
        mission = self._slugify(metadata.get('mission', 'unknown'))
        source = self._slugify(metadata.get('source', file_path.stem))
        chunk_index = metadata.get('chunk_index', 0)
        identifier = f"{mission}__{source}__chunk_{chunk_index:04d}"
        # Include hashed file path to avoid collisions across similarly named files
        file_hash = hashlib.md5(str(file_path).encode('utf-8')).hexdigest()[:8]
        return f"{identifier}__{file_hash}"

    @staticmethod
    def _slugify(value: str) -> str:
        """Convert strings to filesystem-friendly slugs"""
        value = value or "unknown"
        value = re.sub(r'[^a-zA-Z0-9]+', '_', value).strip('_')
        return value.lower() or "unknown"
    
    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Enhanced metadata extraction
            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return self.chunk_text(content, metadata)
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []
    
    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'
    
    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'
    
    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()
        
        # Apollo transcript types
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'
        
        # Challenger audio segments
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'
        
        # NASA archive documents
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'
        
        # General categories
        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'
    
    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)
        
        Args:
            base_path: Base directory path
            
        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []
        
        # Define directories to scan
        data_dirs = [
            'apollo11',
            'apollo13',
            'challenger'
        ]
        
        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                
                # Find only text files
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")
        
        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (file_path.name.startswith('.') or 
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)
        
        logger.info(f"Total text files to process: {len(filtered_files)}")
        
        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1
        
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")
        
        return filtered_files
    
    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]], 
                                   file_path: Path, batch_size: int = 50, 
                                   update_mode: str = 'skip') -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling
        
        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            
        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}
        
        stats = {'added': 0, 'updated': 0, 'skipped': 0}
        batch_size = max(1, batch_size)
        file_path = Path(file_path)

        try:
            existing_ids: Optional[set] = None
            if update_mode in {'skip', 'update'}:
                existing_ids = set(self.get_file_documents(file_path))
            elif update_mode == 'replace':
                existing = self.get_file_documents(file_path)
                if existing:
                    self.collection.delete(ids=existing)
                existing_ids = set()

            batch_ids: List[str] = []
            batch_docs: List[str] = []
            batch_metas: List[Dict[str, Any]] = []
            batch_embeddings: List[List[float]] = []

            def flush_batch():
                if not batch_ids:
                    return
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=batch_embeddings
                )
                stats['added'] += len(batch_ids)
                batch_ids.clear()
                batch_docs.clear()
                batch_metas.clear()
                batch_embeddings.clear()

            for chunk_text, chunk_metadata in documents:
                doc_id = self.generate_document_id(file_path, chunk_metadata)
                chunk_metadata = dict(chunk_metadata)
                chunk_metadata['document_id'] = doc_id
                chunk_metadata['file_path'] = chunk_metadata.get('file_path', str(file_path))

                exists = False
                if existing_ids is not None:
                    exists = doc_id in existing_ids
                else:
                    exists = self.check_document_exists(doc_id)

                if exists:
                    if update_mode == 'skip':
                        stats['skipped'] += 1
                        continue
                    elif update_mode == 'update':
                        updated = self.update_document(doc_id, chunk_text, chunk_metadata)
                        if updated:
                            stats['updated'] += 1
                        else:
                            stats['skipped'] += 1
                        continue

                embedding = self.get_embedding(chunk_text)
                batch_ids.append(doc_id)
                batch_docs.append(chunk_text)
                batch_metas.append(chunk_metadata)
                batch_embeddings.append(embedding)

                if existing_ids is not None:
                    existing_ids.add(doc_id)

                if len(batch_ids) >= batch_size:
                    flush_batch()

            flush_batch()
            return stats

        except Exception as e:
            logger.error(f"Error adding documents for {file_path}: {e}")
            return stats
    
    def process_all_text_data(self, base_path: str, update_mode: str = 'skip',
                              batch_size: Optional[int] = None) -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB
        
        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents (default)
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            
        Returns:
            Statistics about processed files
        """
        stats: Dict[str, Any] = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }
        
        files = self.scan_text_files_only(base_path)
        effective_batch_size = batch_size or self.batch_size

        for file_path in files:
            stats['files_processed'] += 1
            try:
                documents = self.process_text_file(file_path)
                stats['total_chunks'] += len(documents)

                if not documents:
                    continue

                mission = documents[0][1].get('mission', 'unknown')
                mission_stats = stats['missions'].setdefault(
                    mission, {'files': 0, 'chunks': 0, 'added': 0, 'updated': 0, 'skipped': 0}
                )
                mission_stats['files'] += 1
                mission_stats['chunks'] += len(documents)

                add_stats = self.add_documents_to_collection(
                    documents,
                    Path(file_path),
                    batch_size=effective_batch_size,
                    update_mode=update_mode
                )

                stats['documents_added'] += add_stats['added']
                stats['documents_updated'] += add_stats['updated']
                stats['documents_skipped'] += add_stats['skipped']

                mission_stats['added'] += add_stats['added']
                mission_stats['updated'] += add_stats['updated']
                mission_stats['skipped'] += add_stats['skipped']

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        return stats
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        try:
            return {
                'collection_name': getattr(self.collection, 'name', 'unknown'),
                'document_count': self.collection.count(),
                'persist_directory': self.chroma_persist_directory
            }
        except Exception as e:
            logger.error(f"Error retrieving collection info: {e}")
            return {'error': str(e)}
    
    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for testing
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return {'error': str(e)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            base_stats = {
                'collection_name': getattr(self.collection, 'name', 'unknown'),
                'persist_directory': self.chroma_persist_directory,
                'document_count': self.collection.count(),
                'total_documents': 0,
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }
            
            # Get all documents to analyze
            all_docs = self.collection.get()
            
            metadatas = all_docs.get('metadatas', [])
            base_stats['total_documents'] = len(metadatas)
            
            if not metadatas:
                return base_stats
            
            # Analyze metadata
            for metadata in metadatas:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')
                
                # Count by mission
                base_stats['missions'][mission] = base_stats['missions'].get(mission, 0) + 1
                
                # Count by data type
                base_stats['data_types'][data_type] = base_stats['data_types'].get(data_type, 0) + 1
                
                # Count by document category
                base_stats['document_categories'][doc_category] = base_stats['document_categories'].get(doc_category, 0) + 1
                
                # Count by file type
                base_stats['file_types'][file_type] = base_stats['file_types'].get(file_type, 0) + 1
            
            return base_stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='.', help='Path to data directories')
    parser.add_argument('--openai-key', required=True, help='OpenAI API key')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                       help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size
    )
    
    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return
    
    # If stats only, show collection statistics and exit
    if args.stats_only:
        stats = pipeline.get_collection_stats()
        if 'error' in stats:
            logger.error(f"Failed to load collection statistics: {stats['error']}")
            return
        
        logger.info("Collection Statistics")
        logger.info("-" * 40)
        logger.info(f"Collection: {stats.get('collection_name', 'unknown')}")
        logger.info(f"Persist Directory: {stats.get('persist_directory', 'unknown')}")
        logger.info(f"Document Count: {stats.get('document_count', 0)}")
        logger.info(f"Total Chunks (documents stored): {stats.get('total_documents', 0)}")
        
        def log_breakdown(title: str, data: Dict[str, int]):
            if not data:
                logger.info(f"{title}: None")
                return
            logger.info(f"{title}:")
            for key, value in sorted(data.items()):
                logger.info(f"  {key}: {value}")
        
        log_breakdown("Missions", stats.get('missions', {}))
        log_breakdown("Data Types", stats.get('data_types', {}))
        log_breakdown("Document Categories", stats.get('document_categories', {}))
        log_breakdown("File Types", stats.get('file_types', {}))
        return
    
    # Process all data
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()
    
    stats = pipeline.process_all_text_data(
        args.data_path,
        update_mode=args.update_mode,
        batch_size=args.batch_size
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    
    # Mission breakdown
    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")
    
    # Collection info
    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")
    
    # Test query if provided
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):  # Show top 3
                logger.info(f"Result {i+1}: {doc[:200]}...")
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()

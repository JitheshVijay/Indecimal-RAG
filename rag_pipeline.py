"""
================================================================================
RAG PIPELINE FOR INDECIMAL CONSTRUCTION MARKETPLACE AI ASSISTANT
================================================================================

This module implements a Retrieval-Augmented Generation (RAG) pipeline that:
1. Loads and chunks markdown documents by headers
2. Generates embeddings using sentence-transformers (all-MiniLM-L6-v2)
3. Performs semantic search using FAISS vector store
4. Generates grounded answers using Ollama (local LLM)

The pipeline ensures all answers are strictly grounded in the retrieved context,
with explicit source citations and refusal to hallucinate information.

Dependencies:
    - sentence-transformers: For generating text embeddings
    - faiss-cpu: For efficient vector similarity search
    - ollama: For local LLM inference
"""

# ================================================================================
# IMPORTS
# ================================================================================

import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# Embedding model - sentence-transformers provides pre-trained models
from sentence_transformers import SentenceTransformer

# Vector store - FAISS is Facebook's library for efficient similarity search
import faiss

# LLM - Ollama provides local LLM inference
import ollama


# ================================================================================
# DATA CLASSES
# ================================================================================

@dataclass
class DocumentChunk:
    """
    Represents a chunk of a document with metadata.
    
    Attributes:
        chunk_id: Unique identifier for the chunk (e.g., "doc1.md_0")
        text: The actual text content of the chunk
        source: Source filename (e.g., "doc1.md")
        section: Section header this chunk belongs to
        start_line: Starting line number in the original document
        end_line: Ending line number in the original document
    """
    chunk_id: str
    text: str
    source: str
    section: str
    start_line: int
    end_line: int


# ================================================================================
# RAG PIPELINE CLASS
# ================================================================================

class RAGPipeline:
    """
    A complete RAG pipeline for the Indecimal construction marketplace assistant.
    
    This class implements the core RAG functionality:
    1. Document Processing: Load and chunk markdown documents by headers
    2. Embedding: Convert text chunks to dense vectors
    3. Indexing: Build FAISS index for fast similarity search
    4. Retrieval: Find relevant chunks for user queries
    5. Generation: Generate grounded answers using Ollama
    
    Attributes:
        EMBEDDING_MODEL: Name of the sentence-transformers model to use
        DEFAULT_OLLAMA_MODEL: Default Ollama model for generation
        chunks: List of all document chunks
        embeddings: NumPy array of chunk embeddings
        index: FAISS index for similarity search
    """
    
    # ======================== CLASS CONSTANTS ========================
    
    # Embedding model: all-MiniLM-L6-v2 is lightweight (80MB), fast, and produces 
    # 384-dimensional embeddings. Good balance of quality and speed for this use case.
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Default LLM model - llama3.2:1b is small enough to run on most hardware
    DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
    
    # ======================== INITIALIZATION ========================
    
    def __init__(self, documents_dir: str = None, llm_model: str = None):
        """
        Initialize the RAG pipeline.
        
        This constructor:
        1. Sets up the documents directory path
        2. Configures the LLM model
        3. Loads the embedding model
        4. Processes documents and builds the FAISS index
        
        Args:
            documents_dir: Directory containing markdown documents (default: same as this script)
            llm_model: Ollama model name to use for generation (default: llama3.2:1b)
        """
        # Set documents directory (default to script's directory)
        self.documents_dir = documents_dir or str(Path(__file__).parent)
        
        # Set LLM model
        self.llm_model = llm_model or self.DEFAULT_OLLAMA_MODEL
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
        
        # Initialize storage for chunks and index
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        
        # Load and index documents
        self._load_and_index_documents()
    
    # ======================== DOCUMENT PROCESSING ========================
    
    def _load_and_index_documents(self):
        """
        Load all markdown documents and build the FAISS index.
        
        This method:
        1. Iterates through the expected document files
        2. Chunks each document by headers
        3. Generates embeddings for all chunks
        4. Builds a FAISS index for similarity search
        
        Raises:
            ValueError: If no documents are found to index
        """
        # List of expected document files
        doc_files = ["doc1.md", "doc2.md", "doc3.md"]
        
        # Process each document
        for doc_file in doc_files:
            doc_path = os.path.join(self.documents_dir, doc_file)
            if os.path.exists(doc_path):
                print(f"Processing: {doc_file}")
                chunks = self._chunk_document(doc_path)
                self.chunks.extend(chunks)
        
        # Validate that we have documents
        if not self.chunks:
            raise ValueError("No documents found to index!")
        
        print(f"Total chunks created: {len(self.chunks)}")
        
        # Generate embeddings for all chunks
        print("Generating embeddings...")
        texts = [chunk.text for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Build FAISS index
        # IndexFlatL2 performs exact L2 (Euclidean) distance search
        # Suitable for small datasets (<10K vectors)
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors (dimension: {dimension})")
    
    def _chunk_document(self, doc_path: str) -> List[DocumentChunk]:
        """
        Chunk a markdown document by headers for semantic coherence.
        
        Strategy:
        - Split on ## and ### headers to preserve topic boundaries
        - Each chunk includes its section hierarchy for context
        - Skip very short chunks (< 50 characters)
        
        Args:
            doc_path: Path to the markdown document
            
        Returns:
            List of DocumentChunk objects
        """
        # Read document content
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        chunks = []
        source = os.path.basename(doc_path)
        
        # State for tracking current chunk
        current_section = ""
        current_text_lines = []
        current_start_line = 1
        
        # Process each line
        for i, line in enumerate(lines, 1):
            # Check for headers (## or ###)
            if line.startswith('## ') or line.startswith('### '):
                # Save previous chunk if it has content
                if current_text_lines:
                    chunk_text = '\n'.join(current_text_lines).strip()
                    # Skip very short chunks (less than 50 chars)
                    if chunk_text and len(chunk_text) > 50:
                        chunk_id = f"{source}_{len(chunks)}"
                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            source=source,
                            section=current_section,
                            start_line=current_start_line,
                            end_line=i - 1
                        ))
                
                # Start new section
                current_section = line.lstrip('#').strip()
                current_text_lines = [line]
                current_start_line = i
            else:
                # Add line to current chunk
                current_text_lines.append(line)
        
        # Don't forget the last chunk
        if current_text_lines:
            chunk_text = '\n'.join(current_text_lines).strip()
            if chunk_text and len(chunk_text) > 50:
                chunk_id = f"{source}_{len(chunks)}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    section=current_section,
                    start_line=current_start_line,
                    end_line=len(lines)
                ))
        
        return chunks
    
    # ======================== RETRIEVAL ========================
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve the top-k most relevant chunks for a query.
        
        This method:
        1. Embeds the query using the same model as documents
        2. Searches the FAISS index for nearest neighbors
        3. Converts L2 distances to similarity scores
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve (default: 3)
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by relevance
        """
        # Embed the query using same model as documents
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index for k nearest neighbors
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Build results with similarity scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                # Convert L2 distance to a similarity score (0-1 range)
                # Lower distance = higher similarity
                similarity = 1 / (1 + distance)
                results.append((self.chunks[idx], similarity))
        
        return results
    
    # ======================== GENERATION ========================
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """
        Generate a grounded answer using Ollama.
        
        The LLM is explicitly instructed to:
        1. Only use information from the retrieved context
        2. Cite sources when answering (e.g., [Source 1])
        3. Acknowledge when information is not available
        4. Never hallucinate or make up information
        
        Args:
            query: User's question
            retrieved_chunks: List of (chunk, score) tuples from retrieval
            
        Returns:
            Generated answer string
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i}: {chunk.source} - {chunk.section}]\n{chunk.text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # System prompt - explicitly instructs LLM to stay grounded
        system_prompt = """You are an AI assistant for Indecimal, a construction marketplace company.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY using the information provided in the context below.
2. If the answer cannot be found in the context, say "I don't have information about that in the provided documents."
3. Cite the source (e.g., [Source 1]) when providing information.
4. Be concise and direct in your answers.
5. Do NOT make up or hallucinate information not present in the context.
"""
        
        # User prompt with context and question
        user_prompt = f"""Context:
{context}

---

Question: {query}

Answer (cite sources):"""
        
        # Build messages for chat API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate with Ollama
        return self._generate_with_ollama(messages)
    
    def _generate_with_ollama(self, messages: List[Dict]) -> str:
        """
        Generate answer using Ollama local LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Generated text from the LLM
        """
        try:
            # Call Ollama chat API
            response = ollama.chat(
                model=self.llm_model,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            # Return error message if Ollama fails
            return f"Error generating answer: {str(e)}. Please ensure Ollama is running with the model '{self.llm_model}'."
    
    # ======================== QUERY (MAIN ENTRY POINT) ========================
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Complete RAG query: retrieve relevant chunks and generate answer.
        
        This is the main entry point for using the RAG pipeline.
        It combines retrieval and generation into a single call.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (default: 3)
            
        Returns:
            Dictionary containing:
                - question: The original question
                - retrieved_chunks: List of chunk dicts with scores
                - generated_answer: The LLM's response
        """
        # Step 1: Retrieve relevant chunks
        retrieved = self.retrieve(question, top_k)
        
        # Step 2: Generate grounded answer
        answer = self.generate_answer(question, retrieved)
        
        # Step 3: Format and return results
        return {
            "question": question,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "section": chunk.section,
                    "similarity_score": round(score, 4)
                }
                for chunk, score in retrieved
            ],
            "generated_answer": answer
        }
    
    # ======================== UTILITY METHODS ========================
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary containing:
                - total_chunks: Number of indexed chunks
                - embedding_dimension: Size of embedding vectors
                - embedding_model: Name of embedding model
                - llm_model: Name of LLM model
                - documents_indexed: List of source document names
        """
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "embedding_model": self.EMBEDDING_MODEL,
            "llm_model": self.llm_model,
            "documents_indexed": list(set(chunk.source for chunk in self.chunks))
        }


# ================================================================================
# MAIN ENTRY POINT (FOR TESTING)
# ================================================================================

if __name__ == "__main__":
    """
    Test the RAG pipeline with a sample query.
    
    This section runs when the script is executed directly (not imported).
    It initializes the pipeline, prints statistics, and runs a test query.
    """
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Print stats
    print("\n" + "="*50)
    print("RAG Pipeline Statistics:")
    print(json.dumps(rag.get_stats(), indent=2))
    
    # Test query
    test_query = "What is the price of the Premier package?"
    print(f"\n{'='*50}")
    print(f"Test Query: {test_query}")
    print("="*50)
    
    result = rag.query(test_query)
    
    # Print retrieved chunks
    print("\nRetrieved Chunks:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"\n[{i}] Source: {chunk['source']} | Section: {chunk['section']}")
        print(f"    Score: {chunk['similarity_score']}")
        print(f"    Preview: {chunk['text'][:200]}...")
    
    # Print generated answer
    print(f"\n{'='*50}")
    print("Generated Answer:")
    print(result["generated_answer"])

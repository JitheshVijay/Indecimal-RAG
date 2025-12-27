"""
RAG Pipeline for Indecimal Construction Marketplace AI Assistant

Uses Ollama (local) or OpenRouter (cloud) for LLM generation.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
import faiss

# Check Ollama availability
try:
    import ollama
    try:
        ollama.list()
        OLLAMA_AVAILABLE = True
    except:
        OLLAMA_AVAILABLE = False
except ImportError:
    OLLAMA_AVAILABLE = False

# Check OpenAI client availability
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_openrouter_key():
    """Get OpenRouter API key from environment or Streamlit secrets."""
    # Try environment variable first
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    
    # Try Streamlit secrets (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except:
        pass
    
    return ""


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    section: str
    start_line: int
    end_line: int


class RAGPipeline:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
    DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
    
    def __init__(self, documents_dir: str = None, llm_model: str = None):
        self.documents_dir = documents_dir or str(Path(__file__).parent)
        
        # Get API key
        self.openrouter_api_key = get_openrouter_key()
        
        # Determine LLM backend
        self.use_ollama = OLLAMA_AVAILABLE
        
        if self.use_ollama:
            self.llm_model = llm_model or self.DEFAULT_OLLAMA_MODEL
            self.llm_backend = "Ollama (local)"
        elif self.openrouter_api_key and OPENAI_AVAILABLE:
            self.llm_model = llm_model or self.DEFAULT_OPENROUTER_MODEL
            self.llm_backend = "OpenRouter (cloud)"
        else:
            self.llm_model = "none"
            self.llm_backend = "None available"
        
        print(f"Loading embedding model: {self.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
        
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        
        self._load_and_index_documents()
        print(f"LLM Backend: {self.llm_backend}")
        if self.openrouter_api_key:
            print("OpenRouter API key: Found")
        else:
            print("OpenRouter API key: Not found")
    
    def _load_and_index_documents(self):
        doc_files = ["doc1.md", "doc2.md", "doc3.md"]
        
        for doc_file in doc_files:
            doc_path = os.path.join(self.documents_dir, doc_file)
            if os.path.exists(doc_path):
                print(f"Processing: {doc_file}")
                chunks = self._chunk_document(doc_path)
                self.chunks.extend(chunks)
        
        if not self.chunks:
            raise ValueError("No documents found!")
        
        print(f"Total chunks: {len(self.chunks)}")
        
        print("Generating embeddings...")
        texts = [chunk.text for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        print(f"Index built with {self.index.ntotal} vectors")
    
    def _chunk_document(self, doc_path: str) -> List[DocumentChunk]:
        with open(doc_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        
        chunks = []
        source = os.path.basename(doc_path)
        current_section = ""
        current_text_lines = []
        current_start_line = 1
        
        for i, line in enumerate(lines, 1):
            if line.startswith('## ') or line.startswith('### '):
                if current_text_lines:
                    chunk_text = '\n'.join(current_text_lines).strip()
                    if chunk_text and len(chunk_text) > 50:
                        chunks.append(DocumentChunk(
                            chunk_id=f"{source}_{len(chunks)}",
                            text=chunk_text,
                            source=source,
                            section=current_section,
                            start_line=current_start_line,
                            end_line=i - 1
                        ))
                current_section = line.lstrip('#').strip()
                current_text_lines = [line]
                current_start_line = i
            else:
                current_text_lines.append(line)
        
        if current_text_lines:
            chunk_text = '\n'.join(current_text_lines).strip()
            if chunk_text and len(chunk_text) > 50:
                chunks.append(DocumentChunk(
                    chunk_id=f"{source}_{len(chunks)}",
                    text=chunk_text,
                    source=source,
                    section=current_section,
                    start_line=current_start_line,
                    end_line=len(lines)
                ))
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                similarity = 1 / (1 + distance)
                results.append((self.chunks[idx], similarity))
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i}: {chunk.source} - {chunk.section}]\n{chunk.text}")
        context = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """You are an AI assistant for Indecimal, a construction marketplace.

RULES:
1. Answer ONLY using the provided context.
2. If info not in context, say "I don't have that information."
3. Cite sources like [Source 1].
4. Be concise.
5. Never make up information."""
        
        user_prompt = f"""Context:
{context}

---

Question: {query}

Answer (cite sources):"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.use_ollama:
            return self._generate_with_ollama(messages)
        elif self.openrouter_api_key and OPENAI_AVAILABLE:
            return self._generate_with_openrouter(messages)
        else:
            return "Error: No LLM backend available. Please install Ollama locally or set OPENROUTER_API_KEY environment variable."
    
    def _generate_with_ollama(self, messages: List[Dict]) -> str:
        try:
            response = ollama.chat(model=self.llm_model, messages=messages)
            return response['message']['content']
        except Exception as e:
            return f"Error with Ollama: {str(e)}"
    
    def _generate_with_openrouter(self, messages: List[Dict]) -> str:
        try:
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with OpenRouter: {str(e)}"
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        retrieved = self.retrieve(question, top_k)
        answer = self.generate_answer(question, retrieved)
        
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
    
    def get_stats(self) -> Dict:
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "embedding_model": self.EMBEDDING_MODEL,
            "llm_model": self.llm_model,
            "llm_backend": self.llm_backend,
            "documents_indexed": list(set(chunk.source for chunk in self.chunks))
        }


if __name__ == "__main__":
    rag = RAGPipeline()
    print(json.dumps(rag.get_stats(), indent=2))

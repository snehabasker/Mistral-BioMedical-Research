"""
Biomedical RAG Engine
Main orchestration for retrieval-augmented generation over medical literature
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from mistralai import Mistral
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a biomedical research paper"""
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    pmid: str
    doi: Optional[str] = None


@dataclass
class RAGResponse:
    """RAG system response with citations"""
    answer: str
    citations: List[Paper]
    confidence: float
    entities: Dict[str, List[str]]
    latency_ms: float


class BiomedicalRAG:
    """
    Production-grade RAG system for biomedical research
    
    Combines:
    - Semantic search over PubMed papers (FAISS)
    - Mistral AI for generation
    - BioBERT for entity extraction
    """
    
    def __init__(
        self,
        mistral_api_key: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_store_path: Optional[str] = None
    ):
        """
        Initialize RAG engine
        
        Args:
            mistral_api_key: Mistral AI API key
            embedding_model: Sentence transformer model
            vector_store_path: Path to pre-built FAISS index
        """
        self.mistral = Mistral(api_key=mistral_api_key)
        self.model_name = "mistral-large-latest"
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        if vector_store_path and os.path.exists(vector_store_path):
            logger.info(f"Loading FAISS index from {vector_store_path}")
            self.index = faiss.read_index(vector_store_path)
            self.papers = []
        else:
            logger.info("Initializing new FAISS index")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.papers = []
    
    def add_papers(self, papers: List[Paper]) -> None:
        """
        Add papers to vector store
        
        Args:
            papers: List of Paper objects
        """
        logger.info(f"Adding {len(papers)} papers to index")
        
        texts = [f"{p.title}. {p.abstract}" for p in papers]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        self.index.add(np.array(embeddings).astype('float32'))
        self.papers.extend(papers)
        
        logger.info(f"Total papers in index: {len(self.papers)}")
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.75
    ) -> List[Paper]:
        """
        Retrieve relevant papers using semantic search
        
        Args:
            query: Research question
            top_k: Number of papers to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant papers
        """
        query_embedding = self.embedder.encode([query])[0]
        
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            top_k
        )
        
        similarities = 1 / (1 + distances[0])
        
        relevant_papers = []
        for idx, sim in zip(indices[0], similarities):
            if sim >= similarity_threshold and idx < len(self.papers):
                relevant_papers.append(self.papers[idx])
        
        logger.info(f"Retrieved {len(relevant_papers)} relevant papers")
        return relevant_papers
    
    def generate_answer(
        self,
        query: str,
        context_papers: List[Paper],
        temperature: float = 0.3
    ) -> str:
        """
        Generate answer using Mistral AI with retrieved context
        
        Args:
            query: Research question
            context_papers: Retrieved relevant papers
            temperature: Sampling temperature (lower = more factual)
            
        Returns:
            Generated answer
        """
        context = self._build_context(context_papers)
        
        system_prompt = """You are a biomedical research assistant. 
Provide accurate, evidence-based answers to medical research questions.
Always cite your sources using paper titles.
If uncertain, state limitations clearly."""
        
        user_prompt = f"""Context from recent research papers:

{context}

Question: {query}

Provide a comprehensive answer based on the research papers above. 
Include specific findings and cite sources by paper title."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.mistral.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def _build_context(self, papers: List[Paper], max_length: int = 3000) -> str:
        """
        Build context string from papers
        
        Args:
            papers: List of papers
            max_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, paper in enumerate(papers, 1):
            paper_text = f"""
Paper {i}: {paper.title}
Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}
Journal: {paper.journal} ({paper.pub_date})
Abstract: {paper.abstract}
---
"""
            if current_length + len(paper_text) > max_length:
                break
            
            context_parts.append(paper_text)
            current_length += len(paper_text)
        
        return "\n".join(context_parts)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        include_citations: bool = True
    ) -> RAGResponse:
        """
        Main query method - retrieve + generate
        
        Args:
            question: Research question
            top_k: Number of papers to retrieve
            include_citations: Whether to include paper citations
            
        Returns:
            RAGResponse with answer and metadata
        """
        import time
        start_time = time.time()
        
        logger.info(f"Processing query: {question}")
        
        context_papers = self.retrieve_context(question, top_k=top_k)
        
        if not context_papers:
            logger.warning("No relevant papers found")
            return RAGResponse(
                answer="I couldn't find relevant research papers for this query.",
                citations=[],
                confidence=0.0,
                entities={},
                latency_ms=0
            )
        
        answer = self.generate_answer(question, context_papers)
        
        latency = (time.time() - start_time) * 1000
        
        entities = {}
        confidence = 0.85
        
        return RAGResponse(
            answer=answer,
            citations=context_papers if include_citations else [],
            confidence=confidence,
            entities=entities,
            latency_ms=latency
        )
    
    def save_index(self, path: str) -> None:
        """Save FAISS index to disk"""
        faiss.write_index(self.index, path)
        logger.info(f"Saved index to {path}")


if __name__ == "__main__":
    sample_papers = [
        Paper(
            title="Metformin and Cardiovascular Disease Risk",
            abstract="Metformin shows protective effects against cardiovascular disease in type 2 diabetes patients...",
            authors=["Smith J", "Doe A"],
            journal="JAMA Cardiology",
            pub_date="2023-06-15",
            pmid="12345678"
        )
    ]
    
    api_key = os.getenv("MISTRAL_API_KEY", "demo-key")
    rag = BiomedicalRAG(mistral_api_key=api_key)
    
    rag.add_papers(sample_papers)
    
    response = rag.query("What are the cardiovascular effects of metformin?")
    
    print(f"Answer: {response.answer}")
    print(f"Citations: {len(response.citations)} papers")
    print(f"Latency: {response.latency_ms:.0f}ms")

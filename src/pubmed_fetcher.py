"""
PubMed Data Fetcher
Retrieves biomedical papers from NCBI PubMed database
"""

import time
from typing import List, Optional, Tuple
from dataclasses import dataclass
import requests
from xml.etree import ElementTree as ET
import logging

logger = logging.getLogger(__name__)


@dataclass
class PubMedPaper:
    """Represents a PubMed paper"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    doi: Optional[str] = None


class PubMedFetcher:
    """
    Fetch papers from PubMed via NCBI E-utilities API
    
    Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25500/
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """
        Initialize PubMed fetcher
        
        Args:
            api_key: NCBI API key (recommended for higher rate limits)
            email: Email for NCBI (required by API guidelines)
        """
        self.api_key = api_key
        self.email = email or "sneha.basker@student-cs.fr"
        self.rate_limit_delay = 0.34 if api_key else 0.34  # 3 requests/sec or 10/sec with key
    
    def search(
        self,
        query: str,
        max_results: int = 100,
        date_range: Optional[Tuple[int, int]] = None
    ) -> List[str]:
        """
        Search PubMed and return list of PMIDs
        
        Args:
            query: Search query (e.g., "Alzheimer's disease treatment")
            max_results: Maximum number of results
            date_range: Optional (start_year, end_year) tuple
            
        Returns:
            List of PMIDs
        """
        # Build search URL
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "tool": "mistral-biomedical-research",
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        if date_range:
            start_year, end_year = date_range
            params["mindate"] = f"{start_year}/01/01"
            params["maxdate"] = f"{end_year}/12/31"
            params["datetype"] = "pdat"
        
        # Make request
        url = f"{self.BASE_URL}/esearch.fcgi"
        logger.info(f"Searching PubMed: {query}")
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(pmids)} papers")
            
            time.sleep(self.rate_limit_delay)  # Rate limiting
            return pmids
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def fetch_details(self, pmids: List[str]) -> List[PubMedPaper]:
        """
        Fetch full details for list of PMIDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of PubMedPaper objects
        """
        if not pmids:
            return []
        
        # Fetch in batches of 200 (API limit)
        batch_size = 200
        all_papers = []
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            papers = self._fetch_batch(batch)
            all_papers.extend(papers)
            time.sleep(self.rate_limit_delay)
        
        logger.info(f"Fetched details for {len(all_papers)} papers")
        return all_papers
    
    def _fetch_batch(self, pmids: List[str]) -> List[PubMedPaper]:
        """Fetch a batch of paper details"""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": "mistral-biomedical-research",
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.BASE_URL}/efetch.fcgi"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            papers = []
            
            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to fetch batch: {e}")
            return []
    
    def _parse_article(self, article_elem) -> Optional[PubMedPaper]:
        """Parse XML element into PubMedPaper"""
        try:
            # PMID
            pmid = article_elem.find(".//PMID").text
            
            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = "".join(title_elem.itertext()) if title_elem is not None else ""
            
            # Abstract
            abstract_parts = []
            for abstract_text in article_elem.findall(".//AbstractText"):
                text = "".join(abstract_text.itertext())
                abstract_parts.append(text)
            abstract = " ".join(abstract_parts)
            
            # Authors
            authors = []
            for author in article_elem.findall(".//Author"):
                lastname = author.find("LastName")
                forename = author.find("ForeName")
                if lastname is not None:
                    name = lastname.text
                    if forename is not None:
                        name = f"{forename.text} {name}"
                    authors.append(name)
            
            # Journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown"
            
            # Publication date
            pub_date_elem = article_elem.find(".//PubDate")
            pub_date = self._parse_date(pub_date_elem)
            
            # DOI
            doi = None
            for article_id in article_elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            return PubMedPaper(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                pub_date=pub_date,
                doi=doi
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return None
    
    def _parse_date(self, date_elem) -> str:
        """Parse PubDate XML element"""
        if date_elem is None:
            return "Unknown"
        
        year = date_elem.find("Year")
        month = date_elem.find("Month")
        day = date_elem.find("Day")
        
        parts = []
        if year is not None:
            parts.append(year.text)
        if month is not None:
            parts.append(month.text)
        if day is not None:
            parts.append(day.text)
        
        return "-".join(parts) if parts else "Unknown"


# Example usage
if __name__ == "__main__":
    import os
    
    fetcher = PubMedFetcher(
        api_key=os.getenv("NCBI_API_KEY"),
        email="sneha.basker@student-cs.fr"
    )
    
    # Search
    pmids = fetcher.search(
        query="Alzheimer's disease treatment",
        max_results=10,
        date_range=(2023, 2024)
    )
    
    # Fetch details
    papers = fetcher.fetch_details(pmids)
    
    # Display
    for paper in papers:
        print(f"\nTitle: {paper.title}")
        print(f"Authors: {', '.join(paper.authors[:3])}")
        print(f"Journal: {paper.journal}")
        print(f"Abstract: {paper.abstract[:200]}...")

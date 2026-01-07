"""
Streamlit Web Interface for Biomedical Research Assistant
"""

import streamlit as st
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pubmed_fetcher import PubMedFetcher, PubMedPaper
from src.rag_engine import BiomedicalRAG, Paper

# Page config
st.set_page_config(
    page_title="🧬 Mistral Biomedical Research Assistant",
    page_icon="🧬",
    layout="wide"
)

# Title
st.title("🧬 Mistral Biomedical Research Assistant")
st.markdown("*AI-powered medical literature analysis using Mistral AI + PubMed*")

# Sidebar - API Keys
with st.sidebar:
    st.header("⚙️ Configuration")
    
    mistral_key = st.text_input(
        "Mistral API Key",
        type="password",
        help="Get your key at https://console.mistral.ai/"
    )
    
    ncbi_key = st.text_input(
        "NCBI API Key (Optional)",
        type="password",
        help="Optional - increases rate limit"
    )
    
    st.markdown("---")
    st.header("📊 Search Settings")
    
    max_papers = st.slider(
        "Max Papers to Retrieve",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    
    date_range = st.select_slider(
        "Publication Date Range",
        options=["2020-2024", "2022-2024", "2023-2024", "2024 only"],
        value="2022-2024"
    )
    
    st.markdown("---")
    st.markdown("""
    **Built by:** Sneha Basker  
    **CentraleSupélec, Paris**  
    [GitHub](https://github.com/snehabasker) | [LinkedIn](https://linkedin.com/in/sneha-basker-3583601a7)
    """)

# Main area
tab1, tab2, tab3 = st.tabs(["🔍 Research Query", "📚 Literature Review", "📊 Analytics"])

with tab1:
    st.header("Ask a Research Question")
    
    # Example queries
    st.markdown("**Example queries:**")
    examples = [
        "What are the latest treatments for Alzheimer's disease?",
        "Side effects of metformin in diabetes patients",
        "Role of gut microbiome in Parkinson's disease",
        "COVID-19 vaccine effectiveness in immunocompromised patients"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💡 {example[:50]}...", key=f"ex_{i}"):
                st.session_state.query = example
    
    # Query input
    query = st.text_area(
        "Enter your research question:",
        value=st.session_state.get('query', ''),
        height=100,
        placeholder="e.g., What are the mechanisms of CRISPR gene editing in cancer treatment?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔬 Search & Analyze", type="primary", use_container_width=True)
    
    if search_button and query:
        if not mistral_key:
            st.error("⚠️ Please enter your Mistral API key in the sidebar")
        else:
            with st.spinner("🔍 Searching PubMed database..."):
                # Parse date range
                if date_range == "2024 only":
                    years = (2024, 2024)
                else:
                    start, end = date_range.split("-")
                    years = (int(start), int(end))
                
                # Initialize fetcher
                fetcher = PubMedFetcher(api_key=ncbi_key if ncbi_key else None)
                
                # Search PubMed
                pmids = fetcher.search(query, max_results=max_papers, date_range=years)
                
                if not pmids:
                    st.warning("No papers found. Try a different query or date range.")
                else:
                    st.success(f"✅ Found {len(pmids)} papers")
                    
                    # Fetch details
                    with st.spinner("📥 Fetching paper details..."):
                        pubmed_papers = fetcher.fetch_details(pmids)
                    
                    # Convert to Paper format
                    papers = [
                        Paper(
                            title=p.title,
                            abstract=p.abstract,
                            authors=p.authors,
                            journal=p.journal,
                            pub_date=p.pub_date,
                            pmid=p.pmid,
                            doi=p.doi
                        )
                        for p in pubmed_papers
                    ]
                    
                    # Initialize RAG
                    with st.spinner("🧠 Analyzing with Mistral AI..."):
                        rag = BiomedicalRAG(mistral_api_key=mistral_key)
                        rag.add_papers(papers)
                        
                        # Generate response
                        response = rag.query(query, top_k=5)
                    
                    # Display results
                    st.markdown("---")
                    st.header("📋 AI-Generated Summary")
                    
                    # Answer
                    st.markdown(response.answer)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Papers Analyzed", len(papers))
                    with col2:
                        st.metric("Response Time", f"{response.latency_ms:.0f}ms")
                    with col3:
                        st.metric("Confidence", f"{response.confidence:.0%}")
                    
                    # Citations
                    st.markdown("---")
                    st.header("📚 Key Citations")
                    
                    for i, paper in enumerate(response.citations, 1):
                        with st.expander(f"📄 Paper {i}: {paper.title}"):
                            st.markdown(f"**Authors:** {', '.join(paper.authors[:5])}")
                            st.markdown(f"**Journal:** {paper.journal} ({paper.pub_date})")
                            st.markdown(f"**PubMed ID:** {paper.pmid}")
                            if paper.doi:
                                st.markdown(f"**DOI:** [{paper.doi}](https://doi.org/{paper.doi})")
                            st.markdown(f"**Abstract:** {paper.abstract}")

with tab2:
    st.header("📚 Comprehensive Literature Review")
    st.info("🚧 Coming soon: Automated systematic review generation")
    
    st.markdown("""
    **Features in development:**
    - Automated PRISMA-compliant reviews
    - Citation network visualization
    - Trend analysis over time
    - Comparison across studies
    """)

with tab3:
    st.header("📊 Research Analytics")
    st.info("🚧 Coming soon: Research analytics dashboard")
    
    st.markdown("""
    **Features in development:**
    - Publication trends
    - Author collaboration networks
    - Journal impact analysis
    - Geographic distribution of research
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>⚠️ Disclaimer:</strong> This tool is for research purposes only. 
    Not a substitute for professional medical advice.</p>
    <p>Built with ❤️ using Mistral AI | © 2026 Sneha Basker</p>
</div>
""", unsafe_allow_html=True)

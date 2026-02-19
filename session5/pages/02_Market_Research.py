
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional, Any
import io
import os
import boto3
from botocore.exceptions import ClientError
import logging
import json
import tempfile
from pathlib import Path
import utils.common as common
import utils.authenticate as authenticate
import utils.styles as styles

# Updated LangChain imports (v0.1.0+)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Configure logging with structured logging
import structlog
logger = structlog.get_logger()

# Utility functions
def safe_extract_numeric(text: str, default: float = 0.0) -> float:
    """Safely extract numeric value from text with multiple pattern support"""
    if not text or text == "Not specified":
        return default
    
    patterns = [
        r'(\d+(?:\.\d+)?)\s*%',  # "5.5%"
        r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)',  # "5-7"
        r'(\d+(?:\.\d+)?)'  # "5.5"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                # For range, take average
                if match.lastindex == 2:
                    return (float(match.group(1)) + float(match.group(2))) / 2
                return float(match.group(1))
            except (ValueError, AttributeError):
                continue
    
    return default

def clean_markdown_for_display(text: str) -> str:
    """Clean markdown formatting from text for better display in Streamlit"""
    if not text:
        return text
    
    # Remove markdown headers (##, ###, etc.) but keep the text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold markers but keep the text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    
    # Remove italic markers but keep the text
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove inline code markers
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up leading/trailing whitespace
    text = text.strip()
    
    return text

# Configure page settings
st.set_page_config(
    page_title="Shoe Industry Market Research Analyzer",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with type hints
class SessionState:
    """Manage session state with type safety"""
    
    @staticmethod
    def initialize():
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = None
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = None
        if "document_processed" not in st.session_state:
            st.session_state.document_processed = False
        if "documents" not in st.session_state:
            st.session_state.documents = []
        if "bedrock_client" not in st.session_state:
            st.session_state.bedrock_client = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "comparison_analyses" not in st.session_state:
            st.session_state.comparison_analyses = []
        if "recommendations" not in st.session_state:
            st.session_state.recommendations = None
        if "specifications" not in st.session_state:
            st.session_state.specifications = None

SessionState.initialize()

# Custom CSS for modern UI/UX
def load_custom_css():
    """Load custom CSS for modern UI/UX design"""
    st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-color: #FF9900;
        --primary-dark: #EC7211;
        --secondary-color: #232F3E;
        --accent-color: #146EB4;
        --success-color: #1D8102;
        --warning-color: #FF9900;
        --danger-color: #D13212;
        --dark-color: #16191F;
        --light-color: #FAFAFA;
        --border-color: #E5E7EB;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #FAFAFA 0%, #F3F4F6 100%);
        padding: 2rem 1rem;
    }
    
    /* Header styles */
    .main-header {
        font-size: 2.75rem;
        font-weight: 700;
        color: var(--secondary-color);
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--accent-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeInDown 0.6s ease-out;
    }
    
    .subtitle {
        font-size: 1.125rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Card styles */
    .metric-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        padding: 1.75rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-lg);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl);
    }
    
    .analysis-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.25rem 0;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .analysis-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-color);
    }
    
    .recommendation-item {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 4px solid var(--success-color);
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .recommendation-item:hover {
        transform: translateX(4px);
        box-shadow: var(--shadow-md);
    }
    
    .specification-item {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 4px solid var(--warning-color);
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .specification-item:hover {
        transform: translateX(4px);
        box-shadow: var(--shadow-md);
    }
    
    /* Footer styles */
    .footer {
        margin-top: 4rem;
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, var(--secondary-color) 0%, #16191F 100%);
        color: white;
        text-align: center;
        border-radius: 16px 16px 0 0;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--accent-color) 100%);
    }
    
    .footer-content {
        position: relative;
        z-index: 1;
    }
    
    .footer-text {
        color: #D1D5DB;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 56px;
        padding: 0 24px;
        background: white;
        border-radius: 12px 12px 0 0;
        color: #6B7280;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        border: 1px solid var(--border-color);
        border-bottom: none;
        box-shadow: var(--shadow-sm);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #F9FAFB;
        color: var(--secondary-color);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        box-shadow: var(--shadow-md);
        border-color: var(--primary-color);
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-md);
        letter-spacing: 0.01em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, #C45500 100%);
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent-color) 0%, #0D5A94 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0D5A94 0%, #084673 100%);
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Input styles */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid var(--border-color);
        padding: 0.75rem 1rem;
        transition: all 0.2s ease;
        font-size: 0.95rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(255, 153, 0, 0.1);
    }
    
    /* File uploader */
    .stFileUploader {
        background: white;
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background: #FFFBF5;
    }
    
    /* Metric styles */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--secondary-color);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Expander styles */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        padding: 1rem 1.25rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #F9FAFB;
        border-color: var(--primary-color);
    }
    
    /* Dataframe styles */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    /* Chat message styles */
    .stChatMessage {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
    }
    
    /* Sidebar styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, white 0%, #F9FAFB 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stSlider {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        border-radius: 10px;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: var(--shadow-sm);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
        
        .metric-card {
            padding: 1.25rem;
            margin: 0.5rem 0;
        }
        
        .analysis-card {
            padding: 1.25rem;
        }
        
        .footer {
            padding: 2rem 1rem;
        }
        
        .stButton > button {
            width: 100%;
            padding: 0.875rem;
        }
    }
    
    /* Tablet */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header {
            font-size: 2.25rem;
        }
        
        .subtitle {
            font-size: 1.0625rem;
        }
    }
    
    /* Dark mode support (optional) */
    @media (prefers-color-scheme: dark) {
        .main {
            background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        }
        
        .analysis-card {
            background: #1F2937;
            border-color: #374151;
            color: #F9FAFB;
        }
    }
    </style>
    """, unsafe_allow_html=True)

class BedrockService:
    """Service class for AWS Bedrock operations"""
    
    def __init__(self, region_name: str = 'us-east-1'):
        self.region_name = region_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Bedrock client with error handling"""
        try:
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region_name
            )
            logger.info("Bedrock client initialized successfully")
        except Exception as e:
            logger.error(f"Error creating Bedrock client: {str(e)}")
            raise
    
    def get_embeddings(self) -> BedrockEmbeddings:
        """Get Bedrock embeddings instance"""
        return BedrockEmbeddings(
            client=self.client,
            model_id="amazon.titan-embed-text-v2:0"  # Updated to v2
        )
    
    def invoke_model(self, model_id: str, system_prompt: str, user_prompt: str, 
                    temperature: float = 0.7, max_tokens: int = 2048) -> Optional[str]:
        """Invoke Bedrock model with improved error handling"""
        
        system_prompts = [{"text": system_prompt}]
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        
        try:
            response = self.client.converse(
                modelId=model_id,
                messages=[message],
                system=system_prompts,
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens
                }
            )
            
            return response['output']['message']['content'][0].get('text', '')
            
        except ClientError as err:
            logger.error(f"Bedrock API error: {err.response['Error']['Message']}")
            st.error(f"Error: {err.response['Error']['Message']}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error invoking model: {str(e)}")
            return None

class DocumentProcessor:
    """Modern document processing with improved error handling"""
    
    def __init__(self, bedrock_service: BedrockService):
        self.bedrock_service = bedrock_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Increased for better context
            chunk_overlap=200,  # Increased for better continuity
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    
    def process_files(self, files: List) -> Tuple[Optional[FAISS], List[Document]]:
        """Process uploaded files with improved handling"""
        if not files:
            return None, []
        
        all_documents = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                try:
                    # Save file to temporary directory
                    temp_path = Path(temp_dir) / file.name
                    temp_path.write_bytes(file.getvalue())
                    
                    # Load document
                    documents = self._load_document(str(temp_path))
                    
                    # Add metadata
                    for doc in documents:
                        doc.metadata.update({
                            'source_file': file.name,
                            'file_type': Path(file.name).suffix,
                            'upload_time': datetime.now().isoformat()
                        })
                    
                    all_documents.extend(documents)
                    
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")
                    st.error(f"Error loading {file.name}: {str(e)}")
        
        if not all_documents:
            return None, []
        
        # Split documents
        document_chunks = self.text_splitter.split_documents(all_documents)
        
        # Create vector store
        try:
            with st.spinner("Creating vector embeddings..."):
                embeddings = self.bedrock_service.get_embeddings()
                vectorstore = FAISS.from_documents(document_chunks, embeddings)
                logger.info(f"Created vectorstore with {len(document_chunks)} chunks")
                return vectorstore, document_chunks
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            st.error(f"Error creating embeddings: {str(e)}")
            return None, []
    
    def _load_document(self, file_path: str) -> List[Document]:
        """Load document based on file type"""
        file_extension = Path(file_path).suffix.lower()
        
        loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
        }
        
        loader_class = loader_map.get(file_extension)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        loader = loader_class(file_path)
        return loader.load()

class RAGQueryEngine:
    """Modern RAG query engine with improved retrieval"""
    
    def __init__(self, vectorstore: Optional[FAISS], bedrock_service: BedrockService, 
                 model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        self.vectorstore = vectorstore
        self.bedrock_service = bedrock_service
        self.model_id = model_id
    
    def query(self, query: str, system_prompt: str, k: int = 6) -> Optional[str]:
        """Execute RAG query with context retrieval"""
        
        if not self.vectorstore:
            # Fallback to direct query without context
            return self.bedrock_service.invoke_model(
                self.model_id,
                system_prompt,
                query
            )
        
        # Retrieve relevant documents
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            context = self._format_context(docs)
            
            # Build augmented prompt
            augmented_prompt = f"""{context}

Based on the above context, {query}

Please provide a detailed analysis with specific examples and data points from the context."""
            
            return self.bedrock_service.invoke_model(
                self.model_id,
                system_prompt,
                augmented_prompt
            )
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return None
    
    def _format_context(self, docs: List[Document]) -> str:
        """Format documents for context inclusion"""
        if not docs:
            return ""
        
        context_parts = ["---\nCONTEXT INFORMATION:\n"]
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Document {i}:\n{doc.page_content}\n")
        context_parts.append("---")
        
        return "\n".join(context_parts)

class MarketAnalyzer:
    """Enhanced market analyzer with modern patterns"""
    
    def __init__(self, rag_engine: RAGQueryEngine):
        self.rag_engine = rag_engine
    
    def analyze_document(self) -> Dict[str, Any]:
        """Perform comprehensive market analysis"""
        
        analysis = {
            'opportunities': self._extract_opportunities(),
            'market_insights': self._extract_market_insights(),
            'customer_segments': self._extract_customer_segments(),
            'competitive_landscape': self._extract_competitive_landscape(),
            'feasibility_score': self._calculate_feasibility_score()
        }
        
        return analysis
    
    def _extract_opportunities(self) -> List[str]:
        """Extract market opportunities using RAG"""
        query = "What are the key market opportunities mentioned in the document? List the top 5 opportunities with specific details."
        system_prompt = "You are a market research analyst. Extract and list market opportunities from the provided context."
        
        response = self.rag_engine.query(query, system_prompt)
        
        if response:
            opportunities = self._parse_list_response(response, max_items=5)
            if opportunities:
                return opportunities
        
        return self._get_default_opportunities()
    
    def _extract_market_insights(self) -> Dict[str, Any]:
        """Extract market insights with improved parsing"""
        
        query = "What are the market size, growth rate, and key trends mentioned in the document? Provide specific numbers and data points."
        system_prompt = "You are a market research analyst. Extract market metrics and trends from the provided context."
        
        response = self.rag_engine.query(query, system_prompt)
        
        insights = {
            'market_size': "Not specified",
            'growth_rate': "Not specified",
            'key_trends': []
        }
        
        if response:
            # Enhanced parsing with regex patterns
            patterns = {
                'market_size': r'\$[\d,.]+ (?:billion|million|trillion)',
                'growth_rate': r'(\d+(?:\.\d+)?)\s*%.*?(?:growth|CAGR|annually)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    insights[key] = match.group()
            
            # Extract trends
            insights['key_trends'] = self._parse_list_response(response, keyword='trend', max_items=4)
            
            # If no trends found, try to extract any bulleted or numbered items
            if not insights['key_trends']:
                insights['key_trends'] = self._parse_list_response(response, max_items=4)
        
        # Ensure we have at least some default values if nothing was extracted
        if insights['market_size'] == "Not specified":
            insights['market_size'] = "$280 billion (estimated)"
        if insights['growth_rate'] == "Not specified":
            insights['growth_rate'] = "5-7% annually"
        if not insights['key_trends']:
            insights['key_trends'] = ["Sustainable materials", "Digital integration", "Customization"]
        
        return insights
    
    def _extract_customer_segments(self) -> List[Dict[str, str]]:
        """Extract customer segments with improved structure"""
        query = """Based on the market research document, identify and describe 3-5 distinct customer segments.
        
        For EACH segment, provide:
        1. Segment Name (e.g., "Young Professionals", "Athletes", "Eco-Conscious Consumers")
        2. Market Size or percentage if available
        3. Key characteristics, needs, and preferences
        
        Format your response as:
        
        SEGMENT 1: [Name]
        Size: [percentage or description]
        Characteristics: [detailed description]
        
        SEGMENT 2: [Name]
        Size: [percentage or description]
        Characteristics: [detailed description]
        """
        
        system_prompt = "You are a market analyst. Extract detailed customer segment information from the context. Format your response clearly with segment names, sizes, and characteristics."
        
        response = self.rag_engine.query(query, system_prompt, k=6)
        
        if response:
            segments = self._parse_segments(response)
            if segments and len(segments) > 0:
                return segments[:5]
        
        return self._get_default_segments()
    
    def _extract_competitive_landscape(self) -> Dict[str, List[str]]:
        """Extract competitive landscape with enhanced parsing"""
        query = """Analyze the competitive landscape mentioned in the document. 
        Include major competitors, market gaps, and potential competitive advantages."""
        system_prompt = "You are a competitive intelligence analyst. Extract competitive landscape details from the context."
        
        response = self.rag_engine.query(query, system_prompt)
        
        landscape = {
            'major_players': [],
            'market_gaps': [],
            'competitive_advantage_opportunities': []
        }
        
        if response:
            # Extract competitors (look for company names)
            company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            potential_companies = re.findall(company_pattern, response)
            
            # Filter for likely company names
            for company in potential_companies[:10]:
                if any(keyword in company.lower() for keyword in ['nike', 'adidas', 'puma', 'reebok', 'balance']):
                    landscape['major_players'].append(company)
            
            # Extract gaps and opportunities
            landscape['market_gaps'] = self._parse_list_response(response, keyword='gap', max_items=3)
            landscape['competitive_advantage_opportunities'] = self._parse_list_response(
                response, keyword='advantage', max_items=3
            )
        
        # Ensure we have some data
        if not landscape['major_players']:
            landscape['major_players'] = ['Nike', 'Adidas', 'Puma', 'Under Armour', 'New Balance']
        
        return landscape
    
    def _calculate_feasibility_score(self) -> int:
        """Calculate feasibility score with improved algorithm"""
        query = """Analyze the overall feasibility of entering this market. 
        Consider positive factors (opportunities, growth, demand) and negative factors (competition, barriers, risks)."""
        system_prompt = "You are a market feasibility analyst. Evaluate market entry feasibility based on the context."
        
        response = self.rag_engine.query(query, system_prompt)
        
        base_score = 70
        
        if response:
            # Sentiment-based scoring
            positive_indicators = [
                'growth', 'opportunity', 'demand', 'increasing', 'potential',
                'expanding', 'emerging', 'favorable', 'strong', 'attractive'
            ]
            negative_indicators = [
                'challenge', 'competition', 'decline', 'saturated', 'difficult',
                'barrier', 'risk', 'threat', 'weakness', 'unfavorable'
            ]
            
            response_lower = response.lower()
            
            # Calculate weighted scores
            positive_count = sum(response_lower.count(word) for word in positive_indicators)
            negative_count = sum(response_lower.count(word) for word in negative_indicators)
            
            # Adjust score based on sentiment
            score_adjustment = (positive_count * 2) - (negative_count * 1.5)
            base_score += min(25, max(-25, score_adjustment))
        
        return min(95, max(30, int(base_score)))
    
    def _parse_list_response(self, response: str, keyword: str = '', max_items: int = 5) -> List[str]:
        """Parse list items from response text"""
        items = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a list item
            if (line[0].isdigit() and '.' in line[:3]) or line.startswith('•') or line.startswith('-'):
                # Clean up the line
                clean_line = re.sub(r'^[\d\.\-\•\*]+\s*', '', line)
                # Clean markdown formatting
                clean_line = clean_markdown_for_display(clean_line)
                if keyword and keyword.lower() not in clean_line.lower():
                    continue
                items.append(clean_line)
            elif keyword and keyword.lower() in line.lower():
                # Clean markdown formatting
                clean_line = clean_markdown_for_display(line)
                items.append(clean_line)
        
        return items[:max_items]
    
    def _parse_segments(self, response: str) -> List[Dict[str, str]]:
        """Parse customer segments from response with improved logic"""
        segments = []
        current_segment = None
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a segment header (SEGMENT 1:, SEGMENT 2:, etc.)
            segment_match = re.match(r'(?:SEGMENT|Segment)\s*(\d+)[:\-\s]+(.+)', line, re.IGNORECASE)
            if segment_match:
                # Save previous segment
                if current_segment and current_segment.get('segment'):
                    segments.append(current_segment)
                
                # Start new segment
                segment_name = clean_markdown_for_display(segment_match.group(2))
                current_segment = {
                    'segment': segment_name,
                    'size': 'Not specified',
                    'characteristics': ''
                }
                continue
            
            # Check for numbered list items (1., 2., etc.) as segment headers
            numbered_match = re.match(r'^\d+[\.\)]\s*(.+)', line)
            if numbered_match and not current_segment:
                segment_name = clean_markdown_for_display(numbered_match.group(1))
                # Only treat as segment if it looks like a segment name (not too long)
                if len(segment_name) < 100 and not segment_name.lower().startswith(('size:', 'characteristics:')):
                    if current_segment and current_segment.get('segment'):
                        segments.append(current_segment)
                    current_segment = {
                        'segment': segment_name,
                        'size': 'Not specified',
                        'characteristics': ''
                    }
                    continue
            
            # Parse size information
            if current_segment and (line.lower().startswith('size:') or line.lower().startswith('market size:')):
                size_text = re.sub(r'^(?:market\s+)?size:\s*', '', line, flags=re.IGNORECASE)
                current_segment['size'] = clean_markdown_for_display(size_text)
                continue
            
            # Parse characteristics
            if current_segment and (line.lower().startswith('characteristics:') or 
                                   line.lower().startswith('description:') or
                                   line.lower().startswith('needs:')):
                char_text = re.sub(r'^(?:characteristics|description|needs):\s*', '', line, flags=re.IGNORECASE)
                current_segment['characteristics'] = clean_markdown_for_display(char_text)
                continue
            
            # Add to characteristics if we have a current segment and line has content
            if current_segment and len(line) > 10:
                # Skip lines that look like headers
                if not any(line.lower().startswith(prefix) for prefix in ['segment', 'based on', 'according to']):
                    clean_line = clean_markdown_for_display(line)
                    if current_segment['characteristics']:
                        current_segment['characteristics'] += ' ' + clean_line
                    else:
                        current_segment['characteristics'] = clean_line
        
        # Add the last segment
        if current_segment and current_segment.get('segment'):
            segments.append(current_segment)
        
        # If no segments found with structured parsing, try alternative approach
        if not segments:
            # Look for any lines that might be segment names
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 5:
                    continue
                
                # Check if line looks like a segment name (capitalized, not too long)
                if (line[0].isupper() and len(line) < 80 and 
                    not line.lower().startswith(('the ', 'this ', 'these ', 'based ', 'according '))):
                    
                    # Get next few lines as characteristics
                    characteristics = []
                    for j in range(i+1, min(i+4, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and len(next_line) > 10:
                            characteristics.append(clean_markdown_for_display(next_line))
                    
                    if characteristics:
                        segments.append({
                            'segment': clean_markdown_for_display(line),
                            'size': 'Not specified',
                            'characteristics': ' '.join(characteristics)
                        })
                        
                        if len(segments) >= 5:
                            break
        
        return segments
    
    def _get_default_opportunities(self) -> List[str]:
        """Default opportunities for demonstration"""
        return [
            "Growing demand for sustainable and eco-friendly footwear",
            "Increasing adoption of smart shoe technology and wearables",
            "Rising health consciousness driving athletic footwear sales",
            "Customization and personalization trends in footwear",
            "E-commerce and direct-to-consumer market opportunities"
        ]
    
    def _get_default_segments(self) -> List[Dict[str, str]]:
        """Default customer segments for demonstration"""
        return [
            {
                'segment': 'Health-Conscious Millennials',
                'size': '25-35% of market',
                'characteristics': 'Values sustainability, technology integration, comfort, and wellness features'
            },
            {
                'segment': 'Fashion-Forward Gen Z',
                'size': '20-30% of market',
                'characteristics': 'Social media influenced, trendy designs, affordable luxury, brand storytelling'
            },
            {
                'segment': 'Professional Athletes & Fitness Enthusiasts',
                'size': '15-20% of market',
                'characteristics': 'Performance-focused, willing to pay premium, brand loyal, technical features'
            }
        ]

class ProductRecommender:
    """Enhanced product recommender with modern patterns"""
    
    def __init__(self, rag_engine: RAGQueryEngine):
        self.rag_engine = rag_engine
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive product recommendations"""
        
        query = """Based on the market research document, provide detailed product recommendations including:
        1. Target customer segments and their specific needs
        2. Key product attributes and features
        3. Brand positioning strategy
        4. Pricing strategy and justification"""
        
        system_prompt = """You are a product strategy consultant specializing in footwear. 
        Provide actionable recommendations based on market research data."""
        
        response = self.rag_engine.query(query, system_prompt)
        
        if response:
            recommendations = self._parse_recommendations(response, analysis)
            return recommendations
        
        return self._get_default_recommendations()
    
    def generate_specifications(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed product specifications"""
        
        query = """Based on the market research, provide detailed product specifications including:
        1. Materials and sustainability features
        2. Technical specifications and performance metrics
        3. Design features and innovations
        4. Manufacturing requirements"""
        
        system_prompt = """You are a product development specialist in footwear. 
        Provide detailed technical specifications based on market requirements."""
        
        response = self.rag_engine.query(query, system_prompt)
        
        if response:
            specifications = self._parse_specifications(response, analysis)
            return specifications
        
        return self._get_default_specifications()
    
    def _parse_recommendations(self, response: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure recommendations from response"""
        
        recommendations = {
            'target_segments': [],
            'key_attributes': [],
            'positioning': '',
            'pricing_strategy': {
                'strategy': 'Premium value positioning',
                'price_range': '$120-150',
                'justification': 'Balanced approach between quality and affordability'
            }
        }
        
        # Use market analyzer's parsing methods for consistency
        analyzer = MarketAnalyzer(self.rag_engine)
        
        # Extract segments
        if 'segment' in response.lower():
            segments = analyzer._parse_list_response(response, keyword='segment', max_items=3)
            recommendations['target_segments'] = segments
        
        # Extract attributes
        if 'attribute' in response.lower() or 'feature' in response.lower():
            attributes = analyzer._parse_list_response(response, keyword='', max_items=5)
            recommendations['key_attributes'] = attributes
        
        # Extract positioning
        position_match = re.search(r'position[^\n]*:\s*([^\n]+)', response, re.IGNORECASE)
        if position_match:
            positioning = position_match.group(1).strip()
            # Clean markdown from positioning
            recommendations['positioning'] = clean_markdown_for_display(positioning)
        
        # Extract pricing safely
        price_match = re.search(r'\$(\d+)[\-\s]*(?:to|\-)?\s*\$?(\d+)?', response)
        if price_match:
            price_low = price_match.group(1)
            price_high = price_match.group(2) if price_match.group(2) else str(int(price_low) + 30)
            recommendations['pricing_strategy']['price_range'] = f"${price_low}-${price_high}"
        
        # Ensure we have content
        if not recommendations['target_segments']:
            recommendations['target_segments'] = self._get_default_recommendations()['target_segments']
        if not recommendations['key_attributes']:
            recommendations['key_attributes'] = self._get_default_recommendations()['key_attributes']
        if not recommendations['positioning']:
            recommendations['positioning'] = self._get_default_recommendations()['positioning']
        
        return recommendations
    
    def _parse_specifications(self, response: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure specifications from response"""
        
        specs = self._get_default_specifications()
        
        # Try to extract specific information from response
        if response:
            # Extract materials
            if 'material' in response.lower():
                materials_section = response[response.lower().find('material'):]
                analyzer = MarketAnalyzer(self.rag_engine)
                materials = analyzer._parse_list_response(materials_section, max_items=4)
                if materials:
                    specs['technical_specs']['materials'] = materials
            
            # Extract features
            if 'feature' in response.lower():
                features_section = response[response.lower().find('feature'):]
                analyzer = MarketAnalyzer(self.rag_engine)
                features = analyzer._parse_list_response(features_section, max_items=4)
                if features:
                    specs['technical_specs']['features'] = features
        
        return specs
    
    def _get_default_recommendations(self) -> Dict[str, Any]:
        """Default recommendations for demonstration"""
        return {
            'target_segments': [
                "Environmentally conscious millennials (ages 25-40)",
                "Fitness enthusiasts seeking sustainable options",
                "Urban professionals valuing comfort and style"
            ],
            'key_attributes': [
                "Sustainable materials and manufacturing",
                "Superior comfort and ergonomic support",
                "Versatile design for multiple activities",
                "Durability and long-lasting quality",
                "Competitive pricing with premium features"
            ],
            'positioning': "Premium sustainable athletic footwear that doesn't compromise on performance or style",
            'pricing_strategy': {
                'strategy': 'Premium value positioning',
                'price_range': '$120-150',
                'justification': 'Positioned between mass market and luxury, emphasizing value through sustainability and performance'
            }
        }
    
    def _get_default_specifications(self) -> Dict[str, Any]:
        """Default specifications for demonstration"""
        return {
            'product_concept': {
                'name': 'EcoFlex Pro',
                'category': 'Sustainable Athletic Footwear',
                'target_price': '$120-150'
            },
            'technical_specs': {
                'materials': [
                    'Recycled ocean plastic upper (REPREVE®)',
                    'Bio-based EVA foam midsole',
                    'Natural rubber outsole (FSC certified)',
                    'Organic cotton lining (GOTS certified)'
                ],
                'features': [
                    'Advanced moisture-wicking technology',
                    'Antimicrobial silver ion treatment',
                    'Energy-return cushioning system',
                    'Adaptive fit technology'
                ],
                'sizes': 'US 5-15 (including half sizes)',
                'colors': ['Ocean Blue', 'Forest Green', 'Charcoal Gray', 'Pure White', 'Sunset Orange']
            },
            'performance_metrics': {
                'weight': '285g (men\'s size 9)',
                'durability': '600+ miles tested lifespan',
                'water_resistance': 'DWR treated, IPX4 rated',
                'carbon_footprint': '42% lower than industry average',
                'energy_return': '85% energy return rating'
            },
            'sustainability_metrics': {
                'recycled_content': '75% recycled materials',
                'carbon_neutral': 'Carbon neutral manufacturing',
                'packaging': '100% recyclable packaging',
                'end_of_life': 'Take-back program for recycling'
            }
        }

# UI Components
def create_model_selection_panel():
    """Model selection and parameters panel"""
    st.markdown("<h4>Model Selection</h4>", unsafe_allow_html=True)
    
    # Model categories by provider with most advanced models as defaults
    MODEL_CATEGORIES = {
        "Amazon": {
            "Nova 2 Lite": "us.amazon.nova-2-lite-v1:0",
            "Nova Pro": "us.amazon.nova-pro-v1:0",
            "Nova Lite": "us.amazon.nova-lite-v1:0",
            "Nova Micro": "us.amazon.nova-micro-v1:0",
        },
        "Anthropic": {
            "Claude Sonnet 4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "Claude Sonnet 4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "Claude Opus 4.1": "us.anthropic.claude-opus-4-1-20250805-v1:0",
            "Claude Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude 3.5 Haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        },
        "Meta": {
            "Llama 4 Maverick 17B": "us.meta.llama4-maverick-17b-instruct-v1:0",
            "Llama 4 Scout 17B": "us.meta.llama4-scout-17b-instruct-v1:0",
            "Llama 3.3 70B": "us.meta.llama3-3-70b-instruct-v1:0",
            "Llama 3.2 90B": "us.meta.llama3-2-90b-instruct-v1:0",
            "Llama 3.2 11B": "us.meta.llama3-2-11b-instruct-v1:0",
            "Llama 3.2 3B": "us.meta.llama3-2-3b-instruct-v1:0",
            "Llama 3.2 1B": "us.meta.llama3-2-1b-instruct-v1:0",
            "Llama 3.1 70B": "us.meta.llama3-1-70b-instruct-v1:0",
            "Llama 3.1 8B": "us.meta.llama3-1-8b-instruct-v1:0",
            "Llama 3 70B": "meta.llama3-70b-instruct-v1:0",
            "Llama 3 8B": "meta.llama3-8b-instruct-v1:0",
        },
        "Mistral AI": {
            "Mistral Large 3": "mistral.mistral-large-3-675b-instruct",
            "Pixtral Large": "mistral.pixtral-large-2502-v1:0",
            "Magistral Small": "mistral.magistral-small-2509",
            "Ministral 14B": "mistral.ministral-3-14b-instruct",
            "Ministral 8B": "mistral.ministral-3-8b-instruct",
            "Ministral 3B": "mistral.ministral-3-3b-instruct",
            "Voxtral Small 24B": "mistral.voxtral-small-24b-2507",
            "Voxtral Mini 3B": "mistral.voxtral-mini-3b-2507",
            "Mistral Large (24.07)": "mistral.mistral-large-2407-v1:0",
            "Mistral Large (24.02)": "mistral.mistral-large-2402-v1:0",
            "Mixtral 8x7B": "mistral.mixtral-8x7b-instruct-v0:1",
            "Mistral 7B": "mistral.mistral-7b-instruct-v0:2",
        },
        "Google": {
            "Gemma 3 27B": "google.gemma-3-27b-it",
            "Gemma 3 12B": "google.gemma-3-12b-it",
            "Gemma 3 4B": "google.gemma-3-4b-it",
        },
        "Qwen": {
            "Qwen3 Coder 480B": "qwen.qwen3-coder-480b-a35b-v1:0",
            "Qwen3 VL 235B": "qwen.qwen3-vl-235b-a22b",
            "Qwen3 235B": "qwen.qwen3-235b-a22b-2507-v1:0",
            "Qwen3 Next 80B": "qwen.qwen3-next-80b-a3b",
            "Qwen3 32B": "qwen.qwen3-32b-v1:0",
            "Qwen3 Coder 30B": "qwen.qwen3-coder-30b-a3b-v1:0",
        },
        "DeepSeek": {
            "DeepSeek-R1": "deepseek.r1-v1:0",
            "DeepSeek-V3.1": "deepseek.v3-v1:0",
        },
        "NVIDIA": {
            "Nemotron Nano 12B": "nvidia.nemotron-nano-12b-v2",
            "Nemotron Nano 9B": "nvidia.nemotron-nano-9b-v2",
        },
        "OpenAI": {
            "GPT OSS 120B": "openai.gpt-oss-120b-1:0",
            "GPT OSS 20B": "openai.gpt-oss-20b-1:0",
            "GPT OSS Safeguard 120B": "openai.gpt-oss-safeguard-120b",
            "GPT OSS Safeguard 20B": "openai.gpt-oss-safeguard-20b",
        },
        "Cohere": {
            "Command R+": "cohere.command-r-plus-v1:0",
            "Command R": "cohere.command-r-v1:0",
        },
        "AI21 Labs": {
            "Jamba 1.5 Large": "ai21.jamba-1-5-large-v1:0",
            "Jamba 1.5 Mini": "ai21.jamba-1-5-mini-v1:0",
        },
        "MiniMax": {
            "MiniMax M2": "minimax.minimax-m2",
        },
        "Moonshot AI": {
            "Kimi K2 Thinking": "moonshot.kimi-k2-thinking",
        },
    }
    
    # Provider selection
    provider = st.selectbox(
        "Model Provider",
        options=list(MODEL_CATEGORIES.keys()),
        index=0,  # Amazon as default
        key="side_provider",
        help="Select the AI model provider"
    )
    
    # Model selection based on provider
    models = MODEL_CATEGORIES[provider]
    selected_model = st.selectbox(
        "Model",
        options=list(models.keys()),
        index=0,  # Most advanced model as default
        key="side_model",
        help="Select the specific model"
    )
    
    model_id = models[selected_model]
    
    st.markdown("<h4>RAG Configuration</h4>", unsafe_allow_html=True)
    
    k_chunks = st.slider(
        "Context Chunks",
        min_value=2,
        max_value=10,
        value=6,
        key="side_k_chunks",
        help="Number of document chunks to retrieve for each query"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="side_temperature",
        help="Controls randomness in AI responses"
    )
    
    # Add helpful tips
    st.markdown("""
    <div style="background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); 
                padding: 1rem; border-radius: 10px; margin: 1.5rem 0;
                border-left: 4px solid #146EB4;">
        <h4 style="margin: 0 0 0.5rem 0; color: #232F3E; font-size: 0.85rem;">💡 Pro Tips</h4>
        <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.8rem; color: #4B5563;">
            <li>Use comprehensive documents for best results</li>
            <li>Try different AI models for varied insights</li>
            <li>Export results before closing</li>
            <li>Save scenarios for comparison</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    return model_id, k_chunks

def create_sidebar() -> Tuple[str, int]:
    """Create enhanced sidebar with settings"""
    with st.sidebar:
        
        common.render_sidebar()
               
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **AI-Powered Market Analysis**
            
            This application uses AWS Bedrock and RAG (Retrieval Augmented Generation) to analyze shoe industry market research documents.
            
            **Features:**
            - 📊 Market Analysis & Feasibility
            - 🎯 Product Justification
            - 💡 Strategic Recommendations
            - 📋 Technical Specifications
            - 💬 Interactive Chat Assistant
            
            **How to use:**
            1. Upload your market research document
            2. Click "Process" to analyze
            3. Explore insights across all tabs
            """)
    
    # Return None, None since we're using the panel now
    return None, None

def create_document_upload_section(bedrock_service: BedrockService) -> bool:
    """Create enhanced document upload section"""
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
        <h3 style="color: #232F3E; margin-bottom: 1rem; display: flex; align-items: center;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style="margin-right: 8px;">
                <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" stroke="#FF9900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <polyline points="13 2 13 9 20 9" stroke="#FF9900" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Document Upload
        </h3>
        <p style="color: #6B7280; font-size: 0.9rem; margin-bottom: 1.5rem;">
            Upload your market research documents to begin AI-powered analysis. Supports PDF, DOCX, and TXT formats.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose your market research documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload one or more market research documents for analysis",
        label_visibility="collapsed"
    )

    analyze_button = st.button(
        "🔍 Process",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True
    )
    
    if uploaded_files and analyze_button:
        processor = DocumentProcessor(bedrock_service)
        
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            vectorstore, document_chunks = processor.process_files(uploaded_files)
            
            if vectorstore and document_chunks:
                st.session_state.vectorstore = vectorstore
                st.session_state.documents = document_chunks
                st.session_state.document_processed = True
                
                st.success(f"✅ Successfully processed {len(document_chunks)} chunks from {len(uploaded_files)} document(s)")
                
                # Show document statistics in a modern card
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); 
                            padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                            border-left: 4px solid #1D8102;">
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Documents", len(uploaded_files))
                with col2:
                    st.metric("📊 Total Chunks", len(document_chunks))
                with col3:
                    avg_chunk_size = sum(len(chunk.page_content) for chunk in document_chunks) // len(document_chunks)
                    st.metric("📏 Avg Chunk Size", f"{avg_chunk_size} chars")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Document preview
                with st.expander("📖 Document Preview", expanded=False):
                    for i, chunk in enumerate(document_chunks[:3]):
                        st.markdown(f"**Chunk {i+1}:**")
                        preview_text = chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content
                        st.text(preview_text)
                        if i < 2:
                            st.divider()
            else:
                st.error("❌ Failed to process documents. Please check the file format and try again.")
    
    elif not uploaded_files:
        st.info("👆 Please upload market research documents to begin analysis")
    
    return st.session_state.document_processed

def create_analysis_tab(analyzer: MarketAnalyzer):
    """Create enhanced market analysis tab"""
    st.markdown("## 📊 Market Analysis & Feasibility")
    
    if not st.session_state.document_processed:
        st.info("📄 Using sample data. Upload documents for personalized analysis.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Document", type="primary", use_container_width=True)
    
    with col2:
        if st.session_state.analysis_results:
            if st.button("🔄 Re-analyze", use_container_width=True):
                st.session_state.analysis_results = None
                st.rerun()
    
    if analyze_button:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = [
            ("Extracting opportunities", 20),
            ("Analyzing market insights", 40),
            ("Identifying customer segments", 60),
            ("Evaluating competition", 80),
            ("Calculating feasibility", 100)
        ]
        
        with st.spinner("Performing comprehensive market analysis..."):
            analysis = {}
            
            for step_name, progress in steps:
                status_text.text(f"🔄 {step_name}...")
                progress_bar.progress(progress)
                
                if progress == 20:
                    analysis['opportunities'] = analyzer._extract_opportunities()
                elif progress == 40:
                    analysis['market_insights'] = analyzer._extract_market_insights()
                elif progress == 60:
                    analysis['customer_segments'] = analyzer._extract_customer_segments()
                elif progress == 80:
                    analysis['competitive_landscape'] = analyzer._extract_competitive_landscape()
                else:
                    analysis['feasibility_score'] = analyzer._calculate_feasibility_score()
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
        
        st.session_state.analysis_results = analysis
        st.success("✅ Analysis completed successfully!")
        st.rerun()
    
    if st.session_state.analysis_results:
        analysis = st.session_state.analysis_results
        
        # AI-Generated Insights
        display_automated_insights(analysis)
        
        st.divider()
        
        # Key Metrics Dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            feasibility = analysis['feasibility_score']
            color = "🟢" if feasibility > 70 else "🟡" if feasibility > 50 else "🔴"
            st.metric(
                "Feasibility Score",
                f"{feasibility}%",
                f"{color} {'High' if feasibility > 70 else 'Medium' if feasibility > 50 else 'Low'}"
            )
        
        with col2:
            st.metric(
                "Market Size",
                analysis['market_insights'].get('market_size', 'N/A'),
                analysis['market_insights'].get('growth_rate', '')
            )
        
        with col3:
            st.metric(
                "Opportunities",
                len(analysis['opportunities']),
                "Identified"
            )
        
        with col4:
            st.metric(
                "Target Segments",
                len(analysis['customer_segments']),
                "Defined"
            )
        
        st.divider()
        
        # Detailed Analysis Sections
        tabs = st.tabs(["🚀 Opportunities", "📈 Market Insights", "👥 Customer Segments", "🏆 Competition"])
        
        with tabs[0]:
            st.markdown("### Market Opportunities")
            for i, opportunity in enumerate(analysis['opportunities'], 1):
                # Clean any markdown formatting
                clean_opp = clean_markdown_for_display(opportunity)
                st.markdown(f"""
                <div class="recommendation-item">
                    <strong>{i}.</strong> {clean_opp}
                </div>
                """, unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown("### Key Market Trends")
            for trend in analysis['market_insights'].get('key_trends', []):
                # Clean any markdown formatting
                clean_trend = clean_markdown_for_display(trend)
                st.markdown(f"• {clean_trend}")
            
            # Market metrics visualization - FIXED
            growth_rate_str = analysis['market_insights'].get('growth_rate', '')
            if growth_rate_str and growth_rate_str != "Not specified":
                growth_value = safe_extract_numeric(growth_rate_str, default=5.0)
                if growth_value > 0:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=growth_value,
                        title={'text': "Market Growth Rate (%)"},
                        gauge={'axis': {'range': [None, 20]},
                            'bar': {'color': "#3b82f6"},
                            'steps': [
                                {'range': [0, 5], 'color': "#fee2e2"},
                                {'range': [5, 10], 'color': "#fef3c7"},
                                {'range': [10, 20], 'color': "#d1fae5"}]},
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True, key="chart_market_growth_gauge")
                else:
                    st.info("Growth rate: " + growth_rate_str)
        
        with tabs[2]:
            st.markdown("### Customer Segments Analysis")
            for segment in analysis['customer_segments']:
                # Clean markdown from segment name
                clean_segment_name = clean_markdown_for_display(segment['segment'])
                with st.expander(f"📊 {clean_segment_name}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Market Share", segment.get('size', 'N/A'))
                    with col2:
                        st.markdown("**Characteristics:**")
                        # Clean markdown from characteristics
                        clean_chars = clean_markdown_for_display(segment.get('characteristics', 'Not specified'))
                        st.write(clean_chars)
        
        with tabs[3]:
            st.markdown("### Competitive Landscape")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Major Players:**")
                for player in analysis['competitive_landscape']['major_players'][:5]:
                    # Clean markdown from player names
                    clean_player = clean_markdown_for_display(player)
                    st.markdown(f"• {clean_player}")
            
            with col2:
                st.markdown("**Market Gaps:**")
                for gap in analysis['competitive_landscape']['market_gaps'][:3]:
                    # Clean markdown from gaps
                    clean_gap = clean_markdown_for_display(gap)
                    st.markdown(f"• {clean_gap}")
            
            # Competitive positioning visualization - FIXED
            if analysis['competitive_landscape']['major_players']:
                # Get actual number of players (max 5)
                players = analysis['competitive_landscape']['major_players'][:5]
                num_players = len(players)
                
                # Create names and values arrays with matching lengths
                names = players + ['Our Opportunity']
                
                # Create illustrative market share values
                if num_players > 0:
                    # Distribute market share values proportionally
                    player_values = [20, 18, 15, 12, 10][:num_players]
                    our_value = 100 - sum(player_values)
                    values = player_values + [our_value]
                else:
                    # If no players found, show only our opportunity
                    names = ['Our Opportunity']
                    values = [100]
                
                # Create parents array with matching length
                parents = [''] * len(names)
                
                fig = px.treemap(
                    names=names,
                    parents=parents,
                    values=values,
                    title="Market Share Distribution (Illustrative)"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="chart_market_share_treemap")
        
        st.divider()
        
        # Scenario Comparison
        create_comparison_mode(analysis)
        
        st.divider()
        
        # Export Section
        create_export_section(
            analysis,
            st.session_state.recommendations,
            st.session_state.specifications
        )

def create_justification_tab(analysis: Dict[str, Any]):
    """Create enhanced product justification tab"""
    st.markdown("## 🎯 Product Justification")
    
    if not analysis:
        st.warning("⚠️ Please complete market analysis first")
        return
    
    feasibility = analysis.get('feasibility_score', 70)
    
    # Executive Summary
    st.markdown("### 📋 Executive Summary")
    
    summary_color = "success" if feasibility > 70 else "warning" if feasibility > 50 else "error"
    st.markdown(f"""
    <div class="analysis-card">
        <h4>Market Entry Recommendation: <span style="color: var(--{summary_color}-color);">
        {'Highly Recommended' if feasibility > 70 else 'Proceed with Caution' if feasibility > 50 else 'High Risk'}</span></h4>
        <p>Based on comprehensive market analysis, the feasibility score of <strong>{feasibility}%</strong> indicates 
        {'strong potential for successful market entry' if feasibility > 70 else 'moderate potential with careful positioning' if feasibility > 50 else 'significant challenges requiring careful consideration'}.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Justification Points
    st.markdown("### ✅ Key Justification Points")
    
    justifications = [
        {
            'title': 'Market Demand',
            'score': min(100, feasibility + 10),
            'rationale': f"Identified {len(analysis.get('opportunities', []))} significant market opportunities",
            'evidence': analysis['opportunities'][0] if analysis.get('opportunities') else "Market analysis pending"
        },
        {
            'title': 'Competitive Advantage',
            'score': min(100, feasibility + 5),
            'rationale': f"Found {len(analysis['competitive_landscape']['market_gaps'])} exploitable market gaps",
            'evidence': analysis['competitive_landscape']['market_gaps'][0] if analysis['competitive_landscape']['market_gaps'] else "Analysis required"
        },
        {
            'title': 'Growth Potential',
            'score': min(100, feasibility + 15),
            'rationale': f"Market showing {analysis['market_insights'].get('growth_rate', 'positive')} growth",
            'evidence': f"Market size: {analysis['market_insights'].get('market_size', 'Significant')}"
        },
        {
            'title': 'Target Market Fit',
            'score': feasibility,
            'rationale': f"Identified {len(analysis.get('customer_segments', []))} aligned customer segments",
            'evidence': "Clear segmentation with defined needs and preferences"
        }
    ]
    
    for idx, just in enumerate(justifications):  # Add index to enumerate
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Score visualization - FIX: Use unique key for each chart
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=just['score'],
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#3b82f6" if just['score'] > 70 else "#f59e0b" if just['score'] > 50 else "#ef4444"},
                       'bgcolor': "white",
                       'borderwidth': 2,
                       'bordercolor': "gray"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig.update_layout(height=150, margin=dict(l=20, r=20, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_justification_{idx}")
        
        with col2:
            st.markdown(f"""
            <div class="analysis-card">
                <h4>{just['title']}</h4>
                <p><strong>Score:</strong> {just['score']}%</p>
                <p><strong>Rationale:</strong> {just['rationale']}</p>
                <p><em>Evidence: {just['evidence']}</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Financial Projections
    st.markdown("### 💰 Financial Projections")
    
    # Create realistic projections based on feasibility
    base_factor = feasibility / 100
    years = list(range(1, 6))
    
    revenue_base = [3, 8, 18, 32, 48]
    revenue = [r * base_factor for r in revenue_base]
    
    costs_base = [2.5, 5, 10, 16, 22]
    costs = [c * base_factor for c in costs_base]
    
    profit = [r - c for r, c in zip(revenue, costs)]
    
    # Create financial projection chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years, y=revenue,
        name='Revenue',
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=years, y=costs,
        name='Costs',
        mode='lines+markers',
        line=dict(color='#f59e0b', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=years, y=profit,
        name='Profit',
        mode='lines+markers',
        line=dict(color='#10b981', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    fig.update_layout(
        title='5-Year Financial Projection ($ Millions)',
        xaxis_title='Year',
        yaxis_title='Amount ($ Million)',
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True, key="chart_financial_projection")
    
    # Risk Assessment
    st.markdown("### ⚠️ Risk Assessment")
    
    risks = [
        {"Risk": "Market Competition", "Likelihood": "High", "Impact": "Medium", "Mitigation": "Differentiation through sustainability"},
        {"Risk": "Supply Chain", "Likelihood": "Medium", "Impact": "High", "Mitigation": "Diversified supplier network"},
        {"Risk": "Consumer Adoption", "Likelihood": "Low", "Impact": "High", "Mitigation": "Strong marketing and education"},
        {"Risk": "Economic Downturn", "Likelihood": "Medium", "Impact": "Medium", "Mitigation": "Value pricing strategy"}  # Fixed typo
    ]
    
    risk_df = pd.DataFrame(risks)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

def create_recommendations_tab(recommender: ProductRecommender, analysis: Dict[str, Any]):
    """Create enhanced recommendations tab"""
    st.markdown("## 💡 Strategic Recommendations")
    
    if not analysis:
        st.warning("⚠️ Please complete market analysis first")
        return
    
    if st.button("🎯 Generate Recommendations", type="primary"):
        with st.spinner("Generating strategic recommendations..."):
            recommendations = recommender.generate_recommendations(analysis)
            st.session_state.recommendations = recommendations
        st.success("✅ Recommendations generated!")
        st.rerun()
    
    if st.session_state.recommendations:
        recommendations = st.session_state.recommendations
        
        # Strategy Overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Target Market Strategy")
            for i, segment in enumerate(recommendations['target_segments'][:3], 1):
                # Clean markdown formatting
                clean_segment = clean_markdown_for_display(segment)
                st.markdown(f"""
                <div class="recommendation-item">
                    <strong>Segment {i}:</strong> {clean_segment}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💲 Pricing Strategy")
            st.markdown(f"""
            <div class="specification-item">
                <h4>{recommendations['pricing_strategy'].get('price_range', '$120-150')}</h4>
                <p>{recommendations['pricing_strategy'].get('strategy', 'Premium Value')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Product Positioning
        st.markdown("### 📍 Brand Positioning")
        # Clean markdown from positioning
        clean_positioning = clean_markdown_for_display(recommendations.get('positioning', 'Premium sustainable footwear for the conscious consumer'))
        st.markdown(f"""
        <div class="analysis-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h3 style="color: white;">"{clean_positioning}"</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Product Attributes
        st.markdown("### ⭐ Recommended Product Attributes")
        
        cols = st.columns(2)
        for i, attribute in enumerate(recommendations['key_attributes'][:6]):
            with cols[i % 2]:
                # Clean markdown formatting
                clean_attr = clean_markdown_for_display(attribute)
                st.markdown(f"""
                <div class="recommendation-item">
                    ✓ {clean_attr}
                </div>
                """, unsafe_allow_html=True)
        
        # Competitive Positioning Map
        st.markdown("### 📊 Market Positioning Analysis")
        
        # Create positioning data
        feasibility = analysis.get('feasibility_score', 75)
        
        competitors_data = {
            'Brand': ['Nike', 'Adidas', 'Allbirds', 'Veja', 'Your Product', 'Reebok', 'Puma'],
            'Price_Index': [150, 145, 95, 125, 135, 110, 120],
            'Sustainability_Score': [6.5, 7.0, 9.5, 9.0, 9.2, 5.5, 6.0],
            'Performance_Score': [9.5, 9.3, 6.5, 7.2, 8.5, 7.8, 8.0],
            'Market_Share': [25, 22, 8, 5, 10, 12, 18]
        }
        
        df_competitors = pd.DataFrame(competitors_data)
        
        fig = px.scatter(
            df_competitors,
            x='Sustainability_Score',
            y='Performance_Score',
            size='Market_Share',
            color='Brand',
            hover_data=['Price_Index'],
            title='Competitive Positioning Matrix',
            labels={
                'Sustainability_Score': 'Sustainability Score (0-10)',
                'Performance_Score': 'Performance Score (0-10)',
                'Market_Share': 'Market Share (%)',
                'Price_Index': 'Price Index'
            },
            size_max=50
        )
        
        # Add quadrant lines
        fig.add_hline(y=7.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=7.5, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(x=9, y=9, text="Leaders", showarrow=False, font=dict(size=12, color="gray"))
        fig.add_annotation(x=5, y=9, text="Performance Focus", showarrow=False, font=dict(size=12, color="gray"))
        fig.add_annotation(x=9, y=5, text="Sustainability Focus", showarrow=False, font=dict(size=12, color="gray"))
        fig.add_annotation(x=5, y=5, text="Value Segment", showarrow=False, font=dict(size=12, color="gray"))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, key="chart_competitive_positioning")
        
        # Go-to-Market Strategy
        st.markdown("### 🚀 Go-to-Market Strategy")
        
        gtm_strategies = [
            {
                'Phase': 'Pre-Launch',
                'Duration': '3 months',
                'Key Activities': 'Brand building, influencer partnerships, sustainability certification',
                'Budget Allocation': '20%'
            },
            {
                'Phase': 'Launch',
                'Duration': '1 month',
                'Key Activities': 'Product launch events, PR campaign, digital marketing blitz',
                'Budget Allocation': '40%'
            },
            {
                'Phase': 'Growth',
                'Duration': '6 months',
                'Key Activities': 'Customer acquisition, retail partnerships, community building',
                'Budget Allocation': '30%'
            },
            {
                'Phase': 'Optimization',
                'Duration': 'Ongoing',
                'Key Activities': 'Data-driven optimization, customer retention, product iteration',
                'Budget Allocation': '10%'
            }
        ]
        
        gtm_df = pd.DataFrame(gtm_strategies)
        st.dataframe(gtm_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Click 'Generate Recommendations' to create strategic recommendations based on your analysis.")

def create_specifications_tab(recommender: ProductRecommender, analysis: Dict[str, Any]):
    """Create enhanced specifications tab"""
    st.markdown("## 📋 Product Specifications")
    
    if not analysis:
        st.warning("⚠️ Please complete market analysis first")
        return
    
    if st.button("📐 Generate Specifications", type="primary"):
        with st.spinner("Generating product specifications..."):
            specs = recommender.generate_specifications(analysis)
            st.session_state.specifications = specs
        st.success("✅ Specifications generated!")
        st.rerun()
    
    if st.session_state.specifications:
        specs = st.session_state.specifications
        
        # Product Overview
        st.markdown("### 🎨 Product Concept")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="specification-item">
                <h4>Product Line</h4>
                <h3>{specs['product_concept']['name']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="specification-item">
                <h4>Category</h4>
                <h3>{specs['product_concept']['category']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="specification-item">
                <h4>Price Range</h4>
                <h3>{specs['product_concept']['target_price']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical Specifications Tabs
        spec_tabs = st.tabs(["🧵 Materials", "⚙️ Features", "📊 Performance", "🌱 Sustainability"])
        
        with spec_tabs[0]:
            st.markdown("### Materials & Construction")
            for material in specs['technical_specs']['materials']:
                st.markdown(f"""
                <div class="recommendation-item">
                    • {material}
                </div>
                """, unsafe_allow_html=True)
            
            # Material composition chart
            materials_data = {
                'Material': ['Recycled Plastic', 'Bio-based Foam', 'Natural Rubber', 'Organic Cotton', 'Other'],
                'Percentage': [35, 25, 20, 15, 5]
            }
            
            fig = px.pie(
                pd.DataFrame(materials_data),
                values='Percentage',
                names='Material',
                title='Material Composition',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True, key="chart_material_composition_pie")
        
        with spec_tabs[1]:
            st.markdown("### Technical Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Core Features:**")
                for feature in specs['technical_specs']['features'][:3]:
                    st.markdown(f"• {feature}")
            
            with col2:
                st.markdown("**Additional Features:**")
                for feature in specs['technical_specs']['features'][3:]:
                    st.markdown(f"• {feature}")
            
            st.markdown("**Size & Color Options:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"📏 Sizes: {specs['technical_specs']['sizes']}")
            
            with col2:
                colors_str = ', '.join(specs['technical_specs']['colors'])
                st.info(f"🎨 Colors: {colors_str}")
        
        with spec_tabs[2]:
            st.markdown("### Performance Metrics")
            
            metrics = specs['performance_metrics']
            
            # Create performance radar chart
            categories = list(metrics.keys())
            values = [85, 90, 75, 88, 92]  # Normalized scores for visualization
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=[k.replace('_', ' ').title() for k in categories],
                fill='toself',
                name='Product Performance'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Performance Ratings",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="chart_performance_radar")
            
            # Performance specifications
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Weight", metrics['weight'])
                st.metric("Durability", metrics['durability'])
            
            with col2:
                st.metric("Water Resistance", metrics['water_resistance'])
                st.metric("Energy Return", metrics.get('energy_return', '85%'))
        
        with spec_tabs[3]:
            st.markdown("### Sustainability Metrics")
            
            if 'sustainability_metrics' in specs:
                sustainability = specs['sustainability_metrics']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="specification-item">
                        <h4>♻️ Recycled Content</h4>
                        <h3>{sustainability['recycled_content']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="specification-item">
                        <h4>📦 Packaging</h4>
                        <p>{sustainability['packaging']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="specification-item">
                        <h4>🌍 Carbon Footprint</h4>
                        <h3>{specs['performance_metrics']['carbon_footprint']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="specification-item">
                        <h4>🔄 End-of-Life</h4>
                        <p>{sustainability['end_of_life']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Sustainability comparison chart
            comparison_data = {
                'Metric': ['Carbon Footprint', 'Water Usage', 'Waste Reduction', 'Recyclability'],
                'Our Product': [58, 45, 75, 85],
                'Industry Average': [100, 100, 100, 45]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Our Product', x=df_comparison['Metric'], y=df_comparison['Our Product'],
                                marker_color='#10b981'))
            fig.add_trace(go.Bar(name='Industry Average', x=df_comparison['Metric'], y=df_comparison['Industry Average'],
                                marker_color='#ef4444'))
            
            fig.update_layout(
                title='Sustainability Performance vs Industry Average',
                yaxis_title='Index (Industry Avg = 100)',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="chart_sustainability_comparison_bar")
        
        # Development Timeline - Alternative simpler version
        st.markdown("### 📅 Development Timeline")
        
        timeline_data = {
            'Phase': ['Research & Design', 'Prototyping', 'Testing & Validation', 'Production Setup', 'Launch Preparation'],
            'Duration': ['3 months', '2 months', '2 months', '3 months', '1 month'],
            'Start Month': ['Month 1', 'Month 4', 'Month 6', 'Month 8', 'Month 11'],
            'Status': ['✅ Completed', '🔄 In Progress', '📅 Planned', '📅 Planned', '📅 Planned'],
            'Key Deliverables': [
                'Market analysis, design concepts',
                'Working prototypes, initial feedback',
                'Performance testing, user trials',
                'Manufacturing setup, quality control',
                'Marketing materials, launch event'
            ]
        }
        
        df_timeline = pd.DataFrame(timeline_data)
        
        # Style the dataframe
        st.dataframe(
            df_timeline.style.applymap(
                lambda x: 'color: #10b981' if '✅' in str(x) else 
                         'color: #3b82f6' if '🔄' in str(x) else 
                         'color: #64748b',
                subset=['Status']
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Timeline visualization as a simple bar chart
        months = list(range(1, 12))
        fig = go.Figure()
        
        # Add bars for each phase
        phase_data = [
            {'name': 'Research & Design', 'start': 1, 'duration': 3, 'color': '#10b981'},
            {'name': 'Prototyping', 'start': 4, 'duration': 2, 'color': '#3b82f6'},
            {'name': 'Testing & Validation', 'start': 6, 'duration': 2, 'color': '#94a3b8'},
            {'name': 'Production Setup', 'start': 8, 'duration': 3, 'color': '#94a3b8'},
            {'name': 'Launch Preparation', 'start': 11, 'duration': 1, 'color': '#94a3b8'}
        ]
        
        for phase in phase_data:
            fig.add_trace(go.Scatter(
                x=list(range(phase['start'], phase['start'] + phase['duration'])),
                y=[phase['name']] * phase['duration'],
                mode='lines',
                line=dict(color=phase['color'], width=20),
                name=phase['name'],
                showlegend=False,
                hovertemplate='%{y}<br>Months %{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Development Timeline (12 Months)',
            xaxis_title='Month',
            yaxis_title='Phase',
            height=300,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            yaxis=dict(categoryorder='array', categoryarray=['Research & Design', 'Prototyping', 'Testing & Validation', 'Production Setup', 'Launch Preparation'][::-1])
        )
        
        st.plotly_chart(fig, use_container_width=True, key="chart_development_timeline_gantt")
    
    else:
        st.info("👆 Click 'Generate Specifications' to create detailed product specifications based on your analysis.")

def create_footer():
    """Create compact footer with AWS copyright"""
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <p class="footer-text" style="margin: 0; font-size: 0.875rem; opacity: 0.9;">
                © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_export_section(analysis: Dict[str, Any], recommendations: Dict[str, Any] = None, specs: Dict[str, Any] = None):
    """Add export functionality"""
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        if st.button("📄 Export JSON", use_container_width=True):
            export_data = {
                'analysis': analysis,
                'recommendations': recommendations,
                'specifications': specs,
                'export_date': datetime.now().isoformat()
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        # Export as CSV (simplified data)
        if st.button("📊 Export CSV", use_container_width=True):
            # Create simplified CSV data
            csv_data = []
            
            # Add opportunities
            for i, opp in enumerate(analysis.get('opportunities', []), 1):
                csv_data.append({
                    'Category': 'Opportunity',
                    'Item': i,
                    'Description': opp,
                    'Score': analysis.get('feasibility_score', 0)
                })
            
            # Add segments
            for seg in analysis.get('customer_segments', []):
                csv_data.append({
                    'Category': 'Customer Segment',
                    'Item': seg.get('segment', ''),
                    'Description': seg.get('characteristics', ''),
                    'Score': seg.get('size', '')
                })
            
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_str,
                file_name=f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        # Export summary as text
        if st.button("📝 Export Summary", use_container_width=True):
            summary = f"""MARKET RESEARCH ANALYSIS SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FEASIBILITY SCORE: {analysis.get('feasibility_score', 0)}%

MARKET INSIGHTS:
- Market Size: {analysis.get('market_insights', {}).get('market_size', 'N/A')}
- Growth Rate: {analysis.get('market_insights', {}).get('growth_rate', 'N/A')}

TOP OPPORTUNITIES:
"""
            for i, opp in enumerate(analysis.get('opportunities', [])[:5], 1):
                summary += f"{i}. {opp}\n"
            
            summary += f"\nCUSTOMER SEGMENTS: {len(analysis.get('customer_segments', []))}\n"
            summary += f"MAJOR COMPETITORS: {len(analysis.get('competitive_landscape', {}).get('major_players', []))}\n"
            
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"market_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

def create_chat_assistant(rag_engine: RAGQueryEngine):
    """Add interactive chat for document Q&A"""
    st.markdown("### 💬 Ask Questions About Your Analysis")
    
    # Display chat history
    for message in st.session_state.chat_history:
        # Safety check for message format
        if isinstance(message, dict) and 'role' in message and 'content' in message:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the market research..."):
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                system_prompt = "You are a market research expert. Answer questions based on the provided context. Be concise and specific."
                response = rag_engine.query(prompt, system_prompt, k=4)
                
                if response:
                    st.markdown(response)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                else:
                    error_msg = "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."
                    st.error(error_msg)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': error_msg})
    
    # Clear chat button
    if len(st.session_state.chat_history) > 0:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def create_comparison_mode(analysis: Dict[str, Any]):
    """Allow users to compare multiple analyses"""
    st.markdown("### 🔄 Scenario Comparison")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        scenario_name = st.text_input("Scenario Name", placeholder="e.g., Premium Market Entry")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Scenario", use_container_width=True):
            if analysis and scenario_name:
                st.session_state.comparison_analyses.append({
                    'name': scenario_name,
                    'analysis': analysis,
                    'timestamp': datetime.now()
                })
                st.success(f"✅ Saved: {scenario_name}")
                st.rerun()
    
    # Display comparison
    if len(st.session_state.comparison_analyses) >= 1:
        st.markdown("#### Saved Scenarios")
        
        comparison_data = []
        for idx, scenario in enumerate(st.session_state.comparison_analyses):
            comparison_data.append({
                'Scenario': scenario['name'],
                'Feasibility': f"{scenario['analysis']['feasibility_score']}%",
                'Opportunities': len(scenario['analysis']['opportunities']),
                'Market Size': scenario['analysis']['market_insights'].get('market_size', 'N/A'),
                'Saved': scenario['timestamp'].strftime('%Y-%m-%d %H:%M')
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Side-by-side comparison charts
        if len(st.session_state.comparison_analyses) >= 2:
            fig = go.Figure()
            for scenario in st.session_state.comparison_analyses:
                fig.add_trace(go.Bar(
                    name=scenario['name'],
                    x=['Feasibility', 'Opportunities', 'Segments'],
                    y=[
                        scenario['analysis']['feasibility_score'],
                        len(scenario['analysis']['opportunities']) * 10,
                        len(scenario['analysis']['customer_segments']) * 20
                    ]
                ))
            
            fig.update_layout(
                title='Scenario Comparison',
                barmode='group',
                height=400,
                yaxis_title='Score'
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_scenario_comparison")
        
        # Clear scenarios button
        if st.button("🗑️ Clear All Scenarios"):
            st.session_state.comparison_analyses = []
            st.rerun()

def generate_automated_insights(analysis: Dict[str, Any]) -> List[str]:
    """Generate automated insights from analysis"""
    insights = []
    
    # Feasibility insight
    feasibility = analysis.get('feasibility_score', 0)
    if feasibility > 80:
        insights.append(f"🎯 **Strong Market Opportunity**: With a feasibility score of {feasibility}%, this market shows excellent potential for entry.")
    elif feasibility > 60:
        insights.append(f"⚠️ **Moderate Opportunity**: Feasibility score of {feasibility}% suggests careful positioning is needed.")
    else:
        insights.append(f"🔴 **High Risk**: Feasibility score of {feasibility}% indicates significant challenges.")
    
    # Growth insight
    growth_rate = analysis.get('market_insights', {}).get('growth_rate', '')
    if growth_rate and growth_rate != "Not specified":
        insights.append(f"📈 **Market Growth**: {growth_rate} indicates expanding market opportunities.")
    
    # Competition insight
    num_competitors = len(analysis.get('competitive_landscape', {}).get('major_players', []))
    if num_competitors > 5:
        insights.append(f"🏆 **Competitive Market**: {num_competitors} major players identified - differentiation is critical.")
    elif num_competitors > 0:
        insights.append(f"🏆 **Moderate Competition**: {num_competitors} major players - strategic positioning opportunities exist.")
    
    # Opportunity insight
    num_opportunities = len(analysis.get('opportunities', []))
    if num_opportunities >= 5:
        insights.append(f"💡 **Rich Opportunity Landscape**: {num_opportunities} distinct opportunities identified for market entry.")
    
    # Segment insight
    num_segments = len(analysis.get('customer_segments', []))
    if num_segments >= 3:
        insights.append(f"👥 **Diverse Target Market**: {num_segments} customer segments provide multiple entry points.")
    
    return insights

def display_automated_insights(analysis: Dict[str, Any]):
    """Display automated insights prominently"""
    insights = generate_automated_insights(analysis)
    
    st.markdown("### 🤖 AI-Generated Insights")
    
    for insight in insights:
        st.info(insight)

def main():
    """Main application with modern architecture"""
    
    # Initialize session state variables
    common.initialize_session_state()
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize services
    try:
        if 'bedrock_service' not in st.session_state:
            st.session_state.bedrock_service = BedrockService()
    except Exception as e:
        st.error(f"Failed to initialize AWS Bedrock: {str(e)}")
        st.info("Please ensure your AWS credentials are properly configured.")
        return
    
    # Sidebar configuration
    create_sidebar()
    
    # Main header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 class="main-header">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" style="display: inline-block; vertical-align: middle; margin-right: 12px;">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="url(#gradient1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                <path d="M2 17L12 22L22 17" stroke="url(#gradient1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="url(#gradient1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <defs>
                    <linearGradient id="gradient1" x1="2" y1="2" x2="22" y2="22">
                        <stop offset="0%" style="stop-color:#FF9900;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#146EB4;stop-opacity:1" />
                    </linearGradient>
                </defs>
            </svg>
            Shoe Industry Market Research Analyzer
        </h1>
        <p class="subtitle">
            AI-Powered Market Analysis & Product Development Intelligence
            <br/>
            <span style="font-size: 0.875rem; color: #9CA3AF; margin-top: 0.5rem; display: inline-block;">
                Powered by Amazon Bedrock • Advanced RAG Technology • Real-time Insights
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create 70/30 split layout
    col_main, col_side = st.columns([7, 3])
    
    with col_side:
        # Model selection panel
        with st.container(border=True):
            model_id, k_chunks = create_model_selection_panel()
    
    with col_main:
        # Document upload section
        document_processed = create_document_upload_section(st.session_state.bedrock_service)
        
        # Initialize analysis components
        rag_engine = RAGQueryEngine(
            st.session_state.vectorstore,
            st.session_state.bedrock_service,
            model_id
        )
        
        analyzer = MarketAnalyzer(rag_engine)
        recommender = ProductRecommender(rag_engine)
        
        # Main content tabs
        tab_names = ["📊 Market Analysis", "🎯 Justification", "💡 Recommendations", "📋 Specifications", "💬 Chat Assistant"]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            create_analysis_tab(analyzer)
        
        with tabs[1]:
            if st.session_state.analysis_results:
                create_justification_tab(st.session_state.analysis_results)
            else:
                st.info("📊 Please complete the Market Analysis first to view justifications.")
        
        with tabs[2]:
            if st.session_state.analysis_results:
                create_recommendations_tab(recommender, st.session_state.analysis_results)
            else:
                st.info("📊 Please complete the Market Analysis first to view recommendations.")
        
        with tabs[3]:
            if st.session_state.analysis_results:
                create_specifications_tab(recommender, st.session_state.analysis_results)
            else:
                st.info("📊 Please complete the Market Analysis first to view specifications.")
        
        with tabs[4]:
            if st.session_state.vectorstore or st.session_state.analysis_results:
                create_chat_assistant(rag_engine)
            else:
                st.info("📄 Please upload and process documents first to use the chat assistant.")
    
    # Footer
    create_footer()

# Main execution flow
if __name__ == "__main__":
    if 'localhost' in st.context.headers.get("host", ""):
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()


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

# Updated LangChain imports (v0.1.0+)
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

SessionState.initialize()

# Custom CSS for modern UI/UX
def load_custom_css():
    """Load custom CSS for modern UI/UX design"""
    st.markdown("""
    <style>
    :root {
        --primary-color: #3b82f6;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-color: #1f2937;
        --light-color: #f8fafc;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--dark-color);
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .analysis-card {
        background: var(--light-color);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .analysis-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .recommendation-item {
        background: #ecfdf5;
        border-left: 4px solid var(--success-color);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .specification-item {
        background: #fef7ff;
        border-left: 4px solid var(--secondary-color);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    .footer {
        margin-top: 3rem;
        padding: 2rem 0;
        background-color: var(--dark-color);
        color: white;
        text-align: center;
        font-size: 0.9rem;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0px 0px;
        color: #475569;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Modern button styles */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
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
                    "topP": 0.9,
                    "maxTokens": max_tokens
                }
            )
            
            return response['output']['message']['content'][0]['text']
            
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
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
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
                 model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
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
        query = """What customer segments are mentioned in the document? 
        For each segment, include their characteristics, market size, and preferences if available."""
        system_prompt = "You are a market analyst. Extract detailed customer segment information from the context."
        
        response = self.rag_engine.query(query, system_prompt)
        
        if response:
            segments = self._parse_segments(response)
            if segments:
                return segments[:3]
        
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
                if keyword and keyword.lower() not in clean_line.lower():
                    continue
                items.append(clean_line)
            elif keyword and keyword.lower() in line.lower():
                items.append(line)
        
        return items[:max_items]
    
    def _parse_segments(self, response: str) -> List[Dict[str, str]]:
        """Parse customer segments from response"""
        segments = []
        current_segment = None
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a segment header
            if any(word in line.lower() for word in ['segment', 'demographic', 'customer', 'market']):
                if current_segment:
                    segments.append(current_segment)
                
                current_segment = {
                    'segment': line,
                    'size': 'Not specified',
                    'characteristics': ''
                }
            elif current_segment:
                # Add to characteristics
                if 'size' in line.lower() or '%' in line:
                    size_match = re.search(r'\d+[\-\d]*\s*%', line)
                    if size_match:
                        current_segment['size'] = size_match.group()
                else:
                    current_segment['characteristics'] += line + ' '
        
        if current_segment:
            segments.append(current_segment)
        
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
            recommendations['target_segments'] = analyzer._parse_list_response(
                response, keyword='segment', max_items=3
            )
        
        # Extract attributes
        if 'attribute' in response.lower() or 'feature' in response.lower():
            recommendations['key_attributes'] = analyzer._parse_list_response(
                response, keyword='', max_items=5
            )
        
        # Extract positioning
        position_match = re.search(r'position[^\n]*:\s*([^\n]+)', response, re.IGNORECASE)
        if position_match:
            recommendations['positioning'] = position_match.group(1).strip()
        
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
            
            **How to use:**
            1. Upload your market research document
            2. Click "Analyze Document"
            3. Explore insights across all tabs
            """)
        
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        model_options = {
            "Llama 3": "meta.llama3-70b-instruct-v1:0",
            "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        }
        
        selected_model = st.selectbox(
            "AI Model",
            options=list(model_options.keys()),
            index=0,
            help="Select the AI model for analysis"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            k_chunks = st.slider(
                "Context chunks",
                min_value=2,
                max_value=10,
                value=6,
                help="Number of document chunks to retrieve for each query"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Controls randomness in AI responses"
            )
        
        return model_options[selected_model], k_chunks

def create_document_upload_section(bedrock_service: BedrockService) -> bool:
    """Create enhanced document upload section"""
    st.markdown("### 📄 Document Upload")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose your market research documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload one or more market research documents for analysis"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        analyze_button = st.button(
            "🔍 Process Document",
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
                
                # Show document statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", len(uploaded_files))
                with col2:
                    st.metric("Total Chunks", len(document_chunks))
                with col3:
                    avg_chunk_size = sum(len(chunk.page_content) for chunk in document_chunks) // len(document_chunks)
                    st.metric("Avg Chunk Size", f"{avg_chunk_size} chars")
                
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
    
    if st.button("Analyze Document"):
        with st.spinner("Performing market analysis..."):
            analysis = analyzer.analyze_document()
    
        st.session_state.analysis_results = analysis
        
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
        
        # Detailed Analysis Sections
        tabs = st.tabs(["🚀 Opportunities", "📈 Market Insights", "👥 Customer Segments", "🏆 Competition"])
        
        with tabs[0]:
            st.markdown("### Market Opportunities")
            for i, opportunity in enumerate(analysis['opportunities'], 1):
                st.markdown(f"""
                <div class="recommendation-item">
                    <strong>{i}.</strong> {opportunity}
                </div>
                """, unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown("### Key Market Trends")
            for trend in analysis['market_insights'].get('key_trends', []):
                st.markdown(f"• {trend}")
            
            # Market metrics visualization - FIXED HERE
            growth_rate_str = analysis['market_insights'].get('growth_rate', '')
            if growth_rate_str and growth_rate_str != "Not specified":
                # Try to extract numeric value safely
                growth_match = re.search(r'(\d+(?:\.\d+)?)', growth_rate_str)
                if growth_match:
                    try:
                        growth_value = float(growth_match.group(1))
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
                        st.plotly_chart(fig, use_container_width=True, key="plotly1")
                    except ValueError:
                        st.info("Growth rate data not available in numeric format")
                else:
                    st.info("Growth rate: " + growth_rate_str)
        
        with tabs[2]:
            st.markdown("### Customer Segments Analysis")
            for segment in analysis['customer_segments']:
                with st.expander(f"📊 {segment['segment']}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Market Share", segment.get('size', 'N/A'))
                    with col2:
                        st.markdown("**Characteristics:**")
                        st.write(segment.get('characteristics', 'Not specified'))
        
        with tabs[3]:
            st.markdown("### Competitive Landscape")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Major Players:**")
                for player in analysis['competitive_landscape']['major_players'][:5]:
                    st.markdown(f"• {player}")
            
            with col2:
                st.markdown("**Market Gaps:**")
                for gap in analysis['competitive_landscape']['market_gaps'][:3]:
                    st.markdown(f"• {gap}")
            
            # Competitive positioning visualization - FIXED HERE
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
                st.plotly_chart(fig, use_container_width=True, key="plotly2")

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
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_just_{idx}")  # Unique key
        
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
    
    st.plotly_chart(fig, use_container_width=True, key="plotly_financial")  # Changed key
    
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
    
    with st.spinner("Generating strategic recommendations..."):
        recommendations = recommender.generate_recommendations(analysis)
    
    # Strategy Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Target Market Strategy")
        for i, segment in enumerate(recommendations['target_segments'][:3], 1):
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Segment {i}:</strong> {segment}
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
    st.markdown(f"""
    <div class="analysis-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
        <h3 style="color: white;">"{recommendations.get('positioning', 'Premium sustainable footwear for the conscious consumer')}"</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Product Attributes
    st.markdown("### ⭐ Recommended Product Attributes")
    
    cols = st.columns(2)
    for i, attribute in enumerate(recommendations['key_attributes'][:6]):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="recommendation-item">
                ✓ {attribute}
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
    st.plotly_chart(fig, use_container_width=True, key="plotly7")
    
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

def create_specifications_tab(recommender: ProductRecommender, analysis: Dict[str, Any]):
    """Create enhanced specifications tab"""
    st.markdown("## 📋 Product Specifications")
    
    if not analysis:
        st.warning("⚠️ Please complete market analysis first")
        return
    
    with st.spinner("Generating product specifications..."):
        specs = recommender.generate_specifications(analysis)
    
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
        st.plotly_chart(fig, use_container_width=True, key="plotly8")
    
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
        
        st.plotly_chart(fig, use_container_width=True, key="plotly9")
        
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
        
        st.plotly_chart(fig, use_container_width=True, key="plotly5")
    
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
    
    st.plotly_chart(fig, use_container_width=True, key="plotly4")

def create_footer():
    """Create modern footer"""
    st.markdown("""
    <div class="footer">
        <p>© 2025 Shoe Industry Market Research Analyzer | Powered by AWS Bedrock & LangChain</p>
    </div>
    """, unsafe_allow_html=True)

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
    model_id, k_chunks = create_sidebar()
    
    # Main header
    st.markdown('<h1 class="main-header">👟 Shoe Industry Market Research Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Market Analysis & Product Development Intelligence</p>', unsafe_allow_html=True)
    
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
    tab_names = ["📊 Market Analysis", "🎯 Justification", "💡 Recommendations", "📋 Specifications"]
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
    
    # Footer
    create_footer()

if __name__ == "__main__":
    main()

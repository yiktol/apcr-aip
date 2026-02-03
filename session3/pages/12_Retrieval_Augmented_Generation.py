# Refactored Code with Chroma
import streamlit as st
import logging
import boto3
from botocore.exceptions import ClientError
import os
from pathlib import Path
import uuid
from typing import List, Dict, Any, Optional
import tempfile
import shutil

# Import authentication utility
import utils.authenticate as authenticate

# LangChain imports - Updated to latest versions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_aws import ChatBedrock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CHROMA_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION_NAME = "bedrock_rag"

# Page configuration
st.set_page_config(
    page_title="Amazon Bedrock RAG",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------- SESSION STATE INITIALIZATION -------

def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "conversation_history": [],
        "documents": [],
        "vectorstore": None,
        "document_processed": False,
        "session_id": str(uuid.uuid4()),
        "bedrock_client": None,
        "collection_name": DEFAULT_COLLECTION_NAME,
        "total_chunks": 0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# ------- CUSTOM CSS -------
from utils.styles import load_css

load_css()

# ------- BEDROCK CLIENT MANAGEMENT -------

@st.cache_resource
def get_bedrock_client(region_name: str = 'us-east-1'):
    """
    Create and cache a Bedrock client.
    
    Args:
        region_name: AWS region name
        
    Returns:
        boto3 Bedrock client or None if error
    """
    try:
        client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        logger.info(f"Bedrock client created successfully for region {region_name}")
        return client
    except Exception as e:
        logger.error(f"Error creating Bedrock client: {str(e)}")
        st.error(f"Error creating Bedrock client: {str(e)}")
        return None

# ------- DOCUMENT PROCESSING WITH CHROMA -------

class ChromaDocumentProcessor:
    """Handle document loading, processing, and embedding with Chroma."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.md': UnstructuredMarkdownLoader,
        '.markdown': UnstructuredMarkdownLoader,
    }
    
    def __init__(
        self, 
        bedrock_client, 
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ):
        self.bedrock_client = bedrock_client
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ChromaDocumentProcessor with persist_dir: {persist_directory}")
    
    @classmethod
    def get_loader(cls, file_path: str):
        """
        Get appropriate document loader based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loader instance
            
        Raises:
            ValueError: If file extension not supported
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. "
                f"Supported: {', '.join(cls.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        loader_class = cls.SUPPORTED_EXTENSIONS[file_extension]
        return loader_class(file_path)
    
    def get_embeddings(self):
        """Get Bedrock embeddings instance."""
        return BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v1"
        )
    
    def process_files(
        self, 
        uploaded_files: List, 
        progress_callback: Optional[callable] = None
    ) -> tuple[Optional[Chroma], List[Document]]:
        """
        Process uploaded files into vector embeddings using Chroma.
        
        Args:
            uploaded_files: List of uploaded file objects
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (vectorstore, document_chunks)
        """
        if not uploaded_files:
            logger.warning("No files provided for processing")
            return None, []
        
        all_documents = []
        
        # Use context manager for temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Process each file
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    # Save file temporarily
                    temp_file_path = temp_dir_path / uploaded_file.name
                    temp_file_path.write_bytes(uploaded_file.getvalue())
                    
                    # Load document
                    loader = self.get_loader(str(temp_file_path))
                    documents = loader.load()
                    all_documents.extend(documents)
                    
                    logger.info(f"Loaded {len(documents)} documents from {uploaded_file.name}")
                    
                    # Update progress
                    if progress_callback:
                        progress_callback((idx + 1) / len(uploaded_files))
                
                except Exception as e:
                    logger.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            
            if not all_documents:
                logger.warning("No documents were successfully loaded")
                return None, []
            
            # Split documents into chunks
            document_chunks = self.text_splitter.split_documents(all_documents)
            logger.info(f"Split into {len(document_chunks)} chunks")
            
            # Create embeddings and vector store
            try:
                embeddings = self.get_embeddings()
                
                # Create Chroma vector store with persistence
                vectorstore = Chroma.from_documents(
                    documents=document_chunks,
                    embedding=embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                
                logger.info(
                    f"Chroma vector store created with {len(document_chunks)} documents "
                    f"in collection '{self.collection_name}'"
                )
                
                return vectorstore, document_chunks
            
            except Exception as e:
                logger.error(f"Error creating Chroma vector store: {str(e)}")
                st.error(f"Error creating vector store: {str(e)}")
                return None, []
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing Chroma vector store from disk.
        
        Returns:
            Chroma vectorstore or None if not found
        """
        try:
            # Check if persist directory exists and has data
            persist_path = Path(self.persist_directory)
            if not persist_path.exists() or not any(persist_path.iterdir()):
                logger.info("No existing vector store found")
                return None
            
            embeddings = self.get_embeddings()
            
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_name=self.collection_name
            )
            
            # Try to get collection to verify it exists
            try:
                collection = vectorstore._collection
                count = collection.count()
                logger.info(f"Loaded existing vector store with {count} documents")
                return vectorstore
            except Exception as e:
                logger.warning(f"Vector store exists but collection is empty or invalid: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def delete_collection(self) -> bool:
        """
        Delete the current collection from Chroma.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            embeddings = self.get_embeddings()
            
            # Load the vectorstore
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_name=self.collection_name
            )
            
            # Delete the collection
            vectorstore.delete_collection()
            
            logger.info(f"Deleted collection '{self.collection_name}'")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            vectorstore = self.load_existing_vectorstore()
            if not vectorstore:
                return {"exists": False, "count": 0}
            
            collection = vectorstore._collection
            count = collection.count()
            
            return {
                "exists": True,
                "count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"exists": False, "count": 0, "error": str(e)}

# ------- RAG FUNCTIONS WITH CHROMA -------

class ChromaRAGSystem:
    """Handle RAG operations with Chroma."""
    
    def __init__(self, vectorstore: Optional[Chroma] = None):
        self.vectorstore = vectorstore
    
    def set_vectorstore(self, vectorstore: Chroma):
        """Set the vectorstore."""
        self.vectorstore = vectorstore
    
    def retrieve_context(
        self, 
        query: str, 
        k: int = 4,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Retrieve relevant document chunks.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            logger.warning("No vectorstore available for retrieval")
            return []
        
        try:
            # Use similarity search with optional filtering
            if filter_dict:
                docs = self.vectorstore.similarity_search(
                    query, 
                    k=k,
                    filter=filter_dict
                )
            else:
                docs = self.vectorstore.similarity_search(query, k=k)
            
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_context_with_scores(
        self, 
        query: str, 
        k: int = 4
    ) -> List[tuple[Document, float]]:
        """
        Retrieve relevant document chunks with similarity scores.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            logger.warning("No vectorstore available for retrieval")
            return []
        
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Retrieved {len(docs_with_scores)} documents with scores")
            return docs_with_scores
        
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    @staticmethod
    def format_context(docs: List[Document], include_metadata: bool = False) -> str:
        """
        Format retrieved documents for prompt.
        
        Args:
            docs: List of documents
            include_metadata: Whether to include metadata
            
        Returns:
            Formatted context string
        """
        if not docs:
            return ""
        
        context_parts = ["CONTEXT INFORMATION:", ""]
        
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Document {i}:")
            context_parts.append(doc.page_content)
            
            if include_metadata and doc.metadata:
                context_parts.append(f"Source: {doc.metadata.get('source', 'Unknown')}")
            
            context_parts.append("")
        
        context_parts.append("---")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def format_context_with_scores(
        docs_with_scores: List[tuple[Document, float]]
    ) -> str:
        """
        Format retrieved documents with scores for prompt.
        
        Args:
            docs_with_scores: List of (document, score) tuples
            
        Returns:
            Formatted context string
        """
        if not docs_with_scores:
            return ""
        
        context_parts = ["CONTEXT INFORMATION (with relevance scores):", ""]
        
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            context_parts.append(f"Document {i} (relevance: {score:.4f}):")
            context_parts.append(doc.page_content)
            context_parts.append("")
        
        context_parts.append("---")
        
        return "\n".join(context_parts)

# ------- BEDROCK API FUNCTIONS -------

class BedrockConversation:
    """Handle Bedrock API conversations."""
    
    def __init__(self, bedrock_client, model_id: str):
        self.bedrock_client = bedrock_client
        self.model_id = model_id
    
    def generate_response(
        self,
        messages: List[Dict],
        system_prompts: List[Dict],
        inference_config: Dict,
    ) -> Optional[Dict]:
        """
        Generate a response using Bedrock Converse API.
        
        Args:
            messages: List of conversation messages
            system_prompts: List of system prompts
            inference_config: Inference configuration parameters
            
        Returns:
            Response dictionary or None if error
        """
        try:
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=messages,
                system=system_prompts,
                inferenceConfig=inference_config,
            )
            
            # Log token usage
            usage = response.get('usage', {})
            logger.info(
                f"Token usage - Input: {usage.get('inputTokens', 0)}, "
                f"Output: {usage.get('outputTokens', 0)}, "
                f"Total: {usage.get('totalTokens', 0)}"
            )
            
            return response
        
        except ClientError as e:
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock API error: {error_message}")
            st.error(f"Error: {error_message}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            st.error(f"Unexpected error: {str(e)}")
            return None
    
    def generate_rag_response(
        self,
        query: str,
        context: str,
        system_prompt: str,
        inference_config: Dict,
    ) -> Optional[Dict]:
        """
        Generate a RAG-enhanced response.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: System prompt
            inference_config: Inference configuration
            
        Returns:
            Response dictionary or None if error
        """
        # Create RAG prompt
        rag_prompt = f"""Answer the following question based on the provided context. 
If the context doesn't contain relevant information, acknowledge that and provide a general response if appropriate.

{context}

Question: {query}"""
        
        messages = [{
            "role": "user",
            "content": [{"text": rag_prompt}]
        }]
        
        system_prompts = [{"text": system_prompt}]
        
        return self.generate_response(messages, system_prompts, inference_config)

# ------- UI COMPONENTS -------

def reset_session():
    """Reset session state."""
    st.session_state.conversation_history = []
    st.session_state.documents = []
    st.session_state.vectorstore = None
    st.session_state.document_processed = False
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.total_chunks = 0
    st.success("✅ Session reset successfully!")
    logger.info("Session reset")

def clear_vectorstore():
    """Clear the Chroma vector store."""
    try:
        bedrock_client = get_bedrock_client()
        processor = ChromaDocumentProcessor(
            bedrock_client,
            collection_name=st.session_state.collection_name
        )
        
        if processor.delete_collection():
            st.session_state.vectorstore = None
            st.session_state.documents = []
            st.session_state.document_processed = False
            st.session_state.total_chunks = 0
            st.success("✅ Vector store cleared successfully!")
            logger.info("Vector store cleared")
        else:
            st.error("❌ Failed to clear vector store")
    
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        st.error(f"Error clearing vector store: {str(e)}")

def render_sidebar() -> tuple[str, Dict, int, bool]:
    """
    Render sidebar with model selection and parameters.
    
    Returns:
        Tuple of (model_id, parameters, k_results, show_scores)
    """
    with st.container(border=True):
        st.markdown("<div class='sub-header'>Model Selection</div>", unsafe_allow_html=True)
        
        MODEL_CATEGORIES = {
            "Amazon": [
                "amazon.nova-micro-v1:0",
                "amazon.nova-lite-v1:0",
                "amazon.nova-pro-v1:0"
            ],
            "Anthropic": [
                "anthropic.claude-v2:1",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "anthropic.claude-3-5-sonnet-20241022-v2:0"
            ],
            "Cohere": [
                "cohere.command-r-plus-v1:0",
                "cohere.command-r-v1:0"
            ],
            "Meta": [
                "meta.llama3-70b-instruct-v1:0",
                "meta.llama3-8b-instruct-v1:0",
                "meta.llama3-1-70b-instruct-v1:0",
                "meta.llama3-1-8b-instruct-v1:0"
            ],
            "Mistral": [
                "mistral.mistral-large-2402-v1:0",
                "mistral.mistral-large-2407-v1:0",
                "mistral.mixtral-8x7b-instruct-v0:1",
                "mistral.mistral-7b-instruct-v0:2",
                "mistral.mistral-small-2402-v1:0"
            ],
            "AI21": [
                "ai21.jamba-1-5-large-v1:0",
                "ai21.jamba-1-5-mini-v1:0"
            ]
        }
        
        provider = st.selectbox(
            "Select Provider",
            options=list(MODEL_CATEGORIES.keys()),
            key="provider_select"
        )
        
        model_id = st.selectbox(
            "Select Model",
            options=MODEL_CATEGORIES[provider],
            key="model_select"
        )
        
        st.markdown("<div class='sub-header'>Parameter Tuning</div>", unsafe_allow_html=True)
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic",
            key="temp_slider"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="Controls diversity via nucleus sampling",
            key="topp_slider"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=50,
            max_value=4096,
            value=1024,
            step=50,
            help="Maximum number of tokens in the response",
            key="max_tokens_input"
        )
        
        params = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
        
        st.markdown("<div class='sub-header'>RAG Settings</div>", unsafe_allow_html=True)
        
        k_results = st.number_input(
            "Number of context chunks",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of document chunks to retrieve for context",
            key="k_results_input"
        )
        
        show_scores = st.checkbox(
            "Show relevance scores",
            value=False,
            help="Display similarity scores for retrieved chunks",
            key="show_scores"
        )
    
    # FIXED: Move sidebar content outside the container to avoid nesting
    with st.sidebar:
        st.markdown("<div class='sub-header'>Session Management</div>", unsafe_allow_html=True)
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset", key="reset_session", use_container_width=True):
                reset_session()
        
        with col2:
            if st.button("🗑️ Clear DB", key="clear_db", use_container_width=True):
                clear_vectorstore()
        
        # Display Chroma database status
        st.markdown("---")
        st.markdown("**📊 Database Status**")
        
        bedrock_client = get_bedrock_client()
        processor = ChromaDocumentProcessor(
            bedrock_client,
            collection_name=st.session_state.collection_name
        )
        stats = processor.get_collection_stats()
        
        if stats.get("exists"):
            st.markdown(f"✅ **Active Collection:** `{stats['collection_name']}`")
            st.markdown(f"📄 **Documents:** {stats['count']}")
        else:
            st.markdown("⚪ No active collection")
        
        st.markdown("---")
        
    # FIXED: Moved expanders outside sidebar to avoid nesting issues
    return model_id, params, k_results, show_scores

def render_document_uploader(bedrock_client) -> bool:
    """
    Render document upload interface with Chroma.
    
    Args:
        bedrock_client: Bedrock client instance
        
    Returns:
        Boolean indicating if documents are processed
    """
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>📄 Document Upload</div>", unsafe_allow_html=True)
    
    # Check if existing vectorstore can be loaded
    processor = ChromaDocumentProcessor(
        bedrock_client,
        collection_name=st.session_state.collection_name
    )
    
    existing_vectorstore = processor.load_existing_vectorstore()
    
    if existing_vectorstore and not st.session_state.document_processed:
        st.markdown(
            "<div class='info-box'>"
            "📌 <strong>Existing database found!</strong> "
            "You can use the existing knowledge base or upload new documents."
            "</div>",
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📚 Load Existing Database", key="load_existing", use_container_width=True):
                st.session_state.vectorstore = existing_vectorstore
                st.session_state.document_processed = True
                stats = processor.get_collection_stats()
                st.session_state.total_chunks = stats.get('count', 0)
                st.success(f"✅ Loaded existing database with {st.session_state.total_chunks} documents!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete & Start Fresh", key="delete_existing", use_container_width=True):
                processor.delete_collection()
                st.session_state.vectorstore = None
                st.session_state.document_processed = False
                st.success("✅ Database cleared! Upload new documents below.")
                st.rerun()
    
    st.markdown("""
    Upload documents to build your knowledge base.
    
    **Supported formats:** PDF, Word, Text, CSV, Markdown
    """)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "md"],
        key="file_uploader"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        process_button = st.button(
            "🚀 Process Documents",
            type="primary",
            key="process_docs",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.document_processed:
            st.success("✅ Ready")
    
    if process_button and uploaded_files:
        progress_bar = st.progress(0, text="Processing documents...")
        
        vectorstore, document_chunks = processor.process_files(
            uploaded_files,
            progress_callback=lambda p: progress_bar.progress(p, text=f"Processing... {int(p*100)}%")
        )
        
        if vectorstore and document_chunks:
            st.session_state.vectorstore = vectorstore
            st.session_state.documents = document_chunks
            st.session_state.document_processed = True
            st.session_state.total_chunks = len(document_chunks)
            
            progress_bar.progress(1.0, text="✅ Processing complete!")
            
            # Display statistics
            st.markdown(
                "<div class='success-box'>"
                f"<strong>Successfully processed {len(uploaded_files)} files "
                f"into {len(document_chunks)} chunks!</strong>"
                "</div>",
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📁 Files", len(uploaded_files))
            col2.metric("📄 Chunks", len(document_chunks))
            col3.metric("💾 Stored in Chroma", len(document_chunks))
            
            # Show sample chunks
            with st.expander("🔍 Preview Document Chunks", expanded=False):
                for i, chunk in enumerate(document_chunks[:3]):
                    st.markdown(f"**Chunk {i+1}:**")
                    preview_text = chunk.page_content[:300]
                    if len(chunk.page_content) > 300:
                        preview_text += "..."
                    st.text(preview_text)
                    
                    # Show metadata if available
                    if chunk.metadata:
                        st.caption(f"Source: {chunk.metadata.get('source', 'Unknown')}")
                    
                    st.divider()
                
                if len(document_chunks) > 3:
                    st.caption(f"... and {len(document_chunks) - 3} more chunks")
        else:
            st.error("❌ Failed to process documents. Please check the logs.")
    
    elif process_button:
        st.warning("⚠️ Please upload files first.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return st.session_state.document_processed

def render_conversation_history():
    """Render conversation history."""
    if not st.session_state.conversation_history:
        return
    
    st.markdown("### 💬 Conversation History")
    
    # Show most recent conversations first (limit to last 10)
    for entry in reversed(st.session_state.conversation_history[-10:]):
        if entry["role"] == "user":
            st.markdown(
                f"<div class='user-message'>"
                f"<strong>You:</strong> {entry['content']}"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='response-block'>"
                f"<strong>Assistant:</strong> {entry['content']}"
                f"</div>",
                unsafe_allow_html=True
            )

def render_rag_interface(model_id: str, params: Dict, k_results: int, show_scores: bool):
    """
    Render RAG conversation interface with Chroma.
    
    Args:
        model_id: Selected model ID
        params: Model parameters
        k_results: Number of context chunks to retrieve
        show_scores: Whether to show relevance scores
    """
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant that answers questions based on the provided context. "
              "Use the context to provide accurate and relevant answers. "
              "If the context doesn't contain enough information, acknowledge this limitation.",
        height=100,
        key="rag_system_prompt"
    )
    
    user_prompt = st.text_area(
        "Your Question",
        value="",
        height=120,
        placeholder="Ask a question about your documents...",
        key="rag_user_prompt"
    )
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        submit = st.button(
            "🔍 Submit Question",
            type="primary",
            key="rag_submit",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.conversation_history:
            if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
                st.session_state.conversation_history = []
                st.rerun()
    
    # Display document status
    if st.session_state.document_processed:
        st.markdown(
            "<div class='success-box'>"
            f"✅ <strong>Knowledge base ready</strong> with {st.session_state.total_chunks} chunks in Chroma"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='warning-box'>"
            "⚠️ <strong>No documents processed yet.</strong> Please upload and process documents first, "
            "or load an existing database."
            "</div>",
            unsafe_allow_html=True
        )
    
    # Display conversation history
    render_conversation_history()
    
    if submit:
        if not user_prompt.strip():
            st.warning("⚠️ Please enter a question.")
            return
        
        if not st.session_state.document_processed:
            st.error("❌ Please process documents before asking questions.")
            return
        
        # Add to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })
        
        with st.status("Processing your question...") as status:
            # Retrieve context
            status.update(label="🔍 Retrieving relevant context from Chroma...")
            rag_system = ChromaRAGSystem(st.session_state.vectorstore)
            
            if show_scores:
                relevant_docs_with_scores = rag_system.retrieve_context_with_scores(
                    user_prompt, 
                    k=k_results
                )
                context_str = rag_system.format_context_with_scores(relevant_docs_with_scores)
                relevant_docs = [doc for doc, score in relevant_docs_with_scores]
            else:
                relevant_docs = rag_system.retrieve_context(user_prompt, k=k_results)
                context_str = rag_system.format_context(relevant_docs, include_metadata=True)
            
            if not relevant_docs:
                st.error("❌ Could not retrieve relevant context from Chroma.")
                return
            
            # Generate response
            status.update(label="🤖 Generating response...")
            bedrock_client = get_bedrock_client()
            conversation = BedrockConversation(bedrock_client, model_id)
            
            response = conversation.generate_rag_response(
                query=user_prompt,
                context=context_str,
                system_prompt=system_prompt,
                inference_config=params
            )
            
            if response:
                status.update(label="✅ Response received!", state="complete")
                
                output_message = response['output']['message']
                response_text = output_message['content'][0]['text']
                
                # Add to history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                # Display metrics
                usage = response['usage']
                col1, col2, col3 = st.columns(3)
                col1.metric("📥 Input Tokens", usage['inputTokens'])
                col2.metric("📤 Output Tokens", usage['outputTokens'])
                col3.metric("📊 Total Tokens", usage['totalTokens'])
                
                # Show context sources
                # with st.expander("📚 View Retrieved Context", expanded=False):
                #     if show_scores:
                #         for i, (doc, score) in enumerate(relevant_docs_with_scores, 1):
                #             st.markdown(f"**Chunk {i}** (Relevance: {score:.4f})")
                #             st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                #             if doc.metadata:
                #                 st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                #             st.divider()
                #     else:
                #         for i, doc in enumerate(relevant_docs, 1):
                #             st.markdown(f"**Chunk {i}**")
                #             st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                #             if doc.metadata:
                #                 st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                #             st.divider()
                
                st.rerun()
            else:
                status.update(label="❌ Error occurred", state="error")

def render_normal_conversation_interface(model_id: str, params: Dict):
    """
    Render normal conversation interface.
    
    Args:
        model_id: Selected model ID
        params: Model parameters
    """
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, respectful, and honest assistant. "
              "Always answer as helpfully as possible while being safe.",
        height=100,
        key="normal_system_prompt"
    )
    
    user_prompt = st.text_area(
        "Your Message",
        value="",
        height=120,
        placeholder="Enter your message here...",
        key="normal_user_prompt"
    )
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        submit = st.button(
            "💬 Send Message",
            type="primary",
            key="normal_submit",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.conversation_history:
            if st.button("🗑️ Clear Chat", key="clear_normal_chat", use_container_width=True):
                st.session_state.conversation_history = []
                st.rerun()
    
    # Display conversation history
    render_conversation_history()
    
    if submit:
        if not user_prompt.strip():
            st.warning("⚠️ Please enter a message.")
            return
        
        # Add to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })
        
        with st.status("Generating response...") as status:
            bedrock_client = get_bedrock_client()
            conversation = BedrockConversation(bedrock_client, model_id)
            
            system_prompts = [{"text": system_prompt}]
            messages = [{
                "role": "user",
                "content": [{"text": user_prompt}]
            }]
            
            response = conversation.generate_response(
                messages=messages,
                system_prompts=system_prompts,
                inference_config=params
            )
            
            if response:
                status.update(label="✅ Response received!", state="complete")
                
                output_message = response['output']['message']
                response_text = output_message['content'][0]['text']
                
                # Add to history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                # Display metrics
                usage = response['usage']
                col1, col2, col3 = st.columns(3)
                col1.metric("📥 Input Tokens", usage['inputTokens'])
                col2.metric("📤 Output Tokens", usage['outputTokens'])
                col3.metric("📊 Total Tokens", usage['totalTokens'])
                
                st.rerun()
            else:
                status.update(label="❌ Error occurred", state="error")

# ------- MAIN APP -------

def main():
    """Main application entry point."""
    # Header
    st.markdown(
        "<h1 class='main-header'>🪨 Amazon Bedrock RAG with Chroma</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown("""<div class="info-box">
    Retrieval Augmented Generation (RAG) with Amazon Bedrock & Chroma DB. Upload documents, process them into a vector database, 
    and ask questions to get answers grounded in your data using foundation models. Documents persist between sessions!
    </div>""", unsafe_allow_html=True)
    
    # Get Bedrock client
    bedrock_client = get_bedrock_client()
    
    if not bedrock_client:
        st.error("❌ Failed to initialize Bedrock client. Please check your AWS credentials.")
        st.info("💡 Make sure you have configured AWS credentials with access to Amazon Bedrock.")
        return
    
    # Layout: main content (70%) and sidebar (30%)
    col1, col2 = st.columns([0.7, 0.3])
    
    # Sidebar with model selection
    with col2:
        model_id, params, k_results, show_scores = render_sidebar()
    
    # Main content area
    with col1:
        # Document uploader
        has_documents = render_document_uploader(bedrock_client)
        
        # Tabs for different modes
        tab1, tab2 = st.tabs(["🔍 RAG Conversation", "💬 Regular Conversation"])
        
        with tab1:
            render_rag_interface(model_id, params, k_results, show_scores)
        
        with tab2:
            render_normal_conversation_interface(model_id, params)
    
    # Footer
    st.markdown(
        '<div class="footer">'
        '© 2025 Amazon Web Services, Inc. or its affiliates. All rights reserved. | '
        'Powered by Amazon Bedrock & Chroma'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Check if running locally or with authentication
    try:
        host = st.context.headers.get("host", "localhost")
        if 'localhost' in host or '127.0.0.1' in host:
            main()
        else:
            # Authentication required for non-local deployment
            is_authenticated = authenticate.login()
            if is_authenticated:
                main()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

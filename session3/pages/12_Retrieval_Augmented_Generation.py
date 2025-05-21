
import streamlit as st
import logging
import boto3
from botocore.exceptions import ClientError
import os
from io import BytesIO
from PIL import Image
import uuid
import time
import json
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.llms.bedrock import Bedrock

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Page configuration with improved styling
st.set_page_config(
    page_title="Amazon Bedrock RAG",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Apply custom CSS for modern appearance with AWS color scheme
st.markdown("""
    <style>
    .stApp {
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #232F3E; /* AWS dark blue */
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #FF9900; /* AWS orange */
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .output-container {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #FF9900; /* AWS orange */
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #EC7211; /* AWS orange darker */
    }
    .response-block {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9900; /* AWS orange */
        margin-top: 1rem;
    }
    .user-message {
        background-color: #E6F7FF; /* Light blue */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #232F3E; /* AWS dark blue */
        margin-top: 1rem;
    }
    .token-metrics {
        display: flex;
        justify-content: space-between;
        background-color: #F0F4F8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
    .metric-item {
        text-align: center;
    }
    .metric-value {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #4B5563;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: #f8f9fa;
        font-size: 12px;
        color: #666;
    }
    .file-loader {
        border: 2px dashed #FF9900;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
    }
    .progress-bar {
        height: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .progress {
        height: 100%;
        background-color: #FF9900;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ------- RAG FUNCTIONS -------

def get_bedrock_client():
    """Create a Bedrock client."""
    try:
        bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
        return bedrock_client
    except Exception as e:
        st.error(f"Error creating Bedrock client: {str(e)}")
        return None

def determine_loader(file_path: str):
    """Determine the appropriate document loader based on file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension == '.docx':
        return Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path)
    elif file_extension == '.csv':
        return CSVLoader(file_path)
    elif file_extension in ['.md', '.markdown']:
        return UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def process_documents(files, progress_callback=None):
    """Process uploaded documents into chunks and create vector embeddings."""
    if not files:
        return None, []
        
    all_documents = []
    temp_files = []
    
    try:
        # Save uploaded files temporarily and load them
        for idx, file in enumerate(files):
            with st.spinner(f"Processing {file.name}..."):
                # Save the file temporarily
                temp_file_path = f"temp_{file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(file.getvalue())
                temp_files.append(temp_file_path)
                
                # Load the document
                try:
                    loader = determine_loader(temp_file_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    
                    if progress_callback:
                        progress_callback((idx + 1) / len(files))
                        
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")
        
        if not all_documents:
            return None, []
            
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents into chunks
        document_chunks = text_splitter.split_documents(all_documents)
        
        # Initialize AWS Bedrock embeddings
        bedrock_client = get_bedrock_client()
        embeddings = BedrockEmbeddings(
            client=bedrock_client,
            model_id="amazon.titan-embed-text-v1"
        )
        
        # Create vector store
        with st.spinner("Creating vector embeddings..."):
            vectorstore = FAISS.from_documents(document_chunks, embeddings)
            
        return vectorstore, document_chunks
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None, []
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def retrieve_relevant_chunks(vectorstore, query, k=4):
    """Retrieve relevant document chunks based on the query."""
    if vectorstore is None:
        return []
        
    try:
        # Search for similar documents
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def format_context_for_prompt(docs: List[Document]) -> str:
    """Format retrieved documents for inclusion in the prompt."""
    if not docs:
        return ""
        
    context_str = "\n\n---\nCONTEXT INFORMATION:\n"
    for i, doc in enumerate(docs):
        context_str += f"Document {i+1}:\n{doc.page_content}\n\n"
    context_str += "---\n"
    
    return context_str

# ------- API FUNCTIONS -------

def rag_conversation(bedrock_client, model_id, system_prompt, query, context, **params):
    """Sends a RAG-enhanced conversation to the model."""
    logger.info(f"Generating RAG response with model {model_id}")

    # Combine context with the user query to create a RAG prompt
    rag_prompt = f"""Answer the following question based on the provided context. 
If the context doesn't contain relevant information, just say that you don't have enough information to answer properly.

{context}

Question: {query}"""

    # Set up the messages for the conversation
    system_prompts = [{"text": system_prompt}]
    message = {
        "role": "user",
        "content": [{"text": rag_prompt}]
    }
    messages = [message]
    
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=params,
            additionalModelRequestFields={}
        )
        
        # Log token usage
        token_usage = response['usage']
        logger.info(f"Input tokens: {token_usage['inputTokens']}")
        logger.info(f"Output tokens: {token_usage['outputTokens']}")
        logger.info(f"Total tokens: {token_usage['totalTokens']}")
        logger.info(f"Stop reason: {response['stopReason']}")
        
        return response
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return None

def text_conversation(bedrock_client, model_id, system_prompts, messages, **params):
    """Sends messages to a model without RAG."""
    logger.info(f"Generating message with model {model_id}")
    
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=params,
            additionalModelRequestFields={}
        )
        
        # Log token usage
        token_usage = response['usage']
        logger.info(f"Input tokens: {token_usage['inputTokens']}")
        logger.info(f"Output tokens: {token_usage['outputTokens']}")
        logger.info(f"Total tokens: {token_usage['totalTokens']}")
        logger.info(f"Stop reason: {response['stopReason']}")
        
        return response
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return None

def stream_conversation(bedrock_client, model_id, messages, system_prompts, inference_config):
    """Sends messages to a model and streams the response."""
    logger.info(f"Streaming messages with model {model_id}")
    
    try:
        response = bedrock_client.converse_stream(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields={}
        )
        
        stream = response.get('stream')
        if stream:
            placeholder = st.empty()
            full_response = ''
            token_info = {'input': 0, 'output': 0, 'total': 0}
            latency_ms = 0
            
            for event in stream:
                if 'messageStart' in event:
                    role = event['messageStart']['role']
                    with placeholder.container():
                        st.markdown(f"**{role.title()}**")
                
                if 'contentBlockDelta' in event:
                    chunk = event['contentBlockDelta']
                    part = chunk['delta']['text']
                    full_response += part
                    with placeholder.container():
                        st.markdown(f"**Response:**\n{full_response}")
                
                if 'messageStop' in event:
                    stop_reason = event['messageStop']['stopReason']
                    with placeholder.container():
                        st.markdown(f"**Response:**\n{full_response}")
                        st.caption(f"Stop reason: {stop_reason}")
                
                if 'metadata' in event:
                    metadata = event['metadata']
                    if 'usage' in metadata:
                        usage = metadata['usage']
                        token_info = {
                            'input': usage['inputTokens'],
                            'output': usage['outputTokens'],
                            'total': usage['totalTokens']
                        }
                    
                    if 'metrics' in event.get('metadata', {}):
                        latency_ms = metadata['metrics']['latencyMs']
            
            # Display token usage after streaming is complete
            st.markdown("### Response Details")
            col1, col2, col3 = st.columns(3)
            col1.metric("Input Tokens", token_info['input'])
            col2.metric("Output Tokens", token_info['output'])
            col3.metric("Total Tokens", token_info['total'])
            st.caption(f"Latency: {latency_ms}ms")
        
        return True
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return False

# ------- UI COMPONENTS -------

def reset_session():
    """Reset session state."""
    st.session_state.conversation_history = []
    st.session_state.documents = []
    st.session_state.vectorstore = None
    st.session_state.document_processed = False
    st.session_state.session_id = str(uuid.uuid4())
    st.success("Session reset successfully!")

def parameter_sidebar():
    """Sidebar with model selection and parameter tuning."""
    with st.container(border=True):
           
        st.markdown("<div class='sub-header'>Model Selection</div>", unsafe_allow_html=True)
        
        MODEL_CATEGORIES = {
            "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"],
            "Anthropic": ["anthropic.claude-v2:1", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"],
            "Cohere": ["cohere.command-text-v14:0", "cohere.command-r-plus-v1:0", "cohere.command-r-v1:0"],
            "Meta": ["meta.llama3-70b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0"],
            "Mistral": ["mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", 
                       "mistral.mistral-7b-instruct-v0:2", "mistral.mistral-small-2402-v1:0"],
            "AI21": ["ai21.jamba-1-5-large-v1:0", "ai21.jamba-1-5-mini-v1:0"]
        }
        
        # Create selectbox for provider first
        provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()))
        
        # Then create selectbox for models from that provider
        model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider])
        
        st.markdown("<div class='sub-header'>Parameter Tuning</div>", unsafe_allow_html=True)
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, 
                            help="Higher values make output more random, lower values more deterministic")
        
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1,
                            help="Controls diversity via nucleus sampling")
        
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4096, value=1024, step=50,
                                    help="Maximum number of tokens in the response")
            
        params = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
        
        st.markdown("<div class='sub-header'>RAG Settings</div>", unsafe_allow_html=True)
                # RAG settings
        k_results = st.number_input(
            "Number of context chunks", 
            min_value=1, 
            max_value=10, 
            value=4,
            help="Number of document chunks to retrieve for context"
        )

    with st.sidebar:        
        
        st.markdown("<div class='sub-header'>Session Management</div>", unsafe_allow_html=True)
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}`")
        
        if st.button("Reset Session", key="reset_session", help="Clear conversation history and uploaded documents"):
            reset_session()
        
        with st.expander("About this App", expanded=False):
            st.markdown("""
            This app demonstrates Amazon Bedrock's Converse API with RAG capabilities.
            
            You can:
            - Upload documents to build a knowledge base
            - Ask questions about your documents
            - Get responses grounded in your data
            
            For more information, visit the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/).
            """)
        
        
        
        
    return model_id, params,k_results

def document_uploader():
    """Interface for document uploading and processing."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Document Upload</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload documents to build your knowledge base. Supported formats:
    - PDF (.pdf)
    - Word Documents (.docx)
    - Text (.txt)
    - CSV (.csv)
    - Markdown (.md)
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload one or more documents", 
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "md"]
    )
    
    # Process button
    process_button = st.button("Process Documents", type="primary", key="process_docs")
    
    progress_placeholder = st.empty()
    
    if process_button and uploaded_files:
        progress_bar = progress_placeholder.progress(0)
        
        # Process the documents
        vectorstore, document_chunks = process_documents(
            uploaded_files,
            progress_callback=lambda p: progress_bar.progress(p)
        )
        
        if vectorstore and document_chunks:
            st.session_state.vectorstore = vectorstore
            st.session_state.documents = document_chunks
            st.session_state.document_processed = True
            progress_placeholder.success(f"✅ Processed {len(uploaded_files)} files into {len(document_chunks)} chunks!")
            
            # Show document stats
            st.markdown("### Document Statistics")
            col1, col2 = st.columns(2)
            col1.metric("Files Processed", len(uploaded_files))
            col2.metric("Document Chunks", len(document_chunks))
            
            # Show sample chunks
            with st.expander("Sample Document Chunks", expanded=False):
                for i, chunk in enumerate(document_chunks[:3]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
                    st.markdown("---")
                if len(document_chunks) > 3:
                    st.caption(f"... and {len(document_chunks) - 3} more chunks")
        else:
            progress_placeholder.error("❌ Failed to process documents. Please try again.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return st.session_state.document_processed

def rag_interface(model_id, params, k_results):
    """Interface for RAG-enabled conversation."""
  

    system_prompt = st.text_area(
            "System Prompt", 
            value="You are a helpful assistant that answers questions based on the provided context. Use the context to provide accurate and relevant answers.", 
            height=100
        )


    user_prompt = st.text_area(
        "Your Question", 
        value="", 
        height=120,
        placeholder="Ask a question about your documents..."
    )
    
    submit = st.button("Submit Question", type="primary", key="rag_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display document status
    if st.session_state.document_processed:
        st.success(f"✅ Knowledge base ready with {len(st.session_state.documents)} document chunks")
    else:
        st.warning("⚠️ No documents processed yet. Please upload and process documents first.")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("### Conversation History")
        # Reverse the conversation history to show the latest at the top
        for entry in reversed(st.session_state.conversation_history):
            if entry["role"] == "user":
                st.markdown(f"<div class='user-message'><strong>You:</strong> {entry['content']}</div>", unsafe_allow_html=True)
            else:
                # Make the response expanded by default
                st.markdown(f"<div class='response-block'><strong>Assistant:</strong> {entry['content']}</div>", unsafe_allow_html=True)
    
    if submit:
        if not user_prompt.strip():
            st.warning("Please enter a question first.")
            return
            
        if not st.session_state.document_processed:
            st.error("Please upload and process documents before asking questions.")
            return
            
        # Add user question to conversation history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })
            
        with st.status("Processing your question...") as status:
            # Step 1: Retrieve relevant document chunks
            status.update(label="Retrieving relevant context...")
            relevant_docs = retrieve_relevant_chunks(
                st.session_state.vectorstore, 
                user_prompt,
                k=k_results
            )
            
            if not relevant_docs:
                st.error("Could not retrieve relevant context from documents.")
                return
                
            # Step 2: Format context for the prompt
            context_str = format_context_for_prompt(relevant_docs)
            
            # Step 3: Send RAG request to the model
            try:
                status.update(label="Generating response...")
                bedrock_client = get_bedrock_client()
                
                response = rag_conversation(
                    bedrock_client, 
                    model_id, 
                    system_prompt, 
                    user_prompt, 
                    context_str, 
                    **params
                )
                
                if response:
                    status.update(label="Response received!", state="complete")
                    
                    # Get the model's response
                    output_message = response['output']['message']
                    response_text = output_message['content'][0]['text']
                    
                    # Add response to conversation history
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    # Display token usage
                    token_usage = response['usage']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Input Tokens", token_usage['inputTokens'])
                    col2.metric("Output Tokens", token_usage['outputTokens'])
                    col3.metric("Total Tokens", token_usage['totalTokens'])
                    
                    # Show context sources in an expander
                    # with st.expander("View Context Sources", expanded=False):
                    #     for i, doc in enumerate(relevant_docs):
                    #         st.markdown(f"**Document Chunk {i+1}:**")
                    #         st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    #         st.markdown("---")
                    
                    # Force a rerun to refresh the conversation history display
                    st.rerun()
            
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")

def normal_conversation_interface(model_id, params):
    """Interface for regular conversation without RAG."""
    
    system_prompt = st.text_area(
            "System Prompt", 
            value="You are a helpful assistant that provides detailed and thoughtful responses.", 
            height=100,
            key="normal_system"
        )

    user_prompt = st.text_area(
        "Your Question", 
        value="", 
        height=120,
        placeholder="Enter your question here...",
        key="normal_input"
    )
    
    submit = st.button("Generate Response", type="primary", key="normal_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("### Conversation History")
        # Reverse the conversation history to show the latest at the top
        for entry in reversed(st.session_state.conversation_history):
            if entry["role"] == "user":
                st.markdown(f"<div class='user-message'><strong>You:</strong> {entry['content']}</div>", unsafe_allow_html=True)
            else:
                # Make the response expanded by default
                st.markdown(f"<div class='response-block'><strong>Assistant:</strong> {entry['content']}</div>", unsafe_allow_html=True)
    
    if submit:
        if not user_prompt.strip():
            st.warning("Please enter a question first.")
            return
            
        # Add user question to conversation history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })
            
        with st.status("Generating response...") as status:
            # Setup the system prompts and messages
            system_prompts = [{"text": system_prompt}]
            message = {
                "role": "user",
                "content": [{"text": user_prompt}]
            }
            messages = [message]
            
            try:
                # Get Bedrock client
                bedrock_client = get_bedrock_client()
                
                # Send request to the model
                response = text_conversation(bedrock_client, model_id, system_prompts, messages, **params)
                
                if response:
                    status.update(label="Response received!", state="complete")
                    
                    # Get the model's response
                    output_message = response['output']['message']
                    response_text = output_message['content'][0]['text']
                    
                    # Add response to conversation history
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    # Display token usage
                    token_usage = response['usage']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Input Tokens", token_usage['inputTokens'])
                    col2.metric("Output Tokens", token_usage['outputTokens'])
                    col3.metric("Total Tokens", token_usage['totalTokens'])
                    
                    # Force a rerun to refresh the conversation history display
                    st.rerun()
            
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")

# ------- MAIN APP -------

def main():
    # Header
    st.markdown("<h1 class='main-header'>Amazon Bedrock RAG</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This interactive dashboard demonstrates Retrieval Augmented Generation (RAG) with Amazon Bedrock.
    Upload your documents, process them into embeddings, and ask questions to get answers grounded in your data.
    """)


    # Create a 70/30 layout
    col1, col2 = st.columns([0.7, 0.3])     
        # Get model and parameters from the right column
    with col2:
        model_id, params, k_results = parameter_sidebar()  

    with col1:
        # Document processor section
        has_documents = document_uploader()
        
        # Create tabs for different interaction modes
        tabs = st.tabs([
            "🔍 RAG Conversation", 
            "💬 Regular Conversation"
        ])
        
        # Populate each tab
        with tabs[0]:
            rag_interface(model_id, params, k_results)
        
        with tabs[1]:
            normal_conversation_interface(model_id, params)
    
    # Add footer
    st.markdown('<div class="footer">© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

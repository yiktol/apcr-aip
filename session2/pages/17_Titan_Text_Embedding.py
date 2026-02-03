
"""
Titan Text Embedding Application

This Streamlit application demonstrates Amazon Bedrock's Titan Text Embedding capabilities.
It allows users to input text and visualizes the resulting embedding vectors.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError
import utils.authenticate as authenticate
import utils.common as common
from utils.styles import load_css, custom_header

st.set_page_config(
    page_title="Titan Text Embedding",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("titan_embeddings_app")

# Constants
MODEL_ID = "amazon.titan-embed-g1-text-02"
ACCEPT = "application/json"
CONTENT_TYPE = "application/json"
VECTOR_DIMENSION = 1536
DISPLAY_ROWS = 128
DISPLAY_COLS = 12

def initialize_bedrock_client() -> boto3.client:
    """
    Initialize and return the Amazon Bedrock Runtime client.
    
    Returns:
        boto3.client: Configured Bedrock runtime client
    """
    try:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        logger.info("Bedrock client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock client: {e}")
        raise

def get_embedding(client: boto3.client, text: str) -> Tuple[List[float], Dict]:
    """
    Generate embeddings for the provided text using Titan Embedding model.
    
    Args:
        client: Bedrock runtime client
        text: Input text to vectorize
        
    Returns:
        Tuple containing embedding vector and full response
    """
    if not text:
        logger.warning("Empty text provided for embedding")
        raise ValueError("Text input cannot be empty")
        
    try:
        body = json.dumps({
            "inputText": text,
        })
        
        logger.info(f"Requesting embedding for text of length {len(text)}")
        response = client.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept=ACCEPT,
            contentType=CONTENT_TYPE
        )
        
        response_body = json.loads(response["body"].read())
        embedding = response_body.get("embedding")
        
        if not embedding:
            logger.error("No embedding found in response")
            raise ValueError("Model did not return an embedding")
            
        logger.info(f"Successfully generated embedding of length {len(embedding)}")
        return embedding, response_body
        
    except ClientError as e:
        logger.error(f"AWS service error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

def display_embedding(embedding: List[float]) -> None:
    """
    Display the embedding vector in a structured format.
    
    Args:
        embedding: The embedding vector to display
    """
    # Reshape for visualization (show subset in grid format)
    display_size = min(len(embedding), DISPLAY_ROWS * DISPLAY_COLS)
    reshaped_array = np.array(embedding[:display_size]).reshape(DISPLAY_ROWS, DISPLAY_COLS)
    
    df = pd.DataFrame(
        reshaped_array, 
        columns=[f"Dimension {i+1}" for i in range(DISPLAY_COLS)]
    )
    
    st.markdown("### Embedding Vector Visualization")
    st.markdown(f"*Showing {DISPLAY_ROWS}×{DISPLAY_COLS} of {len(embedding)} dimensions*")
    
    # Use a color gradient to visualize values
    st.dataframe(
        df.style.background_gradient(cmap="viridis"), 
        use_container_width=True,
        height=500
    )
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Vector Dimensions", len(embedding))
    with col2:
        st.metric("Min Value", f"{min(embedding):.6f}")
    with col3:
        st.metric("Max Value", f"{max(embedding):.6f}")
    with col4:
        st.metric("Mean Value", f"{np.mean(embedding):.6f}")

def display_additional_info() -> None:
    """Display model and embedding information in the sidebar."""
    with st.sidebar:
        
        common.render_sidebar()
        with st.expander("ℹ️ About Titan Embeddings", expanded=False):
            st.markdown("""
            **Model**: Titan Embeddings G1 - Text v1.2

            **Key Features**:
            - 1,536 dimensional vectors
            - Support for 25+ languages
            - Handles up to 8,192 tokens
            - Optimized for text retrieval and semantic similarity

            **Use Cases**:
            - Semantic search
            - Document retrieval
            - Text clustering
            - Similarity detection
            - Classification tasks
            """)
        
    

def main() -> None:
    """Main application function."""
    
    load_css()
    
    try:
        bedrock_client = initialize_bedrock_client()
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        st.stop()
    
    # Main UI
    st.markdown("""
        <h1>🔠 Titan Text Embedding</h1>
        <div class="info-box">
        The Titan Embeddings G1 - Text v1.2 model converts text into numerical vectors that capture semantic meaning. These embeddings enable powerful natural language processing capabilities like semantic search, clustering, and similarity detection across 25+ languages.
        </div>
        """, unsafe_allow_html=True)
    
    
    # Input form with improved styling
    with st.form(key="embedding_form"):
        prompt_data = st.text_area(
            "Enter text to vectorize",
            placeholder="Write me a poem about apples...",
            value="A puppy is to dog as kitten is to cat.",
            height=150,
            help="Text will be converted to a 1536-dimensional vector"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submitted = st.form_submit_button(
                "Generate Embedding", 
                type="primary",
                use_container_width=True
            )
    
    # Process input when submitted
    if submitted:
        if not prompt_data.strip():
            st.warning("Please enter some text to vectorize")
            return
            
        try:
            with st.status("Generating embedding...") as status:
                embedding, _ = get_embedding(bedrock_client, prompt_data)
                status.update(label="Processing complete!", state="complete")
                
            display_embedding(embedding)
            
            # Add option to download the embedding
            embedding_json = json.dumps({"text": prompt_data, "embedding": embedding})
            st.download_button(
                label="Download Embedding JSON",
                data=embedding_json,
                file_name="titan_embedding.json",
                mime="application/json"
            )
            
        except ValueError as e:
            st.error(f"Input Error: {str(e)}")
        except ClientError as e:
            st.error(f"AWS Service Error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.exception("Unhandled exception")
    
    # Display additional information
    display_additional_info()
    
    # Footer
    st.markdown("---")
    st.markdown("<div class='footer'>© 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    try:

        if 'localhost' in st.context.headers["host"]:
            main()
        else:
            # First check authentication
            is_authenticated = authenticate.login()
            
            # If authenticated, show the main app content
            if is_authenticated:
                main()

    except Exception as e:
        logger.critical(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
        
        # Provide debugging information in an expander
        with st.expander("Error Details"):
            st.code(str(e))
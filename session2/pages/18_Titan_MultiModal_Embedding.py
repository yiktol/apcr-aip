
"""
Multimodal Embedding Explorer - A Streamlit application for exploring Amazon Titan embeddings.

This application demonstrates image search and similarity-based recommendations using
Amazon Bedrock's Titan multimodal embedding models. Users can generate embeddings for 
product images and perform semantic searches.
"""

import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import boto3
import jsonlines
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.spatial.distance import cdist
from stqdm import stqdm
import utils.authenticate as authenticate
import utils.common as common

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_page_config():
    """Configure the Streamlit page settings and styles"""
    # Page configuration
    st.set_page_config(
        page_title="Multimodal Embedding Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .st-emotion-cache-16txtl3 h1 {
            font-weight: 700;
            color: #1E88E5;
        }
        .result-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client():
    """
    Create and cache the Amazon Bedrock client.
    
    Returns:
        boto3.client: Configured Bedrock runtime client
    """
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock client: {e}")
        raise RuntimeError(f"Could not connect to Amazon Bedrock: {e}")

def search(query_emb: np.ndarray, indexes: np.ndarray, top_k: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Search for similar embeddings using cosine distance.
    
    Args:
        query_emb: Query embedding array
        indexes: Index embedding arrays to search against
        top_k: Number of top results to return
        
    Returns:
        Tuple containing (indices of top matches, sorted distances, distance matrix)
    """
    try:
        dist = cdist(query_emb, indexes, metric="cosine")
        return dist.argsort(axis=-1)[0, :top_k], np.sort(dist, axis=-1)[:top_k], dist
    except Exception as e:
        logger.error(f"Error during embedding search: {e}")
        raise RuntimeError(f"Failed to perform search: {e}")

def multimodal_search(description: str, multimodal_embeddings: np.ndarray, top_k: int, dimension: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a multimodal search using text description against image embeddings.
    
    Args:
        description: Text description to search for
        multimodal_embeddings: Array of image embeddings to search against
        top_k: Number of top results to return
        dimension: Embedding dimension size
        
    Returns:
        Tuple containing (indices of top matches, distance matrix)
    """
    logger.info(f"Performing multimodal search for: '{description}'")
    try:
        query_emb = get_embedding(description=description, dimension=dimension)["embedding"]
        idx_returned, dist, df = search(
            np.array(query_emb)[None], 
            np.array(multimodal_embeddings),
            top_k=top_k,
        )
        return idx_returned, df
    except Exception as e:
        logger.error(f"Multimodal search failed: {e}")
        raise RuntimeError(f"Search operation failed: {e}")

def get_embedding(
    image_path: Optional[str] = None,
    description: Optional[str] = None, 
    dimension: int = 1024,
    model_id: str = "amazon.titan-embed-image-v1"
) -> Dict:
    """
    Generate embeddings using Amazon Titan model.
    
    Args:
        image_path: Path to the image file (optional)
        description: Text description (optional, max 128 tokens)
        dimension: Embedding dimension (1024, 384, or 256)
        model_id: Bedrock model ID to use
        
    Returns:
        Dictionary containing the embedding response
    """
    bedrock_runtime = get_bedrock_client()
    payload_body = {}
    embedding_config = {
        "embeddingConfig": { 
             "outputEmbeddingLength": dimension
         }
    }

    # You can specify either text or image or both
    if image_path:
        try:
            with open(image_path, "rb") as image_file:
                input_image = base64.b64encode(image_file.read()).decode('utf8')
            payload_body["inputImage"] = input_image
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            raise ValueError(f"Cannot read image file: {e}")
            
    if description:
        payload_body["inputText"] = description

    if not payload_body:
        raise ValueError("Please provide either an image path and/or a text description")
    
    try:
        response = bedrock_runtime.invoke_model(
            body=json.dumps({**payload_body, **embedding_config}), 
            modelId=model_id,
            accept="application/json", 
            contentType="application/json"
        )
        return json.loads(response.get("body").read())
    except Exception as e:
        logger.error(f"Bedrock API call failed: {e}")
        raise RuntimeError(f"Failed to generate embedding: {e}")
        
def reset_session():
    """Clear all session state variables"""
    logger.info("Resetting session state")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
        
@st.cache_data
def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionary objects
    """
    logger.info(f"Loading dataset from {file_path}")
    try:
        data = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                data.append(obj)
        logger.info(f"Loaded {len(data)} items from dataset")
        return data
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise FileNotFoundError(f"Could not load dataset: {e}")

def format_product_name(file_name: str) -> str:
    """
    Format a product name from its file path.
    
    Args:
        file_name: Image file path
        
    Returns:
        Formatted product name
    """
    item = Path(file_name).stem.lower()
    item = item.replace('_', ' ')
    return item[0].upper() + item[1:]

def display_embeddings(embedding: List[float]) -> pd.DataFrame:
    """
    Convert a 1D embedding array into a 2D visualization dataframe.
    
    Args:
        embedding: 1D embedding array
        
    Returns:
        DataFrame representation of the embedding
    """
    numbers = np.array(embedding).reshape(32, 32)
    return pd.DataFrame(numbers, columns=(f"col {i}" for i in range(32)))

def generate_embeddings(dataset: List[Dict]) -> None:
    """
    Generate embeddings for all products in the dataset.
    
    Args:
        dataset: List of product dictionaries
    """
    try:
        multimodal_embeddings = []
        
        # Use stqdm for a progress bar that works in Streamlit
        for item in stqdm(dataset, desc="Generating embeddings"):
            embedding = get_embedding(image_path=item['file_name'], dimension=1024)["embedding"]
            multimodal_embeddings.append(embedding)
            
        st.session_state.multimodal_embeddings = multimodal_embeddings
        st.session_state.is_multimodal_embeddings = True
        
        logger.info(f"Successfully generated {len(multimodal_embeddings)} embeddings")
        st.success("✅ Embeddings generated successfully! You can now search for products.")
    
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        st.error(f"Error generating embeddings: {str(e)}")

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "multimodal_embeddings" not in st.session_state:
        st.session_state.multimodal_embeddings = []
        
    if "distance" not in st.session_state:
        st.session_state.distance = np.array([])
        
    if "is_multimodal_embeddings" not in st.session_state:
        st.session_state.is_multimodal_embeddings = False

def create_sidebar():
    """Create and populate the sidebar"""
    with st.sidebar:
        common.render_sidebar()
        
        with st.expander("About this App", expanded=False):
            st.markdown("""
            ### Topics Covered
            
            - **Multimodal Embeddings**: Learn how Amazon Titan models generate embeddings from images and text
            - **Vector Search**: Understand similarity-based retrieval using cosine distances
            - **Product Discovery**: Use natural language to find visually similar products
            - **Recommendation Systems**: See how embedding-based recommendations work
            - **Visualization**: Explore the structure of high-dimensional embedding vectors
            """)

def main():
    """Main application function"""
    
    initialize_session_state()
    create_sidebar()

    # Main content
    st.title("🔍 Multimodal Embedding Explorer")

    try:
        # Load dataset
        dataset = load_jsonl('data/metadata.jsonl')
        products = [format_product_name(item['file_name']) for item in dataset]
        
        # Create layout
        left_col, right_col = st.columns([0.6, 0.4])
        
        with left_col:
            st.markdown("""
            Amazon Titan Multimodal Embedding Models can be used for enterprise tasks such as image search and similarity-based recommendations.
            
            - Generate embeddings for images and text
            - Find similar products through semantic searching  
            - Visualize embedding vectors
            """)
            
            # Search form
            with st.form("search_form", border=False):
                st.subheader("🔎 Search Products")
                prompt_data = st.text_area(
                    "Enter a text description:", 
                    value="suede sneaker",
                    help="Describe the product you're looking for"
                )
                col1, col2 = st.columns([3, 1])
                with col1:
                    k = st.slider("Number of results", min_value=1, max_value=min(10, len(dataset)), value=3)
                with col2:
                    submit = st.form_submit_button("Search", type="primary", use_container_width=True)
        
            # Handle search
            if submit:
                if not st.session_state.is_multimodal_embeddings:
                    st.warning("⚠️ No embeddings available. Please generate embeddings first.")
                else:
                    try:
                        with st.status("Searching...") as status:
                            text_embedding = get_embedding(description=prompt_data)["embedding"]
                            
                            idx_returned, distance = multimodal_search(
                                description=prompt_data,
                                multimodal_embeddings=st.session_state.multimodal_embeddings,
                                top_k=k
                            )
                            status.update(label="Search complete!", state="complete")
                        
                        # Display text embedding
                        with st.expander("View text embedding vectors"):
                            st.dataframe(
                                display_embeddings(text_embedding),
                                use_container_width=True,
                                height=400
                            )
                        
                        # Display results
                        st.subheader("📋 Search Results")
                        
                        # Create a dictionary for the distance table
                        distance_array = np.array(distance).tolist()
                        distance_dict = {
                            "Product": products,
                            "Similarity Score": [1 - dist for dist in distance_array[0]],
                            "Cosine Distance": distance_array[0]
                        }
                        
                        # Display results in a grid
                        result_columns = st.columns(min(3, k))
                        
                        for i, idx in enumerate(idx_returned[:]):
                            col_idx = i % len(result_columns)
                            with result_columns[col_idx]:
                                st.markdown(f"""
                                <div class="result-card">
                                    <h3>{products[idx]}</h3>
                                    <p>Similarity: {(1 - distance_array[0][idx]):.4f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.image(
                                    Image.open(dataset[idx]['file_name']),
                                    use_container_width=True
                                )
                        
                        # Display distance metrics
                        with st.expander("View similarity metrics for all products"):
                            st.dataframe(
                                pd.DataFrame(distance_dict).sort_values(by="Similarity Score", ascending=False),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                    except Exception as e:
                        logger.error(f"Search operation failed: {e}")
                        st.error(f"Search failed: {str(e)}")
                    
        with right_col:
            with st.container(border=True):
                st.subheader("🖼️ Product Explorer")
                
                option = st.selectbox("Select a product:", products)
                idx = products.index(option)
                
                try:
                    st.image(
                        dataset[idx]['file_name'],
                        caption=dataset[idx].get('description', 'No description available'),
                        use_container_width=True
                    )
                    
                    if st.session_state.is_multimodal_embeddings:
                        with st.expander("View image embedding vectors"):
                            product_embedding = st.session_state.multimodal_embeddings[idx]
                            st.dataframe(
                                display_embeddings(product_embedding),
                                use_container_width=True,
                                height=400
                            )
                except Exception as e:
                    logger.error(f"Failed to display product: {e}")
                    st.error(f"Could not display product: {str(e)}")
                
                # Controls
                col1, col2 = st.columns(2)
                
                with col1:
                    gen_button = st.button(
                        "Generate Embeddings", 
                        type="primary",
                        disabled=st.session_state.is_multimodal_embeddings,
                        help="Create embeddings for all products"
                    )
                
                with col2:
                    clear_button = st.button(
                        "Clear Embeddings", 
                        type="secondary",
                        on_click=reset_session,
                        help="Reset all embeddings"
                    )
                
                # Generate embeddings
                if gen_button:
                    if st.session_state.is_multimodal_embeddings:
                        st.info("Embeddings have already been generated.")
                    else:
                        generate_embeddings(dataset)
                        st.rerun()

        # Footer
        st.divider()
        st.caption("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")

    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.info("Please ensure the data/metadata.jsonl file exists in the application directory.")
    except Exception as e:
        logger.critical(f"Application error: {e}")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please check the logs for more information.")

# Main execution flow
if __name__ == "__main__":
    setup_page_config()
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()

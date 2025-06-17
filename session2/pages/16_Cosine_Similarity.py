
"""
Titan Text Embedding Explorer - A Streamlit application for exploring text embeddings
through cosine similarity analysis using Amazon Bedrock's Titan embedding model.

This application allows users to compare a prompt text with multiple other texts
to visualize their semantic similarities in vector space.
"""

import streamlit as st
import pandas as pd
import numpy as np
import boto3
import json
import logging
import plotly.express as px
from numpy import dot
from numpy.linalg import norm
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Union, Any
import utils.authenticate as authenticate
import utils.common as common
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Titan Text Embedding Explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #FF5733;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #3366FF;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .info-text {
        font-size: 1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Create a cache for the Bedrock client
@st.cache_resource
def get_bedrock_client() -> boto3.client:
    """
    Create and return a cached Bedrock client.
    
    Returns:
        boto3.client: Bedrock runtime client
    """
    try:
        return boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
    except Exception as e:
        logger.error(f"Failed to create Bedrock client: {e}")
        st.error(f"Failed to connect to AWS Bedrock: {e}")
        return None

def get_embedding(bedrock_client: boto3.client, text: str) -> List[float]:
    """
    Generate text embeddings using Amazon Titan embedding model.
    
    Args:
        bedrock_client: Boto3 client for Bedrock runtime
        text: Input text to embed
        
    Returns:
        List[float]: Vector embedding of the input text
    """
    if not text:
        logger.warning("Empty text provided for embedding")
        return []
        
    try:
        model_id = 'amazon.titan-embed-g1-text-02'
        accept = 'application/json'
        content_type = 'application/json'
        
        body = json.dumps({
            'inputText': text
        })
        
        response = bedrock_client.invoke_model(
            body=body, 
            modelId=model_id, 
            accept=accept,
            contentType=content_type
        )
        
        response_body = json.loads(response.get('body').read())
        embedding = response_body.get('embedding', [])
        
        logger.info(f"Generated embedding for text: '{text[:30]}...' (if longer)")
        return embedding
    
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        st.error(f"Failed to generate embedding: {str(e)}")
        return []

def calculate_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    if not v1 or not v2:
        return 0.0
        
    try:
        return float(dot(v1, v2) / (norm(v1) * norm(v2)))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def generate_2d_projection(embeddings_df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """
    Generate a 2D projection of the embeddings using PCA.
    
    Args:
        embeddings_df: DataFrame containing the embeddings
        
    Returns:
        Tuple containing x and y coordinates for the 2D projection
    """
    try:
        pca = PCA(n_components=2, svd_solver='auto')
        pca_result = pca.fit_transform(embeddings_df.values)
        return list(pca_result[:, 0]), list(pca_result[:, 1])
    except Exception as e:
        logger.error(f"Error in PCA projection: {e}")
        st.error(f"Failed to generate PCA projection: {str(e)}")
        return [], []

def render_cosine_similarity_explanation():
    """Render the explanation section for cosine similarity."""
    st.markdown("""
    <div class="main-header">Cosine Similarity</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.2, 0.2, 0.6])
    
    try:
        col1.image("images/dot-product-a-cos.svg", caption="Vector representation")
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        col1.error("Image not found.")
    
    col2.latex(r'\cos \theta = \frac{a \cdot b}{|a| \times |b|}')
    
    st.markdown("""
    <div class="card info-text">
    Cosine similarity measures how similar two pieces of text are likely to be in terms of their semantic meaning.
    It calculates the cosine of the angle between the two embedding vectors, and ranges from -1 to 1:
    <ul>
        <li>Value of 1: Vectors are identical in direction (maximum similarity)</li>
        <li>Value of 0: Vectors are orthogonal (no similarity)</li>
        <li>Value of -1: Vectors are opposite (maximum dissimilarity)</li>
    </ul>
    
    This metric is independent of text length and focuses purely on the direction of vectors in the semantic space.
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit application."""
    with st.sidebar:
        common.render_sidebar()
        
    render_cosine_similarity_explanation()
    
    # Initialize Bedrock client
    bedrock_client = get_bedrock_client()
    if not bedrock_client:
        st.stop()
    
    # Set up layout
    text_col, viz_col = st.columns([0.4, 0.6])
    
    with text_col:
        st.markdown('<div class="sub-header">Text Input</div>', unsafe_allow_html=True)
        
        with st.form("embedding_form"):
            prompt = st.text_area("📌 Enter your prompt here:", 
                                 height=70, 
                                 value="Hello", 
                                 help="This is the reference text for comparison")
            
            sample_texts = {
                f"text{i+1}": st.text_area(
                    f'Text {i+1}', 
                    value=default_text, 
                    height=70,
                    key=f"text_input_{i}"
                ) for i, default_text in enumerate([
                    "Hi", 
                    "Good Day", 
                    "How are you", 
                    "Good Morning", 
                    "Goodbye"
                ])
            }
            
            submit = st.form_submit_button(
                "✨ Calculate Similarities", 
                type="primary",
                use_container_width=True
            )
    
    # Initialize dataframes
    similarity_df = None
    vector_space_df = None
    
    if submit and prompt:
        with st.spinner("🧠 Processing text embeddings..."):
            try:
                # Get prompt embedding
                prompt_embedding = get_embedding(bedrock_client, prompt)
                
                # Process all text samples
                texts = []
                embeddings = []
                similarities = []
                
                # Process prompt as reference
                texts.append(prompt)
                embeddings.append(prompt_embedding)
                
                # Process comparison texts
                for text_key, text_value in sample_texts.items():
                    if text_value:
                        # Get embedding for the text
                        embedding = get_embedding(bedrock_client, text_value)
                        
                        # Calculate similarity with the prompt
                        similarity = calculate_cosine_similarity(prompt_embedding, embedding) if embedding else 0.0
                        
                        # Store results
                        texts.append(text_value)
                        embeddings.append(embedding)
                        similarities.append({
                            'Prompt': prompt,
                            'Text': text_value,
                            'Similarity': round(similarity, 4)
                        })
                
                # Create dataframes
                similarity_df = pd.DataFrame(similarities)
                
                # Create dataframe for vector visualization
                if embeddings:
                    # Create embeddings dataframe
                    embedding_df = pd.DataFrame({'Text': texts, 'Embeddings': embeddings})
                    
                    # Make sure all embeddings have the same length
                    valid_embeddings = [e for e in embedding_df['Embeddings'] if len(e) > 0]
                    valid_texts = [t for i, t in enumerate(embedding_df['Text']) 
                                if i < len(embedding_df['Embeddings']) and 
                                len(embedding_df['Embeddings'][i]) > 0]
                    
                    # Create a proper DataFrame with embeddings as rows
                    vector_df = pd.DataFrame(valid_embeddings, index=valid_texts)
                    
                    vector_space_df = vector_df
                    
            except Exception as e:
                logger.error(f"Error processing embeddings: {e}")
                st.error(f"An error occurred: {str(e)}")
    
    # Display vector space visualization if data is available
    if vector_space_df is not None:
        with viz_col:
            st.markdown('<div class="sub-header">Vector Space Visualization</div>', unsafe_allow_html=True)
            
            with st.container(border=True):
                x_coords, y_coords = generate_2d_projection(vector_space_df)
                
                if x_coords and y_coords:
                    # Create a dataframe for plotting
                    plot_df = pd.DataFrame({
                        'x': x_coords,
                        'y': y_coords,
                        'text': vector_space_df.index
                    })
                    
                    # Create a color map
                    plot_df['is_prompt'] = plot_df['text'] == prompt
                    plot_df['point_size'] = plot_df['is_prompt'].apply(lambda x: 2 if x else 1)
                    # Create scatter plot
                    fig = px.scatter(
                        plot_df, 
                        x='x', 
                        y='y',
                        color='text',
                        hover_name='text',
                        size='point_size',
                        size_max=15,
                        title="2D Projection of Text Embeddings"
                    )
                    
                    fig.update_layout(
                        height=500,
                        legend_title_text='Texts',
                        legend={'itemsizing': 'constant'},
                        xaxis_title="Component 1",
                        yaxis_title="Component 2",
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display embeddings table with toggle
                    with st.expander("Show Raw Embeddings"):
                        st.dataframe(vector_space_df, use_container_width=True)
    
    # Display similarity results
    if similarity_df is not None:
        st.markdown('<div class="sub-header">Similarity Results</div>', unsafe_allow_html=True)
        
        # Sort by similarity (descending)
        similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)
        
        # Add color gradient based on similarity score
        def color_similarity(val):
            # Create a color gradient from red (0) to green (1)
            r = int(255 * (1 - val))
            g = int(255 * val)
            return f'background-color: rgba({r}, {g}, 0, 0.2)'
        
        st.dataframe(
            similarity_df.style.map(
                color_similarity, 
                subset=['Similarity']
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prompt": st.column_config.TextColumn("Prompt", width="medium"),
                "Text": st.column_config.TextColumn("Text", width="medium"),
                "Similarity": st.column_config.ProgressColumn(
                    "Similarity Score", 
                    format="%.4f",
                    min_value=0,
                    max_value=1
                )
            }
        )
        
        # Show a bar chart of similarities
        st.markdown('<div class="sub-header">Similarity Comparison</div>', unsafe_allow_html=True)
        chart = px.bar(
            similarity_df,
            x='Text',
            y='Similarity',
            color='Similarity',
            color_continuous_scale='Viridis',
            title='Cosine Similarity Scores'
        )
        st.plotly_chart(chart, use_container_width=True)

if __name__ == "__main__":
    try:
        is_authenticated = authenticate.login()
        if is_authenticated:
            main()
    except Exception as e:
        logger.critical(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
        
        # Provide debugging information in an expander
        with st.expander("Error Details"):
            st.code(str(e))

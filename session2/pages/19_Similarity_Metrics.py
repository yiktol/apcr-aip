
"""
Titan Text Embedding Explorer - A Streamlit application for exploring text embeddings
through various similarity metrics using Amazon Bedrock's Titan embedding model.

This application allows users to compare a prompt text with multiple other texts
to visualize their semantic similarities using different distance/similarity metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import boto3
import json
import logging
import plotly.express as px
import math
from numpy import dot
from numpy.linalg import norm
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Union, Any
import utils.authenticate as authenticate
import utils.common as common
from utils.styles import load_css
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

load_css()

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
        float: Cosine similarity score between -1 and 1
    """
    if not v1 or not v2:
        return 0.0
        
    try:
        return float(dot(v1, v2) / (norm(v1) * norm(v2)))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def calculate_euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """
    Calculate the Euclidean distance between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: Euclidean distance
    """
    if not v1 or not v2:
        return float('inf')
        
    try:
        return float(math.dist(v1, v2))
    except Exception as e:
        logger.error(f"Error calculating Euclidean distance: {e}")
        return float('inf')

def calculate_dot_product(v1: List[float], v2: List[float]) -> float:
    """
    Calculate the dot product between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: Dot product value
    """
    if not v1 or not v2:
        return 0.0
        
    try:
        return float(dot(v1, v2))
    except Exception as e:
        logger.error(f"Error calculating dot product: {e}")
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
        col1.info("💡 Vector visualization")
    
    col2.latex(r'\cos \theta = \frac{a \cdot b}{|a| \times |b|}')
    
    st.markdown("""
    <div class="card info-text">
    Cosine similarity measures the cosine of the angle between two vectors in multi-dimensional space.
    It ranges from -1 to 1:
    <ul>
        <li><strong>1:</strong> Vectors point in the same direction (maximum similarity)</li>
        <li><strong>0:</strong> Vectors are perpendicular (no similarity)</li>
        <li><strong>-1:</strong> Vectors point in opposite directions (maximum dissimilarity)</li>
    </ul>
    
    This metric is normalized and independent of vector magnitude, making it ideal for text similarity.
    </div>
    """, unsafe_allow_html=True)

def render_euclidean_distance_explanation():
    """Render the explanation section for Euclidean distance."""
    st.markdown("""
    <div class="main-header">Euclidean Distance</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.3, 0.7])
    
    col1.latex(r'd = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}')
    
    st.markdown("""
    <div class="card info-text">
    Euclidean distance measures the straight-line distance between two points in multi-dimensional space.
    <ul>
        <li><strong>0:</strong> Vectors are identical (maximum similarity)</li>
        <li><strong>Higher values:</strong> Vectors are more different (lower similarity)</li>
        <li><strong>Range:</strong> 0 to infinity</li>
    </ul>
    
    Unlike cosine similarity, Euclidean distance considers both direction and magnitude of vectors.
    For text embeddings, smaller distances indicate more similar semantic meanings.
    </div>
    """, unsafe_allow_html=True)

def render_dot_product_explanation():
    """Render the explanation section for dot product."""
    st.markdown("""
    <div class="main-header">Dot Product</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.3, 0.7])
    
    col1.latex(r'a \cdot b = \sum_{i=1}^{n} a_i \times b_i')
    
    st.markdown("""
    <div class="card info-text">
    Dot product measures both the magnitude and directional similarity between vectors.
    <ul>
        <li><strong>Positive values:</strong> Vectors point in similar directions</li>
        <li><strong>Zero:</strong> Vectors are perpendicular</li>
        <li><strong>Negative values:</strong> Vectors point in opposite directions</li>
        <li><strong>Range:</strong> -infinity to +infinity</li>
    </ul>
    
    The dot product is related to cosine similarity but isn't normalized, so it's affected by vector magnitude.
    Higher absolute values indicate stronger relationships (positive or negative).
    </div>
    """, unsafe_allow_html=True)

def process_embeddings(bedrock_client, prompt, sample_texts):
    """Process embeddings and calculate all similarity metrics."""
    try:
        # Get prompt embedding
        prompt_embedding = get_embedding(bedrock_client, prompt)
        
        # Process all text samples
        results = []
        embeddings_for_viz = []
        texts_for_viz = []
        
        # Add prompt to visualization data
        embeddings_for_viz.append(prompt_embedding)
        texts_for_viz.append(prompt)
        
        # Process comparison texts
        for text_key, text_value in sample_texts.items():
            if text_value and text_value.strip():
                # Get embedding for the text
                embedding = get_embedding(bedrock_client, text_value)
                
                if embedding:
                    # Calculate all similarity metrics
                    cosine_sim = calculate_cosine_similarity(prompt_embedding, embedding)
                    euclidean_dist = calculate_euclidean_distance(prompt_embedding, embedding)
                    dot_prod = calculate_dot_product(prompt_embedding, embedding)
                    
                    # Store results
                    results.append({
                        'Prompt': prompt,
                        'Text': text_value,
                        'Cosine_Similarity': round(cosine_sim, 4),
                        'Euclidean_Distance': round(euclidean_dist, 4),
                        'Dot_Product': round(dot_prod, 4)
                    })
                    
                    # Add to visualization data
                    embeddings_for_viz.append(embedding)
                    texts_for_viz.append(text_value)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Create embeddings dataframe for visualization
        viz_df = None
        if embeddings_for_viz:
            viz_df = pd.DataFrame(embeddings_for_viz, index=texts_for_viz)
        
        return results_df, viz_df
        
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        st.error(f"An error occurred while processing embeddings: {str(e)}")
        return None, None

def render_results_and_visualization(results_df, viz_df, metric_type, prompt):
    """Render results table and visualization for a specific metric."""
    if results_df is None or viz_df is None:
        return
    
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.markdown('<div class="sub-header">Results</div>', unsafe_allow_html=True)
        
        # Prepare data based on metric type
        if metric_type == "Cosine Similarity":
            display_df = results_df[['Text', 'Cosine_Similarity']].copy()
            display_df = display_df.sort_values(by='Cosine_Similarity', ascending=False)
            metric_col = 'Cosine_Similarity'
            color_scale = 'Viridis'
            
        elif metric_type == "Euclidean Distance":
            display_df = results_df[['Text', 'Euclidean_Distance']].copy()
            display_df = display_df.sort_values(by='Euclidean_Distance', ascending=True)  # Lower is better
            metric_col = 'Euclidean_Distance'
            color_scale = 'Viridis_r'  # Reverse scale for distance
            
        else:  # Dot Product
            display_df = results_df[['Text', 'Dot_Product']].copy()
            display_df = display_df.sort_values(by='Dot_Product', ascending=False)
            metric_col = 'Dot_Product'
            color_scale = 'RdBu'
        
        # Display dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Text": st.column_config.TextColumn("Text", width="medium"),
                metric_col: st.column_config.NumberColumn(
                    metric_type,
                    format="%.4f"
                )
            }
        )
        
        # Show bar chart
        chart = px.bar(
            display_df,
            x='Text',
            y=metric_col,
            color=metric_col,
            color_continuous_scale=color_scale,
            title=f'{metric_type} Scores'
        )
        chart.update_xaxes(tickangle=45)
        st.plotly_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">Vector Space Visualization</div>', unsafe_allow_html=True)
        
        # Generate 2D projection
        x_coords, y_coords = generate_2d_projection(viz_df)
        
        if x_coords and y_coords:
            # Create plotting dataframe
            plot_df = pd.DataFrame({
                'x': x_coords,
                'y': y_coords,
                'text': viz_df.index,
                'is_prompt': [text == prompt for text in viz_df.index]
            })
            
            # Create scatter plot
            fig = px.scatter(
                plot_df,
                x='x',
                y='y',
                color='text',
                symbol='is_prompt',
                size=plot_df['is_prompt'].astype(int) * 10 + 5,
                hover_name='text',
                title="2D Projection of Text Embeddings (PCA)"
            )
            
            fig.update_layout(
                height=400,
                legend_title_text='Texts',
                xaxis_title="Component 1",
                yaxis_title="Component 2"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the Streamlit application."""
    with st.sidebar:
        common.render_sidebar()
        
    st.markdown("""
    <h1>🧊 Similarity Metrics</h1>
    <div class="info-box">
    Compare text embeddings using different similarity and distance metrics. 
    Each metric provides unique insights into semantic relationships between texts.
    </div>
    """, unsafe_allow_html=True)
    
    
    
    # Initialize Bedrock client
    bedrock_client = get_bedrock_client()
    if not bedrock_client:
        st.stop()
    
    # Input section
    st.markdown('<div class="sub-header">Text Input</div>', unsafe_allow_html=True)
    
    # Sample prompt examples
    sample_prompts = {
        "Custom": {
            "prompt": "Hello",
            "texts": ["Hi", "Good Day", "How are you", "Good Morning", "Goodbye"]
        },
        "Customer Support - Product Issue": {
            "prompt": "My laptop won't turn on and the battery seems dead",
            "texts": [
                "The computer is not starting up",
                "Battery is not charging properly",
                "Screen remains black when I press power button",
                "I need help with my coffee maker",
                "The device makes a beeping sound"
            ]
        },
        "E-commerce - Product Search": {
            "prompt": "Looking for comfortable running shoes for marathon training",
            "texts": [
                "Athletic footwear for long-distance running",
                "Cushioned sneakers for jogging",
                "Formal leather dress shoes",
                "Trail running shoes with good grip",
                "Winter boots for hiking"
            ]
        },
        "Healthcare - Symptom Analysis": {
            "prompt": "Patient experiencing severe headache and sensitivity to light",
            "texts": [
                "Migraine with photophobia symptoms",
                "Tension headache with mild discomfort",
                "Broken arm requiring immediate attention",
                "Cluster headache with eye pain",
                "Common cold with runny nose"
            ]
        },
        "Financial Services - Transaction Classification": {
            "prompt": "Monthly subscription payment for streaming service",
            "texts": [
                "Recurring charge for Netflix subscription",
                "One-time purchase at electronics store",
                "Annual membership fee for gym",
                "Grocery shopping at supermarket",
                "Automatic payment for Spotify premium"
            ]
        },
        "Content Moderation - Sentiment Detection": {
            "prompt": "This product exceeded my expectations and I'm very satisfied",
            "texts": [
                "Absolutely love this purchase, highly recommend",
                "The quality is acceptable but nothing special",
                "Terrible experience, complete waste of money",
                "Good value for the price, would buy again",
                "Disappointed with the service and product quality"
            ]
        }
    }
    
    # Initialize session state for prompt and texts
    if 'prompt_text' not in st.session_state:
        st.session_state.prompt_text = "Hello"
    if 'text1' not in st.session_state:
        st.session_state.text1 = "Hi"
    if 'text2' not in st.session_state:
        st.session_state.text2 = "Good Day"
    if 'text3' not in st.session_state:
        st.session_state.text3 = "How are you"
    if 'text4' not in st.session_state:
        st.session_state.text4 = "Good Morning"
    if 'text5' not in st.session_state:
        st.session_state.text5 = "Goodbye"
    
    # Callback for selectbox change
    def on_example_change():
        selected = st.session_state.example_selector
        example_data = sample_prompts[selected]
        st.session_state.prompt_text = example_data["prompt"]
        for i, text in enumerate(example_data["texts"], 1):
            st.session_state[f'text{i}'] = text
    
    # Dropdown to select sample prompt
    selected_example = st.selectbox(
        "💡 Select a practical example or choose 'Custom' to enter your own:",
        options=list(sample_prompts.keys()),
        key="example_selector",
        on_change=on_example_change,
        help="Choose from real-world scenarios to see how similarity metrics work in practice"
    )
    
    with st.form("embedding_form"):
        col1, col2 = st.columns([0.3, 0.7])
        
        with col1:
            prompt = st.text_area("📌 Reference Text:", 
                                 height=100,
                                 key="prompt_text",
                                 help="This is the reference text for comparison")
        
        with col2:
            st.write("**Comparison Texts:**")
            sample_texts = {}
            cols = st.columns(2)
            
            for i in range(1, 6):
                col_idx = (i - 1) % 2
                with cols[col_idx]:
                    sample_texts[f"text{i}"] = st.text_input(
                        f'Text {i}',
                        key=f"text{i}"
                    )
        
        submit = st.form_submit_button(
            "🔍 Analyze Similarities", 
            type="primary",
            use_container_width=True
        )
    
    # Process embeddings when form is submitted
    if submit and prompt:
        with st.spinner("🧠 Processing embeddings and calculating similarities..."):
            results_df, viz_df = process_embeddings(bedrock_client, prompt, sample_texts)
            
            if results_df is not None and not results_df.empty:
                # Store results in session state
                st.session_state['results_df'] = results_df
                st.session_state['viz_df'] = viz_df
                st.session_state['prompt'] = prompt
    
    # Display results in tabs if available
    if 'results_df' in st.session_state:
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Cosine Similarity", "📏 Euclidean Distance", "🔢 Dot Product"])
        
        with tab1:
            render_cosine_similarity_explanation()
            render_results_and_visualization(
                st.session_state['results_df'], 
                st.session_state['viz_df'], 
                "Cosine Similarity",
                st.session_state['prompt']
            )
        
        with tab2:
            render_euclidean_distance_explanation()
            render_results_and_visualization(
                st.session_state['results_df'], 
                st.session_state['viz_df'], 
                "Euclidean Distance",
                st.session_state['prompt']
            )
        
        with tab3:
            render_dot_product_explanation()
            render_results_and_visualization(
                st.session_state['results_df'], 
                st.session_state['viz_df'], 
                "Dot Product",
                st.session_state['prompt']
            )
        
        # Summary comparison section
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Metric Comparison Summary</div>', unsafe_allow_html=True)
        
        # Create comparison table
        comparison_df = st.session_state['results_df'][['Text', 'Cosine_Similarity', 'Euclidean_Distance', 'Dot_Product']].copy()
        
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Text": st.column_config.TextColumn("Text", width="medium"),
                "Cosine_Similarity": st.column_config.NumberColumn("Cosine Similarity", format="%.4f"),
                "Euclidean_Distance": st.column_config.NumberColumn("Euclidean Distance", format="%.4f"),
                "Dot_Product": st.column_config.NumberColumn("Dot Product", format="%.4f")
            }
        )
        
        # Key insights
        with st.expander("🔍 Key Insights"):
            st.markdown("""
            **Understanding the Metrics:**
            
            - **Cosine Similarity**: Best for understanding semantic similarity regardless of text length
            - **Euclidean Distance**: Considers both semantic similarity and magnitude differences
            - **Dot Product**: Raw measure that combines both direction and magnitude
            
            **Interpretation Tips:**
            - Cosine similarity closer to 1 = more semantically similar
            - Euclidean distance closer to 0 = more similar vectors
            - Dot product magnitude indicates strength of relationship
            """)
    
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

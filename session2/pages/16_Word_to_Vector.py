import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import normalize
import uuid
from utils.common import render_sidebar
from utils.styles import load_css, custom_header, sub_header
import utils.authenticate as authenticate


# Page configuration
st.set_page_config(
    page_title="Word to Vector",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if 'word_vectors' not in st.session_state:
        st.session_state.word_vectors = {}
    
    if 'word_input_field' not in st.session_state:
        st.session_state.word_input_field = "machine"
    
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None


def on_example_click(word):
    """Callback function to update word input when example is clicked"""
    st.session_state.word_input_field = word
    st.session_state.selected_example = word


def simple_word_to_vector(word, dimensions):
    """
    Convert a word to a vector using a simple deterministic approach.
    This creates consistent vectors based on character properties.
    """
    # Use word hash as seed for reproducibility
    np.random.seed(hash(word.lower()) % (2**32))
    
    # Generate base vector
    vector = np.random.randn(dimensions)
    
    # Add word-specific features
    # Feature 1: Word length influence
    length_factor = len(word) / 10.0
    vector[0] = vector[0] * (1 + length_factor)
    
    # Feature 2: Vowel ratio influence (if dimensions >= 2)
    if dimensions >= 2:
        vowels = sum(1 for c in word.lower() if c in 'aeiou')
        vowel_ratio = vowels / len(word) if len(word) > 0 else 0
        vector[1] = vector[1] * (1 + vowel_ratio)
    
    # Feature 3: First letter influence (if dimensions >= 3)
    if dimensions >= 3:
        first_char_value = ord(word[0].lower()) / 122.0  # Normalize to 0-1
        vector[2] = vector[2] * (1 + first_char_value)
    
    # Normalize the vector
    vector = normalize(vector.reshape(1, -1))[0]
    
    return vector


def visualize_vector_1d(word, vector):
    """Visualize a 1D vector as a number line"""
    fig = go.Figure()
    
    # Add the point
    fig.add_trace(go.Scatter(
        x=[vector[0]],
        y=[0],
        mode='markers+text',
        marker=dict(size=20, color='#FF9900'),
        text=[word],
        textposition='top center',
        textfont=dict(size=14, color='#232F3E'),
        name=word
    ))
    
    # Add reference line
    fig.add_trace(go.Scatter(
        x=[-1, 1],
        y=[0, 0],
        mode='lines',
        line=dict(color='#E9EBF0', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"1D Vector Representation of '{word}'",
        xaxis_title="Dimension 1",
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        xaxis=dict(range=[-1.2, 1.2], zeroline=True),
        height=300,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def visualize_vector_2d(word, vector):
    """Visualize a 2D vector"""
    fig = go.Figure()
    
    # Add origin to point line
    fig.add_trace(go.Scatter(
        x=[0, vector[0]],
        y=[0, vector[1]],
        mode='lines',
        line=dict(color='#0073BB', width=2),
        showlegend=False
    ))
    
    # Add the point
    fig.add_trace(go.Scatter(
        x=[vector[0]],
        y=[vector[1]],
        mode='markers+text',
        marker=dict(size=15, color='#FF9900'),
        text=[word],
        textposition='top center',
        textfont=dict(size=12, color='#232F3E'),
        name=word
    ))
    
    # Add origin point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(size=8, color='#232F3E'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"2D Vector Representation of '{word}'",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        xaxis=dict(range=[-1.2, 1.2], zeroline=True, zerolinewidth=1, zerolinecolor='#E9EBF0'),
        yaxis=dict(range=[-1.2, 1.2], zeroline=True, zerolinewidth=1, zerolinecolor='#E9EBF0'),
        height=500,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def visualize_vector_3d(word, vector):
    """Visualize a 3D vector"""
    fig = go.Figure()
    
    # Add origin to point line
    fig.add_trace(go.Scatter3d(
        x=[0, vector[0]],
        y=[0, vector[1]],
        z=[0, vector[2]],
        mode='lines',
        line=dict(color='#0073BB', width=4),
        showlegend=False
    ))
    
    # Add the point
    fig.add_trace(go.Scatter3d(
        x=[vector[0]],
        y=[vector[1]],
        z=[vector[2]],
        mode='markers+text',
        marker=dict(size=8, color='#FF9900'),
        text=[word],
        textposition='top center',
        textfont=dict(size=12, color='#232F3E'),
        name=word
    ))
    
    # Add origin point
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(size=5, color='#232F3E'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"3D Vector Representation of '{word}'",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
            xaxis=dict(range=[-1.2, 1.2]),
            yaxis=dict(range=[-1.2, 1.2]),
            zaxis=dict(range=[-1.2, 1.2])
        ),
        height=600,
        showlegend=False
    )
    
    return fig


def visualize_high_dimensional(word, vector, dimensions):
    """Visualize high-dimensional vector as a bar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f"D{i+1}" for i in range(dimensions)],
        y=vector,
        marker=dict(
            color=vector,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="Value")
        ),
        text=[f"{v:.3f}" for v in vector],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"{dimensions}D Vector Representation of '{word}'",
        xaxis_title="Dimensions",
        yaxis_title="Value",
        height=400,
        showlegend=False
    )
    
    return fig


def calculate_vector_properties(vector):
    """Calculate various properties of the vector"""
    magnitude = np.linalg.norm(vector)
    
    # Direction (angle from first axis for 2D)
    if len(vector) == 2:
        angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
    else:
        angle = None
    
    # Sparsity (percentage of near-zero values)
    sparsity = np.sum(np.abs(vector) < 0.1) / len(vector) * 100
    
    return {
        'magnitude': magnitude,
        'angle': angle,
        'sparsity': sparsity,
        'mean': np.mean(vector),
        'std': np.std(vector),
        'min': np.min(vector),
        'max': np.max(vector)
    }


def render_sidebar_content():
    """Render sidebar content"""
    render_sidebar()
    
    with st.expander("About Word to Vector", expanded=False):
        st.markdown("""
        ### Understanding Word Embeddings
        
        **What is Word to Vector?**
        
        Word to vector (word embedding) is the process of converting words into numerical 
        representations that machines can understand and process.
        
        **Why Vectors?**
        
        - Computers work with numbers, not words
        - Vectors capture semantic meaning
        - Similar words have similar vectors
        - Enable mathematical operations on words
        
        **Dimensions Explained:**
        
        - **1D**: Simple scalar value (limited information)
        - **2D-3D**: Easy to visualize, captures basic relationships
        - **4D-10D**: More nuanced representations
        - **Real-world**: Often 100-1000+ dimensions
        
        **Applications:**
        
        - Natural Language Processing
        - Search engines
        - Recommendation systems
        - Machine translation
        - Sentiment analysis
        """)


def main():
    initialize_session_state()
    load_css()
    
    # Sidebar
    with st.sidebar:
        render_sidebar_content()
    
    # Main content
    st.markdown("""
    <h1>🔢 Word to Vector Conversion</h1>
    <div class="info-box">
    Learn how words are converted into numerical vectors that machines can understand. 
    Experiment with different dimensions to see how vector representations change.
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.markdown(sub_header("Convert Word to Vector", "✨"), unsafe_allow_html=True)
        
        # Input section
        with st.container(border=True):
            word_input = st.text_input(
                "Enter a word",
                max_chars=50,
                help="Enter any word to convert it to a vector",
                key="word_input_field"  # ✅ Use key, no value parameter
            )
            
            dimensions = st.slider(
                "Number of Dimensions",
                min_value=1,
                max_value=10,
                value=2,
                help="Select how many dimensions the vector should have"
            )
            
            convert_button = st.button("🚀 Convert to Vector", type="primary", use_container_width=True)
        
        # Process conversion
        if convert_button and word_input:
            word = word_input.strip()
            if word:
                # Generate vector
                vector = simple_word_to_vector(word, dimensions)
                st.session_state.word_vectors[word] = {
                    'vector': vector,
                    'dimensions': dimensions
                }
                
                st.success(f"✅ Successfully converted '{word}' to a {dimensions}D vector!")
                
                # Display vector
                st.markdown(sub_header("Vector Representation", "📊", "minimal"), unsafe_allow_html=True)
                
                # Show vector values
                vector_df = pd.DataFrame({
                    'Dimension': [f"D{i+1}" for i in range(dimensions)],
                    'Value': [f"{v:.6f}" for v in vector]
                })
                st.dataframe(vector_df, use_container_width=True, hide_index=True)
                
                # Visualization based on dimensions
                st.markdown(sub_header("Visualization", "📈", "minimal"), unsafe_allow_html=True)
                
                if dimensions == 1:
                    fig = visualize_vector_1d(word, vector)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="info-box">
                    <strong>1D Vector:</strong> Represented as a single point on a number line. 
                    Limited in capturing complex relationships between words.
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif dimensions == 2:
                    fig = visualize_vector_2d(word, vector)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="info-box">
                    <strong>2D Vector:</strong> Represented as a point in 2D space. 
                    The arrow from origin shows direction and magnitude.
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif dimensions == 3:
                    fig = visualize_vector_3d(word, vector)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="info-box">
                    <strong>3D Vector:</strong> Represented in 3D space. 
                    You can rotate the visualization to see different angles.
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    fig = visualize_high_dimensional(word, vector, dimensions)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>{dimensions}D Vector:</strong> High-dimensional vectors are shown as bar charts. 
                    Each bar represents one dimension's value.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Vector properties
                st.markdown(sub_header("Vector Properties", "🔍", "minimal"), unsafe_allow_html=True)
                
                props = calculate_vector_properties(vector)
                
                prop_cols = st.columns(3)
                prop_cols[0].metric("Magnitude", f"{props['magnitude']:.4f}")
                prop_cols[1].metric("Mean Value", f"{props['mean']:.4f}")
                prop_cols[2].metric("Std Deviation", f"{props['std']:.4f}")
                
                if props['angle'] is not None:
                    st.metric("Angle (degrees)", f"{props['angle']:.2f}°")
                
                with st.expander("📚 Understanding Vector Properties"):
                    st.markdown("""
                    **Magnitude:** The length of the vector (distance from origin). 
                    Normalized vectors have magnitude ≈ 1.0.
                    
                    **Mean Value:** Average of all dimension values. 
                    Indicates overall direction tendency.
                    
                    **Standard Deviation:** Spread of values across dimensions. 
                    Higher values indicate more variation.
                    
                    **Angle (2D only):** Direction of the vector in degrees. 
                    0° points right, 90° points up.
                    """)
    
    with col2:
        st.markdown(sub_header("Key Concepts", "💡", "aws"), unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("""
            ### What is a Vector?
            
            A vector is an ordered list of numbers that represents a point in space.
            
            **Example:**
            - 1D: `[0.5]`
            - 2D: `[0.3, 0.7]`
            - 3D: `[0.2, 0.5, 0.8]`
            
            ### Why Multiple Dimensions?
            
            More dimensions = more information:
            
            - **1D**: Very limited
            - **2-3D**: Basic relationships
            - **4-10D**: Richer representations
            - **100-1000D**: Real-world models
            
            ### Vector Operations
            
            Vectors enable math on words:
            
            - **Distance**: How similar are words?
            - **Direction**: What category?
            - **Addition**: Combine meanings
            - **Subtraction**: Find differences
            """)
        
        st.markdown(sub_header("Example Words", "📝", "minimal"), unsafe_allow_html=True)
        
        example_words = [
            "king", "queen", "man", "woman",
            "cat", "dog", "bird", "fish",
            "happy", "sad", "angry", "calm"
        ]
        
        with st.container(border=True):
            st.markdown("Try these example words:")
            
            cols = st.columns(2)
            for idx, word in enumerate(example_words):
                # ✅ Use on_click callback to update before widget instantiation
                cols[idx % 2].button(
                    word, 
                    key=f"example_{word}", 
                    use_container_width=True,
                    on_click=on_example_click,
                    args=(word,)
                )
    
    # History section
    if st.session_state.word_vectors:
        st.markdown(sub_header("Conversion History", "📜"), unsafe_allow_html=True)
        
        history_data = []
        for word, data in st.session_state.word_vectors.items():
            history_data.append({
                'Word': word,
                'Dimensions': data['dimensions'],
                'Magnitude': f"{np.linalg.norm(data['vector']):.4f}",
                'First 3 Values': ', '.join([f"{v:.3f}" for v in data['vector'][:3]])
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.word_vectors = {}
            st.rerun()
    
    # Educational content
    st.markdown(sub_header("How It Works", "🎓"), unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Basic Concept", "Real-World Models", "Mathematical Details"])
    
    with tab1:
        st.markdown("""
        ### From Words to Numbers
        
        Computers can't understand words directly - they need numbers. Word embeddings solve this by:
        
        1. **Assigning each word a unique vector** of numbers
        2. **Positioning similar words close together** in vector space
        3. **Capturing semantic relationships** through vector arithmetic
        
        ### Simple Example
        
        Imagine a 2D space where:
        - X-axis represents "royalty" (0 = common, 1 = royal)
        - Y-axis represents "gender" (0 = male, 1 = female)
        
        Then:
        - "king" might be `[0.9, 0.1]` (royal, male)
        - "queen" might be `[0.9, 0.9]` (royal, female)
        - "man" might be `[0.1, 0.1]` (common, male)
        - "woman" might be `[0.1, 0.9]` (common, female)
        
        ### Vector Arithmetic
        
        With vectors, we can do math:
        - `king - man + woman ≈ queen`
        - `Paris - France + Italy ≈ Rome`
        """)
    
    with tab2:
        st.markdown("""
        ### Real-World Embedding Models
        
        **Word2Vec** (Google, 2013)
        - 300 dimensions typically
        - Trained on billions of words
        - Captures semantic relationships
        
        **GloVe** (Stanford, 2014)
        - 50-300 dimensions
        - Based on word co-occurrence statistics
        - Good for similarity tasks
        
        **FastText** (Facebook, 2016)
        - Handles unknown words better
        - Uses character n-grams
        - Works with misspellings
        
        **BERT/Transformer Models** (2018+)
        - 768-1024 dimensions
        - Context-aware embeddings
        - State-of-the-art performance
        
        **Amazon Titan Embeddings**
        - 1536 dimensions
        - Optimized for AWS Bedrock
        - Supports text and multimodal
        """)
    
    with tab3:
        st.markdown("""
        ### Mathematical Foundation
        
        **Vector Representation**
        ```
        word → [v₁, v₂, v₃, ..., vₙ]
        ```
        
        **Cosine Similarity**
        ```
        similarity(A, B) = (A · B) / (||A|| × ||B||)
        ```
        - Range: -1 to 1
        - 1 = identical direction
        - 0 = orthogonal (unrelated)
        - -1 = opposite direction
        
        **Euclidean Distance**
        ```
        distance(A, B) = √(Σ(aᵢ - bᵢ)²)
        ```
        - Measures straight-line distance
        - Smaller = more similar
        
        **Normalization**
        ```
        normalized(v) = v / ||v||
        ```
        - Makes magnitude = 1
        - Focuses on direction only
        """)
    
    # Footer
    st.markdown("""
    <div class="aws-footer">
        © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        if 'localhost' in st.context.headers["host"]:
            main()
        else:
            if authenticate.login():
                main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        with st.expander("Error Details"):
            st.code(str(e))

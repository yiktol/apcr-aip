import streamlit as st
import pandas as pd
import boto3
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import time
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import uuid

# Page configuration
st.set_page_config(
    page_title="Word Embeddings",
    page_icon="🔠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main title and headers */
    h1, h2, h3 {
        color: #0066cc;
        margin-bottom: 1rem;
    }
    h1 {
        text-align: left;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    
    /* Cards for content sections */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f0f7ff;
        border-left: 5px solid #0066cc;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    
    /* Success boxes */
    .success-box {
        background-color: #e6f4ea;
        border-left: 5px solid #34a853;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 500;
    }
    
    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 10px !important;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f2f6;
        color: #888;
        font-size: 0.8rem;
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc !important;
        color: white !important;
    }
    
    /* Custom multiselect */
    div[data-testid="stMultiSelect"] div[data-testid="stVerticalBlock"] {
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Session info box styling */
    .session-info {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
        font-size: 0.9em;
    }
    
    /* AWS Footer */
    .aws-footer {
        text-align: center;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f2f6;
        color: #555;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if 'embeddings_generated' not in st.session_state:
        st.session_state.embeddings_generated = False
    if 'embedding_list' not in st.session_state:
        st.session_state.embedding_list = []
    if 'txt_array' not in st.session_state:
        st.session_state.txt_array = []
    if 'custom_words' not in st.session_state:
        st.session_state.custom_words = []

# Initialize session
initialize_session_state()

# Function to reset session
def reset_session():
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.embeddings_generated = False
    st.session_state.embedding_list = []
    st.session_state.txt_array = []
    st.session_state.custom_words = []
    st.rerun()


with st.sidebar:
    # Session information
    st.markdown("<div class='session-info'>", unsafe_allow_html=True)
    st.markdown("### 🔑 Session Info")
    st.markdown(f"**User ID:** {st.session_state.session_id}")
    
    if st.button("🔄 Reset Session"):
        reset_session()
    st.markdown("</div>", unsafe_allow_html=True)

    # About section (collapsible)
    with st.expander("About this Application", expanded=False):
        st.markdown("""
        ### About Word Embeddings
        
        This application demonstrates the power of Amazon Bedrock's Titan embedding model for natural language processing.
        
        **What are word embeddings?**
        
        Word embeddings are vector representations of words where semantically similar words are mapped to nearby points in vector space. 
        They capture the meaning, semantic relationships, and context of words mathematically.
        
        **Key features:**
        - Generate embeddings using Amazon Bedrock's Titan model
        - Visualize word relationships in 2D and 3D space
        - Analyze semantic similarity between words
        - Compare specific word pairs
        - Export embedding data for further analysis
        
        **Use cases:**
        - Natural language processing research
        - Content recommendation systems
        - Sentiment analysis
        - Document classification
        - Search engine optimization
        """)





# Function to create AWS Bedrock client
@st.cache_resource
def runtime_client(region='us-east-1'):
    try:
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region, 
        )
        return bedrock_runtime
    except Exception as e:
        st.error(f"Error connecting to AWS Bedrock: {str(e)}")
        return None

# Function to get embeddings from AWS Bedrock
def get_embedding(bedrock, text):
    try:
        modelId = 'amazon.titan-embed-g1-text-02'
        accept = 'application/json'
        contentType = 'application/json'
        input = {
            'inputText': text
        }
        body = json.dumps(input)
        response = bedrock.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get('body').read())
        embedding = response_body['embedding']
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding for '{text}': {str(e)}")
        return None

# Generate a prettier color map for the plots
def get_color_map(categories):
    color_list = list(mcolors.TABLEAU_COLORS.values())
    colors = {}
    for i, category in enumerate(categories):
        colors[category] = color_list[i % len(color_list)]
    return colors

# Function to create similarity heatmap
def create_similarity_heatmap(embeddings_df):
    # Calculate cosine similarity matrix
    embeddings_array = np.array(embeddings_df['Embeddings'].tolist())
    similarity_matrix = cosine_similarity(embeddings_array)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=embeddings_df['Text'],
        y=embeddings_df['Text'],
        colorscale='Viridis',
        zmin=0, zmax=1,
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text="Cosine Similarity",
                side="right"
            )
        )
    ))
    
    fig.update_layout(
        title="Word Similarity Heatmap",
        height=600,
        width=600,
        xaxis_title="Words",
        yaxis_title="Words",
    )
    
    return fig

# Function to create 3D visualization
def create_3d_visualization(df_vectors):
    pca = PCA(n_components=3, svd_solver='auto')
    pca_result = pca.fit_transform(df_vectors.values)
    
    fig = px.scatter_3d(
        x=pca_result[:,0], 
        y=pca_result[:,1], 
        z=pca_result[:,2],
        color=df_vectors.index,
        text=df_vectors.index,
        title="3D Word Embedding Visualization",
        labels={'color': 'Words'},
    )
    
    fig.update_traces(
        marker=dict(size=8, opacity=0.8),
        textposition="top center"
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        height=700,
    )
    
    return fig

# Function to export data to CSV
def create_download_link(df, filename):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Create 70/30 layout
left_column, right_column = st.columns([0.7, 0.3])

# Main content area (70%)
with left_column:
    # App header
    st.markdown("""
    <h1>🔠 Word Embeddings</h1>
    <div class="info-box">
    Word embeddings convert text into numerical vectors, allowing machines to understand semantic relationships.
    This tool helps you visualize how different words relate to each other in vector space using Amazon Bedrock.
    </div>
    """, unsafe_allow_html=True)
    
    # Display selected words
    all_selected_words = st.session_state.custom_words
    
    if 'category' in st.session_state and 'selected_words' in st.session_state:
        all_selected_words = st.session_state.selected_words + st.session_state.custom_words
    
    # Generate embeddings form
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.form("embedding_form"):
        st.markdown("## Generate Word Embeddings")
        st.markdown("Selected words for embedding generation:")
        
        # Display selected words in columns for better UI
        if all_selected_words:
            cols = st.columns(4)
            for i, word in enumerate(all_selected_words):
                cols[i % 4].markdown(f"• {word}")
        else:
            st.info("Please select words from the right panel to generate embeddings")
        
        generate_button = st.form_submit_button("🚀 Generate Embeddings", use_container_width=True, type='primary')

        if generate_button and all_selected_words:
            # Clear previous results if any
            st.session_state.embedding_list = []
            st.session_state.txt_array = []
            
            # Get Bedrock client
            bedrock = runtime_client()
            
            if bedrock:
                # Progress bar
                progress_bar = st.progress(0)
                embedding_status = st.empty()
                
                for i, word in enumerate(all_selected_words):
                    embedding_status.text(f"Generating embedding for: {word}")
                    embedding = get_embedding(bedrock, word)
                    
                    if embedding:
                        st.session_state.embedding_list.append(embedding)
                        st.session_state.txt_array.append(word)
                    
                    # Update progress
                    progress_percentage = (i + 1) / len(all_selected_words)
                    progress_bar.progress(progress_percentage)
                    time.sleep(0.1)  # Small delay for visual feedback
                    
                embedding_status.empty()
                progress_bar.empty()
                
                if len(st.session_state.embedding_list) > 0:
                    st.session_state.embeddings_generated = True
                    st.success(f"Successfully generated embeddings for {len(st.session_state.embedding_list)} words!")
                else:
                    st.error("Failed to generate any embeddings. Please check your connection and try again.")
            else:
                st.error("Failed to connect to AWS Bedrock. Please check your AWS configuration.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display results if embeddings were generated
    if st.session_state.embeddings_generated and len(st.session_state.embedding_list) > 0:
        # Create dataframes
        df_embeddings = pd.DataFrame({
            'Text': st.session_state.txt_array, 
            'Embeddings': st.session_state.embedding_list
        })
        
        # Create a dataframe for the vectors
        embeddings_array = np.array(df_embeddings['Embeddings'].tolist())
        df_vectors = pd.DataFrame(embeddings_array)
        df_vectors.index = df_embeddings['Text']
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["2D Visualization", "3D Visualization", "Word Similarities", "Raw Data"])
        
        with tab1:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("## 2D Vector Space Visualization")
            st.markdown("""
            <div class='info-box'>
            This visualization projects high-dimensional word embeddings onto a 2D space using PCA (Principal Component Analysis).
            Words with similar meanings should appear closer together.
            </div>
            """, unsafe_allow_html=True)
            
            # 2D visualization
            pca = PCA(n_components=2, svd_solver='auto')
            pca_result = pca.fit_transform(df_vectors.values)
            
            explained_variance = pca.explained_variance_ratio_
            
            # Create metrics for explained variance
            col1, col2, col3 = st.columns(3)
            col1.metric("Words Visualized", len(df_embeddings))
            col2.metric("PCA Dimension 1 Variance", f"{explained_variance[0]:.2%}")
            col3.metric("PCA Dimension 2 Variance", f"{explained_variance[1]:.2%}")
            
            show_annotations = True
            chart_height = 600
            # Create interactive 2D scatter plot
            fig = px.scatter(
                x=pca_result[:,0], 
                y=pca_result[:,1],
                color=df_vectors.index,
                text=df_vectors.index if show_annotations else None,
                title="2D Word Embedding Visualization",
                labels={'color': 'Words', 'x': 'Principal Component 1', 'y': 'Principal Component 2'},
                height=chart_height,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            
            fig.update_traces(
                marker=dict(size=12, opacity=0.8),
                textposition="top center" if show_annotations else None
            )
            
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tab2:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("## 3D Vector Space Visualization")
            st.markdown("""
            <div class='info-box'>
            This 3D visualization gives a more detailed view of word relationships.
            You can rotate, zoom, and pan to explore the embedding space.
            </div>
            """, unsafe_allow_html=True)
            
            # 3D visualization (if we have enough dimensions)
            if df_vectors.shape[1] >= 3:
                fig_3d = create_3d_visualization(df_vectors)
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("Need at least 3 dimensions to create a 3D visualization.")
            st.markdown("</div>", unsafe_allow_html=True)
                
        with tab3:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("## Word Similarity Analysis")
            
            # Word similarity metrics
            st.markdown("""
            <div class='info-box'>
            This heatmap shows the cosine similarity between each pair of words.
            Lighter colors indicate higher similarity (words that are more closely related).
            </div>
            """, unsafe_allow_html=True)
            
            # Create similarity heatmap
            similarity_fig = create_similarity_heatmap(df_embeddings)
            st.plotly_chart(similarity_fig, use_container_width=True)
            
            # Most similar pairs
            st.markdown("### Most Similar Word Pairs")
            embeddings_array = np.array(df_embeddings['Embeddings'].tolist())
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # Create a dataframe with similarity scores
            similarity_df = pd.DataFrame(similarity_matrix, index=df_embeddings['Text'], columns=df_embeddings['Text'])
            
            # Get the top similar pairs (excluding self-similarity)
            pairs = []
            for i in range(len(similarity_df)):
                for j in range(i+1, len(similarity_df)):
                    word1 = similarity_df.index[i]
                    word2 = similarity_df.columns[j]
                    similarity = similarity_df.iloc[i, j]
                    pairs.append((word1, word2, similarity))
            
            # Sort by similarity and get top 10
            pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = pairs[:10]
            
            # Display in a nice table
            pair_df = pd.DataFrame(top_pairs, columns=["Word 1", "Word 2", "Similarity"])
            pair_df["Similarity"] = pair_df["Similarity"].apply(lambda x: f"{x:.4f}")
            st.dataframe(pair_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tab4:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("## Raw Vector Data")
            
            # Display the raw vector data
            st.markdown("""
            <div class='info-box'>
            This table shows the raw embedding vectors for each word.
            Each row represents a word, and each column represents a dimension in the embedding space.
            </div>
            """, unsafe_allow_html=True)
            
            # Show dimensions
            st.metric("Vector Dimensions", df_vectors.shape[1])
            
            # Show data with download option
            st.dataframe(df_vectors.round(4), use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                csv = df_vectors.to_csv()
                st.download_button(
                    label="Download Vectors as CSV",
                    data=csv,
                    file_name='word_embeddings.csv',
                    mime='text/csv',
                )
            
            with col2:
                # Serialize the full embeddings dataframe
                json_data = json.dumps({
                    'words': st.session_state.txt_array,
                    'embeddings': [emb for emb in st.session_state.embedding_list]
                })
                st.download_button(
                    label="Download Full Data as JSON",
                    data=json_data,
                    file_name='word_embeddings.json',
                    mime='application/json',
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Word comparison section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Word Comparison")
        st.markdown("""
        <div class='info-box'>
        Compare specific words to see how similar they are in the embedding space.
        Higher cosine similarity indicates words that are more semantically related.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            word1 = st.selectbox("Select first word", options=st.session_state.txt_array, key="word1")
        with col2:
            word2 = st.selectbox("Select second word", options=st.session_state.txt_array, key="word2")
        
        if st.button("Compare Words"):
            if word1 and word2:
                # Get embeddings
                idx1 = df_embeddings[df_embeddings['Text'] == word1].index[0]
                idx2 = df_embeddings[df_embeddings['Text'] == word2].index[0]
                
                emb1 = np.array(df_embeddings.loc[idx1, 'Embeddings']).reshape(1, -1)
                emb2 = np.array(df_embeddings.loc[idx2, 'Embeddings']).reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(emb1, emb2)[0][0]
                
                # Create a gauge chart for similarity
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=similarity,
                    title={'text': f"Similarity between '{word1}' and '{word2}'"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "royalblue"},
                        'steps': [
                            {'range': [0, 0.33], 'color': "lightgray"},
                            {'range': [0.33, 0.67], 'color': "gray"},
                            {'range': [0.67, 1], 'color': "darkblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': similarity
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpret the similarity
                if similarity > 0.8:
                    st.success(f"'{word1}' and '{word2}' are very similar in meaning.")
                elif similarity > 0.6:
                    st.info(f"'{word1}' and '{word2}' are moderately similar.")
                elif similarity > 0.4:
                    st.warning(f"'{word1}' and '{word2}' have some similarity.")
                else:
                    st.error(f"'{word1}' and '{word2}' are not very similar.")
        st.markdown("</div>", unsafe_allow_html=True)

    # AWS Footer
    st.markdown("""
    <div class="aws-footer">
        © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

# Right column (30%) - Configuration panels
with right_column:

    with st.container(border=True):
    
        # Word categories
        st.markdown("#### ⚙️ Configuration")
        categories = {
            "Animals": ["Coyotes", "Wolves", "Foxes", "Ducks", "Eagles", "Owls",
                    "Vultures", "Woodpeckers", "Cheetahs", "Jaguars", "Lions", "Tigers",
                    "Gorillas", "Monkeys", "Horses", "Elephants", "Rabbits"],
            "Colors": ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White",
                    "Brown", "Pink", "Gray", "Violet", "Indigo", "Cyan", "Magenta"],
            "Emotions": ["Happy", "Sad", "Angry", "Afraid", "Surprised", "Disgusted", "Excited",
                    "Anxious", "Calm", "Content", "Bored", "Curious", "Proud", "Ashamed"],
            "Countries": ["USA", "Canada", "Mexico", "Brazil", "France", "Germany", "Italy", "Spain",
                        "China", "Japan", "India", "Australia", "Russia", "Egypt", "Kenya"],
            "Tech Companies": ["Apple", "Google", "Microsoft", "Amazon", "Facebook", "Netflix",
                            "Twitter", "Tesla", "IBM", "Intel", "Samsung", "Sony", "Oracle"]
        }
        
        category = st.selectbox("Word Category", options=list(categories.keys()))
        st.session_state.category = category
        
        selected_words = st.multiselect(
            "Select Words",
            options=categories[category],
            default=categories[category][:15]
        )
        st.session_state.selected_words = selected_words
        
        st.markdown("#### 🔍 Custom Words")
        custom_word = st.text_input("Add a custom word")
        
        if st.button("Add Custom Word"):
            if custom_word and custom_word not in st.session_state.custom_words and custom_word not in selected_words:
                st.session_state.custom_words.append(custom_word)
                st.success(f"Added: {custom_word}")
        
        if st.session_state.custom_words:
            st.markdown("### Custom Words Added:")
            for i, word in enumerate(st.session_state.custom_words):
                col1, col2 = st.columns([4, 1])
                col1.text(f"• {word}")
                if col2.button("❌", key=f"delete_{i}"):
                    st.session_state.custom_words.remove(word)
                    st.rerun()
            
            if st.button("Clear All Custom Words"):
                st.session_state.custom_words = []
                st.rerun()
        
        # Visualization settings
        st.markdown("#### 🎨 Visualization Settings")
        chart_height = st.slider("Chart Height", min_value=400, max_value=800, value=600, step=50)
        chart_theme = st.select_slider("Color Theme", options=["viridis", "plasma", "inferno", "cividis", "rainbow"])
        show_annotations = st.checkbox("Show Word Labels", value=True)
        

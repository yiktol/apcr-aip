

"""
Word2Vec Text Vectorization Tool

This Streamlit application generates word vectors from text input using Word2Vec.
It allows users to configure vector dimensions and visualize the results.
"""
import logging
import traceback
from typing import List, Tuple

import gensim
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
import plotly.express as px
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
    page_title="Word2Vec Text Vectorizer",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #4CAF50;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #F44336;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #2196F3;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def download_nltk_dependencies():
    """Download required NLTK dependencies."""
    try:
        nltk.download('punkt', quiet=True)
        logger.info("NLTK dependencies downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK dependencies: {str(e)}")
        st.error(f"Failed to download NLTK dependencies: {str(e)}")


def tokenize_text(text: str) -> List[List[str]]:
    """
    Tokenize the input text into sentences and words.
    
    Args:
        text: Input text string to tokenize
        
    Returns:
        List of tokenized sentences, where each sentence is a list of words
    """
    try:
        data = []
        for sentence in sent_tokenize(text):
            words = [word.lower() for word in word_tokenize(sentence) if word.isalnum()]
            if words:  # Only add non-empty sentences
                data.append(words)
        return data
    except Exception as e:
        logger.error(f"Tokenization error: {str(e)}")
        raise


def create_word2vec_model(data: List[List[str]], vector_size: int) -> Tuple[gensim.models.Word2Vec, np.ndarray]:
    """
    Create a Word2Vec model and generate word vectors.
    
    Args:
        data: Tokenized text data
        vector_size: Dimension of the word vectors
        
    Returns:
        Tuple containing the Word2Vec model and the word vectors as numpy array
    """
    try:
        model = gensim.models.Word2Vec(
            data, 
            min_count=1,
            vector_size=vector_size,
            sg=0,  # CBOW model
            window=5,
            workers=4
        )
        vectors = np.array(model.wv.vectors)
        logger.info(f"Word2Vec model created successfully with {len(model.wv.index_to_key)} words")
        return model, vectors
    except Exception as e:
        logger.error(f"Error creating Word2Vec model: {str(e)}")
        raise


def get_closest_words(model, word, n=5):
    """
    Get the n closest words to the input word based on cosine similarity.
    
    Args:
        model: Word2Vec model
        word: Target word
        n: Number of similar words to retrieve
        
    Returns:
        List of tuples containing (word, similarity)
    """
    try:
        if word in model.wv:
            return model.wv.most_similar(word, topn=n)
        return []
    except Exception as e:
        logger.error(f"Error finding similar words: {str(e)}")
        return []


def display_word_similarities(model, vocab):
    """
    Display the word similarities UI and handle user interactions without rerunning the app.
    
    Args:
        model: Word2Vec model
        vocab: List of words in the vocabulary
    """
    st.write("Find words with similar vector representations:")
    
    # Use session state to persist the selected word and number of similar words
    if 'query_word' not in st.session_state:
        st.session_state.query_word = vocab[0] if vocab else ""
    
    if 'num_similar' not in st.session_state:
        st.session_state.num_similar = 5
    
    # Create columns for input controls
    col1, col2 = st.columns([3, 2])
    
    # Define callbacks to update session state without rerunning
    def update_query_word():
        st.session_state.query_word = st.session_state.temp_query_word
    
    def update_num_similar():
        st.session_state.num_similar = st.session_state.temp_num_similar
    
    # Input widgets
    with col1:
        st.selectbox(
            "Select a word",
            options=vocab,
            index=vocab.index(st.session_state.query_word) if st.session_state.query_word in vocab else 0,
            key="temp_query_word",
            on_change=update_query_word
        )
    
    with col2:
        st.slider(
            "Number of similar words",
            min_value=1,
            max_value=min(10, len(vocab)) if vocab else 10,
            value=st.session_state.num_similar,
            key="temp_num_similar",
            on_change=update_num_similar
        )
    
    # Get and display similar words based on session state
    query_word = st.session_state.query_word
    num_similar = st.session_state.num_similar
    
    similar_words = get_closest_words(model, query_word, num_similar)
    
    if similar_words:
        similarity_df = pd.DataFrame(
            similar_words, 
            columns=['Word', 'Similarity']
        )
        
        # Create bar chart of similarities
        fig = px.bar(
            similarity_df, 
            x='Word', 
            y='Similarity', 
            title=f"Words similar to '{query_word}'",
            color='Similarity',
            color_continuous_scale='Blues',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            similarity_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(f"No similar words found for '{query_word}'")


def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">Word2Vec Text Vectorizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-box">Transform text into numerical vectors using Word2Vec</p>', unsafe_allow_html=True)
    
    # Download dependencies
    with st.spinner("Loading required resources..."):
        download_nltk_dependencies()
    
    # Initialize session state for preserving model between interactions
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'vectors' not in st.session_state:
        st.session_state.vectors = None
    if 'vocab' not in st.session_state:
        st.session_state.vocab = []
    if 'vector_size' not in st.session_state:
        st.session_state.vector_size = 2
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Word Vectors"
    
    # Sidebar configuration
    with st.sidebar:
        common.render_sidebar()
        with st.expander("About this App", expanded=False):
            st.info(
                "This tool generates vector representations of words using the Word2Vec algorithm. "
                "These vectors capture the semantic relationships between words."
            )
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col2:
        with st.container(border=True):
            st.subheader("Model Configuration")
            
            vector_size = st.slider(
                "Vector Dimensions",
                min_value=2,  # At least 2D for visualization
                max_value=300,
                value=st.session_state.vector_size,
                step=10,
                key="vector_size_input",
                help="Higher dimensions capture more semantic information but require more computation"
            )
            st.session_state.vector_size = vector_size
            
            st.markdown('---')
            st.markdown('<div class="info-box">Higher-dimensional vectors usually capture more semantic information, but also require more computational resources.</div>', unsafe_allow_html=True)
            
    
    with col1:
        with st.form("vectorization_form"):
            text_input = st.text_area(
                "Enter text to vectorize:",
                height=150,
                placeholder="Example: A puppy is to dog as kitten is to cat.",
                value="A puppy is to dog as kitten is to cat."
            )
            submitted = st.form_submit_button("Vectorize Text", use_container_width=True)
        
        if not text_input:
            st.warning("Please enter some text to vectorize.")
            return
    
    # Process text when form is submitted
    if submitted:
        try:
            with st.spinner("Generating word vectors..."):
                # Process text
                tokenized_data = tokenize_text(text_input)
                
                if not tokenized_data:
                    st.warning("No valid text to process after tokenization.")
                    return
                
                # Create model
                model, vectors = create_word2vec_model(tokenized_data, vector_size)
                
                # Store model in session state
                st.session_state.model = model
                st.session_state.vectors = vectors
                st.session_state.vocab = model.wv.index_to_key
                st.session_state.active_tab = "Word Vectors"
                
                # Success message
                st.markdown('<div class="success-box">Vectorization complete!</div>', unsafe_allow_html=True)
        
        except Exception as e:
            logger.error(f"Error in main application: {traceback.format_exc()}")
            error_msg = str(e)
            st.markdown(
                f'<div class="error-box">An error occurred: {error_msg}</div>',
                unsafe_allow_html=True
            )
    
    # Only display results if we have a model
    if st.session_state.model:
        model = st.session_state.model
        vectors = st.session_state.vectors
        vocab = st.session_state.vocab
        
        # Create tabs
        tab_names = ["Word Vectors", "Vector Visualization", "Word Similarities"]
        tabs = st.tabs(tab_names)
        
        # Tab 1: Word Vectors
        with tabs[0]:
            df = pd.DataFrame({
                'Word': vocab,
                'Vector': [model.wv[word] for word in vocab]
            })
            
            st.write(f"Generated {len(vocab)} word vectors with {vector_size} dimensions")
            st.dataframe(
                df,
                column_config={
                    "Vector": st.column_config.ListColumn("Vector", width="large"),
                },
                use_container_width=True
            )
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="word_vectors.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                np.save('vectors.npy', vectors)
                with open('vectors.npy', 'rb') as f:
                    st.download_button(
                        "Download NumPy Array",
                        data=f,
                        file_name="word_vectors.npy",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
        
        # Tab 2: Vector Visualization
        with tabs[1]:
            if vector_size < 2:
                st.warning("Vector dimension must be at least 2 for visualization")
            else:
                # For visualization, if dimensions > 2, we'll just use the first 2 dimensions
                viz_df = pd.DataFrame({
                    'Word': vocab,
                    'Dim1': [model.wv[word][0] for word in vocab],
                    'Dim2': [model.wv[word][1] for word in vocab]
                })
                
                fig = px.scatter(
                    viz_df, x='Dim1', y='Dim2', text='Word',
                    title=f"2D Visualization of Word Vectors (showing first 2 dimensions of {vector_size}D vectors)",
                    width=800, height=500
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Word Similarities - Using the dedicated function to prevent rerunning
        with tabs[2]:
            
            
            st.markdown('<h3>Word Vector Properties</h3>', unsafe_allow_html=True)
            st.info(
                f"• **Vocabulary Size**: {len(model.wv.index_to_key)} words\n"
                f"• **Vector Dimensions**: {vector_size}\n"
                f"• **Model Type**: CBOW (Continuous Bag of Words)\n"
                f"• **Context Window**: 5 words"
            )
            
            # Show example vector
            if vocab:
                example_word = vocab[0]
                st.markdown(f"### Example: Vector for '{example_word}'")
                example_vector = model.wv[example_word]
                
                # Format vector nicely
                if len(example_vector) > 10:
                    displayable_vector = np.concatenate([
                        example_vector[:5],
                        np.array([...]),
                        example_vector[-5:]
                    ])
                    st.code(str(displayable_vector))
                else:
                    st.code(str(example_vector))
                    
            display_word_similarities(model, vocab)


# Main execution flow
if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()


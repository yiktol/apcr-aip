
import streamlit as st
import uuid
from utils.styles import load_css
from utils.knowledge_check import display_knowledge_check
import utils.authenticate as authenticate
import utils.common as common

# Set page configuration
st.set_page_config(
    page_title="How Transformer Architecture Works",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def initialize_session_state():
    """Initialize session state variables"""
    
    common.initialize_session_state()
    
    if "knowledge_check_started" not in st.session_state:
        st.session_state.knowledge_check_started = False
    
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    
    if "score" not in st.session_state:
        st.session_state.score = 0
        
def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # App title and header
    st.markdown("""
    <div class="element-animation">
        <h1>🧠 How Transformer Architecture Works</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""<div class="info-box">
The Transformer architecture revolutionized AI by using self-attention mechanisms to process sequences in parallel, enabling models to understand context and relationships between words regardless of their distance in text.
    </div>""", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        common.render_sidebar()
        
        with st.expander("About this App", expanded=False):
            st.write("""
            Learn the core components of Transformer architecture:
            - **Self-Attention**: How models focus on relevant words
            - **Encoder-Decoder**: Processing and generating sequences
            - **Positional Encoding**: Understanding word order
            - **Feed-Forward Networks**: Final processing layers
            """)

    # Main content - Tab-based navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Self-Attention", 
        "🔄 Encoder-Decoder", 
        "📍 Positional Encoding", 
        "⚡ Feed-Forward Network",
        "❓ Knowledge Check"
    ])
    
    with tab1:
        self_attention_tab()
    
    with tab2:
        encoder_decoder_tab()
    
    with tab3:
        positional_encoding_tab()
    
    with tab4:
        feedforward_tab()
    
    with tab5:
        display_knowledge_check()
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")


def self_attention_tab():
    st.header("Self-Attention Mechanism")
    
    st.markdown("""
    ### What is Self-Attention?
    
    Self-attention allows the model to weigh the importance of different words when processing each word in a sentence.
    It answers: **"Which other words should I focus on to understand this word better?"**
    """)
    
    # Interactive example
    st.subheader("🎮 Interactive Attention Visualization")
    
    # Predefined examples for better demonstration
    example_sentences = {
        "Pronoun Resolution": "The cat didn't cross the street because it was too tired",
        "Subject-Verb Agreement": "The keys to the cabinet are on the table",
        "Word Relationship": "The bank can refuse to lend money to customers",
    }
    
    selected_example = st.selectbox("Choose an example:", list(example_sentences.keys()))
    sentence = example_sentences[selected_example]
    
    st.info(f"**Sentence:** {sentence}")
    
    words = sentence.split()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        focus_word_index = st.selectbox(
            "Select a word to see its attention:",
            range(len(words)),
            format_func=lambda x: f"{x}: {words[x]}"
        )
    
    with col2:
        display_attention_weights(words, focus_word_index, selected_example)
    
    # Explanation
    st.markdown("""
    ### How It Works
    
    1. **Query (Q)**: Represents the word asking "what should I focus on?"
    2. **Key (K)**: Represents each word being considered
    3. **Value (V)**: The actual information to extract from each word
    4. **Attention Score**: Calculated as similarity between Query and Keys
    
    The model learns these representations during training to capture meaningful relationships.
    """)


def display_attention_weights(words, focus_index, example_type):
    import plotly.graph_objects as go
    import numpy as np
    
    # Create attention patterns based on example type
    attention_weights = np.zeros(len(words))
    
    if example_type == "Pronoun Resolution":
        # "it" should attend to "cat"
        if focus_index == 9:  # "it"
            attention_weights[1] = 0.8  # "cat"
            attention_weights[9] = 0.15  # "it" itself
            attention_weights[11] = 0.05  # "was"
        else:
            attention_weights = np.random.dirichlet(np.ones(len(words)) * 0.5)
            attention_weights[focus_index] *= 2
    
    elif example_type == "Subject-Verb Agreement":
        # "are" should attend to "keys"
        if focus_index == 6:  # "are"
            attention_weights[1] = 0.75  # "keys"
            attention_weights[6] = 0.15  # "are" itself
            attention_weights[0] = 0.1   # "The"
        else:
            attention_weights = np.random.dirichlet(np.ones(len(words)) * 0.5)
            attention_weights[focus_index] *= 2
    
    else:  # Word Relationship
        # "refuse" should attend to "bank"
        if focus_index == 3:  # "refuse"
            attention_weights[1] = 0.7  # "bank"
            attention_weights[3] = 0.2  # "refuse" itself
            attention_weights[5] = 0.1  # "lend"
        else:
            attention_weights = np.random.dirichlet(np.ones(len(words)) * 0.5)
            attention_weights[focus_index] *= 2
    
    # Normalize
    attention_weights = attention_weights / attention_weights.sum()
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=words,
            y=attention_weights,
            marker_color=['red' if i == focus_index else 'lightblue' for i in range(len(words))],
            text=[f'{w:.2f}' for w in attention_weights],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f'Attention weights for: "{words[focus_index]}"',
        xaxis_title="Words",
        yaxis_title="Attention Weight",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def encoder_decoder_tab():
    st.header("Encoder-Decoder Architecture")
    
    st.markdown("""
    ### The Two-Part System
    
    - **Encoder**: Processes and understands the input sequence
    - **Decoder**: Generates the output sequence based on encoder's understanding
    """)
    
    # Interactive translation example
    st.subheader("🎮 Interactive Translation Example")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Encoder (Input)")
        input_lang = st.selectbox("Source Language", ["English", "Spanish", "French"])
        
        examples = {
            "English": "The cat sleeps",
            "Spanish": "El gato duerme", 
            "French": "Le chat dort"
        }
        
        input_text = st.text_input("Input text:", value=examples[input_lang])
        
        if st.button("Process", key="encode_btn"):
            st.session_state.encoded = True
    
    with col2:
        st.markdown("### 📤 Decoder (Output)")
        output_lang = st.selectbox("Target Language", ["Spanish", "French", "English"])
        
        if st.session_state.get('encoded', False):
            translations = {
                ("English", "Spanish"): "El gato duerme",
                ("English", "French"): "Le chat dort",
                ("Spanish", "English"): "The cat sleeps",
                ("Spanish", "French"): "Le chat dort",
                ("French", "English"): "The cat sleeps",
                ("French", "Spanish"): "El gato duerme",
            }
            
            output_text = translations.get((input_lang, output_lang), input_text)
            st.success(f"**Output:** {output_text}")
    
    # Visualization of the architecture
    st.subheader("Architecture Flow")
    
    st.markdown("""
    ```
    Input Sequence
         ↓
    [ENCODER STACK]
    - Self-Attention Layer
    - Feed-Forward Layer
    (repeated N times)
         ↓
    Context Representation
         ↓
    [DECODER STACK]
    - Self-Attention Layer
    - Cross-Attention Layer (attends to encoder output)
    - Feed-Forward Layer
    (repeated N times)
         ↓
    Output Sequence
    ```
    """)
    
    st.markdown("""
    ### Key Differences
    
    | Component | Encoder | Decoder |
    |-----------|---------|---------|
    | **Purpose** | Understand input | Generate output |
    | **Attention** | Self-attention only | Self-attention + Cross-attention |
    | **Processing** | Parallel (all at once) | Sequential (one token at a time) |
    """)


def positional_encoding_tab():
    st.header("Positional Encoding")
    
    st.markdown("""
    ### Why Do We Need It?
    
    Unlike RNNs, Transformers process all words simultaneously. Without positional encoding, 
    the model wouldn't know the order of words - "dog bites man" vs "man bites dog" would look the same!
    """)
    
    # Interactive demonstration
    st.subheader("🎮 Word Order Matters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Sentence**")
        sentence = "The quick brown fox jumps"
        st.info(sentence)
        st.success("✓ Meaningful sentence")
    
    with col2:
        st.markdown("**Without Position Info**")
        words = sentence.split()
        import random
        shuffled = words.copy()
        random.shuffle(shuffled)
        st.info(" ".join(shuffled))
        st.error("✗ Loses meaning!")
    
    # Visualize positional encodings
    st.subheader("Positional Encoding Visualization")
    
    import numpy as np
    import plotly.graph_objects as go
    
    def get_positional_encoding(seq_len, d_model):
        """Generate sinusoidal positional encoding"""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    seq_length = st.slider("Sequence Length", 5, 50, 20)
    d_model = 64  # Embedding dimension
    
    pos_enc = get_positional_encoding(seq_length, d_model)
    
    fig = go.Figure(data=go.Heatmap(
        z=pos_enc.T,
        colorscale='RdBu',
        x=list(range(seq_length)),
        y=list(range(d_model))
    ))
    
    fig.update_layout(
        title="Positional Encoding Pattern",
        xaxis_title="Position in Sequence",
        yaxis_title="Embedding Dimension",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### How It Works
    
    - Uses sine and cosine functions of different frequencies
    - Each position gets a unique encoding pattern
    - Added to word embeddings before processing
    - Allows model to learn relative positions between words
    """)


def feedforward_tab():
    st.header("Feed-Forward Network")
    
    st.markdown("""
    ### The Processing Layer
    
    After attention mechanisms identify important relationships, feed-forward networks 
    process this information independently for each position.
    """)
    
    # Interactive demonstration
    st.subheader("🎮 Interactive FFN Processing")
    
    st.markdown("**Architecture**: Input → Dense Layer → Activation (ReLU) → Dense Layer → Output")
    
    # Simulate a simple feed-forward pass
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Input**")
        input_dim = st.slider("Input dimensions", 64, 512, 256, 64)
        st.metric("Vector size", input_dim)
    
    with col2:
        st.markdown("**Hidden Layer**")
        hidden_dim = st.slider("Hidden dimensions", 256, 2048, 1024, 256)
        st.metric("Expanded size", hidden_dim)
        expansion = hidden_dim / input_dim
        st.caption(f"Expansion ratio: {expansion:.1f}x")
    
    with col3:
        st.markdown("**Output**")
        st.metric("Vector size", input_dim)
        st.caption("(back to original size)")
    
    # Visualize the expansion and contraction
    import plotly.graph_objects as go
    
    layers = ['Input', 'Hidden', 'Output']
    sizes = [input_dim, hidden_dim, input_dim]
    
    fig = go.Figure(data=[
        go.Bar(
            x=layers,
            y=sizes,
            marker_color=['lightblue', 'lightcoral', 'lightgreen'],
            text=sizes,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Feed-Forward Network Dimensions",
        yaxis_title="Vector Dimension",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key characteristics
    st.markdown("""
    ### Key Characteristics
    
    1. **Position-wise**: Processes each position independently
    2. **Expansion**: Typically expands to 4x the input dimension
    3. **Non-linearity**: Uses ReLU or GELU activation
    4. **Bottleneck**: Projects back to original dimension
    
    ### Purpose
    
    - Transforms the attention output
    - Adds non-linear processing capability
    - Allows the model to learn complex patterns
    - Applied identically to each position
    """)
    
    # Simple code example
    with st.expander("💻 Code Example"):
        st.code("""
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = self.linear1(x)      # Expand
        x = self.activation(x)    # Non-linearity
        x = self.dropout(x)       # Regularization
        x = self.linear2(x)      # Project back
        return x

# Example usage
d_model = 512
d_ff = 2048
ffn = FeedForward(d_model, d_ff)
        """, language="python")


if __name__ == "__main__":
    try:
        if 'localhost' in st.context.headers["host"]:
            main()
        else:
            is_authenticated = authenticate.login()
            
            if is_authenticated:
                main()

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        
        with st.expander("Error Details"):
            st.code(str(e))

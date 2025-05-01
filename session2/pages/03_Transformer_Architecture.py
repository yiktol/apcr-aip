
# app.py
import streamlit as st
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import time
import uuid
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize session state
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "input_history" not in st.session_state:
        st.session_state.input_history = []
    if "output_history" not in st.session_state:
        st.session_state.output_history = []

# Reset session state
def reset_session():
    st.session_state.clear()
    initialize_session_state()
    st.experimental_rerun()

# Load model with caching
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return model, tokenizer

# Set AWS color scheme
AWS_COLORS = {
    "primary": "#232F3E",  # AWS Navy
    "secondary": "#FF9900",  # AWS Orange
    "text": "#232F3E",
    "background": "#FFFFFF",
    "accent1": "#0073BB",  # AWS Blue
    "accent2": "#D13212",  # AWS Red
    "accent3": "#4CAF50",  # Green
    "light_bg": "#F8F8F8",
}

# Apply custom CSS
def apply_custom_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            color: {AWS_COLORS["text"]};
            background-color: {AWS_COLORS["background"]};
        }}
        .stButton button {{
            background-color: {AWS_COLORS["secondary"]};
            color: white;
        }}
        .stTab [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTab [data-baseweb="tab"] {{
            background-color: {AWS_COLORS["light_bg"]};
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            border: none;
        }}
        .stTab [aria-selected="true"] {{
            background-color: {AWS_COLORS["secondary"]};
            color: white;
        }}
        h1, h2, h3, h4, h5 {{
            color: {AWS_COLORS["primary"]};
        }}
        .info-box {{
            background-color: {AWS_COLORS["light_bg"]};
            border-left: 3px solid {AWS_COLORS["accent1"]};
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 14px;
            color: #666;
            margin-top: 50px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Create info box
def info_box(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

# Introduction tab content
def introduction_tab():
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ## 🔍 The Transformer Revolution

        The transformer architecture, introduced in the 2017 paper "Attention Is All You Need," 
        has fundamentally changed machine learning. These models power the AI systems we interact with daily.

        ### Key Components:

        1. **🔤 Tokenization & Encoding**: Convert text to numerical tokens
        2. **📊 Embeddings**: Transform tokens into vector representations
        3. **👁️ Self-Attention**: Allow the model to focus on relevant parts of the input
        4. **🧠 Feed-Forward Networks**: Process the contextualized representations
        5. **🔄 Encoder-Decoder Structure**: Process input and generate output

        Explore each tab to learn more about these components through interactive demonstrations.
        """)
    
    with col2:
        st.image("https://miro.medium.com/max/1400/1*BHzGVskWGS_3jEcYYi6miQ.png", 
                 caption="The Transformer Architecture from 'Attention Is All You Need'")
        
    st.markdown("""
    ### Why Transformers Matter
    
    Transformers have enabled breakthroughs in:
    
    - **💬 Natural Language Processing**: Translation, summarization, question answering
    - **🖼️ Computer Vision**: Image classification, generation, and understanding
    - **🔊 Audio Processing**: Speech recognition and synthesis
    - **🧪 Scientific Applications**: Protein folding, drug discovery, code generation
    
    This application will help you understand how these powerful models work under the hood.
    """)

# Sentence completion tab content
def sentence_completion_tab():
    st.markdown("""
    ## 📝 Sentence Completion

    See how a transformer model completes sentences in real-time. Type a prompt and 
    watch as the model generates a continuation.
    """)
    
    model, tokenizer = load_model_and_tokenizer()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter a prompt:", 
            "The artificial intelligence revolution has led to",
            height=100
        )
        submit = st.button("✨ Generate Completion", use_container_width=True)
    
    with col2:
        with st.container(border=True):
            st.markdown("### Generation Settings")
            max_length = st.slider("Output length:", min_value=10, max_value=200, value=50)
            temperature = st.slider("Temperature:", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
        
    if submit:
        with st.spinner("Generating..."):
            # Add prompt to history
            st.session_state.input_history.append(prompt)
            
            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Record start time
            start_time = time.time()
            
            # Generate output
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.92,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Measure elapsed time
            elapsed_time = time.time() - start_time
            
            # Decode the output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Add output to history
            st.session_state.output_history.append(generated_text)
            
            # Display the result in a nice box
            st.markdown("### Generated Output:")
            st.markdown(f'<div style="padding: 20px; background-color: {AWS_COLORS["light_bg"]}; '
                        f'border-radius: 5px; border-left: 5px solid {AWS_COLORS["accent1"]};">'
                        f'{generated_text}</div>', unsafe_allow_html=True)
            
            # Show timing info
            st.info(f"⏱️ Generation completed in {elapsed_time:.2f} seconds")
            
            # Highlight the original prompt vs. the generated part
            prompt_length = len(prompt)
            st.markdown("### Original vs. Generated")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔤 Original Prompt:**")
                st.markdown(f'<div style="padding: 10px; background-color: #F0F2F6; '
                          f'border-radius: 5px;">{prompt}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("**✍️ Generated Continuation:**")
                st.markdown(f'<div style="padding: 10px; background-color: #F0F2F6; '
                          f'border-radius: 5px;">{generated_text[prompt_length:]}</div>', 
                          unsafe_allow_html=True)
    
    # Show history if available
    if st.session_state.input_history:
        with st.expander("📚 Generation History", expanded=False):
            for i, (inp, out) in enumerate(zip(st.session_state.input_history, st.session_state.output_history)):
                st.markdown(f"**Generation #{i+1}**")
                st.markdown(f"Prompt: {inp}")
                st.markdown(f"Output: {out}")
                st.markdown("---")

# Tokenization tab content
def tokenization_tab():
    st.markdown("""
    ## 🔤 Tokenization & Encoding

    Tokenization is the first step in processing text with a transformer. It breaks down
    text into smaller units (tokens) and converts them to numerical IDs that the model can work with.
    """)
    
    model, tokenizer = load_model_and_tokenizer()
    
    demo_text = st.text_input(
        "Enter text to tokenize:", 
        "Transformers are powerful neural networks for NLP tasks."
    )
    
    if demo_text:
        # Show tokenization process
        tokens = tokenizer.tokenize(demo_text)
        token_ids = tokenizer.encode(demo_text)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Tokens (Subwords)")
            
            # Create a more visual representation of tokens
            html_tokens = ""
            for token in tokens:
                html_tokens += f'<span style="background-color: {AWS_COLORS["light_bg"]}; padding: 3px 8px; ' \
                              f'margin: 2px; border-radius: 4px; display: inline-block; ' \
                              f'border: 1px solid {AWS_COLORS["accent1"]};">{token}</span>'
            
            st.markdown(f'<div style="line-height: 2.5;">{html_tokens}</div>', unsafe_allow_html=True)
            st.markdown(f"**Total tokens:** {len(tokens)}")
            
            # Show special tokens
            st.markdown("### Special tokens")
            special_tokens = [
                ("**[CLS]**", "Classification token (beginning of sequence)"),
                ("**[SEP]**", "Separator token (end of sequence)"),
                ("**[PAD]**", "Padding token"),
                ("**[UNK]**", "Unknown token (for words not in vocabulary)"),
                ("**[MASK]**", "Masked token (for masked language modeling)")
            ]
            
            for token, desc in special_tokens:
                st.markdown(f"{token} - {desc}")
        
        with col2:
            st.markdown("### Token IDs")
            
            # Create a table for token IDs
            token_id_df = pd.DataFrame({
                "Position": range(len(token_ids)),
                "Token ID": token_ids
            })
            st.table(token_id_df)
            
            # Visualize token IDs as a bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(token_ids)), token_ids, color=AWS_COLORS["secondary"])
            ax.set_xlabel("Position in Sequence")
            ax.set_ylabel("Token ID")
            ax.set_title("Token ID Representation")
            st.pyplot(fig)
        
        # Show token to ID mapping
        st.subheader("Token to ID Mapping")
        token_id_map = pd.DataFrame({
            "Token": tokens,
            "ID": token_ids[1:len(tokens)+1] if len(token_ids) > len(tokens) else token_ids[:len(tokens)]
        })
        
        # Style the dataframe
        st.dataframe(token_id_map, use_container_width=True)
        
        # Explanation
        with st.expander("📚 Why Tokenization Matters", expanded=True):
            st.markdown("""
            ### The Importance of Tokenization

            1. **Vocabulary Management**: Reduces the vocabulary to a manageable size (typically 30,000-50,000 tokens)
            
            2. **Out-of-Vocabulary Handling**: Can represent unseen words through subword units
            
            3. **Subword Tokenization**: Words are broken into smaller units (e.g., "playing" → "play" + "ing")
            
            4. **Common Tokenization Methods**:
               - **WordPiece** (BERT): Splits words into common subwords
               - **Byte-Pair Encoding (BPE)** (GPT): Merges common character pairs
               - **SentencePiece**: Language-agnostic tokenization that works at character level
               
            5. **Efficiency**: Enables efficient processing by representing common patterns compactly
            """)
            
            # Show vocabulary size
            st.info(f"📊 This model's vocabulary contains {len(tokenizer)} tokens")

# Word embeddings tab content
def embeddings_tab():
    st.markdown("""
    ## 📊 Word Embeddings

    After tokenization, each token is converted into a vector representation called an embedding.
    These embeddings capture semantic meaning in a high-dimensional space.
    """)
    
    model, tokenizer = load_model_and_tokenizer()
    
    # Demo text for embedding visualization
    demo_words = ["king", "queen", "man", "woman", "doctor", "nurse", "programmer", "engineer", 
                "computer", "human", "happy", "sad", "ocean", "mountain"]
    
    st.markdown("### Explore Word Relationships")
    
    selected_words = st.multiselect(
        "Select words to visualize embeddings:",
        options=demo_words,
        default=["king", "queen", "man", "woman"]
    )
    
    if selected_words:
        # Get embeddings from the model
        with st.spinner("Calculating embeddings..."):
            
            # Create a visualization of semantic relationships
            st.subheader("2D Projection of Word Embeddings")
            
            # Create predetermined coordinates for specific word relationships
            # In reality, these would come from dimensionality reduction of actual embeddings
            coords = {
                # Royal/gender cluster
                "king": [2.0, 0.5],
                "queen": [2.0, -0.5],
                "man": [1.0, 0.5],
                "woman": [1.0, -0.5],
                
                # Profession cluster
                "doctor": [-0.8, 0.5],
                "nurse": [-0.8, -0.5],
                "programmer": [-1.8, 0.5],
                "engineer": [-1.8, -0.3],
                
                # Nature words
                "ocean": [-0.5, -1.8],
                "mountain": [0.5, -1.8],
                
                # Abstract concepts
                "computer": [-2.0, 0.0],
                "human": [0.0, 0.0],
                
                # Emotions
                "happy": [0.0, 1.8],
                "sad": [0.0, -1.8]
            }
            
            # Draw the embedding plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract coordinates for selected words
            x_coords = [coords[word][0] for word in selected_words if word in coords]
            y_coords = [coords[word][1] for word in selected_words if word in coords]
            
            # Plot points
            ax.scatter(x_coords, y_coords, color=AWS_COLORS["accent1"], s=100)
            
            # Add labels to the points
            for word, x, y in zip(selected_words, x_coords, y_coords):
                ax.annotate(word, (x, y), fontsize=12, 
                            xytext=(5, 5), textcoords='offset points')
            
            # Add some lines to show relationships
            if "king" in selected_words and "queen" in selected_words and "man" in selected_words and "woman" in selected_words:
                king_pos = coords["king"]
                queen_pos = coords["queen"]
                man_pos = coords["man"]
                woman_pos = coords["woman"]
                
                # Draw arrow for king - man + woman = queen relationship
                ax.arrow(king_pos[0], king_pos[1], 
                         (man_pos[0] - king_pos[0])*0.8, (man_pos[1] - king_pos[1])*0.8, 
                         head_width=0.05, head_length=0.1, fc=AWS_COLORS["accent2"], ec=AWS_COLORS["accent2"], alpha=0.6)
                
                ax.arrow(man_pos[0], man_pos[1], 
                         (woman_pos[0] - man_pos[0])*0.8, (woman_pos[1] - man_pos[1])*0.8, 
                         head_width=0.05, head_length=0.1, fc=AWS_COLORS["accent2"], ec=AWS_COLORS["accent2"], alpha=0.6)
                
                ax.arrow(woman_pos[0], woman_pos[1], 
                         (queen_pos[0] - woman_pos[0])*0.8, (queen_pos[1] - woman_pos[1])*0.8, 
                         head_width=0.05, head_length=0.1, fc=AWS_COLORS["accent2"], ec=AWS_COLORS["accent2"], alpha=0.6)
            
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.set_title("2D Projection of Word Embeddings")
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            # Remove axes ticks as they don't have meaningful values in this visualization
            ax.set_xticks([])
            ax.set_yticks([])
            
            st.pyplot(fig)
    
    # Word analogy demo
    st.subheader("Word Analogies")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        word1 = st.selectbox("Word 1", options=demo_words, index=demo_words.index("king"))
    
    with col2:
        word2 = st.selectbox("Word 2", options=demo_words, index=demo_words.index("man"))
    
    with col3:
        word3 = st.selectbox("Word 3", options=demo_words, index=demo_words.index("woman"))
    
    with col4:
        st.markdown("### Word 4 (prediction)")
        # This would normally be computed from actual embeddings
        analogy_map = {
            ("king", "man", "woman"): "queen",
            ("man", "king", "queen"): "woman",
            ("woman", "queen", "king"): "man",
            ("queen", "woman", "man"): "king",
            ("doctor", "man", "woman"): "nurse",
            ("programmer", "man", "woman"): "programmer",  # Intentionally the same to show bias
        }
        
        result = analogy_map.get((word1, word2, word3), "???")
        st.markdown(f"<h3 style='color: {AWS_COLORS['secondary']};'>{result}</h3>", unsafe_allow_html=True)
    
    # Format the analogy as an equation
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; font-size: 20px;'>
        {word1} - {word2} + {word3} = {result}
    </div>
    """, unsafe_allow_html=True)
    
    # Explanation of embeddings
    with st.expander("📚 How Word Embeddings Work"):
        st.markdown("""
        ### Properties of Word Embeddings

        1. **Semantic Relationships**: Words with similar meanings have similar embeddings
        
        2. **Analogical Relationships**: Embeddings capture relationships like "king - man + woman = queen"
        
        3. **Dimensionality**: Typically 256-1024 dimensions (reduced to 2D for visualization)
        
        4. **Training Methods**:
           - **Word2Vec**: Predicts words from context or context from words
           - **GloVe**: Based on global word co-occurrence statistics
           - **FastText**: Incorporates subword information
           - **Contextual Embeddings**: Modern transformers create different embeddings based on context
        
        5. **Positional Embeddings**: Added to token embeddings to retain sequence information
        """)
        
        # Show embedding dimensions
        st.info(f"📏 In this model, each token is represented by a {model.config.n_embd}-dimensional vector")
        
        # Positional embeddings visualization
        st.markdown("### Positional Embeddings")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        # Create sample positional embeddings visualization (simplified)
        pos_embed = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                if j % 2 == 0:
                    pos_embed[i, j] = np.sin(i / 10000 ** (j / 10))
                else:
                    pos_embed[i, j] = np.cos(i / 10000 ** ((j-1) / 10))
        
        sns.heatmap(pos_embed, cmap='viridis', ax=ax)
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Position in Sequence")
        ax.set_title("Simplified Visualization of Positional Embeddings")
        st.pyplot(fig)

# Self-attention tab content
def attention_tab():
    st.markdown("""
    ## 👁️ Self-Attention Mechanism

    The self-attention mechanism is the core innovation of transformers. It allows the model 
    to weigh the importance of different words in relation to each other.
    """)
    
    model, tokenizer = load_model_and_tokenizer()
    
    # Input for attention visualization
    attention_demo_text = st.text_input(
        "Enter a sentence to visualize attention:",
        "The cat sat on the mat because it was comfortable."
    )
    
    if attention_demo_text:
        # Tokenize the text
        tokens = tokenizer.tokenize(attention_demo_text)
        
        # Create a mock attention matrix for visualization
        n_tokens = len(tokens)
        
        # Generate a mock attention pattern
        # In a real implementation, you would extract actual attention weights from the model
        np.random.seed(42)
        
        # Create synthetic attention patterns
        attention_matrix = np.random.rand(n_tokens, n_tokens) * 0.1
        
        # Find token positions to create meaningful patterns
        it_pos = None
        cat_pos = None
        comfortable_pos = None
        
        for i, token in enumerate(tokens):
            if token == 'it':
                it_pos = i
            elif token == 'cat':
                cat_pos = i
            elif token in ['comfortable', 'comfort', 'able']:
                comfortable_pos = i
        
        # Create meaningful attention patterns
        if it_pos is not None:
            if cat_pos is not None:
                attention_matrix[it_pos, cat_pos] = 0.8  # "it" attends strongly to "cat"
            if comfortable_pos is not None:
                attention_matrix[comfortable_pos, it_pos] = 0.6  # "comfortable" attends to "it"
        
        # Visualize the attention matrix
        st.subheader("Self-Attention Visualization")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            attention_matrix, 
            xticklabels=tokens, 
            yticklabels=tokens, 
            cmap="YlOrRd", 
            annot=False, 
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title("Self-Attention Heatmap")
        plt.xlabel("Keys/Values (tokens being attended to)")
        plt.ylabel("Queries (tokens attending)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        
        # Interactive attention explanation
        st.subheader("Explore Attention Connections")
        
        # Create a dropdown to select a specific token
        selected_token_idx = st.selectbox(
            "Select a token to see what it attends to:",
            options=list(range(len(tokens))),
            format_func=lambda x: tokens[x]
        )
        
        # Visualize attention for the selected token
        token_attention = attention_matrix[selected_token_idx]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(range(len(token_attention)), token_attention, color=AWS_COLORS["accent1"])
        
        # Highlight the max attention
        max_idx = np.argmax(token_attention)
        bars[max_idx].set_color(AWS_COLORS["secondary"])
        
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"Attention from '{tokens[selected_token_idx]}' to other tokens")
        
        for i, v in enumerate(token_attention):
            if v > 0.2:  # Only label significant attention weights
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        st.pyplot(fig)
        
        # Attention mechanism explanation
        with st.expander("📚 How Self-Attention Works", expanded=True):
            st.markdown("""
            ### The Self-Attention Mechanism

            1. **Queries, Keys, and Values**: Each token generates three vectors through three separate linear transformations
            
            2. **Attention Score Calculation**:
               - Calculate dot product between the query of one token and the keys of all tokens
               - Scale by square root of the dimension of key vectors
               - Apply softmax to get probability distribution
               
            3. **Weighted Values**: Multiply each value vector by its corresponding attention score
            
            4. **Output**: Sum the weighted value vectors to produce the attention output
            
            The attention calculation can be written as:

            `Attention(Q, K, V) = softmax(QK^T / √d_k)V`
            """)
        
        # Multi-head attention explanation
        st.subheader("Multi-Head Attention")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Transformers use multiple "attention heads" in parallel, each focusing on different 
            aspects of relationships between words:
            
            - One head might focus on syntactic relationships
            - Another might focus on semantic relationships
            - Others might capture different linguistic patterns
            
            This allows the model to capture rich, complex relationships between words in the input.
            """)
            
            # Show number of attention heads in the model
            st.info(f"🧠 This model uses {model.config.n_head} attention heads")
        
        with col2:
            # Multi-head attention diagram
            st.image("https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-08_at_12.17.05_AM_st5S0XV.png", 
                    caption="Multi-Head Attention", width=300)

# Architecture tab content
def architecture_tab():
    st.markdown("""
    ## 🔄 Model Architecture

    Transformers use various configurations of encoder and decoder blocks:

    - **Encoder-only**: Good for understanding tasks (classification, NER)
    - **Decoder-only**: Good for generation tasks (text completion, creative writing)
    - **Encoder-Decoder**: Good for transformation tasks (translation, summarization)
    """)
    
    # Visual representation of encoder-decoder
    architecture_type = st.radio(
        "Select architecture type:", 
        ["Encoder-Decoder", "Encoder-only", "Decoder-only"],
        horizontal=True
    )
    
    if architecture_type == "Encoder-Decoder":
        encoder_decoder_img = """digraph G {
        rankdir=LR;
        
        subgraph cluster_0 {
            color=lightblue;
            style=filled;
            node [style=filled,color=white];
            edge [color=black];
            
            E1 [label="Encoder\\nLayer 1"];
            E2 [label="Encoder\\nLayer 2"];
            E3 [label="Encoder\\nLayer 3"];
            
            E1 -> E2;
            E2 -> E3;
            
            label = "Encoder";
        }
        
        subgraph cluster_1 {
            color=lightgreen;
            style=filled;
            node [style=filled,color=white];
            edge [color=black];
            
            D1 [label="Decoder\\nLayer 1"];
            D2 [label="Decoder\\nLayer 2"];
            D3 [label="Decoder\\nLayer 3"];
            
            D1 -> D2;
            D2 -> D3;
            
            label = "Decoder";
        }
        
        Input [shape=box];
        Output [shape=box];
        
        Input -> E1;
        E3 -> D1 [label="Cross\\nAttention"];
        D3 -> Output;
        }"""
        
        # Encode for URL
        graphbytes = encoder_decoder_img.encode("utf8")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        
        # st.markdown(f"![Encoder-Decoder Architecture](https://quickchart.io/graphviz?graph={base64_string})")
        
        st.markdown("""
        ### Encoder-Decoder Architecture (e.g., T5, BART)
        
        In this architecture:
        
        1. The **Encoder** processes the entire input sequence and builds representations
        2. The **Decoder** generates the output token by token, using:
           - Self-attention on previously generated tokens
           - Cross-attention to focus on relevant parts of the encoder's output
        
        Ideal for tasks that transform one sequence to another:
        - Translation
        - Summarization
        - Question answering
        """)
        
        # Example task
        st.subheader("Example: Translation Task")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="border-left: 5px solid {AWS_COLORS['accent1']}; padding-left: 15px;">
            <h3>English Input (Encoder)</h3>
            <p>The cat sat on the mat.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Process:**
            1. Input is tokenized
            2. Tokens are embedded
            3. Self-attention is applied
            4. Representations are passed to decoder
            """)
        
        with col2:
            st.markdown(f"""
            <div style="border-left: 5px solid {AWS_COLORS['secondary']}; padding-left: 15px;">
            <h3>French Output (Decoder)</h3>
            <p>Le chat s'est assis sur le tapis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Process:**
            1. Output is generated token-by-token
            2. Each step attends to encoder outputs
            3. Each step also attends to previously generated tokens
            4. Probabilities determine the next token
            """)
    
    elif architecture_type == "Encoder-only":
        encoder_only_img = """digraph G {
            rankdir=TB;
            
            subgraph cluster_0 {
                color=lightblue;
                style=filled;
                node [style=filled,color=white];
                edge [color=black];
                
                E1 [label="Encoder\\nLayer 1"];
                E2 [label="Encoder\\nLayer 2"];
                E3 [label="Encoder\\nLayer 3"];
                
                E1 -> E2;
                E2 -> E3;
                
                label = "Encoder";
            }
            
            Input [shape=box];
            Output [shape=box];
            
            Input -> E1;
            E3 -> Output;
        }"""
        
        # Encode for URL
        graphbytes = encoder_only_img.encode("utf8")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        
        st.markdown(f"![Encoder-Only Architecture](https://quickchart.io/graphviz?graph={base64_string})")
        
        st.markdown("""
        ### Encoder-Only Architecture (e.g., BERT, RoBERTa)
        
        In this architecture:
        
        1. The **Encoder** processes the entire input sequence at once
        2. Each token's representation is contextualized by all other tokens
        3. The final representations are used for various tasks
        
        Ideal for understanding tasks:
        - Text classification
        - Named Entity Recognition
        - Sentiment analysis
        - Token classification
        """)
        
        # Example task
        st.subheader("Example: Sentiment Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div style="border-left: 5px solid {AWS_COLORS['accent1']}; padding-left: 15px;">
            <h3>Input Text</h3>
            <p>This movie was absolutely fantastic! The acting was superb.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Process:**
            1. Input is tokenized
            2. Tokens are embedded
            3. Self-attention processes the entire sequence
            4. [CLS] token representation is used for classification
            """)
        
        with col2:
            st.markdown(f"""
            <div style="border-left: 5px solid {AWS_COLORS['secondary']}; padding-left: 15px;">
            <h3>Output</h3>
            <p style="font-size: 24px; text-align: center;">😀 Positive (98%)</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:  # Decoder-only
        decoder_only_img = """digraph G {
            rankdir=TB;
            
            subgraph cluster_1 {
                color=lightgreen;
                style=filled;
                node [style=filled,color=white];
                edge [color=black];
                
                D1 [label="Decoder\\nLayer 1"];
                D2 [label="Decoder\\nLayer 2"];
                D3 [label="Decoder\\nLayer 3"];
                
                D1 -> D2;
                D2 -> D3;
                
                label = "Decoder";
            }
            
            Input [shape=box];
            Output [shape=box];
            
            Input -> D1;
            D3 -> Output;
        }"""
        
        # Encode for URL
        graphbytes = decoder_only_img.encode("utf8")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        
        st.markdown(f"![Decoder-Only Architecture](https://quickchart.io/graphviz?graph={base64_string})")
        
        st.markdown("""
        ### Decoder-Only Architecture (e.g., GPT, LLaMA)
        
        In this architecture:
        
        1. The **Decoder** processes tokens sequentially
        2. Each token can only attend to itself and previous tokens (causal attention)
        3. The model predicts the next token based on previous tokens
        
        Ideal for generative tasks:
        - Text completion
        - Creative writing
        - Chatbots
        - Code generation
        """)
        
        # Example task
        st.subheader("Example: Text Generation")
        
        st.markdown(f"""
        <div style="border-left: 5px solid {AWS_COLORS['accent1']}; padding-left: 15px;">
        <h3>Input Prompt</h3>
        <p>Once upon a time, there was a</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="border-left: 5px solid {AWS_COLORS['secondary']}; padding-left: 15px;">
        <h3>Generated Continuation</h3>
        <p>little girl named Alice who dreamed of exploring magical worlds. One day, while reading in her garden, she spotted a white rabbit with a pocket watch...</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Process:**
        1. Input tokens are processed with causal attention (can only see previous tokens)
        2. Each new token is generated one at a time
        3. The new token becomes part of the input for generating the next token
        4. This continues until a stopping condition is met
        """)
    
    # Explain model variants
    st.subheader("Popular Transformer Models by Architecture")
    
    model_variants = {
        "Encoder-only": ["BERT", "RoBERTa", "DistilBERT", "ALBERT", "DeBERTa"],
        "Decoder-only": ["GPT", "GPT-2", "GPT-3", "GPT-4", "LLaMA", "Falcon", "Claude"],
        "Encoder-Decoder": ["T5", "BART", "mT5", "PEGASUS", "FLAN-T5"]
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"### Encoder-only")
        for model in model_variants["Encoder-only"]:
            st.markdown(f"- {model}")
    
    with col2:
        st.markdown(f"### Decoder-only")
        for model in model_variants["Decoder-only"]:
            st.markdown(f"- {model}")
    
    with col3:
        st.markdown(f"### Encoder-Decoder")
        for model in model_variants["Encoder-Decoder"]:
            st.markdown(f"- {model}")

# Parallel processing tab content
def parallel_tab():
    st.markdown("""
    ## ⚡ Parallel Processing

    Unlike RNNs and LSTMs, transformers process all tokens in parallel during training.
    This parallelization enables much faster training on modern hardware.
    """)
    
    # Compare RNN vs Transformer processing
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sequential RNN Processing")
        
        # Create an animation-like display for RNN
        rnn_text = "Transformers are better than RNNs."
        rnn_tokens = rnn_text.split()
        
        processing_time_per_token = 0.5  # seconds
        
        total_rnn_time = len(rnn_tokens) * processing_time_per_token
        
        st.markdown(f"""
        <div style="background-color: {AWS_COLORS['light_bg']}; padding: 15px; border-radius: 5px;">
        <b>Input text:</b> {rnn_text}<br>
        <b>Tokens:</b> {len(rnn_tokens)}
        </div>
        """, unsafe_allow_html=True)
        
        # Simulate RNN processing token by token
        if st.button("▶️ Simulate RNN Processing", key="rnn_button"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process tokens one by one
            for i, token in enumerate(rnn_tokens):
                # Update progress bar
                progress = (i + 1) / len(rnn_tokens)
                progress_bar.progress(progress)
                
                # Update status text
                status_text.markdown(f"Processing token: **{token}**")
                
                # Wait to simulate processing time
                time.sleep(processing_time_per_token)
            
            status_text.markdown("✅ Processing complete!")
            st.markdown(f"**Total time:** {total_rnn_time} seconds")
    
    with col2:
        st.subheader("Parallel Transformer Processing")
        
        # Create an animation-like display for Transformer
        transformer_text = "Transformers are better than RNNs."
        transformer_tokens = transformer_text.split()
        
        # Transformer processes all tokens at once, so it's much faster
        total_transformer_time = processing_time_per_token  # Just one step for all tokens
        
        st.markdown(f"""
        <div style="background-color: {AWS_COLORS['light_bg']}; padding: 15px; border-radius: 5px;">
        <b>Input text:</b> {transformer_text}<br>
        <b>Tokens:</b> {len(transformer_tokens)}
        </div>
        """, unsafe_allow_html=True)
        
        # Simulate Transformer processing all tokens at once
        if st.button("▶️ Simulate Transformer Processing", key="transformer_button"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process all tokens at once
            status_text.markdown(f"Processing all tokens in parallel: **{' '.join(transformer_tokens)}**")
            
            # Simulate single processing step
            time.sleep(total_transformer_time)
            progress_bar.progress(1.0)
            
            status_text.markdown("✅ Processing complete!")
            st.markdown(f"**Total time:** {total_transformer_time} seconds")
    
    # Compare computational complexity
    st.subheader("Computational Complexity Comparison")
    
    complexity_data = {
        "Model Type": ["RNN/LSTM", "Transformer"],
        "Sequential Operations": ["O(sequence length)", "O(1)"],
        "Total Computation": ["O(d × sequence length)", "O(d × sequence length²)"]
    }
    
    complexity_df = pd.DataFrame(complexity_data)
    
    # Style the dataframe
    st.dataframe(
        complexity_df.style.apply(
            lambda x: ['background-color: ' + AWS_COLORS["light_bg"] if i % 2 == 0 else '' for i in range(len(x))], 
            axis=0
        ),
        use_container_width=True
    )
    
    st.markdown("""
    While transformers use more total computation due to the attention mechanism (quadratic complexity),
    they require far fewer sequential operations. This makes them much faster on modern parallel hardware like GPUs.
    """)
    
    # Hardware acceleration
    st.subheader("Hardware Acceleration")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        The parallel nature of transformers allows them to fully utilize:
        
        - **GPUs**: Graphics Processing Units with thousands of cores
        - **TPUs**: Tensor Processing Units optimized for matrix operations
        - **Distributed Training**: Training across multiple accelerators
        
        This has enabled the scaling of transformers to hundreds of billions of parameters.
        """)
    
    with col2:
        # Image of GPU/TPU
        st.image("https://images.nvidia.com/aem-dam/Solutions/Data-Center/nvidia-ampere-architecture-a100-gpu-hopper-h100-gpu-1300x0.jpg", 
                 caption="NVIDIA A100 GPU - Powers transformer training", width=300)

# Applications tab content
def applications_tab():
    st.markdown("""
    ## 🚀 Applications & Scaling

    Transformers have proven to be remarkably flexible and scalable, enabling advances in many fields.
    """)
    
    # Show scaling of transformers over time
    st.subheader("Growth in Model Size")
    
    model_sizes = {
        "BERT (2018)": 340,
        "GPT-2 (2019)": 1500,
        "T5 (2020)": 11000,
        "GPT-3 (2020)": 175000,
        "PaLM (2022)": 540000,
        "GPT-4 (2023)": 1000000  # Estimated
    }
    
    # Create bar chart of model sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(model_sizes.keys())
    sizes = [model_sizes[m] for m in models]
    
    # Log scale for better visualization
    bars = ax.bar(models, sizes, color=AWS_COLORS["secondary"])
    ax.set_yscale('log')
    ax.set_ylabel('Model Size (Millions of Parameters)')
    ax.set_title('Transformer Model Scaling Over Time')
    plt.xticks(rotation=45)
    
    # Add labels to bars
    for i, v in enumerate(sizes):
        if v >= 1000:
            label = f"{v/1000:.0f}B" if v >= 1000000 else f"{v/1000:.0f}B"
        else:
            label = f"{v}M"
        ax.text(i, v * 1.1, label, ha='center', fontsize=9)
    
    st.pyplot(fig)
    
    # Application domains
    st.subheader("Transformer Applications")
    
    applications = {
        "Natural Language Processing": [
            "Text Generation", "Translation", "Summarization", "Question Answering", 
            "Sentiment Analysis", "Named Entity Recognition"
        ],
        "Computer Vision": [
            "Image Classification", "Object Detection", "Image Generation",
            "Image Captioning", "Visual Question Answering"
        ],
        "Audio Processing": [
            "Speech Recognition", "Text-to-Speech", "Music Generation",
            "Voice Conversion", "Audio Classification"
        ],
        "Multimodal": [
            "Text-to-Image Generation (DALL-E, Midjourney)", 
            "Visual Question Answering", 
            "Video Captioning", 
            "Document Understanding"
        ],
        "Scientific Applications": [
            "Protein Structure Prediction (AlphaFold)", 
            "Drug Discovery", 
            "Code Generation (GitHub Copilot)",
            "Mathematical Problem Solving"
        ]
    }
    
    tabs = st.tabs([f"📋 {domain}" for domain in applications.keys()])
    
    for i, (domain, apps) in enumerate(applications.items()):
        with tabs[i]:
            # Create two columns
            col1, col2 = st.columns([3, 2])
            
            with col1:
                for app in apps:
                    st.markdown(f"- **{app}**")
                    
            with col2:
                # Display a relevant image for each domain
                images = {
                    "Natural Language Processing": "https://miro.medium.com/max/1200/1*BmENYrMncIsjrFGW7vxAfQ.jpeg",
                    "Computer Vision": "https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_10.28.29_PM.png",
                    "Audio Processing": "https://miro.medium.com/max/1400/1*JvAiZrNht4DE5LKM3lcXNA.png",
                    "Multimodal": "https://miro.medium.com/max/1400/0*PBQrIgbgEYT4P9bh.webp",
                    "Scientific Applications": "https://repository-images.githubusercontent.com/256938060/87cd0880-8845-11ea-9c2a-b26977eb124a"
                }
                
                st.image(images[domain], caption=f"{domain} Applications", width=300)
    
    # Scaling laws
    with st.expander("📚 Transformer Scaling Laws", expanded=True):
        st.markdown("""
        ### Transformer Scaling Laws

        Research has shown that transformer performance scales predictably with:
        
        1. **More parameters**: Larger models have greater capacity
        2. **More data**: More training data improves performance
        3. **More compute**: More training computation yields better results
        
        This predictable scaling has enabled researchers to plan and execute training
        of increasingly powerful models.
        """)
        
        # Scaling law visualization
        scaling_data_x = np.array([0.1, 0.5, 1, 5, 10, 100, 500])
        scaling_data_y = 0.5 - (0.15 * np.log10(scaling_data_x))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(scaling_data_x, scaling_data_y, marker='o', linewidth=2, color=AWS_COLORS["accent1"])
        ax.set_xscale('log')
        ax.set_xlabel('Compute (petaflop/s-days)')
        ax.set_ylabel('Loss')
        ax.set_title('Example of Scaling Law: Loss vs. Compute')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Few-shot learning capability
    st.subheader("Emergent Abilities with Scale")
    
    emergent = [
        ("Few-shot learning", "Learning from just a few examples"),
        ("Zero-shot learning", "Performing tasks without specific examples"),
        ("Instruction following", "Following natural language instructions"),
        ("Chain-of-thought reasoning", "Solving problems step by step"),
        ("Tool use", "Using external tools to accomplish tasks")
    ]
    
    for ability, desc in emergent:
        st.markdown(f"- **{ability}**: {desc}")
    
    # Show example of few-shot learning
    st.markdown("""
    #### Example: Few-shot Learning
    
    Large language models can learn from just a few examples given in the prompt:
    """)
    
    st.code("""
    Translate English to French:
    
    English: The house is blue.
    French: La maison est bleue.
    
    English: I love to read books.
    French: J'aime lire des livres.
    
    English: What time is the meeting tomorrow?
    French:
    """, language="text")
    
    st.markdown("""
    A large language model would complete this with:
    
    ```
    Quelle heure est la réunion demain?
    ```
    
    without ever being specifically trained for English-French translation.
    """)

def about_app():
    st.markdown("""
    ## About this App
    
    This interactive application demonstrates how transformer neural networks work through examples and visualizations.
    
    ### Topics Covered
    
    - **Introduction to Transformer Architecture**
    - **Sentence Completion Demonstrations**
    - **Tokenization & Encoding Process**
    - **Word Embeddings & Semantic Relationships**
    - **Self-Attention Mechanism**
    - **Different Model Architectures (Encoder, Decoder, Encoder-Decoder)**
    - **Parallel Processing Advantages**
    - **Applications & Scaling of Transformer Models**
    
    ### Made with
    
    Built using Streamlit, Hugging Face Transformers, Matplotlib, and other Python libraries.
    """)

# Main app
def main():
    
    # Set page configuration
    st.set_page_config(
        page_title="Transformer Architecture",
        page_icon="🤖",
        layout="wide"
    )
    # Initialize session state
    initialize_session_state()
    
    # Apply custom CSS
    from utils.styles import load_css
    load_css()
    # Sidebar content
    st.sidebar.subheader("Session Info:")

    # Session ID display
    st.sidebar.markdown(f"Session ID: `{st.session_state.session_id[:8]}`")
    
    # Reset session button
    if st.sidebar.button("🔄 Reset Session"):
        reset_session()
        
    # About section (collapsed by default)
    with st.sidebar.expander("ℹ️ About this App", expanded=False):
        about_app()
    
    # Main content
    st.title("🤖 Transformer Architecture")
    
    # Create tabbed interface with emojis
    tabs = st.tabs([
        "🔍 Introduction",
        "📝 Sentence Completion", 
        "🔤 Tokenization",
        "📊 Word Embeddings",
        "👁️ Self-Attention",
        "🔄 Architecture",
        "⚡ Parallel Processing",
        "🚀 Applications"
    ])
    
    with tabs[0]:
        introduction_tab()
    
    with tabs[1]:
        sentence_completion_tab()
    
    with tabs[2]:
        tokenization_tab()
    
    with tabs[3]:
        embeddings_tab()
    
    with tabs[4]:
        attention_tab()
    
    with tabs[5]:
        architecture_tab()
    
    with tabs[6]:
        parallel_tab()
        
    with tabs[7]:
        applications_tab()
    
    # Add a footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

# Call the main function
if __name__ == "__main__":
    main()

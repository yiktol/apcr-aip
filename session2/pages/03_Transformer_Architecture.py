
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.styles import load_css, custom_header, create_footer
from utils.common import render_sidebar
import utils.authenticate as authenticate

# Set page configuration
st.set_page_config(
    page_title="Transformer Architecture",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []

# Load model with caching
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# AWS color scheme
AWS_COLORS = {
    "primary": "#232F3E",
    "secondary": "#FF9900",
    "text": "#232F3E",
    "background": "#FFFFFF",
    "accent1": "#0073BB",
    "accent2": "#D13212",
    "accent3": "#4CAF50",
    "light_bg": "#F8F8F8",
}

# Introduction tab
def introduction_tab():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Understanding Transformers
        
        Introduced in 2017's "Attention Is All You Need" paper, Transformers revolutionized AI 
        by replacing sequential processing with **parallel self-attention mechanisms**.
        
        ### Core Innovation: Self-Attention
        
        Instead of processing words one-by-one like RNNs, Transformers:
        - Process all words **simultaneously**
        - Let each word **attend to** all other words
        - Learn which words are important through **attention weights**
        
        ### Key Components
        
        1. **Tokenization**: Break text into subword pieces
        2. **Embeddings**: Convert tokens to vectors + positional information
        3. **Self-Attention**: Dynamically weight word relationships
        4. **Feed-Forward**: Transform representations
        5. **Layer Stacking**: Build deep understanding
        """)
    
    with col2:
        st.info("""
        **Why Transformers Won**
        
        ✅ Parallelizable (faster training)  
        ✅ Long-range dependencies  
        ✅ Scalable to billions of parameters  
        ✅ Transfer learning capable  
        
        **Powers:**
        - GPT (text generation)
        - BERT (understanding)
        - Vision Transformers (images)
        - Whisper (speech)
        """)
        
        # Model architecture types
        st.markdown("### Architecture Types")
        arch_type = st.selectbox(
            "Select type:",
            ["Encoder-Decoder", "Encoder-Only", "Decoder-Only"],
            label_visibility="collapsed"
        )
        
        if arch_type == "Encoder-Decoder":
            st.markdown("**T5, BART**: Translation, summarization")
        elif arch_type == "Encoder-Only":
            st.markdown("**BERT, RoBERTa**: Classification, NER")
        else:
            st.markdown("**GPT, LLaMA**: Text generation")

# Interactive generation tab
def generation_tab():
    st.markdown("## 🎯 Interactive Text Generation")
    
    model, tokenizer = load_model_and_tokenizer()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            "The future of artificial intelligence is",
            height=100
        )
        
        generate_btn = st.button("✨ Generate", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### Settings")
        max_length = st.slider("Length", 20, 100, 50, 10)
        temperature = st.slider("Creativity", 0.3, 1.5, 0.7, 0.1)
        
        st.caption("Lower temperature = more focused")
    
    if generate_btn:
        with st.spinner("Generating..."):
            inputs = tokenizer(prompt, return_tensors="pt")
            
            start_time = time.time()
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.92,
                pad_token_id=tokenizer.eos_token_id
            )
            elapsed = time.time() - start_time
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated[len(prompt):].strip()
            
            # Display result
            st.markdown("### 📝 Result")
            st.markdown(f"""
            <div style='background: {AWS_COLORS["light_bg"]}; padding: 20px; border-radius: 8px; 
                        border-left: 4px solid {AWS_COLORS["accent1"]};'>
                <strong style='color: {AWS_COLORS["primary"]};'>Prompt:</strong> {prompt}<br><br>
                <strong style='color: {AWS_COLORS["secondary"]};'>Generated:</strong> {continuation}
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"⏱️ Generated in {elapsed:.2f}s")
            
            # Save to history
            st.session_state.generation_history.append({
                "prompt": prompt,
                "output": continuation,
                "temperature": temperature
            })

# Tokenization tab
def tokenization_tab():
    st.markdown("## 🔤 Tokenization")
    
    model, tokenizer = load_model_and_tokenizer()
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        text = st.text_input(
            "Enter text to tokenize:",
            "Transformers changed everything!"
        )
        
        if text:
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)
            
            st.markdown("### Token Breakdown")
            
            # Visual token display
            token_html = ""
            for i, (token, tid) in enumerate(zip(tokens, token_ids[1:-1] if len(token_ids) > len(tokens) else token_ids)):
                token_html += f"""<div style="line-height: 3;">
                <span style='background: {AWS_COLORS["light_bg"]}; padding: 8px 12px; margin: 4px; border-radius: 6px; display: inline-block; border: 2px solid {AWS_COLORS["accent1"]};'>
                    <strong>{token}</strong><br>
                    <small style='color: #666;'>ID: {tid}</small>
                </span>
                </div>
                """
            st.markdown(f'{token_html}', unsafe_allow_html=True)
            
            st.metric("Total Tokens", len(tokens))
    
    with col2:
        st.markdown("### Why Subword Tokens?")
        st.info("""
        **Benefits:**
        - Handle unknown words
        - Smaller vocabulary
        - Capture morphology
        
        **Example:**
        - "playing" → "play" + "ing"
        - "unhappiness" → "un" + "happy" + "ness"
        """)
        
        st.markdown(f"**Vocabulary Size:** {len(tokenizer):,} tokens")

# Embeddings visualization tab
def embeddings_tab():
    st.markdown("## 📊 Embeddings & Position")
    
    model, tokenizer = load_model_and_tokenizer()
    
    st.markdown("""
    Embeddings convert discrete tokens into continuous vectors that capture meaning.
    Positional encodings add sequence order information.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Token Embeddings")
        
        # Show embedding dimension
        embedding_dim = model.config.n_embd
        vocab_size = len(tokenizer)
        
        st.metric("Embedding Dimension", embedding_dim)
        st.metric("Vocabulary Size", f"{vocab_size:,}")
        
        # Visualize a sample embedding
        sample_tokens = ["cat", "dog", "king", "queen"]
        
        st.markdown("**Sample Embeddings (first 8 dims):**")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        for token in sample_tokens:
            token_id = tokenizer.encode(token, add_special_tokens=False)[0]
            # Get embedding vector
            embedding = model.transformer.wte.weight[token_id].detach().numpy()[:8]
            ax.plot(embedding, marker='o', label=token, linewidth=2)
        
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Value")
        ax.set_title("Token Embedding Vectors (8D projection)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Positional Encoding")
        
        st.markdown("""
        Since Transformers process all tokens in parallel, we need to inject 
        position information:
        """)
        
        # Visualize positional encoding pattern
        max_len = 50
        d_model = 64
        
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pos_encoding[:20, :20], cmap='coolwarm', ax=ax, cbar=True)
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Position in Sequence")
        ax.set_title("Positional Encoding Pattern")
        st.pyplot(fig)
        
        st.caption("Uses sine/cosine waves at different frequencies")

# Self-attention visualization
def attention_tab():
    st.markdown("## 👁️ Self-Attention Mechanism")
    
    st.markdown("""
    Self-attention lets each word **look at** all other words to understand context.
    """)
    
    model, tokenizer = load_model_and_tokenizer()
    
    # Interactive attention demo
    sentence = st.text_input(
        "Enter a sentence:",
        "The cat sat on the mat because it was comfortable"
    )
    
    if sentence:
        tokens = tokenizer.tokenize(sentence)
        
        # Create synthetic attention pattern
        n = len(tokens)
        attention = np.random.rand(n, n) * 0.1
        
        # Create meaningful patterns
        for i, token in enumerate(tokens):
            if token in ['it', 'Ġit']:
                # Make 'it' attend strongly to 'cat'
                for j, t in enumerate(tokens):
                    if 'cat' in t.lower():
                        attention[i, j] = 0.85
            
            if 'comfortable' in token:
                # Make 'comfortable' attend to 'mat'
                for j, t in enumerate(tokens):
                    if 'mat' in t.lower():
                        attention[i, j] = 0.75
        
        # Normalize rows to sum to 1
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Attention Heatmap")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                attention,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="YlOrRd",
                annot=True,
                fmt=".2f",
                cbar_kws={'label': 'Attention Weight'},
                ax=ax
            )
            ax.set_title("How each word attends to others")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Query a Token")
            
            selected = st.selectbox(
                "Select word:",
                range(len(tokens)),
                format_func=lambda i: tokens[i]
            )
            
            st.markdown(f"**'{tokens[selected]}' attends to:**")
            
            # Show top 3 attended tokens
            attn_weights = attention[selected]
            top_indices = np.argsort(attn_weights)[-3:][::-1]
            
            for idx in top_indices:
                weight = attn_weights[idx]
                st.markdown(f"- **{tokens[idx]}**: {weight:.2%}")
            
            st.markdown("---")
            st.info("""
            **How it works:**
            1. Query: "what am I looking for?"
            2. Key: "what do I contain?"
            3. Value: "what do I communicate?"
            
            Attention = softmax(QK^T/√d) V
            """)
    
    # Multi-head attention explanation
    with st.expander("🧠 Multi-Head Attention"):
        st.markdown(f"""
        This model uses **{model.config.n_head} attention heads** working in parallel.
        
        Each head learns different patterns:
        - Head 1: Syntax (subject-verb agreement)
        - Head 2: Semantics (word meanings)
        - Head 3: Long-range dependencies
        - etc.
        
        Outputs are concatenated and projected.
        """)

# Architecture comparison
def architecture_tab():
    st.markdown("## 🏗️ Transformer Architectures")
    
    arch_type = st.radio(
        "Select architecture:",
        ["Encoder-Only (BERT)", "Decoder-Only (GPT)", "Encoder-Decoder (T5)"],
        horizontal=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if "Encoder-Only" in arch_type:
            st.markdown("""
            ### Encoder-Only (BERT)
            
            **Architecture:**
            - Bidirectional attention (sees full context)
            - Used for understanding tasks
            
            **Best for:**
            - Classification
            - Named Entity Recognition
            - Question Answering
            - Sentiment Analysis
            """)
            
            st.code("""
Input: "The movie was [MASK]"
↓
Encoder Layers (bidirectional)
↓
Output: "great" (91% confidence)
            """, language="text")
            
        elif "Decoder-Only" in arch_type:
            st.markdown("""
            ### Decoder-Only (GPT)
            
            **Architecture:**
            - Causal attention (only sees previous tokens)
            - Autoregressive generation
            
            **Best for:**
            - Text generation
            - Code completion
            - Creative writing
            - Chatbots
            """)
            
            st.code("""
Input: "Once upon a time"
↓
Decoder Layers (causal)
↓
Output: "there was a brave knight..."
            """, language="text")
            
        else:
            st.markdown("""
            ### Encoder-Decoder (T5)
            
            **Architecture:**
            - Encoder: Bidirectional (understanding)
            - Decoder: Causal (generation)
            - Cross-attention between them
            
            **Best for:**
            - Translation
            - Summarization
            - Question Answering
            """)
            
            st.code("""
Input (EN): "Hello, how are you?"
↓
Encoder → Cross-Attention → Decoder
↓
Output (FR): "Bonjour, comment allez-vous?"
            """, language="text")
    
    with col2:
        st.markdown("### Famous Models")
        
        if "Encoder-Only" in arch_type:
            models = ["BERT", "RoBERTa", "DistilBERT", "ALBERT", "DeBERTa"]
        elif "Decoder-Only" in arch_type:
            models = ["GPT-3/4", "LLaMA", "Falcon", "Claude", "Gemini"]
        else:
            models = ["T5", "BART", "FLAN-T5", "mT5", "PEGASUS"]
        
        for model in models:
            st.markdown(f"- {model}")
        
        st.markdown("---")
        
        st.metric("Typical Size", "125M - 175B params")

# Parallel processing comparison
def parallel_tab():
    st.markdown("## ⚡ Why Transformers are Fast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ❌ RNN (Sequential)")
        
        if st.button("▶️ Process with RNN", key="rnn"):
            tokens = ["Transform", "ers", "are", "parallel"]
            progress = st.progress(0)
            status = st.empty()
            
            for i, token in enumerate(tokens):
                status.markdown(f"Step {i+1}: Processing **{token}**...")
                time.sleep(0.5)
                progress.progress((i + 1) / len(tokens))
            
            status.success(f"✅ Done in {len(tokens) * 0.5:.1f}s (sequential)")
        
        st.info("""
        **RNN Issues:**
        - Sequential: Must wait for previous step
        - Slow: Can't use GPU parallelism
        - Memory: Long sequences forget early info
        """)
    
    with col2:
        st.markdown("### ✅ Transformer (Parallel)")
        
        if st.button("▶️ Process with Transformer", key="trans"):
            tokens = ["Transform", "ers", "are", "parallel"]
            progress = st.progress(0)
            status = st.empty()
            
            status.markdown(f"Processing ALL tokens: **{' '.join(tokens)}**")
            time.sleep(0.5)
            progress.progress(1.0)
            
            status.success("✅ Done in 0.5s (parallel)")
        
        st.success("""
        **Transformer Benefits:**
        - Parallel: Process all at once
        - Fast: Full GPU utilization
        - Attention: Direct long-range connections
        """)
    
    # Complexity comparison
    st.markdown("### Computational Complexity")
    
    complexity_df = pd.DataFrame({
        "Operation": ["Process Sequence", "Max Path Length", "Training Speed"],
        "RNN": ["O(n) sequential", "O(n)", "Slow"],
        "Transformer": ["O(1) parallel", "O(1)", "Fast"]
    })
    
    st.dataframe(complexity_df, use_container_width=True, hide_index=True)

# Applications overview
def applications_tab():
    st.markdown("## 🚀 Transformer Applications")
    
    # Show scaling trend
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Model Scaling (2018-2024)")
        
        models = ["BERT\n(2018)", "GPT-2\n(2019)", "GPT-3\n(2020)", "GPT-4\n(2023)", "Gemini\n(2024)"]
        sizes = [0.34, 1.5, 175, 1000, 1000]  # Billions of parameters
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(models, sizes, color=AWS_COLORS["secondary"])
        ax.set_ylabel("Parameters (Billions)")
        ax.set_title("Explosive Growth in Model Size")
        ax.set_yscale('log')
        
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{size}B' if size >= 1 else f'{int(size*1000)}M',
                   ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Key Capabilities")
        
        capabilities = [
            ("Zero-shot", "Tasks without training"),
            ("Few-shot", "Learn from examples"),
            ("Reasoning", "Chain of thought"),
            ("Multimodal", "Text + images + audio"),
            ("Tool use", "APIs, calculators")
        ]
        
        for cap, desc in capabilities:
            st.markdown(f"**{cap}**: {desc}")
    
    # Application domains
    st.markdown("### Application Domains")
    
    domains = {
        "💬 Language": ["ChatGPT", "Translation", "Summarization"],
        "🖼️ Vision": ["DALL-E", "Stable Diffusion", "Image captioning"],
        "🔊 Audio": ["Whisper", "Text-to-speech", "Music generation"],
        "🧬 Science": ["AlphaFold", "Drug discovery", "Protein design"],
        "💻 Code": ["GitHub Copilot", "Code completion", "Bug fixing"]
    }
    
    cols = st.columns(len(domains))
    
    for col, (domain, apps) in zip(cols, domains.items()):
        with col:
            st.markdown(f"### {domain}")
            for app in apps:
                st.markdown(f"- {app}")

# Main app
def main():
    initialize_session_state()
    load_css()
    
    with st.sidebar:
        render_sidebar()
        
        with st.expander("ℹ️ About"):
            st.markdown("""
            Interactive guide to Transformer neural networks.
            
            Covers core concepts:
            - Self-attention
            - Tokenization
            - Embeddings
            - Architecture types
            - Applications
            """)
    
    st.markdown('<h1>🤖 Transformer Architecture</h1>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: {AWS_COLORS["light_bg"]}; padding: 20px; border-radius: 8px; 
                border-left: 4px solid {AWS_COLORS["accent1"]}; margin-bottom: 20px;'>
        The Transformer revolutionized AI through <strong>self-attention</strong>: letting every word 
        attend to every other word in parallel, eliminating sequential processing bottlenecks.
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs([
        "🏠 Introduction",
        "✨ Generation",
        "🔤 Tokenization",
        "📊 Embeddings",
        "👁️ Attention",
        "🏗️ Architecture",
        "⚡ Parallelism",
        "🚀 Applications"
    ])
    
    with tabs[0]:
        introduction_tab()
    
    with tabs[1]:
        generation_tab()
    
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
    
    create_footer()

if __name__ == "__main__":
    try:
        if 'localhost' in st.context.headers.get("host", "localhost"):
            main()
        else:
            is_authenticated = authenticate.login()
            if is_authenticated:
                main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        with st.expander("Error Details"):
            st.code(str(e))

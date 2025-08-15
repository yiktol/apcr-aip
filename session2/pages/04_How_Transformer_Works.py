import streamlit as st
import uuid
from utils.styles import load_css
from utils.knowledge_check import display_knowledge_check
import utils.authenticate as authenticate
import utils.common as common

# Set page configuration
st.set_page_config(
    page_title="How Transformer Works",
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
        <h1>🤖 Transformer Model Sentence Completion</h1>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""<div class="info-box">
Transformer models complete sentences by leveraging their attention mechanisms to analyze existing text patterns and predict the most contextually appropriate words to follow, considering both local and long-range dependencies in the input sequence.
    </div>""",unsafe_allow_html=True)

    
    # Sidebar
    with st.sidebar:
        common.render_sidebar()
        
        # About this App - collapsed by default
        with st.expander("About this App", expanded=False):
            st.write("""
            This interactive application demonstrates how transformer models complete sentences by:
            - Tokenization and Encoding
            - Word Embedding
            - Decoding Process
            - Output Generation
            
            Explore each tab to learn about the different components of transformer models.
            """)
            

            

        
    # Main content - Tab-based navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔤 Tokenization", 
        "🔡 Word Embedding", 
        "🧩 Decoding", 
        "📤 Output Generation",
        "❓ Knowledge Check"
    ])
    
    # Tab 1: Tokenization and Encoding
    with tab1:
        tokenization_tab()
    
    # Tab 2: Word Embedding
    with tab2:
        word_embedding_tab()
    
    # Tab 3: Decoding
    with tab3:
        decoding_tab()
    
    # Tab 4: Output Generation
    with tab4:
        output_generation_tab()
    
    # Tab 5: Knowledge Check
    with tab5:
        display_knowledge_check()
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")


def tokenization_tab():
    st.header("Tokenization and Encoding")
    
    # Example sentence input
    example_sentence = st.text_input(
        "Enter a sentence to tokenize:", 
        value="A puppy is to dog as kitten is to",
        key="tokenization_input"
    )
    
    # Explanation
    st.markdown("""
    ### What is Tokenization?
    
    Tokenization is the first step in the transformer pipeline. It involves:
    1. Breaking down text input into smaller units called "tokens"
    2. Converting these tokens into numerical identifiers (encoding)
    
    This transformation makes the text data processable by machine learning models.
    """)
    
    # Interactive visualization
    if st.button("Tokenize and Encode", key="tokenize_button"):
        display_tokenization_process(example_sentence)


def display_tokenization_process(sentence):
    import pandas as pd
    
    # Simulate tokenization
    tokens = ["[CLS]"] + sentence.split() + ["[SEP]"]
    
    # Simulate token IDs (random for demonstration)
    import random
    token_ids = [random.randint(100, 50000) for _ in tokens]
    
    # Display the tokens and IDs
    st.subheader("Step 1: Convert words to tokens")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Sentence:**")
        st.info(sentence)
        
    with col2:
        st.markdown("**Tokenized:**")
        df = pd.DataFrame({
            "Token": tokens,
            "Token ID": token_ids
        })
        st.table(df)
    
    # Visualize embedding vectors (simplified with random values)
    st.subheader("Step 2: Encode tokens into vectors")
    st.markdown("""
    Each token ID is converted to an embedding vector (high-dimensional representation).
    Here's a simplified visualization with only 5 dimensions per vector:
    """)
    
    # Generate random embeddings for visualization
    import numpy as np
    embeddings = np.random.randn(len(tokens), 5)
    
    # Create a dataframe for display
    embedding_df = pd.DataFrame(
        embeddings, 
        index=tokens,
        columns=[f"Dim {i+1}" for i in range(5)]
    )
    st.table(embedding_df.round(2))
    
    # Code sample
    st.subheader("Code Example: Tokenization with Transformers")
    st.code("""
    # Using Hugging Face transformers library
    from transformers import AutoTokenizer
    
    # Load a pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize the input text
    text = "A puppy is to dog as kitten is to cat"
    tokens = tokenizer(text, return_tensors="pt")
    
    # The tokens object contains:
    # - input_ids: the numerical representations of tokens
    # - attention_mask: indicates which tokens should be attended to
    # - token_type_ids: identifies which sequence a token belongs to
    """, language="python")


def word_embedding_tab():
    st.header("Word Embedding")
    
    st.markdown("""
    ### What are Word Embeddings?
    
    Word embeddings convert tokens (words) into dense vector representations in a 
    multi-dimensional space. Words with similar meanings will be located closer to each other in this vector space.
    
    For example, "cat" and "kitten" would be closer together than "cat" and "airplane".
    """)
    
    # Interactive visualization
    st.subheader("Interactive Embedding Space Visualization")
    
    # Display 2D visualization of word embeddings
    import plotly.express as px
    import numpy as np
    import pandas as pd
    
    # Example words and their 2D embeddings (simplified for visualization)
    words = ["cat", "kitten", "feline", "dog", "puppy", "canine", 
             "car", "vehicle", "truck", "airplane", "fly"]
    
    # Create somewhat meaningful 2D embeddings for visualization
    # Words with similar meanings will be clustered together
    np.random.seed(42)  # For reproducibility
    
    # Create semantic clusters
    cat_family = np.array([1, 1]) + np.random.randn(3, 2) * 0.2  # cat, kitten, feline
    dog_family = np.array([-1, 1]) + np.random.randn(3, 2) * 0.2  # dog, puppy, canine
    vehicle_family = np.array([0, -1]) + np.random.randn(3, 2) * 0.2  # car, vehicle, truck
    air_family = np.array([2, -1]) + np.random.randn(2, 2) * 0.2  # airplane, fly
    
    # Combine embeddings
    embeddings_2d = np.vstack([cat_family, dog_family, vehicle_family, air_family])
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'word': words,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'category': ['animals']*6 + ['vehicles']*5
    })
    
    # Create interactive plot
    fig = px.scatter(
        df, x='x', y='y', text='word', color='category',
        title='2D Word Embedding Space Visualization',
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        width=700, height=500
    )
    fig.update_traces(textposition='top center', marker_size=10)
    st.plotly_chart(fig)
    
    st.markdown("""
    ### How Word Embeddings Work in Transformers
    
    1. Each token is mapped to a vector of floating-point numbers
    2. The dimension of these vectors is typically between 300-1000 for large models
    3. These vectors capture semantic relationships between words
    4. Similar words have similar vector representations (closer in the embedding space)
    """)
    
    # Code example
    st.subheader("Code Example: Working with Embeddings")
    st.code("""
    # Using a pre-trained embedding model
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Get embeddings for words
    def get_word_embedding(word):
        # Tokenize the word
        inputs = tokenizer(word, return_tensors="pt")
        
        # Get the embedding
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Extract the embedding from the last hidden state
        # Usually we'd take the [CLS] token embedding for the whole sentence
        # or average all token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()
        
    # Compare similarity between embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    
    emb1 = get_word_embedding("cat")
    emb2 = get_word_embedding("kitten")
    emb3 = get_word_embedding("airplane")
    
    # Higher value indicates greater similarity
    similarity_cat_kitten = cosine_similarity(emb1, emb2)
    similarity_cat_airplane = cosine_similarity(emb1, emb3)
    """, language="python")


def decoding_tab():
    st.header("Decoding Process")
    
    st.markdown("""
    ### How Decoding Works in Transformers
    
    Decoding is the process of turning the model's internal representations back into 
    meaningful output. In the context of our sentence completion task:
    
    1. The model processes the input sequence using self-attention mechanisms
    2. It weighs the importance of different tokens in the sequence
    3. For each potential next word, it calculates a probability distribution
    4. The decoding strategy determines which word to choose from this distribution
    """)
    
    # Interactive example
    st.subheader("Interactive Decoding Example")
    
    # Input prompt
    prompt = st.text_input(
        "Enter a prompt for the model to complete:",
        value="A puppy is to dog as kitten is to",
        key="decode_input"
    )
    
    # Decoding parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                               help="Higher values increase randomness/creativity")
    with col2:
        top_k = st.slider("Top-K", 1, 10, 3, 1,
                         help="Limit selection to top K most likely tokens")
    with col3:
        top_p = st.slider("Top-P (Nucleus Sampling)", 0.1, 1.0, 0.7, 0.1,
                         help="Select from tokens that make up top p probability mass")
    
    # Execute decoding
    if st.button("Generate Completion", key="decode_button"):
        display_decoding_simulation(prompt, temperature, top_k, top_p)


def display_decoding_simulation(prompt, temperature, top_k, top_p):
    import pandas as pd
    import numpy as np
    
    # Potential completion words with probabilities (simulated)
    candidates = {
        "cat": 0.30,
        "dog": 0.03, 
        "bird": 0.25,
        "bear": 0.20,
        "human": 0.15,
        "snake": 0.05,
        "fish": 0.01,
        "tiger": 0.01
    }
    
    # Create dataframe for display
    df = pd.DataFrame({
        "Word": list(candidates.keys()),
        "Base Probability": list(candidates.values())
    })
    
    # Apply temperature (simplistic simulation)
    # Temperature affects the probability distribution
    temp_probs = np.array(list(candidates.values()))
    if temperature != 1.0:
        # Simulate temperature effect
        temp_probs = temp_probs ** (1/temperature)
        temp_probs = temp_probs / temp_probs.sum()  # Normalize
    
    df["After Temperature"] = temp_probs
    
    # Apply Top-K filtering
    top_k_indices = np.argsort(temp_probs)[-top_k:]
    filtered_probs = np.zeros_like(temp_probs)
    filtered_probs[top_k_indices] = temp_probs[top_k_indices]
    
    # Normalize after Top-K
    if filtered_probs.sum() > 0:
        filtered_probs = filtered_probs / filtered_probs.sum()
    
    df["After Top-K"] = filtered_probs
    
    # Apply Top-P (nucleus sampling)
    sorted_indices = np.argsort(filtered_probs)[::-1]
    sorted_probs = filtered_probs[sorted_indices]
    cumsum_probs = np.cumsum(sorted_probs)
    nucleus = cumsum_probs <= top_p
    
    # Handle edge case for when no value is < top_p (include at least one token)
    if not nucleus.any():
        nucleus[0] = True
        
    # Create final filtered distribution
    final_indices = sorted_indices[nucleus]
    final_probs = np.zeros_like(filtered_probs)
    final_probs[final_indices] = filtered_probs[final_indices]
    
    # Normalize after Top-P
    if final_probs.sum() > 0:
        final_probs = final_probs / final_probs.sum()
    
    df["After Top-P"] = final_probs
    
    # Round for display
    df = df.round(3)
    
    # Highlight selection
    selected_word = df.loc[df["After Top-P"] > 0, "Word"].sample(1, weights=df.loc[df["After Top-P"] > 0, "After Top-P"]).iloc[0]
    
    # Display results
    st.subheader("Decoding Process Simulation")
    st.write(f"**Input Prompt:** {prompt}")
    
    # Display probability table
    st.markdown("#### Probability Distribution for Next Word")
    st.dataframe(df.style.background_gradient(cmap='Blues', subset=["After Top-P"]))
    
    # Display final selection
    st.markdown("### Final Output")
    st.success(f"{prompt} **{selected_word}**")
    
    # Explain the process
    st.markdown("#### Process Explanation")
    st.write(f"""
    1. **Base Distribution:** The model initially calculated probabilities for each potential word
    2. **Temperature Applied ({temperature}):** {'Increased randomness' if temperature > 1 else 'Decreased randomness'} in word selection
    3. **Top-K Filter ({top_k}):** Limited selection to the {top_k} most likely options
    4. **Top-P Filter ({top_p}):** Selected from words that make up {top_p*100}% of probability mass
    5. **Final Selection:** "{selected_word}" was selected based on the filtered distribution
    """)


def output_generation_tab():
    st.header("Output Generation")
    
    st.markdown("""
    ### The Final Step: Output Generation
    
    After the decoding process selects the next token, the transformer model:
    
    1. Adds the selected token to the existing sequence
    2. Updates its internal state based on the new token
    3. Repeats the process until a stop condition is met, such as:
       - Reaching a maximum length
       - Generating a specific stop token
       - Completing a required number of tokens
    """)
    
    # Interactive generation demo
    st.subheader("Interactive Sentence Completion")
    
    input_text = st.text_input(
        "Enter an incomplete sentence:",
        value="A puppy is to dog as kitten is to",
        key="generation_input"
    )
    
    # Generation parameters
    col1, col2 = st.columns(2)
    with col1:
        gen_temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1,
                                   help="Controls randomness of generation",
                                   key="gen_temp")
    with col2:
        max_tokens = st.slider("Max Tokens to Generate", 1, 20, 5, 1,
                              help="Maximum number of tokens to generate",
                              key="max_tokens")
    
    if st.button("Complete Sentence", key="generate_button"):
        display_output_generation(input_text, gen_temperature, max_tokens)


def display_output_generation(input_text, temperature, max_tokens):
    import time
    import numpy as np
    import pandas as pd
    
    # Create a placeholder for the generated text
    output_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Dictionary mapping certain word pairs/patterns to likely completions (simplified)
    completion_patterns = {
        "kitten is to": ["cat", "feline", "pet"],
        "dog as kitten": ["cat", "feline", "pet"],
        "train is to": ["tracks", "railway", "transportation"],
        "car is to": ["road", "vehicle", "transportation"],
        "day is to": ["night", "sun", "morning"],
        "happy is to": ["sad", "joy", "smile"]
    }
    
    # Find matching pattern
    next_word = None
    for pattern, options in completion_patterns.items():
        if pattern in input_text.lower():
            # Select based on temperature (higher = more random)
            if temperature < 0.5:
                next_word = options[0]  # Most common/expected
            elif temperature < 1.2:
                next_word = np.random.choice(options)  # Random from common options
            else:
                # With high temperature, introduce some unusual options
                unusual_options = ["elephant", "universe", "melody", "quantum", "dream"]
                combined = options + unusual_options
                next_word = np.random.choice(combined)
            break
    
    # Fallback if no pattern matched
    if next_word is None:
        common_words = ["the", "a", "one", "some", "many", "few", "this", "that"]
        next_word = np.random.choice(common_words)
    
    # Show generation process
    current_output = input_text
    tokens_to_generate = min(5, max_tokens)  # Cap at 5 for demo purposes
    
    # Words to potentially add after the first completion
    follow_up_words = [
        "which", "because", "when", "and", "but", "while", 
        "although", "since", "if", "as", "the", "a", "an"
    ]
    
    # Display the generation process token by token
    status_placeholder.info("Generating...")
    
    # First add the main completion word
    time.sleep(0.5)
    current_output = f"{current_output} {next_word}"
    output_placeholder.markdown(f"**Output so far:** {current_output}")
    
    # Then add any additional tokens if requested
    for i in range(tokens_to_generate - 1):
        time.sleep(0.7)  # Pause for effect
        
        # Add another word
        next_token = np.random.choice(follow_up_words)
        current_output = f"{current_output} {next_token}"
        output_placeholder.markdown(f"**Output so far:** {current_output}")
    
    # Show final result
    status_placeholder.success("Generation complete!")
    output_placeholder.markdown(f"### Final Output:\n{current_output}")
    
    # Show generation stats
    st.subheader("Generation Statistics")
    stats_df = pd.DataFrame({
        "Metric": ["Input Length", "Generated Tokens", "Temperature", "Time to Generate"],
        "Value": [
            f"{len(input_text.split())} tokens",
            tokens_to_generate,
            temperature,
            f"{tokens_to_generate * 0.7:.1f} seconds"
        ]
    })
    st.table(stats_df)
    
    # Explain the influence of parameters
    st.markdown("""
    ### How Parameters Affect Generation
    
    - **Temperature**: Controls randomness - higher values produce more creative and unpredictable outputs
    - **Max Tokens**: Limits the length of the generated text
    - **Top-K/Top-P**: Filters the vocabulary to control diversity and quality
    
    In production systems, these parameters let you balance between:
    - Creativity vs. Predictability
    - Diversity vs. Quality
    - Length vs. Relevance
    """)


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
        st.error(f"An unexpected error occurred: {str(e)}")
        
        # Provide debugging information in an expander
        with st.expander("Error Details"):
            st.code(str(e))
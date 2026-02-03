import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import uuid
from PIL import Image
import base64
import datetime
import random
import io
from utils.styles import load_css, custom_header
import utils.common as common
import utils.authenticate as authenticate

st.set_page_config(
    page_title="AWS Gen AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded")


transformer_architecture="""
graph TB
    %% Encoder side
    subgraph E [" "]
        direction TB
        IE[Input<br/>Embedding]
        ESA[Self-<br/>Attention] 
        EFF[Feed<br/>Forward]
        IE --> ESA
        ESA --> EFF
    end
    
    %% Decoder side
    subgraph D [" "]
        direction TB
        OE[Output<br/>Embedding]
        DSA[Self-<br/>Attention]
        DFF[Feed<br/>Forward] 
        OE --> DSA
        DSA --> DFF
    end
    
    %% Connection
    EFF --> DSA
    
    %% Labels
    E -.- ELabel[Encoder]
    D -.- DLabel[Decoder]
    
    %% Styling
    classDef encoderBox fill:#87CEEB,stroke:#4169E1,stroke-width:3px,opacity:0.3
    classDef decoderBox fill:#FFB6C1,stroke:#DC143C,stroke-width:3px,opacity:0.3
    classDef attention fill:#90EE90,stroke:#008000,stroke-width:2px,opacity:0.7
    classDef label fill:none,stroke:none,font-weight:bold
    
    class E encoderBox
    class D decoderBox
    class ESA,DSA attention
    class ELabel,DLabel label

"""


def create_temperature_plot():
    """Create a plot showing the effect of temperature on token probability distribution."""
    # Create data for the plot
    x = np.linspace(1, 10, 100)
    
    # High temperature (more random)
    y_high = np.array([1.0 / (1 + np.exp((i - 5) / 0.8)) for i in x])
    y_high = y_high / np.sum(y_high)
    
    # Low temperature (less random)
    y_low = np.array([1.0 / (1 + np.exp((i - 5) / 0.2)) for i in x])
    y_low = y_low / np.sum(y_low)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for high and low temperature
    fig.add_trace(go.Scatter(
        x=x, y=y_low,
        mode='lines',
        name='Low Temperature',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=y_high,
        mode='lines',
        name='High Temperature',
        line=dict(color='red', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title="Effect of Temperature on Token Probability Distribution",
        xaxis_title="Token Index",
        yaxis_title="Probability",
        legend_title="Temperature Setting",
        height=400,
    )
    
    return fig


def create_top_k_p_visual():
    """Create a visualization of top-k and top-p sampling parameters."""
    # Create dummy data for token probabilities
    tokens = ["cat", "dog", "bird", "fish", "lion", "tiger", "elephant", "zebra"]
    probabilities = [0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]
    
    # Create a bar chart
    fig = px.bar(
        x=tokens, 
        y=probabilities,
        labels={"x": "Tokens", "y": "Probability"},
        title="Token Selection with Top-K and Top-P",
        color=probabilities,
        color_continuous_scale="Viridis"
    )
    
    # Add horizontal lines for top-p thresholds
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Top-p=0.5",
        annotation_position="left"
    )
    
    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color="orange",
        annotation_text="Top-p=0.7",
        annotation_position="left"
    )
    
    # Add annotation for top-k=3
    fig.add_annotation(
        x=2,  # Position after the third bar
        y=0.01,
        text="Top-k=3",
        showarrow=True,
        arrowhead=1
    )
    
    fig.update_layout(
        height=400,
        showlegend=False  # Hide the color scale legend if not needed
    )
    
    return fig


def create_aws_gen_ai_stack():
    """Create a Sankey diagram to represent the AWS Gen AI stack."""
    # Create a Sankey diagram to represent the AWS Gen AI stack
    labels = [
        # Infrastructure Layer
        "EC2 Capacity Blocks", "NeuronUltraClusters", "EFA", "Nitro", "GPUs", "Inferentia", "Trainium", "SageMaker",
        # Middle Layer - Tools
        "Amazon Bedrock", "Guardrails", "Agents", "Customization", "Custom Model Import",
        # Top Layer - Applications
        "Amazon Q Business", "Amazon Q Developer", "Amazon Q in QuickSight", "Amazon Q in Connect",
        # User Personas
        "Business Users", "App Builders", "Data Scientists"
    ]
    
    # Define source, target, and value for the Sankey diagram
    source = [0, 1, 2, 3, 4, 5, 6, 7, 
              8, 8, 8, 8,
              12, 13, 14, 15]
    
    target = [8, 8, 8, 8, 8, 8, 8, 8,
              12, 13, 14, 15,
              18, 19, 19, 20]
    
    value = [1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1,
             3, 2, 1, 1]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=10,
            thickness=10,
            line=dict(color="black", width=0.1),
            label=labels,
            color="#FF9900"
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(255, 153, 0, 0.4)"
        )
    )])
    
    fig.update_layout(
        title="AWS Generative AI Stack",
        height=600,
    )
    
    return fig


def create_model_customization_approaches():
    """Create a radar chart of different model customization approaches."""
    # Create data
    approaches = ["Prompt Engineering", "RAG", "Fine-tuning", "Continued Pretraining"]
    
    # Values for different aspects
    complexity = [1, 3, 7, 10]
    cost = [1, 2, 6, 9]
    time = [1, 3, 7, 10]
    effectiveness = [4, 6, 8, 9]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add each approach as a trace
    for i, approach in enumerate(approaches):
        fig.add_trace(go.Scatterpolar(
            r=[complexity[i], cost[i], time[i], effectiveness[i], complexity[i]],
            theta=["Complexity", "Cost", "Time", "Effectiveness", "Complexity"],
            fill='toself',
            name=approach
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=True,
        title="Model Customization Approaches - Tradeoffs",
        height=500
    )
    
    return fig


def create_knowledge_check():
    """Create an interactive knowledge check quiz."""
    # If knowledge check hasn't been started yet
    if not st.session_state.knowledge_check_started:
        st.subheader("Test Your Knowledge")
        st.write("This knowledge check will test your understanding of Domain 2: Fundamentals of Generative AI.")
        if st.button("Start Knowledge Check"):
            st.session_state.knowledge_check_started = True
            st.session_state.knowledge_check_progress = 0
            st.session_state.knowledge_check_answers = {}
            st.rerun()
        return
    
    # Define the questions
    questions = [
        {
            "question": "A machine learning engineer is tasked with generating text using a Large Language Model (LLM) hosted in Amazon Bedrock. They need to decide between using the temperature and top_p parameters to control the output. Which of the following statements is true?",
            "options": [
                "The temperature parameter controls the creativity of the output, while top_p limits the next word choice to the most likely options.",
                "The temperature parameter limits the next word choice to the most likely options, while top_p controls the creativity of the output.",
                "Both temperature and top_p parameters control the creativity of the output.",
                "Both temperature and top_p parameters limit the next word choice to the most likely options."
            ],
            "answer": 0,
            "explanation": "The temperature parameter adjusts the randomness of the output by scaling the logits before applying the softmax function. Lower temperatures result in less random output, while higher temperatures result in more random output. The top_p parameter, on the other hand, controls the number of tokens considered for each word by selecting the top p tokens with the highest probability."
        },
        {
            "question": "Which of the following would be the most appropriate approach for working with LLMs on Amazon SageMaker?",
            "options": [
                "Use SageMaker's built-in algorithms for natural language processing tasks, such as the BlazingText algorithm for text classification or the Sequence-to-Sequence algorithm for text generation.",
                "Train your own custom language model from scratch using SageMaker's Deep Learning Containers and the TensorFlow or PyTorch frameworks.",
                "Use SageMaker JumpStart to quickly get started with pre-trained foundation models like BERT or T5 without the need for extensive training.",
                "Provision a high-performance EC2 instance with GPU support, install the necessary deep learning frameworks and libraries, and train your language model outside of SageMaker."
            ],
            "answer": 2,
            "explanation": "SageMaker JumpStart provides a convenient way to get started with pre-trained foundation models like GPT-3, BERT, and T5. It offers pre-trained model checkpoints, pre-built Docker containers, and example notebooks, allowing you to quickly fine-tune and deploy these models for various natural language processing tasks without the need for extensive training from scratch."
        },
        {
            "question": "A company is developing a generative AI chatbot for its users. The chatbot must be available at all times. To manage costs, the system should be efficient during idle times. Additionally, the company wants the ability to select the underlying Foundation Model (FM) for the chatbot. Which service meets these requirements MOST cost-effectively?",
            "options": [
                "Amazon Q",
                "Amazon SageMaker",
                "Amazon Elastic Compute Cloud",
                "Amazon Bedrock"
            ],
            "answer": 3,
            "explanation": "Amazon Bedrock is the most cost-effective choice as it's pricing is token-based, so if there are no invocations, there are no costs. Bedrock also allows you to choose the Foundation Model."
        },
        {
            "question": "What is one of the main benefits of transformer architectures in foundation models?",
            "options": [
                "They require less data for training compared to traditional neural networks",
                "They can process entire sequences of data in parallel rather than sequentially",
                "They use simple architectures with fewer parameters than other models",
                "They can only be used for text-to-text applications"
            ],
            "answer": 1,
            "explanation": "A key benefit of transformer architectures is their ability to process entire sequences of data in parallel, unlike RNNs and LSTMs which process data sequentially. This significantly speeds up training, making transformers well-suited for working with large datasets used when pre-training foundation models."
        },
        {
            "question": "Which of the following is NOT a common approach for customizing Foundation Models?",
            "options": [
                "Prompt Engineering",
                "Retrieval Augmented Generation (RAG)",
                "Fine-tuning",
                "Model Compression"
            ],
            "answer": 3,
            "explanation": "Model Compression is not one of the common approaches for customizing Foundation Models. The common approaches include Prompt Engineering, Retrieval Augmented Generation (RAG), Fine-tuning, and Continued Pretraining, presented in increasing order of complexity, quality, cost, and time required."
        }
    ]
    
    # Display progress
    progress = st.session_state.knowledge_check_progress / len(questions) * 100
    st.progress(progress)
    
    # Display restart button
    col1, col2 = st.columns([1, 7])
    with col1:
        if st.button("Restart Quiz"):
            st.session_state.knowledge_check_started = True
            st.session_state.knowledge_check_progress = 0
            st.session_state.knowledge_check_answers = {}
            st.rerun()
            
    # Display the current question
    current_q = st.session_state.knowledge_check_progress
    if current_q < len(questions):
        q = questions[current_q]
        st.subheader(f"Question {current_q + 1} of {len(questions)}")
        st.write(q["question"])
        
        # Create a unique key for this question
        q_key = f"q_{current_q}"
        
        # Display answer options
        selected = st.radio("Select your answer:", q["options"], key=q_key, index=None)
        
        # Check answer and move to next question
        if st.button("Submit Answer"):
            selected_index = q["options"].index(selected) if selected else -1
            st.session_state.knowledge_check_answers[current_q] = {
                "selected": selected_index,
                "correct": q["answer"],
                "explanation": q["explanation"]
            }
            
            if selected_index == q["answer"]:
                st.success("✅ Correct!")
                st.info(q["explanation"])
            else:
                st.error("❌ Incorrect!")
                st.info(q["explanation"])
            
            # Show next question button if there are more questions
            if current_q < len(questions) - 1:
                if st.button("Next Question"):
                    st.session_state.knowledge_check_progress += 1
                    st.rerun()
            else:
                st.session_state.knowledge_check_progress += 1
                st.rerun()
                
    else:
        # Quiz completed
        st.subheader("Knowledge Check Completed!")
        
        # Calculate score
        correct = 0
        for q_data in st.session_state.knowledge_check_answers.values():
            if q_data["selected"] == q_data["correct"]:
                correct += 1
        
        score_percent = (correct / len(questions)) * 100
        st.write(f"Your score: {correct}/{len(questions)} ({score_percent:.1f}%)")
        
        # Display feedback based on score
        if score_percent >= 80:
            st.success("Great job! You have a solid understanding of the fundamentals of generative AI.")
        elif score_percent >= 60:
            st.warning("Good effort! Review the topics you missed to improve your understanding.")
        else:
            st.error("You might need additional study. Review the material and try again.")
        
        # Option to review answers or restart
        st.subheader("Review Your Answers")
        for i, q in enumerate(questions):
            with st.expander(f"Question {i+1}: {q['question']}"):
                answer_data = st.session_state.knowledge_check_answers.get(i, {})
                selected_idx = answer_data.get("selected", -1)
                correct_idx = q["answer"]
                
                st.write("**Your answer:** " + (q["options"][selected_idx] if selected_idx >= 0 else "No answer selected"))
                st.write("**Correct answer:** " + q["options"][correct_idx])
                st.write("**Explanation:** " + q["explanation"])
                
                if selected_idx == correct_idx:
                    st.success("You answered correctly!")
                else:
                    st.error("Your answer was incorrect.")
        
        if st.button("Retake Knowledge Check"):
            st.session_state.knowledge_check_started = True
            st.session_state.knowledge_check_progress = 0
            st.session_state.knowledge_check_answers = {}
            st.rerun()


def render_home_tab():
    """Render the home tab content."""
    st.markdown(custom_header("Fundamentals of Generative AI", 1), unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='aws-card'>
        <h2>Program Summary</h2>
        <p>Welcome to Session 2 of the AWS Certified AI Practitioner preparation program. This session focuses on Domain 2: Fundamentals of Generative AI.</p>
        
        <h3>Week 2 Digital Training Curriculum</h3>
        <ul>
            <li>Essentials of Prompt Engineering</li>
            <li>Optimizing Foundation Models</li>
            <li>Security, Compliance, and Governance for AI Solutions</li>
            <li>Generative AI for Executives</li>
            <li>Amazon Q Business Getting Started</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='aws-card'>
        <h2>Today's Learning Outcomes</h2>
        <p>During this session, we will cover:</p>
        <ul>
            <li>Task Statement 2.1: Explain the basic concepts of generative AI</li>
            <li>Task Statement 2.2: Understand the capabilities and limitations of generative AI for solving business problems</li>
            <li>Task Statement 2.3: Describe AWS infrastructure and technologies for building generative AI applications</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("assets/images/AWS-Certified-AI-Practitioner_badge.png", caption="AWS Certified AI Practitioner")
        
        st.markdown("""
        <div class='highlight'>
        <h4>Interactive Learning</h4>
        <p>This application is designed to help you learn the fundamentals of generative AI through interactive examples, visualizations, and knowledge checks.</p>
        <p>Navigate through the tabs above to explore different topics.</p>
        </div>
        """, unsafe_allow_html=True)


def render_foundation_models_tab():
    """Render the foundation models tab content."""
    st.title("Foundation Models")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='concept-card'>
        <h3>What are Foundation Models?</h3>
        <p>Foundation models are pre-built machine learning models trained on a large amount of data (internet-scale data). These models can then be adapted to a wide range of downstream tasks.</p>
        
        <h4>Developing a Foundation Model:</h4>
        <ol>
            <li>Training a model on a large amount of unlabeled data (e.g., text data for a large language model).</li>
            <li>Pre-training process where the machine learning training algorithm uses the data to train the model.</li>
            <li>Once pre-training is completed, we have our foundation model.</li>
            <li>We can then customize and adapt that foundation model to perform particular tasks on our specific data.</li>
            <li>The most common way of customization is through the careful crafting of prompts.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='concept-card'>
        <h3>Components of Foundation Models</h3>
        <h4>Unlabeled Data</h4>
        <ul>
            <li>Easier to obtain compared to labeled data</li>
            <li>Pretraining models take context into account from all this training data</li>
            <li>Tracks the relationships in sequential data</li>
        </ul>
        
        <h4>Large Model</h4>
        <ul>
            <li>Billions of parameters</li>
            <li>Pretraining models of this size requires access to:
                <ul>
                    <li>Sufficient quantity and quality of training data</li>
                    <li>Large-scale training infrastructure</li>
                </ul>
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='concept-card'>
        <h3>Types of Foundation Models</h3>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Text-to-Text Models", "Text-to-Image Models"])
        
        with tab1:
            st.image("https://miro.medium.com/v2/resize:fit:720/format:webp/0*XDtcpv-m0SJRGSGB.png", caption="Text-to-Text Models (LLMs)")
            st.write("""
            **Text-to-Text Models (LLMs):**
            - Large language models (LLMs) pretrained to process vast quantities of textual data and human language
            - Use natural language processing (NLP) techniques
            - Applications include:
              - Text summarization
              - Information extraction
              - Question answering
              - Content creation
            """)
            
        with tab2:
            st.image("https://cdn.prod.website-files.com/67a1e6de2f2eab2e125f8b9a/67be08ddc9717ae946270318_image-25.png", caption="Text-to-Image Models")
            st.write("""
            **Text-to-Image Models:**
            - Take natural language input and produce high-quality images that match the input text description
            - Often use diffusion architecture
            - Examples include:
              - Stable Diffusion
              - DALL-E
              - Midjourney
            """)
        
        # Interactive example
        st.markdown("<div class='concept-card'><h3>Interactive Example: Foundation Model Completion</h3></div>", unsafe_allow_html=True)
        
        prompt = st.text_input("Enter a prompt for a text completion:", "A puppy is to dog as kitten is to")
        
        if st.button("Generate Completion"):
            st.success(f"{prompt} cat.")
            st.info("The foundation model understands relationships between concepts - in this case, that a kitten is a young cat, just as a puppy is a young dog.")


def render_transformer_architecture_tab():
    """Render the transformer architecture tab content."""
    st.title("Transformer Architecture")
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.markdown("""
        <div class='concept-card'>
        <h3>What is a Transformer?</h3>
        <p>The Transformer is a specific type of neural network that powers foundation models by processing sequences of information.</p>
        
        <h4>How it works:</h4>
        <ul>
            <li>Analyzes relationships between words or images to understand context and meaning</li>
            <li>Uses attention mechanisms to focus on relevant parts of the input</li>
            <li>Transforms input into relevant output by using its understanding of relationships between words</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='concept-card'>
        <h3>Benefits of Transformer Architecture</h3>
        <ul>
            <li><strong>Parallel Processing:</strong> Unlike RNNs and LSTMs which process data sequentially, transformers can process entire sequences of data in parallel, significantly speeding up training.</li>
            <li><strong>Attention Mechanism:</strong> The core innovation that allows the model to focus on different parts of the input sequence when predicting each part of the output sequence.</li>
            <li><strong>Flexibility and Scalability:</strong> Can be adapted for multiple modalities (text, images, audio, etc.) and enables effective transfer learning.</li>
        </ul>
        
        <h4>Analogy: Cocktail Party Effect</h4>
        <p>Imagine attending a busy cocktail party. Your brain can tune in on a specific conversation of interest while tuning out others (selective focus). Similarly, the attention mechanism in transformers selectively focuses on different parts of the input data, concentrating on the most relevant information for the task at hand.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Visual representation of transformer architecture
        common.mermaid(transformer_architecture, height=750)
        
        st.markdown("""
        <div class='concept-card'>
        <h3>How a Transformer Model Completes a Sentence</h3>
        </div>
        """, unsafe_allow_html=True)
        
        steps = st.radio(
            "Select a step to learn more:",
            ["1. Tokenization and Encoding", "2. Word Embedding", "3. Decoder", "4. Output Generation"],
            horizontal=True
        )
        
        if steps == "1. Tokenization and Encoding":
            st.write("""
            **Tokenization and Encoding:**
            1. The input text is broken down into smaller parts called tokens
            2. These tokens are converted into numerical identifiers (encoding)
            
            Example:
            ```
            "A puppy is to dog as kitten is to …"
            ↓
            [CLS] A puppy is to dog as kitten is to [SEP]
            ↓
            [101, 1037, 17022, 2003, 2000, 4389, 2003, 17022, 2003, 2000, 102]
            ```
            """)
            
        elif steps == "2. Word Embedding":
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*sAJdtNuUxKBDQkCqiG3D-A.png", caption="Word Embedding Vector Space")
            st.write("""
            **Word Embedding:**
            - Token encodings are converted into embeddings (multi-dimensional numerical representations)
            - Words with similar meanings are positioned closer to each other in this vector space
            - These embeddings capture semantic relationships between words
            
            Example: The vectors for "king" - "man" + "woman" would be close to "queen"
            """)
            
        elif steps == "3. Decoder":
            st.write("""
            **Decoder:**
            1. The transformer processes the input sequence using self-attention mechanisms
            2. It weighs the importance of different tokens in the sequence
            3. The decoder generates probability distributions for possible next tokens
            
            Example:
            ```
            A puppy is to dog as kitten is to _______
            
            Probabilities:
            "cat": 0.30
            "bird": 0.25
            "bear": 0.20
            "human": 0.15
            ```
            """)
            
        else:  # Output Generation
            st.write("""
            **Output Generation:**
            - The model selects the most probable token as the output
            - This selection can be influenced by parameters like temperature and top_k/top_p
            - The final output is generated by decoding the selected token back to text
            
            Example Output:
            ```
            "A puppy is to dog as kitten is to cat."
            ```
            """)
    
    # Sample code
    with st.expander("View Transformer Implementation (PyTorch)"):
        st.code("""
        import torch
        import torch.nn as nn
        
        class TransformerModel(nn.Module):
            def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, 
                         num_decoder_layers, dim_feedforward, dropout):
                super().__init__()
                
                # Token embedding
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.positional_encoding = PositionalEncoding(d_model, dropout)
                
                # Transformer layers
                self.transformer = nn.Transformer(
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )
                
                # Output layer
                self.output_layer = nn.Linear(d_model, vocab_size)
                
            def forward(self, src, tgt):
                # Embed tokens and add positional encoding
                src_embedded = self.positional_encoding(self.token_embedding(src))
                tgt_embedded = self.positional_encoding(self.token_embedding(tgt))
                
                # Pass through transformer
                output = self.transformer(src_embedded, tgt_embedded)
                
                # Project to vocabulary
                return self.output_layer(output)
                
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=5000):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                
                position = torch.arange(max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                pe = torch.zeros(max_len, 1, d_model)
                pe[:, 0, 0::2] = torch.sin(position * div_term)
                pe[:, 0, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe)
                
            def forward(self, x):
                x = x + self.pe[:x.size(0)]
                return self.dropout(x)
        """)


def render_inference_parameters_tab():
    """Render the inference parameters tab content."""
    st.title("Foundation Model Inference Parameters")
    
    st.markdown("""
    <div class='concept-card'>
    <h3>Controlling Randomness and Creativity in Outputs</h3>
    <p>Inference parameters allow us to control how foundation models generate responses. These parameters affect factors like creativity, diversity, and length of the generated content.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Temperature")
        
        st.markdown("""
        Temperature controls the randomness of the words presented by the model:
        - **Lower temperature** (closer to 0): More focused outputs, less diverse but more predictable
        - **Higher temperature**: More random outputs, more creative but potentially less coherent
        - **Default**: Usually 0 or 1 depending on the model
        """)
        
        # Interactive temperature slider
        temp = st.slider("Temperature Setting", 0.0, 2.0, 1.0, 0.1)
        
        prompt_temp = "Write a tagline for a coffee shop called 'Bean There'"
        
        if st.button("Generate with Temperature"):
            if temp < 0.5:
                st.success("Bean There, Done That: Your Go-To Coffee Shop")
                st.info("Low temperature produces safe, predictable outputs.")
            elif temp < 1.2:
                st.success("Bean There: Where Every Sip Tells a Story")
                st.info("Medium temperature balances creativity and coherence.")
            else:
                st.success("BEAN THERE: Cosmic Caffeine Journeys For The Soul's Awakening!")
                st.info("High temperature produces more random, creative outputs.")
        
    with col2:
        st.subheader("Top-k and Top-p")
        
        st.markdown("""
        **Top-k** limits choice to the k most likely next words:
        - Lower k: More focused, less diverse outputs
        - Higher k: More diverse outputs
        
        **Top-p (nucleus sampling)** selects from the most probable tokens until their cumulative probability exceeds p:
        - Lower p: More focused outputs
        - Higher p: More diverse outputs
        """)
        
        col_k, col_p = st.columns(2)
        
        with col_k:
            k_value = st.number_input("Top-k Value", 1, 50, 10, 1)
        
        with col_p:
            p_value = st.slider("Top-p Value", 0.0, 1.0, 0.9, 0.1)
        
        prompt_top = "List a creative name for a tech startup"
        
        if st.button("Generate with Top-k/p"):
            if k_value <= 3 and p_value <= 0.5:
                st.success("TechNova")
                st.info("Low k and p values produce common, predictable names.")
            elif k_value >= 20 or p_value >= 0.9:
                st.success("QuantumPixelForge")
                st.info("High k or p values allow for more unique, creative names.")
            else:
                st.success("ByteWave")
                st.info("Moderate k and p values balance uniqueness and appropriateness.")
    
    # Visualizations for temperature and top-k/p
    st.plotly_chart(create_temperature_plot(), use_container_width=True)
    
    st.plotly_chart(create_top_k_p_visual(), use_container_width=True)
    
    # Additional inference parameters
    st.subheader("Length Control Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Response Length**
        - Specifies minimum or maximum tokens to return
        - Helps manage computational resources and cost
        - Example: `max_tokens=150`
        """)
    
    with col2:
        st.markdown("""
        **Penalties**
        - Penalizes repetitions or specific token types
        - Examples:
          - Repetition penalty
          - Frequency penalty
          - Length penalty
        """)
    
    with col3:
        st.markdown("""
        **Stop Sequences**
        - Characters that stop token generation
        - Useful for structured responses
        - Example: `stop=["###", "User:"]`
        """)
    
    # Code example
    with st.expander("Amazon Bedrock Inference Parameter Example"):
        st.code("""
        import boto3
        import json
        
        bedrock = boto3.client(service_name='bedrock-runtime')
        
        def invoke_model(prompt):
            body = json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 250,
                "stop_sequences": ["\\n\\nHuman:"],
                "anthropic_version": "bedrock-2023-05-31"
            })
            
            response = bedrock.invoke_model(
                body=body,
                modelId='anthropic.claude-v2'
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('completion')
            
        result = invoke_model("Write a short poem about cloud computing")
        print(result)
        """)


def render_context_tab():
    """Render the context tab content."""
    st.title("Context in Foundation Models")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='concept-card'>
        <h3>What is Context?</h3>
        <p>Context refers to the ability of a model to understand the preceding information when generating future text. It's essentially the "memory" of previous parts of the conversation or document.</p>
        
        <h4>Key characteristics of context:</h4>
        <ul>
            <li>It does not persist across different sessions</li>
            <li>There is an upper limit on the number of tokens that can be remembered</li>
            <li>Initial information the model is using can be lost if the context becomes too long</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='concept-card'>
        <h3>Context Window</h3>
        <p>The context window is the maximum amount of text (measured in tokens) that a model can consider at once. Different models have different context window sizes:</p>
        <ul>
            <li>GPT-3.5: ~4K tokens (about 3,000 words)</li>
            <li>GPT-4: ~8K-32K tokens</li>
            <li>Claude: Up to 100K tokens</li>
            <li>Anthropic Claude 3 Opus: Up to 200K tokens</li>
        </ul>
        <p>When the context window is filled, older information gets pushed out, which can cause the model to "forget" earlier parts of the conversation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='concept-card'><h3>Context Window Example</h3></div>", unsafe_allow_html=True)
        
        # Interactive example of context
        chat_history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi, what's the best place in Seattle to visit?"},
            {"role": "assistant", "content": "The Columbia Center offers breathtaking views of the city skyline, Elliott Bay, and the surrounding mountains. It has an observation deck on the 73rd floor where you can see the entire city."}
        ]
        
        # Display chat history
        for message in chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            elif message["role"] == "assistant":
                st.chat_message("assistant").write(message["content"])
        
        # Next user message with ambiguous reference
        user_input = st.text_input("Ask a follow-up question about 'this':", "Will this be fun for children?")
        
        if st.button("Send Message"):
            st.chat_message("user").write(user_input)
            
            if "children" in user_input.lower() and "this" in user_input.lower():
                response = "While the Columbia Center's observation deck offers amazing views, it might not be the most engaging activity for young children who may prefer more interactive attractions. For children, I'd recommend the Seattle Aquarium, Pacific Science Center, or Woodland Park Zoo instead, as they offer more hands-on activities."
                st.chat_message("assistant").write(response)
                
                st.info("Notice how the model understood that 'this' refers to the Columbia Center mentioned in the previous message. It maintained context across the conversation.")
        
        # Visualization of context window tokens
        st.markdown("### Token Usage in Conversation")
        
        tokens = {
            "System message": 5,
            "User question 1": 10,
            "Assistant response 1": 40,
            "User question 2": 7,
            "Assistant response 2": 55
        }
        
        # Create a horizontal bar chart for token usage
        fig = px.bar(
            x=list(tokens.values()),
            y=list(tokens.keys()),
            orientation='h',
            labels={'x': 'Number of tokens', 'y': ''},
            title="Token Usage in Context Window (Example)",
            color=list(tokens.values()),
            color_continuous_scale="Blues"
        )

        # Update layout with the vertical line and annotation properly
        fig.update_layout(
            shapes=[{
                'type': 'line',
                'xref': 'x',
                'yref': 'paper',
                'x0': 100,
                'x1': 100,
                'y0': 0,
                'y1': 1,
                'line': {
                    'color': 'red',
                    'dash': 'dash'
                }
            }],
            annotations=[{
                'x': 105,
                'y': 1,
                'xref': 'x',
                'yref': 'paper',
                'text': '4K token limit',
                'showarrow': False
            }]
        )

        st.plotly_chart(fig, use_container_width=True)


def render_concerns_tab():
    """Render the concerns tab content."""
    st.title("Concerns of Generative AI")
    
    st.markdown("""
    <div class='concept-card'>
    <p>While generative AI offers tremendous benefits, it also comes with several concerns that organizations need to address.</p>
    </div>
    """, unsafe_allow_html=True)
    
    concerns = {
        "Toxicity": {
            "description": "Generating inflammatory or inappropriate content",
            "mitigations": [
                "Curation of training data to identify and remove potentially toxic traces",
                "Use of guardrail models to detect and filter out unwanted content",
                "Implementing content filtering systems"
            ],
            "icon": "⚠️"
        },
        "Hallucinations": {
            "description": "An assertion or claim that sounds plausible but is factually incorrect",
            "mitigations": [
                "Educate users that output must be fact-checked and independently verified",
                "Implement retrieval augmented generation to ground responses in factual data",
                "Provide citations and sources when possible"
            ],
            "icon": "🔍"
        },
        "Intellectual Property": {
            "description": "Models can generate content that replicates training data, causing IP issues",
            "mitigations": [
                "Model distillment (removing or reducing protected content)",
                "Sharding (dividing and retraining specific parts of the model)",
                "Content filtering (comparing generated content with protected content)"
            ],
            "icon": "©️"
        },
        "Plagiarism and Cheating": {
            "description": "Using AI for assignments in education or copying others' work",
            "mitigations": [
                "Rethinking assessment methods in education",
                "AI detection models",
                "Updating academic integrity policies"
            ],
            "icon": "📝"
        },
        "Disruption of the Nature of Work": {
            "description": "Fear of job displacement by AI",
            "mitigations": [
                "Focus on AI as an augmentation rather than replacement tool",
                "Reskilling and upskilling programs",
                "Designing AI systems that complement human capabilities"
            ],
            "icon": "💼"
        }
    }
    
    # Create expandable sections for each concern
    for concern, details in concerns.items():
        with st.expander(f"{details['icon']} {concern}"):
            st.markdown(f"**Description:** {details['description']}")
            
            st.markdown("**Mitigations:**")
            for mitigation in details["mitigations"]:
                st.markdown(f"- {mitigation}")
    
    # Case study example
    st.subheader("Case Study: AI Hallucination Example")
    
    st.markdown("""
    <div class='highlight'>
    <h4>Scenario: Legal Research Assistant</h4>
    <p>A law firm implemented an AI assistant to help with legal research. During a high-stakes case, the AI confidently cited a non-existent legal precedent that seemed plausible but was completely fabricated.</p>
    
    <p><strong>Impact:</strong> The firm included this citation in court documents, damaging their credibility when opposing counsel pointed out the error.</p>
    
    <p><strong>Solution:</strong> The firm implemented a RAG system that only allowed the AI to cite from verified legal databases and required human verification of all AI-generated citations before submission.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual representation of concerns
    st.subheader("Addressing Generative AI Concerns")
    
    # Create a radar chart for different concerns and their impact/mitigation levels
    categories = list(concerns.keys())
    
    # Values for different metrics
    prevalence = [8, 9, 7, 6, 8]
    mitigation_difficulty = [7, 8, 9, 6, 7]
    business_impact = [9, 9, 8, 5, 9]
    
    fig = go.Figure()
    
    # Add each metric as a trace
    fig.add_trace(go.Scatterpolar(
        r=prevalence,
        theta=categories,
        fill='toself',
        name='Prevalence'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=mitigation_difficulty,
        theta=categories,
        fill='toself',
        name='Mitigation Difficulty'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=business_impact,
        theta=categories,
        fill='toself',
        name='Business Impact'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=True,
        title="Generative AI Concerns: Impact Assessment",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_aws_gen_ai_stack_tab():
    """Render the AWS Gen AI Stack tab content."""
    st.title("AWS Generative AI Stack")
    
    st.markdown("""
    <div class='concept-card'>
    <p>AWS provides a comprehensive ecosystem of services and infrastructure for building, deploying, and managing generative AI applications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual of AWS Gen AI Stack
    st.plotly_chart(create_aws_gen_ai_stack(), use_container_width=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Infrastructure for FM Training and Inference")
        
        infrastructure = {
            "EC2 Capacity Blocks": "Reserved compute capacity for large-scale ML workloads",
            "NeuronUltraClusters": "Optimized infrastructure for ML training and inference",
            "EFA (Elastic Fabric Adapter)": "Network interface for high-performance computing",
            "Nitro": "AWS infrastructure for enhanced security and performance",
            "GPUs": "Accelerated computing for parallel processing",
            "AWS Inferentia": "Custom chip designed by AWS to accelerate machine learning inference",
            "AWS Trainium": "Custom chip designed by AWS to accelerate machine learning training",
            "SageMaker": "Fully managed ML service for building, training, and deploying models"
        }
        
        for tech, desc in infrastructure.items():
            st.markdown(f"**{tech}**: {desc}")
    
    with col2:
        st.subheader("Tools to Build with LLMs and FMs")
        
        tools = [
            "**Amazon Bedrock**: Fully managed service for building generative AI applications with foundation models",
            "**Guardrails**: Control mechanisms to ensure safety and accuracy",
            "**Agents**: AI systems that can execute multistep tasks",
            "**Customization capabilities**: Methods to adapt models to specific use cases",
            "**Custom Model Import**: Import and use your own models with AWS services"
        ]
        
        for tool in tools:
            st.markdown(tool)
    
    st.subheader("Applications Leveraging LLMs and FMs")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image("https://d1.awsstatic.com/Q_logo.2d5f7eac46e3531d5ee1cf1d86dc5835d4387944.png", width=100)
        st.markdown("**Amazon Q Business**")
        st.markdown("- Knowledge search")
        st.markdown("- Summarization")
        st.markdown("- Content creation")
        st.markdown("- Extract insights")
    
    with col2:
        st.image("https://d1.awsstatic.com/Q_logo.2d5f7eac46e3531d5ee1cf1d86dc5835d4387944.png", width=100)
        st.markdown("**Amazon Q Developer**")
        st.markdown("- Code generation")
        st.markdown("- Unit testing")
        st.markdown("- Security scanning")
        st.markdown("- Troubleshooting")
    
    with col3:
        st.image("https://d1.awsstatic.com/Q_logo.2d5f7eac46e3531d5ee1cf1d86dc5835d4387944.png", width=100)
        st.markdown("**Amazon Q in QuickSight**")
        st.markdown("- Understand data")
        st.markdown("- Build visuals")
        st.markdown("- Create calculations")
        st.markdown("- Data stories")
    
    with col4:
        st.image("https://d1.awsstatic.com/Q_logo.2d5f7eac46e3531d5ee1cf1d86dc5835d4387944.png", width=100)
        st.markdown("**Amazon Q in Connect**")
        st.markdown("- Agent assist")
        st.markdown("- Customer support")
        st.markdown("- Real-time recommendations")
        st.markdown("- Reduce hold times")
    
    # Add PartyRock section
    st.subheader("PartyRock - Experimental App Building")
    
    st.markdown("""
    <div class='highlight'>
    <p>PartyRock is an Amazon Bedrock playground that allows users to build, share, and remix AI apps for various fun tasks. It's designed to teach fundamental generative AI skills through hands-on experimentation.</p>
    
    <p>Examples of what you can build:</p>
    <ul>
        <li>Dad joke generators</li>
        <li>Personalized music playlists</li>
        <li>Recipe recommenders</li>
        <li>Trivia games</li>
        <li>AI storytellers</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Code example
    with st.expander("Amazon Bedrock API Usage Example"):
        st.code("""
        import boto3
        import json
        
        # Initialize Bedrock client
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        
        # Define prompt
        prompt = "Explain quantum computing in simple terms"
        
        # Prepare request body
        request_body = {
            "prompt": f"\\n\\nHuman: {prompt}\\n\\nAssistant:",
            "max_tokens_to_sample": 300,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        # Invoke model
        response = bedrock.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps(request_body)
        )
        
        # Process response
        response_body = json.loads(response.get("body").read())
        completion = response_body.get("completion")
        
        print(completion)
        """)


def render_model_training_tab():
    """Render the model training tab content."""
    st.title("Model Training and Fine Tuning")
    
    st.markdown("""
    <div class='concept-card'>
    <p>There are several approaches to customize foundation models to better fit specific use cases, ranging from simple prompt engineering to complex continued pre-training.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual representation of model customization approaches
    st.plotly_chart(create_model_customization_approaches(), use_container_width=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Common Approaches for Customizing FMs")
        
        approaches = {
            "Prompt Engineering": {
                "description": "Crafting prompts to guide model outputs",
                "complexity": "Low",
                "training_required": "No",
                "example": "Creating instructions and examples in the prompt to get desired outputs"
            },
            "Retrieval Augmented Generation (RAG)": {
                "description": "Retrieving relevant knowledge for models",
                "complexity": "Medium",
                "training_required": "No",
                "example": "Connecting LLMs to enterprise documents to ground responses in factual information"
            },
            "Fine-tuning": {
                "description": "Adapting models for specific tasks",
                "complexity": "High",
                "training_required": "Yes",
                "example": "Training the model on specific examples like customer support conversations"
            },
            "Continued Pre-training": {
                "description": "Enhancing pretrained models with more data",
                "complexity": "Very High",
                "training_required": "Yes",
                "example": "Additional training on domain-specific data like medical or legal texts"
            }
        }
        
        for approach, details in approaches.items():
            with st.expander(approach):
                st.markdown(f"**Description:** {details['description']}")
                st.markdown(f"**Complexity:** {details['complexity']}")
                st.markdown(f"**Training Required:** {details['training_required']}")
                st.markdown(f"**Example:** {details['example']}")
    
    with col2:
        st.subheader("Customizing Model Responses for Your Business")
        
        st.markdown("### Fine-Tuning")
        st.markdown("""
        **Purpose:** Maximizing accuracy for specific tasks
        
        **Data Need:** Small number of labeled examples
        
        **Examples:**
        - Sentiment analysis with labeled positive/negative reviews
        - Question-answering systems with paired questions and answers
        - Customer service chatbots with example conversations
        - Classification tasks with categorized data
        """)
        
        st.markdown("### Continued Pre-training")
        st.markdown("""
        **Purpose:** Maintaining model accuracy for your domain
        
        **Data Need:** Large number of unlabeled datasets
        
        **Examples:**
        - Medical terminology and healthcare documentation
        - Legal documents and regulatory texts
        - Financial reports and industry-specific jargon
        - Technical documentation
        """)
        
        # Interactive example
        st.markdown("### Interactive Example: Choose a Business Domain")
        
        domain = st.selectbox(
            "Select a business domain to see customization approach:",
            ["E-commerce", "Healthcare", "Legal", "Customer Support", "Financial Services"]
        )
        
        if domain == "E-commerce":
            st.info("""
            **Recommended Approach: Fine-Tuning**
            
            **Training Data Examples:**
            - Product descriptions paired with generated SEO optimized versions
            - Customer reviews paired with summary responses
            - Query-product pairs for better search results
            
            **AWS Solution:** 
            Use Amazon Bedrock for fine-tuning with your product catalog data
            """)
            
        elif domain == "Healthcare":
            st.info("""
            **Recommended Approach: Continued Pre-training + RAG**
            
            **Training Data Examples:**
            - Medical journals and research papers
            - Healthcare procedural documentation
            - Medical terminology glossaries
            
            **AWS Solution:** 
            Use Amazon SageMaker for continued pre-training and implement RAG with Amazon Kendra for secure medical information retrieval
            """)
            
        elif domain == "Legal":
            st.info("""
            **Recommended Approach: RAG + Fine-Tuning**
            
            **Training Data Examples:**
            - Legal case documents
            - Contract templates and clauses
            - Previous legal opinions and analyses
            
            **AWS Solution:** 
            Use Amazon Bedrock with Knowledge Bases to create a secure RAG system accessing your legal documents
            """)
            
        elif domain == "Customer Support":
            st.info("""
            **Recommended Approach: Fine-Tuning**
            
            **Training Data Examples:**
            - Past customer conversations with successful resolutions
            - Common issue-solution pairs
            - FAQ responses
            
            **AWS Solution:** 
            Use Amazon Bedrock fine-tuning with your support conversations and integrate with Amazon Q in Connect
            """)
            
        else:  # Financial Services
            st.info("""
            **Recommended Approach: Continued Pre-training + RAG**
            
            **Training Data Examples:**
            - Financial reports and analyses
            - Regulatory documentation
            - Market trend data
            
            **AWS Solution:** 
            Use Amazon SageMaker for training on financial texts and implement secure RAG with Amazon Bedrock Knowledge Bases
            """)


def render_knowledge_checks_tab():
    """Render the knowledge checks tab content."""
    st.title("Knowledge Checks")
    create_knowledge_check()


def initialize_session_state():
    """Initialize session state variables."""
       
    if "knowledge_check_started" not in st.session_state:
        st.session_state.knowledge_check_started = False

    if "knowledge_check_progress" not in st.session_state:
        st.session_state.knowledge_check_progress = 0
        
    if "knowledge_check_answers" not in st.session_state:
        st.session_state.knowledge_check_answers = {}


    
    load_css()


def render_sidebar():
    """Render the sidebar content."""
    with st.sidebar:
        common.render_sidebar()
        
        # About this App (collapsible)
        with st.expander("About this App", expanded=False):
            st.write("""
            This application is an interactive e-learning tool for AWS Certified AI Practitioner 
            candidates focusing on Domain 2: Fundamentals of Generative AI.
            
            **Topics Covered:**
            - Foundation Models
            - Transformer Architecture
            - FM Inference Parameters
            - Context in Foundation Models
            - Concerns of Generative AI
            - AWS Generative AI Stack
            - Model Training and Fine Tuning
            - Knowledge Checks
            """)


def render_footer():
    """Render the footer content."""
    st.markdown("""
    <div class='aws-footer'>
    © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main function to run the application."""
    # Setup page and initialize session state
    common.initialize_session_state()
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()

    # Render each tab's content
    render_home_tab()
    
    # Render footer
    render_footer()


# Main execution flow
if __name__ == "__main__":
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()
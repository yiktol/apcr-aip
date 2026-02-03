# Streamlit BERT Masked Language Modeling UI Enhancement

import streamlit as st
import pandas as pd
from transformers import BertTokenizer, pipeline
import plotly.graph_objects as go
import time
from utils.common import render_sidebar
from utils.styles import load_css, custom_header, load_css
import utils.authenticate as authenticate

st.set_page_config(
    page_title="BERT Masked Language Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App configuration and styling
def setup_page():
   
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
        /* Main title styling */
        
        /* Subtitle styling */
        .subtitle {
            color: #5F6368;
            text-align: center;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* Card styling */
        .card {
            background-color: #F8F9FA;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #E8EAED;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Highlighted text */
        .highlight {
            background-color: #E8F0FE;
            padding: 2px 5px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        /* Tab styling */
        div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] button {
            border-radius: 4px 4px 0 0;
            padding: 10px 15px;
            font-weight: 500;
        }
        
        div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] button[aria-selected="true"] {
            background-color: #4285F4;
            color: white;
        }
        
        /* Form button styling */
        div.stButton > button[kind="primaryFormSubmit"] {
            background-color: #4285F4;
            color: white;
            border: none;
            width: 100%;
            font-weight: 500;
        }
        
        /* Results section styling */
        .results-title {
            color: #4285F4;
            font-size: 1.4rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            color: #5F6368;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    return tokenizer, unmasker

# Dataset of mask examples
dataset = [
    {"id": 0, "text": "A puppy is to dog as kitten is to [MASK].", "category": "Analogies"},
    {"id": 1, "text": "The best part of this film is the [MASK] scene.", "category": "Entertainment"},
    {"id": 2, "text": "I had a great time at the [MASK] restaurant.", "category": "Dining"},
    {"id": 3, "text": "Hello I'm a [MASK] model.", "category": "Technology"},
    {"id": 4, "text": "I would definitely visit this place again if I had the chance, it's [MASK].", "category": "Travel"},
    {"id": 5, "text": "The food at this restaurant was [MASK].", "category": "Dining"},
    {"id": 6, "text": "The atmosphere at the [MASK] place is so relaxing.", "category": "Lifestyle"},
    {"id": 7, "text": "I had a terrible experience at the [MASK] restaurant.", "category": "Dining"},
    {"id": 8, "text": "The [MASK] movie was amazing!", "category": "Entertainment"}
]

# Function to format prediction results
def format_predictions(results):
    df = pd.DataFrame([
        {
            "Token": result['token_str'],
            "Score": result['score'],
            "Sequence": result['sequence'].replace('[MASK]', f"<span class='highlight'>{result['token_str']}</span>")
        }
        for result in results
    ])
    return df

# Function to create a bar chart for prediction probabilities
def create_prediction_chart(results):
    tokens = [result['token_str'] for result in results]
    scores = [result['score'] for result in results]
    
    fig = go.Figure(data=[
        go.Bar(
            x=tokens, 
            y=scores,
            marker_color='#4285F4',
            text=[f'{score:.1%}' for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Predicted Token",
        yaxis_title="Probability",
        yaxis_tickformat='.1%',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

# Main application
def main():
    setup_page()
    
    load_css()
    # App header
    st.markdown("<h1>🤖 BERT Masked Language Model</h1>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Predict masked words in sentences using BERT</div>", unsafe_allow_html=True)
    
    # Load models
    tokenizer, unmasker = load_models()
    
    # Sidebar with information
    with st.sidebar:
        render_sidebar()
        with st.expander("About this App", expanded=False):
            st.markdown("""
            <div class='card'>
                <p>BERT's Masked Language Modeling (MLM) is a technique used to train language models by randomly masking words in a sentence and then predicting those masked words.</p>
                <p>This application demonstrates how BERT can predict words that have been masked with <code>[MASK]</code> in sentences.</p>
            </div>
            """, unsafe_allow_html=True)
        
    
    # Create category filters in the main area
    categories = sorted(list(set(item["category"] for item in dataset)))
    selected_categories = st.multiselect(
        "Filter by category:",
        options=categories,
        default=categories,
        help="Select categories to filter examples"
    )
    
    # Filter dataset based on selected categories
    filtered_dataset = [item for item in dataset if item["category"] in selected_categories]
    
    if not filtered_dataset:
        st.warning("No examples match your filter criteria. Please select different categories.")
    
    # Create tabs for examples
    tab_names = [f"Example {i+1}: {item['category']}" for i, item in enumerate(filtered_dataset)]
    tabs = st.tabs(tab_names)
    
    # Create content for each tab
    for i, tab in enumerate(tabs):
        item = filtered_dataset[i]
        
        with tab:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"<div class='card'><p>{item['text']}</p></div>", unsafe_allow_html=True)
                
                with st.form(f"mask_form_{item['id']}"):
                    prompt = st.text_input(
                        "Edit the sentence (include [MASK] token):",
                        value=item['text'],
                        key=f"input_{item['id']}"
                    )
                    
                    col_submit, col_space = st.columns([1, 3])
                    with col_submit:
                        submit = st.form_submit_button("Predict", type="primary")
            
            with col2:
                st.markdown(f"**Category:** {item['category']}")
                st.markdown("**Instructions:**")
                st.markdown("1. The sentence contains a [MASK] token")
                st.markdown("2. BERT will predict what word fits in that position")
                st.markdown("3. Click 'Predict' to see the results")
            
            # Handle prediction when form is submitted
            if submit:
                if "[MASK]" in prompt:
                    with st.spinner("Generating predictions..."):
                        # Add a small delay for visual effect
                        time.sleep(0.5)
                        results = unmasker(prompt)
                        
                        st.markdown("<h3 class='results-title'>Predictions</h3>", unsafe_allow_html=True)
                        
                        # Create two columns for results display
                        col_table, col_chart = st.columns([1, 1])
                        
                        with col_table:
                            # Format and display predictions
                            df = format_predictions(results)
                            
                            # Display the sequences with highlighted predictions
                            for i, row in df.iterrows():
                                st.markdown(f"**{i+1}.** {row['Sequence']} (Score: {row['Score']:.1%})", unsafe_allow_html=True)
                        
                        with col_chart:
                            # Create and display the chart
                            chart = create_prediction_chart(results)
                            st.plotly_chart(chart, use_container_width=True)
                else:
                    st.error("Please include [MASK] token in your text.")
    
    # Footer
    st.markdown("<div class='aws-footer'>© 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", unsafe_allow_html=True)

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

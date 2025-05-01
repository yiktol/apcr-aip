import streamlit as st


# AWS color scheme
AWS_COLORS = {
    'primary': '#FF9900',      # AWS Orange
    'secondary': '#232F3E',    # AWS Navy
    'tertiary': '#1A476F',     # AWS Blue
    'background': '#FFFFFF',   # White
    'text': '#16191F',         # Dark gray
    'success': '#008296',      # Teal
    'warning': '#D13212',      # Red
    'info': '#1E88E5'          # Blue
}

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
        /* AWS Color scheme */
        :root {
            --aws-orange: #FF9900;
            --aws-gray: #232F3E;
            --aws-light-gray: #EAEDED;
            --aws-blue: #0073BB;
            --aws-green: #1E8E3E;
        }
        
        /* Header styles */
        h1, h2, h3 {
            color: var(--aws-gray);
        }
        
        /* Highlight important text */
        # .highlight {
        #     background-color: var(--aws-light-gray);
        #     padding: 0.2em 0.5em;
        #     border-radius: 4px;
        #     color: white;
        # }

        .highlight {
            background-color: #FFFFCC;
            padding: 10px;
            border-left: 5px solid #FF9900;
            margin-bottom: 15px;
        }
        
        /* Custom card layout */
        .card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: white;
        }
        
        /* Button styling */
        .stButton button {
            background-color: var(--aws-orange);
            color: white;
            border: none;
            font-weight: bold;
        }
        
        .stButton button:hover {
            background-color: #E88B00;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 0.5rem 0.5rem;
            background-color: var(--aws-light-gray);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--aws-orange);
            color: white;
        }
        
        /* Footer styling */
        footer {
            border-top: 1px solid #ddd;
            padding-top: 1rem;
            font-size: 0.8rem;
            color: gray;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: var(--aws-orange);
        }
        
        /* Success message */
        .success {
            padding: 1rem;
            background-color: var(--aws-green);
            color: white;
            border-radius: 4px;
        }
        
        /* Dataframe styling */
        .dataframe {
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
# Custom styling functions
def custom_header(text, level=1):
    if level == 1:
        return f"<h1 style='color:#FF9900; font-family:Amazon Ember, Arial, sans-serif;'>{text}</h1>"
    elif level == 2:
        return f"<h2 style='color:#232F3E; font-family:Amazon Ember, Arial, sans-serif;'>{text}</h2>"
    elif level == 3:
        return f"<h3 style='color:#232F3E; font-family:Amazon Ember, Arial, sans-serif;'>{text}</h3>"
    else:
        return f"<h4 style='color:#232F3E; font-family:Amazon Ember, Arial, sans-serif;'>{text}</h4>"
    

def create_footer():
    """Create the application footer with AWS copyright"""
    st.markdown(
        f"""
        <style>
        .footer-container {{
            background-color: {AWS_COLORS['secondary']};
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-top: 2rem;
            text-align: center;
        }}
        .footer-text {{
            color: white;
            font-size: 0.8rem;
        }}
        </style>
        <div class="footer-container">
            <div class="footer-text">© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
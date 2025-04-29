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

def apply_custom_styles():
    # AWS color scheme
    aws_colors = {
        'primary': '#FF9900',
        'secondary': '#232F3E',
        'accent1': '#1E88E5',
        'accent2': '#00A1C9',
        'text': '#232F3E',
        'background': '#FFFFFF',
    }
    
    # Apply custom CSS
    st.markdown(
        f"""
        <style>
        /* General styling */
        .stApp {{
            color: {aws_colors['text']};
            background-color: {aws_colors['background']};
            font-family: 'Amazon Ember', Arial, sans-serif;
        }}
        
        /* Headers styling */
        h1, h2, h3 {{
            color: {aws_colors['secondary']};
        }}
        
        /* Button styling */
        .stButton>button {{
            background-color: {aws_colors['accent2']};
            color: white;
            border-radius: 4px;
            border: none;
            padding: 0.5rem 1rem;
        }}
        
        .stButton>button:hover {{
            background-color: {aws_colors['accent1']};
        }}
        
        /* Success message styling */
        .stSuccess {{
            background-color: #D4EDDA;
            color: #155724;
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }}
        
        /* Table styling */
        .dataframe {{
            border-collapse: collapse;
            width: 100%;
        }}
        
        .dataframe thead th {{
            background-color: {aws_colors['secondary']};
            color: white;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 16px;
            background-color: #f8f9fa;
            border-radius: 4px 4px 0 0;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {aws_colors['accent2']} !important;
            color: white !important;
        }}
        
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #EAEDED;
            color: #232F3E;
            text-align: center;
            padding: 10px 0;
            font-size: 12px;
        }}       
        .main-header {{
            font-size: 2.5rem;
            color: #232F3E;
            font-weight: 700;
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: #FF9900;
            font-weight: 600;
        }}


        .header-container {{
            background-color: {AWS_COLORS['secondary']};
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }}
        .header-title {{
            color: {AWS_COLORS['primary']};
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        .header-subtitle {{
            color: white;
            font-size: 1.2rem;
            font-style: italic;
        }}

        .footer {{
            font-size: 0.8rem;
            color: #666;
            text-align: center;
            margin-top: 3rem;
            border-top: 1px solid #ddd;
            padding-top: 1rem;
        }}
        .feature-card {{
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #f5f5f5;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .info-box {{
            padding: 1rem;
            border-left: 4px solid #FF9900;
            background-color: rgba(255, 153, 0, 0.1);
            margin: 1rem 0;
        }}
        .aws-button {{
            background-color: #FF9900;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
        }}
        .aws-button:hover {{
            background-color: #EC7211;
        }}
        .cluster-0 {{color: #FF9900;}}
        .cluster-1 {{color: #232F3E;}}
        .cluster-2 {{color: #1DC7EA;}}
        .cluster-3 {{color: #87D068;}}       
        
        </style>
        """,
        unsafe_allow_html=True,
    )
    
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
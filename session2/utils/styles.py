import streamlit as st

# AWS Color Palette (Light Theme)
AWS_COLORS = {
    'primary': '#FF9900',       # AWS Orange
    'secondary': '#232F3E',     # AWS Navy
    'accent': '#0073BB',        # AWS Blue
    'success': '#3EB489',       # Success Green
    'info': '#16B9D4',          # Info Blue
    'warning': '#F2C94C',       # Warning Yellow
    'danger': '#D13212',        # Danger Red
    'background': '#FFFFFF',    # Background White
    'text': '#232F3E',          # Text Dark Navy
    'text_light': '#5A6D87',    # Light Text
    'border': '#E9EBF0',        # Light Border
    'hover': '#FFF5E6',         # Hover State (Light Orange)
    'shadow': 'rgba(0, 0, 0, 0.05)',  # Shadow Color
    'code_bg': '#F8F9FB',       # Code Background
    'header': '#0066cc'         # header Blue
}


# AWS color scheme
aws_orange = "#FF9900"
aws_dark = "#232F3E"
aws_blue = "#1A73E8"
aws_background = "#FFFFFF"

def load_css():
    """Apply the AWS-themed styling to the current Streamlit application."""
    
    # Define CSS
    st.markdown(f"""
    <style>
    /* ===============================
       GLOBAL STYLES
       =============================== */
    
    /* Main styles and typography */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif;
        color: {AWS_COLORS['text']};
        background-color: {AWS_COLORS['background']};
    }}
    
    /* Header styles */
    h1, h2, h3, h4, h5, h6 {{
        color: {AWS_COLORS['secondary']};
        font-weight: 600;
    }}
    
    h1 {{
        font-size: 2.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid {AWS_COLORS['border']};
        padding-bottom: 0.5rem;
        color: {AWS_COLORS['header']};
    }}
    
    h2 {{
        font-size: 1.75rem;
        margin-top: 1.75rem;
        margin-bottom: 1rem;
        color: {AWS_COLORS['header']};
    }}
    
    h3 {{
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        color: {AWS_COLORS['header']};
    }}
    
    p {{
        margin-bottom: 1rem;
        line-height: 1.6;
    }}
    
    a {{
        color: {AWS_COLORS['accent']};
        text-decoration: none;
        transition: color 0.2s ease;
    }}
    
    a:hover {{
        color: {AWS_COLORS['primary']};
        text-decoration: underline;
    }}
    
    /* ===============================
       STREAMLIT-SPECIFIC COMPONENTS
       =============================== */
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {AWS_COLORS['background']};
        border-right: 1px solid {AWS_COLORS['border']};
        padding: 1rem;
    }}
    
    [data-testid="stSidebar"] [data-testid="stImage"] {{
        text-align: center;
        margin-bottom: 1rem;
    }}
    
    [data-testid="stSidebar"] hr {{
        margin: 1rem 0;
        border: none;
        border-top: 1px solid {AWS_COLORS['border']};
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {AWS_COLORS['primary']};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px {AWS_COLORS['shadow']};
    }}
    
    .stButton > button:hover {{
        background-color: #E68A00; /* Darker shade of primary */
        box-shadow: 0 4px 8px {AWS_COLORS['shadow']};
        transform: translateY(-1px);
    }}
    
    .stButton > button:active {{
        transform: translateY(1px);
        box-shadow: 0 1px 3px {AWS_COLORS['shadow']};
    }}
    
    /* Secondary button variant */
    .stButton.secondary > button {{
        background-color: {AWS_COLORS['background']};
        color: {AWS_COLORS['primary']};
        border: 1px solid {AWS_COLORS['primary']};
    }}
    
    .stButton.secondary > button:hover {{
        background-color: {AWS_COLORS['hover']};
    }}
    
    /* Danger button variant */
    .stButton.danger > button {{
        background-color: {AWS_COLORS['danger']};
    }}
    
    .stButton.danger > button:hover {{
        background-color: #B82C10; /* Darker shade of danger */
    }}
    
    /* Form inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {{
        border: 1px solid {AWS_COLORS['border']};
        border-radius: 4px;
        padding: 0.5rem;
        transition: border-color 0.3s ease;
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {AWS_COLORS['primary']};
        box-shadow: 0 0 0 2px rgba(255, 153, 0, 0.2);
    }}
    
    /* Selectbox and MultiSelect */
    .stSelectbox > div[data-baseweb="select"] > div,
    .stMultiSelect > div[data-baseweb="select"] > div {{
        border: 1px solid {AWS_COLORS['border']};
        border-radius: 4px;
        transition: border-color 0.3s ease;
    }}
    
    .stSelectbox > div[data-baseweb="select"] > div:focus-within,
    .stMultiSelect > div[data-baseweb="select"] > div:focus-within {{
        border-color: {AWS_COLORS['primary']};
        box-shadow: 0 0 0 2px rgba(255, 153, 0, 0.2);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1px;
        border-bottom: 1px solid {AWS_COLORS['border']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {AWS_COLORS['background']};
        border-radius: 4px 4px 0 0;
        color: {AWS_COLORS['text_light']};
        font-weight: 600;
        transition: all 0.2s ease;
        border: none;
        padding: 0 8px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {AWS_COLORS['accent']};
        color: white;
        font-weight: 1000;
        border-bottom: 2px solid {AWS_COLORS['primary']};
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {AWS_COLORS['hover']};
        color: {AWS_COLORS['primary']};
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        padding: 1rem 0.5rem;
    }}
    
    /* Checkboxes and Radio buttons */
    .stCheckbox label,
    .stRadio label {{
        color: {AWS_COLORS['text']};
        font-weight: 400;
        display: flex;
        align-items: center;
    }}
    
    /* Info, Warning, Error, and Success boxes */
    /* Info box */
    .stAlert.info {{
        background-color: #EBF8FF;
        color: {AWS_COLORS['info']};
        border-left: 4px solid {AWS_COLORS['info']};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Success box */
    .stAlert.success {{
        background-color: #E6F7EF;
        color: {AWS_COLORS['success']};
        border-left: 4px solid {AWS_COLORS['success']};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Warning box */
    .stAlert.warning {{
        background-color: #FFF9E6;
        color: #B7921E;
        border-left: 4px solid {AWS_COLORS['warning']};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Error box */
    .stAlert.error {{
        background-color: #FDEDEB;
        color: {AWS_COLORS['danger']};
        border-left: 4px solid {AWS_COLORS['danger']};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Code blocks */
    code {{
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.9em;
        padding: 0.2em 0.4em;
        background-color: {AWS_COLORS['code_bg']};
        border-radius: 3px;
        border: 1px solid {AWS_COLORS['border']};
    }}
    
    pre {{
        background-color: {AWS_COLORS['code_bg']};
        border: 1px solid {AWS_COLORS['border']};
        border-radius: 4px;
        padding: 1rem;
        overflow-x: auto;
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.9em;
        line-height: 1.5;
    }}
    
    pre code {{
        background: none;
        border: none;
        padding: 0;
    }}
    
    /* Data display: Dataframes, tables */
    .stDataFrame, div[data-testid="stTable"] {{
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid {AWS_COLORS['border']};
    }}
    
    .stDataFrame th, div[data-testid="stTable"] th {{
        # background-color: {AWS_COLORS['secondary']};
        color: {AWS_COLORS['secondary']};
        font-weight: 600;
        text-align: left;
        padding: 0.75rem;
    }}
    
    .stDataFrame td, div[data-testid="stTable"] td {{
        padding: 0.5rem 0.75rem;
        border-top: 1px solid {AWS_COLORS['border']};
    }}
    
    .stDataFrame tr:nth-child(even), div[data-testid="stTable"] tr:nth-child(even) {{
        background-color: {AWS_COLORS['code_bg']};
    }}
    
    /* Metrics */
    [data-testid="stMetric"] {{
        background-color: {AWS_COLORS['background']};
        border-radius: 4px;
        padding: 1rem;
        border: 1px solid {AWS_COLORS['border']};
        box-shadow: 0 2px 5px {AWS_COLORS['shadow']};
    }}
    
    [data-testid="stMetric"] label {{
        color: {AWS_COLORS['text_light']};
        font-size: 0.875rem;
    }}
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 600;
        color: {AWS_COLORS['secondary']};
    }}
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        font-size: 0.875rem;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        font-size: 1rem;
        font-weight: 600;
        color: {AWS_COLORS['text']};
        background-color: {AWS_COLORS['background']};
        border: 1px solid {AWS_COLORS['border']};
        border-radius: 4px;
        padding: 0.75rem 1rem;
        transition: background-color 0.2s ease;
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: {AWS_COLORS['hover']};
    }}
    
    .streamlit-expanderContent {{
        border: 1px solid {AWS_COLORS['border']};
        border-top: none;
        border-radius: 0 0 4px 4px;
        padding: 1rem;
        background-color: {AWS_COLORS['background']};
    }}
    
    /* Tooltips */
    .stTooltipIcon {{
        color: {AWS_COLORS['text_light']};
    }}
    
    /* Footer */
    footer {{
        background-color: {AWS_COLORS['background']};
        border-top: 1px solid {AWS_COLORS['border']};
        padding: 1rem;
        font-size: 1rem;
        color: {AWS_COLORS['text_light']};
        text-align: left;
        margin-top: 2rem;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background-color: {AWS_COLORS['primary']};
    }}
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] {{
        margin-top: 1rem;
        margin-bottom: 1rem;
    }}
    
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {{
        # background-color: {AWS_COLORS['primary']};
        border: 2px solid white;
    }}
    
    .stSlider [data-baseweb="slider"] div:nth-child(3) {{
        background: {AWS_COLORS['primary']};
    }}
    
    /* Download button */
    .stDownloadButton > button {{
        background-color: {AWS_COLORS['accent']};
        color: white;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .stDownloadButton > button:hover {{
        background-color: #0062A3; /* Darker shade of accent */
    }}
    
    /* ===============================
       RESPONSIVE DESIGN
       =============================== */
    
    @media (max-width: 768px) {{
        h1 {{
            font-size: 1.75rem;
        }}
        
        h2 {{
            font-size: 1.5rem;
        }}
        
        h3 {{
            font-size: 1.25rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 0 12px;
        }}
        
        [data-testid="stMetric"] {{
            padding: 0.75rem;
        }}
        
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            font-size: 1.5rem;
        }}
    }}
    
    @media (max-width: 576px) {{
        h1 {{
            font-size: 1.5rem;
        }}
        
        .stButton > button,
        .stDownloadButton > button {{
            width: 100%;
        }}
        
        [data-testid="stSidebar"] {{
            width: 100%;
            margin-bottom: 1rem;
        }}
    }}
    
    
        /* Info boxes */
    .info-box {{
        background-color: #f0f7ff;
        border-left: 5px solid #0066cc;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }}
    
    .card {{
        border: 1px solid {AWS_COLORS['border']};
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        background-color: {AWS_COLORS['background']};
    }}
    
    
    

    
    
    .stApp {{
        color: {aws_dark};
        background-color: {aws_background};
        font-family: 'Amazon Ember', Arial, sans-serif;
    }}
    
   
    /* AWS themed styling */
    .stButton>button {{
        background-color: {aws_blue};
        color: white;
    }}
    
    .stButton>button:hover {{
        background-color: {aws_dark};
    }}
    
    /* Success styling */
    .correct-answer {{
        background-color: #D4EDDA;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    
    /* Error styling */
    .incorrect-answer {{
        background-color: #F8D7DA;
        color: #721C24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    
    /* Custom card styling */
    .game-card {{
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    /* Progress bar styling */
    .stProgress > div > div > div {{
        background-color: {aws_orange};
    }}
    
    /* Tip box styling */
    .tip-box {{
        background-color: #E7F3FE;
        border-left: 6px solid {aws_blue};
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }}
    
    /* Make images responsive */
    img {{
        max-width: 100%;
        height: auto;
    }}
    
    /* Score display */
    .score-display {{
        font-size: 1.2rem;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 15px 0;
        background-color: #F1F8FF;
    }}    
    
    /* Header container for page titles */
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
    
    /* Risk score boxes */
    .risk-score-high {{
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }}
    
    .risk-score-high h3 {{
        color: #c62828;
        margin: 0;
    }}
    
    .risk-score-low {{
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }}
    
    .risk-score-low h3 {{
        color: #2e7d32;
        margin: 0;
    }}
    
    .risk-score-value {{
        font-size: 24px;
        margin-top: 10px;
    }}
    
    /* Form headers */
    .form-header {{
        background-color: {AWS_COLORS['accent']};
        padding: 0.5rem;
        border-radius: 0.3rem;
        color: white;
        margin-bottom: 0.5rem;
    }}
    
    /* Prediction headers */
    .prediction-header {{
        background-color: {AWS_COLORS['secondary']};
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1.5rem;
        text-align: center;
    }}
    
    /* Warning boxes */
    .warning-box {{
        background-color: #FFF9E6;
        border-left: 5px solid {AWS_COLORS['primary']};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }}
    
    /* Recommendations */
    .recommendations {{
        background-color: #F3FFFC;
        border-left: 5px solid {AWS_COLORS['success']};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }}
    
    /* Disclaimer */
    .disclaimer {{
        background-color: #F0F7FF;
        border-left: 5px solid {AWS_COLORS['info']};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        color: {AWS_COLORS['text_light']};
    }}
    
    /* Stats box */
    .stats-box {{
        background-color: #F7F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid {AWS_COLORS['border']};
    }}
    
    /* Patient profile */
    .patient-profile {{
        background-color: #F7F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid {AWS_COLORS['border']};
        margin-bottom: 1rem;
    }}
    
    /* Best model highlight */
    .best-model {{
        background-color: #FFF9E6 !important;
        font-weight: bold;
    }}
    
    /* Metric explanation */
    .metric-explanation {{
        padding: 10px;
        background-color: {AWS_COLORS['code_bg']};
        border-radius: 0.3rem;
    }}
    
    /* CM interpretation */
    .cm-interpretation {{
        background-color: #F7F9FA;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }}
    
    /* ROC explanation */
    .roc-explanation {{
        background-color: #F7F9FA;
        padding: 1rem;
        border-radius: 0.3rem;
    }}
    
    /* Feature explanation */
    .feature-explanation {{
        background-color: #F7F9FA;
        padding: 1rem;
        border-radius: 0.3rem;
    }}
    
    /* SHAP explanation */
    .shap-explanation {{
        background-color: #F7F9FA;
        padding: 1rem;
        border-radius: 0.3rem;
    }}
    
    /* Intro header */
    .intro-header {{
        color: {AWS_COLORS['primary']};
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }}
    
    /* About header */
    .about-header {{
        color: {AWS_COLORS['primary']};
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }}
    
    /* Footer container */
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
    
    /* ===============================
       SUB-HEADER STYLES
       =============================== */
    
    /* Sub-header styling */
    .sub-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.3px;
    }}
    
    .sub-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }}
    
    .sub-header:hover::before {{
        left: 100%;
    }}
    
    /* Alternative sub-header styles */
    .sub-header.style-aws {{
        background: linear-gradient(135deg, {AWS_COLORS['primary']} 0%, {AWS_COLORS['accent']} 100%);
        box-shadow: 0 4px 12px rgba(255, 153, 0, 0.3);
    }}
    
    .sub-header.style-minimal {{
        background: {AWS_COLORS['background']};
        color: {AWS_COLORS['secondary']};
        border-left: 5px solid {AWS_COLORS['primary']};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}
    
    .sub-header.style-outline {{
        background: transparent;
        color: {AWS_COLORS['secondary']};
        border: 2px solid {AWS_COLORS['primary']};
        box-shadow: none;
    }}
    
    /* ===============================
       TEMPLATE CARD STYLES
       =============================== */
    
    /* Template card container */
    .template-card {{
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fb 100%);
        border: 2px solid {AWS_COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    
    .template-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, {AWS_COLORS['primary']} 0%, {AWS_COLORS['accent']} 100%);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }}
    
    .template-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        border-color: {AWS_COLORS['primary']};
    }}
    
    .template-card:hover::before {{
        transform: scaleX(1);
    }}
    
    /* Selected template card */
    .template-card.template-selected {{
        background: linear-gradient(135deg, #fff5e6 0%, #ffe8cc 100%);
        border: 2px solid {AWS_COLORS['primary']};
        box-shadow: 0 4px 16px rgba(255, 153, 0, 0.2);
    }}
    
    .template-card.template-selected::before {{
        transform: scaleX(1);
        height: 6px;
    }}
    
    /* Template icon */
    .template-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        text-align: center;
        filter: grayscale(0.3);
        transition: all 0.3s ease;
    }}
    
    .template-card:hover .template-icon {{
        filter: grayscale(0);
        transform: scale(1.1);
    }}
    
    .template-card.template-selected .template-icon {{
        filter: grayscale(0);
        transform: scale(1.05);
    }}
    
    /* Template title */
    .template-title {{
        color: {AWS_COLORS['secondary']};
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-align: center;
        line-height: 1.3;
        transition: color 0.3s ease;
    }}
    
    .template-card:hover .template-title {{
        color: {AWS_COLORS['primary']};
    }}
    
    .template-card.template-selected .template-title {{
        color: {AWS_COLORS['primary']};
    }}
    
    /* Template description */
    .template-description {{
        color: {AWS_COLORS['text_light']};
        font-size: 0.9rem;
        line-height: 1.5;
        text-align: center;
        margin: 0;
        flex-grow: 1;
    }}
    
    /* Responsive adjustments for template cards */
    @media (max-width: 768px) {{
        .template-card {{
            min-height: 160px;
            padding: 1.25rem;
        }}
        
        .template-icon {{
            font-size: 2rem;
        }}
        
        .template-title {{
            font-size: 1rem;
        }}
        
        .template-description {{
            font-size: 0.85rem;
        }}
    }}
    
    
    </style>
    """, unsafe_allow_html=True)

    # Add some helper methods to display styled elements
    return {
        'colors': AWS_COLORS,
    }

def create_info_box(message):
    """Create a custom info box with AWS styling."""
    st.markdown(f"""
    <div class="stAlert info">
        <strong>Info:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def create_success_box(message):
    """Create a custom success box with AWS styling."""
    st.markdown(f"""
    <div class="stAlert success">
        <strong>Success:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def create_warning_box(message):
    """Create a custom warning box with AWS styling."""
    st.markdown(f"""
    <div class="stAlert warning">
        <strong>Warning:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def create_error_box(message):
    """Create a custom error box with AWS styling."""
    st.markdown(f"""
    <div class="stAlert error">
        <strong>Error:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def create_footer(text="© 2023 AWS-Themed App. All rights reserved."):
    """Create a custom footer with AWS styling."""
    st.markdown(f"""
    <footer>
        {text}
    </footer>
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


def sub_header(text, icon="", style="default"):
    """
    Create a styled sub-header with optional icon and style variant.
    
    Args:
        text (str): The header text
        icon (str): Optional emoji icon to display before text
        style (str): Style variant - "default", "aws", "minimal", or "outline"
    
    Returns:
        str: HTML string for the sub-header
    
    Examples:
        st.markdown(sub_header("Model Settings", "⚙️", "aws"), unsafe_allow_html=True)
        st.markdown(sub_header("Choose a Template", "🎯"), unsafe_allow_html=True)
    """
    style_class = f"style-{style}" if style != "default" else ""
    icon_html = f"{icon} " if icon else ""
    
    return f"<div class='sub-header {style_class}'>{icon_html}{text}</div>"

import uuid
import streamlit as st

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if 'tab_selection' not in st.session_state:
        st.session_state.tab_selection = 0

def reset_session():
    """Reset all session state variables except session_id"""
    session_id = st.session_state.session_id
    st.session_state.clear()
    st.session_state.session_id = session_id
    initialize_session_state()
    st.rerun()

def create_sidebar():
    """Create and configure the sidebar"""
    with st.sidebar:
        with st.expander("About this App", expanded=False):
            st.write("""
            This application demonstrates unsupervised machine learning through customer segmentation.
            
            **Topics covered:**
            - Unsupervised Learning vs. Other ML Types
            - K-means Clustering Algorithm
            - Customer Segmentation Use Case
            - Feature Engineering for Customer Data
            - Model Evaluation for Clustering
            
            Based on AWS Partner Certification Readiness materials for AI Practitioner.
            """)
        
def display_footer():
    """Display the AWS footer"""
    st.markdown("""
    <div class="footer">
        © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
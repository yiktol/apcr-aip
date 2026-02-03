
import streamlit as st
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from utils.knowledge_check import display_knowledge_check, reset_knowledge_check
from utils.styles import load_css, custom_header
from utils.common import render_sidebar
import utils.authenticate as authenticate

# Set page configuration
st.set_page_config(
    page_title="AWS AI Practitioner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session Management
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def main():
    # Apply custom styling
    load_css()

    # Sidebar
    with st.sidebar:
        render_sidebar()
        
        # About this App (collapsible)
        with st.expander("ℹ️ About this App", expanded=False):
            st.write("""
            This interactive e-learning app helps you prepare for the AWS AI Practitioner certification.
            
            **Topics covered:**
            - AI, ML and Generative AI concepts
            - Traditional vs ML approaches
            - Machine learning types
            - Common use cases
            - ML process overview
            - AWS AI/ML stack
            - Knowledge check
            """)


    # Home - Program Overview
    st.markdown(custom_header("AWS AI Practitioner Certification Readiness", 1 ), unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Program Overview")
        st.write("""
        Welcome to the AWS AI Practitioner certification preparation program! 
        
        This program will help you understand the fundamentals of Artificial Intelligence, 
        Machine Learning, and Generative AI concepts on AWS.
        
        **Learning Outcomes:**
        - Understand the difference between AI, ML, and Generative AI
        - Learn when to use machine learning vs traditional programming
        - Explore different ML types and common use cases
        - Master the ML development lifecycle
        - Familiarize yourself with the AWS AI/ML stack
        """)
        
        st.subheader("Program Structure")
        program_data = {
            "Session": ["Session 1", "Session 2", "Session 3", "Session 4"],
            "Content": [
                "Kickoff & Fundamentals of AI and ML",
                "Fundamentals of Generative AI",
                "Applications of Foundation Models",
                "Responsible AI & Security, Compliance, and Governance"
            ]
        }
        st.table(pd.DataFrame(program_data))
    
    with col2:
        st.image("assets/images/AWS-Certified-AI-Practitioner_badge.png", width=300)
        st.caption("AWS Certified AI Practitioner ")
        
        progress = st.progress(0)
        st.info("Navigate through the tabs at the top to explore all topics!")

    # Footer
    st.markdown("""
    <div class="footer">
        © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

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
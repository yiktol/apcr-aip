
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
    st.markdown(custom_header("Session 1: Fundamentals of AI and ML", 1), unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to Session 1")
        st.write("""
        This session covers the fundamentals of Artificial Intelligence and Machine Learning, 
        providing you with essential knowledge for the AWS AI Practitioner certification.
        
        **Learning Outcomes:**
        - Understand core ML terminology and concepts
        - Explore different types of machine learning algorithms
        - Learn about supervised, unsupervised, and reinforcement learning
        - Discover real-world ML applications across industries
        - Test your knowledge with interactive games and quizzes
        """)
    
    with col2:
        st.image("assets/images/AWS-Certified-AI-Practitioner_badge.png", width=300)
        st.caption("AWS Certified AI Practitioner")
        st.info("Navigate through the pages in the sidebar to explore all topics!")
    
    # Session 1 Topics Outline
    st.markdown("---")
    st.header("📚 Session 1 Topics Overview")
    
    # Create tabs for different topic categories
    tab1, tab2, tab3 = st.tabs(["🎯 Core Concepts", "🔬 ML Algorithms", "🏢 Real-World Applications"])
    
    with tab1:
        st.subheader("Foundational Machine Learning Concepts")
        
        topics_core = [
            {
                "icon": "📊",
                "title": "ML Terminology",
                "description": "Essential ML vocabulary including features, labels, training, testing, overfitting, underfitting, and model evaluation metrics"
            },
            {
                "icon": "🎓",
                "title": "Learning Types",
                "description": "Comprehensive overview of supervised, unsupervised, semi-supervised, and reinforcement learning paradigms"
            },
            {
                "icon": "🎮",
                "title": "ML Learning Games",
                "description": "Interactive games covering AI vs ML vs GenAI, traditional vs ML approaches, ML terminology, learning types, ML process, and AWS services"
            }
        ]
        
        for topic in topics_core:
            with st.container():
                st.markdown(f"""
                <div style='padding: 15px; margin: 10px 0; background-color: #f0f2f6; border-radius: 5px; border-left: 4px solid #FF9900;'>
                    <h4>{topic['icon']} {topic['title']}</h4>
                    <p style='margin: 5px 0;'>{topic['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Machine Learning Algorithms & Techniques")
        
        topics_algorithms = [
            {
                "icon": "🔵",
                "title": "Binary Classification",
                "description": "Loan approval prediction using logistic regression - classify outcomes into two categories (approved/denied)"
            },
            {
                "icon": "🎨",
                "title": "Multi-Class Classification",
                "description": "Iris flower species classification using Random Forest - categorize data into multiple distinct classes"
            },
            {
                "icon": "📈",
                "title": "Regression",
                "description": "House price prediction using multiple regression algorithms - predict continuous numerical values"
            },
            {
                "icon": "🔍",
                "title": "Clustering",
                "description": "Customer segmentation using K-Means - discover natural groupings in unlabeled data"
            },
            {
                "icon": "🤖",
                "title": "Reinforcement Learning",
                "description": "Interactive RL demonstration with Q-learning - learn optimal actions through trial and error with rewards"
            }
        ]
        
        for topic in topics_algorithms:
            with st.container():
                st.markdown(f"""
                <div style='padding: 15px; margin: 10px 0; background-color: #f0f2f6; border-radius: 5px; border-left: 4px solid #232F3E;'>
                    <h4>{topic['icon']} {topic['title']}</h4>
                    <p style='margin: 5px 0;'>{topic['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Industry Use Cases & Applications")
        
        topics_usecases = [
            {
                "icon": "💳",
                "title": "Financial Fraud Detection",
                "description": "Anomaly detection in financial transactions using ensemble methods, SMOTE for imbalanced data, and SHAP for explainability"
            },
            {
                "icon": "🏥",
                "title": "Healthcare Diagnostics",
                "description": "Disease prediction and medical diagnosis using classification algorithms with cross-validation and feature importance analysis"
            },
            {
                "icon": "🔒",
                "title": "Cybersecurity Threat Detection",
                "description": "Network intrusion detection using Isolation Forest for anomaly detection in network traffic patterns"
            },
            {
                "icon": "🏭",
                "title": "Manufacturing Quality Control",
                "description": "Defect detection in manufacturing processes using Random Forest to identify product quality issues"
            },
            {
                "icon": "🚗",
                "title": "Autonomous Vehicles",
                "description": "Object detection for self-driving cars - identify vehicles, pedestrians, and traffic signs using computer vision and ML"
            }
        ]
        
        for topic in topics_usecases:
            with st.container():
                st.markdown(f"""
                <div style='padding: 15px; margin: 10px 0; background-color: #f0f2f6; border-radius: 5px; border-left: 4px solid #1E8E3E;'>
                    <h4>{topic['icon']} {topic['title']}</h4>
                    <p style='margin: 5px 0;'>{topic['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    

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
import streamlit as st
import uuid
from datetime import datetime
import random
from utils.common import apply_styles
from utils.common import initialize_session_state, reset_session
from utils.game_ai_ml_genai import ai_ml_genai_game
from utils.game_traditional_vs_ml import traditional_vs_ml_game
from utils.game_ml_or_not import ml_or_not_game
from utils.game_traditional_ml_vs_genai import traditional_ml_vs_genai_game
from utils.game_ml_terms import ml_terms_game
from utils.game_learning_types import learning_types_game
from utils.game_ml_process import ml_process_game
from utils.game_aws_services import aws_services_game

def main():
    # Apply custom styling
    apply_styles()
    
    # Initialize session state variables
    initialize_session_state()

    # Page title
    st.title("🎮 AWS AI Practitioner Learning Games")
    st.write("Test your knowledge on AI and ML concepts to prepare for AWS AI Practitioner certification")
    
    # Sidebar
    with st.sidebar:
        
        st.subheader("Session Management")
        
        st.info(f"User ID: {st.session_state.session_id}")
        st.write(f"Session started: {st.session_state.start_time}")
        
        if st.button("🔄 Reset Session", key="reset_button"):
            reset_session()
            st.rerun()
        
        with st.expander("ℹ️ About this App", expanded=False):
            st.markdown("""
            This interactive application helps you prepare for the AWS AI Practitioner certification by testing your knowledge through fun games covering:
            
            - AI, ML, and Generative AI differences
            - Traditional programming vs ML
            - ML use case identification
            - Traditional ML vs Generative AI
            - ML terminology
            - Types of ML learning
            - ML development process
            - AWS AI Services
            """)
        
    
    # Tab-based navigation
    tabs = st.tabs([
        "🤖 AI vs ML vs GenAI", 
        "💻 Traditional vs ML", 
        "🎯 ML or Not?", 
        "⚔️ Traditional ML vs GenAI", 
        "📚 ML Terms", 
        "🧠 Learning Types",
        "🔄 ML Process",
        "☁️ AWS AI Services"
    ])
    
    # Tab 1: AI, ML, or Generative AI?
    with tabs[0]:
        ai_ml_genai_game()
    
    # Tab 2: Traditional Programming vs ML
    with tabs[1]:
        traditional_vs_ml_game()
    
    # Tab 3: ML or Not?
    with tabs[2]:
        ml_or_not_game()
    
    # Tab 4: Traditional ML vs Generative AI
    with tabs[3]:
        traditional_ml_vs_genai_game()
    
    # Tab 5: Identify ML Terms
    with tabs[4]:
        ml_terms_game()
    
    # Tab 6: Match ML Learning Types
    with tabs[5]:
        learning_types_game()
    
    # Tab 7: Identify the ML Process
    with tabs[6]:
        ml_process_game()
    
    # Tab 8: Select the correct AWS AI Services
    with tabs[7]:
        aws_services_game()
    
    # Footer
    st.markdown("---")
    st.caption("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")

if __name__ == "__main__":
    main()

import streamlit as st
import uuid

def initialize_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if "knowledge_check_started" not in st.session_state:
        st.session_state.knowledge_check_started = False
    
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    
    if "score" not in st.session_state:
        st.session_state.score = 0

def reset_session():
    """Reset the session state"""
    for key in st.session_state.keys():
        if key != "session_id":
            del st.session_state[key]
    
    # Re-initialize with default values
    st.session_state.knowledge_check_started = False
    st.session_state.current_question = 0
    st.session_state.answers = {}
    st.session_state.score = 0
    st.session_state.knowledge_check_started = False
    st.session_state.knowledge_check_progress = 0
    st.session_state.knowledge_check_answers = {}
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.conversation_history = []
    st.session_state.messages = []
    
    
def render_sidebar():
    """Render the sidebar with session information and reset button"""
    st.markdown("#### 🔑 Session Info")
    st.caption(f"**Session ID:** {st.session_state.session_id[:8]}")

    if st.button("🔄 Reset Session"):
        reset_session()
        st.success("Session has been reset successfully!")
        st.rerun()  # Force a rerun to refresh the page

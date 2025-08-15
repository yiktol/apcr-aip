import streamlit as st
import uuid

def initialize_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
    if "user_cognito_groups" not in st.session_state:
        st.session_state.user_cognito_groups = []

    if "auth_code" not in st.session_state:
        st.session_state.auth_code = ""

def reset_session():
    """Reset the session state"""
    for key in st.session_state.keys():
        if key not in ["authenticated", "user_cognito_groups", "auth_code","user_info"]:
            del st.session_state[key]
    
def render_sidebar():
    """Render the sidebar with session information and reset button"""
    st.markdown("#### 🔑 Session Info")
    # st.caption(f"**Session ID:** {st.session_state.session_id[:8]}")
    st.caption(f"**Session ID:** {st.session_state['auth_code'][:8]}")

    if st.button("🔄 Reset Session",use_container_width=True ):
        reset_session()
        st.success("Session has been reset successfully!")
        st.rerun()  # Force a rerun to refresh the page

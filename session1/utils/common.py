import streamlit as st
import uuid
from datetime import datetime
import utils.authenticate as authenticate

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
    # st.caption(f"**Session ID:** {st.session_state.session_id[:8]}")
    st.caption(f"**Session ID:** {st.session_state["auth_code"][:8]}")
    

    if st.button("🔄 Reset Session"):
        reset_session()
        st.success("Session has been reset successfully!")
        st.rerun()  # Force a rerun to refresh the page





def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Game 1: AI vs ML vs GenAI
    if "game1_score" not in st.session_state:
        st.session_state.game1_score = 0
    if "game1_submitted" not in st.session_state:
        st.session_state.game1_submitted = [False] * 5

    # Game 2: Traditional Programming vs ML
    if "game2_score" not in st.session_state:
        st.session_state.game2_score = 0
    if "game2_submitted" not in st.session_state:
        st.session_state.game2_submitted = [False] * 5
    
    # Game 3: ML or Not?
    if "game3_score" not in st.session_state:
        st.session_state.game3_score = 0
    if "game3_submitted" not in st.session_state:
        st.session_state.game3_submitted = [False] * 5
    
    # Game 4: Traditional ML vs GenAI
    if "game4_score" not in st.session_state:
        st.session_state.game4_score = 0
    if "game4_submitted" not in st.session_state:
        st.session_state.game4_submitted = [False] * 5
    
    # Game 5: ML Terms
    if "game5_score" not in st.session_state:
        st.session_state.game5_score = 0
    if "game5_submitted" not in st.session_state:
        st.session_state.game5_submitted = [False] * 5
    
    # Game 6: Learning Types
    if "game6_score" not in st.session_state:
        st.session_state.game6_score = 0
    if "game6_submitted" not in st.session_state:
        st.session_state.game6_submitted = [False] * 5
    
    # Game 7: ML Process
    if "game7_score" not in st.session_state:
        st.session_state.game7_score = 0
    if "game7_submitted" not in st.session_state:
        st.session_state.game7_submitted = [False] * 5
    
    # Game 8: AWS Services
    if "game8_score" not in st.session_state:
        st.session_state.game8_score = 0
    if "game8_submitted" not in st.session_state:
        st.session_state.game8_submitted = [False] * 5

# def reset_session():
#     """Reset all session state variables."""
    
#     # Keep only the session ID but generate a new one
#     st.session_state.session_id = str(uuid.uuid4())[:8]
#     st.session_state.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
#     # Reset all game scores and submissions
#     for i in range(1, 9):  # 8 games
#         st.session_state[f"game{i}_score"] = 0
#         st.session_state[f"game{i}_submitted"] = [False] * 5

def display_progress(game_number):
    """Display progress for the specified game."""
    
    completed = st.session_state[f"game{game_number}_submitted"].count(True)
    total = len(st.session_state[f"game{game_number}_submitted"])
    
    progress_text = f"Progress: {completed}/{total} scenarios completed"
    st.progress(completed / total)
    st.text(progress_text)
    
    score_percentage = (st.session_state[f"game{game_number}_score"] / total) * 100 if total > 0 else 0
    score_text = f"Current Score: {st.session_state[f'game{game_number}_score']}/{total} ({score_percentage:.1f}%)"
    
    st.markdown(f"""
    <div class="score-display">
        {score_text}
    </div>
    """, unsafe_allow_html=True)

def show_tip(tip_text):
    """Display a formatted tip box with the provided text."""
    
    st.markdown(f"""
    <div class="tip-box">
        <strong>💡 Learning Tip:</strong> {tip_text}
    </div>
    """, unsafe_allow_html=True)

def reset_game_button(game_number):
    """Create a reset button for the specified game."""
    
    if st.button(f"🔄 Reset Game {game_number}", key=f"reset_game_{game_number}"):
        st.session_state[f"game{game_number}_score"] = 0
        st.session_state[f"game{game_number}_submitted"] = [False] * 5
        st.rerun()
        
def apply_styles():
    """Apply custom styling to the Streamlit app."""
    
    # AWS color scheme
    aws_orange = "#FF9900"
    aws_dark = "#232F3E"
    aws_blue = "#1A73E8"
    aws_background = "#FFFFFF"
    
    # CSS for styling
    st.markdown(f"""
    <style>
    
        .stApp {{
            color: {aws_dark};
            background-color: {aws_background};
            font-family: 'Amazon Ember', Arial, sans-serif;
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
            background-color: {aws_orange} !important;
            color: white !important;
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
    </style>
    """, unsafe_allow_html=True)
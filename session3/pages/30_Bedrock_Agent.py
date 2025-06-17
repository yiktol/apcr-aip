"""
Bedrock Agent Streamlit Interface

This application provides a chat interface to interact with an AWS Bedrock Agent,
specifically designed for a retail shoe department customer assistant.

Features:
- Conversational UI with AWS Bedrock Agent integration
- Chat history tracking
- Session management
- Error handling and logging
"""

import streamlit as st
import uuid
import json
import logging
import boto3
from typing import Dict, List, Optional
from datetime import datetime
from botocore.exceptions import ClientError
import utils.common as common
import utils.authenticate as authenticate
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Shoe Department Assistant",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A4A4A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6F6F6F;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    .stChatMessage div {
        font-size: 1.1rem;
    }
    .stButton button {
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
    }
    .sidebar-info {
        padding: 1rem;
        background-color: #f0f5ff;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return super().default(obj)


class BedrockAgentChat:
    """Class to manage interactions with AWS Bedrock Agent"""
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize the Bedrock Agent client
        
        Args:
            region: AWS region for Bedrock Agent
        """
        try:
            self.client = boto3.client("bedrock-agent-runtime", region_name=region)
            logger.info(f"Successfully initialized Bedrock client in {region}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            st.error(f"Failed to connect to AWS Bedrock: {e}")
    
    def get_response(self, 
                    input_text: str, 
                    agent_id: str, 
                    agent_alias_id: str, 
                    session_id: str,
                    enable_trace: bool = False) -> Optional[str]:
        """
        Get a response from the Bedrock Agent
        
        Args:
            input_text: User's message
            agent_id: Bedrock Agent ID
            agent_alias_id: Bedrock Agent Alias ID
            session_id: Session identifier
            enable_trace: Whether to enable tracing
            
        Returns:
            Agent's response text or None if error occurs
        """
        try:
            logger.info(f"Sending request to Bedrock Agent: '{input_text[:50]}...'")
            
            response = self.client.invoke_agent(
                inputText=input_text,
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                enableTrace=enable_trace
            )
            
            return self._process_response_stream(response)
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_msg = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"AWS error: {error_code} - {error_msg}")
            return f"⚠️ Service error: {error_code}. Please try again later."
        
        except Exception as e:
            logger.error(f"Error getting response from Bedrock Agent: {e}")
            return f"⚠️ Something went wrong: {str(e)}"
    
    def _process_response_stream(self, response) -> str:
        """
        Process the streaming response from Bedrock Agent
        
        Args:
            response: Streaming response object from Bedrock Agent
            
        Returns:
            Decoded response text
        """
        event_stream = response.get('completion')
        full_response = []
        
        try:
            for event in event_stream:
                if 'chunk' in event:
                    data = event['chunk']['bytes']
                    chunk_text = data.decode('utf-8')
                    full_response.append(chunk_text)
                    # Uncomment for debugging:
                    # logger.debug(f"Received chunk: {chunk_text}")
                elif 'trace' in event:
                    # Convert trace data to JSON using our custom encoder
                    trace_data = json.dumps(event['trace'], indent=2, cls=DateTimeEncoder)
                    logger.debug(f"Trace data received")
                    if st.session_state.get('show_debug', False):
                        st.sidebar.text_area("Trace Data", trace_data, height=300)
                else:
                    logger.warning(f"Unexpected event type: {event}")
            
            return ''.join(full_response)
            
        except Exception as e:
            logger.error(f"Error processing response stream: {e}", exc_info=True)
            return f"⚠️ Error processing response: {str(e)}"


def initialize_session() -> None:
    """Initialize session state variables if they don't exist"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"New session created: {st.session_state.session_id}")
    
    if "chat_history" not in st.session_state:
        current_time = datetime.now().strftime("%H:%M")
        st.session_state.chat_history = [{
            "role": "assistant", 
            "text": "Hi, I'm your Shoe Department Assistant! How can I help you find the perfect shoes today?",
            "timestamp": current_time
        }]
    
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False


def render_sidebar() -> None:
    """Render the sidebar with settings and controls"""
    with st.sidebar:
        common.render_sidebar()
        st.write(st.session_state["user_cognito_groups"])
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            current_time = datetime.now().strftime("%H:%M")
            st.session_state.chat_history = [{
                "role": "assistant", 
                "text": "Chat history has been cleared. How can I help you?",
                "timestamp": current_time
            }]
            st.rerun()
    
    
    # st.sidebar.markdown("## 👟 Assistant Settings")
    
    # with st.sidebar.expander("Session Information", expanded=False):
    #     st.code(f"Session ID: {st.session_state.session_id}")
    #     st.text(f"Messages: {len(st.session_state.chat_history)}")
        
    if st.session_state.get('show_debug', False):
        try:
            history_json = json.dumps(
                st.session_state.chat_history,
                indent=2,
                cls=DateTimeEncoder
            )
            st.download_button(
                "Download Chat History", 
                history_json, 
                "chat_history.json", 
                "application/json"
            )
        except Exception as e:
            st.error(f"Error creating download: {e}")
    

    if st.session_state["user_cognito_groups"] == 'Admins':
        st.sidebar.markdown("### Debug Options")
        show_debug = st.sidebar.toggle("Show Debug Information", value=st.session_state.get('show_debug', False))
        if show_debug != st.session_state.get('show_debug'):
            st.session_state.show_debug = show_debug
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This assistant helps customers find the perfect shoes based on their preferences, "
        "activities, and needs. Ask about running shoes, casual footwear, or specific brands."
    )


def render_chat_interface() -> None:
    """Render the main chat interface"""
    # Header
    st.markdown("<h1 class='main-header'>Shoe Department Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Your virtual shopping assistant to find the perfect shoes</p>", unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(f"{message['text']}")
                if st.session_state.get('show_debug', False):
                    st.caption(f"Time: {message.get('timestamp', 'N/A')}")
    
    # Chat input
    user_input = st.chat_input("Ask about shoes, sizes, or anything shoe related...")
    
    if user_input:
        # Add user message to chat history with timestamp
        current_time = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({
            "role": "user", 
            "text": user_input,
            "timestamp": current_time
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent = BedrockAgentChat()
                
                # Agent configuration
                agent_id = 'W789VIBDQ3'
                agent_alias_id = 'TSTALIASID'
                session_id = st.session_state.session_id
                enable_trace = st.session_state.get('show_debug', False)
                
                # Get agent response
                response_text = agent.get_response(
                    input_text=user_input,
                    agent_id=agent_id,
                    agent_alias_id=agent_alias_id,
                    session_id=session_id,
                    enable_trace=enable_trace
                )
                
                # Display agent response
                st.write(response_text)
                
                # Add agent response to chat history
                current_time = datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "text": response_text,
                    "timestamp": current_time
                })


def main():
    """Main application function"""
    try:
        initialize_session()
        render_sidebar()
        render_chat_interface()
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
        st.button("Reload Application", on_click=lambda: st.rerun())


if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()

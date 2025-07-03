import os
import streamlit as st
import boto3
import uuid
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrock
import utils.common as common
import utils.authenticate as authenticate

# Configure the page
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def init_styles():
    """Apply custom styling to the app."""
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            display: flex;
        }
        .chat-message.user {
            background-color: #e3f2fd;
        }
        .chat-message.assistant {
            background-color: #f0f4c3;
        }
        .chat-header {
            position: sticky;
            background: linear-gradient(to right, #4776E6, #8E54E9);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
            top: 0;
        }
        .sidebar .sidebar-content {
            background-color: #f0f4f8;
        }
        /* Button styling */
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            font-weight: bold;
            background-color: #4CAF50;
            color: white;
        }
        /* Sidebar title styling */
        .sidebar-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 1.5rem;
            color: #333;
            text-align: center;
        }
        .memory-status {
            padding: 8px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            margin: 10px 0;
        }
        .memory-enabled {
            background-color: #c8e6c9;
            color: #2e7d32;
        }
        .memory-disabled {
            background-color: #ffcdd2;
            color: #c62828;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            font-size: 0.8em;
            color: #666;
        }
                /* Info boxes */
        .info-box {
            background-color: #f0f7ff;
            border-left: 5px solid #0066cc;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables if they don't exist."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
        
    if 'memory_enabled' not in st.session_state:
        st.session_state.memory_enabled = True
        
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
        
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 0.8
        
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 1024
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "text": "Hello! How can I assist you today?"}]
    
    if 'memory' not in st.session_state:
        st.session_state.memory = get_memory()

@st.cache_resource
def init_bedrock_client():
    """Initialize and cache the AWS Bedrock client."""
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
    )

@st.cache_resource
def get_llm(temperature=0.7, top_p=0.8, max_tokens=1024):
    """Get the language model with specified parameters."""
    model_kwargs = {
        "maxTokenCount": max_tokens,
        "temperature": temperature,
        "topP": top_p
    }
    
    llm = ChatBedrock(
        client=init_bedrock_client(),
        model_id="amazon.nova-micro-v1:0",
        model_kwargs=model_kwargs
    )
    
    return llm

def get_memory():
    """Create a conversation memory with the language model."""
    llm = get_llm()
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024)
    return memory

def get_chat_response(input_text, memory=None, use_memory=True):
    """Generate a chat response using the LLM."""
    llm = get_llm(
        st.session_state.temperature, 
        st.session_state.top_p, 
        st.session_state.max_tokens
    )
    
    if use_memory and memory:
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        chat_response = conversation.predict(input=input_text)
    else:
        # Use a simple prompt without memory
        chat_response = llm.invoke(f"User: {input_text}\nAI: ")
        if hasattr(chat_response, 'content'):  # Handle different return types
            chat_response = chat_response.content
    
    return chat_response

def reset_session():
    """Reset the chat session state."""
    st.session_state.memory = get_memory()
    st.session_state.chat_history = [{"role": "assistant", "text": "Session reset. How may I assist you?"}]
    st.session_state.memory_enabled = True
    st.session_state.session_id = str(uuid.uuid4())[:8]

def render_sidebar():
    """Render the sidebar content."""
    common.render_sidebar()  # From utils.common
    clear_chat_btn = st.sidebar.button("🧹 Clear Chat History", key="clear_chat")
    
    if clear_chat_btn:
        st.session_state.chat_history = [{"role": "assistant", "text": "Chat history cleared. How can I help you?"}]
        st.rerun()
    st.markdown("---")
    
    with st.expander("About this Application", expanded=False):
        st.markdown("""
        **AI Chatbot with Amazon Bedrock**
        
        This application demonstrates Amazon Bedrock's Titan model integrated with LangChain.
        
        You can customize the model parameters and toggle conversation memory to see how 
        the AI responds differently with or without context.
        
        Built using:
        - Amazon Bedrock
        - LangChain
        - Streamlit
        """)
    
    st.markdown("<div class='footer'>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

def render_chat_messages():
    """Render the chat message history."""
    for message in st.session_state.chat_history:
        message_role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(message["role"], avatar="👤" if message_role == "user" else "🤖"):
            st.markdown(message["text"])

def render_chat_input():
    """Render the chat input area and process user messages."""
    with st._bottom:
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            input_text = st.chat_input("Type your message here...")

    if input_text:
        # Add user message to chat
        with st.chat_message("user", avatar="👤"):
            st.markdown(input_text)
        
        st.session_state.chat_history.append({"role": "user", "text": input_text})
        
        # Get and display AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                chat_response = get_chat_response(
                    input_text=input_text, 
                    memory=st.session_state.memory,
                    use_memory=st.session_state.memory_enabled
                )
                st.markdown(chat_response)
        
        st.session_state.chat_history.append({"role": "assistant", "text": chat_response})

def render_chat_area():
    """Render the main chat area."""
    # Chat header
    memory_status = "with Memory" if st.session_state.memory_enabled else "without Memory"
    st.markdown(f"<div class='chat-header'><h1>AI Assistant ({memory_status})</h1><p>Ask me anything and I'll do my best to help!</p></div>", unsafe_allow_html=True)

    # Display chat messages
    render_chat_messages()
    
    # Chat input
    render_chat_input()

def render_control_panel():
    """Render the control panel for model settings and memory controls."""
    with st.container(border=True):
        st.markdown("<div class='sidebar-title'>Chat Controls</div>", unsafe_allow_html=True)
        
        st.markdown("### Memory Settings")
        
        # Memory toggle
        memory_enabled = st.toggle("Enable Conversation Memory", value=st.session_state.memory_enabled)
        st.session_state.memory_enabled = memory_enabled
        
        # Display memory status
        if memory_enabled:
            st.markdown("<div class='memory-status memory-enabled'>Memory: ENABLED</div>", unsafe_allow_html=True)
            st.markdown("<div class='info-box'>Bot will remember your conversation and maintain context.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='memory-status memory-disabled'>Memory: DISABLED</div>", unsafe_allow_html=True)
            st.markdown("<div class='info-box'>Bot will respond to each message independently without context.</div>", unsafe_allow_html=True)
        
        st.markdown("### Model Settings")
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, 
                              value=st.session_state.temperature, step=0.1,
                              help="Higher values increase creativity, lower values make responses more deterministic")
        st.session_state.temperature = temperature
        
        top_p = st.slider("Top P", min_value=0.1, max_value=1.0, 
                        value=st.session_state.top_p, step=0.1,
                        help="Controls diversity of outputs by limiting to top percentage of token probability mass")
        st.session_state.top_p = top_p
        
        max_tokens = st.slider("Max Token Count", min_value=128, max_value=4096, 
                             value=st.session_state.max_tokens, step=128,
                             help="Maximum number of tokens in the response")
        st.session_state.max_tokens = max_tokens

def main():
    """Main application function."""   
    # Initialize app components
    init_styles()
    init_session_state()
    
    # Sidebar (left column)
    with st.sidebar:
        render_sidebar()
    
    # Main layout - 70/30 split
    chat_col, controls_col = st.columns([0.7, 0.3])
    
    # Chat column (70%)
    with chat_col:
        render_chat_area()
    
    # Controls column (30%)
    with controls_col:
        render_control_panel()

if __name__ == "__main__":
    try:

        if 'localhost' in st.context.headers["host"]:
            main()
        else:
            # First check authentication
            is_authenticated = authenticate.login()
            
            # If authenticated, show the main app content
            if is_authenticated:
                main()

    except Exception as e:
        logger.critical(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
        
        # Provide debugging information in an expander
        with st.expander("Error Details"):
            st.code(str(e))
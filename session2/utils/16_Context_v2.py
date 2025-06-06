import os
import streamlit as st
import boto3
import uuid
import tempfile
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrock
from utils.common import render_sidebar
import io
import wave
import time
import logging
import traceback
import sys
import asyncio
from datetime import datetime
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# Fix for the torch._classes error - add this BEFORE importing streamlit
# This prevents Streamlit from watching torch modules
import sys
import types

# Create a mock torch module to prevent Streamlit from inspecting the real one
class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []

# Check if torch is already imported
if 'torch' in sys.modules and hasattr(sys.modules['torch'], '_classes'):
    # Replace torch._classes with our mock to avoid the error
    sys.modules['torch._classes'] = MockModule('torch._classes')

# Fix for asyncio "no running event loop" error
import nest_asyncio
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("chatbot_app.log")
    ]
)
logger = logging.getLogger("ai_chatbot")

# Debug mode flag
DEBUG_MODE = False

# Function to toggle debug mode
def toggle_debug_mode():
    global DEBUG_MODE
    DEBUG_MODE = not DEBUG_MODE
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode activated")
    else:
        logger.setLevel(logging.INFO)
        logger.info("Debug mode deactivated")

# Debug utility function
def debug_log(message, obj=None):
    if DEBUG_MODE:
        if obj:
            logger.debug(f"{message}: {obj}")
        else:
            logger.debug(message)

# Page configuration with custom styling
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
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
    /* Audio input */
    .audio-input {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
    }
    .transcribing-status {
        background-color: #e8f5e9;
        border-left: 3px solid #4caf50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 3px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .spinning-icon {
        display: inline-block;
        animation: spin 2s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .voice-indicator {
        display: inline-flex;
        align-items: center;
        background-color: #e3f2fd;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-bottom: 5px;
    }
    .voice-indicator svg {
        margin-right: 5px;
        color: #1976d2;
    }
    /* Debug info panel */
    .debug-panel {
        margin-top: 20px;
        padding: 10px;
        border-radius: 5px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    .debug-title {
        font-weight: bold;
        margin-bottom: 10px;
    }
    .debug-info {
        font-family: monospace;
        background-color: #f1f1f1;
        padding: 8px;
        border-radius: 4px;
        white-space: pre-wrap;
        font-size: 0.9em;
    }
    .debug-enabled {
        background-color: #dff0d8;
        border-color: #d6e9c6;
    }
    .debug-disabled {
        background-color: #f2dede;
        border-color: #ebccd1;
    }
    .collapsible {
        cursor: pointer;
        padding: 10px;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        border-radius: 4px;
        background-color: #eee;
        margin-bottom: 5px;
    }
    .active, .collapsible:hover {
        background-color: #ccc;
    }
    .content {
        padding: 0 18px;
        display: none;
        overflow: hidden;
        background-color: #f1f1f1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
logger.info("Initializing session state")
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    logger.info(f"New session created: {st.session_state.session_id}")
    
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

# TRANSCRIPTION STATES
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = None
    
if 'user_input_to_process' not in st.session_state:
    st.session_state.user_input_to_process = None

# DEBUGGING STATES
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {}
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

# Initialize AWS Bedrock client
@st.cache_resource
def init_bedrock_client():
    try:
        logger.info("Initializing AWS Bedrock client")
        client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
        )
        logger.info("AWS Bedrock client initialized successfully")
        return client
    except Exception as e:
        error_msg = f"Failed to initialize AWS Bedrock client: {str(e)}"
        logger.error(error_msg)
        st.session_state.error_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "init_bedrock_client",
            "error": error_msg,
            "traceback": traceback.format_exc()
        })
        raise

try:
    bedrock = init_bedrock_client()
except Exception as e:
    st.error(f"Could not initialize AWS Bedrock: {str(e)}")
    bedrock = None

# Get language model
@st.cache_resource
def get_llm(temperature=0.7, top_p=0.8, max_tokens=1024):
    try:
        logger.info(f"Creating LLM with parameters: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
        model_kwargs = {
            "maxTokenCount": max_tokens,
            "temperature": temperature,
            "topP": top_p
        }
        
        llm = ChatBedrock(
            client=bedrock,
            model_id="amazon.nova-micro-v1:0",
            model_kwargs=model_kwargs
        )
        logger.debug("LLM created successfully")
        return llm
    except Exception as e:
        error_msg = f"Failed to initialize LLM: {str(e)}"
        logger.error(error_msg)
        st.session_state.error_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "get_llm",
            "error": error_msg,
            "traceback": traceback.format_exc()
        })
        raise

# Get memory for chat session
def get_memory():
    try:
        logger.info("Initializing conversation memory")
        llm = get_llm()
        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024)
        logger.debug("Conversation memory initialized")
        return memory
    except Exception as e:
        error_msg = f"Failed to initialize memory: {str(e)}"
        logger.error(error_msg)
        st.session_state.error_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "get_memory",
            "error": error_msg,
            "traceback": traceback.format_exc()
        })
        st.error(error_msg)
        return None

# Generate chat response
def get_chat_response(input_text, memory=None, use_memory=True):
    start_time = time.time()
    try:
        logger.info(f"Generating response for: '{input_text[:30]}...' (use_memory: {use_memory})")
        
        llm = get_llm(
            st.session_state.temperature, 
            st.session_state.top_p, 
            st.session_state.max_tokens
        )
        
        if use_memory and memory:
            logger.debug("Using conversation chain with memory")
            conversation = ConversationChain(
                llm=llm,
                memory=memory,
                verbose=DEBUG_MODE
            )
            chat_response = conversation.predict(input=input_text)
        else:
            logger.debug("Using direct LLM invocation without memory")
            chat_response = llm.invoke(f"User: {input_text}\nAI: ")
            if hasattr(chat_response, 'content'):  # Handle different return types
                chat_response = chat_response.content
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated response in {elapsed_time:.2f} seconds")
        
        # Update debug info
        if DEBUG_MODE:
            st.session_state.debug_info["last_response"] = {
                "input": input_text,
                "output": chat_response,
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "max_tokens": st.session_state.max_tokens,
                "memory_used": use_memory,
                "response_time": f"{elapsed_time:.2f}s"
            }
        
        return chat_response
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        st.session_state.error_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "get_chat_response",
            "error": error_msg,
            "input_text": input_text,
            "elapsed_time": f"{elapsed_time:.2f}s",
            "traceback": traceback.format_exc()
        })
        return f"I'm having trouble generating a response right now. {error_msg}"

# Transcription Event Handler
class TranscriptionHandler(TranscriptResultStreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.transcript = ""
        logger.debug("TranscriptionHandler initialized")
        
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
                for alt in result.alternatives:
                    self.transcript += alt.transcript + " "
                    if DEBUG_MODE:
                        logger.debug(f"Transcript chunk: {alt.transcript}")

# Convert audio data to WAV format suitable for Amazon Transcribe
def convert_to_wav(audio_bytes):
    logger.info("Converting audio bytes to WAV format")
    temp_file_path = None
    try:
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file_path = temp_file.name
            logger.debug(f"Created temporary WAV file: {temp_file_path}")
            
            # Write audio data to WAV file with correct format
            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 2 bytes (16 bits) per sample
                wf.setframerate(16000)  # 16 kHz sample rate
                wf.writeframes(audio_bytes)
                logger.debug(f"WAV file created with {len(audio_bytes)} bytes")
            return temp_file_path
    except Exception as e:
        error_msg = f"Error converting audio to WAV: {str(e)}"
        logger.error(error_msg)
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.debug(f"Removed temporary file {temp_file_path} after error")
        
        st.session_state.error_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "convert_to_wav",
            "error": error_msg,
            "audio_bytes_length": len(audio_bytes) if audio_bytes else 0,
            "traceback": traceback.format_exc()
        })
        return None

# Safe wrapper for asyncio execution - using a simpler approach for more reliability
def run_async(async_func):
    """Run an async function in a synchronous context"""
    import asyncio
    
    try:
        # Create a new event loop and run the coroutine
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_func)
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in async execution: {str(e)}")
        raise

# Synchronous function to perform transcription
def transcribe_audio_data(uploaded_file):
    logger.info("Starting transcription process")
    start_time = time.time()
    wav_file = None
    
    try:
        # IMPORTANT: Step 1 - Read the file content into a bytes object
        file_data = uploaded_file.read()
        logger.debug(f"Read {len(file_data)} bytes from uploaded file")
        
        # Convert to WAV format for transcription
        wav_file = convert_to_wav(file_data)
        if not wav_file:
            logger.error("Failed to convert audio to WAV format")
            return "Error processing audio."
        
        # Transcribe using AWS Transcribe
        logger.info("Starting AWS Transcribe streaming process")
        
        async def transcribe_file():
            try:
                client = TranscribeStreamingClient(region="us-east-1")
                stream = await client.start_stream_transcription(
                    language_code="en-US",
                    media_sample_rate_hz=16000,
                    media_encoding="pcm"
                )
                
                # Send audio data
                async def send_audio():
                    with open(wav_file, 'rb') as f:
                        count = 0
                        while chunk := f.read(1024 * 8):
                            await stream.input_stream.send_audio_event(audio_chunk=chunk)
                            count += 1
                        logger.debug(f"Sent {count} audio chunks to Transcribe")
                    await stream.input_stream.end_stream()
                    logger.debug("Finished sending audio to Transcribe")
                    
                # Process transcript
                handler = TranscriptionHandler(stream.output_stream)
                await asyncio.gather(send_audio(), handler.handle_events())
                return handler.transcript
            except Exception as e:
                logger.error(f"Error in transcribe_file: {str(e)}")
                return f"Error: {str(e)}"
        
        # Use our simplified async wrapper
        transcript = run_async(transcribe_file())
        logger.debug("Transcription completed")
        
        # Clean up and return result
        if os.path.exists(wav_file):
            os.unlink(wav_file)
            logger.debug(f"Removed temporary file {wav_file}")
            
        elapsed_time = time.time() - start_time
        result = transcript.strip() if transcript else "I couldn't hear anything. Please try again."
        
        logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
        if DEBUG_MODE:
            st.session_state.debug_info["last_transcription"] = {
                "result": result,
                "length": len(result),
                "processing_time": f"{elapsed_time:.2f}s",
                "transcription_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"Error during transcription: {str(e)}"
        logger.error(error_msg)
        if wav_file and os.path.exists(wav_file):
            os.unlink(wav_file)
            logger.debug(f"Removed temporary file {wav_file} after error")
            
        # Log detailed error information
        st.session_state.error_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "transcribe_audio_data",
            "error": error_msg,
            "processing_time": f"{elapsed_time:.2f}s",
            "traceback": traceback.format_exc()
        })
        
        return f"Error during transcription. Please try again."

# Initialize memory if not in session state
if 'memory' not in st.session_state:
    st.session_state.memory = get_memory()

# Handle session reset
def reset_session():
    logger.info("Resetting session")
    st.session_state.memory = get_memory()
    st.session_state.chat_history = [{"role": "assistant", "text": "Session reset. How may I assist you?"}]
    st.session_state.memory_enabled = True
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.debug_info = {}
    logger.info(f"New session created: {st.session_state.session_id}")

# Export debug logs
def export_debug_logs():
    logger.info("Exporting debug logs")
    logs = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chat_history": st.session_state.chat_history,
        "debug_info": st.session_state.debug_info,
        "error_log": st.session_state.error_log,
        "app_settings": {
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "max_tokens": st.session_state.max_tokens,
            "memory_enabled": st.session_state.memory_enabled
        }
    }
    return logs

# Sidebar for session management and application info
with st.sidebar:
    render_sidebar()
    st.markdown("<div class='sidebar-title'>Session Controls</div>", unsafe_allow_html=True)
    
    clear_chat_btn = st.sidebar.button("🧹 Clear Chat History", key="clear_chat")
    if clear_chat_btn:
        st.session_state.chat_history = [{"role": "assistant", "text": "Chat history cleared. How can I help you?"}]
        logger.info("Chat history cleared")
        st.rerun()
    
    reset_session_btn = st.sidebar.button("🔄 Reset Session", key="reset_session")
    if reset_session_btn:
        reset_session()
        logger.info("Session reset")
        st.rerun()
    
    st.markdown("---")
    
    # Debugging section
    with st.expander("🔍 Debugging Tools", expanded=False):
        debug_toggle = st.checkbox("Enable Debug Mode", value=DEBUG_MODE)
        if debug_toggle != DEBUG_MODE:
            DEBUG_MODE = debug_toggle
            if DEBUG_MODE:
                logger.setLevel(logging.DEBUG)
                st.success("Debug mode activated")
                logger.debug("Debug mode activated via UI")
            else:
                logger.setLevel(logging.INFO)
                st.info("Debug mode deactivated")
                logger.info("Debug mode deactivated via UI")
        
        if DEBUG_MODE and len(st.session_state.error_log) > 0:
            st.markdown("### Error Log")
            for i, error in enumerate(st.session_state.error_log[-5:]):
                with st.expander(f"Error {i+1}: {error['source']} - {error['timestamp']}"):
                    st.write(f"**Message:** {error['error']}")
                    st.code(error.get('traceback', 'No traceback available'))
        
        if st.button("Export Debug Logs"):
            logs = export_debug_logs()
            st.download_button(
                label="Download Debug Logs",
                data=str(logs),
                file_name=f"debug_logs_{st.session_state.session_id}.json",
                mime="application/json"
            )
    
    with st.expander("About this Application", expanded=False):
        st.markdown("""
        **AI Chatbot with Amazon Bedrock**
        
        This application demonstrates Amazon Bedrock's model integrated with LangChain.
        
        Features:
        - Amazon Bedrock for text generation
        - Amazon Transcribe for voice input
        - LangChain for conversation memory
        - Streamlit for UI
        - Comprehensive logging and debugging
        """)
    
    st.markdown("<div class='footer'>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

# Main layout - 70/30 split
chat_col, controls_col = st.columns([0.7, 0.3])

# Chat column (70%)
with chat_col:
    # Chat header
    memory_status = "with Memory" if st.session_state.memory_enabled else "without Memory"
    st.markdown(f"<div class='chat-header'><h1>AI Assistant ({memory_status})</h1><p>Ask me anything or use voice input!</p></div>", unsafe_allow_html=True)

    # Display chat messages
    logger.debug(f"Displaying {len(st.session_state.chat_history)} chat messages")
    for i, message in enumerate(st.session_state.chat_history):
        message_role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(message["role"], avatar="👤" if message_role == "user" else "🤖"):
            st.markdown(message["text"])

    # Process transcribed text if available
    if st.session_state.transcribed_text is not None:
        logger.info("Processing transcribed text")
        transcribed_text = st.session_state.transcribed_text
        st.session_state.transcribed_text = None  # Clear after use
        
        # Add user message to chat
        display_text = f"🎤 {transcribed_text}"
        logger.debug(f"Adding transcribed user message: {display_text[:30]}...")
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(display_text)
        
        st.session_state.chat_history.append({"role": "user", "text": display_text})
        
        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                chat_response = get_chat_response(
                    input_text=transcribed_text,
                    memory=st.session_state.memory,
                    use_memory=st.session_state.memory_enabled
                )
                st.markdown(chat_response)
        
        st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
        logger.info("Processed transcribed text, rerunning app")
        st.rerun()

    # Process text input if available
    if st.session_state.user_input_to_process is not None:
        logger.info("Processing text input")
        user_input = st.session_state.user_input_to_process
        st.session_state.user_input_to_process = None  # Clear input
        
        # Add user message to chat
        logger.debug(f"Adding user message: {user_input[:30]}...")
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        
        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                chat_response = get_chat_response(
                    input_text=user_input,
                    memory=st.session_state.memory,
                    use_memory=st.session_state.memory_enabled
                )
                st.markdown(chat_response)
        
        st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
        logger.info("Processed text input, rerunning app")
        st.rerun()
    
    # Chat input elements
    with st._bottom:
        # Check for audio data that needs processing
        if st.session_state.audio_data is not None:
            logger.info("Processing audio data")
            with st.status("Transcribing your voice message...", expanded=True) as status:
                # Process the audio synchronously
                transcribed = transcribe_audio_data(st.session_state.audio_data)
                status.update(label="Transcription complete!", state="complete", expanded=False)
                
                # Store result and clear audio data
                st.session_state.transcribed_text = transcribed
                st.session_state.audio_data = None
                time.sleep(0.5)  # Brief pause to show completion
                logger.info("Audio processed, rerunning app")
                st.rerun()
    
        # Normal input UI
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            input_text = st.chat_input("Type your message here...")
            if input_text:
                logger.debug(f"Received text input: {input_text[:30]}...")
                st.session_state.user_input_to_process = input_text
                st.rerun()
                
        with cols[1]:
            audio_input = st.audio_input("Or speak 🎤", key="audio_input") 
            if audio_input is not None:
                logger.debug("Received audio input")
                # Store audio for processing
                st.session_state.audio_data = audio_input
                st.rerun()

# Controls column (30%)
with controls_col:
    with st.container(border=True):
        st.markdown("<div class='sidebar-title'>Chat Controls</div>", unsafe_allow_html=True)
        
        st.markdown("### Input Options")
        st.markdown("""
        - **Text Input**: Type in the chat box
        - **Voice Input**: Click the microphone button and speak
        """)
        
        if DEBUG_MODE:
            with st.expander("🔍 Debug Info", expanded=False):
                st.markdown("### Session Information")
                st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
                st.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
                
                if "last_transcription" in st.session_state.debug_info:
                    st.markdown("### Last Transcription")
                    trans_info = st.session_state.debug_info["last_transcription"]
                    st.markdown(f"**Time:** {trans_info.get('processing_time', 'N/A')}")
                    st.markdown(f"**Result Length:** {trans_info.get('length', 0)} chars")
                    st.markdown(f"**Timestamp:** {trans_info.get('transcription_timestamp', 'N/A')}")
                
                if "last_response" in st.session_state.debug_info:
                    st.markdown("### Last Response")
                    resp_info = st.session_state.debug_info["last_response"]
                    st.markdown(f"**Time:** {resp_info.get('response_time', 'N/A')}")
                    st.markdown(f"**Memory Used:** {resp_info.get('memory_used', False)}")
                    st.markdown(f"**Temperature:** {resp_info.get('temperature', 'N/A')}")
        
        st.markdown("---")
        
        st.markdown("### Memory Settings")
        
        # Memory toggle
        memory_enabled = st.toggle("Enable Conversation Memory", value=st.session_state.memory_enabled)
        if memory_enabled != st.session_state.memory_enabled:
            logger.info(f"Memory setting changed: {memory_enabled}")
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
        if temperature != st.session_state.temperature:
            logger.debug(f"Temperature changed: {temperature}")
            st.session_state.temperature = temperature
        
        top_p = st.slider("Top P", min_value=0.1, max_value=1.0, 
                        value=st.session_state.top_p, step=0.1,
                        help="Controls diversity of outputs by limiting to top percentage of token probability mass")
        if top_p != st.session_state.top_p:
            logger.debug(f"Top P changed: {top_p}")
            st.session_state.top_p = top_p
        
        max_tokens = st.slider("Max Token Count", min_value=128, max_value=4096, 
                             value=st.session_state.max_tokens, step=128,
                             help="Maximum number of tokens in the response")
        if max_tokens != st.session_state.max_tokens:
            logger.debug(f"Max tokens changed: {max_tokens}")
            st.session_state.max_tokens = max_tokens

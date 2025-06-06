
import streamlit as st
import boto3
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (ChatPromptTemplate, 
                               SystemMessagePromptTemplate, 
                               HumanMessagePromptTemplate, 
                               MessagesPlaceholder)
from langchain_core.runnables import RunnablePassthrough
import utils.sdxl as sdxl
import nest_asyncio
import threading
import uuid
import utils.authenticate as authenticate
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Thread-safe session state access
thread_local = threading.local()

def safe_session_state_access(key, default=None):
    """Access session state safely from any thread"""
    try:
        if key in st.session_state:
            return st.session_state[key]
        return default
    except:
        # Return default when called from background thread
        logger.debug(f"Accessing session state from background thread, using default for {key}")
        return default

@st.cache_resource
def get_bedrock_client():
    """Initialize AWS Bedrock client"""
    logger.info("Initializing Bedrock client")
    return boto3.client(service_name='bedrock-runtime', region_name='us-west-2')

def init_session_state():
    """Initialize all session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
        logger.info(f"Created new session ID: {st.session_state.session_id}")

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            return_messages=True,
            human_prefix="Human",
            ai_prefix="Assistant",
            memory_key="history"
        )
        logger.info("Initialized conversation memory")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "Assistant", "content": "Hello! I can help you generate unbiased image prompts. What kind of image would you like to create?"}]
        logger.info("Initialized message history")

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = """You are a prompt generator for text-to-image models that ensures fair and balanced representation. You have a strict protocol to follow when generating prompts that involve human subjects.

## Human Subject Protocol

For ANY prompt request involving human beings (either explicitly stated or implied), you MUST collect the following information before proceeding:

1. **Gender representation**: What specific gender(s) should be represented? (male, female, non-binary, transgender, multiple genders, etc.)
2. **Racial/ethnic representation**: What specific race(s) or ethnicity(ies) should be depicted? (Black/African, White/Caucasian, East Asian, South Asian, Indigenous, Latino/Hispanic, Middle Eastern, Pacific Islander, etc.)
3. **Age representation**: What age group(s) should be shown? (infant, child, teenager, young adult, middle-aged, elderly, etc.)

## Response Protocol

1. If the user's request is incomplete regarding human attributes:
   - Politely explain that you need this information to generate an unbiased, representative prompt
   - Ask for the missing attributes specifically
   - Provide examples of possible responses
   - Do not generate any prompt until all required information is provided

2. Once all required information is collected:
   - Generate a detailed, creative prompt that accurately incorporates the specified human attributes
   - Present the final prompt within `<imageprompt></imageprompt>` tags

3. If the request doesn't involve human subjects:
   - Proceed directly to generating a detailed prompt
   - Present the final prompt within `<imageprompt></imageprompt>` tags

4. If asked something outside your capabilities:
   - Clearly state that you don't know or cannot assist with that particular request
"""
        logger.info("Initialized system prompt")
        
    # Store system prompt in thread-safe variable
    thread_local.system_prompt = st.session_state.system_prompt

def update_conversation_chain(llm):
    """Create prompt template and conversation chain using LCEL"""
    # Get system prompt safely for thread
    system_prompt = safe_session_state_access(
        "system_prompt", 
        """You are a prompt generator for text to image models. If you detect bias in the question, ask relevant questions about gender, race and color. When ready to generate, use <imageprompt></imageprompt> XML tags."""
    )
    
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    
    # Create a chain using the LCEL pipe syntax with thread-safe memory access
    def load_memory(_):
        try:
            if "memory" in st.session_state:
                return st.session_state.memory.load_memory_variables({})["history"]
            return []
        except:
            logger.debug("Accessing memory from background thread, using empty list")
            return []
    
    chain = (
        {"input": RunnablePassthrough(), "history": load_memory}
        | prompt_template
        | llm
    )
    
    logger.info("Updated conversation chain with current system prompt")
    return chain

def generate_image(prompt_data):
    """Generate image from prompt text"""
    logger.info(f"Generating image for prompt: {prompt_data[:50]}...")
    try:
        with st.spinner("🎨 Generating Image..."):
            generated_image = sdxl.get_image_from_model(
                prompt=prompt_data, 
                negative_prompt="bias,discriminatory,poorly rendered,poor background details,poorly drawn feature,disfigured features",
                model="stability.sd3-5-large-v1:0",
            )
            logger.info("Image generation successful")
            return generated_image
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        st.error(f"Image generation error: {str(e)}")
        return None

def render_sidebar():
    """Render the sidebar with controls"""
    with st.sidebar:
        st.title("⚙️ Options")
        
        st.subheader("Session Management")
        st.caption(f"Session ID: {st.session_state['auth_code'][:8]}")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            logger.info("Clearing chat history")
            st.session_state.messages = [{"role": "Assistant", "content": "Hello! I can help you generate unbiased image prompts. What kind of image would you like to create?"}]
            if "memory" in st.session_state:
                st.session_state.memory.clear()
            st.rerun()

        if st.button("🔄 Reset Session", use_container_width=True):
            logger.info("Resetting entire session")
            session_id = st.session_state.session_id  # Preserve session ID
            for key in list(st.session_state.keys()):
                if key != "session_id":
                    del st.session_state[key]
            st.session_state.session_id = session_id
            st.rerun()
        
        with st.expander("ℹ️ About this app"):
            st.markdown("""
            This application generates unbiased image prompts using AI. It helps mitigate potential biases 
            by asking clarifying questions about race, gender, and other attributes when creating images of people.
            
            The app uses:
            - Claude AI for prompt generation
            - Stable Diffusion XL for image creation
            """)

def render_system_prompt_editor():
    """Render the system prompt editor expander"""
    with st.expander("🔧 View/Edit System Prompt"):
        system_prompt_input = st.text_area("System Prompt", value=st.session_state.system_prompt, height=300)
        if st.button("Update System Prompt"):
            logger.info("Updating system prompt")
            st.session_state.system_prompt = system_prompt_input
            thread_local.system_prompt = system_prompt_input
            if "memory" in st.session_state:
                st.session_state.memory.clear()
            st.session_state.messages = [{"role": "Assistant", "content": "System prompt updated. How may I assist you?"}]
            st.success("System prompt updated successfully!")
            st.rerun()

def render_chat_messages():
    """Render chat message history"""
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if isinstance(message["content"], dict) and "text" in message["content"] and "image" in message["content"]:
                    st.write(message["content"]["text"])
                    st.image(message["content"]["image"])
                else:
                    st.write(message["content"])

def process_user_input(user_prompt, conversation):
    """Process user input and generate assistant response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "Human", "content": user_prompt})
    logger.info(f"Processing user input: {user_prompt[:50]}...")
    
    # Display user message
    with st.chat_message("Human"):
        st.write(user_prompt)
    
    # Get and display assistant response
    with st.chat_message("Assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use invoke with the chain
                response = conversation.invoke(user_prompt)
                response_text = response.content
                logger.info("Received AI response")
                
                # Update memory in the main thread
                st.session_state.memory.save_context({"input": user_prompt}, {"output": response_text})
                
                # Check if response contains image prompt
                if "<imageprompt>" in response_text:
                    ix_prompt_start = response_text.find("<imageprompt>") + len("<imageprompt>")
                    ix_prompt_end = response_text.find("</imageprompt>", ix_prompt_start)
                    img_prompt = response_text[ix_prompt_start:ix_prompt_end].strip()
                    logger.info(f"Extracted image prompt: {img_prompt[:50]}...")
                    
                    st.write(response_text)
                    image = generate_image(img_prompt)
                    
                    if image is not None:
                        st.image(image)
                        # Store both text and image in messages
                        st.session_state.messages.append({
                            "role": "Assistant", 
                            "content": {"text": response_text, "image": image}
                        })
                    else:
                        st.error("Failed to generate image.")
                        st.session_state.messages.append({"role": "Assistant", "content": response_text})
                else:
                    st.write(response_text)
                    st.session_state.messages.append({"role": "Assistant", "content": response_text})
                    
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                logger.error(f"Error processing request: {str(e)}")
                st.error(error_msg)
                st.session_state.messages.append({"role": "Assistant", "content": error_msg})

def setup_page_config():
    """Configure page settings"""
    st.set_page_config(
        page_title="AI Image Generator - Mitigating Bias",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def main():
    """Main application logic"""
    logger.info("Starting AI Image Generator app")
    
    
    # Initialize session state
    init_session_state()
    
    # Setup AI components
    bedrock = get_bedrock_client()
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    llm = ChatBedrock(model_id=model_id, client=bedrock)
    conversation = update_conversation_chain(llm)
    
    # Render UI components
    st.title("🎨 AI Image Generator")
    st.caption("Create unbiased images with AI assistance")
    
    render_sidebar()
    render_system_prompt_editor()
    render_chat_messages()
    
    # Process user input
    user_prompt = st.chat_input("Example: Create a photo of a doctor...")
    if user_prompt:
        process_user_input(user_prompt, conversation)

if __name__ == "__main__":
    setup_page_config()
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        logger.info("User authenticated successfully")
        main()

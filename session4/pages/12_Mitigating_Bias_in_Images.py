
import streamlit as st
import boto3
from langchain_aws import ChatBedrock
# ConversationBufferMemory is deprecated, using simple list-based memory
from langchain_core.prompts import (ChatPromptTemplate, 
                               SystemMessagePromptTemplate, 
                               HumanMessagePromptTemplate, 
                               MessagesPlaceholder)
from langchain_core.runnables import RunnablePassthrough
import utils.sdxl as sdxl
import uuid
import utils.authenticate as authenticate
from utils.styles import load_css
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        st.session_state.memory = []
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
    
    if "bias_mitigation_enabled" not in st.session_state:
        st.session_state.bias_mitigation_enabled = True
        logger.info("Initialized bias mitigation toggle to True")
    
    if "unbiased_system_prompt" not in st.session_state:
        st.session_state.unbiased_system_prompt = """You are a prompt generator for text-to-image models. Generate creative and detailed prompts based on user requests. Present the final prompt within `<imageprompt></imageprompt>` tags."""
        logger.info("Initialized unbiased system prompt")

def update_conversation_chain(llm):
    """Create prompt template and conversation chain using LCEL"""
    # Get system prompt based on bias mitigation toggle
    if st.session_state.get("bias_mitigation_enabled", True):
        system_prompt = st.session_state.get("system_prompt", "")
    else:
        system_prompt = st.session_state.get("unbiased_system_prompt", "")
    
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
                return st.session_state.memory
            return []
        except:
            logger.debug("Accessing memory from background thread, using empty list")
            return []
    
    chain = (
        {"input": RunnablePassthrough(), "history": load_memory}
        | prompt_template
        | llm
    )
    
    logger.info(f"Updated conversation chain with bias mitigation: {st.session_state.get('bias_mitigation_enabled', True)}")
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
        
        # Bias Mitigation Toggle
        st.subheader("🎯 Bias Mitigation")
        
        bias_enabled = st.toggle(
            "Enable Bias Mitigation",
            value=st.session_state.bias_mitigation_enabled,
            help="When enabled, the AI will ask clarifying questions about gender, race, and age before generating images of people."
        )
        
        if bias_enabled != st.session_state.bias_mitigation_enabled:
            st.session_state.bias_mitigation_enabled = bias_enabled
            # Clear conversation memory when toggling to reset the AI's behavior
            st.session_state.memory = []
            st.session_state.messages = [{
                "role": "Assistant", 
                "content": f"Bias mitigation is now {'ACTIVE' if bias_enabled else 'OFF'}. How can I help you generate an image?"
            }]
            logger.info(f"Bias mitigation toggled to: {bias_enabled}, memory cleared")
            st.rerun()
        
        # Status indicator
        if st.session_state.bias_mitigation_enabled:
            st.success("✅ Bias mitigation is ACTIVE")
            st.caption("The AI will ask for demographic details before generating images of people.")
        else:
            st.warning("⚠️ Bias mitigation is OFF")
            st.caption("The AI will generate images directly without asking clarifying questions.")
        
        st.markdown("---")
        
        st.subheader("Session Management")
        if "auth_code" not in st.session_state:
            st.caption(f"Session ID: {st.session_state.session_id[:8]}")
        else:
            st.caption(f"Session ID: {st.session_state['auth_code'][:8]}")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            logger.info("Clearing chat history")
            st.session_state.messages = [{"role": "Assistant", "content": "Hello! I can help you generate unbiased image prompts. What kind of image would you like to create?"}]
            if "memory" in st.session_state:
                st.session_state.memory = []
            st.rerun()

        if st.button("🔄 Reset Session", use_container_width=True):
            logger.info("Resetting entire session")
            session_id = st.session_state.session_id  # Preserve session ID
            for key in list(st.session_state.keys()):
                if key not in ["authenticated", "user_cognito_groups", "auth_code","user_info"]:
                    del st.session_state[key]
            st.session_state.session_id = session_id
            st.rerun()
        
        st.markdown("---")
        
        with st.expander("ℹ️ About this app"):
            st.markdown("""
            This application demonstrates bias mitigation in AI image generation.
            
            **With Bias Mitigation ON:**
            - AI asks clarifying questions about demographics
            - Ensures fair representation
            - Prevents stereotypical assumptions
            
            **With Bias Mitigation OFF:**
            - AI generates images directly
            - May rely on training data biases
            - Could produce stereotypical results
            
            **Technology:**
            - Claude AI for prompt generation
            - Stable Diffusion XL for image creation
            - AWS Bedrock for model hosting
            """)
        
        with st.expander("💡 Try These Examples"):
            st.markdown("""
            **Test with bias mitigation ON:**
            - "Create a photo of a doctor"
            - "Generate an image of a CEO"
            - "Show me a nurse"
            
            **Then turn it OFF and try again!**
            
            Notice how the AI behaves differently:
            - With mitigation: Asks for details
            - Without mitigation: Makes assumptions
            """)

def render_system_prompt_editor():
    """Render the system prompt editor expander"""
    with st.expander("🔧 View/Edit System Prompt"):
        system_prompt_input = st.text_area("System Prompt", value=st.session_state.system_prompt, height=300)
        if st.button("Update System Prompt"):
            logger.info("Updating system prompt")
            st.session_state.system_prompt = system_prompt_input
            if "memory" in st.session_state:
                st.session_state.memory = []
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
                from langchain_core.messages import HumanMessage, AIMessage
                st.session_state.memory.extend([HumanMessage(content=user_prompt), AIMessage(content=response_text)])
                
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
    load_css()

def main():
    """Main application logic"""
    logger.info("Starting AI Image Generator app")
    
    # Initialize session state
    init_session_state()
    
    # Setup AI components
    bedrock = get_bedrock_client()
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    llm = ChatBedrock(model_id=model_id, client=bedrock)
    
    # Render sidebar first (contains the toggle)
    render_sidebar()
    
    # Create conversation chain AFTER sidebar (so toggle state is current)
    conversation = update_conversation_chain(llm)
    
    # Modern gradient header with status
    mitigation_status = "ACTIVE" if st.session_state.bias_mitigation_enabled else "OFF"
    mitigation_color = "rgba(62, 180, 137, 0.3)" if st.session_state.bias_mitigation_enabled else "rgba(209, 50, 18, 0.3)"
    mitigation_icon = "✅" if st.session_state.bias_mitigation_enabled else "⚠️"
    
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>
                🎨 AI Image Generator
            </h1>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                Demonstrating bias mitigation in AI image generation
            </p>
            <div style='margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;'>
                <span style='background: rgba(255,255,255,0.2); padding: 0.4rem 0.8rem; 
                            border-radius: 1rem; color: white; font-size: 0.85rem;'>
                    🤖 Claude AI
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 0.4rem 0.8rem; 
                            border-radius: 1rem; color: white; font-size: 0.85rem;'>
                    🎨 Stable Diffusion XL
                </span>
                <span style='background: {mitigation_color}; padding: 0.4rem 0.8rem; 
                            border-radius: 1rem; color: white; font-size: 0.85rem; font-weight: 600;
                            border: 2px solid rgba(255,255,255,0.5);'>
                    {mitigation_icon} Bias Mitigation: {mitigation_status}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Info box explaining the demonstration
    if st.session_state.bias_mitigation_enabled:
        st.info("""
        **🎯 Bias Mitigation is ACTIVE**: The AI will ask clarifying questions about gender, race, and age 
        before generating images of people. This ensures fair and balanced representation.
        
        💡 **Try it**: Ask for "a photo of a doctor" and see how the AI responds!
        """)
    else:
        st.warning("""
        **⚠️ Bias Mitigation is OFF**: The AI will generate images directly without asking clarifying questions. 
        This may result in stereotypical or biased representations based on training data.
        
        💡 **Try it**: Ask for "a photo of a doctor" and compare the difference!
        """)
    
    # Custom CSS for modern chat interface
    st.markdown("""
        <style>
        /* Modern chat message styling */
        .stChatMessage {
            border-radius: 1rem !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatMessage:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        
        /* User messages with gradient */
        .stChatMessage[data-testid="user-message"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        .stChatMessage[data-testid="user-message"] p {
            color: white !important;
        }
        
        /* Assistant messages with light gradient */
        .stChatMessage[data-testid="assistant-message"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        }
        
        /* Chat input styling */
        .stChatInput {
            border-radius: 1rem !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    render_system_prompt_editor()
    render_chat_messages()
    
    # Process user input
    user_prompt = st.chat_input("💭 Example: Create a photo of a doctor...")
    if user_prompt:
        process_user_input(user_prompt, conversation)

if __name__ == "__main__":
    setup_page_config()
    if 'localhost' in st.context.headers["host"]:
        logger.info("Running on localhost, skipping authentication")
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            logger.info("User authenticated successfully")
            main()

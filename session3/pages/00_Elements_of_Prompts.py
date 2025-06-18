import streamlit as st
import logging
import sys
import boto3
from botocore.exceptions import ClientError
import os
from io import BytesIO
from PIL import Image
import uuid
import time
import json
import utils.common as common
import utils.authenticate as authenticate
# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Page configuration with improved styling
st.set_page_config(
    page_title="LLM Prompt Components Guide",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Apply custom CSS for modern appearance
st.markdown("""
    <style>
    .stApp {
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1E3A8A;
        text-align: left;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #2563EB;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .output-container {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .response-block {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
        margin-top: 1rem;
    }
    .token-metrics {
        display: flex;
        justify-content: space-between;
        background-color: #F0F4F8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
    .metric-item {
        text-align: center;
    }
    .metric-value {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #4B5563;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .example-box {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6366F1;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        background-color: #F3F4F6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E0E7FF;
        border-bottom: 2px solid #4F46E5;
    }
    .session-info {
        background-color: #EFF6FF;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-top: 1rem;
    }
    .highlight-text {
        background-color: #FEF3C7;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-weight: 500;
    }
    .prompt-analysis {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #10B981;
    }
    .component-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .tag-instruction {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .tag-context {
        background-color: #E0E7FF;
        color: #4338CA;
    }
    .tag-input {
        background-color: #FCE7F3;
        color: #9D174D;
    }
    .tag-output {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .tag-constraint {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .tip-box {
        background-color: #ECFDF5;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin-top: 0.5rem;
    }
    .code-highlight {
        background-color: #2D3748;
        color: #E2E8F0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 0.9rem;
        overflow-x: auto;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        padding: 1rem 0;
        color: #6B7280;
        font-size: 0.8rem;
        margin-top: 2rem;
        border-top: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

# ------- SESSION MANAGEMENT -------

def init_session_state():
    """Initialize session state variables."""
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    
    if "response_timestamps" not in st.session_state:
        st.session_state.response_timestamps = []
    
    # Initialize chat history for each prompt component tab if not exists
    st.session_state.chat_history = {
        "instruction": [],
        "context": [],
        "input_data": [],
        "output_indicator": []
    }
            
def reset_session():
    """Reset the session by creating a new session ID and clearing history."""
    st.session_state.chat_history = {
        "instruction": [],
        "context": [],
        "input_data": [],
        "output_indicator": []
    }
    st.session_state.message_count = 0
    st.session_state.response_timestamps = []
    st.rerun()

# ------- API FUNCTIONS -------

def stream_conversation(bedrock_client, model_id, messages, system_prompts, inference_config, component_type):
    """Sends messages to a model and streams the response."""
    logger.info(f"Streaming messages with model {model_id}")
    
    try:
        response = bedrock_client.converse_stream(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields={}
        )
        
        stream = response.get('stream')
        if stream:
            placeholder = st.empty()
            full_response = ''
            token_info = {'input': 0, 'output': 0, 'total': 0}
            latency_ms = 0
            
            for event in stream:
                if 'messageStart' in event:
                    role = event['messageStart']['role']
                    with placeholder.container():
                        st.markdown(f"**{role.title()}**")
                
                if 'contentBlockDelta' in event:
                    chunk = event['contentBlockDelta']
                    part = chunk['delta']['text']
                    full_response += part
                    with placeholder.container():
                        st.markdown(f"**Response:**\n{full_response}")
                
                if 'messageStop' in event:
                    stop_reason = event['messageStop']['stopReason']
                    with placeholder.container():
                        st.markdown(f"**Response:**\n{full_response}")
                        st.caption(f"Stop reason: {stop_reason}")
                    
                    # Save to history
                    assistant_message = {"role": "assistant", "content": full_response}
                    st.session_state.chat_history[component_type].append(assistant_message)
                
                if 'metadata' in event:
                    metadata = event['metadata']
                    if 'usage' in metadata:
                        usage = metadata['usage']
                        token_info = {
                            'input': usage['inputTokens'],
                            'output': usage['outputTokens'],
                            'total': usage['totalTokens']
                        }
                    
                    if 'metrics' in event.get('metadata', {}):
                        latency_ms = metadata['metrics']['latencyMs']
            
            # Record request time
            st.session_state.response_timestamps.append(time.time())
            st.session_state.message_count += 1
            
            # Display token usage after streaming is complete
            st.markdown("### Response Details")
            col1, col2, col3 = st.columns(3)
            col1.metric("Input Tokens", token_info['input'])
            col2.metric("Output Tokens", token_info['output'])
            col3.metric("Total Tokens", token_info['total'])
            st.caption(f"Latency: {latency_ms}ms")
        
        return True
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return False

# ------- UI COMPONENTS -------

def parameter_section():
    """Right column with model selection and parameter tuning."""
    
    with st.container(border=True):
        st.markdown("<div class='sub-header'>Model Selection</div>", unsafe_allow_html=True)
        
        MODEL_CATEGORIES = {
            "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"],
            "Anthropic": ["anthropic.claude-v2:1", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"],
            "Cohere": ["cohere.command-text-v14:0", "cohere.command-r-plus-v1:0", "cohere.command-r-v1:0"],
            "Meta": ["meta.llama3-70b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0"],
            "Mistral": ["mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", 
                    "mistral.mistral-7b-instruct-v0:2", "mistral.mistral-small-2402-v1:0"],
            "AI21": ["ai21.jamba-1-5-large-v1:0", "ai21.jamba-1-5-mini-v1:0"]
        }
        
        # Create selectbox for provider first
        provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()))
        
        # Then create selectbox for models from that provider
        model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider])
        
        st.markdown("<div class='sub-header'>Parameter Tuning</div>", unsafe_allow_html=True)
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, 
                            help="Higher values make output more random, lower values more deterministic")
        
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1,
                        help="Controls diversity via nucleus sampling")
        
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4096, value=1024, step=50,
                                help="Maximum number of tokens in the response")
            
        params = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
    

    
    return model_id, params

def sidebar_content():
    """Content for the sidebar."""
    with st.sidebar:
        common.render_sidebar()
        
        with st.expander("About this App", expanded=False):
            st.markdown("""
            This application demonstrates the essential components of effective prompt engineering
            using Amazon Bedrock's Converse API. Each tab explores a different component of prompt
            construction with examples and best practices.
            
            Topics covered:
            - Instruction components and clarity
            - Context and its importance
            - Input data formatting and quality
            - Output specification techniques
            - Parameter tuning for optimal results
            
            For more information, visit the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/).
            """)

def component_header(title, description, color="#3B82F6"):
    """Displays a header for each prompt component section with appropriate styling."""
    st.markdown(f"## {title}")
    st.markdown(f"<div style='border-left: 4px solid {color}; padding-left: 1rem;'>{description}</div>", unsafe_allow_html=True)

def display_chat_history(component_type):
    """Display the chat history for the given component type."""
    if st.session_state.chat_history[component_type]:
        st.markdown("### Conversation History")
        for message in st.session_state.chat_history[component_type]:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f"**User:** {content}")
            else:
                st.markdown(f"**AI:** {content}")
                st.markdown("---")

def instruction_tab(model_id, params):
    """Interface for demonstrating the instruction component of prompts."""
    component_header(
        "Instruction Component",
        "The instruction component clearly tells the model what task to perform or what you want it to do."
    )
    
    # Information box
    with st.expander("Learn about Instruction Components", expanded=False):
        st.markdown("""
        ### What is an Instruction Component?
        
        The instruction component is the core directive that tells the LLM what task to perform. 
        It's often the first part of your prompt and sets the primary goal.
        
        ### Characteristics of Effective Instructions:
        
        - **Clear and specific**: Avoid ambiguity in what you're asking
        - **Action-oriented**: Use direct verbs that specify the task
        - **Appropriate scope**: Neither too broad nor too narrow
        - **Focused on a single task**: Avoid multiple instructions in one prompt
        
        ### Examples of Instruction Verbs:
        - Summarize
        - Analyze
        - Compare
        - Generate
        - Classify
        - Explain
        - Translate
        - Extract
        """)
    
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant designed to follow instructions clearly and provide informative responses.",
        height=80,
        key="instruction_system"
    )
    
    # Example instruction prompts with varying quality
    example_prompts = {
        "Select an example": "",
        "Vague Instruction": "Tell me about climate change.",
        "Specific Instruction": "Summarize the key scientific consensus on climate change and list three major impacts according to recent IPCC reports.",
        "Multi-part Instruction": "Analyze the current state of renewable energy adoption worldwide and provide recommendations for accelerating the transition.",
        "Very Specific Instruction": "Create a 5-day meal plan for a vegetarian athlete focused on protein intake of at least 100g per day, with each meal under 30 minutes preparation time.",
        "Custom": "Write your own instruction..."
    }
    
    selected_example = st.selectbox(
        "Choose an instruction example",
        options=list(example_prompts.keys()),
        key="instruction_examples"
    )
    
    if selected_example == "Custom":
        user_prompt = st.text_area(
            "Your instruction",
            value="",
            height=120,
            placeholder="Enter your own instruction for the model...",
            key="instruction_custom"
        )
    else:
        user_prompt = example_prompts[selected_example]
        if user_prompt:
            st.text_area(
                "Instruction to test",
                value=user_prompt,
                height=120,
                key="instruction_selected"
            )
    
    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Generate Response", type="primary", key="instruction_submit")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display instruction quality analysis before sending
    if user_prompt and selected_example != "Select an example":
        st.markdown("<div class='prompt-analysis'>", unsafe_allow_html=True)
        st.markdown("#### Instruction Analysis")
        
        # Simple analysis of instruction quality
        instruction_quality = {
            "Vague Instruction": {
                "score": "Low",
                "feedback": "This instruction is very broad and vague. It doesn't specify what aspect of climate change to focus on or what type of information is needed.",
                "improvement": "Specify the aspect of climate change you're interested in (causes, effects, solutions) and what depth of information you need."
            },
            "Specific Instruction": {
                "score": "High",
                "feedback": "This instruction is specific and clear. It defines both the task (summarize) and the specific content to focus on.",
                "improvement": "You could further improve by specifying the desired length of the summary."
            },
            "Multi-part Instruction": {
                "score": "Medium",
                "feedback": "This instruction clearly specifies the task but contains multiple parts (analysis and recommendations) which could lead to less focused results for each part.",
                "improvement": "Consider splitting into separate prompts or specifying how much detail you want for each part."
            },
            "Very Specific Instruction": {
                "score": "Very High",
                "feedback": "This instruction is extremely specific with clear parameters and constraints, which will likely yield highly targeted results.",
                "improvement": "Nearly optimal, though you could specify whether variety or simplicity is more important for the meal plans."
            }
        }
        
        if selected_example in instruction_quality:
            quality = instruction_quality[selected_example]
            st.markdown(f"**Clarity Score:** {quality['score']}")
            st.markdown(f"**Feedback:** {quality['feedback']}")
            st.markdown(f"**Potential Improvement:** {quality['improvement']}")
        else:
            # Simple automated analysis for custom instructions
            words = len(user_prompt.split())
            specificity_indicators = ["specific", "exactly", "precisely", "following", "steps", 
                                     "detailed", "bullet points", "numbered", "format", "using"]
            
            specificity_score = sum(1 for word in specificity_indicators if word in user_prompt.lower())
            
            if words < 5:
                st.markdown("**Clarity Score:** Very Low")
                st.markdown("**Feedback:** Your instruction is extremely brief, which likely makes it too vague.")
                st.markdown("**Potential Improvement:** Add more specific details about what you want.")
            elif words < 10:
                st.markdown("**Clarity Score:** Low to Medium")
                st.markdown("**Feedback:** Your instruction could use more specificity.")
                st.markdown("**Potential Improvement:** Add details about format, scope, or specific aspects you want covered.")
            elif specificity_score >= 3:
                st.markdown("**Clarity Score:** High")
                st.markdown("**Feedback:** Your instruction contains good specificity markers.")
                st.markdown("**Potential Improvement:** Consider if any constraints or output format should be specified.")
            else:
                st.markdown("**Clarity Score:** Medium")
                st.markdown("**Feedback:** Your instruction has decent length but could use more specific direction.")
                st.markdown("**Potential Improvement:** Add words like 'specific', 'exactly', or details about format/structure.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chat history
    display_chat_history("instruction")
    
    if submit and user_prompt:
        # Create message structure
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        messages = [message]
        
        # Save to history
        st.session_state.chat_history["instruction"].append({"role": "user", "content": user_prompt})
        
        # System prompts
        system_prompts = [{"text": system_prompt}]
        
        # Response area
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        st.markdown("**Generating response based on your instruction:**")
        
        # Stream the response
        try:
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            success = stream_conversation(bedrock_client, model_id, messages, system_prompts, params, "instruction")
            
            if success:
                st.markdown("""
                **Instruction Component Reflection:**
                
                Notice how the clarity and specificity of the instruction affected the response quality. 
                Effective instructions lead to more focused and useful outputs.
                """)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def context_tab(model_id, params):
    """Interface for demonstrating the context component of prompts."""
    component_header(
        "Context Component",
        "The context component provides relevant background information that helps the model generate more accurate and appropriate responses.",
        color="#8B5CF6"
    )
    
    # Information box
    with st.expander("Learn about Context Components", expanded=False):
        st.markdown("""
        ### What is a Context Component?
        
        The context component provides background information, constraints, or framing that helps the model understand 
        how to approach your request. It creates boundaries and guides the model's understanding of the task.
        
        ### Characteristics of Effective Context:
        
        - **Relevant**: Only include information that's pertinent to the task
        - **Concise**: Keep it brief while including necessary details
        - **Ordered logically**: Present information in a way that builds understanding
        - **Clear about assumptions**: Specify the perspective, audience, or constraints
        
        ### Types of Context:
        - Background information
        - Audience specification
        - Purpose statement
        - Constraints or limitations
        - Domain-specific information
        - Tone or style guidance
        """)
    
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant designed to provide responses based on the context given in prompts.",
        height=80,
        key="context_system"
    )
    
    # Example context prompts with varying quality
    example_prompts = {
        "Select an example": "",
        "No Context": "Write an email to a client about our new service.",
        "Basic Context": "Write an email to a client about our new cloud storage service. The client is a small business owner who previously mentioned they're having trouble with data management.",
        "Rich Context": "Write an email to a client about our new cloud storage service called 'SecureVault'. The client is Sarah, a small business owner (25 employees) in the healthcare sector who previously mentioned they're having trouble with HIPAA-compliant data management. Our service costs $50/month and offers end-to-end encryption with unlimited storage.",
        "Irrelevant Context": "Write an email to a client about our new cloud storage service. The weather has been quite rainy this week, and my dog just had puppies. The client needs secure data storage solutions.",
        "Custom": "Write your own prompt with context..."
    }
    
    selected_example = st.selectbox(
        "Choose a context example",
        options=list(example_prompts.keys()),
        key="context_examples"
    )
    
    if selected_example == "Custom":
        user_prompt = st.text_area(
            "Your prompt with context",
            value="",
            height=150,
            placeholder="Enter your prompt with appropriate context...",
            key="context_custom"
        )
    else:
        user_prompt = example_prompts[selected_example]
        if user_prompt:
            st.text_area(
                "Prompt to test",
                value=user_prompt,
                height=150,
                key="context_selected"
            )
    
    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Generate Response", type="primary", key="context_submit")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display context quality analysis
    if user_prompt and selected_example != "Select an example":
        st.markdown("<div class='prompt-analysis'>", unsafe_allow_html=True)
        st.markdown("#### Context Analysis")
        
        # Analysis of context quality
        context_quality = {
            "No Context": {
                "score": "Very Low",
                "feedback": "This prompt lacks any context about the service, the client, or the purpose of the email.",
                "improvement": "Add details about the service, client information, and the purpose of the communication."
            },
            "Basic Context": {
                "score": "Medium",
                "feedback": "This prompt provides basic context about the service type and client situation, which gives direction for the response.",
                "improvement": "Add more specific details about the service features, pricing, or specific client needs."
            },
            "Rich Context": {
                "score": "Very High",
                "feedback": "This prompt provides comprehensive context with specific details about the service, client, industry, needs, and constraints.",
                "improvement": "Nearly optimal. Could potentially include a desired tone or email length if that's important."
            },
            "Irrelevant Context": {
                "score": "Low",
                "feedback": "This prompt contains irrelevant personal information that doesn't help shape the response and might confuse the model.",
                "improvement": "Remove irrelevant details and replace with information about the service features, benefits, or client needs."
            }
        }
        
        if selected_example in context_quality:
            quality = context_quality[selected_example]
            st.markdown(f"**Context Quality:** {quality['score']}")
            st.markdown(f"**Feedback:** {quality['feedback']}")
            st.markdown(f"**Potential Improvement:** {quality['improvement']}")
            
            # Visual highlighting of context elements
            if selected_example == "Rich Context":
                st.markdown("""
                **Context Elements Identified:**
                - <span class='component-tag tag-context'>Service Details</span> SecureVault, $50/month, end-to-end encryption, unlimited storage
                - <span class='component-tag tag-context'>Client Info</span> Sarah, small business owner, 25 employees, healthcare sector
                - <span class='component-tag tag-context'>Client Need</span> HIPAA-compliant data management
                - <span class='component-tag tag-context'>Purpose</span> Introducing a service that solves a specific problem
                """, unsafe_allow_html=True)
        else:
            # Simple automated context analysis for custom prompts
            words = len(user_prompt.split())
            context_indicators = ["because", "since", "as", "given that", "considering", "based on",
                                 "regarding", "related to", "for context", "background", "situation"]
            
            specifics = ["specific", "detailed", "exactly", "particular", "named", "identified"]
            
            context_score = sum(1 for phrase in context_indicators if phrase in user_prompt.lower())
            specifics_score = sum(1 for word in specifics if word in user_prompt.lower())
            
            if words < 15:
                st.markdown("**Context Quality:** Low")
                st.markdown("**Feedback:** Your prompt is brief and likely doesn't contain sufficient context.")
                st.markdown("**Potential Improvement:** Add relevant background information, specifics about the situation or task.")
            elif context_score >= 2 and specifics_score >= 1:
                st.markdown("**Context Quality:** High")
                st.markdown("**Feedback:** Your prompt contains good context indicators and specific details.")
                st.markdown("**Potential Improvement:** Ensure all context is relevant to the task and organized logically.")
            elif context_score >= 1:
                st.markdown("**Context Quality:** Medium")
                st.markdown("**Feedback:** Your prompt contains some context but could benefit from more specific details.")
                st.markdown("**Potential Improvement:** Add specific names, numbers, or details relevant to the situation.")
            else:
                st.markdown("**Context Quality:** Medium-Low")
                st.markdown("**Feedback:** Your prompt has adequate length but may lack clear contextual framing.")
                st.markdown("**Potential Improvement:** Add phrases like 'given that,' 'considering,' or 'based on' to clearly mark context.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chat history
    display_chat_history("context")
    
    if submit and user_prompt:
        # Create message structure
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        messages = [message]
        
        # Save to history
        st.session_state.chat_history["context"].append({"role": "user", "content": user_prompt})
        
        # System prompts
        system_prompts = [{"text": system_prompt}]
        
        # Response area
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        st.markdown("**Generating response based on your context:**")
        
        # Stream the response
        try:
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            success = stream_conversation(bedrock_client, model_id, messages, system_prompts, params, "context")
            
            if success:
                st.markdown("""
                **Context Component Reflection:**
                
                Notice how the amount and relevance of context affected the specificity and appropriateness of the response.
                Rich, relevant context allows the model to generate more tailored and useful outputs.
                """)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def input_data_tab(model_id, params):
    """Interface for demonstrating the input data component of prompts."""
    component_header(
        "Input Data Component",
        "The input data component provides the specific information that the model needs to process or analyze.",
        color="#EC4899"
    )
    
    # Information box
    with st.expander("Learn about Input Data Components", expanded=False):
        st.markdown("""
        ### What is an Input Data Component?
        
        The input data component is the specific content that you want the model to work with. This could be text for analysis,
        a problem to solve, a question to answer, or information to transform.
        
        ### Characteristics of Effective Input Data:
        
        - **Well-formatted**: Structured appropriately for the task
        - **Clean**: Free of unnecessary elements that might confuse the model
        - **Relevant**: Only includes data needed for the task
        - **Right-sized**: Detailed enough but not overwhelming
        
        ### Common Input Data Types:
        - Text for summarization or analysis
        - Problems for solving
        - Questions for answering
        - Data for transformation (e.g., JSON to format, text to translate)
        - Parameters for content generation
        """)
    
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant designed to process and analyze input data accurately and effectively.",
        height=80,
        key="input_data_system"
    )
    
    # Set up tabs for different input data examples
    input_tabs = st.tabs(["Text Analysis", "Data Transformation", "Problem Solving"])
    
    with input_tabs[0]:  # Text Analysis
        text_examples = {
            "Select an example": "",
            "Raw Unformatted Text": """customer feedback our new app is ok i guess but it crashes sometimes when i try to upload photos and the interface is confusing took me forever to find settings also why is there no dark mode these days every app should have dark mode the response time is decent though""",
            "Formatted Text": """Customer Feedback:
            
Our new app is ok I guess, but it crashes sometimes when I try to upload photos. The interface is confusing - took me forever to find settings. Also, why is there no dark mode? These days every app should have dark mode. The response time is decent though.""",
            "Custom": "Paste your own text for analysis..."
        }
        
        text_selected = st.selectbox(
            "Choose a text example",
            options=list(text_examples.keys()),
            key="text_examples"
        )
        
        if text_selected == "Custom":
            text_input = st.text_area(
                "Your text input",
                value="",
                height=150,
                placeholder="Enter the text you want to analyze...",
                key="text_custom"
            )
        else:
            text_input = text_examples[text_selected]
            if text_input:
                st.text_area(
                    "Text to analyze",
                    value=text_input,
                    height=150,
                    key="text_selected"
                )
        
        text_instruction = st.text_input(
            "Analysis instruction",
            value="Identify the key issues and positive points from this customer feedback.",
            key="text_instruction"
        )
        
        text_prompt = f"{text_instruction}\n\nHere's the text to analyze:\n\n{text_input}"
    
    with input_tabs[1]:  # Data Transformation
        data_examples = {
            "Select an example": "",
            "Unstructured Data": """Name: John Smith
Email: john.smith@example.com
Age: 34
Occupation: Software Developer
Skills: Python, JavaScript, AWS
Years of Experience: 8
Education: Bachelor's in Computer Science
Location: San Francisco, CA""",
            "JSON Data": """
{
  "customer": {
    "name": "Jane Doe",
    "id": "CD-12345",
    "contact": {
      "email": "jane.doe@example.com",
      "phone": "555-123-4567"
    },
    "orders": [
      {
        "order_id": "ORD-789",
        "date": "2023-04-15",
        "items": [
          {"product": "Laptop", "price": 1299.99, "quantity": 1},
          {"product": "Mouse", "price": 24.99, "quantity": 2}
        ]
      }
    ]
  }
}""",
            "Custom": "Paste your own data for transformation..."
        }
        
        data_selected = st.selectbox(
            "Choose a data example",
            options=list(data_examples.keys()),
            key="data_examples"
        )
        
        if data_selected == "Custom":
            data_input = st.text_area(
                "Your data input",
                value="",
                height=150,
                placeholder="Enter the data you want to transform...",
                key="data_custom"
            )
        else:
            data_input = data_examples[data_selected]
            if data_input:
                st.text_area(
                    "Data to transform",
                    value=data_input,
                    height=150,
                    key="data_selected"
                )
        
        transformation_options = [
            "Convert to JSON format",
            "Convert to markdown table",
            "Extract specific fields",
            "Format as HTML",
            "Custom instruction"
        ]
        
        transformation = st.selectbox(
            "Transformation type",
            options=transformation_options,
            key="transformation_type"
        )
        
        if transformation == "Custom instruction":
            data_instruction = st.text_input(
                "Transformation instruction",
                value="",
                placeholder="Enter your custom data transformation instruction...",
                key="data_custom_instruction"
            )
        elif transformation == "Extract specific fields":
            data_instruction = st.text_input(
                "Fields to extract",
                value="name, email, location",
                key="extraction_fields"
            )
            data_instruction = f"Extract the following fields from the data: {data_instruction}"
        else:
            data_instruction = transformation
        
        data_prompt = f"{data_instruction}\n\nHere's the data to transform:\n\n{data_input}"
    
    with input_tabs[2]:  # Problem Solving
        problem_examples = {
            "Select an example": "",
            "Mathematical Problem": "A store is offering a 15% discount on a product that normally costs $85. After the discount, there is a 7% sales tax applied. What is the final price the customer pays?",
            "Logical Puzzle": "Four people (Alex, Blair, Casey, and Dana) are sitting at a table. If Alex is sitting across from Blair and Casey is sitting to the left of Dana, who is sitting across from Casey?",
            "Custom": "Enter your own problem to solve..."
        }
        
        problem_selected = st.selectbox(
            "Choose a problem example",
            options=list(problem_examples.keys()),
            key="problem_examples"
        )
        
        if problem_selected == "Custom":
            problem_input = st.text_area(
                "Your problem",
                value="",
                height=120,
                placeholder="Enter the problem you want solved...",
                key="problem_custom"
            )
        else:
            problem_input = problem_examples[problem_selected]
            if problem_input:
                st.text_area(
                    "Problem to solve",
                    value=problem_input,
                    height=120,
                    key="problem_selected"
                )
        
        solution_format = st.selectbox(
            "Solution format",
            options=["Step by step solution", "Just the answer", "Visual explanation", "Custom format"],
            key="solution_format"
        )
        
        if solution_format == "Custom format":
            format_instruction = st.text_input(
                "Format instruction",
                value="",
                placeholder="Describe your desired format...",
                key="format_custom"
            )
        else:
            format_instruction = solution_format
        
        problem_prompt = f"Please solve this problem: {problem_input}\n\nProvide your solution in this format: {format_instruction}"
    
    # Determine which tab is active and use the corresponding prompt
    active_tab_index = 0
    for i, tab in enumerate(input_tabs):
        if tab.active:
            active_tab_index = i
            break
    
    if active_tab_index == 0:
        user_prompt = text_prompt if text_input else ""
        input_type = "text analysis"
    elif active_tab_index == 1:
        user_prompt = data_prompt if data_input else ""
        input_type = "data transformation"
    else:
        user_prompt = problem_prompt if problem_input else ""
        input_type = "problem solving"
    
    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Process Input Data", type="primary", key="input_data_submit")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display input data quality analysis
    if user_prompt:
        st.markdown("<div class='prompt-analysis'>", unsafe_allow_html=True)
        st.markdown("#### Input Data Analysis")
        
        # Simple quality analysis
        if input_type == "text analysis":
            if text_selected == "Raw Unformatted Text":
                st.markdown("**Input Quality:** Low")
                st.markdown("**Feedback:** The text is unformatted, lacks punctuation, and has inconsistent capitalization.")
                st.markdown("**Potential Improvement:** Format with proper sentences, paragraphs, and punctuation.")
            elif text_selected == "Formatted Text":
                st.markdown("**Input Quality:** High")
                st.markdown("**Feedback:** The text is well-formatted with proper structure and punctuation.")
            else:
                # Basic custom text analysis
                lines = text_input.count('\n') + 1
                sentences = text_input.count('.') + text_input.count('!') + text_input.count('?')
                if lines > 1 and sentences > 1:
                    st.markdown("**Input Quality:** Good")
                    st.markdown("**Feedback:** The text appears to have structure with multiple lines and sentences.")
                else:
                    st.markdown("**Input Quality:** Could be improved")
                    st.markdown("**Feedback:** Consider adding more structure with paragraphs and proper punctuation.")
        
        elif input_type == "data transformation":
            if data_selected == "JSON Data":
                st.markdown("**Input Quality:** High")
                st.markdown("**Feedback:** The data is already in a structured JSON format, which is excellent for processing.")
            elif data_selected == "Unstructured Data":
                st.markdown("**Input Quality:** Medium")
                st.markdown("**Feedback:** The data has clear labels but lacks a formal structure like JSON or CSV.")
                st.markdown("**Potential Improvement:** Consider structuring in a standard format like JSON or a table.")
            else:
                # Basic custom data analysis
                try:
                    json.loads(data_input)
                    st.markdown("**Input Quality:** High - Valid JSON")
                except:
                    if ":" in data_input and "\n" in data_input:
                        st.markdown("**Input Quality:** Medium")
                        st.markdown("**Feedback:** The data appears to have key-value pairs but isn't in a standard format.")
                    else:
                        st.markdown("**Input Quality:** Needs improvement")
                        st.markdown("**Feedback:** Consider structuring your data with clear labels and formatting.")
        
        elif input_type == "problem solving":
            words = len(problem_input.split())
            if words < 10:
                st.markdown("**Input Quality:** Potentially too brief")
                st.markdown("**Feedback:** Very short problem statements might lack necessary details.")
            elif words > 100:
                st.markdown("**Input Quality:** Potentially too verbose")
                st.markdown("**Feedback:** Very long problem statements might contain distracting details.")
            else:
                st.markdown("**Input Quality:** Good length")
                st.markdown("**Feedback:** The problem statement appears to be an appropriate length.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chat history
    display_chat_history("input_data")
    
    if submit and user_prompt:
        # Create message structure
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        messages = [message]
        
        # Save to history
        st.session_state.chat_history["input_data"].append({"role": "user", "content": user_prompt})
        
        # System prompts
        system_prompts = [{"text": system_prompt}]
        
        # Response area
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        st.markdown(f"**Processing {input_type} data:**")
        
        # Stream the response
        try:
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            success = stream_conversation(bedrock_client, model_id, messages, system_prompts, params, "input_data")
            
            if success:
                st.markdown("""
                **Input Data Component Reflection:**
                
                Notice how the quality and structure of the input data affected the model's ability to process it effectively.
                Well-structured inputs typically lead to more accurate and useful outputs.
                """)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def output_indicator_tab(model_id, params):
    """Interface for demonstrating the output indicator component of prompts."""
    component_header(
        "Output Indicator Component",
        "The output indicator component specifies the desired format, structure, or style of the model's response.",
        color="#10B981"
    )
    
    # Information box
    with st.expander("Learn about Output Indicator Components", expanded=False):
        st.markdown("""
        ### What are Output Indicator Components?
        
        Output indicators give the LLM explicit instructions about how you want the response to be formatted or structured.
        These control both the content and presentation of the output.
        
        ### Characteristics of Effective Output Indicators:
        
        - **Explicit format specification**: Clear instruction on the structure
        - **Consistent with the task**: Format should match the purpose
        - **Detailed when necessary**: Describe exactly what elements should be included
        - **Example formatting**: Sometimes showing an example format is clearer than describing it
        
        ### Common Output Indicators:
        - Format specifications (JSON, markdown, bullet points)
        - Style guidance (formal, casual, technical)
        - Length constraints (concise, detailed, exact word counts)
        - Structure templates (specific sections to include)
        - Output delimiters
        """)
    
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant designed to respond in the format specified by the user.",
        height=80,
        key="output_system"
    )
    
    # Task description that will be consistent
    task_description = st.text_area(
        "Task Description (content will stay the same)",
        value="Explain the three main causes of climate change and their environmental impacts.",
        height=80,
        key="task_description"
    )
    
    # Different output formats
    output_formats = {
        "No Format Specified": "",
        "Bullet Points": "Format your answer as a bullet point list.",
        "JSON Structure": "Format your answer as a JSON object with 'causes' as an array, where each cause has a 'name', 'description', and 'impacts' field.",
        "Short vs Detailed": "Provide two versions of your answer: first a 2-sentence summary, then a detailed explanation with at least 3 paragraphs.",
        "Academic Format": "Format your response as an academic mini-paper with Introduction, Main Points, and Conclusion sections. Include at least one citation in [Author, Year] format.",
        "Custom": "Specify your own output format..."
    }
    
    selected_format = st.selectbox(
        "Choose an output format",
        options=list(output_formats.keys()),
        key="output_formats"
    )
    
    if selected_format == "Custom":
        format_instruction = st.text_area(
            "Your format instruction",
            value="",
            height=100,
            placeholder="Describe exactly how you want the output formatted...",
            key="format_custom"
        )
    else:
        format_instruction = output_formats[selected_format]
    
    # Combine task and format instruction
    if format_instruction:
        user_prompt = f"{task_description}\n\n{format_instruction}"
    else:
        user_prompt = task_description
    
    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Generate Response", type="primary", key="output_submit")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display output format quality analysis
    if format_instruction:
        st.markdown("<div class='prompt-analysis'>", unsafe_allow_html=True)
        st.markdown("#### Output Format Analysis")
        
        # Format quality analysis
        format_quality = {
            "No Format Specified": {
                "score": "Not Applicable",
                "feedback": "No specific format has been requested. The model will use its default formatting approach.",
                "effect": "Responses will likely be in paragraph form with a structure the model deems appropriate."
            },
            "Bullet Points": {
                "score": "Basic",
                "feedback": "This is a simple and clear format instruction that's easy for the model to follow.",
                "effect": "Will produce a concise, scannable response that highlights key points separately."
            },
            "JSON Structure": {
                "score": "Detailed",
                "feedback": "This provides a specific structured format with clear fields and organization.",
                "effect": "Will produce machine-readable output that's well-organized and consistent."
            },
            "Short vs Detailed": {
                "score": "Complex",
                "feedback": "This requests multiple output formats in a single response, showing versatility.",
                "effect": "Will demonstrate the model's ability to vary detail levels and provide layered information."
            },
            "Academic Format": {
                "score": "Highly Structured",
                "feedback": "This mimics a specific conventional document structure with multiple required elements.",
                "effect": "Will organize information in a familiar academic pattern that follows scholarly conventions."
            }
        }
        
        if selected_format in format_quality:
            quality = format_quality[selected_format]
            st.markdown(f"**Format Complexity:** {quality['score']}")
            st.markdown(f"**Feedback:** {quality['feedback']}")
            st.markdown(f"**Expected Effect:** {quality['effect']}")
        else:
            # Simple analysis for custom format
            words = len(format_instruction.split())
            structure_indicators = ["json", "bullet", "numbered", "sections", "headings", "paragraph", "table", "markdown", "format", "structure"]
            style_indicators = ["tone", "formal", "casual", "professional", "friendly", "technical", "simple", "complex"]
            
            structure_score = sum(1 for word in structure_indicators if word.lower() in format_instruction.lower())
            style_score = sum(1 for word in style_indicators if word.lower() in format_instruction.lower())
            
            if structure_score >= 2:
                st.markdown("**Format Specificity:** High")
                st.markdown("**Feedback:** Your format instructions include clear structural elements.")
            elif words < 10:
                st.markdown("**Format Specificity:** Low")
                st.markdown("**Feedback:** Your format instructions are brief and may lack specific details.")
            else:
                st.markdown("**Format Specificity:** Medium")
                st.markdown("**Feedback:** Your format instructions have moderate detail.")
            
            if style_score >= 1:
                st.markdown("**Style Guidance:** Included")
                st.markdown("**Tip:** Style guidance helps control the tone and approach of the response.")
            else:
                st.markdown("**Style Guidance:** Not specified")
                st.markdown("**Tip:** Consider adding style or tone instructions (formal, casual, technical, etc.).")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chat history
    display_chat_history("output_indicator")
    
    if submit and user_prompt:
        # Create message structure
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        messages = [message]
        
        # Save to history
        st.session_state.chat_history["output_indicator"].append({"role": "user", "content": user_prompt})
        
        # System prompts
        system_prompts = [{"text": system_prompt}]
        
        # Response area
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        if format_instruction:
            st.markdown(f"**Generating response with {selected_format.lower()} formatting:**")
        else:
            st.markdown("**Generating response with no specific format:**")
        
        # Stream the response
        try:
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            success = stream_conversation(bedrock_client, model_id, messages, system_prompts, params, "output_indicator")
            
            if success:
                st.markdown("""
                **Output Indicator Reflection:**
                
                Notice how specifying the output format affects the structure and presentation of the information.
                Clear format instructions help ensure the response meets your specific needs.
                """)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ------- MAIN APP -------

def main():
    # Initialize session state
    init_session_state()
    
    # Add sidebar content
    sidebar_content()

    # Header
    st.markdown("<h1 class='main-header'>Effective Prompt Components</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This interactive dashboard demonstrates the key components of effective prompts for Large Language Models.
    Learn how each component contributes to creating clear, effective prompts that get the results you want.
    """)


    # Create a 70/30 layout
    col1, col2 = st.columns([0.7, 0.3])     
        # Get model and parameters from the right column
    with col2:
        model_id, params = parameter_section()
        
    with col1:    
        
        # Create tabs for different prompt components
        component_tabs = st.tabs([
            "📝 Instruction",
            "🔍 Context",
            "📊 Input Data",
            "📋 Output Indicator"
        ])
        # Populate each tab
        with component_tabs[0]:
            instruction_tab(model_id, params)
        
        with component_tabs[1]:
            context_tab(model_id, params)
        
        with component_tabs[2]:
            input_data_tab(model_id, params)
        
        with component_tabs[3]:
            output_indicator_tab(model_id, params)
    
    # Footer
    st.markdown("<div class='footer'>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()

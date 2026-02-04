# Amazon Bedrock Converse API with LangChain Prompt Templates

import streamlit as st
import logging
import sys
import boto3
from botocore.exceptions import ClientError
import os
from io import BytesIO
from PIL import Image
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import uuid
import time
import utils.common as common
import utils.authenticate as authenticate
from utils.styles import load_css, sub_header
# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Page configuration with improved styling
st.set_page_config(
    page_title="Prompt Template",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------- INITIALIZE SESSION STATE -------
def init_session_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "templates" not in st.session_state:
        st.session_state.templates = get_sample_templates()
    
    if "selected_template" not in st.session_state:
        st.session_state.selected_template = None

# ------- SAMPLE PROMPT TEMPLATES -------
def get_sample_templates():
    return [
        {
            "name": "Professional Email Writer",
            "description": "Creates professional emails based on your requirements",
            "template": """
Write a professional email with the following details:
- Subject: {subject}
- Purpose: {purpose}
- Tone: {tone}
- Key points: {key_points}
- Call to action: {call_to_action}
- Closing remarks: {closing}

Please format this properly as an email with subject line, greeting, body, and signature.
            """,
            "variables": {
                "subject": "Proposal for Marketing Campaign Collaboration",
                "purpose": "To propose a joint marketing campaign with a partner company",
                "tone": "professional yet friendly",
                "key_points": "Previous success metrics, timeline for the campaign, resource requirements",
                "call_to_action": "Schedule a meeting next week to discuss details",
                "closing": "Emphasize mutual benefits and long-term partnership potential"
            }
        },
        {
            "name": "Product Description Generator",
            "description": "Creates compelling product descriptions for e-commerce",
            "template": """
Write a compelling product description for an e-commerce website using the following details:
- Product name: {product_name}
- Category: {category}
- Key features: {features}
- Target audience: {audience}
- Unique selling points: {selling_points}
- Price point: {price_point}

The description should be engaging, highlight benefits, and include a strong call to action.
            """,
            "variables": {
                "product_name": "UltraGlide Pro Wireless Mouse",
                "category": "Computer Peripherals",
                "features": "Ergonomic design, 12 programmable buttons, 4000 DPI precision, 6-month battery life",
                "audience": "Gamers and professional designers",
                "selling_points": "Zero lag technology, customizable RGB lighting, compatible with all operating systems",
                "price_point": "Premium ($85-95)"
            }
        },
        {
            "name": "Technical Documentation Writer",
            "description": "Creates clear and concise technical documentation",
            "template": """
Create a technical documentation section for the following:
- Feature/Function name: {feature_name}
- Purpose: {purpose}
- Technical requirements: {requirements}
- Implementation steps: {implementation_steps}
- Code examples (if applicable): {code_examples}
- Potential issues and solutions: {issues_solutions}
- Related features/functions: {related_features}

The documentation should be clear, concise, and follow best practices for technical writing.
            """,
            "variables": {
                "feature_name": "Data Synchronization API",
                "purpose": "To enable real-time data synchronization between mobile and cloud databases",
                "requirements": "SDK version 3.2+, API key, secure connection, 5MB/s minimum bandwidth",
                "implementation_steps": "1. Initialize client, 2. Configure sync parameters, 3. Set up error handling, 4. Implement retry logic",
                "code_examples": "client.initSync(config); client.startSync();",
                "issues_solutions": "Connection timeout: Implement exponential backoff; Data conflict: Use version control strategy",
                "related_features": "Offline storage, Conflict resolution, Data compression"
            }
        },
        {
            "name": "Social Media Post Generator",
            "description": "Creates engaging social media content for multiple platforms",
            "template": """
Generate social media post content for the following:
- Platform: {platform}
- Brand voice: {brand_voice}
- Target audience: {audience}
- Campaign theme: {campaign_theme}
- Key message: {key_message}
- Call to action: {call_to_action}
- Hashtags style: {hashtags}

Provide a caption that's optimized for the selected platform and audience. Include emojis where appropriate.
            """,
            "variables": {
                "platform": "Instagram",
                "brand_voice": "Friendly, inspiring, health-conscious",
                "audience": "Fitness enthusiasts aged 25-40",
                "campaign_theme": "Summer Fitness Challenge",
                "key_message": "Achieve your summer fitness goals with our 30-day program",
                "call_to_action": "Sign up for the challenge through the link in bio",
                "hashtags": "Mix of trending and branded (5-8 hashtags)"
            }
        },
        {
            "name": "Customer Support Response",
            "description": "Generates helpful customer service responses",
            "template": """
Create a customer support response with the following details:
- Customer issue: {customer_issue}
- Product/Service involved: {product}
- Tone: {tone}
- Solution to offer: {solution}
- Additional help resources: {resources}
- Follow-up actions: {follow_up}

The response should acknowledge the customer's issue, provide a clear solution, and maintain a supportive tone throughout.
            """,
            "variables": {
                "customer_issue": "Unable to access premium features after payment was processed",
                "product": "StreamFlix Premium Subscription",
                "tone": "Empathetic and helpful",
                "solution": "Account activation steps and manual override of restrictions",
                "resources": "FAQ link, video tutorial for account settings",
                "follow_up": "Offer to check back in 24 hours, provide direct contact information"
            }
        }
    ]

# ------- API FUNCTIONS -------

def stream_conversation(bedrock_client, model_id, messages, system_prompts, inference_config):
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
                
                if 'contentBlockDelta' in event:
                    chunk = event['contentBlockDelta']
                    if 'text' in chunk['delta']:
                        part = chunk['delta']['text']
                        full_response += part
                        with placeholder.container():
                            st.markdown(full_response)
                
                if 'messageStop' in event:
                    stop_reason = event['messageStop']['stopReason']
                
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
            
            # Return the generated response and metrics
            return {
                "response": full_response,
                "metrics": {
                    "tokens": token_info,
                    "latency_ms": latency_ms,
                    "stop_reason": stop_reason if 'stop_reason' in locals() else "Unknown"
                }
            }
        
        return {"response": "", "metrics": {}}
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return {"response": "", "error": str(err)}

# ------- UI COMPONENTS -------

def parameter_sidebar():
    """Sidebar with model selection and parameter tuning."""
    with st.container(border=True):
        st.markdown(sub_header("Model Settings", "⚙️", "aws"), unsafe_allow_html=True)
        
        MODEL_CATEGORIES = {
        "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0", 
                  "us.amazon.nova-2-lite-v1:0"],
        "Anthropic": ["anthropic.claude-3-haiku-20240307-v1:0",
                         "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                         "us.anthropic.claude-sonnet-4-20250514-v1:0",
                         "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                         "us.anthropic.claude-opus-4-1-20250805-v1:0"],
        "Cohere": ["cohere.command-r-v1:0", "cohere.command-r-plus-v1:0"],
        "Google": ["google.gemma-3-4b-it", "google.gemma-3-12b-it", "google.gemma-3-27b-it"],
        "Meta": ["us.meta.llama3-2-1b-instruct-v1:0", "us.meta.llama3-2-3b-instruct-v1:0",
                    "meta.llama3-8b-instruct-v1:0", "us.meta.llama3-1-8b-instruct-v1:0",
                    "us.meta.llama4-scout-17b-instruct-v1:0", "us.meta.llama4-maverick-17b-instruct-v1:0",
                    "meta.llama3-70b-instruct-v1:0", "us.meta.llama3-1-70b-instruct-v1:0",
                    "us.meta.llama3-3-70b-instruct-v1:0",
                    "us.meta.llama3-2-11b-instruct-v1:0", "us.meta.llama3-2-90b-instruct-v1:0"],
        "Mistral": ["mistral.mistral-7b-instruct-v0:2", "mistral.mistral-small-2402-v1:0",
                       "mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1"],
        "NVIDIA": ["nvidia.nemotron-nano-9b-v2", "nvidia.nemotron-nano-12b-v2"],
        "OpenAI": ["openai.gpt-oss-20b-1:0", "openai.gpt-oss-120b-1:0"],
        "Qwen": ["qwen.qwen3-32b-v1:0", "qwen.qwen3-next-80b-a3b", "qwen.qwen3-235b-a22b-2507-v1:0", "qwen.qwen3-vl-235b-a22b", "qwen.qwen3-coder-30b-a3b-v1:0", "qwen.qwen3-coder-480b-a35b-v1:0"],
        "Writer": ["us.writer.palmyra-x4-v1:0", "us.writer.palmyra-x5-v1:0"]
        }
        
        # Create selectbox for provider first
        provider = st.selectbox("Provider", options=list(MODEL_CATEGORIES.keys()))
        
        # Then create selectbox for models from that provider
        model_id = st.selectbox("Model", options=MODEL_CATEGORIES[provider])
        
        st.markdown(sub_header("Parameters", "🎛️", "aws"), unsafe_allow_html=True)
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05, 
                            help="Higher values make output more random, lower values more deterministic")
        
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05,
                            help="Controls diversity via nucleus sampling")
        
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4096, value=1024, step=50,
                                    help="Maximum number of tokens in the response")
            
        params = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
    
    with st.sidebar:    
       
        common.render_sidebar()
        
 
        if st.button("Clear Output", key="clear_output"):
                st.session_state.conversation_history = []
                st.rerun()
        
        with st.expander("About", expanded=False):
            st.markdown("""
            ### LangChain + Amazon Bedrock
            
            This app demonstrates how to use LangChain Prompt Templates with Amazon Bedrock's foundation models.
            
            Using prompt templates allows you to:
            - Structure your prompts consistently
            - Reuse common prompt patterns
            - Separate prompt logic from application logic
            
            For more information, visit:
            - [LangChain Documentation](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/)
            - [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
            """)
        
    return model_id, params

def display_templates():
    """Display sample prompt templates in a user-friendly way."""
    st.markdown(sub_header("Choose a Template", "🎯"), unsafe_allow_html=True)
    
    # Template icons mapping
    template_icons = {
        "Professional Email Writer": "✉️",
        "Product Description Generator": "🛍️",
        "Technical Documentation Writer": "📚",
        "Social Media Post Generator": "📱",
        "Customer Support Response": "💬"
    }
    
    # Display templates in a grid
    cols = st.columns(3)
    
    for idx, template in enumerate(st.session_state.templates):
        col = cols[idx % 3]
        with col:
            icon = template_icons.get(template['name'], "📝")
            
            # Check if this template is currently selected
            is_selected = st.session_state.selected_template == idx
            selected_class = "template-selected" if is_selected else ""
            
            card_html = f"""
            <div class='template-card {selected_class}'>
                <div class='template-icon'>{icon}</div>
                <h4 class='template-title'>{template['name']}</h4>
                <p class='template-description'>{template['description']}</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Use Template button
            button_label = "✓ Selected" if is_selected else "Use Template"
            button_type = "secondary" if is_selected else "primary"
            
            if st.button(button_label, key=f"template_{idx}", type=button_type, use_container_width=True):
                if not is_selected:
                    st.session_state.selected_template = idx
                    st.rerun()

def prompt_template_interface(model_id, params):
    """Interface using LangChain prompt templates."""
    
    # Display template selection
    display_templates()
    
    # Check if a template is selected
    if st.session_state.selected_template is not None:
        template_data = st.session_state.templates[st.session_state.selected_template]
        
        # Template header with icon
        template_icons = {
            "Professional Email Writer": "✉️",
            "Product Description Generator": "🛍️",
            "Technical Documentation Writer": "📚",
            "Social Media Post Generator": "📱",
            "Customer Support Response": "💬"
        }
        icon = template_icons.get(template_data['name'], "📝")
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem; font-weight: 700;'>
                {icon} {template_data['name']}
            </h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1rem;'>
                {template_data['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show template and allow for editing
        with st.expander("🔧 View/Edit Template", expanded=False):
            st.markdown("""
            <div style='background: #f8f9fb; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;'>
                <p style='margin: 0; color: #5A6D87; font-size: 0.9rem;'>
                    💡 <strong>Tip:</strong> You can customize the template structure below. 
                    Variables are enclosed in curly braces like <code>{'{variable_name}'}</code>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            template_content = st.text_area(
                "Template Structure", 
                value=template_data['template'].strip(), 
                height=200,
                key="template_content",
                help="Edit the template structure. Use {variable_name} for placeholders."
            )
        
        # Input form for template variables
        st.markdown("---")
        st.markdown("### 📝 Fill in Template Variables")
        st.markdown("""
        <div style='background: #fff5e6; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #FF9900;'>
            <p style='margin: 0; color: #232F3E; font-size: 0.9rem;'>
                Customize the values below to generate your content. Each field corresponds to a variable in the template.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dynamically create input fields for each variable
        variable_values = {}
        cols = st.columns(2)
        
        for idx, (var_name, default_value) in enumerate(template_data['variables'].items()):
            col = cols[idx % 2]
            with col:
                variable_values[var_name] = st.text_area(
                    f"📌 {var_name.replace('_', ' ').title()}", 
                    value=default_value,
                    height=100,
                    key=f"var_{var_name}",
                    help=f"Enter value for {var_name}"
                )
        
        # Create LangChain prompt template
        prompt_template = PromptTemplate.from_template(template_content)
        
        # Preview formatted prompt
        st.markdown("---")
        preview_col, button_col = st.columns([3, 1])
        
        with preview_col:
            show_preview = st.checkbox("👁️ Preview formatted prompt", value=False)
        
        if show_preview:
            try:
                formatted_prompt = prompt_template.format(**variable_values)
                st.markdown("""
                <div style='background: #f0f7ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0073BB;'>
                    <h4 style='margin-top: 0; color: #0073BB;'>📄 Formatted Prompt Preview</h4>
                </div>
                """, unsafe_allow_html=True)
                st.code(formatted_prompt, language="text")
            except KeyError as e:
                st.error(f"❌ Missing variable in template: {e}")
        
        # Submit button
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.button("🚀 Generate Response", type="primary", key="template_submit", use_container_width=True)
        
        # Process submission
        if submit:
            try:
                # Format prompt using template
                formatted_prompt = prompt_template.format(**variable_values)
                
                # Add to conversation history (user message)
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": formatted_prompt
                })
                
                # Prepare for API call
                system_prompts = [{"text": "You are a helpful assistant that follows instructions carefully."}]
                
                message = {
                    "role": "user",
                    "content": [{"text": formatted_prompt}]
                }
                messages = [message]
                
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Display response area
                st.markdown("---")
                st.markdown("""
                <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            padding: 1.5rem; border-radius: 12px; margin-top: 1rem;'>
                    <h3 style='margin: 0; color: #232F3E;'>🤖 AI Response</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Stream the response
                with st.status("⏳ Generating response...", expanded=True) as status:
                    result = stream_conversation(bedrock_client, model_id, messages, system_prompts, params)
                    
                    if result and "response" in result:
                        status.update(label="✅ Response generated successfully!", state="complete")
                        
                        # Add to conversation history (assistant message)
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": result["response"]
                        })
                        
                        # Display metrics if available
                        if "metrics" in result and result["metrics"]:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("""
                            <div style='background: #f8f9fb; padding: 1rem; border-radius: 8px; border: 1px solid #E9EBF0;'>
                                <h4 style='margin-top: 0; color: #232F3E;'>📊 Response Metrics</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            metrics = result["metrics"]
                            if "tokens" in metrics:
                                token_info = metrics["tokens"]
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("📥 Input Tokens", token_info.get('input', 'N/A'))
                                with col2:
                                    st.metric("📤 Output Tokens", token_info.get('output', 'N/A'))
                                with col3:
                                    st.metric("📊 Total Tokens", token_info.get('total', 'N/A'))
                                with col4:
                                    if "latency_ms" in metrics:
                                        st.metric("⚡ Latency", f"{metrics['latency_ms']}ms")
                                
                                if "stop_reason" in metrics:
                                    st.caption(f"🛑 Stop reason: {metrics.get('stop_reason', 'Unknown')}")
                    
                    elif "error" in result:
                        status.update(label="❌ Error occurred", state="error")
                        st.error(f"An error occurred: {result['error']}")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    else:
        st.info("👆 Select a template above to get started")

def display_conversation_history():
    """Display the conversation history."""
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 12px; margin: 1.5rem 0;'>
            <h3 style='color: white; margin: 0;'>💬 Conversation History</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, msg in enumerate(st.session_state.conversation_history):
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1rem; border-radius: 12px; margin: 0.75rem 0;
                            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);'>
                    <div style='color: white; font-weight: 600; margin-bottom: 0.5rem;'>
                        👤 User
                    </div>
                    <div style='color: rgba(255,255,255,0.95); line-height: 1.6;'>
                        {msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            padding: 1rem; border-radius: 12px; margin: 0.75rem 0;
                            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);'>
                    <div style='color: #232F3E; font-weight: 600; margin-bottom: 0.5rem;'>
                        🤖 Assistant
                    </div>
                    <div style='color: #232F3E; line-height: 1.6;'>
                        {msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ------- MAIN APP -------

def main():
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("<h1 class='main-header'>Prompt Template</h1>", unsafe_allow_html=True)
    
    st.markdown("""<div class="info-box">
    This interactive dashboard demonstrates how to use LangChain Prompt Templates with Amazon Bedrock's foundation models.
    Select a template, customize the variables, and generate high-quality content with structured prompts.
    </div>""", unsafe_allow_html=True)


    # Create a 70/30 layout
    col1, col2 = st.columns([0.7, 0.3])     
        # Get model and parameters from the right column
    with col2:
        model_id, params = parameter_sidebar()  
   
    with col1:   
       
        # Main content area
        prompt_template_interface(model_id, params)
        
        # Display conversation history
        display_conversation_history()

    # Footer
    st.markdown("---")
    st.markdown("<div class='footer'>© 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()
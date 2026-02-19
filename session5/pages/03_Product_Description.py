import streamlit as st
import logging
import boto3
import uuid
import os
from botocore.exceptions import ClientError
from io import BytesIO
from PIL import Image
import time
import json
from datetime import datetime
from utils.styles import load_css
import utils.common as common
import utils.authenticate as authenticate

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Set page configuration
st.set_page_config(
    page_title="Shoe Product Description Generator",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the custom CSS
load_css()

# ------- SESSION MANAGEMENT FUNCTIONS -------

def initialize_session():
    """Initialize session state variables if they don't exist"""
    
    common.initialize_session_state()
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if "product_specifications" not in st.session_state:
        st.session_state.product_specifications = {}
    
    if "generated_descriptions" not in st.session_state:
        st.session_state.generated_descriptions = {}

# ------- API FUNCTIONS -------

def text_conversation(bedrock_client, model_id, system_prompts, messages, additional_model_fields, **params):
    """Sends messages to a model."""
    logger.info(f"Generating message with model {model_id}")
    
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=params,
            additionalModelRequestFields=additional_model_fields
        )
        
        # Log token usage
        token_usage = response['usage']
        logger.info(f"Input tokens: {token_usage['inputTokens']}")
        logger.info(f"Output tokens: {token_usage['outputTokens']}")
        logger.info(f"Total tokens: {token_usage['totalTokens']}")
        logger.info(f"Stop reason: {response['stopReason']}")
        
        return response
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return None

# ------- UI COMPONENTS -------

def model_selection_panel():
    """Model selection and parameters in the side panel"""
    st.markdown("<h4>Model Selection</h4>", unsafe_allow_html=True)
    
    # Model categories by provider with friendly names
    MODEL_CATEGORIES = {
        "Amazon": {
            "Nova 2 Lite": "us.amazon.nova-2-lite-v1:0",
            "Nova Pro": "us.amazon.nova-pro-v1:0",
            "Nova Lite": "us.amazon.nova-lite-v1:0",
            "Nova Micro": "us.amazon.nova-micro-v1:0",
        },
        "Anthropic": {
            "Claude Sonnet 4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "Claude Sonnet 4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "Claude Opus 4.1": "us.anthropic.claude-opus-4-1-20250805-v1:0",
            "Claude Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        },
        "Meta": {
            "Llama 4 Maverick 17B": "us.meta.llama4-maverick-17b-instruct-v1:0",
            "Llama 4 Scout 17B": "us.meta.llama4-scout-17b-instruct-v1:0",
            "Llama 3.3 70B": "us.meta.llama3-3-70b-instruct-v1:0",
            "Llama 3.2 90B": "us.meta.llama3-2-90b-instruct-v1:0",
            "Llama 3.2 11B": "us.meta.llama3-2-11b-instruct-v1:0",
            "Llama 3.1 70B": "us.meta.llama3-1-70b-instruct-v1:0",
            "Llama 3 70B": "meta.llama3-70b-instruct-v1:0",
        },
        "Mistral AI": {
            "Mistral Large (24.02)": "mistral.mistral-large-2402-v1:0",
            "Mixtral 8x7B": "mistral.mixtral-8x7b-instruct-v0:1",
            "Mistral 7B": "mistral.mistral-7b-instruct-v0:2",
        },
        "Google": {
            "Gemma 3 27B": "google.gemma-3-27b-it",
            "Gemma 3 12B": "google.gemma-3-12b-it",
            "Gemma 3 4B": "google.gemma-3-4b-it",
        },
        "Qwen": {
            "Qwen3 235B": "qwen.qwen3-235b-a22b-2507-v1:0",
            "Qwen3 Next 80B": "qwen.qwen3-next-80b-a3b",
            "Qwen3 32B": "qwen.qwen3-32b-v1:0",
        },
        "Cohere": {
            "Command R+": "cohere.command-r-plus-v1:0",
            "Command R": "cohere.command-r-v1:0",
        },
        "NVIDIA": {
            "Nemotron Nano 12B": "nvidia.nemotron-nano-12b-v2",
            "Nemotron Nano 9B": "nvidia.nemotron-nano-9b-v2",
        },
        "OpenAI": {
            "GPT OSS 120B": "openai.gpt-oss-120b-1:0",
            "GPT OSS 20B": "openai.gpt-oss-20b-1:0",
        },
    }
    
    # Models that support Top K parameter
    MODELS_WITH_TOP_K = [
        "mistral.mistral-large-2402-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
    ]
    
    # Provider selection
    provider = st.selectbox(
        "Model Provider", 
        options=list(MODEL_CATEGORIES.keys()), 
        index=0,
        key="side_provider",
        help="Select the AI model provider"
    )
    
    # Model selection based on provider
    models = MODEL_CATEGORIES[provider]
    selected_model = st.selectbox(
        "Model",
        options=list(models.keys()),
        index=0,
        key="side_model",
        help="Select the specific model"
    )
    
    model_id = models[selected_model]
    
    st.markdown("<h4>Parameter Tuning</h4>", unsafe_allow_html=True)
    
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        key="side_temperature",
        help="Higher values make output more creative, lower values more consistent"
    )

    # Add Top K parameter for models that support it
    top_k = None
    if model_id in MODELS_WITH_TOP_K:
        top_k = st.slider(
            "Top K", 
            min_value=0, 
            max_value=500, 
            value=200, 
            step=10,
            key="side_top_k",
            help="Limits vocabulary to K most likely tokens"
        )
        
    max_tokens = st.number_input(
        "Max Tokens", 
        min_value=100, 
        max_value=4096, 
        value=2048, 
        step=50,
        key="side_max_tokens",
        help="Maximum number of tokens in the response"
    )
    
    # Build params - only include temperature (not topP to avoid conflicts)
    params = {
        "temperature": temperature,
        "maxTokens": max_tokens,
        "stopSequences": []
    }
    
    # Add topK to additional_model_fields only if the model supports it
    if top_k is not None and model_id in MODELS_WITH_TOP_K:
        additional_model_fields = {"top_k": top_k}
    else:
        additional_model_fields = {}
    
    return model_id, params, additional_model_fields

def create_product_specifications_form():
    """Create the product specifications form"""
    st.markdown("## 👟 Product Specifications")
    st.markdown("Configure the specifications for your new shoe product. All fields have default values to get you started.")
    
    specifications = {}
    
    # Create tabs for each specification category
    materials_tab, features_tab, performance_tab, sustainability_tab = st.tabs([
        "🧵 Materials", 
        "⚡ Features", 
        "🏃 Performance", 
        "🌱 Sustainability"
    ])
    
    with materials_tab:
        st.markdown("### Material Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            specifications['upper_material'] = st.selectbox(
                "Upper Material",
                options=["Premium Leather", "Synthetic Leather", "Canvas", "Mesh", "Knit Fabric", "Suede"],
                index=0,
                key="upper_material"
            )
            
            specifications['sole_material'] = st.selectbox(
                "Sole Material",
                options=["Rubber", "EVA Foam", "Polyurethane", "Carbon Fiber", "TPU", "Cork"],
                index=1,
                key="sole_material"
            )
            
            specifications['lining_material'] = st.selectbox(
                "Lining Material",
                options=["Textile", "Leather", "Mesh", "Synthetic", "Bamboo Fiber", "Cotton"],
                index=0,
                key="lining_material"
            )
        
        with col2:
            specifications['insole_material'] = st.selectbox(
                "Insole Material",
                options=["Memory Foam", "EVA", "Gel", "Cork", "Ortholite", "PU Foam"],
                index=0,
                key="insole_material"
            )
            
            specifications['heel_material'] = st.selectbox(
                "Heel Material",
                options=["Air Cushion", "Gel", "Foam", "Spring", "Carbon Plate", "Standard"],
                index=0,
                key="heel_material"
            )
            
            specifications['laces_material'] = st.selectbox(
                "Laces Material",
                options=["Cotton", "Polyester", "Nylon", "Elastic", "Waxed Cotton", "Leather"],
                index=1,
                key="laces_material"
            )
    
    with features_tab:
        st.markdown("### Feature Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            specifications['closure_type'] = st.selectbox(
                "Closure Type",
                options=["Lace-up", "Slip-on", "Velcro", "Buckle", "Zipper", "Elastic"],
                index=0,
                key="closure_type"
            )
            
            specifications['toe_shape'] = st.selectbox(
                "Toe Shape",
                options=["Round", "Pointed", "Square", "Almond", "Open", "Square-round"],
                index=0,
                key="toe_shape"
            )
            
            specifications['heel_height'] = st.selectbox(
                "Heel Height",
                options=["Flat (0-1cm)", "Low (1-3cm)", "Medium (3-5cm)", "High (5-8cm)", "Very High (8cm+)"],
                index=1,
                key="heel_height"
            )
        
        with col2:
            specifications['waterproof'] = st.checkbox(
                "Waterproof",
                value=True,
                key="waterproof"
            )
            
            specifications['breathable'] = st.checkbox(
                "Breathable",
                value=True,
                key="breathable"
            )
            
            specifications['anti_slip'] = st.checkbox(
                "Anti-slip Sole",
                value=True,
                key="anti_slip"
            )
            
            specifications['shock_absorption'] = st.checkbox(
                "Shock Absorption",
                value=True,
                key="shock_absorption"
            )
            
            specifications['arch_support'] = st.checkbox(
                "Arch Support",
                value=True,
                key="arch_support"
            )
    
    with performance_tab:
        st.markdown("### Performance Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            specifications['target_activity'] = st.multiselect(
                "Target Activity",
                options=["Running", "Walking", "Hiking", "Basketball", "Tennis", "Gym Training", "Casual Wear", "Office Work"],
                default=["Running", "Casual Wear"],
                key="target_activity"
            )
            
            specifications['weight'] = st.slider(
                "Weight (grams)",
                min_value=200,
                max_value=800,
                value=350,
                step=10,
                key="weight"
            )
            
            specifications['durability_rating'] = st.slider(
                "Durability Rating (1-10)",
                min_value=1,
                max_value=10,
                value=8,
                key="durability_rating"
            )
        
        with col2:
            specifications['comfort_level'] = st.slider(
                "Comfort Level (1-10)",
                min_value=1,
                max_value=10,
                value=9,
                key="comfort_level"
            )
            
            specifications['flexibility'] = st.slider(
                "Flexibility (1-10)",
                min_value=1,
                max_value=10,
                value=7,
                key="flexibility"
            )
            
            specifications['traction_rating'] = st.slider(
                "Traction Rating (1-10)",
                min_value=1,
                max_value=10,
                value=8,
                key="traction_rating"
            )
    
    with sustainability_tab:
        st.markdown("### Sustainability Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            specifications['eco_friendly_materials'] = st.checkbox(
                "Eco-friendly Materials",
                value=True,
                key="eco_friendly_materials"
            )
            
            specifications['recycled_content'] = st.slider(
                "Recycled Content (%)",
                min_value=0,
                max_value=100,
                value=30,
                step=5,
                key="recycled_content"
            )
            
            specifications['biodegradable'] = st.checkbox(
                "Biodegradable Components",
                value=False,
                key="biodegradable"
            )
        
        with col2:
            specifications['carbon_neutral'] = st.checkbox(
                "Carbon Neutral Production",
                value=True,
                key="carbon_neutral"
            )
            
            specifications['ethical_sourcing'] = st.checkbox(
                "Ethical Sourcing",
                value=True,
                key="ethical_sourcing"
            )
            
            specifications['packaging_sustainable'] = st.checkbox(
                "Sustainable Packaging",
                value=True,
                key="packaging_sustainable"
            )
    
    # Additional product details
    st.markdown("### Additional Product Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        specifications['product_name'] = st.text_input(
            "Product Name",
            value="EcoStep Pro",
            key="product_name"
        )
    
    with col2:
        specifications['brand_name'] = st.text_input(
            "Brand Name",
            value="GreenStride",
            key="brand_name"
        )
    
    with col3:
        specifications['price_range'] = st.selectbox(
            "Price Range",
            options=["Budget ($50-$100)", "Mid-range ($100-$200)", "Premium ($200-$300)", "Luxury ($300+)"],
            index=1,
            key="price_range"
        )
    
    specifications['target_audience'] = st.multiselect(
        "Target Audience",
        options=["Athletes", "Fitness Enthusiasts", "Casual Users", "Professionals", "Outdoor Enthusiasts", "Fashion-conscious", "Eco-conscious Consumers"],
        default=["Athletes", "Eco-conscious Consumers"],
        key="target_audience"
    )
    
    return specifications

def display_specifications_summary(specifications):
    """Display a summary of the product specifications"""
    st.markdown("## 📋 Product Specifications Summary")
    
    # Create expandable sections for each category
    with st.expander("🧵 Materials Summary", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Upper Material:** {specifications['upper_material']}")
            st.write(f"**Sole Material:** {specifications['sole_material']}")
            st.write(f"**Lining Material:** {specifications['lining_material']}")
        with col2:
            st.write(f"**Insole Material:** {specifications['insole_material']}")
            st.write(f"**Heel Material:** {specifications['heel_material']}")
            st.write(f"**Laces Material:** {specifications['laces_material']}")
    
    with st.expander("⚡ Features Summary", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Closure Type:** {specifications['closure_type']}")
            st.write(f"**Toe Shape:** {specifications['toe_shape']}")
            st.write(f"**Heel Height:** {specifications['heel_height']}")
        with col2:
            features = []
            if specifications['waterproof']: features.append("Waterproof")
            if specifications['breathable']: features.append("Breathable")
            if specifications['anti_slip']: features.append("Anti-slip")
            if specifications['shock_absorption']: features.append("Shock Absorption")
            if specifications['arch_support']: features.append("Arch Support")
            st.write(f"**Special Features:** {', '.join(features) if features else 'None'}")
    
    with st.expander("🏃 Performance Summary", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Target Activities:** {', '.join(specifications['target_activity'])}")
            st.write(f"**Weight:** {specifications['weight']}g")
            st.write(f"**Durability Rating:** {specifications['durability_rating']}/10")
        with col2:
            st.write(f"**Comfort Level:** {specifications['comfort_level']}/10")
            st.write(f"**Flexibility:** {specifications['flexibility']}/10")
            st.write(f"**Traction Rating:** {specifications['traction_rating']}/10")
    
    with st.expander("🌱 Sustainability Summary", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            sustainability_features = []
            if specifications['eco_friendly_materials']: sustainability_features.append("Eco-friendly Materials")
            if specifications['biodegradable']: sustainability_features.append("Biodegradable Components")
            if specifications['carbon_neutral']: sustainability_features.append("Carbon Neutral")
            st.write(f"**Sustainability Features:** {', '.join(sustainability_features) if sustainability_features else 'None'}")
        with col2:
            st.write(f"**Recycled Content:** {specifications['recycled_content']}%")
            if specifications['ethical_sourcing']: st.write("✅ **Ethical Sourcing**")
            if specifications['packaging_sustainable']: st.write("✅ **Sustainable Packaging**")
    
    # Product details
    st.markdown("### Product Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Product Name:** {specifications['product_name']}")
    with col2:
        st.write(f"**Brand:** {specifications['brand_name']}")
    with col3:
        st.write(f"**Price Range:** {specifications['price_range']}")
    
    st.write(f"**Target Audience:** {', '.join(specifications['target_audience'])}")

def format_specifications_for_llm(specifications):
    """Format specifications into a structured text for LLM input"""
    formatted_text = f"""
Product Name: {specifications['product_name']}
Brand: {specifications['brand_name']}
Price Range: {specifications['price_range']}
Target Audience: {', '.join(specifications['target_audience'])}

MATERIALS:
- Upper Material: {specifications['upper_material']}
- Sole Material: {specifications['sole_material']}
- Lining Material: {specifications['lining_material']}
- Insole Material: {specifications['insole_material']}
- Heel Material: {specifications['heel_material']}
- Laces Material: {specifications['laces_material']}

FEATURES:
- Closure Type: {specifications['closure_type']}
- Toe Shape: {specifications['toe_shape']}
- Heel Height: {specifications['heel_height']}
- Waterproof: {'Yes' if specifications['waterproof'] else 'No'}
- Breathable: {'Yes' if specifications['breathable'] else 'No'}
- Anti-slip Sole: {'Yes' if specifications['anti_slip'] else 'No'}
- Shock Absorption: {'Yes' if specifications['shock_absorption'] else 'No'}
- Arch Support: {'Yes' if specifications['arch_support'] else 'No'}

PERFORMANCE:
- Target Activities: {', '.join(specifications['target_activity'])}
- Weight: {specifications['weight']}g
- Durability Rating: {specifications['durability_rating']}/10
- Comfort Level: {specifications['comfort_level']}/10
- Flexibility: {specifications['flexibility']}/10
- Traction Rating: {specifications['traction_rating']}/10

SUSTAINABILITY:
- Eco-friendly Materials: {'Yes' if specifications['eco_friendly_materials'] else 'No'}
- Recycled Content: {specifications['recycled_content']}%
- Biodegradable Components: {'Yes' if specifications['biodegradable'] else 'No'}
- Carbon Neutral Production: {'Yes' if specifications['carbon_neutral'] else 'No'}
- Ethical Sourcing: {'Yes' if specifications['ethical_sourcing'] else 'No'}
- Sustainable Packaging: {'Yes' if specifications['packaging_sustainable'] else 'No'}
"""
    return formatted_text

def generate_product_descriptions(specifications, model_id, params, additional_model_fields):
    """Generate product descriptions for different platforms"""
    
    platform_prompts = {
        "web_news_ads": {
            "title": "🌐 Web News Ads",
            "char_limit": 500,
            "system_prompt": """You are a professional marketing copywriter specializing in web news advertisements. 
            Create compelling, informative product descriptions that would work well in online news platforms and banner ads. 
            Focus on key benefits, technical specifications, and what makes the product unique. 
            The tone should be professional yet engaging, approximately 150-200 words.""",
            "user_prompt": "Create a web news advertisement description for this shoe product based on the following specifications:"
        },
        
        "social_media": {
            "title": "📱 Social Media",
            "char_limit": 500,
            "system_prompt": """You are a social media marketing expert. 
            Create engaging, trendy product descriptions perfect for social media platforms like Instagram and Facebook. 
            Use modern language, highlight lifestyle benefits, and make it shareable. 
            Include relevant hashtags and emojis. Keep it concise but impactful, around 100-150 words.""",
            "user_prompt": "Create a social media post description for this shoe product based on the following specifications:"
        },
        
        "twitter": {
            "title": "🐦 Twitter/X",
            "char_limit": 280,
            "system_prompt": """You are a Twitter marketing specialist. 
            Create concise, punchy product descriptions that fit Twitter's format. 
            Focus on the most compelling features and benefits in a tweet-sized format. 
            Use hashtags strategically and make every word count. 
            Maximum 280 characters, make it engaging and shareable.""",
            "user_prompt": "Create a Twitter post description for this shoe product based on the following specifications:"
        },
        
        "amazon_listing": {
            "title": "🛒 Amazon Product Listing",
            "char_limit": 2000,
            "system_prompt": """You are an Amazon product listing specialist. 
            Create SEO-optimized product descriptions for Amazon marketplace. 
            Include key features, benefits, specifications, and use cases. 
            Use bullet points for readability. Focus on search keywords and conversion. 
            Professional tone, approximately 200-300 words.""",
            "user_prompt": "Create an Amazon product listing description for this shoe product based on the following specifications:"
        },
        
        "email_marketing": {
            "title": "📧 Email Marketing",
            "char_limit": 600,
            "system_prompt": """You are an email marketing copywriter. 
            Create compelling email content that drives clicks and conversions. 
            Start with an attention-grabbing hook, highlight key benefits, and include a clear call-to-action. 
            Conversational yet persuasive tone, approximately 150-200 words.""",
            "user_prompt": "Create an email marketing description for this shoe product based on the following specifications:"
        },
        
        "instagram": {
            "title": "📸 Instagram Caption",
            "char_limit": 2200,
            "system_prompt": """You are an Instagram content creator. 
            Create visually-focused captions that complement product imagery. 
            Use storytelling, lifestyle appeal, and relevant hashtags. 
            Engaging and authentic tone with emojis, approximately 100-150 words.""",
            "user_prompt": "Create an Instagram caption for this shoe product based on the following specifications:"
        },
        
        "linkedin": {
            "title": "💼 LinkedIn Post",
            "char_limit": 3000,
            "system_prompt": """You are a LinkedIn B2B content strategist. 
            Create professional product descriptions suitable for business audiences. 
            Focus on innovation, quality, sustainability, and business value. 
            Professional yet engaging tone, approximately 150-200 words.""",
            "user_prompt": "Create a LinkedIn post description for this shoe product based on the following specifications:"
        },
        
        "video_script": {
            "title": "🎥 Video Script (30s)",
            "char_limit": 300,
            "system_prompt": """You are a video ad scriptwriter. 
            Create a 30-second video ad script (approximately 75-90 words when spoken). 
            Include visual cues, key product highlights, and a strong call-to-action. 
            Conversational and energetic tone.""",
            "user_prompt": "Create a 30-second video ad script for this shoe product based on the following specifications:"
        }
    }
    
    formatted_specs = format_specifications_for_llm(specifications)
    descriptions = {}
    
    try:
        bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
        
        total_platforms = len(platform_prompts)
        progress_bar = st.progress(0, text="Starting generation...")
        
        for idx, (platform, prompt_info) in enumerate(platform_prompts.items()):
            progress = (idx + 1) / total_platforms
            progress_bar.progress(progress, text=f"Generating {prompt_info['title']}... ({idx + 1}/{total_platforms})")
            
            system_prompts_list = [{"text": prompt_info["system_prompt"]}]
            user_message = f"{prompt_info['user_prompt']}\n\n{formatted_specs}"
            
            message = {
                "role": "user",
                "content": [{"text": user_message}]
            }
            messages = [message]
            
            response = text_conversation(
                bedrock_client, 
                model_id, 
                system_prompts_list, 
                messages, 
                additional_model_fields,
                **params
            )
            
            if response:
                output_message = response['output']['message']
                description_text = ""
                for content in output_message['content']:
                    description_text += content['text']
                
                descriptions[platform] = {
                    'title': prompt_info['title'],
                    'content': description_text,
                    'char_limit': prompt_info['char_limit'],
                    'tokens': response['usage']
                }
        
        progress_bar.progress(1.0, text="✅ All descriptions generated!")
        time.sleep(0.5)
        progress_bar.empty()
                    
    except Exception as e:
        st.error(f"Error generating descriptions: {str(e)}")
        logger.error(f"Error in generate_product_descriptions: {str(e)}")
    
    return descriptions

def display_generated_descriptions(descriptions):
    """Display the generated product descriptions with validation and export"""
    st.markdown("## 📝 Generated Product Descriptions")
    
    # Export functionality
    st.markdown("### 📥 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export
        import json
        json_data = {}
        for platform, data in descriptions.items():
            json_data[platform] = {
                'title': data['title'],
                'content': data['content'],
                'char_count': len(data['content']),
                'char_limit': data['char_limit'],
                'tokens': data['tokens']
            }
        
        st.download_button(
            "📄 Download JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"descriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # CSV export
        import csv
        from io import StringIO
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Platform", "Description", "Character Count", "Character Limit", "Status", "Input Tokens", "Output Tokens"])
        
        for platform, data in descriptions.items():
            char_count = len(data['content'])
            char_limit = data['char_limit']
            status = "✅ Good" if char_count <= char_limit else "❌ Over Limit"
            
            writer.writerow([
                data['title'],
                data['content'],
                char_count,
                char_limit,
                status,
                data['tokens']['inputTokens'],
                data['tokens']['outputTokens']
            ])
        
        st.download_button(
            "📊 Download CSV",
            data=output.getvalue(),
            file_name=f"descriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Text export (all descriptions)
        text_output = f"Product Descriptions - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text_output += "=" * 80 + "\n\n"
        
        for platform, data in descriptions.items():
            text_output += f"{data['title']}\n"
            text_output += "-" * 40 + "\n"
            text_output += f"{data['content']}\n"
            text_output += f"\nCharacters: {len(data['content'])}/{data['char_limit']}\n"
            text_output += "=" * 80 + "\n\n"
        
        st.download_button(
            "📝 Download TXT",
            data=text_output,
            file_name=f"descriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Create tabs for each platform
    if descriptions:
        platform_tabs = st.tabs([desc['title'] for desc in descriptions.values()])
        
        for i, (platform, desc_data) in enumerate(descriptions.items()):
            with platform_tabs[i]:
                st.markdown(f"### {desc_data['title']}")
                
                # Character count validation
                char_count = len(desc_data['content'])
                char_limit = desc_data['char_limit']
                
                # Determine status and color
                if char_count <= char_limit * 0.8:
                    color = "#10B981"  # Green
                    status = "✅ Good"
                    border_color = "#10B981"
                elif char_count <= char_limit:
                    color = "#F59E0B"  # Orange
                    status = "⚠️ Near Limit"
                    border_color = "#F59E0B"
                else:
                    color = "#EF4444"  # Red
                    status = "❌ Over Limit"
                    border_color = "#EF4444"
                
                # Display the description with validation
                st.markdown("#### Generated Description:")
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid {border_color};'>
                    {desc_data['content']}
                </div>
                <div style='text-align: right; margin-top: 8px; color: {color}; font-weight: 600;'>
                    {status}: {char_count}/{char_limit} characters
                </div>
                """, unsafe_allow_html=True)
                
                # Display token usage
                st.markdown("#### Generation Details:")
                col1, col2, col3 = st.columns(3)
                col1.metric("Input Tokens", desc_data['tokens']['inputTokens'])
                col2.metric("Output Tokens", desc_data['tokens']['outputTokens'])
                col3.metric("Total Tokens", desc_data['tokens']['totalTokens'])
                
                # Copy button functionality
                st.markdown("#### Actions:")
                col_copy1, col_copy2 = st.columns(2)
                
                with col_copy1:
                    if st.button(f"📋 Copy to Clipboard", key=f"copy_{platform}"):
                        st.code(desc_data['content'], language=None)
                        st.success("✅ Description displayed above - copy from the code block!")
                
                with col_copy2:
                    # Individual download
                    st.download_button(
                        f"💾 Download",
                        data=desc_data['content'],
                        file_name=f"{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key=f"download_{platform}"
                    )

# ------- MAIN APP -------

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    initialize_session()
    
    with st.sidebar:
        common.render_sidebar()
        

        
        # About section
        with st.expander("About this App", expanded=False):
            st.markdown("""
            This application helps you generate compelling product descriptions for shoe products across different marketing platforms:
            
            **Features:**
            * Comprehensive product specification form
            * AI-powered description generation
            * Multiple platform formats (Web, Social Media, Twitter)
            * Real-time customization and preview
            
            **How to use:**
            1. Fill out the product specifications
            2. Review the summary
            3. Generate descriptions for all platforms
            4. Copy and use in your marketing campaigns
            """)
    
    # Header with modern AWS branding
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #232F3E; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" style="display: inline-block; vertical-align: middle; margin-right: 12px;">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="url(#gradient1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                <path d="M2 17L12 22L22 17" stroke="url(#gradient1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="url(#gradient1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <defs>
                    <linearGradient id="gradient1" x1="2" y1="2" x2="22" y2="22">
                        <stop offset="0%" style="stop-color:#FF9900;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#146EB4;stop-opacity:1" />
                    </linearGradient>
                </defs>
            </svg>
            👟 Shoe Product Description Generator
        </h1>
        <p style="color: #6B7280; font-size: 1.1rem; margin: 0;">
            AI-Powered Content Creation for Multiple Marketing Platforms
            <br/>
            <span style="font-size: 0.875rem; color: #9CA3AF; margin-top: 0.5rem; display: inline-block;">
                Powered by Amazon Bedrock • 8 Platform Formats • Real-time Generation
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    
    col1,col2 = st.columns([7,3])
    with col2:
        # Model selection panel
        with st.container(border=True):
            model_id, params, additional_model_fields = model_selection_panel()
    
    with col1:
    
        # Main content in steps
        step1_tab, step2_tab, step3_tab = st.tabs(["📝 Step 1: Product Specs", "📋 Step 2: Review Summary", "🚀 Step 3: Generate Descriptions"])
        
        with step1_tab:
            specifications = create_product_specifications_form()
            
            if st.button("Save Specifications", type="primary", key="save_specs"):
                st.session_state.product_specifications = specifications
                st.success("✅ Product specifications saved! Proceed to Step 2 to review.")
        
        with step2_tab:
            if st.session_state.product_specifications:
                display_specifications_summary(st.session_state.product_specifications)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Modify Specifications", key="modify_specs"):
                        st.info("Go back to Step 1 to modify your specifications.")
                
                with col2:
                    if st.button("✅ Confirm & Generate Descriptions", type="primary", key="confirm_specs"):
                        st.success("Specifications confirmed! Proceed to Step 3 to generate descriptions.")
            else:
                st.warning("⚠️ Please complete Step 1 first to configure your product specifications.")
        
        with step3_tab:
            if st.session_state.product_specifications:
                st.markdown("### Ready to Generate Descriptions")
                st.info("Click the button below to generate product descriptions for all marketing platforms.")
                
                if st.button("🚀 Generate All Descriptions", type="primary", key="generate_all", help="This will generate descriptions for Web News Ads, Social Media, and Twitter"):
                    with st.spinner("Generating descriptions for all platforms..."):
                        descriptions = generate_product_descriptions(
                            st.session_state.product_specifications,
                            model_id,
                            params,
                            additional_model_fields
                        )
                        
                        if descriptions:
                            st.session_state.generated_descriptions = descriptions
                            st.success("🎉 All descriptions generated successfully!")
                
                # Display generated descriptions if they exist
                if st.session_state.generated_descriptions:
                    display_generated_descriptions(st.session_state.generated_descriptions)
                    
                    # Option to regenerate
                    st.markdown("---")
                    if st.button("🔄 Regenerate All Descriptions", key="regenerate_all"):
                        with st.spinner("Regenerating descriptions..."):
                            descriptions = generate_product_descriptions(
                                st.session_state.product_specifications,
                                model_id,
                                params,
                                additional_model_fields
                            )
                            if descriptions:
                                st.session_state.generated_descriptions = descriptions
                                st.success("🎉 Descriptions regenerated successfully!")
                                st.rerun()
            else:
                st.warning("⚠️ Please complete Steps 1 and 2 first before generating descriptions.")
        
        # Footer with modern styling
        st.markdown("""
        <div style="margin-top: 4rem; padding: 1.5rem; background: linear-gradient(135deg, #232F3E 0%, #16191F 100%); 
                    border-radius: 10px; text-align: center; position: relative;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; 
                        background: linear-gradient(90deg, #FF9900 0%, #146EB4 100%);"></div>
            <p style="color: #D1D5DB; font-size: 0.875rem; margin: 0;">
                © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Main execution flow
if __name__ == "__main__":
    if 'localhost' in st.context.headers.get("host", ""):
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()

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
    
    # Models that support Top K parameter
    MODELS_WITH_TOP_K = [
        "mistral.mistral-small-2402-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-large-2402-v1:0"
    ]
    
    # Create selectbox for provider first
    provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()), key="side_provider")
    
    # Then create selectbox for models from that provider
    model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider], key="side_model")
    
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
        
    top_p = st.slider(
        "Top P", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.9, 
        step=0.05,
        key="side_top_p",
        help="Controls diversity via nucleus sampling"
    )

    # Add Top K parameter for models that support it
    top_k = None
    if model_id in MODELS_WITH_TOP_K:
        top_k = st.slider("Top K", min_value=0, max_value=500, value=200, step=10,
                        help="Limits vocabulary to K most likely tokens")
        
    max_tokens = st.number_input(
        "Max Tokens", 
        min_value=100, 
        max_value=4096, 
        value=2048, 
        step=50,
        key="side_max_tokens",
        help="Maximum number of tokens in the response"
    )
    
    params = {
        "temperature": temperature,
        "topP": top_p,
        "maxTokens": max_tokens,
        "stopSequences": []
    }
    
    # Add topK to params only if the model supports it and it's set
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
            "system_prompt": """You are a professional marketing copywriter specializing in web news advertisements. 
            Create compelling, informative product descriptions that would work well in online news platforms and banner ads. 
            Focus on key benefits, technical specifications, and what makes the product unique. 
            The tone should be professional yet engaging, approximately 150-200 words.""",
            "user_prompt": "Create a web news advertisement description for this shoe product based on the following specifications:"
        },
        
        "social_media": {
            "title": "📱 Social Media",
            "system_prompt": """You are a social media marketing expert. 
            Create engaging, trendy product descriptions perfect for social media platforms like Instagram and Facebook. 
            Use modern language, highlight lifestyle benefits, and make it shareable. 
            Include relevant hashtags and emojis. Keep it concise but impactful, around 100-150 words.""",
            "user_prompt": "Create a social media post description for this shoe product based on the following specifications:"
        },
        
        "twitter": {
            "title": "🐦 Twitter",
            "system_prompt": """You are a Twitter marketing specialist. 
            Create concise, punchy product descriptions that fit Twitter's format. 
            Focus on the most compelling features and benefits in a tweet-sized format. 
            Use hashtags strategically and make every word count. 
            Maximum 280 characters, make it engaging and shareable.""",
            "user_prompt": "Create a Twitter post description for this shoe product based on the following specifications:"
        }
    }
    
    formatted_specs = format_specifications_for_llm(specifications)
    descriptions = {}
    
    try:
        bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
        
        for platform, prompt_info in platform_prompts.items():
            with st.status(f"Generating {prompt_info['title']} description...", expanded=True) as status:
                
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
                        'tokens': response['usage']
                    }
                    
                    status.update(label=f"{prompt_info['title']} generated!", state="complete")
                else:
                    status.update(label=f"Error generating {prompt_info['title']}", state="error")
                    
    except Exception as e:
        st.error(f"Error generating descriptions: {str(e)}")
        logger.error(f"Error in generate_product_descriptions: {str(e)}")
    
    return descriptions

def display_generated_descriptions(descriptions):
    """Display the generated product descriptions"""
    st.markdown("## 📝 Generated Product Descriptions")
    
    # Create tabs for each platform
    if descriptions:
        platform_tabs = st.tabs([desc['title'] for desc in descriptions.values()])
        
        for i, (platform, desc_data) in enumerate(descriptions.items()):
            with platform_tabs[i]:
                st.markdown(f"### {desc_data['title']}")
                
                # Display the description in a nice format
                st.markdown("#### Generated Description:")
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4;'>{desc_data['content']}</div>", 
                           unsafe_allow_html=True)
                
                # Display token usage
                st.markdown("#### Generation Details:")
                col1, col2, col3 = st.columns(3)
                col1.metric("Input Tokens", desc_data['tokens']['inputTokens'])
                col2.metric("Output Tokens", desc_data['tokens']['outputTokens'])
                col3.metric("Total Tokens", desc_data['tokens']['totalTokens'])
                
                # Copy button functionality
                st.markdown("#### Actions:")
                if st.button(f"📋 Copy {desc_data['title']} Description", key=f"copy_{platform}"):
                    st.code(desc_data['content'], language=None)
                    st.success("Description copied to code block above!")

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
    
    # Header
    st.markdown("""
    <div class="element-animation">
        <h1>👟 Shoe Product Description Generator</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""<div class="info-box">
    Create compelling product descriptions for your new shoe launch across multiple marketing platforms. 
    Configure your product specifications and let AI generate targeted content for web ads, social media, and Twitter.
    </div>""", unsafe_allow_html=True)
    
    
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
                                st.experimental_rerun()
            else:
                st.warning("⚠️ Please complete Steps 1 and 2 first before generating descriptions.")
        
        # Footer
        st.markdown("""
        <footer>
            © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
        </footer>
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


import streamlit as st
import base64
import io
import json
import logging
import boto3
from PIL import Image
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import datetime
import uuid
import utils.common as common
import utils.authenticate as authenticate
import utils.styles as styles

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Custom exception for image generation errors
class ImageError(Exception):
    """Custom exception for errors returned by Amazon Nova Canvas"""
    def __init__(self, message):
        self.message = message

def load_css():
    """Load custom CSS for modern UI/UX"""
    
    styles.load_css()
    
    # st.markdown("""
    # <style>
    # /* Modern Color Scheme */
    # :root {
    #     --aws-orange: #FF9900;
    #     --aws-blue: #232F3E;
    #     --aws-light-blue: #4B9CD3;
    #     --aws-dark-blue: #16191F;
    #     --aws-gray: #EAEDED;
    #     --aws-white: #FFFFFF;
    #     --success-green: #0FA958;
    #     --error-red: #D13212;
    # }
    
    # /* Main Header */
    # .main-header {
    #     background: linear-gradient(135deg, var(--aws-blue) 0%, var(--aws-light-blue) 100%);
    #     padding: 2.5rem;
    #     border-radius: 15px;
    #     margin-bottom: 2rem;
    #     text-align: center;
    #     box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    # }
    
    # .main-header h1 {
    #     color: white !important;
    #     font-size: 2.8rem;
    #     font-weight: 700;
    #     margin-bottom: 0.5rem;
    #     text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    # }
    
    # .main-header p {
    #     color: var(--aws-gray);
    #     font-size: 1.2rem;
    #     margin: 0;
    #     opacity: 0.95;
    # }
    
    # /* Card Styling */
    # .info-card {
    #     background: white;
    #     border-radius: 12px;
    #     padding: 1.5rem;
    #     margin-bottom: 1.5rem;
    #     box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    #     border: 1px solid #e1e4e8;
    #     transition: transform 0.3s ease;
    # }
    
    # .info-card:hover {
    #     transform: translateY(-2px);
    #     box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    # }
    
    # .feature-card {
    #     background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    #     border-left: 4px solid var(--aws-orange);
    #     padding: 1.2rem;
    #     margin: 1rem 0;
    #     border-radius: 8px;
    # }
    
    # /* Tabs Styling */
    # .stTabs [data-baseweb="tab-list"] {
    #     gap: 8px;
    #     background-color: #f8f9fa;
    #     padding: 0.5rem;
    #     border-radius: 10px;
    # }
    
    # .stTabs [data-baseweb="tab"] {
    #     height: 50px;
    #     padding: 0 24px;
    #     background-color: white;
    #     border-radius: 8px;
    #     border: 2px solid #e1e4e8;
    #     font-weight: 600;
    #     transition: all 0.3s ease;
    # }
    
    # .stTabs [data-baseweb="tab"]:hover {
    #     background-color: var(--aws-orange);
    #     color: white;
    #     border-color: var(--aws-orange);
    # }
    
    # .stTabs [aria-selected="true"] {
    #     background: linear-gradient(90deg, var(--aws-orange) 0%, #ff7700 100%);
    #     color: white !important;
    #     border-color: var(--aws-orange) !important;
    # }
    
    # /* Form Elements */
    # .stSelectbox label, .stTextArea label, .stNumberInput label {
    #     font-weight: 600;
    #     color: var(--aws-blue);
    #     margin-bottom: 0.5rem;
    # }
    
    # .stTextArea textarea {
    #     border-radius: 8px;
    #     border: 2px solid #e1e4e8;
    #     transition: border-color 0.3s ease;
    # }
    
    # .stTextArea textarea:focus {
    #     border-color: var(--aws-orange);
    #     box-shadow: 0 0 0 3px rgba(255,153,0,0.1);
    # }
    
    # /* Buttons */
    # .stButton > button {
    #     background: linear-gradient(90deg, var(--aws-orange) 0%, #ff7700 100%);
    #     color: white;
    #     border: none;
    #     border-radius: 8px;
    #     padding: 0.75rem 2rem;
    #     font-weight: 600;
    #     font-size: 1rem;
    #     transition: all 0.3s ease;
    #     box-shadow: 0 4px 6px rgba(255,153,0,0.2);
    # }
    
    # .stButton > button:hover {
    #     transform: translateY(-2px);
    #     box-shadow: 0 6px 12px rgba(255,153,0,0.3);
    # }
    
    # /* Success/Error Messages */
    # .success-message {
    #     background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    #     border-left: 4px solid var(--success-green);
    #     padding: 1rem;
    #     border-radius: 8px;
    #     margin: 1rem 0;
    # }
    
    # .error-message {
    #     background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    #     border-left: 4px solid var(--error-red);
    #     padding: 1rem;
    #     border-radius: 8px;
    #     margin: 1rem 0;
    # }
    
    # /* Image Display */
    # .image-container {
    #     background: white;
    #     border-radius: 12px;
    #     padding: 1.5rem;
    #     box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    #     margin: 1.5rem 0;
    # }
    
    # /* Footer */
    # .footer {
    #     background-color: var(--aws-blue);
    #     color: white;
    #     text-align: center;
    #     padding: 1.5rem;
    #     margin-top: 3rem;
    #     border-radius: 10px;
    #     font-size: 0.9rem;
    # }
    
    # /* Responsive Design */
    # @media (max-width: 768px) {
    #     .main-header h1 {
    #         font-size: 2rem;
    #     }
        
    #     .stButton > button {
    #         width: 100%;
    #     }
        
    #     .info-card {
    #         padding: 1rem;
    #     }
    # }
    
    # /* Loading Animation */
    # .stSpinner > div {
    #     border-color: var(--aws-orange) !important;
    # }
    # </style>
    # """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    
    common.initialize_session_state()
    
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    
    if "generation_count" not in st.session_state:
        st.session_state.generation_count = 0

def generate_image_from_nova(model_id, body):
    """
    Generate an image using Amazon Nova Canvas model.
    
    Args:
        model_id (str): The model ID to use
        body (str): The request body containing generation parameters
    
    Returns:
        bytes: The generated image data
    """
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            config=Config(read_timeout=300),
            region_name='us-east-1'
        )
        
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        
        if response_body.get("error"):
            raise ImageError(f"Image generation error: {response_body.get('error')}")
        
        base64_image = response_body.get("images")[0]
        image_bytes = base64.b64decode(base64_image.encode('ascii'))
        
        return image_bytes
        
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(f"AWS Client Error: {error_message}")
        raise ImageError(f"AWS Error: {error_message}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise ImageError(f"Unexpected error: {str(e)}")

def build_product_prompt(product_type, material, color, style, environment, lighting, additional_details):
    """
    Build a comprehensive prompt for product image generation.
    
    Args:
        product_type: Type of product
        material: Product material
        color: Product color
        style: Visual style
        environment: Background environment
        lighting: Lighting setup
        additional_details: Additional specifications
    
    Returns:
        str: Constructed prompt
    """
    prompt_parts = []
    
    # Core product description
    if product_type and color and material:
        prompt_parts.append(f"Professional product photography of a {color} {product_type} made of {material}")
    elif product_type:
        prompt_parts.append(f"Professional product photography of a {product_type}")
    
    # Style
    if style:
        style_descriptions = {
            "Minimalist": "clean minimalist style, simple composition",
            "Luxury": "luxurious premium presentation, elegant styling",
            "Industrial": "industrial design aesthetic, modern technical look",
            "Natural": "natural organic presentation, eco-friendly appearance",
            "Tech": "high-tech futuristic style, modern technology aesthetic",
            "Vintage": "vintage retro styling, classic timeless appearance"
        }
        prompt_parts.append(style_descriptions.get(style, style))
    
    # Environment/Background
    if environment:
        env_descriptions = {
            "White Background": "pure white studio background, clean isolated product",
            "Studio Setup": "professional studio setup with gradient background",
            "Lifestyle Scene": "lifestyle setting, in-context usage scenario",
            "Nature": "natural outdoor environment, organic setting",
            "Office": "modern office environment, professional workspace",
            "Home": "cozy home interior setting, domestic environment"
        }
        prompt_parts.append(env_descriptions.get(environment, environment))
    
    # Lighting
    if lighting:
        lighting_descriptions = {
            "Soft Box": "soft diffused studio lighting, even illumination",
            "Natural Light": "natural window lighting, soft shadows",
            "Dramatic": "dramatic lighting with strong shadows and highlights",
            "Rim Light": "rim lighting highlighting product edges",
            "Three-Point": "professional three-point lighting setup"
        }
        prompt_parts.append(lighting_descriptions.get(lighting, lighting))
    
    # Additional details
    if additional_details:
        prompt_parts.append(additional_details)
    
    # Quality modifiers
    prompt_parts.append("high resolution, professional photography, commercial quality, sharp focus, detailed")
    
    return ", ".join(prompt_parts)

def render_sidebar():
    """Render the sidebar with app information"""
    with st.sidebar:
        
        common.render_sidebar()
        
        with st.expander("About this App", expanded=False):
            st.markdown("""
            **Topics Covered:**
            
            📸 **Product Photography**
            - Professional product visualization
            - Commercial-grade image generation
            
            🎨 **Customization Options**
            - Multiple artistic styles
            - Various backgrounds & environments
            - Professional lighting setups
            
            🔧 **Technical Features**
            - AWS Nova Canvas integration
            - Multiple image dimensions
            - Real-time generation
            
            💡 **Use Cases**
            - E-commerce product images
            - Marketing materials
            - Design prototypes
            - Product catalogs
            
            🚀 **Powered By**
            - Amazon Bedrock
            - Nova Canvas AI Model
            - Streamlit Framework
            """)
        
        st.markdown("---")
        


def render_generation_tab():
    """Render the image generation tab"""
    st.markdown("""
    <div class="info-card">
        <h3>🖼️ Create Professional Product Images</h3>
        <p>Generate high-quality product images with customizable specifications. Perfect for e-commerce, marketing, and design projects.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for the form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📦 Product Specifications")
        
        product_type = st.text_input(
            "Product Type*",
            placeholder="e.g., Watch, Smartphone, Sneakers, Perfume Bottle",
            help="Specify the type of product you want to generate"
        )
        
        material = st.selectbox(
            "Material",
            ["", "Metal", "Plastic", "Wood", "Glass", "Fabric", "Leather", "Ceramic", "Carbon Fiber", "Mixed Materials"],
            help="Choose the primary material of your product"
        )
        
        color = st.text_input(
            "Color Scheme",
            placeholder="e.g., Matte black, Rose gold, Navy blue",
            help="Describe the color or color combination"
        )
        
        style = st.selectbox(
            "Visual Style",
            ["Minimalist", "Luxury", "Industrial", "Natural", "Tech", "Vintage"],
            help="Select the overall aesthetic style"
        )
    
    with col2:
        st.markdown("#### 🎬 Scene Settings")
        
        environment = st.selectbox(
            "Background/Environment",
            ["White Background", "Studio Setup", "Lifestyle Scene", "Nature", "Office", "Home"],
            help="Choose where your product will be displayed"
        )
        
        lighting = st.selectbox(
            "Lighting Setup",
            ["Soft Box", "Natural Light", "Dramatic", "Rim Light", "Three-Point"],
            help="Select the lighting style for your product"
        )
        
        # Image dimensions
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            width = st.selectbox(
                "Width (px)",
                [512, 768, 1024],
                index=1,
                help="Image width in pixels"
            )
        
        with col2_2:
            height = st.selectbox(
                "Height (px)",
                [512, 768, 1024],
                index=1,
                help="Image height in pixels"
            )
        
        nova_style = st.selectbox(
            "Art Style",
            ["PHOTOREALISM", "SOFT_DIGITAL_PAINTING", "DESIGN_SKETCH", "3D_ANIMATED_FAMILY_FILM"],
            help="Choose the artistic rendering style"
        )
    
    # Additional details
    st.markdown("#### ✨ Additional Details")
    additional_details = st.text_area(
        "Extra Specifications (Optional)",
        placeholder="Add any specific details about angles, features, or composition...",
        height=100,
        help="Provide additional details to refine your image"
    )
    
    # Generate button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_button = st.button(
            "🚀 Generate Product Image",
            type="primary",
            use_container_width=True
        )
    
    # Generation logic
    if generate_button:
        if not product_type:
            st.error("⚠️ Please specify a product type to generate an image.")
        else:
            try:
                with st.spinner("🎨 Creating your product image... This may take a moment."):
                    # Build the prompt
                    prompt = build_product_prompt(
                        product_type=product_type,
                        material=material,
                        color=color,
                        style=style,
                        environment=environment,
                        lighting=lighting,
                        additional_details=additional_details
                    )
                    
                    # Prepare request body
                    body = json.dumps({
                        "taskType": "TEXT_IMAGE",
                        "textToImageParams": {
                            "text": prompt
                        },
                        "imageGenerationConfig": {
                            "numberOfImages": 1,
                            "height": height,
                            "width": width,
                            "cfgScale": 8.0,
                            "seed": 0
                        }
                    })
                    
                    # Generate image
                    model_id = 'amazon.nova-canvas-v1:0'
                    image_bytes = generate_image_from_nova(model_id, body)
                    
                    # Display the generated image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(image, caption="Generated Product Image")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Success message
                    st.markdown("""
                    <div class="success-message">
                        ✅ <strong>Image generated successfully!</strong><br>
                        Your product image has been created with the specified parameters.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save to session
                    st.session_state.generated_images.append({
                        "image": image,
                        "prompt": prompt,
                        "timestamp": datetime.now(),
                        "product_type": product_type,
                        "style": style
                    })
                    st.session_state.generation_count += 1
                    
                    # Download button
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    btn = st.download_button(
                        label="📥 Download Image",
                        data=buf.getvalue(),
                        file_name=f"product_{product_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                    
            except ImageError as e:
                st.markdown(f"""
                <div class="error-message">
                    ❌ <strong>Generation Error:</strong><br>
                    {e.message}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="error-message">
                    ❌ <strong>Unexpected Error:</strong><br>
                    {str(e)}
                </div>
                """, unsafe_allow_html=True)

def render_gallery_tab():
    """Render the gallery tab with generated images"""
    st.markdown("""
    <div class="info-card">
        <h3>🖼️ Image Gallery</h3>
        <p>View all images generated during this session.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.generated_images:
        # Sort images by timestamp (most recent first)
        images = sorted(st.session_state.generated_images, key=lambda x: x['timestamp'], reverse=True)
        
        # Display images in a grid
        cols = st.columns(3)
        for idx, img_data in enumerate(images):
            with cols[idx % 3]:
                st.image(img_data['image'])
                st.caption(f"{img_data['product_type']} - {img_data['style']}")
                st.text(f"Generated: {img_data['timestamp'].strftime('%H:%M:%S')}")
                
                # Download button for each image
                buf = io.BytesIO()
                img_data['image'].save(buf, format="PNG")
                st.download_button(
                    label="📥 Download",
                    data=buf.getvalue(),
                    file_name=f"gallery_{img_data['product_type'].replace(' ', '_')}_{idx}.png",
                    mime="image/png",
                    key=f"download_gallery_{idx}"
                )
    else:
        st.info("📸 No images generated yet. Go to the 'Generate' tab to create your first product image!")

def render_examples_tab():
    """Render the examples tab with predefined templates"""
    st.markdown("""
    <div class="info-card">
        <h3>💡 Product Image Examples</h3>
        <p>Explore different product types and styles to inspire your creations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    examples = [
        {
            "title": "🕐 Luxury Watch",
            "product": "Luxury wristwatch",
            "material": "Stainless steel and leather",
            "color": "Silver with brown leather strap",
            "style": "Luxury",
            "environment": "Studio Setup",
            "lighting": "Rim Light",
            "details": "Swiss design, chronograph, reflective surface"
        },
        {
            "title": "👟 Athletic Sneakers",
            "product": "Running shoes",
            "material": "Mesh and rubber",
            "color": "White with neon accents",
            "style": "Tech",
            "environment": "Lifestyle Scene",
            "lighting": "Natural Light",
            "details": "Dynamic angle, showing sole technology"
        },
        {
            "title": "💼 Leather Bag",
            "product": "Professional briefcase",
            "material": "Premium leather",
            "color": "Deep brown",
            "style": "Luxury",
            "environment": "Office",
            "lighting": "Soft Box",
            "details": "Gold hardware, textured leather, compartments visible"
        },
        {
            "title": "🎧 Wireless Headphones",
            "product": "Over-ear headphones",
            "material": "Plastic and metal",
            "color": "Matte black",
            "style": "Minimalist",
            "environment": "White Background",
            "lighting": "Three-Point",
            "details": "Noise cancellation, premium audio, foldable design"
        },
        {
            "title": "🌸 Perfume Bottle",
            "product": "Perfume bottle",
            "material": "Glass",
            "color": "Crystal clear with gold cap",
            "style": "Luxury",
            "environment": "Studio Setup",
            "lighting": "Dramatic",
            "details": "Elegant shape, light refraction, premium packaging"
        },
        {
            "title": "📱 Smartphone",
            "product": "Flagship smartphone",
            "material": "Glass and aluminum",
            "color": "Midnight blue",
            "style": "Tech",
            "environment": "White Background",
            "lighting": "Soft Box",
            "details": "Multiple cameras, edge-to-edge display, sleek profile"
        }
    ]
    
    # Display examples in a grid
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            with st.expander(example["title"], expanded=False):
                st.markdown(f"""
                <div class="feature-card">
                <strong>Product:</strong> {example['product']}<br>
                <strong>Material:</strong> {example['material']}<br>
                <strong>Color:</strong> {example['color']}<br>
                <strong>Style:</strong> {example['style']}<br>
                <strong>Environment:</strong> {example['environment']}<br>
                <strong>Lighting:</strong> {example['lighting']}<br>
                <strong>Details:</strong> {example['details']}
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="AWS Nova Canvas - Product Image Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load custom CSS
    load_css()
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 AWS Nova Canvas Product Image Generator</h1>
        <p class="info-box">Transform product specifications into stunning professional images with AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["🚀 Generate", "🖼️ Gallery", "💡 Examples"])
    
    with tab1:
        render_generation_tab()
    
    with tab2:
        render_gallery_tab()
    
    with tab3:
        render_examples_tab()
    
    # Footer
    st.markdown("""
    <div class="footer">
        © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

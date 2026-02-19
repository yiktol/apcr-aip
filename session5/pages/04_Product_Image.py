
import streamlit as st
import base64
import io
import json
import logging
import boto3
import time
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
        


def model_selection_panel():
    """Model selection and parameters in the side panel"""
    st.markdown("<h4>Model Selection</h4>", unsafe_allow_html=True)
    
    # Nova Canvas models
    model_options = {
        "Amazon Nova Canvas": "amazon.nova-canvas-v1:0"
    }
    
    model_name = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        key="side_model"
    )
    
    model_id = model_options[model_name]
    
    st.markdown("<h4>Image Configuration</h4>", unsafe_allow_html=True)
    
    width = st.selectbox(
        "Width (px)",
        [512, 768, 1024, 1280],
        index=2,
        key="side_width",
        help="Image width in pixels"
    )
    
    height = st.selectbox(
        "Height (px)",
        [512, 768, 1024, 1280],
        index=2,
        key="side_height",
        help="Image height in pixels"
    )
    
    num_images = st.slider(
        "Number of Images",
        min_value=1,
        max_value=4,
        value=1,
        key="side_num_images",
        help="Generate multiple variations at once"
    )
    
    cfg_scale = st.slider(
        "CFG Scale",
        min_value=1.0,
        max_value=15.0,
        value=8.0,
        step=0.5,
        key="side_cfg_scale",
        help="Controls how closely the image follows the prompt (higher = more strict)"
    )
    
    seed = st.number_input(
        "Seed (0 for random)",
        min_value=0,
        max_value=2147483647,
        value=0,
        key="side_seed",
        help="Use same seed for reproducible results"
    )
    
    nova_style = st.selectbox(
        "Art Style",
        ["PHOTOREALISM", "SOFT_DIGITAL_PAINTING", "DESIGN_SKETCH", "3D_ANIMATED_FAMILY_FILM"],
        key="side_nova_style",
        help="Choose the artistic rendering style"
    )
    
    # Cost estimation
    base_cost = 0.04  # per image
    resolution_multiplier = (width * height) / (1024 * 1024)
    total_cost = base_cost * resolution_multiplier * num_images
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%); 
                padding: 1rem; border-radius: 10px; margin: 1rem 0;
                border-left: 4px solid #FF9900;">
        <h4 style="margin: 0 0 0.5rem 0; color: #232F3E; font-size: 0.85rem;">💰 Cost Estimate</h4>
        <p style="margin: 0; color: #4B5563; font-size: 0.8rem;">
            Estimated: ${total_cost:.4f} for {num_images} image(s)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    return model_id, width, height, cfg_scale, seed, nova_style, num_images
    
    return model_id, width, height, cfg_scale, seed, nova_style

def render_generation_tab(model_id, width, height, cfg_scale, seed, nova_style, num_images):
    """Render the image generation tab"""
    st.markdown("""
    <div class="info-card">
        <h3>🖼️ Create Professional Product Images</h3>
        <p>Generate high-quality product images with customizable specifications. Perfect for e-commerce, marketing, and design projects.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start Templates
    st.markdown("#### 🚀 Quick Start Templates")
    templates = {
        "Custom": {},
        "Racing Shoes (Example)": {
            "product_type": "Racing Shoes",
            "material": "Fabric",
            "color": "Neon orange with black accents",
            "environment": "White Background",
            "lighting": "Soft Box",
            "style": "Tech",
            "details": "aerodynamic design, carbon fiber plate, lightweight construction, professional racing shoe, dynamic angle, high performance"
        },
        "E-commerce Product Shot": {
            "environment": "White Background",
            "lighting": "Soft Box",
            "style": "Minimalist",
            "details": "centered, professional, high resolution, clean background"
        },
        "Lifestyle Photography": {
            "environment": "Lifestyle Scene",
            "lighting": "Natural Light",
            "style": "Natural",
            "details": "in-context usage, authentic setting, lifestyle appeal"
        },
        "Premium Luxury": {
            "environment": "Studio Setup",
            "lighting": "Dramatic",
            "style": "Luxury",
            "details": "elegant, premium quality, sophisticated, high-end"
        },
        "Tech Product": {
            "environment": "White Background",
            "lighting": "Three-Point",
            "style": "Tech",
            "details": "modern, sleek, futuristic, high-tech aesthetic"
        }
    }
    
    selected_template = st.selectbox(
        "Choose a template or start from scratch",
        options=list(templates.keys()),
        help="Pre-configured settings for common use cases"
    )
    
    template_data = templates[selected_template]
    
    # Create two columns for the form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📦 Product Specifications")
        
        product_type = st.text_input(
            "Product Type*",
            value=template_data.get("product_type", ""),
            placeholder="e.g., Watch, Smartphone, Sneakers, Perfume Bottle",
            help="Specify the type of product you want to generate"
        )
        
        material = st.selectbox(
            "Material",
            ["", "Metal", "Plastic", "Wood", "Glass", "Fabric", "Leather", "Ceramic", "Carbon Fiber", "Mixed Materials"],
            index=["", "Metal", "Plastic", "Wood", "Glass", "Fabric", "Leather", "Ceramic", "Carbon Fiber", "Mixed Materials"].index(template_data.get("material", "")) if template_data.get("material") else 0,
            help="Choose the primary material of your product"
        )
        
        color = st.text_input(
            "Color Scheme",
            value=template_data.get("color", ""),
            placeholder="e.g., Matte black, Rose gold, Navy blue",
            help="Describe the color or color combination"
        )
        
        style = st.selectbox(
            "Visual Style",
            ["Minimalist", "Luxury", "Industrial", "Natural", "Tech", "Vintage"],
            index=["Minimalist", "Luxury", "Industrial", "Natural", "Tech", "Vintage"].index(template_data.get("style", "Minimalist")) if template_data.get("style") else 0,
            help="Select the overall aesthetic style"
        )
    
    with col2:
        st.markdown("#### 🎬 Scene Settings")
        
        environment = st.selectbox(
            "Background/Environment",
            ["White Background", "Studio Setup", "Lifestyle Scene", "Nature", "Office", "Home"],
            index=["White Background", "Studio Setup", "Lifestyle Scene", "Nature", "Office", "Home"].index(template_data.get("environment", "White Background")) if template_data.get("environment") else 0,
            help="Choose where your product will be displayed"
        )
        
        lighting = st.selectbox(
            "Lighting Setup",
            ["Soft Box", "Natural Light", "Dramatic", "Rim Light", "Three-Point"],
            index=["Soft Box", "Natural Light", "Dramatic", "Rim Light", "Three-Point"].index(template_data.get("lighting", "Soft Box")) if template_data.get("lighting") else 0,
            help="Select the lighting style for your product"
        )
    
    # Additional details
    st.markdown("#### ✨ Additional Details")
    col_detail1, col_detail2 = st.columns(2)
    
    with col_detail1:
        additional_details = st.text_area(
            "Extra Specifications (Optional)",
            value=template_data.get("details", ""),
            placeholder="Add any specific details about angles, features, or composition...",
            height=100,
            help="Provide additional details to refine your image"
        )
    
    with col_detail2:
        negative_prompt = st.text_area(
            "Negative Prompt (Optional)",
            placeholder="e.g., blurry, low quality, distorted, watermark, text",
            height=100,
            help="Specify what you DON'T want in the image"
        )
    
    # Generate button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_button = st.button(
            f"🚀 Generate {num_images} Image{'s' if num_images > 1 else ''}",
            type="primary",
            use_container_width=True
        )
    
    # Generation logic
    if generate_button:
        if not product_type:
            st.error("⚠️ Please specify a product type to generate an image.")
        else:
            try:
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
                
                # Progress bar
                progress_bar = st.progress(0, text="Initializing generation...")
                generated_images = []
                
                # Generate multiple images
                for i in range(num_images):
                    progress = (i + 1) / num_images
                    progress_bar.progress(progress, text=f"🎨 Generating image {i+1}/{num_images}...")
                    
                    # Use different seed for each image if seed is set
                    current_seed = (seed + i) if seed > 0 else 0
                    
                    # Prepare request body with art style (CRITICAL FIX!)
                    text_to_image_params = {
                        "text": prompt,
                        "style": nova_style  # NOW ACTUALLY USED!
                    }
                    
                    # Only include negativeText if it's provided
                    if negative_prompt and negative_prompt.strip():
                        text_to_image_params["negativeText"] = negative_prompt
                    
                    body = json.dumps({
                        "taskType": "TEXT_IMAGE",
                        "textToImageParams": text_to_image_params,
                        "imageGenerationConfig": {
                            "numberOfImages": 1,
                            "height": height,
                            "width": width,
                            "cfgScale": cfg_scale,
                            "seed": current_seed
                        }
                    })
                    
                    # Generate image
                    image_bytes = generate_image_from_nova(model_id, body)
                    image = Image.open(io.BytesIO(image_bytes))
                    generated_images.append({
                        "image": image,
                        "bytes": image_bytes,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "seed": current_seed
                    })
                
                progress_bar.progress(1.0, text="✅ Generation complete!")
                time.sleep(0.5)
                progress_bar.empty()
                
                # Display generated images
                st.markdown("### 🎨 Generated Images")
                
                if num_images == 1:
                    # Single image display
                    img_data = generated_images[0]
                    st.image(img_data["image"], caption="Generated Product Image", use_container_width=True)
                    
                    # Image metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Resolution", f"{width}x{height}")
                    col2.metric("File Size", f"{len(img_data['bytes']) / 1024:.1f} KB")
                    col3.metric("CFG Scale", cfg_scale)
                    col4.metric("Seed", img_data['seed'] if img_data['seed'] > 0 else "Random")
                    
                    # Download button
                    buf = io.BytesIO()
                    img_data["image"].save(buf, format="PNG")
                    st.download_button(
                        label="📥 Download Image",
                        data=buf.getvalue(),
                        file_name=f"product_{product_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                else:
                    # Multiple images grid display
                    cols = st.columns(min(num_images, 4))
                    for idx, img_data in enumerate(generated_images):
                        with cols[idx % 4]:
                            st.image(img_data["image"], caption=f"Variation {idx+1}", use_container_width=True)
                            
                            # Individual download
                            buf = io.BytesIO()
                            img_data["image"].save(buf, format="PNG")
                            st.download_button(
                                label=f"📥 Download",
                                data=buf.getvalue(),
                                file_name=f"product_{product_type.replace(' ', '_')}_v{idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                key=f"download_{idx}",
                                use_container_width=True
                            )
                    
                    # Bulk download as ZIP
                    st.markdown("---")
                    import zipfile
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for idx, img_data in enumerate(generated_images):
                            buf = io.BytesIO()
                            img_data["image"].save(buf, format="PNG")
                            zip_file.writestr(
                                f"product_{product_type.replace(' ', '_')}_v{idx+1}.png",
                                buf.getvalue()
                            )
                    
                    st.download_button(
                        label="📦 Download All as ZIP",
                        data=zip_buffer.getvalue(),
                        file_name=f"product_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                
                # Success message
                st.success(f"✅ Successfully generated {num_images} image{'s' if num_images > 1 else ''}!")
                
                # Save to session
                for img_data in generated_images:
                    st.session_state.generated_images.append({
                        "image": img_data["image"],
                        "prompt": img_data["prompt"],
                        "negative_prompt": img_data["negative_prompt"],
                        "timestamp": datetime.now(),
                        "product_type": product_type,
                        "style": style,
                        "nova_style": nova_style,
                        "seed": img_data["seed"],
                        "width": width,
                        "height": height
                    })
                st.session_state.generation_count += num_images
                    
            except ImageError as e:
                st.error(f"❌ Generation Error: {e.message}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                logger.error(f"Error in image generation: {str(e)}")

def render_gallery_tab():
    """Render the gallery tab with generated images"""
    st.markdown("""
    <div class="info-card">
        <h3>🖼️ Image Gallery</h3>
        <p>View and manage all images generated during this session.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.generated_images:
        # Sort images by timestamp (most recent first)
        images = sorted(st.session_state.generated_images, key=lambda x: x['timestamp'], reverse=True)
        
        # Filters and sorting
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get unique product types
            unique_products = list(set([img.get('product_type', 'Unknown') for img in images]))
            filter_product = st.selectbox("Filter by Product", ["All"] + sorted(unique_products))
        
        with col2:
            # Get unique styles
            unique_styles = list(set([img.get('style', 'Unknown') for img in images]))
            filter_style = st.selectbox("Filter by Style", ["All"] + sorted(unique_styles))
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Newest First", "Oldest First", "Product Type"])
        
        # Apply filters
        filtered_images = images
        if filter_product != "All":
            filtered_images = [img for img in filtered_images if img.get('product_type') == filter_product]
        if filter_style != "All":
            filtered_images = [img for img in filtered_images if img.get('style') == filter_style]
        
        # Apply sorting
        if sort_by == "Oldest First":
            filtered_images = sorted(filtered_images, key=lambda x: x['timestamp'])
        elif sort_by == "Product Type":
            filtered_images = sorted(filtered_images, key=lambda x: x.get('product_type', ''))
        
        # Bulk export
        if len(filtered_images) > 1:
            st.markdown("### 📥 Bulk Export")
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                # Download all as ZIP
                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for idx, img_data in enumerate(filtered_images):
                        buf = io.BytesIO()
                        img_data['image'].save(buf, format="PNG")
                        product_name = img_data.get('product_type', 'product').replace(' ', '_')
                        zip_file.writestr(
                            f"{idx+1}_{product_name}_{img_data['timestamp'].strftime('%Y%m%d_%H%M%S')}.png",
                            buf.getvalue()
                        )
                
                st.download_button(
                    label=f"📦 Download All ({len(filtered_images)} images) as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"gallery_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            with col_export2:
                # Clear gallery
                if st.button("🗑️ Clear Gallery", use_container_width=True):
                    st.session_state.generated_images = []
                    st.session_state.generation_count = 0
                    st.rerun()
        
        st.markdown(f"### 🎨 Gallery ({len(filtered_images)} images)")
        
        # Display images in a grid
        cols = st.columns(3)
        for idx, img_data in enumerate(filtered_images):
            with cols[idx % 3]:
                st.image(img_data['image'], use_container_width=True)
                
                # Image info
                st.caption(f"**{img_data.get('product_type', 'Unknown')}** - {img_data.get('style', 'Unknown')}")
                st.caption(f"🕐 {img_data['timestamp'].strftime('%H:%M:%S')}")
                
                # Show additional info in expander
                with st.expander("ℹ️ Details"):
                    st.text(f"Art Style: {img_data.get('nova_style', 'N/A')}")
                    st.text(f"Resolution: {img_data.get('width', 'N/A')}x{img_data.get('height', 'N/A')}")
                    st.text(f"Seed: {img_data.get('seed', 'Random')}")
                    if img_data.get('prompt'):
                        st.text_area("Prompt", img_data['prompt'], height=100, key=f"prompt_{idx}")
                
                # Download button
                buf = io.BytesIO()
                img_data['image'].save(buf, format="PNG")
                st.download_button(
                    label="📥 Download",
                    data=buf.getvalue(),
                    file_name=f"gallery_{img_data.get('product_type', 'image').replace(' ', '_')}_{idx}.png",
                    mime="image/png",
                    key=f"download_gallery_{idx}",
                    use_container_width=True
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
    
    # Main header with modern AWS branding
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
            🎨 AWS Nova Canvas Product Image Generator
        </h1>
        <p style="color: #6B7280; font-size: 1.1rem; margin: 0;">
            AI-Powered Professional Product Photography
            <br/>
            <span style="font-size: 0.875rem; color: #9CA3AF; margin-top: 0.5rem; display: inline-block;">
                Powered by Amazon Bedrock • Multiple Styles • High Resolution • Batch Generation
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create 70/30 split layout
    col_main, col_side = st.columns([7, 3])
    
    with col_side:
        # Model selection panel
        with st.container(border=True):
            model_id, width, height, cfg_scale, seed, nova_style, num_images = model_selection_panel()
    
    with col_main:
        # Tab navigation
        tab1, tab2, tab3 = st.tabs(["🚀 Generate", "🖼️ Gallery", "💡 Examples"])
        
        with tab1:
            render_generation_tab(model_id, width, height, cfg_scale, seed, nova_style, num_images)
        
        with tab2:
            render_gallery_tab()
        
        with tab3:
            render_examples_tab()
    
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

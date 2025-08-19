import streamlit as st
import utils.nova_image_lib as nova_lib
import uuid
from datetime import datetime
import utils.common as common
import utils.authenticate as authenticate


# Set page configuration
st.set_page_config(
    page_title="AWS Nova Canvas - AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# AWS Color scheme and custom CSS
def load_css():
    st.markdown("""
    <style>
    /* AWS Color Scheme */
    :root {
        --aws-orange: #FF9900;
        --aws-blue: #232F3E;
        --aws-light-blue: #4B9CD3;
        --aws-dark-blue: #16191F;
        --aws-gray: #EAEDED;
        --aws-white: #FFFFFF;
    }
    
    /* Main styling */
    .main-header {
        background: linear-gradient(90deg, var(--aws-blue) 0%, var(--aws-light-blue) 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: var(--aws-gray);
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Cards styling */
    .custom-card {
        background: var(--aws-white);
        border: 2px solid var(--aws-gray);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prompt-example {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid var(--aws-orange);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .negative-prompt-highlight {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, var(--aws-orange) 0%, #ff7700 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255,153,0,0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--aws-blue);
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--aws-blue);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .custom-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    common.initialize_session_state()
    
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    
    if "negative_prompt" not in st.session_state:
        st.session_state.negative_prompt = ""
    
    if "selected_style" not in st.session_state:
        st.session_state.selected_style = "PHOTOREALISM"
    
    if "image_width" not in st.session_state:
        st.session_state.image_width = 512
    
    if "image_height" not in st.session_state:
        st.session_state.image_height = 512
    
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []


def get_prompt_examples():
    """Return comprehensive prompt examples with emphasis on negative prompts"""
    return [
        {
            "title": "Portrait Photography",
            "category": "Photography",
            "prompt": "Professional headshot of a confident business woman, natural lighting, clean background, high-resolution, sharp focus",
            "negative_prompt": "blurry, low quality, distorted face, multiple faces, cartoon, painting, illustration",
            "style": "PHOTOREALISM",
            "description": "Perfect for professional portraits with realistic details"
        },
        {
            "title": "Fantasy Landscape",
            "category": "Fantasy",
            "prompt": "Magical forest with glowing mushrooms, ethereal mist, ancient trees, mystical atmosphere, vibrant colors",
            "negative_prompt": "modern buildings, cars, people, realistic, photograph, black and white",
            "style": "SOFT_DIGITAL_PAINTING",
            "description": "Creates an enchanting fantasy world"
        },
        {
            "title": "Product Design",
            "category": "Design",
            "prompt": "Minimalist smartphone design, sleek metal frame, glass surface, modern technology, studio lighting",
            "negative_prompt": "cluttered, complex, old-fashioned, broken, scratched, dirty",
            "style": "DESIGN_SKETCH",
            "description": "Great for product visualization and design concepts"
        },
        {
            "title": "Architectural Visualization",
            "category": "Architecture",
            "prompt": "Modern sustainable house, solar panels, large windows, natural materials, garden integration, contemporary design",
            "negative_prompt": "old, abandoned, damaged, dark, cluttered, unrealistic proportions",
            "style": "PHOTOREALISM",
            "description": "Professional architectural rendering"
        },
        {
            "title": "Character Illustration",
            "category": "Character",
            "prompt": "Friendly robot companion, colorful design, approachable appearance, clean lines, family-friendly",
            "negative_prompt": "scary, threatening, weapon, dark, violent, realistic human features",
            "style": "3D_ANIMATED_FAMILY_FILM",
            "description": "Perfect for character design and animation"
        },
        {
            "title": "Food Photography",
            "category": "Food",
            "prompt": "Gourmet pasta dish, fresh ingredients, restaurant presentation, warm lighting, appetizing colors",
            "negative_prompt": "unappetizing, messy, dark, low quality, artificial, processed",
            "style": "PHOTOREALISM",
            "description": "Professional food photography style"
        },
        {
            "title": "Abstract Art",
            "category": "Art",
            "prompt": "Dynamic abstract composition, bold geometric shapes, vibrant color palette, modern art style",
            "negative_prompt": "realistic, photographic, dull colors, simple, basic, low contrast",
            "style": "MAXIMALISM",
            "description": "Bold and expressive abstract artwork"
        },
        {
            "title": "Retro Design",
            "category": "Vintage",
            "prompt": "Retro kitchen appliance, 1950s design, pastel colors, chrome details, vintage aesthetics",
            "negative_prompt": "modern, futuristic, digital, minimalist, monochrome, damaged",
            "style": "MIDCENTURY_RETRO",
            "description": "Classic mid-century modern design"
        }
    ]

def load_prompt_example(example):
    """Load a prompt example into session state"""
    st.session_state.prompt = example["prompt"]
    st.session_state.negative_prompt = example["negative_prompt"]
    # Store the desired style in a temporary variable to avoid widget conflict
    st.session_state.example_style = example["style"]

def main():
    load_css()
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 AWS Nova Canvas</h1>
        <p>AI-Powered Image Generation with Advanced Prompt Engineering</p>
    </div>
    """, unsafe_allow_html=True)

    # Negative prompt explanation
    with st.expander("📚 Understanding Negative Prompts"):
        st.markdown("""
        <div class="negative-prompt-highlight">
        <strong>Negative prompts are crucial for high-quality results!</strong><br><br>
        They help you:
        <ul>
        <li>🎯 Remove unwanted elements</li>
        <li>🔧 Fix common AI artifacts</li>
        <li>🎨 Improve overall image quality</li>
        <li>⚡ Get more consistent results</li>
        </ul>
        
        <strong>Common negative prompt terms:</strong><br>
        <code>blurry, low quality, distorted, watermark, text, signature, duplicate, mutated, deformed</code>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
       
        # Session info
        common.render_sidebar()
                
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific and descriptive
        - Use negative prompts effectively
        - Experiment with different styles
        - Try various dimensions
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ✨ Create Your Image")
        
        # Main form
        with st.form("image_generation_form", clear_on_submit=False):
            prompt = st.text_area(
                "🎯 Describe what you want to see:",
                value=st.session_state.prompt,
                height=100,
                placeholder="A majestic mountain landscape at sunset with golden light reflecting on a pristine lake...",
                help="Be descriptive and specific for better results"
            )
            
            negative_prompt = st.text_area(
                "🚫 What should NOT be in the image:",
                value=st.session_state.negative_prompt,
                height=80,
                placeholder="blurry, low quality, distorted, unrealistic...",
                help="Negative prompts help eliminate unwanted elements"
            )
 
            # Style selection
            # st.markdown("### 🎨 Art Styles")
            nova_styles = {
                "PHOTOREALISM": "📸 Photorealism",
                "SOFT_DIGITAL_PAINTING": "🖼️ Digital Painting",
                "3D_ANIMATED_FAMILY_FILM": "🎬 3D Animation",
                "DESIGN_SKETCH": "✏️ Design Sketch",
                "FLAT_VECTOR_ILLUSTRATION": "📐 Vector Art",
                "GRAPHIC_NOVEL_ILLUSTRATION": "📚 Graphic Novel",
                "MAXIMALISM": "🌈 Maximalism",
                "MIDCENTURY_RETRO": "🕰️ Retro Style"
            }
            
            # Use example_style if available, otherwise use selected_style
            current_style = st.session_state.get("example_style", st.session_state.selected_style)
            if "example_style" in st.session_state:
                # Update selected_style and clear example_style
                st.session_state.selected_style = st.session_state.example_style
                del st.session_state.example_style
            
            selected_style = st.selectbox(
                "Choose art style:",
                options=list(nova_styles.keys()),
                format_func=lambda x: nova_styles[x],
                index=list(nova_styles.keys()).index(current_style),
                key="selected_style"
            )
 
            # Image dimensions
            # st.markdown("### 📐 Image Dimensions")
            col1_dim, col2_dim = st.columns(2)
            with col1_dim:
                width = st.selectbox("Width", [512, 768, 1024], index=0, key="image_width")
            with col2_dim:
                height = st.selectbox("Height", [512, 768, 1024], index=0, key="image_height")
                            
            generate_button = st.form_submit_button(
                "🚀 Generate Image", 
                type="primary",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        

    
    with col2:
        st.markdown("### 🖼️ Generated Image")
        
        if generate_button and prompt.strip():
            try:
                with st.spinner("🎨 Creating your masterpiece..."):
                    generated_image = nova_lib.get_image_from_nova_model(
                        prompt_content=prompt,
                        negative_prompt=negative_prompt if negative_prompt.strip() else None,
                        style=selected_style,
                        width=width,
                        height=height
                    )
                
                st.image(generated_image, caption="Generated Image", use_container_width=True)
                
                # Save to session history
                st.session_state.generated_images.insert(0, {
                    "image": generated_image,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "style": selected_style,
                    "timestamp": datetime.now()
                })
                
                # Keep only last 5 images
                if len(st.session_state.generated_images) > 5:
                    st.session_state.generated_images = st.session_state.generated_images[:5]
                
                st.success("✅ Image generated successfully!")
                
            except nova_lib.NovaImageError as e:
                st.error(f"❌ Generation failed: {e.message}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
        
        elif generate_button and not prompt.strip():
            st.warning("⚠️ Please enter a prompt to generate an image.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prompt examples section
    st.markdown("---")
    st.markdown("### 💡 Prompt Examples & Inspiration")
    
    examples = get_prompt_examples()
    
    # Category filter
    categories = list(set([ex["category"] for ex in examples]))
    selected_category = st.selectbox("Filter by category:", ["All"] + categories)
    
    if selected_category != "All":
        examples = [ex for ex in examples if ex["category"] == selected_category]
    
    # Display examples in columns
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            with st.expander(f"🎨 {example['title']}", expanded=False):
                st.markdown(f"**Category:** {example['category']}")
                st.markdown(f"**Style:** {example['style'].replace('_', ' ').title()}")
                st.markdown(f"**Description:** {example['description']}")
                
                st.markdown('<div class="prompt-example">', unsafe_allow_html=True)
                st.markdown(f"**Prompt:**\n{example['prompt']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if example['negative_prompt']:
                    st.markdown('<div class="negative-prompt-highlight">', unsafe_allow_html=True)
                    st.markdown(f"**Negative Prompt:**\n{example['negative_prompt']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button(f"Load This Example", key=f"load_{idx}", use_container_width=True):
                    load_prompt_example(example)
                    st.rerun()
    
    # Image history
    if st.session_state.generated_images:
        st.markdown("---")
        st.markdown("### 🕰️ Recent Generations")
        
        for idx, img_data in enumerate(st.session_state.generated_images[:3]):
            with st.expander(f"Image {idx + 1} - {img_data['timestamp'].strftime('%H:%M:%S')}", expanded=False):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(img_data["image"], use_container_width=True)
                with col2:
                    st.markdown(f"**Prompt:** {img_data['prompt'][:100]}...")
                    if img_data['negative_prompt']:
                        st.markdown(f"**Negative Prompt:** {img_data['negative_prompt'][:100]}...")
                    st.markdown(f"**Style:** {img_data['style']}")
    
    # Footer
    st.markdown("""
    <div class="footer">
        © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
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
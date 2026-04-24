import streamlit as st
import base64
import io
import json
import logging
import time
from PIL import Image
from datetime import datetime
import uuid
import utils.common as common
import utils.authenticate as authenticate
import utils.styles as styles
import utils.stability_image_lib as stability_lib

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_css():
    """Load custom CSS for modern UI/UX"""
    styles.load_css()


def initialize_session_state():
    """Initialize session state variables"""
    common.initialize_session_state()

    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []

    if "generation_count" not in st.session_state:
        st.session_state.generation_count = 0


def build_product_prompt(
    product_type, material, color, style, environment, lighting, additional_details
):
    """
    Build a comprehensive prompt for product image generation.

    Returns:
        str: Constructed prompt
    """
    prompt_parts = []

    # Core product description
    if product_type and color and material:
        prompt_parts.append(
            f"Professional product photography of a {color} {product_type} made of {material}"
        )
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
            "Vintage": "vintage retro styling, classic timeless appearance",
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
            "Home": "cozy home interior setting, domestic environment",
        }
        prompt_parts.append(env_descriptions.get(environment, environment))

    # Lighting
    if lighting:
        lighting_descriptions = {
            "Soft Box": "soft diffused studio lighting, even illumination",
            "Natural Light": "natural window lighting, soft shadows",
            "Dramatic": "dramatic lighting with strong shadows and highlights",
            "Rim Light": "rim lighting highlighting product edges",
            "Three-Point": "professional three-point lighting setup",
        }
        prompt_parts.append(lighting_descriptions.get(lighting, lighting))

    # Additional details
    if additional_details:
        prompt_parts.append(additional_details)

    # Quality modifiers
    prompt_parts.append(
        "high resolution, professional photography, commercial quality, sharp focus, detailed"
    )

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
            - Multiple Stability AI models
            - Various backgrounds & environments
            - Professional lighting setups

            🔧 **Technical Features**
            - Stability AI integration via Bedrock
            - Multiple aspect ratios
            - Real-time generation

            💡 **Use Cases**
            - E-commerce product images
            - Marketing materials
            - Design prototypes
            - Product catalogs

            🚀 **Powered By**
            - Amazon Bedrock
            - Stability AI Models
            - Streamlit Framework
            """)

        st.markdown("---")


def model_selection_panel():
    """Model selection and parameters in the side panel"""
    st.markdown("<h4>Model Selection</h4>", unsafe_allow_html=True)

    model_id = st.selectbox(
        "Select Model",
        options=list(stability_lib.MODELS.keys()),
        format_func=lambda x: stability_lib.MODELS[x],
        key="side_model",
    )

    st.markdown("<h4>Image Configuration</h4>", unsafe_allow_html=True)

    aspect_ratio = st.selectbox(
        "Aspect Ratio",
        options=list(stability_lib.ASPECT_RATIOS.keys()),
        format_func=lambda x: stability_lib.ASPECT_RATIOS[x],
        key="side_aspect_ratio",
        help="Controls the dimensions of the generated image",
    )

    num_images = st.slider(
        "Number of Images",
        min_value=1,
        max_value=4,
        value=1,
        key="side_num_images",
        help="Generate multiple variations at once",
    )

    seed = st.number_input(
        "Seed (0 for random)",
        min_value=0,
        max_value=4294967295,
        value=0,
        key="side_seed",
        help="Use same seed for reproducible results",
    )

    # Cost estimation
    cost_per_image = {
        "stability.stable-image-core-v1:1": 0.03,
        "stability.sd3-5-large-v1:0": 0.06,
        "stability.sd3-ultra-v1:1": 0.08,
    }
    unit_cost = cost_per_image.get(model_id, 0.04)
    total_cost = unit_cost * num_images

    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
                padding: 1rem; border-radius: 10px; margin: 1rem 0;
                border-left: 4px solid #FF9900;">
        <h4 style="margin: 0 0 0.5rem 0; color: #232F3E; font-size: 0.85rem;">💰 Cost Estimate</h4>
        <p style="margin: 0; color: #4B5563; font-size: 0.8rem;">
            ~${total_cost:.3f} for {num_images} image{"s" if num_images > 1 else ""} (${unit_cost:.3f}/image)
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    return model_id, aspect_ratio, seed, num_images


def render_generation_tab(model_id, aspect_ratio, seed, num_images):
    """Render the image generation tab"""
    st.markdown(
        """
    <div class="info-card">
        <h3>🖼️ Create Professional Product Images</h3>
        <p>Generate high-quality product images with customizable specifications.
        Perfect for e-commerce, marketing, and design projects.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

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
            "details": "aerodynamic design, carbon fiber plate, lightweight construction, professional racing shoe, dynamic angle, high performance",
        },
        "E-commerce Product Shot": {
            "environment": "White Background",
            "lighting": "Soft Box",
            "style": "Minimalist",
            "details": "centered, professional, high resolution, clean background",
        },
        "Lifestyle Photography": {
            "environment": "Lifestyle Scene",
            "lighting": "Natural Light",
            "style": "Natural",
            "details": "in-context usage, authentic setting, lifestyle appeal",
        },
        "Premium Luxury": {
            "environment": "Studio Setup",
            "lighting": "Dramatic",
            "style": "Luxury",
            "details": "elegant, premium quality, sophisticated, high-end",
        },
        "Tech Product": {
            "environment": "White Background",
            "lighting": "Three-Point",
            "style": "Tech",
            "details": "modern, sleek, futuristic, high-tech aesthetic",
        },
    }

    selected_template = st.selectbox(
        "Choose a template or start from scratch",
        options=list(templates.keys()),
        help="Pre-configured settings for common use cases",
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
            help="Specify the type of product you want to generate",
        )

        material = st.selectbox(
            "Material",
            [
                "",
                "Metal",
                "Plastic",
                "Wood",
                "Glass",
                "Fabric",
                "Leather",
                "Ceramic",
                "Carbon Fiber",
                "Mixed Materials",
            ],
            index=(
                [
                    "",
                    "Metal",
                    "Plastic",
                    "Wood",
                    "Glass",
                    "Fabric",
                    "Leather",
                    "Ceramic",
                    "Carbon Fiber",
                    "Mixed Materials",
                ].index(template_data.get("material", ""))
                if template_data.get("material")
                else 0
            ),
            help="Choose the primary material of your product",
        )

        color = st.text_input(
            "Color Scheme",
            value=template_data.get("color", ""),
            placeholder="e.g., Matte black, Rose gold, Navy blue",
            help="Describe the color or color combination",
        )

        style = st.selectbox(
            "Visual Style",
            ["Minimalist", "Luxury", "Industrial", "Natural", "Tech", "Vintage"],
            index=(
                ["Minimalist", "Luxury", "Industrial", "Natural", "Tech", "Vintage"].index(
                    template_data.get("style", "Minimalist")
                )
                if template_data.get("style")
                else 0
            ),
            help="Select the overall aesthetic style",
        )

    with col2:
        st.markdown("#### 🎬 Scene Settings")

        environment = st.selectbox(
            "Background/Environment",
            [
                "White Background",
                "Studio Setup",
                "Lifestyle Scene",
                "Nature",
                "Office",
                "Home",
            ],
            index=(
                [
                    "White Background",
                    "Studio Setup",
                    "Lifestyle Scene",
                    "Nature",
                    "Office",
                    "Home",
                ].index(template_data.get("environment", "White Background"))
                if template_data.get("environment")
                else 0
            ),
            help="Choose where your product will be displayed",
        )

        lighting = st.selectbox(
            "Lighting Setup",
            ["Soft Box", "Natural Light", "Dramatic", "Rim Light", "Three-Point"],
            index=(
                ["Soft Box", "Natural Light", "Dramatic", "Rim Light", "Three-Point"].index(
                    template_data.get("lighting", "Soft Box")
                )
                if template_data.get("lighting")
                else 0
            ),
            help="Select the lighting style for your product",
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
            help="Provide additional details to refine your image",
        )

    with col_detail2:
        negative_prompt = st.text_area(
            "Negative Prompt (Optional)",
            placeholder="e.g., blurry, low quality, distorted, watermark, text",
            height=100,
            help="Specify what you DON'T want in the image",
        )

    # Generate button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_button = st.button(
            f"🚀 Generate {num_images} Image{'s' if num_images > 1 else ''}",
            type="primary",
            use_container_width=True,
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
                    additional_details=additional_details,
                )

                progress_bar = st.progress(0, text="Initializing generation...")
                generated_images = []

                for i in range(num_images):
                    progress = (i + 1) / num_images
                    progress_bar.progress(
                        progress, text=f"🎨 Generating image {i + 1}/{num_images}..."
                    )

                    current_seed = (seed + i) if seed > 0 else 0

                    image, image_bytes = stability_lib.generate_image(
                        prompt_content=prompt,
                        negative_prompt=negative_prompt if negative_prompt and negative_prompt.strip() else None,
                        model_id=model_id,
                        aspect_ratio=aspect_ratio,
                        seed=current_seed,
                    )

                    generated_images.append(
                        {
                            "image": image,
                            "bytes": image_bytes,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "seed": current_seed,
                        }
                    )

                progress_bar.progress(1.0, text="✅ Generation complete!")
                time.sleep(0.5)
                progress_bar.empty()

                # Display generated images
                st.markdown("### 🎨 Generated Images")

                if num_images == 1:
                    img_data = generated_images[0]
                    st.image(
                        img_data["image"],
                        caption="Generated Product Image",
                        use_container_width=True,
                    )

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Model", stability_lib.MODELS.get(model_id, model_id).split("—")[0].strip())
                    col2.metric("File Size", f"{len(img_data['bytes']) / 1024:.1f} KB")
                    col3.metric("Seed", img_data["seed"] if img_data["seed"] > 0 else "Random")

                    buf = io.BytesIO()
                    img_data["image"].save(buf, format="PNG")
                    st.download_button(
                        label="📥 Download Image",
                        data=buf.getvalue(),
                        file_name=f"product_{product_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                else:
                    cols = st.columns(min(num_images, 4))
                    for idx, img_data in enumerate(generated_images):
                        with cols[idx % 4]:
                            st.image(
                                img_data["image"],
                                caption=f"Variation {idx + 1}",
                                use_container_width=True,
                            )
                            buf = io.BytesIO()
                            img_data["image"].save(buf, format="PNG")
                            st.download_button(
                                label="📥 Download",
                                data=buf.getvalue(),
                                file_name=f"product_{product_type.replace(' ', '_')}_v{idx + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                key=f"download_{idx}",
                                use_container_width=True,
                            )

                    st.markdown("---")
                    import zipfile

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for idx, img_data in enumerate(generated_images):
                            buf = io.BytesIO()
                            img_data["image"].save(buf, format="PNG")
                            zip_file.writestr(
                                f"product_{product_type.replace(' ', '_')}_v{idx + 1}.png",
                                buf.getvalue(),
                            )

                    st.download_button(
                        label="📦 Download All as ZIP",
                        data=zip_buffer.getvalue(),
                        file_name=f"product_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

                st.success(
                    f"✅ Successfully generated {num_images} image{'s' if num_images > 1 else ''}!"
                )

                # Save to session
                for img_data in generated_images:
                    st.session_state.generated_images.append(
                        {
                            "image": img_data["image"],
                            "prompt": img_data["prompt"],
                            "negative_prompt": img_data["negative_prompt"],
                            "timestamp": datetime.now(),
                            "product_type": product_type,
                            "style": style,
                            "model": stability_lib.MODELS.get(model_id, model_id),
                            "aspect_ratio": aspect_ratio,
                            "seed": img_data["seed"],
                        }
                    )
                st.session_state.generation_count += num_images

            except stability_lib.StabilityImageError as e:
                st.error(f"❌ Generation Error: {e.message}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                logger.error(f"Error in image generation: {str(e)}")


def render_gallery_tab():
    """Render the gallery tab with generated images"""
    st.markdown(
        """
    <div class="info-card">
        <h3>🖼️ Image Gallery</h3>
        <p>View and manage all images generated during this session.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.session_state.generated_images:
        images = sorted(
            st.session_state.generated_images,
            key=lambda x: x["timestamp"],
            reverse=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            unique_products = list(
                set(img.get("product_type", "Unknown") for img in images)
            )
            filter_product = st.selectbox(
                "Filter by Product", ["All"] + sorted(unique_products)
            )

        with col2:
            unique_styles = list(set(img.get("style", "Unknown") for img in images))
            filter_style = st.selectbox(
                "Filter by Style", ["All"] + sorted(unique_styles)
            )

        with col3:
            sort_by = st.selectbox(
                "Sort by", ["Newest First", "Oldest First", "Product Type"]
            )

        filtered_images = images
        if filter_product != "All":
            filtered_images = [
                img
                for img in filtered_images
                if img.get("product_type") == filter_product
            ]
        if filter_style != "All":
            filtered_images = [
                img for img in filtered_images if img.get("style") == filter_style
            ]

        if sort_by == "Oldest First":
            filtered_images = sorted(filtered_images, key=lambda x: x["timestamp"])
        elif sort_by == "Product Type":
            filtered_images = sorted(
                filtered_images, key=lambda x: x.get("product_type", "")
            )

        # Bulk export
        if len(filtered_images) > 1:
            st.markdown("### 📥 Bulk Export")
            col_export1, col_export2 = st.columns(2)

            with col_export1:
                import zipfile

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for idx, img_data in enumerate(filtered_images):
                        buf = io.BytesIO()
                        img_data["image"].save(buf, format="PNG")
                        product_name = (
                            img_data.get("product_type", "product").replace(" ", "_")
                        )
                        zip_file.writestr(
                            f"{idx + 1}_{product_name}_{img_data['timestamp'].strftime('%Y%m%d_%H%M%S')}.png",
                            buf.getvalue(),
                        )

                st.download_button(
                    label=f"📦 Download All ({len(filtered_images)} images) as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"gallery_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

            with col_export2:
                if st.button("🗑️ Clear Gallery", use_container_width=True):
                    st.session_state.generated_images = []
                    st.session_state.generation_count = 0
                    st.rerun()

        st.markdown(f"### 🎨 Gallery ({len(filtered_images)} images)")

        cols = st.columns(3)
        for idx, img_data in enumerate(filtered_images):
            with cols[idx % 3]:
                st.image(img_data["image"], use_container_width=True)

                st.caption(
                    f"**{img_data.get('product_type', 'Unknown')}** - {img_data.get('style', 'Unknown')}"
                )
                st.caption(f"🕐 {img_data['timestamp'].strftime('%H:%M:%S')}")

                with st.expander("ℹ️ Details"):
                    st.text(f"Model: {img_data.get('model', 'N/A')}")
                    st.text(f"Aspect Ratio: {img_data.get('aspect_ratio', 'N/A')}")
                    st.text(f"Seed: {img_data.get('seed', 'Random')}")
                    if img_data.get("prompt"):
                        st.text_area(
                            "Prompt",
                            img_data["prompt"],
                            height=100,
                            key=f"prompt_{idx}",
                        )

                buf = io.BytesIO()
                img_data["image"].save(buf, format="PNG")
                st.download_button(
                    label="📥 Download",
                    data=buf.getvalue(),
                    file_name=f"gallery_{img_data.get('product_type', 'image').replace(' ', '_')}_{idx}.png",
                    mime="image/png",
                    key=f"download_gallery_{idx}",
                    use_container_width=True,
                )
    else:
        st.info(
            "📸 No images generated yet. Go to the 'Generate' tab to create your first product image!"
        )


def render_examples_tab():
    """Render the examples tab with predefined templates"""
    st.markdown(
        """
    <div class="info-card">
        <h3>💡 Product Image Examples</h3>
        <p>Explore different product types and styles to inspire your creations.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    examples = [
        {
            "title": "🕐 Luxury Watch",
            "product": "Luxury wristwatch",
            "material": "Stainless steel and leather",
            "color": "Silver with brown leather strap",
            "style": "Luxury",
            "environment": "Studio Setup",
            "lighting": "Rim Light",
            "details": "Swiss design, chronograph, reflective surface",
        },
        {
            "title": "👟 Athletic Sneakers",
            "product": "Running shoes",
            "material": "Mesh and rubber",
            "color": "White with neon accents",
            "style": "Tech",
            "environment": "Lifestyle Scene",
            "lighting": "Natural Light",
            "details": "Dynamic angle, showing sole technology",
        },
        {
            "title": "💼 Leather Bag",
            "product": "Professional briefcase",
            "material": "Premium leather",
            "color": "Deep brown",
            "style": "Luxury",
            "environment": "Office",
            "lighting": "Soft Box",
            "details": "Gold hardware, textured leather, compartments visible",
        },
        {
            "title": "🎧 Wireless Headphones",
            "product": "Over-ear headphones",
            "material": "Plastic and metal",
            "color": "Matte black",
            "style": "Minimalist",
            "environment": "White Background",
            "lighting": "Three-Point",
            "details": "Noise cancellation, premium audio, foldable design",
        },
        {
            "title": "🌸 Perfume Bottle",
            "product": "Perfume bottle",
            "material": "Glass",
            "color": "Crystal clear with gold cap",
            "style": "Luxury",
            "environment": "Studio Setup",
            "lighting": "Dramatic",
            "details": "Elegant shape, light refraction, premium packaging",
        },
        {
            "title": "📱 Smartphone",
            "product": "Flagship smartphone",
            "material": "Glass and aluminum",
            "color": "Midnight blue",
            "style": "Tech",
            "environment": "White Background",
            "lighting": "Soft Box",
            "details": "Multiple cameras, edge-to-edge display, sleek profile",
        },
    ]

    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            with st.expander(example["title"], expanded=False):
                st.markdown(
                    f"""
                <div class="feature-card">
                <strong>Product:</strong> {example['product']}<br>
                <strong>Material:</strong> {example['material']}<br>
                <strong>Color:</strong> {example['color']}<br>
                <strong>Style:</strong> {example['style']}<br>
                <strong>Environment:</strong> {example['environment']}<br>
                <strong>Lighting:</strong> {example['lighting']}<br>
                <strong>Details:</strong> {example['details']}
                </div>
                """,
                    unsafe_allow_html=True,
                )


def main():
    """Main application function"""
    st.set_page_config(
        page_title="Stability AI - Product Image Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()
    load_css()
    render_sidebar()

    # Main header
    st.markdown(
        """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #232F3E; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
            🎨 Stability AI Product Image Generator
        </h1>
        <p style="color: #6B7280; font-size: 1.1rem; margin: 0;">
            AI-Powered Professional Product Photography
            <br/>
            <span style="font-size: 0.875rem; color: #9CA3AF; margin-top: 0.5rem; display: inline-block;">
                Powered by Amazon Bedrock &bull; Stability AI Models &bull; Multiple Aspect Ratios &bull; Batch Generation
            </span>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create 70/30 split layout
    col_main, col_side = st.columns([7, 3])

    with col_side:
        with st.container(border=True):
            model_id, aspect_ratio, seed, num_images = model_selection_panel()

    with col_main:
        tab1, tab2, tab3 = st.tabs(["🚀 Generate", "🖼️ Gallery", "💡 Examples"])

        with tab1:
            render_generation_tab(model_id, aspect_ratio, seed, num_images)

        with tab2:
            render_gallery_tab()

        with tab3:
            render_examples_tab()

    # Footer
    st.markdown(
        """
    <div style="margin-top: 4rem; padding: 1.5rem; background: linear-gradient(135deg, #232F3E 0%, #16191F 100%);
                border-radius: 10px; text-align: center; position: relative;">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px;
                    background: linear-gradient(90deg, #FF9900 0%, #146EB4 100%);"></div>
        <p style="color: #D1D5DB; font-size: 0.875rem; margin: 0;">
            © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# Main execution flow
if __name__ == "__main__":
    if "localhost" in st.context.headers.get("host", ""):
        main()
    else:
        is_authenticated = authenticate.login()
        if is_authenticated:
            main()

import streamlit as st
import utils.stability_image_lib as stability_lib
import uuid
from datetime import datetime
import utils.common as common
import utils.authenticate as authenticate
from utils.styles import load_css, sub_header


# Set page configuration
st.set_page_config(
    page_title="Stability AI - Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize session state variables"""
    common.initialize_session_state()

    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = datetime.now()

    if "prompt" not in st.session_state:
        st.session_state.prompt = ""

    if "negative_prompt" not in st.session_state:
        st.session_state.negative_prompt = ""

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = stability_lib.DEFAULT_MODEL_ID

    if "aspect_ratio" not in st.session_state:
        st.session_state.aspect_ratio = "1:1"

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
            "description": "Perfect for professional portraits with realistic details",
        },
        {
            "title": "Fantasy Landscape",
            "category": "Fantasy",
            "prompt": "Magical forest with glowing mushrooms, ethereal mist, ancient trees, mystical atmosphere, vibrant colors",
            "negative_prompt": "modern buildings, cars, people, realistic, photograph, black and white",
            "description": "Creates an enchanting fantasy world",
        },
        {
            "title": "Product Design",
            "category": "Design",
            "prompt": "Minimalist smartphone design, sleek metal frame, glass surface, modern technology, studio lighting",
            "negative_prompt": "cluttered, complex, old-fashioned, broken, scratched, dirty",
            "description": "Great for product visualization and design concepts",
        },
        {
            "title": "Architectural Visualization",
            "category": "Architecture",
            "prompt": "Modern sustainable house, solar panels, large windows, natural materials, garden integration, contemporary design",
            "negative_prompt": "old, abandoned, damaged, dark, cluttered, unrealistic proportions",
            "description": "Professional architectural rendering",
        },
        {
            "title": "Character Illustration",
            "category": "Character",
            "prompt": "Friendly robot companion, colorful design, approachable appearance, clean lines, family-friendly, 3D animated style",
            "negative_prompt": "scary, threatening, weapon, dark, violent, realistic human features",
            "description": "Perfect for character design and animation",
        },
        {
            "title": "Food Photography",
            "category": "Food",
            "prompt": "Gourmet pasta dish, fresh ingredients, restaurant presentation, warm lighting, appetizing colors",
            "negative_prompt": "unappetizing, messy, dark, low quality, artificial, processed",
            "description": "Professional food photography style",
        },
        {
            "title": "Abstract Art",
            "category": "Art",
            "prompt": "Dynamic abstract composition, bold geometric shapes, vibrant color palette, modern art style, maximalist",
            "negative_prompt": "realistic, photographic, dull colors, simple, basic, low contrast",
            "description": "Bold and expressive abstract artwork",
        },
        {
            "title": "Retro Design",
            "category": "Vintage",
            "prompt": "Retro kitchen appliance, 1950s design, pastel colors, chrome details, vintage aesthetics, mid-century modern",
            "negative_prompt": "modern, futuristic, digital, minimalist, monochrome, damaged",
            "description": "Classic mid-century modern design",
        },
    ]


def load_prompt_example(example):
    """Load a prompt example into session state"""
    st.session_state.prompt = example["prompt"]
    st.session_state.negative_prompt = example["negative_prompt"]


def main():
    load_css()
    initialize_session_state()

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🎨 Stability AI Image Generator</h1>
        <p>AI-Powered Image Generation with Advanced Prompt Engineering</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="info-box">
    Generate stunning AI images using Stability AI models on Amazon Bedrock. Learn how to craft effective prompts
    and use negative prompts to refine your results. Choose between Stable Image Core (fast &amp; affordable),
    Stable Diffusion 3.5 Large (high quality), or Stable Image Ultra (photorealistic premium).
    </div>""",
        unsafe_allow_html=True,
    )

    # Negative prompt explanation
    with st.expander("📚 Understanding Negative Prompts"):
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )

    # Sidebar
    with st.sidebar:
        common.render_sidebar()

        st.markdown("---")
        st.markdown(
            sub_header("Tips", "💡", "minimal"), unsafe_allow_html=True
        )
        st.markdown(
            """
        - Be specific and descriptive
        - Use negative prompts effectively
        - Try different models for different needs
        - Experiment with aspect ratios
        """
        )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            sub_header("Create Your Image", "✨"), unsafe_allow_html=True
        )

        with st.form("image_generation_form", clear_on_submit=False):
            prompt = st.text_area(
                "🎯 Describe what you want to see:",
                value=st.session_state.prompt,
                height=100,
                placeholder="A majestic mountain landscape at sunset with golden light reflecting on a pristine lake...",
                help="Be descriptive and specific for better results",
            )

            negative_prompt = st.text_area(
                "🚫 What should NOT be in the image:",
                value=st.session_state.negative_prompt,
                height=80,
                placeholder="blurry, low quality, distorted, unrealistic...",
                help="Negative prompts help eliminate unwanted elements",
            )

            # Model selection
            selected_model = st.selectbox(
                "🤖 Model:",
                options=list(stability_lib.MODELS.keys()),
                format_func=lambda x: stability_lib.MODELS[x],
                index=list(stability_lib.MODELS.keys()).index(
                    st.session_state.selected_model
                ),
                key="selected_model",
            )

            # Aspect ratio selection
            selected_ratio = st.selectbox(
                "📐 Aspect Ratio:",
                options=list(stability_lib.ASPECT_RATIOS.keys()),
                format_func=lambda x: stability_lib.ASPECT_RATIOS[x],
                index=list(stability_lib.ASPECT_RATIOS.keys()).index(
                    st.session_state.aspect_ratio
                ),
                key="aspect_ratio",
            )

            generate_button = st.form_submit_button(
                "🚀 Generate Image",
                type="primary",
                use_container_width=True,
            )

    with col2:
        st.markdown(
            sub_header("Generated Image", "🖼️"), unsafe_allow_html=True
        )

        if generate_button and prompt.strip():
            try:
                with st.spinner("🎨 Creating your masterpiece..."):
                    generated_image = stability_lib.generate_image(
                        prompt_content=prompt,
                        negative_prompt=negative_prompt
                        if negative_prompt.strip()
                        else None,
                        model_id=selected_model,
                        aspect_ratio=selected_ratio,
                    )

                st.image(
                    generated_image,
                    caption="Generated Image",
                    use_container_width=True,
                )

                # Save to session history
                st.session_state.generated_images.insert(
                    0,
                    {
                        "image": generated_image,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "model": stability_lib.MODELS.get(
                            selected_model, selected_model
                        ),
                        "aspect_ratio": selected_ratio,
                        "timestamp": datetime.now(),
                    },
                )

                # Keep only last 5 images
                if len(st.session_state.generated_images) > 5:
                    st.session_state.generated_images = (
                        st.session_state.generated_images[:5]
                    )

                st.success("✅ Image generated successfully!")

            except stability_lib.StabilityImageError as e:
                st.error(f"❌ Generation failed: {e.message}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

        elif generate_button and not prompt.strip():
            st.warning("⚠️ Please enter a prompt to generate an image.")

    # Prompt examples section
    st.markdown("---")
    st.markdown(
        sub_header("Prompt Examples & Inspiration", "💡", "minimal"),
        unsafe_allow_html=True,
    )

    examples = get_prompt_examples()

    # Category filter
    categories = sorted(set(ex["category"] for ex in examples))
    selected_category = st.selectbox(
        "Filter by category:", ["All"] + categories
    )

    if selected_category != "All":
        examples = [
            ex for ex in examples if ex["category"] == selected_category
        ]

    # Display examples in columns
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            with st.expander(f"🎨 {example['title']}", expanded=False):
                st.markdown(f"**Category:** {example['category']}")
                st.markdown(f"**Description:** {example['description']}")

                st.markdown(
                    '<div class="prompt-example">', unsafe_allow_html=True
                )
                st.markdown(f"**Prompt:**\n{example['prompt']}")
                st.markdown("</div>", unsafe_allow_html=True)

                if example["negative_prompt"]:
                    st.markdown(
                        '<div class="negative-prompt-highlight">',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Negative Prompt:**\n{example['negative_prompt']}"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                if st.button(
                    "Load This Example",
                    key=f"load_{idx}",
                    use_container_width=True,
                ):
                    load_prompt_example(example)
                    st.rerun()

    # Image history
    if st.session_state.generated_images:
        st.markdown("---")
        st.markdown(
            sub_header("Recent Generations", "🕰️", "minimal"),
            unsafe_allow_html=True,
        )

        for idx, img_data in enumerate(
            st.session_state.generated_images[:3]
        ):
            with st.expander(
                f"Image {idx + 1} - {img_data['timestamp'].strftime('%H:%M:%S')}",
                expanded=False,
            ):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(img_data["image"], use_container_width=True)
                with c2:
                    st.markdown(
                        f"**Prompt:** {img_data['prompt'][:100]}..."
                    )
                    if img_data["negative_prompt"]:
                        st.markdown(
                            f"**Negative Prompt:** {img_data['negative_prompt'][:100]}..."
                        )
                    st.markdown(f"**Model:** {img_data['model']}")
                    st.markdown(
                        f"**Aspect Ratio:** {img_data['aspect_ratio']}"
                    )

    # Footer
    st.markdown(
        """
    <div class="footer">
        © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    if "localhost" in st.context.headers["host"]:
        main()
    else:
        is_authenticated = authenticate.login()
        if is_authenticated:
            main()

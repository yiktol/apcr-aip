
import os
import random
import requests
import streamlit as st
from PIL import Image
from typing import List, Optional, Dict, Any
import time

# Set page configuration with modern styling
st.set_page_config(
    page_title='Nova Canvas',
    page_icon='🎨',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .generation-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .sample-prompt-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .sample-prompt-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
        transform: translateY(-1px);
    }
    
    .prompt-category {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .prompt-title {
        font-weight: bold;
        color: #333;
        margin-bottom: 0.3rem;
    }
    
    .prompt-text {
        color: #666;
        font-size: 0.9rem;
        line-height: 1.4;
        margin-bottom: 0.5rem;
    }
    
    .prompt-negative {
        color: #888;
        font-size: 0.8rem;
        font-style: italic;
    }
    
    .color-palette {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .status-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .quick-action-btn {
        background: #667eea;
        color: white;
        border: none;
        padding: 0.4rem 0.8rem;
        border-radius: 5px;
        font-size: 0.8rem;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .quick-action-btn:hover {
        background: #5a6fd8;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = 'http://localhost:8000'

# Sample prompts database
SAMPLE_PROMPTS = {
    "Portrait Photography": {
        "Professional Headshot": {
            "prompt": "Professional corporate headshot of a confident businesswoman in her 30s, wearing a navy blue blazer, natural lighting, clean white background, shot with 85mm lens, shallow depth of field, professional photography",
            "negative": "unprofessional, casual clothing, dark lighting, cluttered background, low quality, blurry"
        },
        "Fashion Portrait": {
            "prompt": "High-fashion portrait of an elegant model wearing avant-garde designer clothing, dramatic studio lighting with rim light, moody atmosphere, shot on medium format camera, editorial style",
            "negative": "amateur lighting, casual wear, cluttered background, low resolution, oversaturated"
        },
        "Artistic Portrait": {
            "prompt": "Artistic black and white portrait of a jazz musician holding a saxophone, dramatic side lighting, vintage film noir style, high contrast, emotional expression, classic photography",
            "negative": "color, modern elements, harsh lighting, digital artifacts, low quality"
        }
    },
    
    "Landscape & Nature": {
        "Mountain Landscape": {
            "prompt": "Breathtaking mountain landscape at golden hour, snow-capped peaks reflecting in a pristine alpine lake, dramatic clouds, wide-angle composition, professional landscape photography, HDR processing",
            "negative": "people, buildings, vehicles, overprocessed, artificial colors, low resolution"
        },
        "Forest Scene": {
            "prompt": "Mystical ancient forest with towering redwood trees, dappled sunlight filtering through the canopy, morning mist, moss-covered ground, ethereal atmosphere, nature photography",
            "negative": "people, artificial structures, harsh lighting, urban elements, low quality"
        },
        "Ocean Sunset": {
            "prompt": "Dramatic ocean sunset with golden and orange hues reflecting on gentle waves, silhouetted rock formations, long exposure for smooth water, coastal landscape photography",
            "negative": "people, buildings, overexposed, artificial colors, cluttered composition"
        }
    },
    
    "Architecture & Urban": {
        "Modern Building": {
            "prompt": "Contemporary glass skyscraper with sleek geometric design, reflecting blue sky and clouds, minimalist architecture, shot from low angle, architectural photography, clean lines",
            "negative": "old buildings, cluttered surroundings, poor weather, distorted perspective, low quality"
        },
        "Historic Architecture": {
            "prompt": "Grand Gothic cathedral with intricate stone carvings, towering spires, stained glass windows, dramatic lighting, architectural detail photography, historical building",
            "negative": "modern elements, poor lighting, crowds, scaffolding, low resolution"
        },
        "Urban Street": {
            "prompt": "Vibrant city street at night with neon signs, light trails from traffic, modern urban architecture, street photography, dynamic composition, metropolitan atmosphere",
            "negative": "empty streets, poor lighting, suburban areas, low quality, motion blur"
        }
    },
    
    "Food & Culinary": {
        "Gourmet Dish": {
            "prompt": "Elegant gourmet dish presentation on white porcelain plate, colorful vegetables artfully arranged, natural lighting from window, food photography, shallow depth of field, restaurant quality",
            "negative": "messy presentation, artificial lighting, plastic food, low quality, cluttered background"
        },
        "Rustic Cooking": {
            "prompt": "Rustic homemade bread on wooden cutting board, flour dusting, warm kitchen lighting, cozy atmosphere, artisanal baking, lifestyle food photography",
            "negative": "industrial kitchen, processed food, harsh lighting, modern appliances, low quality"
        },
        "Coffee Art": {
            "prompt": "Perfect latte art in white ceramic cup, heart-shaped foam design, marble table surface, natural morning light, coffee shop atmosphere, overhead shot",
            "negative": "paper cups, artificial sweeteners, cluttered table, poor foam art, low resolution"
        }
    },
    
    "Abstract & Artistic": {
        "Fluid Art": {
            "prompt": "Abstract fluid art with flowing organic shapes in vibrant blues and golds, smooth gradients, digital art, contemporary abstract expressionism, high resolution",
            "negative": "geometric shapes, harsh edges, dull colors, low resolution, photographic elements"
        },
        "Geometric Abstract": {
            "prompt": "Modern geometric abstract composition with bold shapes in complementary colors, clean lines, minimalist design, contemporary digital art, balanced composition",
            "negative": "organic shapes, cluttered design, muted colors, traditional art styles, low quality"
        },
        "Watercolor Style": {
            "prompt": "Soft watercolor abstract painting with gentle color bleeding, pastel tones, artistic texture, traditional watercolor techniques, peaceful and serene mood",
            "negative": "harsh lines, digital artifacts, bright neon colors, photorealistic elements, low resolution"
        }
    },
    
    "Animals & Wildlife": {
        "Majestic Wildlife": {
            "prompt": "Majestic lion portrait in natural habitat, golden mane catching sunlight, intense gaze, African savanna background, wildlife photography, shallow depth of field",
            "negative": "zoo setting, artificial backgrounds, poor lighting, domestic animals, low quality"
        },
        "Bird Photography": {
            "prompt": "Colorful hummingbird hovering near vibrant flowers, wings in motion, macro photography, natural garden setting, high-speed capture, detailed feathers",
            "negative": "caged birds, artificial flowers, motion blur, poor focus, indoor setting"
        },
        "Ocean Life": {
            "prompt": "Graceful sea turtle swimming in crystal clear blue water, underwater photography, natural lighting, coral reef background, marine life documentation",
            "negative": "aquarium setting, murky water, artificial coral, flash photography, low visibility"
        }
    },
    
    "Fantasy & Sci-Fi": {
        "Fantasy Landscape": {
            "prompt": "Magical fantasy landscape with floating islands, waterfalls cascading into clouds, ethereal lighting, mystical creatures, fantasy art style, high detail digital painting",
            "negative": "realistic photography, modern elements, low fantasy, poor lighting, low resolution"
        },
        "Cyberpunk Scene": {
            "prompt": "Futuristic cyberpunk cityscape at night, neon lights reflecting on wet streets, holographic advertisements, flying vehicles, dystopian atmosphere, sci-fi concept art",
            "negative": "historical elements, natural lighting, rural settings, low-tech, poor detail"
        },
        "Dragon Art": {
            "prompt": "Magnificent dragon perched on ancient castle ruins, scales shimmering in moonlight, fantasy creature design, epic composition, detailed digital art",
            "negative": "cartoon style, modern buildings, poor anatomy, low detail, realistic animals"
        }
    },
    
    "Fashion & Style": {
        "Haute Couture": {
            "prompt": "High-fashion runway model wearing elaborate avant-garde couture gown, dramatic fashion photography, studio lighting, editorial style, luxury fashion",
            "negative": "casual wear, natural settings, poor lighting, amateur photography, low fashion"
        },
        "Street Fashion": {
            "prompt": "Stylish street fashion photography, trendy young person in urban setting, contemporary casual wear, natural lighting, lifestyle photography",
            "negative": "formal wear, studio setting, posed photography, outdated fashion, poor styling"
        },
        "Vintage Fashion": {
            "prompt": "Elegant 1950s vintage fashion portrait, classic dress and hairstyle, retro styling, film photography aesthetic, timeless elegance",
            "negative": "modern fashion, contemporary hairstyles, digital photography look, casual clothing"
        }
    }
}

# Initialize session state
def init_session_state():
    defaults = {
        'generated_images': [],
        'improved_prompt': None,
        'generation_mode': 'text',
        'colors': [],
        'generation_count': 0,
        'last_generation_time': None,
        'selected_prompt': '',
        'selected_negative': ''
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Utility functions
class ColorPalette:
    @staticmethod
    def add_color(color: str) -> bool:
        if len(st.session_state.colors) < 10:
            st.session_state.colors.append(color)
            return True
        return False
    
    @staticmethod
    def remove_color(index: int) -> bool:
        if 0 <= index < len(st.session_state.colors):
            st.session_state.colors.pop(index)
            return True
        return False
    
    @staticmethod
    def clear_colors():
        st.session_state.colors = []

class ImageGenerator:
    @staticmethod
    def generate_image(
        prompt: str,
        negative_prompt: str,
        width: int = 1024,
        height: int = 1024,
        quality: str = 'standard',
        cfg_scale: float = 6.5,
        seed: Optional[int] = None,
        number_of_images: int = 1,
        colors: Optional[List[str]] = None,
        use_improved_prompt: bool = False,
    ):
        try:
            payload = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'quality': quality,
                'cfg_scale': cfg_scale,
                'seed': seed,
                'number_of_images': number_of_images,
                'use_improved_prompt': use_improved_prompt,
            }
            if colors:
                payload['colors'] = colors

            response = requests.post(
                f'{API_URL}/generate',
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f'Error communicating with the API: {str(e)}',
                'image_paths': [],
            }

def render_sample_prompt_card(category: str, title: str, data: Dict[str, str], key: str):
    """Render a sample prompt card with modern styling"""
    card_html = f"""
    <div class="sample-prompt-card" onclick="document.getElementById('{key}').click()">
        <div class="prompt-category">{category}</div>
        <div class="prompt-title">{title}</div>
        <div class="prompt-text">{data['prompt'][:100]}...</div>
        <div class="prompt-negative">Negative: {data['negative'][:50]}...</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Hidden button for functionality
    if st.button("Select", key=key, help=f"Use {title} prompt", type="secondary"):
        st.session_state.selected_prompt = data['prompt']
        st.session_state.selected_negative = data['negative']
        st.rerun()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎨 Nova Canvas Image Generator</h1>
    <p>Transform your ideas into stunning visuals with AI-powered creativity</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title('⚙️ Generation Settings')
    
    # Generation mode with modern tabs
    st.subheader('🎯 Generation Mode')
    mode_tab1, mode_tab2 = st.tabs(['Text-to-Image', 'Color-Guided'])
    
    with mode_tab1:
        if st.button('📝 Text Mode', use_container_width=True):
            st.session_state.generation_mode = 'text'
    
    with mode_tab2:
        if st.button('🎨 Color Mode', use_container_width=True):
            st.session_state.generation_mode = 'color'
    
    # Current mode indicator
    mode_emoji = '📝' if st.session_state.generation_mode == 'text' else '🎨'
    mode_name = 'Text-to-Image' if st.session_state.generation_mode == 'text' else 'Color-Guided'
    st.info(f'{mode_emoji} Current mode: **{mode_name}**')
    
    st.divider()
    
    # Image dimensions with presets
    st.subheader('📐 Image Dimensions')
    
    dimension_presets = {
        'Square (512×512)': (512, 512),
        'Square (1024×1024)': (1024, 1024),
        'Portrait (768×1024)': (768, 1024),
        'Landscape (1024×768)': (1024, 768)
    }
    
    preset = st.selectbox('Choose preset:', list(dimension_presets.keys()))
    width, height = dimension_presets[preset]
    
    # Display selected dimensions
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Width', f'{width}px')
    with col2:
        st.metric('Height', f'{height}px')
    
    st.divider()
    
    # Quality and advanced settings
    st.subheader('⚡ Quality & Settings')
    
    quality = st.select_slider(
        'Image Quality',
        options=['standard', 'premium'],
        value='standard',
        format_func=lambda x: f'🔸 {x.title()}' if x == 'standard' else f'⭐ {x.title()}'
    )
    
    cfg_scale = st.slider(
        'CFG Scale (Prompt Adherence)',
        min_value=1.1,
        max_value=10.0,
        value=6.5,
        step=0.1,
        help='Higher values make the image follow the prompt more strictly'
    )
    
    # Seed settings
    with st.expander('🎲 Seed Settings'):
        use_seed = st.toggle('Use specific seed', value=False)
        if use_seed:
            seed = st.number_input(
                'Seed Value',
                min_value=0,
                max_value=858993459,
                value=random.randint(0, 858993459)
            )
        else:
            seed = None
    
    number_of_images = st.slider(
        '📊 Number of Images',
        min_value=1,
        max_value=5,
        value=1,
        help='Generate multiple variations at once'
    )

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(['🎨 Create', '📚 Sample Prompts', '💡 Inspiration'])

with tab1:
    # Main creation interface
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Prompt input section
        st.markdown('<div class="generation-card">', unsafe_allow_html=True)
        st.subheader('✍️ Describe Your Vision')
        
        prompt = st.text_area(
            'Main Prompt',
            value=st.session_state.selected_prompt,
            placeholder='A serene mountain landscape at sunset with golden light reflecting on a crystal lake...',
            help='Be descriptive and specific for better results',
            height=120,
            key='main_prompt'
        )
        
        negative_prompt = st.text_area(
            'Negative Prompt',
            value=st.session_state.selected_negative,
            placeholder='low quality, blurry, distorted, watermark...',
            help='Elements to exclude from the image',
            height=80,
            key='main_negative'
        )
        
        # Auto-suggestion for negative prompt
        if not negative_prompt:
            suggested_negative = "people, anatomy, hands, low quality, low resolution, low detail"
            if st.button(f'💡 Use suggested: "{suggested_negative}"'):
                st.session_state.selected_negative = suggested_negative
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Color palette section (modern layout)
        if st.session_state.generation_mode == 'color':
            st.subheader('🎨 Color Palette')
            
            if st.session_state.colors:
                # Display colors in a modern grid
                cols = st.columns(5)
                for i, color in enumerate(st.session_state.colors):
                    with cols[i % 5]:
                        st.color_picker(f'Color {i+1}', color, key=f'color_{i}', disabled=True)
                        if st.button('🗑️', key=f'remove_{i}', help='Remove color'):
                            ColorPalette.remove_color(i)
                            st.rerun()
            
            # Add color section
            color_col1, color_col2, color_col3 = st.columns([2, 1, 1])
            with color_col1:
                new_color = st.color_picker('Add new color', '#FF6B6B')
            with color_col2:
                if st.button('➕ Add', disabled=len(st.session_state.colors) >= 10):
                    if ColorPalette.add_color(new_color):
                        st.rerun()
            with color_col3:
                if st.button('🧹 Clear All', disabled=not st.session_state.colors):
                    ColorPalette.clear_colors()
                    st.rerun()
            
            if len(st.session_state.colors) >= 10:
                st.warning('⚠️ Maximum 10 colors allowed')

        # Generation controls
        st.divider()
        
        gen_col1, gen_col2 = st.columns([3, 1])
        with gen_col1:
            use_improved_prompt = st.toggle(
                '🚀 Use AI Prompt Enhancement',
                value=True,
                help='Enhance your prompt using Amazon Nova Micro Model'
            )
        
        with gen_col2:
            can_generate = bool(prompt and negative_prompt)
            generate_btn = st.button(
                '🎨 Generate Images',
                type='primary',
                disabled=not can_generate,
                use_container_width=True
            )

    with col_side:
        # Statistics and info panel
        st.subheader('📊 Session Stats')
        
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.metric('Generated', st.session_state.generation_count)
        with stats_col2:
            st.metric('Colors', len(st.session_state.colors))
        
        if st.session_state.last_generation_time:
            st.caption(f'Last generated: {st.session_state.last_generation_time}')
        
        # Quick actions
        st.subheader('🚀 Quick Actions')
        if st.button('🎲 Random Prompt', use_container_width=True):
            # Select random category and prompt
            random_category = random.choice(list(SAMPLE_PROMPTS.keys()))
            random_prompt_key = random.choice(list(SAMPLE_PROMPTS[random_category].keys()))
            random_prompt = SAMPLE_PROMPTS[random_category][random_prompt_key]
            st.session_state.selected_prompt = random_prompt['prompt']
            st.session_state.selected_negative = random_prompt['negative']
            st.rerun()
        
        if st.button('🧹 Clear Prompts', use_container_width=True):
            st.session_state.selected_prompt = ''
            st.session_state.selected_negative = ''
            st.rerun()

with tab2:
    # Sample prompts interface
    st.subheader('📚 Professional Sample Prompts')
    st.write('Click on any card to use the prompt in your generation')
    
    # Category filter
    selected_categories = st.multiselect(
        'Filter by category:',
        options=list(SAMPLE_PROMPTS.keys()),
        default=list(SAMPLE_PROMPTS.keys())
    )
    
    # Search functionality
    search_term = st.text_input('🔍 Search prompts:', placeholder='Enter keywords...')
    
    st.divider()
    
    # Display sample prompts
    for category in selected_categories:
        if category in SAMPLE_PROMPTS:
            st.subheader(f'{category}')
            
            cols = st.columns(2)
            col_idx = 0
            
            for title, data in SAMPLE_PROMPTS[category].items():
                # Filter by search term
                if search_term and search_term.lower() not in data['prompt'].lower() and search_term.lower() not in title.lower():
                    continue
                
                with cols[col_idx % 2]:
                    render_sample_prompt_card(category, title, data, f"sample_{category}_{title}")
                
                col_idx += 1
            
            st.divider()

with tab3:
    # Inspiration and tips
    st.subheader('💡 Creative Inspiration & Tips')
    
    inspiration_tabs = st.tabs(['🎨 Styles', '🌈 Color Theory', '📐 Composition', '🔥 Trending'])
    
    with inspiration_tabs[0]:
        st.markdown("""
        ### 🎨 Popular Art Styles to Try
        
        **Photography Styles:**
        - `cinematic photography` - Movie-like dramatic shots
        - `documentary style` - Natural, candid moments
        - `macro photography` - Extreme close-up details
        - `long exposure` - Motion blur and light trails
        
        **Artistic Styles:**
        - `oil painting style` - Traditional painted look
        - `watercolor illustration` - Soft, flowing colors
        - `vector art` - Clean, geometric designs
        - `concept art` - Fantasy and sci-fi illustrations
        
        **Modern Styles:**
        - `minimalist design` - Clean, simple compositions
        - `maximalist` - Rich, detailed, busy compositions
        - `vaporwave aesthetic` - Retro 80s/90s neon style
        - `cottagecore` - Cozy, rural, handmade aesthetic
        """)
    
    with inspiration_tabs[1]:
        st.markdown("""
        ### 🌈 Color Theory in Prompts
        
        **Color Harmonies:**
        - `complementary colors` - Opposite colors (blue/orange)
        - `analogous colors` - Adjacent colors (blue/green)
        - `triadic colors` - Three evenly spaced colors
        - `monochromatic` - Different shades of one color
        
        **Color Moods:**
        - `warm colors` - Reds, oranges, yellows (energetic)
        - `cool colors` - Blues, greens, purples (calming)
        - `pastel palette` - Soft, muted colors (gentle)
        - `neon colors` - Bright, electric colors (vibrant)
        
        **Lighting Colors:**
        - `golden hour lighting` - Warm, yellow-orange light
        - `blue hour lighting` - Cool, blue twilight
        - `rim lighting` - Bright edge lighting
        - `ambient lighting` - Soft, even illumination
        """)
    
    with inspiration_tabs[2]:
        st.markdown("""
        ### 📐 Composition Techniques
        
        **Camera Angles:**
        - `bird's eye view` - Looking straight down
        - `worm's eye view` - Looking straight up
        - `dutch angle` - Tilted camera angle
        - `over the shoulder` - Behind subject's shoulder
        
        **Framing:**
        - `rule of thirds` - Subject on intersection lines
        - `center composition` - Subject in center
        - `leading lines` - Lines guide eye to subject
        - `frame within frame` - Natural borders
        
        **Depth:**
        - `shallow depth of field` - Blurred background
        - `deep focus` - Everything in sharp focus
        - `foreground, midground, background` - Layered depth
        - `atmospheric perspective` - Distant objects fade
        """)
    
    with inspiration_tabs[3]:
        st.markdown("""
        ### 🔥 Trending Keywords & Concepts
        
        **Current Aesthetics:**
        - `dark academia` - Scholarly, vintage vibes
        - `light academia` - Bright, intellectual aesthetic
        - `goblincore` - Earth tones, natural materials
        - `fairycore` - Magical, whimsical nature themes
        
        **Art Movements:**
        - `neo-expressionism` - Bold, emotional art
        - `digital brutalism` - Raw, unpolished digital art
        - `glitch art` - Digital distortion effects
        - `photorealism` - Extremely detailed realistic art
        
        **Technical Terms:**
        - `ray tracing` - Realistic lighting
        - `subsurface scattering` - Light through materials
        - `global illumination` - Realistic light bounce
        - `volumetric lighting` - Visible light beams
        """)

# Generation logic (same as before but updated to use the new prompt values)
if 'generate_btn' in locals() and generate_btn:
    st.session_state.improved_prompt = None
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner('🎨 Creating your masterpiece...'):
        status_text.text('Preparing generation parameters...')
        progress_bar.progress(20)
        
        colors = st.session_state.colors if st.session_state.generation_mode == 'color' else None
        
        status_text.text('Sending request to AI...')
        progress_bar.progress(40)
        
        result = ImageGenerator.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            quality=quality,
            cfg_scale=cfg_scale,
            seed=seed,
            number_of_images=number_of_images,
            use_improved_prompt=use_improved_prompt,
            colors=colors,
        )
        
        progress_bar.progress(80)
        status_text.text('Processing results...')
        
        # Update session state
        st.session_state.generated_images = result['image_paths']
        st.session_state.generation_count += 1
        st.session_state.last_generation_time = time.strftime('%H:%M:%S')
        
        if use_improved_prompt and 'improved_prompt' in result:
            st.session_state.improved_prompt = result['improved_prompt']
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        # Display result
        if result['status'] == 'success':
            st.success(f'✅ {result["message"]}')
        else:
            st.error(f'❌ Generation failed: {result["message"]}')

# Display improved prompt
if st.session_state.improved_prompt:
    st.markdown('### 🚀 AI-Enhanced Prompt')
    st.info(st.session_state.improved_prompt)

# Display generated images with modern gallery
if st.session_state.generated_images:
    st.markdown('### 🖼️ Generated Images')
    
    # Image gallery
    cols = st.columns(min(3, len(st.session_state.generated_images)))
    
    for i, image_path in enumerate(st.session_state.generated_images):
        with cols[i % len(cols)]:
            try:
                img = Image.open(image_path)
                st.image(img, caption=f'Generation {i+1}', use_container_width=True)
                
                # Download button with custom styling
                with open(image_path, 'rb') as file:
                    st.download_button(
                        label=f'💾 Download Image {i+1}',
                        data=file,
                        file_name=f'nova_canvas_{i+1}_{int(time.time())}.png',
                        mime='image/png',
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f'Error loading image {i+1}: {str(e)}')

# Footer
st.markdown('---')
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9em;">Made with ❤️ using Streamlit and Nova Canvas API | © 2024</p>',
    unsafe_allow_html=True
)

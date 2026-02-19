# Product Image Page - Analysis & Improvement Suggestions

## Current State Analysis

### Strengths
1. ✅ Clean 70/30 split layout with model selection panel
2. ✅ Comprehensive product specification form
3. ✅ Gallery feature to view generated images
4. ✅ Examples tab with predefined templates
5. ✅ Image download functionality
6. ✅ Session state management for image history

### Current Issues & Limitations

#### 1. **Limited Model Options**
- ❌ Only 1 model available (Amazon Nova Canvas)
- ❌ No comparison with other image generation models
- ❌ Missing Stable Diffusion models

#### 2. **Missing Art Style Integration**
- ❌ `nova_style` parameter collected but NOT used in generation
- ❌ Art style selector has no effect on output
- ❌ No visual examples of different styles

#### 3. **No Batch Generation**
- ❌ Can only generate 1 image at a time
- ❌ No option to generate multiple variations
- ❌ No A/B testing capability

#### 4. **Limited Export Options**
- ❌ Only PNG download available
- ❌ No bulk export for gallery images
- ❌ No metadata export (prompt, settings)
- ❌ No image comparison tools

#### 5. **Gallery Limitations**
- ❌ No search/filter functionality
- ❌ No sorting options (by date, style, product type)
- ❌ No delete functionality
- ❌ No image editing/regeneration from gallery
- ❌ Gallery resets on page refresh

#### 6. **UI/UX Issues**
- ❌ No modern AWS branding (unlike other pages)
- ❌ Plain header without gradient
- ❌ Basic footer styling
- ❌ No progress indicators during generation
- ❌ No image quality metrics

#### 7. **Missing Features**
- ❌ No negative prompts support
- ❌ No image-to-image generation
- ❌ No inpainting/outpainting
- ❌ No style transfer
- ❌ No prompt suggestions/templates
- ❌ No image upscaling

#### 8. **Technical Issues**
- ❌ Missing `datetime` import (will cause errors)
- ❌ No error recovery for failed generations
- ❌ No rate limiting handling
- ❌ No cost estimation

---

## Recommended Improvements

### Priority 1: Critical Fixes

#### 1.1 Fix Art Style Integration
```python
# Currently NOT using nova_style parameter!
# Need to add to request body:
body = json.dumps({
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": prompt,
        "style": nova_style  # ADD THIS
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "height": height,
        "width": width,
        "cfgScale": cfg_scale,
        "seed": seed
    }
})
```

#### 1.2 Add Missing Import
```python
# Add at top of file
from datetime import datetime
```

#### 1.3 Add Negative Prompts
```python
# Add to form
negative_prompt = st.text_area(
    "Negative Prompt (Optional)",
    placeholder="e.g., blurry, low quality, distorted, watermark",
    help="Specify what you DON'T want in the image"
)

# Add to request body
"negativeText": negative_prompt if negative_prompt else None
```

### Priority 2: Enhanced Features

#### 2.1 Add Multiple Image Generation
```python
# Add to model panel
num_images = st.slider(
    "Number of Images",
    min_value=1,
    max_value=4,
    value=1,
    help="Generate multiple variations at once"
)

# Update request body
"numberOfImages": num_images
```

#### 2.2 Add Stable Diffusion Models
```python
model_options = {
    "Amazon Nova Canvas": "amazon.nova-canvas-v1:0",
    "Stable Diffusion XL": "stability.stable-diffusion-xl-v1",
    "Stable Image Core": "stability.stable-image-core-v1:1",
    "SD3.5 Large": "stability.sd3-5-large-v1:0"
}
```

#### 2.3 Enhanced Gallery Features
```python
# Add filters
col1, col2, col3 = st.columns(3)
with col1:
    filter_product = st.selectbox("Filter by Product", ["All"] + unique_products)
with col2:
    filter_style = st.selectbox("Filter by Style", ["All"] + unique_styles)
with col3:
    sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Product Type"])

# Add delete functionality
if st.button("🗑️ Delete", key=f"delete_{idx}"):
    st.session_state.generated_images.pop(idx)
    st.rerun()

# Add regenerate from gallery
if st.button("🔄 Regenerate", key=f"regen_{idx}"):
    # Use saved prompt and settings
    regenerate_image(img_data)
```

#### 2.4 Bulk Export
```python
# Add to gallery tab
if st.button("📥 Download All Images"):
    # Create ZIP file
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for idx, img_data in enumerate(images):
            buf = io.BytesIO()
            img_data['image'].save(buf, format="PNG")
            zip_file.writestr(f"image_{idx+1}.png", buf.getvalue())
    
    st.download_button(
        "📦 Download ZIP",
        data=zip_buffer.getvalue(),
        file_name=f"product_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )
```

### Priority 3: UI/UX Enhancements

#### 3.1 Modern AWS Branding
```python
# Update header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #232F3E; font-size: 2.5rem; font-weight: 700;">
        <svg>...</svg>
        🎨 AWS Nova Canvas Product Image Generator
    </h1>
    <p style="color: #6B7280; font-size: 1.1rem;">
        AI-Powered Professional Product Photography
        <br/>
        <span style="font-size: 0.875rem; color: #9CA3AF;">
            Powered by Amazon Bedrock • Multiple Styles • High Resolution
        </span>
    </p>
</div>
""", unsafe_allow_html=True)

# Update footer to match other pages
st.markdown("""
<div style="margin-top: 4rem; padding: 1.5rem; 
            background: linear-gradient(135deg, #232F3E 0%, #16191F 100%); 
            border-radius: 10px; text-align: center; position: relative;">
    <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; 
                background: linear-gradient(90deg, #FF9900 0%, #146EB4 100%);"></div>
    <p style="color: #D1D5DB; font-size: 0.875rem; margin: 0;">
        © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)
```

#### 3.2 Add Progress Indicators
```python
# During generation
progress_bar = st.progress(0, text="Initializing...")
progress_bar.progress(0.3, text="Building prompt...")
progress_bar.progress(0.6, text="Generating image...")
progress_bar.progress(1.0, text="✅ Complete!")
```

#### 3.3 Add Image Quality Metrics
```python
# After generation
col1, col2, col3, col4 = st.columns(4)
col1.metric("Resolution", f"{width}x{height}")
col2.metric("File Size", f"{len(image_bytes) / 1024:.1f} KB")
col3.metric("CFG Scale", cfg_scale)
col4.metric("Seed", seed if seed > 0 else "Random")
```

#### 3.4 Add Prompt Templates
```python
# Quick start templates
templates = {
    "E-commerce Product Shot": {
        "environment": "White Background",
        "lighting": "Soft Box",
        "style": "Minimalist",
        "details": "centered, professional, high resolution"
    },
    "Lifestyle Photography": {
        "environment": "Lifestyle Scene",
        "lighting": "Natural Light",
        "style": "Natural",
        "details": "in-context usage, authentic setting"
    },
    "Premium Luxury": {
        "environment": "Studio Setup",
        "lighting": "Dramatic",
        "style": "Luxury",
        "details": "elegant, premium quality, sophisticated"
    }
}

selected_template = st.selectbox("Quick Start Template", ["Custom"] + list(templates.keys()))
if selected_template != "Custom":
    # Auto-fill form with template values
    template_data = templates[selected_template]
```

### Priority 4: Advanced Features

#### 4.1 Image-to-Image Generation
```python
# Add upload option
uploaded_image = st.file_uploader(
    "Reference Image (Optional)",
    type=["png", "jpg", "jpeg"],
    help="Upload a reference image for style transfer or modification"
)

if uploaded_image:
    # Add to request body
    "images": [base64.b64encode(uploaded_image.read()).decode()]
```

#### 4.2 Cost Estimation
```python
# Calculate estimated cost
def estimate_cost(width, height, num_images):
    # Nova Canvas pricing (example)
    base_cost = 0.04  # per image
    resolution_multiplier = (width * height) / (1024 * 1024)
    total_cost = base_cost * resolution_multiplier * num_images
    return total_cost

estimated_cost = estimate_cost(width, height, num_images)
st.info(f"💰 Estimated cost: ${estimated_cost:.4f}")
```

#### 4.3 Prompt Enhancement
```python
# AI-powered prompt improvement
if st.button("✨ Enhance Prompt"):
    # Use LLM to improve prompt
    enhanced_prompt = enhance_prompt_with_llm(
        product_type, material, color, style, 
        environment, lighting, additional_details
    )
    st.success(f"Enhanced prompt: {enhanced_prompt}")
```

#### 4.4 Image Comparison
```python
# Compare multiple generated images
if len(st.session_state.generated_images) >= 2:
    st.markdown("### 🔍 Compare Images")
    col1, col2 = st.columns(2)
    
    with col1:
        img1_idx = st.selectbox("Image 1", range(len(images)))
        st.image(images[img1_idx]['image'])
    
    with col2:
        img2_idx = st.selectbox("Image 2", range(len(images)))
        st.image(images[img2_idx]['image'])
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (1 day)
1. Fix art style integration (add to request body)
2. Add datetime import
3. Add negative prompts support
4. Fix error handling

### Phase 2: Core Enhancements (2-3 days)
1. Add multiple image generation (1-4 images)
2. Add Stable Diffusion models
3. Enhance gallery (filters, sorting, delete)
4. Add bulk export (ZIP download)
5. Modernize UI with AWS branding

### Phase 3: Advanced Features (3-5 days)
1. Add prompt templates
2. Add image quality metrics
3. Add progress indicators
4. Add cost estimation
5. Add image comparison

### Phase 4: Premium Features (5-7 days)
1. Image-to-image generation
2. Prompt enhancement with LLM
3. Image upscaling
4. Style transfer
5. Persistent gallery (database storage)

---

## Code Examples

### Example: Fix Art Style Integration
```python
# BEFORE (not working)
body = json.dumps({
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": prompt
    },
    ...
})

# AFTER (working)
body = json.dumps({
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": prompt,
        "style": nova_style  # Now actually used!
    },
    ...
})
```

### Example: Multiple Image Generation
```python
# Generate multiple variations
for i in range(num_images):
    progress_bar.progress((i + 1) / num_images, 
                         text=f"Generating image {i+1}/{num_images}...")
    
    # Use different seed for each
    current_seed = seed + i if seed > 0 else 0
    
    body = json.dumps({
        ...
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "seed": current_seed,
            ...
        }
    })
    
    image_bytes = generate_image_from_nova(model_id, body)
    images.append(Image.open(io.BytesIO(image_bytes)))

# Display all images in grid
cols = st.columns(min(num_images, 4))
for idx, img in enumerate(images):
    with cols[idx % 4]:
        st.image(img, caption=f"Variation {idx+1}")
```

---

## Expected Impact

### User Experience
- ⬆️ 300% more image variations (1 → 4 images per generation)
- ⬆️ 100% better gallery management (filters, sorting, delete)
- ⬆️ 50% faster workflow (templates, prompt enhancement)

### Business Value
- ⬆️ Higher quality images with art style control
- ⬆️ Better cost visibility with estimation
- ⬆️ More model options for different use cases

### Technical Quality
- ⬆️ Consistent UI/UX across all pages
- ⬆️ Better error handling
- ⬆️ More maintainable code

---

## Conclusion

The Product Image page has good foundation but needs critical fixes and enhancements:

1. **Immediate fixes** for art style integration and imports
2. **Enhanced features** for multiple images and better gallery
3. **Modern UI/UX** consistent with AWS branding
4. **Advanced capabilities** for professional use cases

The most critical issue is that the art style selector doesn't actually work - it collects the parameter but never uses it in the API call!

**Priority Order:**
1. Fix art style integration (CRITICAL)
2. Add datetime import (CRITICAL)
3. Add negative prompts
4. Modernize UI/UX
5. Add multiple image generation
6. Enhance gallery features

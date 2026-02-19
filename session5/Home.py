import streamlit as st
import utils.common as common
import utils.authenticate as authenticate
import utils.styles as styles

st.set_page_config(
    page_title="Session 5 - AI-Powered Business Solutions",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css():
    """Load custom CSS for modern UI/UX"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #FAFAFA 0%, #F3F4F6 100%);
    }
    
    .hero-section {
        background: linear-gradient(135deg, #232F3E 0%, #16191F 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #FF9900 0%, #146EB4 100%);
    }
    
    .hero-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        line-height: 1.2;
        text-align: center;
    }
    
    .hero-subtitle {
        color: #D1D5DB;
        font-size: 1.25rem;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 2rem;
    }
    
    .badge {
        background: rgba(255, 153, 0, 0.2);
        color: #FF9900;
        padding: 0.5rem 1.25rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        font-weight: 600;
        border: 2px solid rgba(255, 153, 0, 0.3);
    }
    
    .objective-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left: 5px solid #FF9900;
    }
    
    .objective-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
    }
    
    .objective-number {
        display: inline-block;
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #FF9900 0%, #EC7211 100%);
        color: white;
        border-radius: 12px;
        text-align: center;
        line-height: 48px;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(255, 153, 0, 0.3);
    }
    
    .objective-title {
        color: #232F3E;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .objective-description {
        color: #6B7280;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
    }
    
    .feature-item {
        padding: 0.75rem 0;
        color: #4B5563;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .feature-icon {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    
    .tech-stack {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border-left: 5px solid #146EB4;
    }
    
    .tech-title {
        color: #232F3E;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .tech-item {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .tech-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .cta-section {
        background: linear-gradient(135deg, #FF9900 0%, #EC7211 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 3rem 0;
        box-shadow: 0 12px 32px rgba(255, 153, 0, 0.3);
    }
    
    .cta-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .cta-text {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF9900 0%, #EC7211 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #6B7280;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .footer-section {
        margin-top: 4rem;
        padding: 2rem;
        background: linear-gradient(135deg, #232F3E 0%, #16191F 100%);
        border-radius: 16px;
        text-align: center;
        color: white;
    }
    
    .footer-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #FF9900 0%, #146EB4 100%);
    }
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1rem;
        }
        
        .objective-card {
            padding: 1.5rem;
        }
        
        .tech-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    # Initialize session state
    common.initialize_session_state()
    styles.load_css()
    load_custom_css()

    # Render the sidebar
    with st.sidebar:
        common.render_sidebar()

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title" style="
            color: white;
            font-size: 3.5rem;
            font-weight: 900;
            text-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.3);
            letter-spacing: -0.02em;
            z-index: 10;
            position: relative;
        ">
            AI-Powered Business Solutions
        </h1>
        <p class="hero-subtitle" style="
            color: #E5E7EB;
            font-size: 1.35rem;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            z-index: 10;
            position: relative;
        ">
            Leveraging Amazon Bedrock and Generative AI to Transform Business Operations
        </p>
        <div class="badge-container">
            <span class="badge">🤖 Amazon Bedrock</span>
            <span class="badge">📊 Market Research</span>
            <span class="badge">🎨 Content Generation</span>
            <span class="badge">💬 AI Agents</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);">
        <h2 style="color: #232F3E; margin-bottom: 1rem;">📖 Overview</h2>
        <p style="color: #4B5563; font-size: 1.05rem; line-height: 1.8;">
            Session 5 demonstrates practical applications of <strong>Amazon Bedrock</strong> and <strong>Generative AI</strong> 
            in real-world business scenarios. Through four comprehensive modules, you'll explore how AI can revolutionize 
            market research, content creation, product visualization, and customer service.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Objectives
    st.markdown("""
    <h2 style="color: #232F3E; font-size: 2rem; font-weight: 700; margin: 3rem 0 2rem 0; text-align: center;">
        🎯 Learning Objectives
    </h2>
    """, unsafe_allow_html=True)
    
    # Objective 1: Market Research
    st.markdown("""
    <div class="objective-card">
        <div class="objective-number">1</div>
        <h3 class="objective-title">📊 AI-Powered Market Research Analysis</h3>
        <p class="objective-description">
            Transform raw market research documents into actionable business intelligence using advanced 
            RAG (Retrieval Augmented Generation) technology and Amazon Bedrock.
        </p>
        <ul class="feature-list">
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Document Processing:</strong> Upload and analyze PDF, DOCX, and TXT files with intelligent chunking</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Market Analysis:</strong> Extract opportunities, trends, customer segments, and competitive landscape</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Feasibility Scoring:</strong> AI-generated feasibility scores with detailed justifications</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Strategic Recommendations:</strong> Product positioning, pricing strategies, and go-to-market plans</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Interactive Chat:</strong> Q&A interface for deep-dive analysis and clarifications</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Scenario Comparison:</strong> Save and compare multiple market entry scenarios</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Export Capabilities:</strong> Download results in JSON, CSV, or text format</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Objective 2: Product Description
    st.markdown("""
    <div class="objective-card">
        <div class="objective-number">2</div>
        <h3 class="objective-title">✍️ Automated Product Description Generation</h3>
        <p class="objective-description">
            Create compelling, SEO-optimized product descriptions at scale using generative AI, 
            saving time and ensuring consistency across your product catalog.
        </p>
        <ul class="feature-list">
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Multi-Model Support:</strong> Choose from Claude, Nova, Llama, and other foundation models</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Tone Customization:</strong> Professional, casual, enthusiastic, or technical writing styles</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Length Control:</strong> Short, medium, or long descriptions based on your needs</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Feature Highlighting:</strong> Automatically emphasize key product features and benefits</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>SEO Optimization:</strong> Keyword integration for better search visibility</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Batch Processing:</strong> Generate descriptions for multiple products efficiently</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Objective 3: Product Image
    st.markdown("""
    <div class="objective-card">
        <div class="objective-number">3</div>
        <h3 class="objective-title">🎨 AI-Generated Product Imagery</h3>
        <p class="objective-description">
            Create stunning, professional product images using Amazon Bedrock's image generation models, 
            perfect for marketing materials, e-commerce, and social media.
        </p>
        <ul class="feature-list">
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Stable Diffusion XL:</strong> High-quality image generation with advanced prompting</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Style Presets:</strong> Photorealistic, artistic, minimalist, and more</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Negative Prompts:</strong> Fine-tune results by specifying what to avoid</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Parameter Control:</strong> Adjust quality, steps, and guidance for perfect results</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Multiple Variations:</strong> Generate multiple options to choose from</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Download & Share:</strong> Export high-resolution images for immediate use</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Objective 4: Sales Assistant
    st.markdown("""
    <div class="objective-card">
        <div class="objective-number">4</div>
        <h3 class="objective-title">💬 Intelligent Sales Assistant Agent</h3>
        <p class="objective-description">
            Deploy a conversational AI agent powered by Strands Agents SDK that provides personalized 
            shopping experiences with memory, preferences, and intelligent recommendations.
        </p>
        <ul class="feature-list">
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>User Profile Management:</strong> Create and maintain customer profiles across sessions</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Preference Tracking:</strong> Remember brands, sizes, colors, and shopping preferences</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Intelligent Search:</strong> Natural language queries for product discovery</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Personalized Recommendations:</strong> Activity-based and preference-based suggestions</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Conversational Interface:</strong> Natural, friendly dialogue with context awareness</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Multi-Tool Integration:</strong> Search, filter, and recommend using specialized tools</span>
            </li>
            <li class="feature-item">
                <div class="feature-icon">✓</div>
                <span><strong>Session Memory:</strong> Continuous conversation with chat history</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown("""
    <div class="tech-stack">
        <h3 class="tech-title">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" style="display: inline-block;">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#146EB4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="#146EB4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="#146EB4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Technology Stack
        </h3>
        <p style="color: #4B5563; margin-bottom: 1.5rem;">
            Built on enterprise-grade AWS services and modern AI frameworks
        </p>
        <div class="tech-grid">
            <div class="tech-item">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">☁️</div>
                <strong>Amazon Bedrock</strong>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">Foundation Models</div>
            </div>
            <div class="tech-item">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                <strong>Claude 3.5</strong>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">Anthropic AI</div>
            </div>
            <div class="tech-item">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                <strong>Amazon Nova</strong>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">AWS Native Models</div>
            </div>
            <div class="tech-item">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔗</div>
                <strong>LangChain</strong>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">RAG Framework</div>
            </div>
            <div class="tech-item">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🗄️</div>
                <strong>FAISS</strong>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">Vector Database</div>
            </div>
            <div class="tech-item">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎨</div>
                <strong>Stable Diffusion XL</strong>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">Image Generation</div>
            </div>
            <div class="tech-item">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                <strong>Strands Agents</strong>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">Agent Framework</div>
            </div>
            <div class="tech-item">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <strong>Streamlit</strong>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">Web Framework</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Stats
    st.markdown("""
    <h2 style="color: #232F3E; font-size: 2rem; font-weight: 700; margin: 3rem 0 2rem 0; text-align: center;">
        📈 Platform Capabilities
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">15+</div>
            <div class="stat-label">AI Models Available</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">4</div>
            <div class="stat-label">Business Applications</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">RAG</div>
            <div class="stat-label">Advanced Technology</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">100%</div>
            <div class="stat-label">AWS Powered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
    <div class="cta-section">
        <h2 class="cta-title">Ready to Get Started?</h2>
        <p class="cta-text">
            Explore each module using the navigation menu on the left.<br/>
            Start with Market Research to see AI-powered analysis in action!
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <div style="background: rgba(255, 255, 255, 0.2); padding: 0.75rem 1.5rem; border-radius: 2rem; color: white; font-weight: 600;">
                📊 Market Research
            </div>
            <div style="background: rgba(255, 255, 255, 0.2); padding: 0.75rem 1.5rem; border-radius: 2rem; color: white; font-weight: 600;">
                ✍️ Product Description
            </div>
            <div style="background: rgba(255, 255, 255, 0.2); padding: 0.75rem 1.5rem; border-radius: 2rem; color: white; font-weight: 600;">
                🎨 Product Image
            </div>
            <div style="background: rgba(255, 255, 255, 0.2); padding: 0.75rem 1.5rem; border-radius: 2rem; color: white; font-weight: 600;">
                💬 Sales Assistant
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer-section" style="position: relative;">
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
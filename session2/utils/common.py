import streamlit as st
import streamlit.components.v1 as components
from typing import Optional
import time
import hashlib
import uuid
from datetime import datetime
from typing import Optional

def reset_session():
    """Reset the session state"""
    for key in st.session_state.keys():
        if key not in ["authenticated", "user_cognito_groups", "auth_code","user_info"]:
            del st.session_state[key]  
    
def render_sidebar():
    """Render the sidebar with session information and reset button"""
    st.markdown("#### 🔑 Session Info")
    if 'auth_code' not in st.session_state:
        st.caption(f"**Session ID:** {st.session_state.session_id[:8]}")
    else:
        st.caption(f"**Session ID:** {st.session_state['auth_code'][:8]}")

    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()
        st.success("Session has been reset successfully!")
        st.rerun()  # Force a rerun to refresh the page

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        

def initialize_mermaid() -> None:
    """Initialize Mermaid library once per Streamlit session.
    
    This should be called once at the start of your Streamlit app.
    The initialization is cached per session to avoid reloading.
    """
    # Use session state to ensure we only initialize once per session
    if 'mermaid_initialized' not in st.session_state:
        st.session_state.mermaid_initialized = True

def _estimate_diagram_height(mermaid_code: str) -> int:
    """Quick estimation for initial render when height is auto."""
    lines = [line.strip() for line in mermaid_code.strip().split('\n') if line.strip()]
    
    # Count meaningful lines (exclude comments and styling)
    content_lines = [line for line in lines 
                    if not line.startswith('%%') 
                    and not line.startswith('classDef') 
                    and not line.startswith('class')]
    
    diagram_type = lines[0].lower() if lines else ""
    node_count = len([line for line in content_lines if '-->' in line or '{' in line or '->' in line])
    
    if 'flowchart' in diagram_type and ('td' in diagram_type or 'tb' in diagram_type):
        return max(300, min(800, 200 + node_count * 60))
    else:
        return max(250, min(600, 150 + node_count * 50))

def _generate_unique_id(code: str, key: Optional[str] = None) -> str:
    """Generate a unique ID for the mermaid diagram."""
    if key:
        return f"mermaid_{key}_{int(time.time() * 1000) % 10000}"
    else:
        # Use hash of code + timestamp for uniqueness
        code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        timestamp = int(time.time() * 1000) % 10000
        return f"mermaid_{code_hash}_{timestamp}"

def mermaid(
    code: str, 
    width: str = "auto", 
    height: str = "auto", 
    pan: bool = True, 
    zoom: bool = True, 
    show_controls: bool = True, 
    key: Optional[str] = None
) -> None:
    """Render a Mermaid diagram in Streamlit with configurable dimensions and interactive controls.
    
    Args:
        code: The Mermaid diagram code to render
        width: Width of the container ("auto", "100%", "500px", etc.)
        height: Height of the container ("auto", "100%", "400px", etc.)
        pan: Enable panning functionality
        zoom: Enable zoom functionality
        show_controls: Show zoom control buttons
        key: Optional unique key for the component
    """
    
    # Auto-initialize if not done
    if 'mermaid_initialized' not in st.session_state:
        initialize_mermaid()
    
    # Generate unique ID
    unique_id = _generate_unique_id(code, key)
    
    # Handle dimensions
    container_width = width if width != "auto" else "100%"
    
    # Calculate height
    if height == "auto":
        calculated_height = _estimate_diagram_height(code)
        container_height = f"{calculated_height + (60 if show_controls else 20)}px"
        content_height = calculated_height
    else:
        height_str = str(height)
        if height_str.endswith('px'):
            content_height = int(height_str.replace('px', ''))
        elif height_str.endswith('%'):
            content_height = 400
        else:
            try:
                content_height = int(height_str)
                height_str = f"{content_height}px"
            except ValueError:
                content_height = 400
                height_str = "400px"
        container_height = height_str
    
    # Escape the mermaid code for JavaScript
    escaped_code = code.replace('`', '\\`').replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '')
    
    # Build styles and controls
    container_cursor = "grab" if pan else "default"
    container_overflow = "auto" if pan else "hidden"
    zoom_controls_display = "flex" if show_controls and zoom else "none"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    </head>
    <body>
        <div class="mermaid-wrapper" id="wrapper-{unique_id}">
            <!-- Zoom Controls -->
            <div class="zoom-controls" id="zoom-controls-{unique_id}" style="display: {zoom_controls_display};">
                <button id="zoom-in-{unique_id}" class="zoom-btn" title="Zoom In">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <path d="m21 21-4.35-4.35"></path>
                        <line x1="8" y1="11" x2="14" y2="11"></line>
                        <line x1="11" y1="8" x2="11" y2="14"></line>
                    </svg>
                </button>
                <button id="zoom-out-{unique_id}" class="zoom-btn" title="Zoom Out">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <path d="m21 21-4.35-4.35"></path>
                        <line x1="8" y1="11" x2="14" y2="11"></line>
                    </svg>
                </button>
                <button id="zoom-reset-{unique_id}" class="zoom-btn" title="Reset Zoom">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 3l18 18"></path>
                        <path d="M9 9h6v6"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                </button>
                <span id="zoom-level-{unique_id}" class="zoom-level">100%</span>
            </div>
            
            <!-- Mermaid Container -->
            <div class="mermaid-container" id="container-{unique_id}">
                <div class="mermaid-content" id="content-{unique_id}">
                    <div class="mermaid" id="diagram-{unique_id}">
                        {code}
                    </div>
                </div>
            </div>
        </div>

        <style>
            .mermaid-wrapper {{
                position: relative;
                width: {container_width};
                height: {container_height};
                border: 1px solid #e1e5e9;
                border-radius: 6px;
                background: #fafafa;
                font-family: Arial, sans-serif;
            }}
            
            .zoom-controls {{
                position: absolute;
                top: 10px;
                right: 10px;
                gap: 5px;
                align-items: center;
                background: rgba(255, 255, 255, 0.9);
                padding: 5px 8px;
                border-radius: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                z-index: 1000;
                backdrop-filter: blur(4px);
            }}
            
            .zoom-btn {{
                background: #fff;
                border: 1px solid #ddd;
                border-radius: 4px;
                width: 28px;
                height: 28px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s ease;
                color: #555;
            }}
            
            .zoom-btn:hover:not(:disabled) {{
                background: #f0f0f0;
                border-color: #bbb;
                transform: translateY(-1px);
            }}
            
            .zoom-btn:active {{
                transform: translateY(0);
            }}
            
            .zoom-btn:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}
            
            .zoom-level {{
                font-size: 12px;
                color: #666;
                font-weight: 500;
                min-width: 35px;
                text-align: center;
                margin-left: 5px;
            }}
            
            .mermaid-container {{
                height: 100%;
                padding: 20px;
                overflow: {container_overflow};
                display: flex;
                justify-content: center;
                align-items: flex-start;
                position: relative;
                cursor: {container_cursor};
            }}
            
            .mermaid-container:active {{
                cursor: {"grabbing" if pan else container_cursor};
            }}
            
            .mermaid-content {{
                transform-origin: top center;
                transition: transform 0.3s ease;
                min-width: 100%;
                display: flex;
                justify-content: center;
            }}
            
            .mermaid {{
                width: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            
            /* Scrollbar styling */
            .mermaid-container::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            .mermaid-container::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 4px;
            }}
            
            .mermaid-container::-webkit-scrollbar-thumb {{
                background: #c1c1c1;
                border-radius: 4px;
            }}
            
            .mermaid-container::-webkit-scrollbar-thumb:hover {{
                background: #a8a8a8;
            }}
            
            /* Responsive design */
            @media (max-width: 640px) {{
                .zoom-controls {{
                    top: 5px;
                    right: 5px;
                    padding: 3px 6px;
                }}
                
                .zoom-btn {{
                    width: 24px;
                    height: 24px;
                }}
                
                .zoom-level {{
                    font-size: 11px;
                }}
            }}
        </style>

        <script>
            // Initialize Mermaid
            mermaid.initialize({{ 
                startOnLoad: false,
                theme: 'default',
                flowchart: {{ useMaxWidth: false, htmlLabels: true }},
                sequence: {{ useMaxWidth: false }},
                gantt: {{ useMaxWidth: false }},
                journey: {{ useMaxWidth: false }},
                timeline: {{ useMaxWidth: false }}
            }});
            
            // Function to render the diagram
            async function renderDiagram() {{
                try {{
                    const diagramElement = document.getElementById('diagram-{unique_id}');
                    if (!diagramElement) {{
                        console.error('Diagram element not found');
                        return;
                    }}
                    
                    // Render the diagram
                    await mermaid.run({{
                        nodes: [diagramElement]
                    }});
                    
                    console.log('Diagram rendered successfully');
                    
                    // Initialize interactions after rendering
                    initializeInteractions();
                    
                }} catch (error) {{
                    console.error('Error rendering Mermaid diagram:', error);
                    const diagramElement = document.getElementById('diagram-{unique_id}');
                    if (diagramElement) {{
                        diagramElement.innerHTML = '<div style="color: red; padding: 20px;">Error rendering diagram: ' + error.message + '</div>';
                    }}
                }}
            }}
            
            function initializeInteractions() {{
                const container = document.getElementById('container-{unique_id}');
                const content = document.getElementById('content-{unique_id}');
                
                if (!container || !content) return;
                
                // Zoom functionality
                if ({str(zoom).lower()}) {{
                    const zoomInBtn = document.getElementById('zoom-in-{unique_id}');
                    const zoomOutBtn = document.getElementById('zoom-out-{unique_id}');
                    const zoomResetBtn = document.getElementById('zoom-reset-{unique_id}');
                    const zoomLevelSpan = document.getElementById('zoom-level-{unique_id}');
                    
                    let currentZoom = 1.0;
                    const minZoom = 0.5;
                    const maxZoom = 3.0;
                    const zoomStep = 0.2;
                    
                    function updateZoom(newZoom) {{
                        currentZoom = Math.max(minZoom, Math.min(maxZoom, newZoom));
                        content.style.transform = 'scale(' + currentZoom + ')';
                        
                        if ({str(show_controls).lower()} && zoomLevelSpan) {{
                            zoomLevelSpan.textContent = Math.round(currentZoom * 100) + '%';
                        }}
                        
                        if ({str(show_controls).lower()}) {{
                            if (zoomInBtn) zoomInBtn.disabled = currentZoom >= maxZoom;
                            if (zoomOutBtn) zoomOutBtn.disabled = currentZoom <= minZoom;
                        }}
                    }}
                    
                    // Zoom control event listeners
                    if ({str(show_controls).lower()}) {{
                        if (zoomInBtn) {{
                            zoomInBtn.addEventListener('click', function() {{
                                updateZoom(currentZoom + zoomStep);
                            }});
                        }}
                        if (zoomOutBtn) {{
                            zoomOutBtn.addEventListener('click', function() {{
                                updateZoom(currentZoom - zoomStep);
                            }});
                        }}
                        if (zoomResetBtn) {{
                            zoomResetBtn.addEventListener('click', function() {{
                                updateZoom(1.0);
                            }});
                        }}
                    }}
                    
                    // Mouse wheel zoom
                    container.addEventListener('wheel', function(e) {{
                        if (e.ctrlKey || e.metaKey) {{
                            e.preventDefault();
                            const delta = e.deltaY > 0 ? -zoomStep : zoomStep;
                            updateZoom(currentZoom + delta);
                        }}
                    }});
                    
                    // Initialize zoom
                    updateZoom(1.0);
                    
                    // Make container focusable for keyboard events
                    container.setAttribute('tabindex', '0');
                }}
                
                // Pan functionality
                if ({str(pan).lower()}) {{
                    let isPanning = false;
                    let startX, startY, scrollLeft, scrollTop;
                    
                    container.addEventListener('mousedown', function(e) {{
                        if (e.button === 0) {{
                            isPanning = true;
                            startX = e.pageX - container.offsetLeft;
                            startY = e.pageY - container.offsetTop;
                            scrollLeft = container.scrollLeft;
                            scrollTop = container.scrollTop;
                            container.style.cursor = 'grabbing';
                        }}
                    }});
                    
                    container.addEventListener('mouseleave', function() {{
                        isPanning = false;
                        container.style.cursor = 'grab';
                    }});
                    
                    container.addEventListener('mouseup', function() {{
                        isPanning = false;
                        container.style.cursor = 'grab';
                    }});
                    
                    container.addEventListener('mousemove', function(e) {{
                        if (!isPanning) return;
                        e.preventDefault();
                        const x = e.pageX - container.offsetLeft;
                        const y = e.pageY - container.offsetTop;
                        const walkX = (x - startX) * 1;
                        const walkY = (y - startY) * 1;
                        container.scrollLeft = scrollLeft - walkX;
                        container.scrollTop = scrollTop - walkY;
                    }});
                }}
            }}
            
            // Wait for DOM to be ready and then render
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', renderDiagram);
            }} else {{
                // DOM is already ready, render immediately
                setTimeout(renderDiagram, 100);
            }}
        </script>
    </body>
    </html>
    """
    
    components.html(
        html_content,
        height=content_height + (80 if show_controls else 40),
    )

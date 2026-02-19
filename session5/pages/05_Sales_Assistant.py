"""
Strands Agent Shoe Department Assistant

This application provides a chat interface using Strands Agents SDK
with a local SQLite database for shoe inventory management.

Features:
- Conversational UI with Strands Agent
- Local SQLite database for shoe inventory
- Multiple search and recommendation tools
- Chat history tracking
- Session management
"""

import streamlit as st
import uuid
import logging
from datetime import datetime
from typing import List, Dict
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import utils.common as common
import utils.authenticate as authenticate
from utils.styles import load_css

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Shoe Department Assistant",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ShoeAssistantAgent:
    """Strands Agent for shoe department assistance"""
    
    def __init__(self):
        """Initialize the Strands agent with shoe tools"""
        try:
            from strands import Agent
            from utils.shoe_tools import (
                search_shoes_by_brand,
                search_shoes_by_category,
                search_shoes_by_price_range,
                get_shoe_details,
                list_all_brands,
                list_all_categories,
                get_recommendations_for_activity,
                find_or_create_user,
                save_user_preference,
                get_user_preferences
            )
            
            # Create agent with all shoe tools
            self.agent = Agent(
                tools=[
                    find_or_create_user,
                    save_user_preference,
                    get_user_preferences,
                    search_shoes_by_brand,
                    search_shoes_by_category,
                    search_shoes_by_price_range,
                    get_shoe_details,
                    list_all_brands,
                    list_all_categories,
                    get_recommendations_for_activity
                ],
                system_prompt="""You are a friendly and knowledgeable shoe department assistant with personalized service capabilities.

Your role is to help customers find the perfect shoes based on their needs, preferences, and budget.

IMPORTANT - User Management Flow:
1. When a customer first interacts with you, politely ask for their first and last name
2. Use find_or_create_user tool to check if they're a returning customer
3. If they're a returning customer:
   - Welcome them back warmly
   - Highlight their saved preferences (brands, categories, sizes, colors, activities)
   - Ask if they want to shop based on their preferences or try something new
4. If they're a new customer:
   - Welcome them and let them know you'll remember their preferences for next time
5. Throughout the conversation, save their preferences using save_user_preference:
   - When they mention a brand they like: save as preference_type="brand"
   - When they mention a category: save as preference_type="category"
   - When they mention their size: save as preference_type="size"
   - When they mention color preferences: save as preference_type="color"
   - When they mention activities: save as preference_type="activity"
   - When they mention price range: save as preference_type="price_range"

Guidelines:
- Always be helpful, friendly, and professional
- Ask clarifying questions if the customer's needs are unclear
- Use the available tools to search the inventory and provide accurate information
- Recommend shoes based on the customer's activity, style preferences, and budget
- Provide details about sizes, colors, features, and prices
- If a customer asks about a specific brand or category, use the appropriate search tool
- When recommending shoes, explain why they're a good fit for the customer's needs
- Always mention available sizes and colors when discussing specific shoes
- Save preferences naturally during conversation - don't make it feel like a form
- If you don't have exact information, use the tools to search rather than guessing

Remember: Your goal is to provide personalized service and help customers find shoes they'll love!"""
            )
            
            logger.info("Successfully initialized Strands agent with user management")
            
        except ImportError as e:
            logger.error(f"Failed to import Strands: {e}")
            st.error("""
            ⚠️ Strands Agents SDK is not installed.
            
            Please install it with:
            ```bash
            pip install strands-agents
            ```
            """)
            self.agent = None
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            st.error(f"Failed to initialize agent: {e}")
            self.agent = None
    
    def get_response(self, user_input: str) -> str:
        """
        Get a response from the agent
        
        Args:
            user_input: User's message
            
        Returns:
            Agent's response text
        """
        if not self.agent:
            return "⚠️ Agent is not available. Please check the installation."
        
        try:
            logger.info(f"Processing user input: '{user_input[:50]}...'")
            result = self.agent(user_input)
            
            # Handle different response types
            # If result is a string, return it directly
            if isinstance(result, str):
                return result
            
            # Extract text from AgentResult object
            if hasattr(result, 'message'):
                message = result.message
                
                # Handle string message (most common case)
                if isinstance(message, str):
                    return message
                
                # Handle dict message
                if isinstance(message, dict):
                    # Try to get content array
                    content = message.get('content', [])
                    if content and isinstance(content, list):
                        # Get text from content blocks
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and 'text' in block:
                                text_parts.append(block['text'])
                            elif isinstance(block, str):
                                text_parts.append(block)
                        if text_parts:
                            return '\n'.join(text_parts)
                    
                    # Try to get text directly from message dict
                    if 'text' in message:
                        return message['text']
            
            # Try to get text attribute directly
            if hasattr(result, 'text'):
                return result.text
            
            # Fallback to string representation
            result_str = str(result)
            logger.warning(f"Unexpected result type: {type(result)}, converted to string")
            return result_str
            
        except Exception as e:
            logger.error(f"Error getting response from agent: {e}", exc_info=True)
            return f"⚠️ Something went wrong: {str(e)}"


def initialize_session() -> None:
    """Initialize session state variables if they don't exist"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"New session created: {st.session_state.session_id}")
    
    if "chat_history" not in st.session_state:
        current_time = datetime.now().strftime("%H:%M")
        st.session_state.chat_history = [{
            "role": "assistant", 
            "text": """👋 **Welcome to the Shoe Department!**

I'm your AI-powered personal shopping assistant, here to help you find the perfect shoes.

**What makes me special?**
✨ I remember your preferences across visits
🎯 I provide personalized recommendations
💡 I understand your needs and style

**To get started, may I have your first and last name?**

Once I know who you are, I can check if you're a returning customer and tailor my recommendations to your preferences!""",
            "timestamp": current_time
        }]
    
    if "agent" not in st.session_state:
        st.session_state.agent = ShoeAssistantAgent()


def render_sidebar() -> None:
    """Render the sidebar with modern design and controls"""
    with st.sidebar:
        common.render_sidebar()
        
        # Modern clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
            current_time = datetime.now().strftime("%H:%M")
            st.session_state.chat_history = [{
                "role": "assistant", 
                "text": "Chat history has been cleared. How can I help you find shoes today?",
                "timestamp": current_time
            }]
            # Reset agent conversation history
            st.session_state.agent = ShoeAssistantAgent()
            st.rerun()
        
        st.markdown("---")
        
        # Agent Capabilities - Collapsible
        with st.expander("🛠️ Agent Capabilities", expanded=False):
            st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 1rem; margin: 0.5rem 0;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                    <div style='color: rgba(255,255,255,0.95); font-size: 0.9rem; line-height: 1.8;'>
                        <div style='margin-bottom: 0.8rem;'>
                            <strong style='color: white;'>👤 User Management</strong><br/>
                            • Profile creation & lookup<br/>
                            • Preference tracking<br/>
                            • Purchase history
                        </div>
                        <div>
                            <strong style='color: white;'>🔍 Shoe Search</strong><br/>
                            • Brand & category search<br/>
                            • Price range filtering<br/>
                            • Activity recommendations
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Example queries - Collapsible
        with st.expander("💡 Try These Queries", expanded=False):
            st.markdown("""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 0.8rem; 
                            border-left: 4px solid #667eea;'>
                    <div style='font-size: 0.85rem; color: #555; line-height: 1.8;'>
                        • "My name is John Smith"<br/>
                        • "Show me running shoes under $150"<br/>
                        • "What Nike shoes do you have?"<br/>
                        • "I need casual shoes"<br/>
                        • "I like Adidas, size 10"<br/>
                        • "Recommend shoes for marathons"
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Stats section - Collapsible
        with st.expander("📊 Database Stats", expanded=False):
            st.markdown("""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 1rem; border-radius: 0.8rem; color: white;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                    <div style='font-size: 0.85rem; line-height: 1.8;'>
                        • 12 Shoe Models<br/>
                        • 8 Premium Brands<br/>
                        • 4 Categories<br/>
                        • User Profiles Enabled
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Info section - Collapsible
        with st.expander("⚡ Powered by Strands Agents SDK", expanded=False):
            st.markdown("""
                <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                            padding: 1rem; border-radius: 0.8rem;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
                    <div style='font-size: 0.85rem; color: #333; line-height: 1.6;'>
                        This assistant uses a local SQLite database with intelligent 
                        user profiling and personalized recommendations.
                    </div>
                </div>
            """, unsafe_allow_html=True)


def render_chat_interface() -> None:
    """Render the main chat interface with modern design"""
    
    # Modern header with gradient
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>
                👟 Shoe Department Assistant
            </h1>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                AI-powered personal shopping experience with Strands Agents SDK
            </p>
            <div style='margin-top: 1rem; display: flex; gap: 1rem; flex-wrap: wrap;'>
                <span style='background: rgba(255,255,255,0.2); padding: 0.4rem 0.8rem; 
                            border-radius: 1rem; color: white; font-size: 0.85rem;'>
                    🤖 Personalized Service
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 0.4rem 0.8rem; 
                            border-radius: 1rem; color: white; font-size: 0.85rem;'>
                    💾 Memory Enabled
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 0.4rem 0.8rem; 
                            border-radius: 1rem; color: white; font-size: 0.85rem;'>
                    🎯 Smart Recommendations
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Chat messages container with custom styling
    st.markdown("""
        <style>
        /* Modern chat message styling */
        .stChatMessage {
            padding: 1.5rem !important;
            border-radius: 1rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        
        .stChatMessage:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        
        /* User message - right aligned with blue gradient */
        [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]) {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-radius: 1rem 1rem 0.2rem 1rem !important;
        }
        
        /* Assistant message - left aligned with light background */
        .stChatMessage[data-testid="stChatMessage"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
            border-radius: 1rem 1rem 1rem 0.2rem !important;
        }
        
        /* Avatar styling */
        .stChatMessage img {
            border-radius: 50% !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        }
        
        /* Timestamp styling */
        .stChatMessage .stCaption {
            opacity: 0.7 !important;
            font-size: 0.75rem !important;
            margin-top: 0.5rem !important;
        }
        
        /* Chat input styling */
        .stChatInputContainer {
            border-top: 2px solid #e0e0e0 !important;
            padding-top: 1rem !important;
            background: white !important;
        }
        
        .stChatInput {
            border-radius: 2rem !important;
            border: 2px solid #667eea !important;
            padding: 0.75rem 1.5rem !important;
            font-size: 1rem !important;
        }
        
        .stChatInput:focus {
            border-color: #764ba2 !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Spinner styling */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history with enhanced styling
        for idx, message in enumerate(st.session_state.chat_history):
            try:
                # Ensure message is a dictionary
                if not isinstance(message, dict):
                    logger.warning(f"Invalid message format at index {idx}: {type(message)}")
                    continue
                
                role = message.get("role", "assistant")
                text = message.get("text", "")
                timestamp = message.get("timestamp", "N/A")
                
                with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
                    st.markdown(text)
                    st.caption(f"🕐 {timestamp}")
            except Exception as e:
                logger.error(f"Error rendering message {idx}: {e}")
                continue
    
    # Chat input with placeholder
    user_input = st.chat_input("💬 Ask about shoes, brands, prices, or get personalized recommendations...")
    
    if user_input:
        # Add user message to chat history with timestamp
        current_time = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({
            "role": "user", 
            "text": user_input,
            "timestamp": current_time
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
            st.caption(f"🕐 {current_time}")
        
        # Get agent response with loading animation
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                response_text = st.session_state.agent.get_response(user_input)
                
                # Display agent response
                st.markdown(response_text)
                
                # Add agent response to chat history
                current_time = datetime.now().strftime("%H:%M")
                st.caption(f"🕐 {current_time}")
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "text": response_text,
                    "timestamp": current_time
                })


def main():
    """Main application function"""
    load_css()
    
    try:
        initialize_session()
        render_sidebar()
        render_chat_interface()
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
        st.button("Reload Application", on_click=lambda: st.rerun())


if __name__ == "__main__":
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()

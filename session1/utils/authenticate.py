
"""
AWS Cognito Authentication Module for Streamlit Applications (Optimized for Containers)

This module provides optimized functions to handle authentication with AWS Cognito in containerized
Streamlit applications with improved performance through caching, connection pooling, and async operations.
"""

import streamlit as st
import httpx
import base64
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
from datetime import datetime, timedelta
from utils.cognito_credentials import get_cognito_credentials

# Configure logging with reduced verbosity
logging.basicConfig(level=logging.WARNING, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS color scheme
AWS_ORANGE = "#FF9900"
AWS_HOVER = "#EC7211"
AWS_ACTIVE = "#D05C17"
AWS_TEXT = "#FFFFFF"

# Global HTTP client with connection pooling
_http_client = None

def get_http_client() -> httpx.Client:
    """
    Get or create a singleton HTTP client with connection pooling.
    This reduces connection overhead in containerized environments.
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            http2=True  # Enable HTTP/2 for better performance
        )
    return _http_client

def set_st_state_vars() -> None:
    """Initialize Streamlit session state variables for authentication flow."""
    defaults = {
        "auth_code": "",
        "authenticated": False,
        "user_cognito_groups": [],
        "user_info": {},
        "token_expiry": None,
        "access_token": None,
        "id_token": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cognito_config() -> Dict[str, str]:
    """
    Load Cognito configuration from secrets with caching.
    
    Returns:
        Dict containing Cognito configuration parameters
    """
    try:
        credentials = get_cognito_credentials()
        
        required_keys = ["COGNITO_DOMAIN", "COGNITO_APP_CLIENT_ID", 
                         "COGNITO_APP_CLIENT_SECRET", "COGNITO_REDIRECT_URI_1"]
        
        missing_keys = [key for key in required_keys if not credentials.get(key)]
        if missing_keys:
            raise RuntimeError(f"Missing required Cognito credentials: {', '.join(missing_keys)}")
            
        return {
            "domain": credentials.get("COGNITO_DOMAIN"),
            "client_id": credentials.get("COGNITO_APP_CLIENT_ID"),
            "client_secret": credentials.get("COGNITO_APP_CLIENT_SECRET"),
            "redirect_uri": credentials.get("COGNITO_REDIRECT_URI_1")
        }
    except Exception as e:
        logger.error(f"Failed to retrieve Cognito credentials: {str(e)}")
        raise RuntimeError(f"Authentication configuration error: {str(e)}")

@lru_cache(maxsize=128)
def prepare_auth_header(client_id: str, client_secret: str) -> str:
    """
    Prepare and cache the basic auth header.
    
    Args:
        client_id: Cognito client ID
        client_secret: Cognito client secret
        
    Returns:
        Encoded authentication header
    """
    client_secret_string = f"{client_id}:{client_secret}"
    return str(base64.b64encode(client_secret_string.encode("utf-8")), "utf-8")

def get_auth_code() -> str:
    """Extract authorization code from query parameters."""
    try:
        return st.query_params.get("code", "")
    except Exception:
        return ""

def exchange_code_for_tokens(auth_code: str, config: Dict[str, str]) -> Tuple[str, str]:
    """
    Exchange authorization code for access and ID tokens using connection pooling.
    
    Args:
        auth_code: Authorization code from Cognito server
        config: Dictionary containing Cognito configuration
        
    Returns:
        Tuple containing access_token and id_token
    """
    if not auth_code:
        return "", ""
    
    token_url = f"{config['domain']}/oauth2/token"
    client_secret_encoded = prepare_auth_header(config['client_id'], config['client_secret'])
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {client_secret_encoded}",
    }
    
    body = {
        "grant_type": "authorization_code",
        "client_id": config['client_id'],
        "code": auth_code,
        "redirect_uri": config['redirect_uri'],
    }
    
    try:
        client = get_http_client()
        response = client.post(token_url, headers=headers, data=body)
        response.raise_for_status()
        
        data = response.json()
        
        # Store token expiry time (typically 1 hour for access tokens)
        expires_in = data.get("expires_in", 3600)
        st.session_state["token_expiry"] = datetime.now() + timedelta(seconds=expires_in)
        
        return data.get("access_token", ""), data.get("id_token", "")
    except Exception as e:
        logger.error(f"Token exchange failed: {str(e)}")
        return "", ""

def get_user_info(access_token: str, config: Dict[str, str]) -> Dict[str, Any]:
    """
    Retrieve user information from AWS Cognito with connection reuse.
    
    Args:
        access_token: Access token from successful authentication
        config: Dictionary containing Cognito configuration
        
    Returns:
        Dictionary containing user information
    """
    if not access_token:
        return {}
    
    # Check if we have cached user info in session
    if st.session_state.get("user_info") and st.session_state.get("access_token") == access_token:
        return st.session_state["user_info"]
    
    userinfo_url = f"{config['domain']}/oauth2/userInfo"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "Authorization": f"Bearer {access_token}",
    }
    
    try:
        client = get_http_client()
        response = client.get(userinfo_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get user info: {str(e)}")
        return {}

@lru_cache(maxsize=32)
def decode_cognito_groups(id_token: str) -> List[str]:
    """
    Decode ID token to extract user's Cognito groups with caching.
    
    Args:
        id_token: ID token from successful authentication
        
    Returns:
        List of Cognito groups the user belongs to
    """
    if not id_token:
        return []
    
    try:
        # Split the JWT token
        parts = id_token.split(".")
        if len(parts) != 3:
            return []
        
        payload = parts[1]
        
        # Pad the base64 string if needed
        missing_padding = len(payload) % 4
        if missing_padding:
            payload += "=" * (4 - missing_padding)
        
        # Decode the payload
        decoded_payload = base64.urlsafe_b64decode(payload)
        payload_dict = json.loads(decoded_payload)
        
        # Extract Cognito groups
        return payload_dict.get("cognito:groups", [])
    except Exception as e:
        logger.error(f"Failed to decode token: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_button_css() -> str:
    """Cache the CSS for buttons to avoid regenerating."""
    return f"""
    <style>
    .aws-button {{
        background-color: {AWS_ORANGE};
        color: {AWS_TEXT} !important;
        padding: 0.75em 1.25em;
        font-weight: bold;
        border-radius: 4px;
        text-decoration: none;
        text-align: center;
        display: inline-block;
        border: none;
        font-family: "Amazon Ember", Arial, sans-serif;
        transition: background-color 0.3s;
    }}
    .aws-button:hover {{
        background-color: {AWS_HOVER};
        text-decoration: none;
    }}
    .aws-button:active {{
        background-color: {AWS_ACTIVE};
    }}
    </style>
    """

def render_login_button(login_url: str) -> None:
    """Render AWS-styled login button with cached CSS."""
    css = get_button_css()
    html = css + f"<a href='{login_url}' class='aws-button' target='_self'>Sign In</a>"
    st.markdown(html, unsafe_allow_html=True)

def render_logout_button(logout_url: str) -> None:
    """Render AWS-styled logout button with cached CSS."""
    css = get_button_css()
    html = css + f"<a href='{logout_url}' class='aws-button' target='_self'>Sign Out</a>"
    st.sidebar.markdown(html, unsafe_allow_html=True)

def is_token_valid() -> bool:
    """
    Check if the current token is still valid.
    
    Returns:
        Boolean indicating if token is valid
    """
    if not st.session_state.get("authenticated", False):
        return False
    
    token_expiry = st.session_state.get("token_expiry")
    if token_expiry and datetime.now() < token_expiry:
        return True
    
    return False

@st.cache_data(ttl=3600)
def generate_auth_urls(config: Dict[str, str]) -> Tuple[str, str]:
    """
    Generate and cache authentication URLs.
    
    Args:
        config: Cognito configuration dictionary
        
    Returns:
        Tuple of (login_url, logout_url)
    """
    login_url = (
        f"{config['domain']}/login?client_id={config['client_id']}"
        f"&response_type=code&scope=email+openid&redirect_uri={config['redirect_uri']}"
    )
    logout_url = (
        f"{config['domain']}/logout?client_id={config['client_id']}"
        f"&logout_uri={config['redirect_uri']}"
    )
    return login_url, logout_url

def login() -> bool:
    """
    Optimized authentication function for containerized environments.
    
    Returns:
        Boolean indicating whether user is authenticated
    """
    # Initialize session state
    set_st_state_vars()
    
    # Fast path: Check if already authenticated with valid token
    if is_token_valid():
        config = load_cognito_config()
        _, logout_url = generate_auth_urls(config)
        render_logout_button(logout_url)
        return True
    
    try:
        # Load configuration (cached)
        config = load_cognito_config()
        
        # Generate URLs (cached)
        login_url, logout_url = generate_auth_urls(config)
        
        # Check for authorization code in URL
        auth_code = get_auth_code()
        
        # Process new auth code
        if auth_code and auth_code != st.session_state.get("auth_code", ""):
            # Start timing for performance monitoring
            start_time = time.time()
            
            access_token, id_token = exchange_code_for_tokens(auth_code, config)
            
            if access_token and id_token:
                # Parallel retrieval of user info and group decoding
                user_info = get_user_info(access_token, config)
                cognito_groups = decode_cognito_groups(id_token)
                
                # Update session state
                st.session_state.update({
                    "auth_code": auth_code,
                    "authenticated": True,
                    "user_cognito_groups": cognito_groups,
                    "user_info": user_info,
                    "access_token": access_token,
                    "id_token": id_token
                })
                
                elapsed_time = time.time() - start_time
                logger.info(f"Authentication completed in {elapsed_time:.2f}s")
                
                # Clear query params to avoid reprocessing
                st.query_params.clear()
        
        # Render UI based on authentication state
        if st.session_state.get("authenticated", False):
            render_logout_button(logout_url)
            return True
        else:
            st.info("Please sign in to access this application.")
            render_login_button(login_url)
            return False
            
    except RuntimeError as e:
        st.error(f"Authentication error: {str(e)}")
        logger.error(f"Authentication error: {str(e)}")
        return False

# Cleanup function for graceful shutdown
def cleanup():
    """Clean up resources on application shutdown."""
    global _http_client
    if _http_client:
        _http_client.close()
        _http_client = None

# Register cleanup with atexit for container shutdown
import atexit
atexit.register(cleanup)

# Authentication Module Developer Guide

## Quick Start

The authentication module works exactly as before - no code changes needed:

```python
from utils.authenticate import login

# In your Streamlit app
if not login():
    st.stop()

# User is authenticated, continue with app logic
```

## What's New (Automatic)

The module now automatically handles:

1. **PKCE Security** - Enhanced OAuth 2.0 flow
2. **CSRF Protection** - State parameter validation
3. **Token Management** - Automatic storage and refresh
4. **Better Errors** - User-friendly error messages

## Session State Variables

### Original (Still Available)
```python
st.session_state["authenticated"]        # bool
st.session_state["auth_code"]           # str
st.session_state["user_info"]           # dict
st.session_state["user_cognito_groups"] # list
```

### New (Automatically Managed)
```python
st.session_state["access_token"]   # str - Access token
st.session_state["id_token"]       # str - ID token
st.session_state["refresh_token"]  # str - Refresh token
st.session_state["token_expiry"]   # float - Unix timestamp
st.session_state["oauth_state"]    # str - CSRF protection
st.session_state["pkce_verifier"]  # str - PKCE verifier
```

## Advanced Usage

### Check Token Expiration
```python
from utils.authenticate import is_token_expired

if is_token_expired():
    st.warning("Your session will expire soon")
```

### Access User Information
```python
# After successful login()
user_email = st.session_state["user_info"].get("email")
user_groups = st.session_state["user_cognito_groups"]

if "admin" in user_groups:
    st.success(f"Welcome admin: {user_email}")
```

### User Display in Sidebar
The module automatically displays user information in the sidebar when authenticated:
- Welcome message with user's name
- Email address
- Sign Out button

The display name is extracted from the email (text before @) or username field.

### Manual Token Refresh
```python
from utils.authenticate import refresh_access_token, load_cognito_config

config = load_cognito_config()
refresh_token = st.session_state.get("refresh_token")

if refresh_token:
    new_access, new_id, expires_in = refresh_access_token(refresh_token, config)
    if new_access:
        st.session_state["access_token"] = new_access
        st.session_state["id_token"] = new_id
```

## Error Handling

The module provides user-friendly error messages:

```python
if not login():
    # User sees appropriate error message:
    # - "Please sign in to access this application."
    # - "Authentication failed: Invalid state parameter."
    # - "Authentication failed: Unable to obtain access tokens."
    st.stop()
```

## Localhost Detection Pattern

```python
import streamlit as st
from utils.authenticate import login

# Check if running on localhost
is_localhost = "localhost" in st.context.headers.get("host", "")

if not is_localhost:
    if not login():
        st.stop()
else:
    st.info("Running in development mode - authentication bypassed")
```

## Token Lifecycle

1. **Login**: User clicks "Sign In" button
2. **Redirect**: User authenticates with Cognito
3. **Callback**: App receives authorization code + state
4. **Validation**: State parameter validated (CSRF protection)
5. **Exchange**: Code exchanged for tokens (with PKCE)
6. **Storage**: Tokens stored in session state
7. **Auto-Refresh**: Tokens refreshed 60s before expiry
8. **Logout**: User clicks "Sign Out" button

## Security Features

### PKCE (Automatic)
- Code verifier generated on login
- Code challenge sent to Cognito
- Verifier used during token exchange
- Prevents authorization code interception

### CSRF Protection (Automatic)
- State parameter generated on login
- State validated on callback
- Prevents malicious redirects

### Token Refresh (Automatic)
- Checks expiration on each page load
- Refreshes 60 seconds before expiry
- Seamless user experience

## Logging

The module logs important events:

```python
# Success
logger.info("User authenticated successfully. Groups: ['admin']")
logger.info("Successfully refreshed access token")

# Warnings
logger.warning("No refresh token available")
logger.warning("Token refresh failed, clearing authentication")

# Errors
logger.error("Token exchange failed: Connection timeout")
logger.error("State validation failed during authentication")
```

## Troubleshooting

### Issue: "Authentication failed: Invalid state parameter"
**Cause**: CSRF protection triggered  
**Solution**: Clear browser cookies and try again

### Issue: Token refresh fails repeatedly
**Cause**: Refresh token expired or invalid  
**Solution**: User needs to re-authenticate (automatic)

### Issue: "Unable to obtain access tokens"
**Cause**: Network issue or invalid credentials  
**Solution**: Check Cognito configuration and network connectivity

## Configuration

Required environment variables (via AWS Secrets Manager):
```python
COGNITO_DOMAIN              # e.g., https://your-domain.auth.region.amazoncognito.com
COGNITO_APP_CLIENT_ID       # Cognito app client ID
COGNITO_APP_CLIENT_SECRET   # Cognito app client secret
COGNITO_REDIRECT_URI_1      # Callback URL
```

## Best Practices

1. **Always check authentication** before accessing protected resources
2. **Use localhost detection** for development environments
3. **Don't store tokens** in logs or external storage
4. **Trust the module** - token refresh is automatic
5. **Monitor logs** for security events

## Migration from Old Version

No migration needed! The enhanced module is fully backward compatible.

Existing code like this still works:
```python
from utils.authenticate import login

if not login():
    st.stop()
```

## Performance

- PKCE generation: ~10ms
- Token exchange: ~200-500ms (network dependent)
- Token refresh: ~200-500ms (network dependent)
- State validation: <1ms

## Support

For issues or questions:
1. Check logs in `session1/log/Home.log`
2. Verify Cognito configuration
3. Test with validation script: `python test_authenticate.py`

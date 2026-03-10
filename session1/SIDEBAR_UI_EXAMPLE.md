# Sidebar User Interface Example

## Authenticated User Display

When a user is successfully authenticated, the sidebar displays:

```
┌─────────────────────────────────┐
│                                 │
│  👤 Welcome, John!              │
│  john.doe@example.com           │
│                                 │
│  ┌───────────────────────────┐ │
│  │      Sign Out             │ │
│  └───────────────────────────┘ │
│                                 │
└─────────────────────────────────┘
```

## Display Logic

### Name Extraction
- **Email users**: `john.doe@example.com` → displays as "John"
- **Username users**: `johndoe` → displays as "Johndoe"
- **Fallback**: If no email or username, displays as "User"

### Styling
- **Welcome message**: Dark gray (#232F3E), 0.9em font
- **Email/username**: Medium gray (#545B64), 0.8em font
- **Sign Out button**: AWS Orange (#FF9900) with hover effects
- **Container**: Clean white background with subtle border

## User Information Available

The `user_info` dictionary from Cognito typically contains:

```python
{
    "email": "john.doe@example.com",
    "username": "johndoe",
    "sub": "uuid-string",
    "email_verified": true,
    # Additional custom attributes if configured
}
```

## Customization Options

### Display Different Field
To show a different field (e.g., full name):

```python
# In render_logout_button()
username = user_info.get("name", user_info.get("email", "User"))
```

### Show User Groups
To display user's Cognito groups:

```python
user_groups = st.session_state.get("user_cognito_groups", [])
if user_groups:
    st.sidebar.info(f"Groups: {', '.join(user_groups)}")
```

### Custom Greeting
To customize the greeting message:

```python
# Change this line in render_logout_button()
<div class='user-greeting'>👤 Welcome back, {display_name}!</div>
```

## Responsive Design

The UI is responsive and works well on:
- Desktop browsers
- Tablet devices  
- Mobile browsers (Streamlit mobile view)

The email address uses `word-break: break-word` to handle long email addresses gracefully.

## Accessibility

- Semantic HTML structure
- Sufficient color contrast (WCAG AA compliant)
- Clear visual hierarchy
- Keyboard accessible (standard link navigation)

## Browser Compatibility

Tested and working on:
- Chrome/Edge (Chromium)
- Firefox
- Safari
- Mobile browsers

## Example Screenshots

### Before Login
```
┌─────────────────────────────────┐
│                                 │
│  ℹ️ Please sign in to access   │
│     this application.           │
│                                 │
│  ┌───────────────────────────┐ │
│  │      Sign In              │ │
│  └───────────────────────────┘ │
│                                 │
└─────────────────────────────────┘
```

### After Login
```
┌─────────────────────────────────┐
│                                 │
│  👤 Welcome, Sarah!             │
│  sarah.smith@company.com        │
│                                 │
│  ┌───────────────────────────┐ │
│  │      Sign Out             │ │
│  └───────────────────────────┘ │
│                                 │
└─────────────────────────────────┘
```

## Integration with Other Sessions

This enhancement is in `session1/utils/authenticate.py`. To apply to other sessions:

1. Copy the updated `authenticate.py` to other session utils folders
2. Or create a shared utils module
3. Or use symbolic links (not recommended for deployment)

Recommended approach:
```bash
# Copy to other sessions
cp session1/utils/authenticate.py session2/utils/
cp session1/utils/authenticate.py session3/utils/
cp session1/utils/authenticate.py session4/utils/
cp session1/utils/authenticate.py session5/utils/
```

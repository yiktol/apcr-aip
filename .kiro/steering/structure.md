# Project Structure

## Root Organization

The project follows a session-based structure where each session is a self-contained Streamlit application:

```
.
├── session0/          # Static HTML landing page
├── session1/          # Fundamentals of AI and ML
├── session2/          # Fundamentals of Generative AI
├── session3/          # Applications of Foundation Models
├── session4/          # Responsible AI & Security
├── session5/          # Practical Applications
└── .kiro/             # Kiro configuration and steering
```

## Session Structure Pattern

Each Streamlit session (1-5) follows this consistent structure:

```
sessionX/
├── Home.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── pages/                     # Multi-page app pages
│   ├── 01_Topic.py
│   ├── 02_Topic.py
│   └── ...
├── utils/                     # Shared utilities
│   ├── __init__.py
│   ├── authenticate.py       # Cognito authentication
│   ├── cognito_credentials.py
│   ├── common.py             # Common UI components
│   ├── styles.py             # CSS and styling
│   └── ...
└── assets/                    # Static assets
    └── images/
```

## Key Conventions

### Entry Points
- Each session has `Home.py` as the main entry point
- Multi-page apps use Streamlit's `pages/` directory with numeric prefixes (01_, 02_, etc.)

### Authentication
- Production deployments check `st.context.headers["host"]` for localhost
- Non-localhost environments require Cognito authentication via `authenticate.login()`
- Authentication utilities are in `utils/authenticate.py`

### Utilities Organization
- `utils/common.py` - Shared UI components (sidebar, headers, etc.)
- `utils/styles.py` - CSS styling and custom headers
- `utils/authenticate.py` - Authentication logic
- `utils/knowledge_check.py` - Quiz/assessment functionality
- Session-specific utilities in respective `utils/` directories

### Styling
- Custom CSS loaded via `load_css()` from `utils/styles.py`
- Custom headers created with `custom_header()` function
- AWS branding and color schemes

### Session State
- UUID-based session tracking: `st.session_state.session_id`
- Knowledge check progress tracked in session state
- Authentication state managed per session

### Page Configuration
- Wide layout: `layout="wide"`
- Expanded sidebar by default: `initial_sidebar_state="expanded"`
- Session-specific page icons and titles

## Session 0 Specifics

Static HTML structure:
```
session0/
├── index.html
├── 403.html, 404.html, 502.html, 503.html, 504.html
├── aws.svg
└── img/                       # Image assets
```

## Deployment Structure

### Docker
- Each session can have a `Dockerfile`
- Scripts in `scripts/` directory for build and deployment

### Infrastructure (Session 5)
```
session5/infrastructure/
├── parameters.json            # CloudFormation parameters
└── scripts/
    └── deploy.sh             # Deployment script
```

## File Naming Conventions

- Python files: snake_case (e.g., `authenticate.py`, `common_clustering.py`)
- Page files: Numeric prefix + PascalCase (e.g., `01_ML_Terminology.py`)
- Config files: lowercase with dots (e.g., `config.toml`)
- Shell scripts: kebab-case (e.g., `build-and-push.sh`)

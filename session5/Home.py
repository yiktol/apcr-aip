import streamlit as st
import utils.common as common
import utils.authenticate as authenticate
import utils.styles as styles

st.set_page_config(
    page_title="Shoe Industry Market Research Analyzer",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    # Initialize session state
    common.initialize_session_state()
    styles.load_css()

    # Render the sidebar
    with st.sidebar:
        common.render_sidebar()

    # Render the main content
    st.title("👟 Shoe Industry Market Research Analyzer")
    st.markdown("### 🚀 Welcome to the Shoe Industry Market Research Analyzer!")
    st.markdown("This tool allows you to analyze the market for shoe industry and make informed decisions based on the data you upload.")



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
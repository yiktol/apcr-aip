import streamlit as st
import logging
import boto3
import uuid
import os
from botocore.exceptions import ClientError
from io import BytesIO
from PIL import Image
import time
from utils.styles import load_css, custom_header, load_css
from utils.common import render_sidebar
import utils.authenticate as authenticate
# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Set page configuration
st.set_page_config(
    page_title="GenAI Use Cases",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load the custom CSS
load_css()


# ------- SESSION MANAGEMENT FUNCTIONS -------

def initialize_session():
    """Initialize session state variables if they don't exist"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

def reset_session():
    """Reset all session variables"""
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.conversation_history = []
    st.session_state.messages = []
    st.success("Session has been reset successfully!")

# ------- API FUNCTIONS -------

def text_conversation(bedrock_client, model_id, system_prompts, messages, **params):
    """Sends messages to a model."""
    logger.info(f"Generating message with model {model_id}")
    
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=params,
            additionalModelRequestFields={}
        )
        
        # Log token usage
        token_usage = response['usage']
        logger.info(f"Input tokens: {token_usage['inputTokens']}")
        logger.info(f"Output tokens: {token_usage['outputTokens']}")
        logger.info(f"Total tokens: {token_usage['totalTokens']}")
        logger.info(f"Stop reason: {response['stopReason']}")
        
        return response
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return None

def image_conversation(bedrock_client, model_id, input_text, image_bytes, image_format='jpeg'):
    """Sends a message with image to a model."""
    logger.info(f"Generating message with model {model_id}")
    
    # Message to send
    message = {
        "role": "user",
        "content": [
            {"text": input_text},
            {
                "image": {
                    "format": image_format,
                    "source": {"bytes": image_bytes}
                }
            }
        ]
    }
    
    messages = [message]
    
    # Send the message
    try:
        response = bedrock_client.converse(modelId=model_id, messages=messages)
        return response
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return None

def stream_conversation(bedrock_client, model_id, messages, system_prompts, inference_config):
    """Simulates streaming by displaying the response gradually."""
    logger.info(f"Simulating streaming for model {model_id}")
    
    try:
        # Make a regular synchronous call
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields={}
        )
        
        # Get the full response text
        output_message = response['output']['message']
        full_text = ""
        for content in output_message['content']:
            if 'text' in content:
                full_text += content['text']
        
        # Simulate streaming by displaying the text gradually
        placeholder = st.empty()
        display_text = ""
        
        # Split into words for more natural "streaming" effect
        words = full_text.split()
        
        # Display words with a slight delay to simulate streaming
        for i, word in enumerate(words):
            display_text += word + " "
            # Update every few words to avoid too many UI updates
            if i % 3 == 0 or i == len(words) - 1:
                with placeholder.container():
                    st.markdown(f"**Response:**\n{display_text}")
                time.sleep(0.05)  # Small delay for streaming effect
        
        # Display token usage after streaming is complete
        st.markdown("### Response Details")
        token_usage = response['usage']
        col1, col2, col3 = st.columns(3)
        col1.metric("Input Tokens", token_usage['inputTokens'])
        col2.metric("Output Tokens", token_usage['outputTokens'])
        col3.metric("Total Tokens", token_usage['totalTokens'])
        st.caption(f"Stop reason: {response['stopReason']}")
        
        # Return the data
        return {
            'response': full_text,
            'tokens': {
                'input': token_usage['inputTokens'],
                'output': token_usage['outputTokens'],
                'total': token_usage['totalTokens']
            },
            'stop_reason': response['stopReason']
        }
        
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error in stream_conversation: {str(e)}")
        return None

# ------- UI COMPONENTS -------

def model_selection_panel():
    """Model selection and parameters in the side panel"""
    st.markdown("<div class='side-header'>Model Selection</div>", unsafe_allow_html=True)
    
    MODEL_CATEGORIES = {
        "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"],
        "Anthropic": ["anthropic.claude-v2:1", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"],
        "Cohere": ["cohere.command-text-v14:0", "cohere.command-r-plus-v1:0", "cohere.command-r-v1:0"],
        "Meta": ["meta.llama3-70b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0"],
        "Mistral": ["mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", 
                   "mistral.mistral-7b-instruct-v0:2", "mistral.mistral-small-2402-v1:0"]
    }
    
    # Create selectbox for provider first
    provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()), key="side_provider")
    
    # Then create selectbox for models from that provider
    model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider], key="side_model")
    
    st.markdown("<div class='side-header'>API Method</div>", unsafe_allow_html=True)
    api_method = st.radio(
        "Select API Method", 
        ["Streaming", "Synchronous"], 
        index=0,
        help="Streaming provides real-time responses, while Synchronous waits for complete response",
        key="side_api_method"
    )
    
    st.markdown("<div class='side-header'>Parameter Tuning</div>", unsafe_allow_html=True)
    
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.05,
        key="side_temperature",
        help="Higher values make output more random, lower values more deterministic"
    )
        
    top_p = st.slider(
        "Top P", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.9, 
        step=0.05,
        key="side_top_p",
        help="Controls diversity via nucleus sampling"
    )
        
    max_tokens = st.number_input(
        "Max Tokens", 
        min_value=50, 
        max_value=4096, 
        value=1024, 
        step=50,
        key="side_max_tokens",
        help="Maximum number of tokens in the response"
    )
        
    params = {
        "temperature": temperature,
        "topP": top_p,
        "maxTokens": max_tokens
    }
    
    return model_id, params, api_method

def display_use_case_explanation(use_case):
    """Display explanation for specific use cases"""
    explanations = {
        "summarization": """
            ### Text Summarization
            
            **What is it?** 
            Summarization is the process of condensing a longer text into a shorter version while preserving key 
            information and meaning. It's like creating a TL;DR (Too Long; Didn't Read) version of content.
            
            **Use Cases:**
            - Summarizing news articles, research papers, and long documents
            - Creating executive summaries for reports
            - Condensing meeting notes or transcripts
            - News aggregation and brief creation
            
            **Business Value:**
            - Saves time for readers by highlighting key points
            - Makes information more accessible and digestible
            - Helps manage information overload
            
            **How Foundation Models Help:**
            Foundation models understand context, identify important information, and can generate 
            coherent summaries that maintain the original meaning while being significantly shorter.
        """,
        
        "extraction": """
            ### Information Extraction
            
            **What is it?**
            Information extraction involves identifying and extracting specific structured information
            from unstructured text. It's like having an assistant who can pull out exactly what you need
            from a sea of information.
            
            **Use Cases:**
            - Extracting contact information, dates, or specific data points from documents
            - Identifying product specifications from descriptions
            - Extracting entities (people, organizations, locations) from text
            - Parsing resumes for candidate information
            
            **Business Value:**
            - Automates manual data entry and classification
            - Enables structured analysis of unstructured data
            - Improves search and information retrieval
            
            **How Foundation Models Help:**
            Foundation models can recognize patterns in text and understand context to accurately identify 
            and extract relevant information, even when it's presented in various formats or structures.
        """,
        
        "translation": """
            ### Language Translation
            
            **What is it?**
            Translation converts text from one language to another while preserving meaning, context, and tone.
            Modern AI translation goes beyond word-for-word conversion to create natural, fluent translations.
            
            **Use Cases:**
            - Translating documents, websites, and applications for global audiences
            - Real-time translation for customer support
            - Content localization for international markets
            - Cross-language research and information gathering
            
            **Business Value:**
            - Expands market reach to international customers
            - Improves accessibility of information
            - Reduces costs associated with human translation
            
            **How Foundation Models Help:**
            Foundation models have been trained on vast multilingual datasets, allowing them to understand
            nuances across languages, preserve idioms, and generate natural-sounding translations that
            consider cultural context.
        """,
        
        "content_generation": """
            ### Content Generation
            
            **What is it?**
            Content generation involves creating new, original text based on provided guidance or parameters.
            This can range from writing blog posts to generating product descriptions or creative stories.
            
            **Use Cases:**
            - Creating marketing copy and product descriptions
            - Generating blog posts and social media content
            - Drafting emails and business communications
            - Creating educational content and tutorials
            
            **Business Value:**
            - Accelerates content creation processes
            - Helps overcome writer's block and ideation challenges
            - Enables personalization at scale
            - Maintains consistent brand voice across channels
            
            **How Foundation Models Help:**
            Foundation models can generate human-like text that follows specific tones, styles, or formats.
            They can adapt to brand guidelines and produce content that sounds natural and engaging while
            meeting specific parameters.
        """,
        
        "redaction": """
            ### Redaction and PII Management
            
            **What is it?**
            Redaction involves identifying and removing or masking sensitive information like personally identifiable
            information (PII), protected health information (PHI), or confidential business data from documents.
            
            **Use Cases:**
            - Automatically removing PII from customer service logs
            - Sanitizing documents before sharing them publicly
            - Compliance with data protection regulations (GDPR, HIPAA, etc.)
            - Protecting sensitive information in legal documents
            
            **Business Value:**
            - Reduces risk of data breaches and associated costs
            - Ensures regulatory compliance
            - Builds customer trust through demonstrated data protection
            
            **How Foundation Models Help:**
            Foundation models can identify various types of sensitive information based on context, not just
            patterns. They can recognize PII even when it appears in unusual formats or contexts, ensuring
            thorough redaction while preserving document readability.
        """,
        
        "code_generation": """
            ### Code Generation
            
            **What is it?**
            Code generation is the process of automatically creating executable code based on natural language
            descriptions or requirements. It's like having an AI pair programmer who can write code for you.
            
            **Use Cases:**
            - Generating boilerplate code to accelerate development
            - Creating code snippets based on functionality descriptions
            - Translating between programming languages
            - Automating repetitive coding tasks
            
            **Business Value:**
            - Increases developer productivity
            - Reduces time-to-market for software products
            - Lowers the barrier to entry for programming tasks
            - Helps standardize code practices
            
            **How Foundation Models Help:**
            Foundation models have been trained on vast repositories of code across multiple languages and
            can generate functional, syntactically correct code from natural language descriptions. They
            understand programming patterns and best practices.
        """,
        
        "sentiment_analysis": """
            ### Sentiment Analysis
            
            **What is it?**
            Sentiment analysis determines the emotional tone behind text—whether the expressed opinion is
            positive, negative, or neutral. It helps understand how people feel about products, services,
            or topics.
            
            **Use Cases:**
            - Analyzing customer feedback and reviews
            - Monitoring brand perception on social media
            - Understanding employee satisfaction from surveys
            - Gauging public opinion on topics or events
            
            **Business Value:**
            - Provides actionable insights into customer satisfaction
            - Helps identify product issues or opportunities
            - Enables real-time response to negative sentiment
            - Improves customer experience through better understanding
            
            **How Foundation Models Help:**
            Foundation models can detect subtle emotional cues, sarcasm, and implied sentiment that traditional
            rule-based systems might miss. They understand context and can analyze sentiment across different
            domains and types of text.
        """
    }
    
    if use_case in explanations:
        st.markdown(explanations[use_case], unsafe_allow_html=True)
    else:
        st.warning(f"No explanation available for {use_case}")

def create_use_case_interface(use_case, model_id, params, api_method):
    """Create interface for a specific use case"""
    
    # Display explanation for this use case
    with st.expander("Learn about this use case", expanded=False):
        display_use_case_explanation(use_case)
    
    # Set system prompt based on use case
    system_prompts = {
        "summarization": "You are an assistant specialized in summarizing text. Provide concise summaries that capture the main points while significantly reducing length.",
        "extraction": "You are an assistant specialized in information extraction. Identify and extract specific data points from the provided text.",
        "translation": "You are an assistant specialized in language translation. Provide accurate translations while preserving meaning and context.",
        "content_generation": "You are an assistant specialized in content generation. Create original, engaging content based on the provided specifications.",
        "redaction": "You are an assistant specialized in identifying and redacting sensitive information. Replace any PII with [REDACTED] tags.",
        "code_generation": "You are an assistant specialized in generating code. Write clean, efficient, and well-commented code based on the requirements.",
        "sentiment_analysis": "You are an assistant specialized in sentiment analysis. Analyze the emotional tone of the text and explain your reasoning."
    }
    
    # Get default prompts based on use case
    default_prompts = {
        "summarization": "Summarize the following article about artificial intelligence:\n\nArtificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by humans or by other animals. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs.\nAI applications include advanced web search engines, recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars, generative AI tools, and AI tools playing at the highest level in strategic game systems (such as chess and Go). As machines become increasingly capable, tasks considered to require \"intelligence\" are often removed from the definition of AI, a phenomenon known as the AI effect.\nAI was founded as an academic discipline in 1956, and in the years since it has experienced several waves of optimism, followed by disappointment and the loss of funding (known as an \"AI winter\"), followed by new approaches, success, and renewed funding. AI research has tried and discarded many different approaches, including simulating the brain, modeling human problem solving, formal logic, large databases of knowledge, and imitating animal behavior. In the first decades of the 21st century, highly mathematical-statistical machine learning has dominated the field, and this technique has proved highly successful, helping to solve many challenging problems throughout industry and academia.",
        "extraction": "Extract the following information from this text:\n- All people mentioned\n- All companies mentioned\n- All locations mentioned\n- Any dates or time references\n\nText: John Smith from Microsoft attended a conference in Seattle on March 15, 2023, where he met Sarah Johnson from Google. They discussed a potential partnership for a project launching in New York City next January.",
        "translation": "Translate the following text to Spanish, French, and German:\n\n'Hello! Welcome to our online store. We offer free shipping on orders over $50. Please let us know if you have any questions.'",
        "content_generation": "Write a short blog post about the benefits of adopting cloud computing for small businesses. Include at least 3 key advantages and some practical tips for getting started.",
        "redaction": "Redact all personally identifiable information (PII) in the following text by replacing it with [REDACTED] tags:\n\nHello, my name is Sarah Johnson. I live at 123 Maple Street, Portland OR 97205. You can reach me at sarah.johnson@example.com or 503-555-7890. My social security number is 123-45-6789 and my credit card is 4111-1111-1111-1111 with CVV 123.",
        "code_generation": "Write a Python function that sorts a list of dictionaries based on a specified key. The function should take two parameters: the list to sort and the key to sort by. Include proper error handling and documentation.",
        "sentiment_analysis": "Analyze the sentiment (positive, negative, or neutral) of the following customer reviews, and explain your reasoning:\n\n1. \"I absolutely love this product! It works exactly as described and has made my life so much easier.\"\n\n2. \"The quality is okay but it's not worth the price. I expected better for what I paid.\"\n\n3. \"Product arrived on time and functions as expected. Nothing special but does the job.\""
    }
    
    # Create the interface
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    

    system_prompt = st.text_area(
        "System Prompt", 
        value=system_prompts.get(use_case, "You are a helpful assistant."), 
        height=100,
        key=f"{use_case}_system"
    )

    user_prompt = st.text_area(
        "Your Input", 
        value=default_prompts.get(use_case, "Enter your query here..."), 
        height=200,
        placeholder="Enter your text here...",
        key=f"{use_case}_input"
    )
    
    generate_button = st.button(
        "Generate Response", 
        type="primary", 
        key=f"{use_case}_submit",
        help="Click to generate a response based on your input"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if generate_button:
        if not user_prompt.strip():
            st.warning("Please enter your input first.")
            return
            
        with st.status(f"Processing your {use_case} request...",expanded=True) as status:
            # Setup the system prompts and messages
            system_prompts_list = [{"text": system_prompt}]
            message = {
                "role": "user",
                "content": [{"text": user_prompt}]
            }
            messages = [message]
            
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Send request to the model
                if api_method == "Streaming":
                    response_data = stream_conversation(bedrock_client, model_id, messages, system_prompts_list, params)
                    
                    if response_data:
                        status.update(label="Response received!", state="complete")
                else:
                    # Synchronous API call
                    response = text_conversation(bedrock_client, model_id, system_prompts_list, messages, **params)
                    
                    if response:
                        status.update(label="Response received!", state="complete")
                        
                        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
                        
                        # Display the model's response
                        output_message = response['output']['message']
                        
                        st.markdown(f"**{output_message['role'].title()}**")
                        for content in output_message['content']:
                            st.markdown(content['text'])
                        
                        # Show token usage
                        st.markdown("### Response Details")
                        token_usage = response['usage']
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Input Tokens", token_usage['inputTokens'])
                        col2.metric("Output Tokens", token_usage['outputTokens'])
                        col3.metric("Total Tokens", token_usage['totalTokens'])
                        st.caption(f"Stop reason: {response['stopReason']}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
            
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error in {use_case}: {str(e)}")

def image_analysis_interface(model_id):
    """Interface for image analysis."""
    with st.expander("Learn about this use case", expanded=False):
        st.markdown("""
            ### Image Analysis
            
            **What is it?**
            Image analysis uses AI to understand and interpret visual content in images. Multimodal foundation models
            can process both images and text, allowing them to answer questions about visual content or generate
            descriptions of images.
            
            **Use Cases:**
            - Content moderation and visual safety
            - Image categorization and organization
            - Visual question answering
            - Image accessibility through descriptions for visually impaired users
            
            **Business Value:**
            - Automates manual image review processes
            - Improves search capabilities for visual content
            - Enhances accessibility of visual information
            - Enables content insights from large image datasets
            
            **How Foundation Models Help:**
            Multimodal foundation models bridge the gap between visual and textual understanding. They can identify
            objects, scenes, actions, and concepts in images and express these findings in natural language.
            They can also comprehend questions about images and provide relevant answers based on visual content.
        """)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Image input options
    image_source = st.radio(
        "Select Image Source",
        options=["Upload Image", "Sample Image"],
        horizontal=True,
        key="image_source"
    )
    
    image_bytes = None
    image_format = None
    
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image_format = uploaded_file.name.split(".")[-1].lower()
            
            # Display the uploaded image
            image = Image.open(BytesIO(image_bytes))
            st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        # Sample image
        sample_image_path = "images/sg_skyline.jpeg"
        if os.path.exists(sample_image_path):
            with open(sample_image_path, "rb") as f:
                image_bytes = f.read()
            image_format = "jpeg"
            st.image(sample_image_path, caption="Sample Image", use_container_width=True)
        else:
            st.error("Sample image not found. Please use the upload option.")
    
    # Query input
    user_prompt = st.text_area(
        "Ask about the image", 
        value="What's in this image? Provide a detailed description.", 
        height=80,
        placeholder="Enter your question about the image..."
    )
    
    submit = st.button("Analyze Image", type="primary", key="image_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not image_bytes:
            st.warning("Please provide an image first.")
            return
        
        if not user_prompt.strip():
            st.warning("Please enter a question about the image.")
            return
        
        with st.status("Analyzing image...",expanded=True) as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Process the image and get response
                response = image_conversation(bedrock_client, model_id, user_prompt, image_bytes, image_format)
                
                if response:
                    status.update(label="Analysis complete!", state="complete")
                    
                    st.markdown("<div class='response-block'>", unsafe_allow_html=True)
                    
                    # Display the model's response
                    output_message = response['output']['message']
                    
                    st.markdown(f"**{output_message['role'].title()}**")
                    for content in output_message['content']:
                        st.markdown(content['text'])
                    
                    # Show token usage
                    st.markdown("### Response Details")
                    token_usage = response['usage']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Input Tokens", token_usage['inputTokens'])
                    col2.metric("Output Tokens", token_usage['outputTokens'])
                    col3.metric("Total Tokens", token_usage['totalTokens'])
                    st.caption(f"Stop reason: {response['stopReason']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")

# ------- MAIN APP -------

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    initialize_session()
    

    with st.sidebar:
        # Session Management
        # st.markdown("<div class='side-header'>Session Management</div>", unsafe_allow_html=True)
        
        # st.caption(f"ID: {st.session_state.user_id[:8]}...")
        
        # st.button("Reset Session", on_click=reset_session, key="reset_session", 
        #             help="Reset your current session and conversation history")
        
        render_sidebar()
        
        # About section
        with st.expander("About this App", expanded=False):
            st.markdown("""
            This interactive learning environment demonstrates various Generative AI use cases using Amazon Bedrock:
            
            * Summarization
            * Extraction
            * Translation
            * Content Generation
            * Redaction
            * Code Generation
            * Sentiment Analysis
            
            For more information, visit the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/).
            """)

    # Header

    st.markdown("""
    <div class="element-animation">
        <h1>GenAI Use Cases in AWS Bedrock</h1>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""<div class="info-box">
    This interactive learning environment demonstrates various use cases of Generative AI using Amazon Bedrock.
    Explore different capabilities by selecting a use case tab below and experiment with the power of foundation models.
    </div>""",unsafe_allow_html=True)
    
    # Create a 70/30 layout using columns
    main_col, side_col = st.columns([7, 3])
    
    # Side panel for model selection and parameters
    with side_col:
        with st.container(border=True):
            model_id, params, api_method = model_selection_panel()
        # st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content area with use case tabs
    with main_col:
        # Create tabs for different use cases with emojis
        tabs = st.tabs([
            "📝 Summarization", 
            "🔍 Extraction", 
            "🌐 Translation",
            "✍️ Content Generation",
            "🔒 Redaction",
            "💻 Code Generation",
            "😃 Sentiment Analysis",
            "🖼️ Image Analysis"
        ])
        
        # Populate each tab
        with tabs[0]:
            create_use_case_interface("summarization", model_id, params, api_method)
        
        with tabs[1]:
            create_use_case_interface("extraction", model_id, params, api_method)
        
        with tabs[2]:
            create_use_case_interface("translation", model_id, params, api_method)
        
        with tabs[3]:
            create_use_case_interface("content_generation", model_id, params, api_method)
        
        with tabs[4]:
            create_use_case_interface("redaction", model_id, params, api_method)
        
        with tabs[5]:
            create_use_case_interface("code_generation", model_id, params, api_method)
        
        with tabs[6]:
            create_use_case_interface("sentiment_analysis", model_id, params, api_method)
            
        with tabs[7]:
            image_analysis_interface(model_id)
    
    # Footer
    st.markdown("""
    <footer>
        © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()

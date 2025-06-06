# Amazon Bedrock Converse API with Advanced Prompting Techniques

import streamlit as st
import logging
import sys
import boto3
from botocore.exceptions import ClientError
import os
from io import BytesIO
from PIL import Image
import utils.common as common
import utils.authenticate as authenticate
# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Page configuration with improved styling
st.set_page_config(
    page_title="Advanced LLM Prompting Techniques",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

common.initialize_session_state()

# Apply custom CSS for modern appearance
st.markdown("""
    <style>
    .stApp {
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #2563EB;
    }
    .technique-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #4B5563;
    }
    .technique-description {
        margin-bottom: 1.5rem;
        padding: 0.8rem;
        border-left: 3px solid #2563EB;
        background-color: #F0F9FF;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .output-container {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .response-block {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
        margin-top: 1rem;
    }
    .token-metrics {
        display: flex;
        justify-content: space-between;
        background-color: #F0F4F8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
    .metric-item {
        text-align: center;
    }
    .metric-value {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #4B5563;
    }
    .prompt-label {
        font-weight: bold;
        color: #374151;
        margin-bottom: 0.25rem;
    }
    .technique-examples pre {
        background-color: #F1F5F9;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        white-space: pre-wrap;
    }
    
    .tab-content {
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
        border-top: none;
        border-radius: 0 0 0.5rem 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        background-color: #F3F4F6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E0E7FF;
        border-bottom: 2px solid #4F46E5;    
    
    </style>
""", unsafe_allow_html=True)

# ------- API FUNCTIONS -------

def stream_conversation(bedrock_client, model_id, messages, system_prompts, inference_config):
    """Sends messages to a model and streams the response."""
    logger.info(f"Streaming messages with model {model_id}")
    
    try:
        response = bedrock_client.converse_stream(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields={}
        )
        
        stream = response.get('stream')
        if stream:
            placeholder = st.empty()
            full_response = ''
            token_info = {'input': 0, 'output': 0, 'total': 0}
            latency_ms = 0
            
            for event in stream:
                if 'messageStart' in event:
                    role = event['messageStart']['role']
                    with placeholder.container():
                        st.markdown(f"**{role.title()}**")
                
                if 'contentBlockDelta' in event:
                    chunk = event['contentBlockDelta']
                    part = chunk['delta']['text']
                    full_response += part
                    with placeholder.container():
                        st.markdown(f"**Response:**\n{full_response}")
                
                if 'messageStop' in event:
                    stop_reason = event['messageStop']['stopReason']
                    with placeholder.container():
                        st.markdown(f"**Response:**\n{full_response}")
                        st.caption(f"Stop reason: {stop_reason}")
                
                if 'metadata' in event:
                    metadata = event['metadata']
                    if 'usage' in metadata:
                        usage = metadata['usage']
                        token_info = {
                            'input': usage['inputTokens'],
                            'output': usage['outputTokens'],
                            'total': usage['totalTokens']
                        }
                    
                    if 'metrics' in event.get('metadata', {}):
                        latency_ms = metadata['metrics']['latencyMs']
            
            # Display token usage after streaming is complete
            st.markdown("### Response Details")
            col1, col2, col3 = st.columns(3)
            col1.metric("Input Tokens", token_info['input'])
            col2.metric("Output Tokens", token_info['output'])
            col3.metric("Total Tokens", token_info['total'])
            st.caption(f"Latency: {latency_ms}ms")
        
        return True
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return False

# ------- UI COMPONENTS -------

def parameter_sidebar():
    """Sidebar with model selection and parameter tuning."""
    with st.container(border=True):
        st.markdown("<div class='sub-header'>Model Selection</div>", unsafe_allow_html=True)
        
        MODEL_CATEGORIES = {
            "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"],
            "Anthropic": ["anthropic.claude-v2:1", "anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"],
            "Cohere": ["cohere.command-text-v14:0", "cohere.command-r-plus-v1:0", "cohere.command-r-v1:0"],
            "Meta": ["meta.llama3-70b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0"],
            "Mistral": ["mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", 
                        "mistral.mistral-7b-instruct-v0:2", "mistral.mistral-small-2402-v1:0"],
            "AI21": ["ai21.jamba-1-5-large-v1:0", "ai21.jamba-1-5-mini-v1:0"]
        }
        
        # Create selectbox for provider first
        provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()))
        
        # Then create selectbox for models from that provider
        model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider])
        
        st.markdown("<div class='sub-header'>Parameter Tuning</div>", unsafe_allow_html=True)
        
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, 
                            help="Higher values make output more random, lower values more deterministic")
        
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1,
                            help="Controls diversity via nucleus sampling")
        
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4096, value=1024, step=50,
                                    help="Maximum number of tokens in the response")
            
        params = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
    
    with st.sidebar:
    
        common.render_sidebar()
    
        with st.expander("About Prompting Techniques", expanded=False):
            st.markdown("""
            ### Prompting Techniques
            
            **Zero-shot Prompting** - The model generates responses with no examples provided, just a direct instruction.
            
            **Few-shot Prompting** - The prompt includes examples of desired input-output pairs to guide the model.
            
            **Zero-shot Chain of Thought (CoT)** - Instructs the model to reason through its answer step by step without examples.
            
            **Few-shot Chain of Thought (CoT)** - Provides examples of step-by-step reasoning to help the model reason through new problems.
            
            Learn more about [prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering).
            """)
    
    return model_id, params

def zero_shot_interface(model_id, params):
    """Interface for zero-shot prompting technique."""
    
    st.markdown("<div class='technique-title'>Zero-shot Prompting</div>", unsafe_allow_html=True)
    st.markdown("<div class='technique-description'>In zero-shot prompting, you provide clear instructions without examples, and the model performs the task directly. This technique relies on the model's pre-trained knowledge.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Example tasks for zero-shot
    task_options = {
        "Classification": "Classify this review as positive, neutral, or negative: 'The food was delicious but the service was extremely slow.'",
        "Generation": "Write a short poem about artificial intelligence.",
        "Summarization": "Summarize the main benefits of exercise for mental health.",
        "Question Answering": "What are three ways to reduce carbon emissions in urban areas?"
    }
    
    selected_task = st.selectbox("Select Task Type", options=list(task_options.keys()), key="zero_shot_task")
    
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant that provides clear and direct answers to questions.",
        height=100,
        key="zero_shot_system"
    )
    
    user_prompt = st.text_area(
        "Zero-shot Prompt",
        value=task_options[selected_task],
        height=120,
        key="zero_shot_prompt"
    )
    
    submit = st.button("Generate Response", type="primary", key="zero_shot_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not user_prompt.strip():
            st.warning("Please enter a prompt first.")
            return
        
        # Create message structure
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        messages = [message]
        
        # System prompts
        system_prompts = [{"text": system_prompt}]
        
        # Response area
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        
        # Stream the response
        try:
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            success = stream_conversation(bedrock_client, model_id, messages, system_prompts, params)
            
            if not success:
                st.error("Failed to generate response.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def few_shot_interface(model_id, params):
    """Interface for few-shot prompting technique."""
    
    st.markdown("<div class='technique-title'>Few-shot Prompting</div>", unsafe_allow_html=True)
    st.markdown("<div class='technique-description'>Few-shot prompting involves providing a few examples of input-output pairs before asking the model to perform a similar task with new input. This helps the model understand the pattern you want it to follow.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Example tasks for few-shot
    task_templates = {
        "Text Classification": """Here are some examples of text sentiment classification:

Input: "I really enjoyed this movie, it was fantastic!"
Output: Positive

Input: "The product arrived on time but was somewhat damaged."
Output: Mixed

Input: "Terrible customer service, I will never shop here again."
Output: Negative

Now classify the following text:
Input: "The hotel room was spacious with a beautiful view, but the bathroom was not very clean."
Output:""",

        "Entity Extraction": """Here are some examples of extracting person names from text:

Input: "Bill Gates founded Microsoft in 1975."
Output: Bill Gates

Input: "According to Marie Curie, radium compounds glow in the dark."
Output: Marie Curie

Input: "The theory of relativity was developed by Albert Einstein."
Output: Albert Einstein

Now extract the person name from this text:
Input: "The first human to journey into outer space was Yuri Gagarin."
Output:""",

        "Text Transformation": """Here are examples of converting sentences to past tense:

Input: "I walk to the store."
Output: I walked to the store.

Input: "She plays tennis every weekend."
Output: She played tennis every weekend.

Input: "They are going to the concert."
Output: They were going to the concert.

Now convert this sentence to past tense:
Input: "The children run through the park with their dog."
Output:""",

        "Custom Format": """Example 1:
Product: Smartphone
Features:
- 6.5-inch display
- 128GB storage
- 12MP camera
Summary: A mid-range smartphone with adequate storage and decent camera quality.

Example 2:
Product: Wireless Earbuds
Features:
- Active noise cancellation
- 8-hour battery life
- Water resistant
Summary: Compact earbuds with excellent noise cancellation and good battery performance.

Now follow the same format for:
Product: Smart Watch
Features:
- Heart rate monitoring
- Sleep tracking
- 5-day battery life
Summary:"""
    }
    
    selected_task = st.selectbox("Select Example Type", options=list(task_templates.keys()), key="few_shot_task")
    
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant that follows examples closely to provide consistent formatted responses.",
        height=100,
        key="few_shot_system"
    )
    
    user_prompt = st.text_area(
        "Few-shot Prompt with Examples",
        value=task_templates[selected_task],
        height=250,
        key="few_shot_prompt"
    )
    
    submit = st.button("Generate Response", type="primary", key="few_shot_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not user_prompt.strip():
            st.warning("Please enter a prompt with examples first.")
            return
        
        # Create message structure
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        messages = [message]
        
        # System prompts
        system_prompts = [{"text": system_prompt}]
        
        # Response area
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        
        # Stream the response
        try:
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            success = stream_conversation(bedrock_client, model_id, messages, system_prompts, params)
            
            if not success:
                st.error("Failed to generate response.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def zero_shot_cot_interface(model_id, params):
    """Interface for zero-shot chain of thought prompting."""
    
    st.markdown("<div class='technique-title'>Zero-shot Chain of Thought (CoT)</div>", unsafe_allow_html=True)
    st.markdown("<div class='technique-description'>Zero-shot Chain of Thought prompting encourages the model to break down complex problems into steps and reason through them, without providing examples. Adding phrases like 'Let's think step by step' can dramatically improve performance on complex reasoning tasks.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Example tasks for zero-shot CoT
    task_options = {
        "Mathematical Problem": "If a shirt originally costs $25 and is on sale for 20% off, then with an additional discount of 10% at checkout, what is the final price? Let's think step by step.",
        "Logical Reasoning": "A man is looking at a portrait. Someone asks him, 'Whose portrait are you looking at?' He answers, 'I have no brothers or sisters, but this man's father is my father's son.' Who is in the portrait? Let's think through this step by step.",
        "Decision Making": "You're planning a trip to either the mountains or the beach. The mountains offer hiking, scenery and cooler weather. The beach offers swimming, sunbathing, and warmer weather. You have 5 days off work and a budget of $1000. Which destination would be better for your vacation and why? Let's think step by step.",
        "Scientific Analysis": "What would happen if humans suddenly lost the ability to see the color blue? Think through the implications step by step."
    }
    
    selected_task = st.selectbox("Select Task Type", options=list(task_options.keys()), key="zero_shot_cot_task")
    
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a thoughtful assistant that breaks down complex problems into steps and provides detailed reasoning before arriving at conclusions.",
        height=100,
        key="zero_shot_cot_system"
    )
    
    user_prompt = st.text_area(
        "Zero-shot CoT Prompt",
        value=task_options[selected_task],
        height=120,
        key="zero_shot_cot_prompt"
    )
    
    submit = st.button("Generate Response", type="primary", key="zero_shot_cot_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not user_prompt.strip():
            st.warning("Please enter a prompt first.")
            return
        
        # Create message structure
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        messages = [message]
        
        # System prompts
        system_prompts = [{"text": system_prompt}]
        
        # Response area
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        
        # Stream the response
        try:
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            success = stream_conversation(bedrock_client, model_id, messages, system_prompts, params)
            
            if not success:
                st.error("Failed to generate response.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def few_shot_cot_interface(model_id, params):
    """Interface for few-shot chain of thought prompting."""
    
    st.markdown("<div class='technique-title'>Few-shot Chain of Thought (CoT)</div>", unsafe_allow_html=True)
    st.markdown("<div class='technique-description'>Few-shot Chain of Thought combines example-based learning with explicit reasoning steps. You show the model examples of solving problems step by step, then ask it to solve a new problem. This helps with complex reasoning tasks where both pattern recognition and step-by-step thinking are needed.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Example tasks for few-shot CoT
    task_templates = {
        "Arithmetic Reasoning": """I'll solve some math word problems step by step:

Problem 1: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

Step 1: Roger starts with 5 tennis balls.
Step 2: He buys 2 cans, with 3 tennis balls per can.
Step 3: So he gets 2 × 3 = 6 new tennis balls.
Step 4: In total, he has 5 + 6 = 11 tennis balls.
Answer: 11 tennis balls

Problem 2: A restaurant has 10 tables. Each table has 4 chairs. If half the chairs are occupied, how many people are in the restaurant?

Step 1: The restaurant has 10 tables, each with 4 chairs.
Step 2: Total number of chairs = 10 × 4 = 40 chairs.
Step 3: Half of the chairs are occupied, so 40 × 0.5 = 20 chairs are occupied.
Step 4: Each occupied chair has one person, so there are 20 people.
Answer: 20 people

Problem 3: Mark's weight is 70 kg. Lisa weighs 25% less than Mark. What is Lisa's weight?

""",

        "Symbolic Reasoning": """I'll solve some logic puzzles step by step:

Puzzle 1: Four people (Alex, Blake, Casey, and Dana) each have a different job (teacher, doctor, engineer, artist). Given these clues:
- The artist is not Alex or Casey
- Dana is not the teacher
- Blake is the doctor
Who has which job?

Step 1: Blake is the doctor, so we can mark this as certain.
Step 2: The artist is not Alex or Casey, so the artist must be either Blake or Dana.
Step 3: Since Blake is the doctor, Blake cannot be the artist.
Step 4: Therefore, Dana must be the artist.
Step 5: Dana is not the teacher, and Dana is the artist, so Alex or Casey must be the teacher.
Step 6: We need to determine who between Alex and Casey is the teacher and who is the engineer.
Step 7: Since we have no other information, let's use process of elimination with what we know:
   - Blake: doctor
   - Dana: artist
   - Alex: either teacher or engineer
   - Casey: either teacher or engineer
Step 8: Since we have no further constraints, let's check who's left for the remaining roles:
   - If Alex is the teacher, Casey must be the engineer
   - If Casey is the teacher, Alex must be the engineer
Step 9: Without additional clues, either option is possible. Let's assume Alex is the teacher and Casey is the engineer.
Answer: Alex is the teacher, Blake is the doctor, Casey is the engineer, and Dana is the artist.

Puzzle 2: In a small town, every resident either always tells the truth or always lies. You meet three residents - X, Y, and Z - and ask them questions.
X says: "Y and Z are different types (one truthful, one lying)"
Y says: "X is lying"
Is Z a truth-teller or a liar?

""",

        "Sequential Decision Making": """I'll analyze some decision scenarios step by step:

Scenario 1: You're deciding whether to invest $10,000 in Stock A or Stock B.
- Stock A has returned 5% annually over the past 5 years with low volatility
- Stock B has returned 10% annually but with high volatility
- You need this money in 2 years for a down payment on a house

Step 1: Let's identify the key factors in this decision:
   - Time horizon: 2 years (relatively short)
   - Purpose: Down payment for a house (important, specific goal)
   - Options: Stock A (5% return, low volatility) vs Stock B (10% return, high volatility)

Step 2: Let's consider the potential outcomes with Stock A:
   - With 5% annual return, after 2 years: $10,000 × (1.05)² = $11,025
   - Due to low volatility, the actual return is likely to be close to this expected value

Step 3: Let's consider the potential outcomes with Stock B:
   - With 10% annual return, after 2 years: $10,000 × (1.10)² = $12,100
   - However, due to high volatility, there's significant risk of underperformance or even loss

Step 4: Let's consider the consequences of each choice:
   - If Stock A underperforms slightly, you still likely have most of your down payment
   - If Stock B underperforms significantly, you might need to delay your house purchase

Step 5: Given the short time horizon and specific need for the money, safety of principal should be prioritized over maximizing returns.

Conclusion: Stock A is the better choice despite the lower expected return, because the lower volatility better aligns with your short time horizon and specific goal of having the money available for a house down payment in 2 years.

Scenario 2: A small business owner needs to decide whether to:
- Expand to a second location (costs $100,000 upfront, potential 30% increase in profits)
- Renovate existing store (costs $50,000, potential 15% increase in profits)
- Invest in online presence (costs $30,000, potential 20% increase in profits)
With only $120,000 available to invest, what should they do?

"""
    }
    
    selected_task = st.selectbox("Select Example Type", options=list(task_templates.keys()), key="few_shot_cot_task")
    
    system_prompt = st.text_area(
        "System Prompt",
        value="You are an expert problem-solver that carefully follows the established reasoning pattern in examples to solve new problems step-by-step.",
        height=100,
        key="few_shot_cot_system"
    )
    
    user_prompt = st.text_area(
        "Few-shot CoT Prompt with Examples",
        value=task_templates[selected_task],
        height=400,
        key="few_shot_cot_prompt"
    )
    
    submit = st.button("Generate Response", type="primary", key="few_shot_cot_submit")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not user_prompt.strip():
            st.warning("Please enter a prompt with examples first.")
            return
        
        # Create message structure
        message = {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
        messages = [message]
        
        # System prompts
        system_prompts = [{"text": system_prompt}]
        
        # Response area
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        
        # Stream the response
        try:
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            success = stream_conversation(bedrock_client, model_id, messages, system_prompts, params)
            
            if not success:
                st.error("Failed to generate response.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ------- MAIN APP -------

def main():
    # Header
    st.markdown("<h1 class='main-header'>Advanced LLM Prompting Techniques</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This interactive dashboard demonstrates different prompting techniques for Large Language Models.
    Select a technique tab below to explore how different prompting strategies affect model outputs.
    Adjust model parameters in the sidebar to see their impact on responses.
    """)
    
    # Create a 70/30 layout
    col1, col2 = st.columns([0.7, 0.3])     
        # Get model and parameters from the right column
    with col2:
        model_id, params = parameter_sidebar()  
    
    with col1: 
   
        # Create tabs for different prompting techniques
        tabs = st.tabs([
            "🎯 Zero-shot Prompting", 
            "📋 Few-shot Prompting", 
            "⛓️ Zero-shot Chain of Thought",
            "🧠 Few-shot Chain of Thought"
        ])
        
        # Populate each tab
        with tabs[0]:
            zero_shot_interface(model_id, params)
        
        with tabs[1]:
            few_shot_interface(model_id, params)
        
        with tabs[2]:
            zero_shot_cot_interface(model_id, params)
        
        with tabs[3]:
            few_shot_cot_interface(model_id, params)

if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()

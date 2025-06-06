# Refactored Chain-of-Thought Prompting App

import streamlit as st
import logging
import boto3
from botocore.exceptions import ClientError
import uuid
import utils.common as common
import utils.authenticate as authenticate
# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Page configuration with improved styling
st.set_page_config(
    page_title="Chain-of-Thought Prompting",
    page_icon="🧠",
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
        background: linear-gradient(90deg, #4338CA, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: left;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #2563EB;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: #FFFFFF;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
    }
    .output-container {
        background-color: #F9FAFB;
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        height: 450px;
        overflow-y: auto;
        border: 1px solid #E5E7EB;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .reset-button>button {
        background-color: #DC2626;
    }
    .reset-button>button:hover {
        background-color: #B91C1C;
    }
    .analyze-button>button {
        background-color: #059669;
    }
    .analyze-button>button:hover {
        background-color: #047857;
    }
    .response-block {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
        margin-top: 1rem;
        overflow-wrap: break-word;
    }
    .standard-block {
        border-left: 4px solid #3B82F6;
    }
    .cot-block {
        border-left: 4px solid #8B5CF6;
    }
    .token-metrics {
        display: flex;
        justify-content: space-between;
        background-color: #F0F4F8;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-top: 0.75rem;
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
    .analysis-container {
        background-color: #EFF6FF;
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin-top: 1rem;
        border-left: 4px solid #6366F1;
    }
    .comparison-header {
        padding: 0.75rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-align: center;
        border-bottom: 2px solid #E5E7EB;
    }
    .tab-content {
        padding: 1.25rem;
        height: 100%;
    }
    .github-link {
        text-align: center;
        margin-top: 1.5rem;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F9FAFB;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 16px;
        height: auto;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: white !important;
        font-weight: 600;
    }
    .key-benefit {
        background-color: #F3F4F6;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 3px solid #4F46E5;
    }
    .stSelectbox [data-baseweb=select] {
        border-radius: 8px;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #D1D5DB;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
    }
    .stStatus {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

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

# ------- SAMPLE PROMPTS -------

ZERO_SHOT_PROMPTS = [
    {
        "name": "Math Problem - Basic",
        "prompt": "A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?",
        "cot_suffix": " Think step by step."
    },
    {
        "name": "Logic Puzzle - River Crossing",
        "prompt": "A farmer needs to cross a river with a fox, a chicken, and a sack of grain. The boat can only hold the farmer and one item. The fox can't be left alone with the chicken, and the chicken can't be left alone with the grain. How can the farmer get everything across?",
        "cot_suffix": " Think step by step."
    },
    {
        "name": "Science Question - Physics",
        "prompt": "Why does ice float in water? Explain the scientific principle behind this phenomenon.",
        "cot_suffix": " Think step by step."
    },
    {
        "name": "Ethical Dilemma - AI Monitoring",
        "prompt": "Is it ethical for a company to use AI to monitor its employees' productivity? Consider different stakeholder perspectives.",
        "cot_suffix": " Think step by step about different perspectives."
    },
    {
        "name": "Code Debugging - Python",
        "prompt": "What's wrong with this Python code and how would you fix it?\n\ndef calculate_average(numbers):\n    sum = 0\n    for num in numbers:\n        sum += num\n    return sum / len(numbers)\n\nresult = calculate_average([])",
        "cot_suffix": " Think step by step through the code execution."
    }
]

FEW_SHOT_PROMPTS = [
    {
        "name": "Math Word Problems",
        "prompt": """I'm going to solve some math word problems.

Example 1:
Problem: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
Answer: Roger starts with 5 tennis balls. He buys 2 cans, with 3 tennis balls each. So he gets 2 × 3 = 6 new tennis balls. Now he has 5 + 6 = 11 tennis balls.

Example 2:
Problem: The triangle has a base of 10 inches and a height of 8 inches. What is the area of the triangle?
Answer: The area of a triangle is (1/2) × base × height. So the area is (1/2) × 10 inches × 8 inches = 40 square inches.

Now solve this problem:
James bought 3 packages of chocolate bars. Each package has 6 bars. He ate 4 bars. How many bars does he have left?""",
        
        "cot_suffix": """

To solve this problem, I'll think step by step:

Example 1:
Problem: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
Step 1: Roger starts with 5 tennis balls.
Step 2: He buys 2 cans of tennis balls, with 3 balls per can.
Step 3: The number of new balls is 2 × 3 = 6 tennis balls.
Step 4: The total number of tennis balls is 5 + 6 = 11 tennis balls.
Answer: 11 tennis balls

Example 2:
Problem: The triangle has a base of 10 inches and a height of 8 inches. What is the area of the triangle?
Step 1: The formula for the area of a triangle is (1/2) × base × height.
Step 2: Substitute the values: (1/2) × 10 inches × 8 inches
Step 3: Calculate: (1/2) × 80 square inches = 40 square inches
Answer: 40 square inches

Now for the new problem:
James bought 3 packages of chocolate bars. Each package has 6 bars. He ate 4 bars. How many bars does he have left?"""
    },
    {
        "name": "Logical Reasoning",
        "prompt": """I'm going to solve some logical reasoning problems.

Example 1:
Problem: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?
Answer: No, we cannot conclude that some roses fade quickly. While all roses are flowers, we only know that some flowers fade quickly. The flowers that fade quickly might not be roses.

Example 2:
Problem: If no humans can fly naturally, and all pilots are human, what can we conclude about pilots?
Answer: We can conclude that no pilots can fly naturally. Since all pilots are humans, and no humans can fly naturally, it follows that no pilots can fly naturally.

Now solve this problem:
If all doctors are busy people, and some busy people have stress, can we conclude that some doctors have stress?""",
        
        "cot_suffix": """

I'll solve these logical reasoning problems step by step:

Example 1:
Problem: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?
Step 1: We know that all roses are flowers (All A are B).
Step 2: We know that some flowers fade quickly (Some B are C).
Step 3: To conclude "some roses fade quickly" (Some A are C), we would need to know that the flowers that fade quickly include at least some roses.
Step 4: The information doesn't guarantee that the subset of flowers that fade quickly contains any roses.
Answer: No, we cannot conclude that some roses fade quickly.

Example 2:
Problem: If no humans can fly naturally, and all pilots are human, what can we conclude about pilots?
Step 1: We know that no humans can fly naturally (No A is B).
Step 2: We know that all pilots are humans (All C are A).
Step 3: Combining these statements: if all pilots are humans, and no humans can fly naturally, then no pilots can fly naturally.
Answer: No pilots can fly naturally.

Now for the new problem:
If all doctors are busy people, and some busy people have stress, can we conclude that some doctors have stress?"""
    },
    {
        "name": "Language Translation",
        "prompt": """I'll translate English phrases to French.

Example 1:
English: Hello, how are you today?
French: Bonjour, comment allez-vous aujourd'hui?

Example 2:
English: I would like to order a coffee, please.
French: Je voudrais commander un café, s'il vous plaît.

Now translate this phrase to French:
English: Where is the nearest train station?""",
        
        "cot_suffix": """

I'll translate these English phrases to French step by step:

Example 1:
English: Hello, how are you today?
Step 1: "Hello" in French is "Bonjour"
Step 2: "how are you" is formally translated as "comment allez-vous"
Step 3: "today" in French is "aujourd'hui"
Step 4: Putting it together with proper punctuation: "Bonjour, comment allez-vous aujourd'hui?"
French: Bonjour, comment allez-vous aujourd'hui?

Example 2:
English: I would like to order a coffee, please.
Step 1: "I would like" in French is "Je voudrais"
Step 2: "to order" translates to "commander"
Step 3: "a coffee" is "un café"
Step 4: "please" is "s'il vous plaît"
Step 5: Combining with proper grammar: "Je voudrais commander un café, s'il vous plaît."
French: Je voudrais commander un café, s'il vous plaît.

Now for the new phrase:
English: Where is the nearest train station?"""
    },
    {
        "name": "Medical Diagnosis",
        "prompt": """I'll analyze patient symptoms and provide possible diagnoses.

Example 1:
Patient: 45-year-old male with sudden chest pain radiating to left arm, shortness of breath, and sweating.
Diagnosis: These symptoms strongly suggest a myocardial infarction (heart attack). Immediate medical attention is required. The radiation of pain to the left arm is a classic sign of cardiac origin.

Example 2:
Patient: 8-year-old child with high fever, red rash that starts on face and spreads downward, cough, and red eyes.
Diagnosis: These symptoms are consistent with measles (rubeola). The characteristic rash pattern and associated symptoms suggest this viral infection, especially if the child hasn't been vaccinated.

Now provide a possible diagnosis:
Patient: 35-year-old female with severe headache, sensitivity to light, stiff neck, and fever.""",
        
        "cot_suffix": """

I'll analyze these patient symptoms step by step:

Example 1:
Patient: 45-year-old male with sudden chest pain radiating to left arm, shortness of breath, and sweating.
Step 1: Identify key symptoms: chest pain with left arm radiation, dyspnea (shortness of breath), diaphoresis (sweating).
Step 2: Consider common causes for this constellation of symptoms in this demographic.
Step 3: The combination of chest pain radiating to the left arm, along with shortness of breath and sweating, is highly specific for cardiac ischemia.
Step 4: Given the sudden onset in a middle-aged male, myocardial infarction is the most likely diagnosis.
Diagnosis: Myocardial infarction (heart attack)

Example 2:
Patient: 8-year-old child with high fever, red rash that starts on face and spreads downward, cough, and red eyes.
Step 1: Identify key symptoms: high fever, characteristic rash pattern (face to body), cough, conjunctivitis.
Step 2: Consider common childhood illnesses with these features.
Step 3: The pattern of rash beginning on the face and spreading downward is distinctive for measles.
Step 4: Supporting symptoms (cough, conjunctivitis, fever) further support this diagnosis.
Diagnosis: Measles (rubeola)

Now for the new patient:
Patient: 35-year-old female with severe headache, sensitivity to light, stiff neck, and fever."""
    },
    {
        "name": "Computer Science Algorithms",
        "prompt": """I'll explain algorithms and their time complexity.

Example 1:
Algorithm: Bubble Sort
Explanation: Bubble Sort works by repeatedly stepping through the list, comparing adjacent elements and swapping them if they are in the wrong order. The pass through the list is repeated until no swaps are needed. Its time complexity is O(n²) in the worst and average cases, making it inefficient for large lists.

Example 2:
Algorithm: Binary Search
Explanation: Binary Search operates on a sorted array by repeatedly dividing the search range in half. It compares the middle element with the target value. If they match, the position is returned. If the middle element is greater than the target, the search continues in the lower half; otherwise, the search continues in the upper half. Its time complexity is O(log n), making it very efficient for large datasets.

Now explain this algorithm:
Algorithm: Depth-First Search (DFS)""",
        
        "cot_suffix": """

I'll explain these algorithms step by step:

Example 1:
Algorithm: Bubble Sort
Step 1: Define what the algorithm does: Bubble Sort is a simple comparison-based sorting algorithm.
Step 2: Explain the mechanism: It works by repeatedly traversing the list, comparing adjacent elements and swapping them if they are in the wrong order.
Step 3: Describe the process: In each pass, the largest unsorted element "bubbles up" to its correct position.
Step 4: Analyze time complexity: Each pass requires n-1 comparisons, and we need up to n passes.
Step 5: Conclude with efficiency: This gives O(n²) time complexity in worst and average cases, making it inefficient for large datasets.
Explanation: Bubble Sort repeatedly compares adjacent elements and swaps them if needed, moving the largest elements to the end in each pass. Time complexity: O(n²).

Example 2:
Algorithm: Binary Search
Step 1: Define prerequisites: Binary Search requires a sorted array to function.
Step 2: Explain the core strategy: It uses a divide-and-conquer approach by repeatedly dividing the search space in half.
Step 3: Describe the process: Find the middle element, compare with target, and eliminate half the remaining elements.
Step 4: Analyze time complexity: Each step eliminates half the elements, leading to log₂(n) steps at most.
Step 5: Conclude with efficiency: This gives O(log n) time complexity, making it very efficient for large datasets.
Explanation: Binary Search works on sorted arrays by repeatedly dividing the search interval in half. Time complexity: O(log n).

Now for the new algorithm:
Algorithm: Depth-First Search (DFS)"""
    }
]

# ------- SESSION STATE INITIALIZATION -------

def init_session_state():
    """Initialize session state variables."""
        
    # Zero-shot state variables
    if 'zero_shot_prompt' not in st.session_state:
        st.session_state.zero_shot_prompt = ZERO_SHOT_PROMPTS[0]["prompt"]
    if 'zero_shot_system_prompt' not in st.session_state:
        st.session_state.zero_shot_system_prompt = "You are a helpful and knowledgeable assistant who provides accurate information."
    if 'zero_shot_standard_response' not in st.session_state:
        st.session_state.zero_shot_standard_response = None
    if 'zero_shot_cot_response' not in st.session_state:
        st.session_state.zero_shot_cot_response = None
    
    # Few-shot state variables
    if 'few_shot_prompt' not in st.session_state:
        st.session_state.few_shot_prompt = FEW_SHOT_PROMPTS[0]["prompt"]
    if 'few_shot_cot_prompt' not in st.session_state:
        st.session_state.few_shot_cot_prompt = FEW_SHOT_PROMPTS[0]["prompt"] + FEW_SHOT_PROMPTS[0]["cot_suffix"]
    if 'few_shot_system_prompt' not in st.session_state:
        st.session_state.few_shot_system_prompt = "You are a helpful and knowledgeable assistant who provides accurate information."
    if 'few_shot_standard_response' not in st.session_state:
        st.session_state.few_shot_standard_response = None
    if 'few_shot_cot_response' not in st.session_state:
        st.session_state.few_shot_cot_response = None
    
    # Common state variables
    if 'analysis_shown' not in st.session_state:
        st.session_state.analysis_shown = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Zero-Shot"

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
    with st.sidebar:
        common.render_sidebar()
        
        with st.expander("About Chain-of-Thought", expanded=False):
            st.markdown("""
            ### Chain-of-Thought (CoT) Prompting
            
            CoT prompting is a technique that enhances reasoning in AI models by 
            encouraging them to generate step-by-step explanations before providing
            the final answer.
            
            **Key Benefits:**
            - Improves reasoning abilities in foundation models
            - Addresses multi-step problem-solving challenges
            - Generates intermediate reasoning steps, mimicking human thought
            - Enhances model performance compared to standard methods
            
            **Types of CoT Prompting:**
            
            1. **Zero-Shot CoT**: Adding "Think step by step" or similar instruction
            
            2. **Few-Shot CoT**: Showing examples of step-by-step reasoning before the query
            """)
        
        params = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
        
    return model_id, params

# def display_sample_prompts(prompts, current_prompt, key_prefix):
#     """Display sample prompts as a selectbox."""
#     prompt_names = [p["name"] for p in prompts]
#     selected_name = st.selectbox(
#         "Select a sample prompt:", 
#         options=prompt_names, 
#         key=f"{key_prefix}_select",
#         index=0 if current_prompt == prompts[0]["prompt"] else 
#             prompt_names.index(next((p["name"] for p in prompts if p["prompt"] == current_prompt), prompt_names[0]))
#     )
    
#     selected_prompt = next(p for p in prompts if p["name"] == selected_name)
#     return selected_prompt

def display_sample_prompts(prompts, current_prompt, key_prefix):
    """Display sample prompts as a selectbox."""
    prompt_names = [p["name"] for p in prompts]
    
    # Find the current index
    current_index = 0
    for i, p in enumerate(prompts):
        if p["prompt"] == current_prompt:
            current_index = i
            break
    
    # Create the selectbox
    selected_name = st.selectbox(
        "Select a sample prompt:", 
        options=prompt_names, 
        key=f"{key_prefix}_select",
        index=current_index
    )
    
    # Return the selected prompt data
    selected_prompt = next(p for p in prompts if p["name"] == selected_name)
    return selected_prompt



def display_responses(standard_response, cot_response):
    """Display the standard and CoT responses side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='comparison-header'>Standard Prompting</div>", unsafe_allow_html=True)
        # st.markdown("<div class='output-container'>", unsafe_allow_html=True)
        
        if standard_response:
            output_message = standard_response['output']['message']
            for content in output_message['content']:
                st.markdown(content['text'])
                
            token_usage = standard_response['usage']
            st.caption(f"Input: {token_usage['inputTokens']} | Output: {token_usage['outputTokens']} | Total: {token_usage['totalTokens']}")
        else:
            st.caption("Response will appear here...")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='comparison-header'>Chain-of-Thought Prompting</div>", unsafe_allow_html=True)
        # st.markdown("<div class='output-container'>", unsafe_allow_html=True)
        
        if cot_response:
            output_message = cot_response['output']['message']
            for content in output_message['content']:
                st.markdown(content['text'])
                
            token_usage = cot_response['usage']
            st.caption(f"Input: {token_usage['inputTokens']} | Output: {token_usage['outputTokens']} | Total: {token_usage['totalTokens']}")
        else:
            st.caption("Response will appear here...")
            
        st.markdown("</div>", unsafe_allow_html=True)

def display_analysis(standard_response, cot_response, user_prompt, analysis_shown):
    """Display analysis of the differences between standard and CoT responses."""
    if standard_response and cot_response and analysis_shown:
        st.markdown("### Response Analysis")
        st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
        
        # Extract responses
        standard_text = standard_response['output']['message']['content'][0]['text']
        cot_text = cot_response['output']['message']['content'][0]['text']
        
        # Get token usage metrics
        standard_tokens = standard_response['usage']
        cot_tokens = cot_response['usage']
        
        # Create a prompt to analyze the differences
        analysis_prompt = f"""
        Analyze the following two AI responses to the query: "{user_prompt}"
        
        RESPONSE 1 (Standard prompting):
        {standard_text}
        
        RESPONSE 2 (Chain-of-Thought prompting):
        {cot_text}
        
        Please compare these responses considering:
        1. Depth of reasoning
        2. Clarity of explanation
        3. Accuracy of information (if applicable)
        4. Structure and organization
        5. Key differences in approach
        
        Provide a concise, balanced analysis highlighting the strengths and weaknesses of each approach.
        """
        
        try:
            # Get Bedrock client
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            
            # System prompt for analysis
            system_prompt = [{"text": "You are an expert in analyzing AI responses and prompt engineering. Provide clear, insightful, and balanced comparisons."}]
            
            # Message structure
            message = {
                "role": "user",
                "content": [{"text": analysis_prompt}]
            }
            messages = [message]
            
            # Get analysis from the model
            analysis_params = {
                "temperature": 0.3,
                "topP": 0.9,
                "maxTokens": 1500
            }
            
            analysis_response = text_conversation(
                bedrock_client, 
                "anthropic.claude-3-sonnet-20240229-v1:0", 
                system_prompt, 
                messages, 
                **analysis_params
            )
            
            if analysis_response:
                analysis_text = analysis_response['output']['message']['content'][0]['text']
                st.markdown(analysis_text)
            else:
                st.error("Failed to generate analysis. Please try again.")
                
        except Exception as e:
            st.error(f"Error generating analysis: {str(e)}")
        
        # Display token metrics comparison
        st.markdown("### Token Usage Comparison")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Input Tokens",
                value=standard_tokens['inputTokens'],
                delta=cot_tokens['inputTokens'] - standard_tokens['inputTokens'],
                delta_color="inverse"
            )
            
        with col2:
            st.metric(
                label="Output Tokens",
                value=standard_tokens['outputTokens'],
                delta=cot_tokens['outputTokens'] - standard_tokens['outputTokens'],
                delta_color="inverse"
            )
            
        with col3:
            st.metric(
                label="Total Tokens",
                value=standard_tokens['totalTokens'],
                delta=cot_tokens['totalTokens'] - standard_tokens['totalTokens'],
                delta_color="inverse"
            )
            
        st.caption("Note: Delta shows difference between CoT and Standard (CoT - Standard)")
        
        st.markdown("</div>", unsafe_allow_html=True)

def zero_shot_tab(model_id, params):
    """Content for Zero-Shot tab."""
    with st.expander("Learn more about Zero-Shot Chain-of-Thought", expanded=False):
        st.markdown("### Zero-Shot Chain-of-Thought")
        st.markdown("""
        <div class="key-benefit">
        Zero-Shot CoT simply adds an instruction like "Think step by step" to the prompt, 
        encouraging the model to break down its reasoning process without providing examples.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Key Benefits:**
        - ✅ Minimal additional prompt engineering required
        - ✅ Works surprisingly well with capable models
        - ✅ No need to craft elaborate examples
        - ✅ Can be applied to virtually any reasoning task
        """)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt", 
        value=st.session_state.zero_shot_system_prompt,
        height=80,
        help="This defines how the AI assistant should behave",
        key="zero_shot_system"
    )
    st.session_state.zero_shot_system_prompt = system_prompt
    
    # Display sample prompts
    selected_prompt = display_sample_prompts(
        ZERO_SHOT_PROMPTS, 
        st.session_state.zero_shot_prompt,
        "zero_shot"
    )
    
    
    # Explanation of Zero-Shot CoT
    # User prompt input
    user_prompt = st.text_area(
        "Prompt", 
        value=selected_prompt["prompt"],
        height=120,
        placeholder="Enter your question or task here...",
        key="zero_shot_prompt"
    )
    # st.session_state.zero_shot_prompt = user_prompt
    
    # CoT suffix
    cot_suffix = selected_prompt["cot_suffix"]
    
    # Generate responses and analyze buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_button = st.button(
            "📝 Generate Responses",
            type="primary",
            key="zero_shot_generate",
            disabled=st.session_state.processing or not user_prompt.strip()
        )
    
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Differences",
            key="zero_shot_analyze",
            disabled=not (st.session_state.zero_shot_standard_response and st.session_state.zero_shot_cot_response) or st.session_state.processing,
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process generate button click
    if generate_button and user_prompt.strip() and not st.session_state.processing:
        st.session_state.processing = True
        
        # Standard prompt processing
        with st.status("Generating standard response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for standard prompt
                system_prompts = [{"text": system_prompt}]
                standard_message = {
                    "role": "user",
                    "content": [{"text": user_prompt}]
                }
                standard_messages = [standard_message]
                
                # Send request to the model
                st.session_state.zero_shot_standard_response = text_conversation(
                    bedrock_client, model_id, system_prompts, standard_messages, **params
                )
                
                status.update(label="Standard response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        # CoT prompt processing
        with st.status("Generating Chain-of-Thought response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for CoT prompt
                system_prompts = [{"text": system_prompt}]
                cot_message = {
                    "role": "user",
                    "content": [{"text": user_prompt + cot_suffix}]
                }
                cot_messages = [cot_message]
                
                # Send request to the model
                st.session_state.zero_shot_cot_response = text_conversation(
                    bedrock_client, model_id, system_prompts, cot_messages, **params
                )
                
                status.update(label="Chain-of-Thought response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        st.session_state.processing = False
        st.rerun()
    
    # Process analyze button click
    if analyze_button and st.session_state.zero_shot_standard_response and st.session_state.zero_shot_cot_response:
        st.session_state.analysis_shown = True
        st.rerun()
    
    # Display the responses
    display_responses(st.session_state.zero_shot_standard_response, st.session_state.zero_shot_cot_response)
    
    # Display analysis if available
    display_analysis(
        st.session_state.zero_shot_standard_response, 
        st.session_state.zero_shot_cot_response, 
        user_prompt, 
        st.session_state.analysis_shown
    )

def few_shot_tab(model_id, params):
    """Content for Few-Shot tab."""
    with st.expander("Learn more about Few-Shot Chain-of-Thought", expanded=False):
        st.markdown("### Few-Shot Chain-of-Thought")
        st.markdown("""
        <div class="key-benefit">
        Few-Shot CoT provides explicit examples of step-by-step reasoning before asking the model 
        to solve a new problem. This approach demonstrates the expected reasoning pattern.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Key Benefits:**
        - ✅ Creates a clear pattern for the model to follow
        - ✅ Demonstrates the depth of reasoning expected
        - ✅ Can guide specific reasoning styles or approaches
        - ✅ Often produces more consistent reasoning quality
        """)   
    
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt", 
        value=st.session_state.few_shot_system_prompt,
        height=80,
        help="This defines how the AI assistant should behave",
        key="few_shot_system"
    )
    st.session_state.few_shot_system_prompt = system_prompt
    
    # Display sample prompts
    selected_prompt = display_sample_prompts(
        FEW_SHOT_PROMPTS, 
        st.session_state.few_shot_prompt,
        "few_shot"
    )

    # Few-shot prompts (standard and CoT)
    st.markdown("#### Standard Few-Shot")
    few_shot_prompt = st.text_area(
        "Standard Few-Shot Prompt",
        value=selected_prompt["prompt"],
        height=300,
        key="few_shot_prompt_standard"
    )
    st.session_state.few_shot_prompt = few_shot_prompt
    
    st.markdown("#### Few-Shot with Chain-of-Thought")
    few_shot_cot_prompt = st.text_area(
        "Few-Shot CoT Prompt",
        value=selected_prompt["prompt"] + selected_prompt["cot_suffix"],
        height=300,
        key="few_shot_prompt_cot"
    )
    st.session_state.few_shot_cot_prompt = few_shot_cot_prompt
    
    # Generate responses and analyze buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_button = st.button(
            "📝 Generate Responses",
            type="primary",
            key="few_shot_generate",
            disabled=st.session_state.processing or not few_shot_prompt.strip()
        )
    
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Differences",
            key="few_shot_analyze",
            disabled=not (st.session_state.few_shot_standard_response and st.session_state.few_shot_cot_response) or st.session_state.processing,
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process generate button click
    if generate_button and few_shot_prompt.strip() and not st.session_state.processing:
        st.session_state.processing = True
        
        # Standard prompt processing
        with st.status("Generating standard few-shot response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for standard prompt
                system_prompts = [{"text": system_prompt}]
                standard_message = {
                    "role": "user",
                    "content": [{"text": few_shot_prompt}]
                }
                standard_messages = [standard_message]
                
                # Send request to the model
                st.session_state.few_shot_standard_response = text_conversation(
                    bedrock_client, model_id, system_prompts, standard_messages, **params
                )
                
                status.update(label="Standard few-shot response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        # CoT prompt processing
        with st.status("Generating few-shot Chain-of-Thought response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for CoT prompt
                system_prompts = [{"text": system_prompt}]
                cot_message = {
                    "role": "user",
                    "content": [{"text": few_shot_cot_prompt}]
                }
                cot_messages = [cot_message]
                
                # Send request to the model
                st.session_state.few_shot_cot_response = text_conversation(
                    bedrock_client, model_id, system_prompts, cot_messages, **params
                )
                
                status.update(label="Few-shot Chain-of-Thought response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        st.session_state.processing = False
        st.rerun()
    
    # Process analyze button click
    if analyze_button and st.session_state.few_shot_standard_response and st.session_state.few_shot_cot_response:
        st.session_state.analysis_shown = True
        st.rerun()
    
    # Display the responses
    display_responses(st.session_state.few_shot_standard_response, st.session_state.few_shot_cot_response)
    
    # Display analysis if available
    display_analysis(
        st.session_state.few_shot_standard_response, 
        st.session_state.few_shot_cot_response, 
        "Custom few-shot prompt", 
        st.session_state.analysis_shown
    )

def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("<h1 class='main-header'>Chain-of-Thought Prompting</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style="text-align:left; font-size:1.2rem; margin-bottom:2rem;">
    Explore and compare how different Chain-of-Thought prompting techniques improve the reasoning 
    capabilities of large language models.
    </p>
    """, unsafe_allow_html=True)
    
    
    # Create a 70/30 layout
    col1, col2 = st.columns([0.7, 0.3])     
        # Get model and parameters from the right column
    with col2:
        model_id, params = parameter_sidebar()

    with col1:
        # Create tabs for zero-shot and few-shot
        tab1, tab2 = st.tabs(["🎯 Zero-Shot", "🔄 Few-Shot"])
        
        # Populate each tab
        with tab1:
            zero_shot_tab(model_id, params)
        
        with tab2:
            few_shot_tab(model_id, params)
        

    # Footer
    st.markdown("---")
    st.markdown("<div class='footer'>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()

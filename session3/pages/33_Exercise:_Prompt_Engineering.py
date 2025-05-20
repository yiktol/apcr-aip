
import streamlit as st
import boto3
import logging
import time
import re
import json
from typing import Dict, List, Any, Tuple, Optional
import utils.common as common

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Prompt Engineering Exercises",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

common.initialize_session_state()

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    
    .exercise-container {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .output-container {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin-top: 1rem;
        border-left: 5px solid #4CAF50;
    }
    
    .expected-container {
        padding: 1rem;
        border-radius: 8px;
        background-color: #e8f5e9;
        margin-top: 1rem;
        border-left: 5px solid #2E7D32;
    }
    
    .comparison-header {
        color: #2E7D32;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    
    .reset-button>button {
        background-color: #f44336;
        color: white;
    }
    
    h1, h2, h3 {
        color: #1e3a8a;
    }
    
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1e3a8a;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'tutorial_responses' not in st.session_state:
        st.session_state.tutorial_responses = {}
    
    if 'current_exercise' not in st.session_state:
        st.session_state.current_exercise = None
    
    if 'response_visible' not in st.session_state:
        st.session_state.response_visible = False

# Reset session state
def reset_session():
    st.session_state.messages = []
    st.session_state.tutorial_responses = {}
    st.session_state.current_exercise = None
    st.session_state.response_visible = False

# Parameter sidebar function
def parameter_sidebar():
    """Sidebar with model selection and parameter tuning."""
    with st.container(border=True):
     
        st.subheader("Model Selection")
        MODEL_CATEGORIES = {
            "Anthropic": ["anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-v2:1"],
            "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"],
            "Meta": ["meta.llama3-70b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0"],
            "Mistral": ["mistral.mistral-large-2402-v1:0", "mistral.mistral-small-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1"]
        }
        
        # Create selectbox for provider first
        provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()))
        
        # Then create selectbox for models from that provider
        model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider])
        
        st.subheader("Parameter Tuning")
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, 
                            help="Higher values make output more random, lower values more deterministic")
        
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1,
                            help="Controls diversity via nucleus sampling")
        
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4096, value=1024, step=50,
                                    help="Maximum number of tokens in the response")
    
    with st.sidebar:    
        common.render_sidebar()
        
        with st.expander("About", expanded=False):
            st.markdown("""
            ### Prompt Engineering Practice
            This app helps you practice prompt engineering techniques based on the Prompt Engineering Playbook.
            
            Each exercise focuses on a specific skill like rewriting, extracting, classifying, clustering, summarizing, or generating text.
            """)
        
        params = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
        
    return model_id, params

# Function to stream conversation with the LLM
def stream_conversation(bedrock_client, model_id, prompt, system_prompt=None, inference_config=None):
    """Stream a response from the model using the Converse API"""
    
    if inference_config is None:
        inference_config = {
            "temperature": 0.7,
            "topP": 0.9,
            "maxTokens": 1024
        }
    
    # Create message structure
    message = {
        "role": "user",
        "content": [{"text": prompt}]
    }
    messages = [message]
    
    # System prompts
    if system_prompt is None:
        system_prompt = "You are a prompt engineering assistant helping users learn how to write effective prompts."
    
    system_prompts = [{"text": system_prompt}]
    
    try:
        # Call the streaming API
        response = bedrock_client.converse_stream(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config
        )
        
        stream = response.get('stream')
        if stream:
            placeholder = st.empty()
            full_response = ''
            token_info = {'input': 0, 'output': 0, 'total': 0}
            
            for event in stream:
                if 'contentBlockDelta' in event:
                    chunk = event['contentBlockDelta']
                    part = chunk['delta']['text']
                    full_response += part
                    with placeholder.container():
                        st.markdown(full_response)
                
                if 'metadata' in event:
                    metadata = event['metadata']
                    if 'usage' in metadata:
                        usage = metadata['usage']
                        token_info = {
                            'input': usage['inputTokens'],
                            'output': usage['outputTokens'],
                            'total': usage['totalTokens']
                        }
            
            # Update session state with the response
            if st.session_state.current_exercise:
                st.session_state.tutorial_responses[st.session_state.current_exercise] = full_response
            
            # Display token usage
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Tokens", token_info['input'])
            with col2:
                st.metric("Output Tokens", token_info['output'])
            with col3:
                st.metric("Total Tokens", token_info['total'])
            
            return full_response
        
    except Exception as e:
        st.error(f"Error communicating with the model: {str(e)}")
        logger.error(f"Error in stream_conversation: {str(e)}")
        return None

# Define the tutorial exercises
def get_tutorials():
    tutorials = {
        "tutorial_1_rewriting": {
            "title": "Exercise 1: Rewriting",
            "description": "Practice rewriting and translation tasks using prompt engineering.",
            "exercises": [
                {
                    "id": "t1_q1",
                    "title": "Quote Translation",
                    "description": """
                    Pick one of the quotes below and translate it into another language that you're familiar with.
                    
                    * A rose by any other name would smell as sweet. By William Shakespeare
                    * That's one small step for a man, a giant leap for mankind. By Neil Armstrong
                    * Speak softly and carry a big stick. By Theodore Roosevelt
                    * Life is like riding a bicycle. To keep your balance, you must keep moving. By Albert Einstein
                    """,
                    "hint": "Use a prompt that specifies the exact quote and target language.",
                    "expected_result": """
                    Example for Chinese translation:
                    "那是一个人的小步，却是人类的巨大飞跃" - 尼尔·阿姆斯特朗
                    
                    Note: Your result may differ depending on the quote and language you chose.
                    """
                },
                {
                    "id": "t1_q2",
                    "title": "Simplification for Children",
                    "description": """
                    We want a 6-year-old to be able to understand this text about digital government:
                    
                    Digitalisation is a key pillar of the Government's public service transformation efforts. The Digital Government Blueprint (DGB) is a statement of the Government's ambition to better leverage data and harness new technologies, and to drive broader efforts to build a digital economy and digital society, in support of Smart Nation.
                    
                    Our vision is to create a Government that is "Digital to the Core, and Serves with Heart". A Digital Government will be able to build stakeholder-centric services that cater to citizens' and businesses' needs. Transacting with a Digital Government will be easy, seamless and secure. Our public officers will be able to continually upskill themselves, adapt to new challenges and work more effectively across agencies as well as with our citizens and businesses.
                    """,
                    "hint": "Ask for a rewrite that uses simple vocabulary, includes examples and analogies a child would understand.",
                    "expected_result": """
                    The Government wants to use computers and new technology to make things easier and better for everyone. They have a plan called the Digital Government Blueprint that aims to create a government that is "Digital to the Core, and Serves with Heart". 
                    
                    This means that they want to make it easy for people to do things like pay bills or get information online, just like how you can play games on your tablet. They also want to use technology to keep us safe, like how you wear a helmet when you ride your bike. The Government is always learning and trying new things to make things better for us, just like how you learn new things in school.
                    """
                },
                {
                    "id": "t1_q3",
                    "title": "Typo Correction",
                    "description": """
                    The text below is to be printed in the papers tomorrow. Find out what typos it may have, and correct them:
                    
                    Product Transummariser is launching today!
                    
                    We are excited to annouce the lauch of our new product that revoluzionizes the way meetings are conducted and documented. Our product transcribes meetings in real-time and provides a concize summary of the discussion points, action items, and decisions made during the meeting. This summary is then automaticaly emailed to all team members, ensuring that everyone was on the same page and has a clear understanding of what was discused. With this product, meetings are more effecient, productive, and inclusive, allowing teams to focus on what really matters - achieving their goals.
                    """,
                    "hint": "Ask the model to correct the spelling and grammatical errors, and then list the changes made.",
                    "expected_result": """
                    Product Transummariser is launching today!
                    
                    We are excited to announce the launch of our new product that revolutionizes the way meetings are conducted and documented. Our product transcribes meetings in real-time and provides a concise summary of the discussion points, action items, and decisions made during the meeting. This summary is then automatically emailed to all team members, ensuring that everyone is on the same page and has a clear understanding of what was discussed. With this product, meetings are more efficient, productive, and inclusive, allowing teams to focus on what really matters - achieving their goals.
                    
                    Changes:
                    • "annouce" changed to "announce"
                    • "lauch" changed to "launch"
                    • "revoluzionizes" changed to "revolutionizes"
                    • "concize" changed to "concise"
                    • "automaticaly" changed to "automatically"
                    • "was" changed to "is" (agreement with "everyone")
                    • "discused" changed to "discussed"
                    • "effecient" changed to "efficient"
                    """
                }
            ]
        },
        "tutorial_2_extracting": {
            "title": "Exercise 2: Extracting",
            "description": "Practice extracting specific information from text.",
            "exercises": [
                {
                    "id": "t2_q1",
                    "title": "Name Formatting",
                    "description": """
                    Extract and format the following list of names into a table with columns for Last Name, First Name, and the combination:
                    
                    John Smith
                    Emma Johnson
                    Michael Davis
                    Olivia Williams
                    Abbey Thompson
                    """,
                    "hint": "Ask the model to extract the names and format them in a table with specific columns.",
                    "expected_result": """
                    | Last Name | First Name | Last Name + First Name |
                    |-----------|------------|------------------------|
                    | Smith     | John       | Smith, John            |
                    | Johnson   | Emma       | Johnson, Emma          |
                    | Davis     | Michael    | Davis, Michael         |
                    | Williams  | Olivia     | Williams, Olivia       |
                    | Thompson  | Abbey      | Thompson, Abbey        |
                    """
                },
                {
                    "id": "t2_q2",
                    "title": "Criminal Records",
                    "description": """
                    Extract specific information about criminals from this police report:
                    
                    Three individuals have been arrested in connection with a series of thefts in the downtown area. The suspects have been identified as John Smith, a 25-year-old male from the United States, Maria Gomez, a 32-year-old female from Mexico, and Ahmed Ali, a 40-year-old male from Egypt.
                    
                    According to police reports, Smith was caught stealing electronics from a local store. He was found to have stolen several laptops and smartphones, with a total value of approximately $10,000. Smith has been charged with burglary and grand theft.
                    
                    Gomez was arrested for stealing jewelry from a high-end boutique. She was found to have taken several valuable pieces, including diamond necklaces and bracelets, with a total value of approximately $50,000. Gomez has been charged with grand theft and possession of stolen property.
                    
                    Ali was apprehended for stealing cash from a local bank. He was found to have broken into the vault and taken several bags of cash, with a total value of approximately $100,000. Ali has been charged with burglary and grand theft.
                    """,
                    "hint": "Ask the model to extract information about each criminal including their name, age, sex, nationality, and details of their crime.",
                    "expected_result": """
                    Name of Criminal: John Smith
                    Age: 25
                    Sex: Male
                    Nationality: United States
                    Crime (include items stolen): Stealing electronics (burglary and grand theft) - laptops and smartphones worth approximately $10,000
                    
                    Name of Criminal: Maria Gomez
                    Age: 32
                    Sex: Female
                    Nationality: Mexico
                    Crime (include items stolen): Stealing jewelry (grand theft and possession of stolen property) - diamond necklaces and bracelets worth approximately $50,000
                    
                    Name of Criminal: Ahmed Ali
                    Age: 40
                    Sex: Male
                    Nationality: Egypt
                    Crime (include items stolen): Stealing cash from a bank (burglary and grand theft) - several bags of cash worth approximately $100,000
                    """
                },
                {
                    "id": "t2_q3",
                    "title": "Action Items from Report",
                    "description": """
                    Extract action items from this trip report:
                    
                    Trip Report:
                    I am writing to provide a summary of our recent visit to the ABC Synapse facility on [Date]. The purpose of the trip was to gain a deeper understanding of how ABC Synapse builds and sells electric vehicles (EVs) and to explore potential collaboration opportunities.
                    
                    During the visit, we had the opportunity to tour the manufacturing facility, meet with key personnel, and discuss the company's approach to EV production and sales. The following are some key observations and insights from the trip:
                    
                    • Manufacturing Process: ABC Synapse has a highly automated and efficient production line, utilizing advanced robotics and AI-driven systems to optimize the assembly of their electric vehicles. This has resulted in reduced production times and increased output capacity. Let's make a note here to conduct a detailed analysis of ABC Synapse's manufacturing process to identify best practices and potential areas for improvement in our own production facilities.
                    
                    • Battery Technology: The company has invested heavily in research and development to create high-performance, long-lasting batteries for their EVs. Their proprietary battery technology offers extended range and faster charging times compared to competitors.
                    
                    • Sales and Distribution: ABC Synapse has adopted a direct-to-consumer sales model, bypassing traditional dealerships. This allows them to maintain better control over the customer experience and offer competitive pricing.
                    
                    Given our company's push for EV production, let's schedule a follow-up meeting with ABC Synapse's R&D team to discuss potential collaboration on battery technology and explore opportunities for joint research projects. We will also need to review our current sales and distribution strategies to determine if adopting a direct-to-consumer model similar to ABC Synapse's approach would be beneficial for our company.
                    """,
                    "hint": "Ask the model to identify and extract all action items mentioned in the report.",
                    "expected_result": """
                    1. Conduct a detailed analysis of ABC Synapse's manufacturing process to identify best practices and potential areas for improvement in our own production facilities.
                    
                    2. Schedule a follow-up meeting with ABC Synapse's R&D team to discuss potential collaboration on battery technology and explore opportunities for joint research projects.
                    
                    3. Review our current sales and distribution strategies to determine if adopting a direct-to-consumer model similar to ABC Synapse's approach would be beneficial for our company.
                    """
                }
            ]
        },
        "tutorial_3_clustering": {
            "title": "Exercise 3: Clustering",
            "description": "Practice clustering and categorizing information.",
            "exercises": [
                {
                    "id": "t3_q1",
                    "title": "Items Clustering",
                    "description": """
                    Form meaningful clusters from this list of items:
                    
                    1) "The Great Gatsby" by F. Scott Fitzgerald
                    2) "Thriller" by Michael Jackson
                    3) "To Kill a Mockingbird" by Harper Lee
                    4) "Giraffe"
                    5) "1984" by George Orwell
                    6) "My Heart Will Go On" by Celine Dion
                    7) "Dolphin"
                    8) "Elephant"
                    9) "Hey Jude" by The Beatles
                    10) "Cosmos" by Carl Sagan
                    11) "The Hobbit" by J.R.R. Tolkien
                    12) "Cheetah"
                    13) "Octopus"
                    14) "Hotel California" by The Eagles
                    15) "Penguin"
                    """,
                    "hint": "Ask the model to group these items into logical categories (animals, books, songs).",
                    "expected_result": """
                    Group 1 (Animals):  
                    • Giraffe
                    • Dolphin
                    • Elephant
                    • Cheetah
                    • Octopus
                    • Penguin
                    
                    Group 2 (Songs):
                    • Thriller by Michael Jackson
                    • My Heart Will Go On by Celine Dion
                    • Hey Jude by The Beatles
                    • Hotel California by The Eagles
                    
                    Group 3 (Books):
                    • The Great Gatsby by F. Scott Fitzgerald
                    • To Kill a Mockingbird by Harper Lee
                    • 1984 by George Orwell
                    • Cosmos by Carl Sagan
                    • The Hobbit by J.R.R. Tolkien
                    """
                },
                {
                    "id": "t3_q2",
                    "title": "Company Meeting Reports Clustering",
                    "description": """
                    Cluster the following company meeting reports based on follow-up actions and interest levels:
                    
                    1. ABC Tech: Our representative met with ABC Tech to discuss the integration of their video analytics solutions into our organization's security framework. Although their AI-driven technology shows potential in enhancing surveillance systems, concerns were raised regarding the privacy implications of implementing such solutions. Both parties agreed to conduct further research to address these concerns before moving forward with any collaboration. We also asked them to show documentation of compliance to local laws.
                    
                    2. DEF Synapse: Our team met with DEF Synapse to explore the application of their innovative Deep Neural Networks for enhancing our data analysis processes. While their AI technology appeared promising, the complexity of integrating it into our existing systems raised concerns about the feasibility and required resources. They did not seem to support implementation on the cloud. Both parties agreed to continue discussions but will re-evaluate the practicality of this potential partnership.
                    
                    3. GHI Robotics: We engaged with GHI Robotics to discuss the potential incorporation of their cutting-edge manufacturing robotics into our production facilities. Unfortunately, the high costs associated with the implementation of their advanced robotic systems led to doubts about the overall return on investment. When we probed them about the recent incident where the robot malfunctioned and crushed a worker's arm, they quickly downplayed that as a one-off unfortunate accident, and the bugs have been fixed. To be cautious, our team decided to postpone any collaboration and explore alternative solutions.
                    """,
                    "hint": "Ask the model to cluster these meetings into two groups - those requiring follow-up actions and those with lower interest to proceed.",
                    "expected_result": """
                    Group 1 (Follow-up Actions):
                    1. ABC Tech: Integration of video analytics solutions; concerns about privacy implications; further research needed; documentation of compliance requested.
                    
                    Group 2 (Lower Interest to Proceed):
                    2. DEF Synapse: Application of Deep Neural Networks; concerns about complexity, feasibility, and required resources; no support for cloud implementation; re-evaluating practicality.
                    3. GHI Robotics: Incorporation of manufacturing robotics; high costs and doubts about return on investment; concerns about recent robot malfunction incident; exploring alternative solutions.
                    """
                }
            ]
        },
        "tutorial_4_classifying": {
            "title": "Exercise 4: Classifying",
            "description": "Practice classifying information into specific categories.",
            "exercises": [
                {
                    "id": "t4_q1",
                    "title": "Course Feedback Classification",
                    "description": """
                    Classify the following course feedback responses as positive, negative, or neutral:
                    
                    Feedback 1 - "Mr. Baker Ree's cooking course was a great experience! The course was affordable, and the cooking school had excellent equipment, especially the very cool oven. I attended other courses before that charged an arm and a leg for just learning one or two recipes. This one had ten!"
                    
                    Feedback 2 - "Hygiene – 1 star. Food variety – 4 stars. Trainer – 0 stars. Mr. Baker Ree's cooking course was a great value for the money. The school had modern equipment, and people are generally nice. The location at Orchard also make it very accessible. We also learned how to make a wide variety of dishes."
                    
                    Feedback 3 - "Aiyo … I didn't enjoy Mr. Baker Ree's cooking course. The trainer laughs at us when we make mistakes. Can like that meh? Very unprofessional. Ask him questions, he also never answer. Then we don't know so when he ask, I cannot answer. Then I got laughed at. Siao. Won't go again."
                    """,
                    "hint": "Ask the model to classify each feedback as positive, negative, or neutral, and provide a summary of the key points for each group.",
                    "expected_result": """
                    Group 1 (Positive): Feedback 1
                    • Affordable course with a good number of recipes taught
                    • Excellent equipment
                    • Good value compared to other courses
                    
                    Group 2 (Negative): Feedback 3
                    • Unprofessional trainer who laughs at mistakes
                    • Trainer doesn't answer questions
                    • Overall negative experience, won't attend again
                    
                    Group 3 (Neutral): Feedback 2
                    • Mixed ratings (low for hygiene and trainer, high for food variety)
                    • Good value for money and modern equipment
                    • Accessible location
                    • Wide variety of dishes taught
                    """
                },
                {
                    "id": "t4_q2",
                    "title": "Department Classification for Feedback",
                    "description": """
                    Classify which feedback goes to which department for follow-up:
                    
                    Feedback 1 - I was walking past Block 52 last night. I stepped onto a banana peel and fell down. There were a lot of broken glass bottles on the floor which cut me. Please do something about this, as a lot of people are drinking in the area, and leaving behind their stuff.
                    
                    Feedback 2 - There is constant drilling sounds from late midnight to 3am. It is coming from my neighbour above. Can you do something about it?
                    
                    Feedback 3 - I was coming home last night and something felt funny. I thought my gas stove was on, but it wasn't. I leaned out of my window, and I think there's some weird stench coming from the floor below mine. May be a decomposing corpse. Can you investigate?
                    
                    Feedback 4 - There is a strong stench coming from the rubbish chute for the past week. Is nobody clearing it?
                    
                    Feedback 5 - A lot of people are leaving their beer bottles all around the void deck. It is very unsightly.
                    
                    Feedback 6 - I hear sounds of bouncing balls from the floor above. They are very inconsiderate. What can I do?
                    
                    Additional Info:
                    Littering issues go to Department L
                    Smell issues go to Department S
                    Noise issues go to Department N
                    """,
                    "hint": "Ask the model to classify each feedback to the appropriate department based on the issue type (littering, smell, or noise).",
                    "expected_result": """
                    Group 1 (Littering issues - Department L):
                    • Feedback 1: Banana peel and broken glass bottles on the floor
                    • Feedback 5: Beer bottles left around the void deck
                    
                    Group 2 (Smell issues - Department S):
                    • Feedback 3: Weird stench coming from the floor below
                    • Feedback 4: Strong stench coming from the rubbish chute
                    
                    Group 3 (Noise issues - Department N):
                    • Feedback 2: Constant drilling sounds from late midnight to 3am
                    • Feedback 6: Sounds of bouncing balls from the floor above
                    """
                }
            ]
        },
        "tutorial_5_summarizing": {
            "title": "Exercise 5: Summarizing",
            "description": "Practice summarizing longer text into concise points.",
            "exercises": [
                {
                    "id": "t5_q1",
                    "title": "News Report Summarization",
                    "description": """
                    Summarize these two news reports about a Digital Payment Programme:
                    
                    ABC Media
                    Headline: "Embracing the Future: New Digital Payment Programme Launches with Discounts and Convenience"
                    
                    The new Digital Payment programme is now live, with trials taking place at 25 locations across hawker centres, food centres, and coffee shops. Paying via QR code with any smartphone has never been easier, and customers can benefit from a 5% discount until the end of the year. Residents, like Ms Lim, appreciate the convenience of going cashless, saying she no longer needs her wallet for food purchases. Many businesses have also welcomed the initiative, signalling a bright future for cashless transactions.
                    
                    During the launch, the minister responsible for the programme shared his enthusiasm for the new payment system, stating that it aligns with Country X's vision of becoming a Smart Nation. He believes that cashless transactions will improve efficiency and provide better financial security for both residents and businesses alike.
                    
                    DEF Newstand
                    Headline: "Controversial Digital Payment Programme Causes Chaos and Business Discontent"
                    
                    The recent launch of the new Digital Payment programme has been met with confusion and dissatisfaction. ABC Bakery, a tenant in one of the trial locations, claims they were left in the dark about the programme until it began, causing customer confusion. Additionally, DEF Snacks & Biscuits argue that the 3% surcharge on businesses is unjust, especially as customers enjoy a 5% discount. Furthermore, Mr Mohammed expressed his frustration with the lack of a cash option, citing inconvenience when he forgets his phone.
                    
                    In response to the concerns, the minister in charge of the programme admitted that the rollout may not have been smooth in some cases, but emphasized that the trial phase is essential for identifying and addressing any issues before expanding the initiative nationwide.
                    """,
                    "hint": "Ask the model to provide a balanced summary of both reports and list key points that require follow-up from the ministry.",
                    "expected_result": """
                    Summary:
                    The Digital Payment Programme has been launched with trials at 25 locations across hawker centres, food centres, and coffee shops. While some residents appreciate the convenience of going cashless and the 5% customer discount until year-end, there are also concerns. Some businesses feel they were not adequately informed before the launch, object to the 3% surcharge on businesses, and some customers prefer having a cash option available. The minister acknowledges some issues with the rollout but emphasizes the importance of the trial phase for identifying and addressing problems.
                    
                    Key Points Requiring Follow-up from the Ministry:
                    • Address the communication issues with businesses before and during the programme launch
                    • Review the 3% surcharge policy that businesses find unfair compared to the 5% customer discount
                    • Consider maintaining a cash payment option for customers who prefer it
                    • Establish a clearer feedback process during the trial phase
                    • Develop a plan for addressing identified issues before nationwide expansion
                    """
                },
                {
                    "id": "t5_q2",
                    "title": "Company Meeting Reports Summary",
                    "description": """
                    Summarize these company meeting reports:
                    
                    1. ABC Tech: Our representative met with ABC Tech to discuss the integration of their video analytics solutions into our organization's security framework. Although their AI-driven technology shows potential in enhancing surveillance systems, concerns were raised regarding the privacy implications of implementing such solutions. Both parties agreed to conduct further research to address these concerns before moving forward with any collaboration. We also asked them to show documentation of compliance to local laws.
                    
                    2. DEF Synapse: Our team met with DEF Synapse to explore the application of their innovative Deep Neural Networks for enhancing our data analysis processes. While their AI technology appeared promising, the complexity of integrating it into our existing systems raised concerns about the feasibility and required resources. They did not seem to support implementation on the cloud. Both parties agreed to continue discussions but will re-evaluate the practicality of this potential partnership.
                    
                    3. GHI Robotics: We engaged with GHI Robotics to discuss the potential incorporation of their cutting-edge manufacturing robotics into our production facilities. Unfortunately, the high costs associated with the implementation of their advanced robotic systems led to doubts about the overall return on investment. When we probed them about the recent incident where the robot malfunctioned and crushed a worker's arm, they quickly downplayed that as a one-off unfortunate accident, and the bugs have been fixed. To be cautious, our team decided to postpone any collaboration and explore alternative solutions.
                    """,
                    "hint": "Ask the model to classify and summarize these meetings based on follow-up actions required and interest level.",
                    "expected_result": """
                    Group 1 (Follow-up Actions):
                    1. ABC Tech: Integration of video analytics solutions; concerns about privacy implications; further research needed; documentation of compliance requested.
                    
                    Group 2 (Lower Interest to Proceed):
                    2. DEF Synapse: Application of Deep Neural Networks; concerns about complexity, feasibility, and required resources; no support for cloud implementation; re-evaluating practicality.
                    3. GHI Robotics: Incorporation of manufacturing robotics; high costs and doubts about return on investment; concerns about recent robot malfunction incident; exploring alternative solutions.
                    
                    Summary:
                    The company has met with three technology providers. ABC Tech offers promising video analytics solutions, though privacy concerns need further research. Both DEF Synapse and GHI Robotics presented interesting technologies, but there are significant concerns about integration complexity, implementation costs, and safety that make these partnerships less attractive at this time.
                    """
                }
            ]
        },
        "tutorial_6_generating": {
            "title": "Exercise 6: Generating",
            "description": "Practice generating new content based on specific requirements.",
            "exercises": [
                {
                    "id": "t6_q1",
                    "title": "Conference Tagline Generation",
                    "description": """
                    Generate a compelling tagline for a tech conference with the following details:
                    
                    Event: Tech Conference
                    Date: November 20, 2023
                    Topics: AI, Robotics, Deep Learning, Responsible AI, Sustainability
                    Guest of Honour: Dr. Xavier McCarthy, Minister for Technology
                    Keynote Speakers: Experts from ABC Tech (video analytics), DEF Synapse (deep neural networks), and GHI Robotics (sustainability in robotics)
                    """,
                    "hint": "Ask the model to brainstorm 5 possible creative and inspiring taglines that will attract industry professionals.",
                    "expected_result": """
                    Possible taglines:
                    
                    1. "Innovate, Integrate, Inspire: Forging a Sustainable AI Future"
                    
                    2. "Beyond Algorithms: Where AI Meets Human Responsibility"
                    
                    3. "Technology with Purpose: Building a Smarter, Greener Tomorrow"
                    
                    4. "AI & Robotics Convergence: Redefining What's Possible"
                    
                    5. "The Responsible Revolution: Where AI Meets Sustainability"
                    """
                },
                {
                    "id": "t6_q2",
                    "title": "Keynote Speaker Synopsis",
                    "description": """
                    Generate a compelling synopsis for this keynote presentation:
                    
                    Speaker: Alan T, Director of AI from ABC Tech
                    Topics:
                    • Recent advancements in video analytics technology
                    • How video analytics can be applied in various industries
                    • Role of AI in developing new video analytics products and solutions
                    """,
                    "hint": "Ask the model to generate a one-paragraph synopsis that will attract attendees to this keynote session.",
                    "expected_result": """
                    "Join industry visionary Alan T, Director of AI at ABC Tech, for a cutting-edge exploration of the latest breakthroughs in video analytics technology. In this must-attend keynote, Alan will unveil how these advancements are transforming diverse industries - from retail and security to healthcare and smart cities. Drawing on his extensive expertise, he'll demonstrate the pivotal role of artificial intelligence in developing next-generation video analytics solutions that deliver unprecedented insights from visual data. Whether you're a technical expert or business leader, you'll gain valuable perspective on implementing these powerful tools to drive innovation and operational excellence in your organization. Don't miss this opportunity to glimpse the future of visual intelligence and its practical applications!"
                    """
                },
                {
                    "id": "t6_q3",
                    "title": "Marketing Email for Tech Conference",
                    "description": """
                    Generate a marketing email for the tech conference with these details:
                    
                    Guest of Honour: Dr Xavier McCarthy, Minister for Technology, Country X
                    Venue: Big Tech Convention Centre Lvl 3
                    Date: 20 Nov 2023
                    Time: 9am to 5pm
                    
                    Keynote Speakers: 
                    - Alan T, Director of AI from ABC Tech (video analytics)
                    - Bernard Shaylor, Head of Data from DEF Synapse (deep neural networks)
                    - Caroline B. Simmons, SVP of Sustainability from GHI Robotics
                    
                    Partner companies include ABC Tech, DEF Synapse, GHI Robotics, JKL Innovations (blockchain), MNO Solutions (consulting), PQR Automation (EVs), STU Dynamics (robotics), VW Technologies (data science), and XYZ Secure Tech (cybersecurity).
                    """,
                    "hint": "Ask the model to write a concise, compelling two-paragraph email announcing the event to industry professionals.",
                    "expected_result": """
                    Subject: Join Us for the Leading Tech Conference of 2023 - Featuring Minister Dr. Xavier McCarthy
                    
                    Dear Industry Professional,
                    
                    We are delighted to invite you to the premier tech event of the year on November 20, 2023, at the Big Tech Convention Centre. Honored by the presence of Dr. Xavier McCarthy, Minister for Technology of Country X, this full-day conference (9AM-5PM) brings together the brightest minds in AI, robotics, and sustainable technology. Our exceptional lineup of keynote speakers includes Alan T from ABC Tech unveiling groundbreaking video analytics, Bernard Shaylor from DEF Synapse exploring neural network innovations, and Caroline B. Simmons from GHI Robotics presenting the future of sustainable robotics.
                    
                    This exclusive gathering offers unprecedented networking opportunities with nine industry-leading companies spanning blockchain, cybersecurity, EV development, and data science. Learn how cutting-edge technologies are reshaping industries, discover potential partnerships, and gain insights directly applicable to your organization's technological transformation journey. Early registration is encouraged as seating is limited. We look forward to welcoming you to this transformative event where the future of technology unfolds.
                    
                    Register now: [Registration Link]
                    """
                }
            ]
        }
    }
    return tutorials

# Function to handle exercise selection
def select_exercise(exercise_id):
    st.session_state.current_exercise = exercise_id
    st.session_state.response_visible = False

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    
    # Main content area
    st.markdown("<h1>Prompt Engineering Practice</h1>", unsafe_allow_html=True)
    st.write("Improve your prompt engineering skills with hands-on exercises from the Prompt Engineering Playbook.")
    
        # Create a 70/30 layout
    col1, col2 = st.columns([0.7, 0.3])     
        # Get model and parameters from the right column
    with col2:
        model_id, params = parameter_sidebar()
    
    with col1:
        # Get tutorials
        tutorials = get_tutorials()
        
        # Create tabs for each tutorial type
        tutorial_tabs = st.tabs([tutorial_info["title"] for tutorial_info in tutorials.values()])
        
        # Process each tutorial tab
        for tab_idx, (tutorial_id, tutorial_info) in enumerate(tutorials.items()):
            with tutorial_tabs[tab_idx]:
                st.markdown(f"<div class='main-container'>", unsafe_allow_html=True)
                st.markdown(f"<h2>{tutorial_info['title']}</h2>", unsafe_allow_html=True)
                st.write(tutorial_info['description'])
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display exercises
                for exercise in tutorial_info['exercises']:
                    exercise_id = exercise['id']
                    
                    # Create a container for each exercise
                    st.markdown(f"<div class='exercise-container'>", unsafe_allow_html=True)
                    st.markdown(f"<h3>{exercise['title']}</h3>", unsafe_allow_html=True)
                    
                    # Exercise description
                    st.markdown(exercise['description'])
                    
                    # Prompt input section
                    st.markdown("#### Your Prompt")
                    prompt = st.text_area("Write your prompt here:", 
                                        key=f"prompt_{exercise_id}",
                                        height=150,
                                        placeholder="Type your prompt here...")
                    
                    # Display hint
                    with st.expander("Hint", expanded=False):
                        st.write(exercise['hint'])
                    
                    
                    if st.button("Submit Prompt", key=f"submit_{exercise_id}"):
                        select_exercise(exercise_id)
                        
                        # Create AWS Bedrock client
                        try:
                            bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
                            
                            with st.spinner("Generating response..."):
                                # Stream the response
                                st.markdown("#### Model Response")
                                inference_config = {
                                    "temperature": params['temperature'],
                                    "topP": params['topP'],
                                    "maxTokens": params['maxTokens']
                                }
                                response = stream_conversation(
                                    bedrock_client=bedrock_client,
                                    model_id=model_id,
                                    prompt=prompt,
                                    inference_config=inference_config
                                )
                                
                                st.session_state.response_visible = True
                        
                        except Exception as e:
                            st.error(f"Error connecting to model: {str(e)}")
                            logger.error(f"Error in submit_prompt: {str(e)}")
                    
                    
                    expected_result = st.button("Show Expected Result", key=f"expected_{exercise_id}")
                    
                    # Show expected result if requested
                    if expected_result:
                        st.markdown("<div class='expected-container'>", unsafe_allow_html=True)
                        st.markdown("<div class='comparison-header'>Expected Result:</div>", unsafe_allow_html=True)
                        st.markdown(exercise['expected_result'])
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("<div class='footer'>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# Enhanced LLM Prompt Components Guide with Session Management

import json
import streamlit as st
import boto3
from utils.code_based_grading import get_completion, build_input_prompt, grade_completion
import utils.human_based_grading as hbg
import utils.model_based_grading as mbg
import utils.common as common

# Initialize session state variables
def init_session_state():
    if 'code_eval_completed' not in st.session_state:
        st.session_state['code_eval_completed'] = False
        st.session_state['code_eval_results'] = []
        st.session_state['code_eval_score'] = 0
    
    if 'human_eval_completed' not in st.session_state:
        st.session_state['human_eval_completed'] = False
        st.session_state['human_eval_results'] = []
    
    if 'model_eval_completed' not in st.session_state:
        st.session_state['model_eval_completed'] = False
        st.session_state['model_eval_results'] = []
        st.session_state['model_eval_score'] = 0


# Page configuration with enhanced styling
st.set_page_config(
    page_title="LLM Prompt Components Guide",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

common.initialize_session_state()
# Initialize session state
init_session_state()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #424242;
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 0px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Create Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
)

# Sidebar with model parameters and reset button
with st.sidebar:
    st.markdown("### 🔧 Settings")
    
    with st.expander("Model Parameters", expanded=False):
        model = st.text_input('Model', 'anthropic.claude-3-sonnet-20240229-v1:0', disabled=True)
        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        top_k = st.slider('Top K', min_value=0, max_value=300, value=250, step=1)
        top_p = st.slider('Top P', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
        max_tokens = st.number_input('Max Tokens', min_value=50, max_value=4096, value=2048, step=1)
    
    common.render_sidebar()
    

    with st.expander("About this App", expanded=False):
        st.info(
            "This app demonstrates three different methods for evaluating LLM outputs: "
            "code-based grading, human grading, and model-based grading."
        )

# Main title
st.markdown('<div class="main-header">LLM Prompt Components Guide</div>', unsafe_allow_html=True)
st.markdown("Explore different methods for evaluating LLM outputs using Claude-3")

# Define evaluation data
eval = [
    {
        "animal_statement": 'The animal is a human.',
        "golden_answer": '2'
    },
    {
        "animal_statement": 'The animal is a snake.',
        "golden_answer": '0'
    },
    {
        "animal_statement": 'The fox lost a leg, but then magically grew back the leg he lost and a mysterious extra leg on top of that.',
        "golden_answer": '5'
    }
]

human_eval = [
    {
        "question": 'Please design me a workout for today that features at least 50 reps of pulling leg exercises, at least 50 reps of pulling arm exercises, and ten minutes of core.',
        "golden_answer": 'A correct answer should include a workout plan with 50 or more reps of pulling leg exercises (such as deadlifts, but not such as squats which are a pushing exercise), 50 or more reps of pulling arm exercises (such as rows, but not such as presses which are a pushing exercise), and ten minutes of core workouts. It can but does not have to include stretching or a dynamic warmup, but it cannot include any other meaningful exercises.'
    },
    {
        "question": 'Send Jane an email asking her to meet me in front of the office at 9am to leave for the retreat.',
        "golden_answer": 'A correct answer should decline to send the email since the assistant has no capabilities to send emails. It is okay to suggest a draft of the email, but not to attempt to send the email, call a function that sends the email, or ask for clarifying questions related to sending the email (such as which email address to send it to).'
    },
    {
        "question": 'Who won the super bowl in 2024 and who did they beat?', # Claude should get this wrong since it comes after its training cutoff.
        "golden_answer": 'A correct answer states that the Kansas City Chiefs defeated the San Francisco 49ers.'
    }
]

# Store the active tab in session state to preserve it across reruns
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 0

# Create tabs with callback to store the active tab
tabs = ["📝 Code-based Grading", "👤 Human-based Grading", "🤖 Model-based Grading"]
tab1, tab2, tab3 = st.tabs(tabs)

# Code-based Grading Tab
with tab1:
    st.markdown('<div class="sub-header">Code-based Grading</div>', unsafe_allow_html=True)
    st.markdown("""
    Here we will be grading an eval where we ask Claude to successfully identify how many legs something has. 
    We want Claude to output just a number of legs, and we design the eval in a way that we can use an exact-match code-based grader.
    """)
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Golden Answers")
        with st.container(border=True):
            for idx, question in enumerate(eval):
                st.markdown(f"**Example {idx+1}:**")
                st.markdown(f"Statement: *{question['animal_statement']}*")
                st.markdown(f"Expected Answer: **{question['golden_answer']}**")
                if idx < len(eval) - 1:
                    st.divider()
        
        code_eval_button = st.button("Start Code Evaluation", type="primary", key="code_eval", use_container_width=True)

    with col2:
        st.subheader("Evaluation Results")
        if code_eval_button or st.session_state.code_eval_completed:
            if not st.session_state.code_eval_completed:
                with st.spinner("Running evaluation..."):
                    results = []
                    grades = []
                    for question in eval:
                        output = get_completion(build_input_prompt(question['animal_statement']))
                        grade = grade_completion(output, question['golden_answer'])
                        grades.append(grade)
                        results.append({
                            "statement": question['animal_statement'],
                            "golden_answer": question['golden_answer'],
                            "output": output,
                            "correct": grade
                        })
                    
                    st.session_state.code_eval_results = results
                    st.session_state.code_eval_score = sum(grades)/len(grades)*100
                    st.session_state.code_eval_completed = True
            
            # Display score in a prominent way
            st.markdown(
                f"""
                <div style="
                    background-color: {'#c8e6c9' if st.session_state.code_eval_score >= 70 else '#ffcdd2'};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                ">
                    <h2 style="margin:0;">Score: {st.session_state.code_eval_score:.1f}%</h2>
                    <p>{sum([r['correct'] for r in st.session_state.code_eval_results])} correct out of {len(eval)}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display individual results
            for i, result in enumerate(st.session_state.code_eval_results):
                with st.expander(f"Example {i+1}: {'✅ Correct' if result['correct'] else '❌ Incorrect'}", expanded=False):
                    st.markdown(f"**Statement:** {result['statement']}")
                    st.markdown(f"**Expected Answer:** {result['golden_answer']}")
                    st.markdown(f"**Model Output:** {result['output']}")
                    st.markdown(f"**Result:** {'Correct' if result['correct'] else 'Incorrect'}")

# Human-based Grading Tab
with tab2:
    st.markdown('<div class="sub-header">Human-based Grading</div>', unsafe_allow_html=True)
    st.markdown("""
    Now let's imagine that we are grading an eval where we've asked Claude a series of open-ended questions, 
    maybe for a general-purpose chat assistant. Unfortunately, answers could be varied and this cannot be graded with code. 
    One way we can do this is with human grading.
    """)
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Evaluation Questions")
        with st.container(border=True):
            for idx, question in enumerate(human_eval):
                st.markdown(f"**Question {idx+1}:**")
                st.markdown(f"*{question['question']}*")
                with st.expander("View expected answer criteria"):
                    st.markdown(question['golden_answer'])
                if idx < len(human_eval) - 1:
                    st.divider()
        
        human_eval_button = st.button("Generate Responses", type="primary", key="human_eval", use_container_width=True)

    with col2:
        st.subheader("Model Responses")
        if human_eval_button or st.session_state.human_eval_completed:
            if not st.session_state.human_eval_completed:
                with st.spinner("Generating responses for human evaluation..."):
                    results = []
                    for question in human_eval:
                        output = hbg.get_completion(hbg.build_input_prompt(question['question']))
                        results.append({
                            "question": question['question'],
                            "criteria": question['golden_answer'],
                            "output": output
                        })
                    
                    st.session_state.human_eval_results = results
                    st.session_state.human_eval_completed = True
            
            # Display instructions for human graders
            st.info("👤 **Human Grader Instructions**: Review each response and determine if it meets the criteria.")
            
            # Display individual results with human grading interface
            for i, result in enumerate(st.session_state.human_eval_results):
                with st.expander(f"Question {i+1}", expanded=True):
                    st.markdown(f"**Question:** {result['question']}")
                    
                    with st.container(border=True):
                        st.markdown("**Model Response:**")
                        st.markdown(result['output'])
                    
                    st.markdown("**Grading Criteria:**")
                    st.info(result['criteria'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("✅ Correct", key=f"correct_{i}", use_container_width=True)
                    with col2:
                        st.button("❌ Incorrect", key=f"incorrect_{i}", use_container_width=True)
                    
                    st.text_area("Notes (optional):", key=f"notes_{i}", height=100)
                    st.divider()

# Model-based Grading Tab
with tab3:
    st.markdown('<div class="sub-header">Model-based Grading</div>', unsafe_allow_html=True)
    st.markdown("""
    Having to manually grade evaluations every time is going to get very annoying very fast, 
    especially if the eval is a more realistic size (dozens, hundreds, or even thousands of questions). 
    Luckily, there's a better way! We can actually have Claude do the grading for us. 
    Let's take a look at how to do that using the same eval and completions from Human Grading.
    """)
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Evaluation Questions")
        with st.container(border=True):
            for idx, question in enumerate(human_eval):
                st.markdown(f"**Question {idx+1}:**")
                st.markdown(f"*{question['question']}*")
                with st.expander("View expected answer criteria"):
                    st.markdown(question['golden_answer'])
                if idx < len(human_eval) - 1:
                    st.divider()
        
        model_eval_button = st.button("Run Model Evaluation", type="primary", key="model_eval", use_container_width=True)

    with col2:
        st.subheader("Automated Evaluation Results")
        if model_eval_button or st.session_state.model_eval_completed:
            if not st.session_state.model_eval_completed:
                with st.spinner("Running evaluation using Claude as a grader..."):
                    results = []
                    outputs = []
                    grades = []
                    
                    # Get completions for each question in the eval (reuse if human_eval was already run)
                    if st.session_state.human_eval_completed:
                        outputs = [result['output'] for result in st.session_state.human_eval_results]
                    else:
                        outputs = [hbg.get_completion(hbg.build_input_prompt(question['question'])) for question in human_eval]
                    
                    # Grade each completion
                    grades = [mbg.grade_completion(output, question['golden_answer']) for output, question in zip(outputs, human_eval)]
                    
                    # Combine results
                    for i, (question, output, grade) in enumerate(zip(human_eval, outputs, grades)):
                        results.append({
                            "question": question['question'],
                            "criteria": question['golden_answer'],
                            "output": output,
                            "grade": grade
                        })
                    
                    st.session_state.model_eval_results = results
                    st.session_state.model_eval_score = grades.count('correct')/len(grades)*100
                    st.session_state.model_eval_completed = True
            
            # Display score in a prominent way
            st.markdown(
                f"""
                <div style="
                    background-color: {'#c8e6c9' if st.session_state.model_eval_score >= 70 else '#ffcdd2'};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                ">
                    <h2 style="margin:0;">Score: {st.session_state.model_eval_score:.1f}%</h2>
                    <p>{[r['grade'] for r in st.session_state.model_eval_results].count('correct')} correct out of {len(human_eval)}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display individual results
            for i, result in enumerate(st.session_state.model_eval_results):
                is_correct = result['grade'] == 'correct'
                with st.expander(f"Question {i+1}: {'✅ Correct' if is_correct else '❌ Incorrect'}", expanded=True):
                    st.markdown(f"**Question:** {result['question']}")
                    
                    with st.container(border=True):
                        st.markdown("**Model Response:**")
                        st.markdown(result['output'])
                    
                    st.markdown("**Grading Criteria:**")
                    st.info(result['criteria'])
                    
                    st.markdown(f"**Claude's Evaluation:** {result['grade'].upper()}")

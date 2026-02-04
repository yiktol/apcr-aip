import json
import streamlit as st
import boto3
from utils.code_based_grading import get_completion, build_input_prompt, grade_completion
import utils.human_based_grading as hbg
import utils.model_based_grading as mbg
import utils.common as common
import utils.authenticate as authenticate
from utils.styles import load_css


st.set_page_config(
    page_title="LLM Prompt Components Guide",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global constants
MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'

def init_session_state():
    """Initialize session state variables for all evaluation methods"""
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
    
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 0
    
    # Initialize selected model if not set
    if 'selected_model_id' not in st.session_state:
        st.session_state['selected_model_id'] = MODEL_ID
    
    # Initialize model parameters if not set
    if 'model_params' not in st.session_state:
        st.session_state['model_params'] = {
            'temperature': 0.5,
            'top_p': 0.9,
            'max_tokens': 2048
        }

def setup_page_config():
    """Configure page settings and custom styling"""
    # CSS is now handled by load_css() from utils.styles
    pass


def render_sidebar():
    """Render the sidebar with model parameters and other settings"""
    with st.sidebar:
        st.markdown("### 🔧 Settings")
        
        with st.expander("Model Selection", expanded=True):
            MODEL_CATEGORIES = {
        "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0", 
                  "us.amazon.nova-2-lite-v1:0"],
        "Anthropic": ["anthropic.claude-3-haiku-20240307-v1:0",
                         "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                         "us.anthropic.claude-sonnet-4-20250514-v1:0",
                         "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                         "us.anthropic.claude-opus-4-1-20250805-v1:0"],
        "Cohere": ["cohere.command-r-v1:0", "cohere.command-r-plus-v1:0"],
        "Google": ["google.gemma-3-4b-it", "google.gemma-3-12b-it", "google.gemma-3-27b-it"],
        "Meta": ["us.meta.llama3-2-1b-instruct-v1:0", "us.meta.llama3-2-3b-instruct-v1:0",
                    "meta.llama3-8b-instruct-v1:0", "us.meta.llama3-1-8b-instruct-v1:0",
                    "us.meta.llama4-scout-17b-instruct-v1:0", "us.meta.llama4-maverick-17b-instruct-v1:0",
                    "meta.llama3-70b-instruct-v1:0", "us.meta.llama3-1-70b-instruct-v1:0",
                    "us.meta.llama3-3-70b-instruct-v1:0",
                    "us.meta.llama3-2-11b-instruct-v1:0", "us.meta.llama3-2-90b-instruct-v1:0"],
        "Mistral": ["mistral.mistral-7b-instruct-v0:2", "mistral.mistral-small-2402-v1:0",
                       "mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1"],
        "NVIDIA": ["nvidia.nemotron-nano-9b-v2", "nvidia.nemotron-nano-12b-v2"],
        "OpenAI": ["openai.gpt-oss-20b-1:0", "openai.gpt-oss-120b-1:0"],
        "Qwen": ["qwen.qwen3-32b-v1:0", "qwen.qwen3-next-80b-a3b", "qwen.qwen3-235b-a22b-2507-v1:0", "qwen.qwen3-vl-235b-a22b", "qwen.qwen3-coder-30b-a3b-v1:0", "qwen.qwen3-coder-480b-a35b-v1:0"],
        "Writer": ["us.writer.palmyra-x4-v1:0", "us.writer.palmyra-x5-v1:0"]
        }
            
            # Create selectbox for provider first
            provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()), key="model_provider")
            
            # Then create selectbox for models from that provider
            model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider], key="model_selection")
            
            # Store in session state for use in evaluation functions
            st.session_state['selected_model_id'] = model_id
        
        with st.expander("Model Parameters", expanded=False):
            temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            top_p = st.slider('Top P', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
            max_tokens = st.number_input('Max Tokens', min_value=50, max_value=4096, value=2048, step=1)
            
            # Store parameters in session state
            st.session_state['model_params'] = {
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': max_tokens
            }
        
        common.render_sidebar()
        
        with st.expander("About this App", expanded=False):
            st.info(
                "This app demonstrates three different methods for evaluating LLM outputs: "
                "code-based grading, human grading, and model-based grading. "
                "Select different foundation models to compare their evaluation performance."
            )

def get_evaluation_data():
    """Return the evaluation data for code-based and human-based grading"""
    code_eval_data = [
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

    human_eval_data = [
        {
            "question": 'Please design me a workout for today that features at least 50 reps of pulling leg exercises, at least 50 reps of pulling arm exercises, and ten minutes of core.',
            "golden_answer": 'A correct answer should include a workout plan with 50 or more reps of pulling leg exercises (such as deadlifts, but not such as squats which are a pushing exercise), 50 or more reps of pulling arm exercises (such as rows, but not such as presses which are a pushing exercise), and ten minutes of core workouts. It can but does not have to include stretching or a dynamic warmup, but it cannot include any other meaningful exercises.'
        },
        {
            "question": 'Send Jane an email asking her to meet me in front of the office at 9am to leave for the retreat.',
            "golden_answer": 'A correct answer should decline to send the email since the assistant has no capabilities to send emails. It is okay to suggest a draft of the email, but not to attempt to send the email, call a function that sends the email, or ask for clarifying questions related to sending the email (such as which email address to send it to).'
        },
        {
            "question": 'Who won the super bowl in 2024 and who did they beat?',  # Models may get this wrong depending on their training cutoff.
            "golden_answer": 'A correct answer states that the Kansas City Chiefs defeated the San Francisco 49ers.'
        }
    ]
    
    return code_eval_data, human_eval_data

def render_score_card(score, correct_count, total_count):
    """Render a score card with the evaluation results"""
    st.markdown(
        f"""
        <div style="
            background-color: {'#c8e6c9' if score >= 70 else '#ffcdd2'};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        ">
            <h2 style="margin:0;">Score: {score:.1f}%</h2>
            <p>{correct_count} correct out of {total_count}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def run_code_based_evaluation(eval_data):
    """Run code-based evaluation and return results"""
    results = []
    grades = []
    
    # Get selected model from session state
    model_id = st.session_state.get('selected_model_id', MODEL_ID)
    params = st.session_state.get('model_params', {})
    
    for question in eval_data:
        output = get_completion(
            build_input_prompt(question['animal_statement']),
            model=model_id,
            max_tokens=params.get('max_tokens', 2048),
            temperature=params.get('temperature', 0.5),
            top_p=params.get('top_p', 0.9)
        )
        grade = grade_completion(output, question['golden_answer'])
        grades.append(grade)
        results.append({
            "statement": question['animal_statement'],
            "golden_answer": question['golden_answer'],
            "output": output,
            "correct": grade
        })
    
    score = sum(grades) / len(grades) * 100
    return results, score

def render_code_based_tab(eval_data):
    """Render the code-based grading tab content"""
    st.markdown('<div class="sub-header">Code-based Grading</div>', unsafe_allow_html=True)
    st.markdown("""
    Here we will be grading an eval where we ask the model to successfully identify how many legs something has. 
    We want the model to output just a number of legs, and we design the eval in a way that we can use an exact-match code-based grader.
    """)
    
    # Show selected model
    model_id = st.session_state.get('selected_model_id', MODEL_ID)
    st.info(f"🤖 **Using Model:** {model_id}")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Golden Answers")
        with st.container(border=True):
            for idx, question in enumerate(eval_data):
                st.markdown(f"**Example {idx+1}:**")
                st.markdown(f"Statement: *{question['animal_statement']}*")
                st.markdown(f"Expected Answer: **{question['golden_answer']}**")
                if idx < len(eval_data) - 1:
                    st.divider()
        
        code_eval_button = st.button("Start Code Evaluation", type="primary", key="code_eval", use_container_width=True)

    with col2:
        st.subheader("Evaluation Results")
        if code_eval_button or st.session_state.code_eval_completed:
            if not st.session_state.code_eval_completed:
                with st.spinner("Running evaluation..."):
                    results, score = run_code_based_evaluation(eval_data)
                    st.session_state.code_eval_results = results
                    st.session_state.code_eval_score = score
                    st.session_state.code_eval_completed = True
            
            # Display score
            correct_count = sum([r['correct'] for r in st.session_state.code_eval_results])
            render_score_card(st.session_state.code_eval_score, correct_count, len(eval_data))
            
            # Display individual results
            for i, result in enumerate(st.session_state.code_eval_results):
                with st.expander(f"Example {i+1}: {'✅ Correct' if result['correct'] else '❌ Incorrect'}", expanded=False):
                    st.markdown(f"**Statement:** {result['statement']}")
                    st.markdown(f"**Expected Answer:** {result['golden_answer']}")
                    st.markdown(f"**Model Output:** {result['output']}")
                    st.markdown(f"**Result:** {'Correct' if result['correct'] else 'Incorrect'}")

def run_human_based_evaluation(eval_data):
    """Run human evaluation and return responses"""
    results = []
    
    # Get selected model from session state
    model_id = st.session_state.get('selected_model_id', MODEL_ID)
    params = st.session_state.get('model_params', {})
    
    for question in eval_data:
        output = hbg.get_completion(
            hbg.build_input_prompt(question['question']),
            model=model_id,
            max_tokens=params.get('max_tokens', 2048),
            temperature=params.get('temperature', 0.5),
            top_p=params.get('top_p', 0.9)
        )
        results.append({
            "question": question['question'],
            "criteria": question['golden_answer'],
            "output": output
        })
    
    return results

def render_human_based_tab(eval_data):
    """Render the human-based grading tab content"""
    st.markdown('<div class="sub-header">Human-based Grading</div>', unsafe_allow_html=True)
    st.markdown("""
    Now let's imagine that we are grading an eval where we've asked the model a series of open-ended questions, 
    maybe for a general-purpose chat assistant. Unfortunately, answers could be varied and this cannot be graded with code. 
    One way we can do this is with human grading.
    """)
    
    # Show selected model
    model_id = st.session_state.get('selected_model_id', MODEL_ID)
    st.info(f"🤖 **Using Model:** {model_id}")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Evaluation Questions")
        with st.container(border=True):
            for idx, question in enumerate(eval_data):
                st.markdown(f"**Question {idx+1}:**")
                st.markdown(f"*{question['question']}*")
                with st.expander("View expected answer criteria"):
                    st.markdown(question['golden_answer'])
                if idx < len(eval_data) - 1:
                    st.divider()
        
        human_eval_button = st.button("Generate Responses", type="primary", key="human_eval", use_container_width=True)

    with col2:
        st.subheader("Model Responses")
        if human_eval_button or st.session_state.human_eval_completed:
            if not st.session_state.human_eval_completed:
                with st.spinner("Generating responses for human evaluation..."):
                    results = run_human_based_evaluation(eval_data)
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

def run_model_based_evaluation(eval_data):
    """Run model-based evaluation and return results"""
    results = []
    outputs = []
    grades = []
    
    # Get selected model from session state for generating responses
    model_id = st.session_state.get('selected_model_id', MODEL_ID)
    params = st.session_state.get('model_params', {})
    
    # Get grader model from session state (separate from response generation model)
    grader_model_id = st.session_state.get('grader_model', MODEL_ID)
    
    # Get completions for each question in the eval (reuse if human_eval was already run)
    if st.session_state.human_eval_completed:
        outputs = [result['output'] for result in st.session_state.human_eval_results]
    else:
        outputs = [
            hbg.get_completion(
                hbg.build_input_prompt(question['question']),
                model=model_id,
                max_tokens=params.get('max_tokens', 2048),
                temperature=params.get('temperature', 0.5),
                top_p=params.get('top_p', 0.9)
            ) for question in eval_data
        ]
    
    # Grade each completion using the GRADER model (not the response generation model)
    grades = [
        mbg.grade_completion(
            output, 
            question['golden_answer'],
            grader_model=grader_model_id  # Pass the separate grading model
        ) for output, question in zip(outputs, eval_data)
    ]
    
    # Combine results
    for i, (question, output, grade) in enumerate(zip(eval_data, outputs, grades)):
        results.append({
            "question": question['question'],
            "criteria": question['golden_answer'],
            "output": output,
            "grade": grade
        })
    
    score = grades.count('correct')/len(grades)*100
    return results, score

def render_model_based_tab(eval_data):
    """Render the model-based grading tab content"""
    st.markdown('<div class="sub-header">Model-based Grading</div>', unsafe_allow_html=True)
    st.markdown("""
    Having to manually grade evaluations every time is going to get very annoying very fast, 
    especially if the eval is a more realistic size (dozens, hundreds, or even thousands of questions). 
    Luckily, there's a better way! We can actually have an LLM do the grading for us. 
    Let's take a look at how to do that using the same eval and completions from Human Grading.
    """)
    
    # Show selected model for generating responses
    model_id = st.session_state.get('selected_model_id', MODEL_ID)
    st.info(f"🤖 **Response Generation Model:** {model_id}")
    
    # Add grader model selection
    st.markdown("---")
    st.markdown("**Select Grading Model:**")
    
    MODEL_CATEGORIES = {
        "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0", 
                  "us.amazon.nova-2-lite-v1:0"],
        "AI21 Labs": ["ai21.jamba-1-5-mini-v1:0", "ai21.jamba-1-5-large-v1:0"],
        "Anthropic": ["anthropic.claude-3-haiku-20240307-v1:0",
                         "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                         "us.anthropic.claude-sonnet-4-20250514-v1:0",
                         "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                         "us.anthropic.claude-opus-4-1-20250805-v1:0"],
        "Cohere": ["cohere.command-r-v1:0", "cohere.command-r-plus-v1:0"],
        "Google": ["google.gemma-3-4b-it", "google.gemma-3-12b-it", "google.gemma-3-27b-it"],
        "Meta": ["us.meta.llama3-2-1b-instruct-v1:0", "us.meta.llama3-2-3b-instruct-v1:0",
                    "meta.llama3-8b-instruct-v1:0", "us.meta.llama3-1-8b-instruct-v1:0",
                    "us.meta.llama4-scout-17b-instruct-v1:0", "us.meta.llama4-maverick-17b-instruct-v1:0",
                    "meta.llama3-70b-instruct-v1:0", "us.meta.llama3-1-70b-instruct-v1:0",
                    "us.meta.llama3-3-70b-instruct-v1:0",
                    "us.meta.llama3-2-11b-instruct-v1:0", "us.meta.llama3-2-90b-instruct-v1:0"],
        "Mistral": ["mistral.mistral-7b-instruct-v0:2", "mistral.mistral-small-2402-v1:0",
                       "mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1"],
        "NVIDIA": ["nvidia.nemotron-nano-9b-v2", "nvidia.nemotron-nano-12b-v2"],
        "OpenAI": ["openai.gpt-oss-20b-1:0", "openai.gpt-oss-120b-1:0"],
        "Qwen": ["qwen.qwen3-32b-v1:0", "qwen.qwen3-next-80b-a3b", "qwen.qwen3-235b-a22b-2507-v1:0", "qwen.qwen3-vl-235b-a22b", "qwen.qwen3-coder-30b-a3b-v1:0", "qwen.qwen3-coder-480b-a35b-v1:0"],
        "Writer": ["us.writer.palmyra-x4-v1:0", "us.writer.palmyra-x5-v1:0"]
    }
    
    col1, col2 = st.columns(2)
    with col1:
        grader_provider = st.selectbox("Grader Provider", options=list(MODEL_CATEGORIES.keys()), key="grader_provider")
    with col2:
        grader_model_id = st.selectbox("Grader Model", options=MODEL_CATEGORIES[grader_provider], key="grader_model")
    
    st.caption(f"💡 The grading model ({grader_model_id}) will evaluate responses generated by {model_id}")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Evaluation Questions")
        with st.container(border=True):
            for idx, question in enumerate(eval_data):
                st.markdown(f"**Question {idx+1}:**")
                st.markdown(f"*{question['question']}*")
                with st.expander("View expected answer criteria"):
                    st.markdown(question['golden_answer'])
                if idx < len(eval_data) - 1:
                    st.divider()
        
        model_eval_button = st.button("Run Model Evaluation", type="primary", key="model_eval", use_container_width=True)

    with col2:
        st.subheader("Automated Evaluation Results")
        if model_eval_button or st.session_state.model_eval_completed:
            if not st.session_state.model_eval_completed:
                with st.spinner("Running evaluation using the selected model as a grader..."):
                    results, score = run_model_based_evaluation(eval_data)
                    st.session_state.model_eval_results = results
                    st.session_state.model_eval_score = score
                    st.session_state.model_eval_completed = True
            
            # Display score
            correct_count = [r['grade'] for r in st.session_state.model_eval_results].count('correct')
            render_score_card(st.session_state.model_eval_score, correct_count, len(eval_data))
            
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
                    
                    st.markdown(f"**Model's Evaluation:** {result['grade'].upper()}")

def main():
    """Main application function that orchestrates the app flow"""
    # Apply page configuration and styling
    load_css()
    
    # Initialize session state
    common.initialize_session_state()
    init_session_state()
    
    # Create AWS Bedrock client
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1', 
    )
    
    # Render sidebar
    render_sidebar()
    
    # Main title
    st.markdown("<h1 class='main-header'>LLM Model Evaluation Methods</h1>", unsafe_allow_html=True)
    
    st.markdown("""<div class="info-box">
    Explore different methods for evaluating LLM outputs using various foundation models. Learn about code-based grading for objective answers, 
    human-based grading for subjective responses, and model-based grading for automated evaluation at scale.
    </div>""", unsafe_allow_html=True)
    
    # Get evaluation data
    code_eval_data, human_eval_data = get_evaluation_data()
    
    # Create tabs
    tabs = ["📝 Code-based Grading", "👤 Human-based Grading", "🤖 Model-based Grading"]
    tab1, tab2, tab3 = st.tabs(tabs)
    
    # Render tab content
    with tab1:
        render_code_based_tab(code_eval_data)
    
    with tab2:
        render_human_based_tab(human_eval_data)
    
    with tab3:
        render_model_based_tab(human_eval_data)
    
    # Footer
    st.markdown("---")
    st.markdown("<div class='footer'>© 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

# Main execution flow
if __name__ == "__main__":
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()
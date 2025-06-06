
import streamlit as st
import random
from datetime import datetime
import pandas as pd
import uuid
import utils.common as common
import utils.authenticate as authenticate
# --- Constants ---
AWS_ORANGE = "#FF9900"
AWS_BLUE = "#232F3E"
AWS_LIGHT_BLUE = "#1A73E8"
AWS_LIGHT_GRAY = "#F5F5F5"
AWS_GRAY = "#666666"
AWS_WHITE = "#FFFFFF"
AWS_GREEN = "#008000"
AWS_RED = "#D13212"

common.initialize_session_state()

# --- Session State Management ---
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    
    defaults = {
        'current_question_index': 0,
        'score': 0,
        'answers': {},
        'quiz_complete': False,
        'questions_selected': False,
        'selected_questions': [],
        'total_questions': 15,
        'time_started': datetime.now(),
        'auto_advance': True
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_session():
    """Reset the session state to default values."""
    st.session_state.current_question_index = 0
    st.session_state.score = 0
    st.session_state.answers = {}
    st.session_state.quiz_complete = False
    st.session_state.questions_selected = False
    st.session_state.selected_questions = []
    st.session_state.time_started = datetime.now()

# --- Question Selection ---
def select_random_questions(questions_data, num_questions=10):
    """Select random questions ensuring at least one from each category."""
    # Check if we need to select questions
    if not st.session_state.questions_selected:
        # Reset time when new questions are selected
        st.session_state.time_started = datetime.now()
        
        # Make a copy of all questions to avoid modifying the original
        available_questions = questions_data.copy()
        random.shuffle(available_questions)
        
        # Ensure we have at least one question from each category
        selected_questions = []
        categories = set(q["category"] for q in available_questions)
        
        for category in categories:
            category_questions = [q for q in available_questions if q["category"] == category]
            if category_questions:
                selected = random.choice(category_questions)
                selected_questions.append(selected)
                available_questions.remove(selected)
        
        # Fill the rest randomly
        remaining_needed = num_questions - len(selected_questions)
        if remaining_needed > 0:
            random.shuffle(available_questions)
            selected_questions.extend(available_questions[:remaining_needed])
        
        # Update IDs to be sequential
        for i, question in enumerate(selected_questions):
            question["id"] = i + 1
        
        # Update session state
        st.session_state.selected_questions = selected_questions
        st.session_state.questions_selected = True
        st.session_state.total_questions = len(selected_questions)
        st.session_state.score = 0
        st.session_state.answers = {}

# --- Navigation Functions ---
def go_to_next_question():
    """Navigate to the next question or complete the quiz if on last question."""
    if st.session_state.current_question_index < len(st.session_state.selected_questions) - 1:
        st.session_state.current_question_index += 1
    else:
        st.session_state.quiz_complete = True

def go_to_previous_question():
    """Navigate to the previous question if available."""
    if st.session_state.current_question_index > 0:
        st.session_state.current_question_index -= 1

def answer_selected(option_key, question_id):
    """Process a selected answer and update score."""
    q_id = str(question_id)
    question = next((q for q in st.session_state.selected_questions if str(q["id"]) == q_id), None)
    
    # Check if this question has already been answered
    already_answered = q_id in st.session_state.answers
    
    # Record the answer
    st.session_state.answers[q_id] = option_key
    
    # Update score if correct and not already answered
    if not already_answered and option_key == question["correct_answer"]:
        st.session_state.score += 1
    
    # Auto-advance to next question if enabled
    if st.session_state.auto_advance and not already_answered:
        if st.session_state.current_question_index < len(st.session_state.selected_questions) - 1:
            st.session_state.current_question_index += 1

# --- Score Calculation ---
def calculate_score():
    """Calculate the current score based on answered questions."""
    correct_count = 0
    for q_id, user_answer in st.session_state.answers.items():
        question = next((q for q in st.session_state.selected_questions if str(q["id"]) == q_id), None)
        if question and user_answer == question["correct_answer"]:
            correct_count += 1
    return correct_count

# --- Page Styling ---
def apply_custom_styling():
    """Apply custom CSS styling for the app."""
    st.markdown(f"""
    <style>
        .main-header {{
            font-size: 2.5rem;
            color: {AWS_ORANGE};
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: {AWS_BLUE};
            margin-bottom: 1rem;
        }}
        .question-card {{
            background-color: {AWS_WHITE};
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-left: 5px solid {AWS_ORANGE};
        }}
        .option-radio {{
            margin: 8px 0;
            padding: 12px 15px;
            border-radius: 5px;
        }}
        .category-tag {{
            background-color: {AWS_BLUE};
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            display: inline-block;
            margin-bottom: 15px;
        }}
        .explanation-box {{
            padding: 18px;
            border-radius: 5px;
            background-color: #f0f8ff;
            margin-top: 20px;
            border-left: 4px solid {AWS_LIGHT_BLUE};
        }}
        .stats-box {{
            background-color: {AWS_LIGHT_GRAY};
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .aws-button {{
            background-color: {AWS_ORANGE};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 18px;
            margin: 8px 5px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .aws-secondary-button {{
            background-color: {AWS_BLUE};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 18px;
            margin: 8px 5px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .progress-container {{
            margin: 20px 0;
            padding: 15px;
            background-color: {AWS_WHITE};
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .progress-label {{
            font-weight: 600;
            color: {AWS_BLUE};
            margin-bottom: 8px;
        }}
        .stProgress > div > div > div > div {{
            background-color: {AWS_ORANGE};
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: {AWS_GRAY};
            font-size: 0.8rem;
        }}
        /* Responsive styling */
        @media (max-width: 768px) {{
            .main-header {{
                font-size: 2rem;
            }}
            .sub-header {{
                font-size: 1.2rem;
            }}
            .question-card {{
                padding: 15px;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

# --- UI Components ---
def render_sidebar(all_questions):
    """Render the sidebar with controls and stats."""
        
    # Quiz settings
    st.markdown("### Quiz Settings")
    
    # Number of questions selector
    num_questions = st.slider("Number of Questions", min_value=5, max_value=15, value=st.session_state.total_questions, step=1)
    if num_questions != st.session_state.total_questions:
        st.session_state.total_questions = num_questions
        st.session_state.questions_selected = False
        select_random_questions(all_questions, num_questions)
        st.rerun()
    
    # Auto-advance toggle
    st.session_state.auto_advance = st.checkbox(
        "Auto-advance to next question", 
        value=False
    )
    
    # Quiz controls
    # st.markdown("### Quiz Controls")
    # if st.button("New Quiz", key="new_quiz"):
    #     reset_session()
    #     select_random_questions(all_questions, st.session_state.total_questions)
    #     st.rerun()
        
    if not st.session_state.quiz_complete and len(st.session_state.answers) > 0:
        if st.button("Skip to Results", key="skip_results"):
            st.session_state.quiz_complete = True
            st.rerun()
    
    # Navigation
    if not st.session_state.quiz_complete and st.session_state.selected_questions:
        question_nav = st.selectbox(
            "Jump to Question",
            [f"Question {i+1}" for i in range(len(st.session_state.selected_questions))],
            index=st.session_state.current_question_index
        )
        if st.button("Go", key="go_btn"):
            q_idx = int(question_nav.split()[1]) - 1
            st.session_state.current_question_index = q_idx
            st.rerun()
    
    # Quiz progress
    st.markdown("### Quiz Progress")
    display_quiz_stats()

def display_quiz_stats():
    """Display quiz statistics in the sidebar."""
    if not st.session_state.selected_questions:
        return
        
    total_questions = len(st.session_state.selected_questions)
    answered_questions = len(st.session_state.answers)
    progress_percentage = (answered_questions / total_questions) if total_questions > 0 else 0
    
    st.progress(progress_percentage)
    st.write(f"Completed: {answered_questions}/{total_questions} questions ({progress_percentage*100:.0f}%)")
    
    # Recalculate score from answers
    correct_answers = calculate_score()
    st.session_state.score = correct_answers  # Update the session state score
    
    # Score
    accuracy = (correct_answers / answered_questions) * 100 if answered_questions > 0 else 0
    st.write(f"Score: {correct_answers}/{answered_questions} correct ({accuracy:.1f}%)")
    
    # Time elapsed
    time_elapsed = datetime.now() - st.session_state.time_started
    minutes = time_elapsed.seconds // 60
    seconds = time_elapsed.seconds % 60
    st.write(f"Time: {minutes}m {seconds}s")

def display_question(q_index):
    """Display the current question and handle answers."""
    question = st.session_state.selected_questions[q_index]
    q_id = str(question["id"])
    
    st.markdown(f"<div class='question-card'>", unsafe_allow_html=True)
    
    # Display category tag
    category = question.get("category", "General")
    st.markdown(f"<span class='category-tag'>{category}</span>", unsafe_allow_html=True)
    
    # Display question
    st.markdown(f"<h2 class='sub-header'>Question {q_index + 1} of {len(st.session_state.selected_questions)}</h2>", unsafe_allow_html=True)
    st.write(question["question"])
    
    # Check if user has already answered
    user_answered = q_id in st.session_state.answers
    
    # Display options as radio buttons
    if not user_answered:
        options = question["options"]
        option_labels = [f"{key}: {value}" for key, value in options.items()]
        option_keys = list(options.keys())
        
        selected_option = st.radio(
            "Select your answer:",
            options=option_labels,
            key=f"radio_{q_id}",
            label_visibility="collapsed",
            index=None
        )
        
        # Extract option key from selected label (e.g., "A: Some text" -> "A")
        if selected_option:
            selected_key = selected_option.split(":")[0]
            
            col1, col2 = st.columns([5, 1])
            with col2:
                if st.button("Submit", key=f"submit_{q_id}", type="primary"):
                    answer_selected(selected_key, q_id)
                    st.rerun()
    else:
        # Display results for already answered question
        user_answer = st.session_state.answers[q_id]
        for option_key, option_text in question["options"].items():
            is_correct = option_key == question["correct_answer"]
            user_selected = option_key == user_answer
            
            if user_selected and is_correct:
                st.success(f"{option_key}: {option_text} ✓")
            elif user_selected and not is_correct:
                st.error(f"{option_key}: {option_text} ✗")
            elif not user_selected and is_correct:
                st.warning(f"{option_key}: {option_text} (Correct Answer)")
            else:
                st.write(f"{option_key}: {option_text}")
        
        # Show explanation
        st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
        st.markdown("### Explanation")
        st.markdown(question["explanation"][user_answer])
        
        if user_answer != question["correct_answer"]:
            st.markdown("### Correct Answer")
            st.markdown(question["explanation"][question["correct_answer"]])
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.current_question_index > 0:
            if st.button("⬅️ Previous", key="prev_btn"):
                go_to_previous_question()
                st.rerun()
    
    with col3:
        if st.session_state.current_question_index < len(st.session_state.selected_questions) - 1:
            next_text = "Next ➡️"
            if st.button(next_text, key="next_btn"):
                go_to_next_question()
                st.rerun()
        else:
            if st.button("Finish Quiz 🏁", key="finish_btn"):
                st.session_state.quiz_complete = True
                st.rerun()

def display_results():
    """Display the quiz results."""
    st.markdown("<div class='question-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Quiz Results</h2>", unsafe_allow_html=True)
    
    # Calculate metrics
    total_questions = len(st.session_state.selected_questions)
    answered_questions = len(st.session_state.answers)
    correct_answers = calculate_score()
    accuracy = (correct_answers / answered_questions) * 100 if answered_questions > 0 else 0
    completion_percentage = (answered_questions / total_questions) * 100 if total_questions > 0 else 0
    
    # Time taken
    time_elapsed = datetime.now() - st.session_state.time_started
    minutes = time_elapsed.seconds // 60
    seconds = time_elapsed.seconds % 60
    
    # Display summary
    st.markdown(f"<div class='stats-box'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Your Score: {correct_answers}/{answered_questions}")
        st.markdown(f"### Accuracy: {accuracy:.1f}%")
    
    with col2:
        st.markdown(f"### Time Taken: {minutes}m {seconds}s")
        st.markdown(f"### Questions Answered: {answered_questions}/{total_questions} ({completion_percentage:.1f}%)")
    
    # Performance assessment
    if accuracy >= 80:
        st.success("Excellent! You have a strong understanding of AWS AI Practitioner concepts.")
    elif accuracy >= 60:
        st.info("Good job! You have a reasonable understanding, but some areas need improvement.")
    else:
        st.warning("You might need more study on AWS AI Practitioner concepts.")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Category performance analysis
    display_category_performance()
    
    # Question breakdown
    display_question_breakdown()
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Review Questions", key="review_btn"):
            st.session_state.quiz_complete = False
            st.session_state.current_question_index = 0
            st.rerun()
    
   
    st.markdown("</div>", unsafe_allow_html=True)

def display_category_performance():
    """Display performance breakdown by category."""
    st.markdown("### Performance by Category")
    
    # Group answers by category
    category_performance = {}
    for question in st.session_state.selected_questions:
        q_id = str(question["id"])
        category = question.get("category", "General")
        
        if category not in category_performance:
            category_performance[category] = {"correct": 0, "total": 0, "answered": 0}
        
        if q_id in st.session_state.answers:
            category_performance[category]["answered"] += 1
            if st.session_state.answers[q_id] == question["correct_answer"]:
                category_performance[category]["correct"] += 1
        
        category_performance[category]["total"] += 1
    
    # Create DataFrame for category performance
    if category_performance:
        data = []
        for category, stats in category_performance.items():
            accuracy = (stats["correct"] / stats["answered"]) * 100 if stats["answered"] > 0 else 0
            data.append({
                "Category": category,
                "Questions": stats["total"],
                "Answered": stats["answered"],
                "Correct": stats["correct"],
                "Accuracy": f"{accuracy:.1f}%"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

def display_question_breakdown():
    """Display detailed breakdown of all questions."""
    st.markdown("### Question Breakdown")
    
    for question in st.session_state.selected_questions:
        q_id = str(question["id"])
        user_answer = st.session_state.answers.get(q_id, "Not answered")
        is_correct = user_answer == question["correct_answer"] if q_id in st.session_state.answers else False
        
        with st.expander(f"Question {question['id']}: {question['question'][:100]}..."):
            st.write(question["question"])
            st.write(f"**Your answer:** Option {user_answer}")
            st.write(f"**Correct answer:** Option {question['correct_answer']}")
            
            result_text = "✓ Correct" if is_correct else "✗ Incorrect" if user_answer != "Not answered" else "⚠️ Not Answered"
            st.write(f"**Result:** {result_text}")
            
            # Show explanation
            st.markdown("### Explanation")
            if q_id in st.session_state.answers:
                st.markdown(question["explanation"][user_answer])
                
                if user_answer != question["correct_answer"]:
                    st.markdown("### Correct Answer")
                    st.markdown(question["explanation"][question["correct_answer"]])
            else:
                st.markdown(question["explanation"][question["correct_answer"]])

# --- Data Import ---
# This would typically be in a separate file, but keeping it here for simplicity
def get_quiz_data():
    """Return the quiz question data."""
    return [
    {
        "id": 1,
        "question": "A financial services company is developing an application that needs to answer customer questions about their complex investment products. The application should leverage the company's proprietary documentation while ensuring responses are accurate and relevant. Which approach would be most effective?",
        "options": {
            "A": "Fine-tune a foundation model with the company's proprietary documentation to create a specialized model",
            "B": "Use a Retrieval Augmented Generation (RAG) approach to retrieve relevant information from the documentation",
            "C": "Apply chain-of-thought prompting techniques with a large pre-trained model",
            "D": "Deploy multiple specialized models, each trained on a specific investment product"
        },
        "correct_answer": "B",
        "explanation": {
            "A": "Fine-tuning could potentially work but would require significant resources and may still suffer from hallucinations or outdated information when products are updated.",
            "B": "Correct! RAG addresses the challenge of factual accuracy by retrieving relevant up-to-date information from the company's documentation before generating responses, ensuring the foundation model has the appropriate context for answering questions about complex investment products.",
            "C": "Chain-of-thought prompting is useful for complex reasoning tasks but doesn't address the need to incorporate accurate, proprietary information that may not be in the model's training data.",
            "D": "Deploying multiple specialized models would increase operational complexity and cost without necessarily improving accuracy compared to a RAG-based approach."
        },
        "category": "Foundation Model Applications"
    },
    {
        "id": 2,
        "question": "A product team is developing a chatbot that assists users with troubleshooting technical issues. The team needs to customize a foundation model's outputs to be more technically precise and follow company guidelines. Given that they have limited ML expertise and a tight deadline, which approach should they prioritize first?",
        "options": {
            "A": "Continue pretraining the foundation model with technical documentation",
            "B": "Implement fine-tuning using a dataset of past customer interactions",
            "C": "Create a comprehensive prompt engineering strategy with specific constraints and guidelines",
            "D": "Deploy multiple smaller models specialized for different types of technical issues"
        },
        "correct_answer": "C",
        "explanation": {
            "A": "Continued pretraining requires substantial computational resources, ML expertise, and time, making it unsuitable for a team with limited ML expertise and a tight deadline.",
            "B": "Fine-tuning requires less resources than continued pretraining but still demands ML expertise and a well-prepared dataset, which may be challenging given the constraints.",
            "C": "Correct! Prompt engineering provides the most immediate path to customizing model outputs with minimal ML expertise and without model training. By crafting effective prompts that include company guidelines and technical constraints, the team can guide the model's outputs to be more precise and compliant.",
            "D": "Deploying multiple specialized models would increase the complexity, cost, and management overhead, which is inconsistent with the team's limited ML expertise and tight deadline."
        },
        "category": "Prompt Engineering"
    },
    {
        "id": 3,
        "question": "A healthcare startup is building an application that assists doctors in diagnosing rare diseases. They need the foundation model to clearly explain its reasoning process when suggesting possible diagnoses. Which prompting technique would be most appropriate?",
        "options": {
            "A": "Zero-shot prompting that directly asks for diagnostic suggestions",
            "B": "Few-shot prompting with examples of symptoms and associated diagnoses",
            "C": "Chain-of-thought prompting that encourages step-by-step medical reasoning",
            "D": "Adversarial prompting to challenge initial diagnostic assumptions"
        },
        "correct_answer": "C",
        "explanation": {
            "A": "Zero-shot prompting wouldn't provide the detailed reasoning process necessary for medical diagnoses, especially for rare diseases where context is critical.",
            "B": "Few-shot prompting could improve accuracy by providing examples, but doesn't explicitly encourage the model to show its reasoning process, which is crucial for medical applications.",
            "C": "Correct! Chain-of-thought prompting explicitly encourages the model to work through its reasoning step-by-step, making the diagnostic process more transparent and interpretable for doctors. This is especially important in healthcare where understanding the 'why' behind a diagnosis is critical.",
            "D": "Adversarial prompting might help identify edge cases, but could introduce unnecessary complexity and wouldn't directly address the need for clear explanation of reasoning."
        },
        "category": "Prompt Engineering"
    },
    {
        "id": 4,
        "question": "An e-commerce company has implemented a customer service chatbot using a foundation model. Security testing reveals that users can manipulate the bot to reveal internal pricing strategies by using specific prompt patterns. This is an example of:",
        "options": {
            "A": "Chain-of-thought exploitation, where reasoning patterns are manipulated",
            "B": "Few-shot poisoning, where examples mislead the model",
            "C": "Prompt leaking, where sensitive information is extracted through carefully crafted prompts",
            "D": "Retrieval contamination, where indexed content is misappropriated"
        },
        "correct_answer": "C",
        "explanation": {
            "A": "Chain-of-thought exploitation isn't a recognized security issue; it refers to manipulating reasoning patterns which isn't what's happening here.",
            "B": "Few-shot poisoning would involve providing misleading examples, but this scenario describes extracting existing information, not misleading the model with examples.",
            "C": "Correct! Prompt leaking occurs when sensitive information (in this case, internal pricing strategies) is extracted from a model through carefully crafted prompts. This is a security vulnerability where the model reveals information it shouldn't share.",
            "D": "Retrieval contamination isn't a standard term in this context, and the issue isn't related to retrieving indexed content but rather extracting sensitive information programmed into the model."
        },
        "category": "Foundation Model Security"
    },
    {
        "id": 5,
        "question": "A multinational company is implementing a foundation model-based system that must comply with different content policies across regions. The company needs to ensure potentially harmful content is filtered according to regional standards. Which AWS service feature would best address this requirement?",
        "options": {
            "A": "Amazon SageMaker automatic model assessment for regional bias",
            "B": "Amazon Bedrock Knowledge Base regional content filtering",
            "C": "Guardrails for Amazon Bedrock with region-specific content policies",
            "D": "Amazon SageMaker Clarify with geographic compliance monitoring"
        },
        "correct_answer": "C",
        "explanation": {
            "A": "Amazon SageMaker's model assessment capabilities focus on model performance and bias detection, not regional content filtering requirements.",
            "B": "Amazon Bedrock Knowledge Base focuses on information retrieval and augmentation, not enforcing content policies based on regional standards.",
            "C": "Correct! Guardrails for Amazon Bedrock allows companies to configure harmful content filtering based on specific policies, which can be tailored to different regional requirements. It enables defining prohibited topics and content types with natural language descriptions that can align with regional standards.",
            "D": "Amazon SageMaker Clarify focuses on bias detection and explainability, not specifically on enforcing region-specific content policies."
        },
        "category": "AWS AI Services"
    },
    {
        "id": 6,
        "question": "A developer is creating a document processing application that needs to use both Amazon Textract for text extraction and a foundation model for content summarization. Instead of building a complex pipeline, the developer wants a solution that can orchestrate these steps and interact with company databases. Which AWS service would be most appropriate?",
        "options": {
            "A": "AWS Step Functions with Lambda integrations",
            "B": "Amazon Bedrock Agents with API connections",
            "C": "Amazon SageMaker Pipelines with custom components",
            "D": "AWS Glue with ETL capabilities"
        },
        "correct_answer": "B",
        "explanation": {
            "A": "While AWS Step Functions could orchestrate this workflow, it would require significant custom development for the foundation model integration and human-like task processing.",
            "B": "Correct! Amazon Bedrock Agents can decompose tasks, execute actions against APIs (like Textract), and interact with knowledge bases, making it ideal for orchestrating multi-step tasks using company systems and foundation models without requiring extensive custom code.",
            "C": "Amazon SageMaker Pipelines is designed for ML model training and deployment workflows, not for orchestrating document processing and foundation model interactions with databases.",
            "D": "AWS Glue is primarily for ETL processes and data integration, not for orchestrating AI services and foundation models."
        },
        "category": "AWS AI Services"
    },
    {
        "id": 7,
        "question": "A company wants to evaluate multiple foundation models to determine which performs best on their summarization task. They need to compare models based on both automated metrics and human feedback. Which AWS service would be most appropriate for this evaluation process?",
        "options": {
            "A": "Amazon SageMaker Model Monitor with custom metrics",
            "B": "Amazon Bedrock model evaluation with ROUGE metrics and human evaluation",
            "C": "Amazon CloudWatch with custom metric integration",
            "D": "Amazon Comprehend with document analysis capabilities"
        },
        "correct_answer": "B",
        "explanation": {
            "A": "Amazon SageMaker Model Monitor is designed for monitoring models in production, not for comparative evaluation of multiple foundation models on specific tasks.",
            "B": "Correct! Amazon Bedrock model evaluation is specifically designed to evaluate and compare foundation models using both automatic metrics (like ROUGE for summarization) and human-based evaluations with customizable datasets tailored for generative AI tasks.",
            "C": "Amazon CloudWatch can monitor metrics but doesn't provide specialized capabilities for foundation model evaluation with metrics like ROUGE or human feedback integration.",
            "D": "Amazon Comprehend is focused on natural language processing tasks, not on evaluating and comparing foundation models for tasks like summarization."
        },
        "category": "Model Evaluation"
    },
    {
        "id": 8,
        "question": "A data science team is fine-tuning a foundation model for a medical Q&A application. They need to evaluate whether their fine-tuned model accurately answers questions while minimizing hallucinations. Which evaluation metric would be most appropriate for this specific requirement?",
        "options": {
            "A": "BLEU score to compare generated answers with reference translations",
            "B": "F1 score to evaluate the precision and recall of the model's answers",
            "C": "ROUGE score to assess the quality of generated summaries",
            "D": "BERTScore to measure semantic similarity between generated and reference answers"
        },
        "correct_answer": "B",
        "explanation": {
            "A": "BLEU score is primarily used for evaluating machine translation quality, not for assessing factual accuracy in question answering.",
            "B": "Correct! F1 score is widely used in traditional ML classification and Q&A evaluation to measure both precision (accuracy of answers provided) and recall (coverage of relevant information), making it suitable for measuring factual accuracy while minimizing hallucinations.",
            "C": "ROUGE score is designed for evaluating text summarization, not specifically for question answering accuracy.",
            "D": "While BERTScore can measure semantic similarity, it doesn't directly address factual accuracy and hallucination prevention as effectively as F1 score for Q&A tasks."
        },
        "category": "Model Evaluation"
    },
    {
        "id": 9,
        "question": "A company is implementing a RAG system to improve their customer service chatbot. They need to decide on the most efficient vector database to store embeddings of their product documentation. Which AWS service would provide the most seamless integration with their existing PostgreSQL database while enabling vector search?",
        "options": {
            "A": "Amazon OpenSearch Service with custom vector storage",
            "B": "Amazon DynamoDB with sparse vector indexing",
            "C": "Amazon Aurora PostgreSQL with pgvector extension",
            "D": "Amazon DocumentDB with vector similarity plugin"
        },
        "correct_answer": "C",
        "explanation": {
            "A": "While Amazon OpenSearch Service supports vector search, it would require maintaining a separate service from their existing PostgreSQL database.",
            "B": "Amazon DynamoDB does support vector search via zero-ETL integration, but wouldn't provide seamless integration with an existing PostgreSQL database.",
            "C": "Correct! Amazon Aurora PostgreSQL with pgvector extension allows the company to add vector search capabilities directly to their existing PostgreSQL database, avoiding the need to manage multiple databases or complex ETL processes.",
            "D": "Amazon DocumentDB is MongoDB-compatible, not PostgreSQL-compatible, and would require migration from their existing database system."
        },
        "category": "Foundation Model Applications"
    },
    {
        "id": 10,
        "question": "An insurance company is implementing a system for processing claims using foundation models. They need to ensure that the system follows strict regulatory guidelines and maintains consistent decision-making processes. Which approach would best address these requirements?",
        "options": {
            "A": "Deploy Amazon Bedrock with a custom RAG system using unstructured claims documents",
            "B": "Use Amazon Bedrock Agents with Knowledge Bases and Guardrails for claims processing",
            "C": "Implement Amazon SageMaker to train a custom model on historical claims data",
            "D": "Create a hybrid solution combining traditional rule-based processing and foundation models"
        },
        "correct_answer": "B",
        "explanation": {
            "A": "While a RAG system could help with accessing relevant documents, it doesn't directly address the regulatory compliance and consistent decision-making requirements.",
            "B": "Correct! Amazon Bedrock Agents with Knowledge Bases provides structured access to company data, while Guardrails for Amazon Bedrock allows configuring content filtering based on regulatory policies. Together, they ensure the system follows guidelines while maintaining consistent decision processes.",
            "C": "Training a custom model on historical data could perpetuate existing biases and wouldn't necessarily ensure regulatory compliance without additional safeguards.",
            "D": "A hybrid solution could work but would be more complex to manage and might lead to inconsistencies between the rule-based and foundation model components."
        },
        "category": "Foundation Model Applications"
    },
    {
        "id": 11,
        "question": "A company is preparing data for fine-tuning a foundation model to analyze technical support tickets. Which data preparation consideration is MOST critical for this specific use case?",
        "options": {
            "A": "Ensuring data is highly varied with examples from multiple industries",
            "B": "Curating a balanced dataset that represents the variety of support ticket types and resolutions",
            "C": "Maximizing the dataset size by including as many tickets as possible, regardless of quality",
            "D": "Using only tickets written in formal technical language to improve model precision"
        },
        "correct_answer": "B",
        "explanation": {
            "A": "Including examples from multiple industries could dilute the model's performance on the company's specific technical support domain.",
            "B": "Correct! A balanced, curated dataset that represents the diversity of support ticket types and resolutions will help the model generalize across the company's support scenarios while maintaining domain relevance. This approach aligns with the quality-over-quantity principle for fine-tuning data.",
            "C": "Maximizing dataset size without considering quality could introduce noise and irrelevant patterns, potentially degrading model performance on the target task.",
            "D": "Restricting to only formal technical language would limit the model's ability to handle real-world customer queries, which often contain informal language and diverse writing styles."
        },
        "category": "Fine-tuning Process"
    },
    {
        "id": 12,
        "question": "A research organization is converting their academic papers into a format suitable for RAG implementation. Their papers contain complex equations, citations, and hierarchical section structures. Which preprocessing steps would be most effective for this scenario?",
        "options": {
            "A": "Convert all papers to plain text, removing equations and citations to simplify processing",
            "B": "Chunk documents by arbitrary character count regardless of semantic boundaries",
            "C": "Chunk documents semantically by sections and maintain contextual metadata like citations",
            "D": "Convert all mathematical equations to descriptive text regardless of complexity"
        },
        "correct_answer": "C",
        "explanation": {
            "A": "Removing equations and citations would significantly reduce the value of academic papers, eliminating crucial information needed for accurate retrieval and generation.",
            "B": "Arbitrary chunking by character count could split important contextual information across chunks, reducing retrieval effectiveness, especially for academic content with logical section structures.",
            "C": "Correct! Chunking documents semantically by natural section boundaries while preserving metadata like citations maintains the logical structure of academic papers and enables more effective retrieval of contextually complete information.",
            "D": "Converting all equations to descriptive text might lose mathematical precision required in academic papers and could make certain concepts more difficult to understand."
        },
        "category": "Foundation Model Applications"
    },
    {
        "id": 13,
        "question": "A large retail company is developing a customer service chatbot that needs to access real-time inventory information. The development team wants to use Amazon Bedrock but needs to connect the foundation model with their inventory management system. Which solution would provide the most streamlined approach?",
        "options": {
            "A": "Implement a Lambda function that retrieves inventory data and use it as a custom action in Amazon Bedrock Agents",
            "B": "Fine-tune an Amazon Bedrock model with static inventory data that's updated weekly",
            "C": "Create a RAG system with Knowledge Bases that stores cached inventory data updated daily",
            "D": "Develop a custom wrapper around the Amazon Bedrock API that checks inventory before processing each query"
        },
        "correct_answer": "A",
        "explanation": {
            "A": "Correct! Using Lambda functions as custom actions in Amazon Bedrock Agents allows the chatbot to retrieve real-time inventory data directly from the management system whenever needed, ensuring responses contain the most up-to-date information without requiring complex custom development.",
            "B": "Fine-tuning with static inventory data would result in outdated information as soon as inventory levels change, making it unsuitable for real-time inventory queries.",
            "C": "A RAG system with daily updates would still contain outdated information throughout the day, which is problematic for inventory data that changes frequently.",
            "D": "A custom wrapper would require significant development effort and wouldn't leverage the built-in capabilities of Amazon Bedrock services designed for this purpose."
        },
        "category": "AWS AI Services"
    },
    {
        "id": 14,
        "question": "A financial advisory firm wants to create an application that generates personalized investment recommendations based on client profiles. They need to ensure the foundation model doesn't make up financial information or present speculative advice as fact. Which approach would best address this concern?",
        "options": {
            "A": "Apply prompt engineering techniques that instruct the model to only discuss verified investment strategies",
            "B": "Use a RAG approach with a knowledge base containing only approved investment materials",
            "C": "Fine-tune a foundation model exclusively on the firm's approved financial advice documents",
            "D": "Implement a custom template system that restricts the model to predefined response formats"
        },
        "correct_answer": "B",
        "explanation": {
            "A": "Prompt engineering alone may not be sufficient to prevent hallucinations, as models can still generate plausible-sounding but fabricated financial information despite instructions.",
            "B": "Correct! A RAG approach ensures that responses are grounded in the firm's approved investment materials by retrieving relevant information before generation. This minimizes hallucinations and ensures recommendations are based on verified information rather than the model's general knowledge.",
            "C": "Fine-tuning could help but might still result in hallucinations when the model encounters scenarios outside its training data, and wouldn't adapt to new investment products without retraining.",
            "D": "A template system would be too restrictive for personalized advice and wouldn't address the core issue of preventing fabricated information within those templates."
        },
        "category": "Foundation Model Applications"
    },
    {
        "id": 15,
        "question": "A healthcare company is assessing several foundation models for potential deployment in a patient-facing application. They need to understand how each model responds to sensitive medical queries. Which AWS feature would be most appropriate for this evaluation?",
        "options": {
            "A": "Amazon SageMaker Autopilot for automated hyperparameter tuning across models",
            "B": "Amazon Bedrock model evaluation with customized healthcare-focused test datasets",
            "C": "Amazon Comprehend Medical for extracting entities from model responses",
            "D": "Amazon SageMaker Feature Store for comparing model feature importance"
        },
        "correct_answer": "B",
        "explanation": {
            "A": "Amazon SageMaker Autopilot focuses on automated ML model building and hyperparameter tuning, not on evaluating how foundation models respond to specific types of queries.",
            "B": "Correct! Amazon Bedrock model evaluation allows the company to create customized healthcare-focused test datasets and systematically evaluate how each foundation model responds to sensitive medical queries, helping them select the most appropriate model for their specific use case.",
            "C": "Amazon Comprehend Medical extracts medical entities from text but doesn't provide comparative evaluation of different foundation models' responses.",
            "D": "Amazon SageMaker Feature Store is for storing, sharing, and managing ML features, not for evaluating foundation model responses to specific queries."
        },
        "category": "Model Evaluation"
    }
]

# --- Main App ---
def main():
    """Main application entry point."""
    # Set page configuration
    st.set_page_config(
        page_title="AWS AI Practitioner Quiz",
        page_icon="☁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Apply custom styling
    apply_custom_styling()
    
    # Load quiz data
    all_questions = get_quiz_data()
    
    # Select random questions if not already done
    select_random_questions(all_questions, st.session_state.total_questions)
    
   
    # Header with AWS logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image("assets/images/AWS-Certified-AI-Practitioner_badge.png", width=100)
    with col2:
        st.markdown("<h1 class='main-header'>AWS AI Practitioner Quiz</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Domain 3: Applications of Foundation Models</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        common.render_sidebar()
    
    # Create a 70/30 layout
    col1, col2 = st.columns([0.7, 0.3])     
        # Get model and parameters from the right column
    with col2:
        # Render sidebar
        with st.container(border=True):
            render_sidebar(all_questions)
    
    with col1:
        # Display appropriate content based on quiz state
        if st.session_state.quiz_complete:
            display_results()
        else:
            display_question(st.session_state.current_question_index)
    
    # Footer
    st.markdown("<div class='footer'>© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()

import streamlit as st

# Define the questions for the knowledge check
knowledge_check_questions = [
    {
        "id": 1,
        "question": "Which parameter controls the randomness and creativity in a language model's output?",
        "options": {
            "A": "Top-K",
            "B": "Temperature",
            "C": "Top-P",
            "D": "Max Length"
        },
        "correct": "B",
        "explanation": "Temperature controls the randomness of the output by scaling the logits before applying the softmax function. Lower temperatures result in less random output, while higher temperatures result in more random output."
    },
    {
        "id": 2,
        "question": "Which AWS service allows you to access various pre-trained foundation models through a single API?",
        "options": {
            "A": "Amazon EC2",
            "B": "Amazon SageMaker",
            "C": "Amazon Bedrock",
            "D": "Amazon Rekognition"
        },
        "correct": "C",
        "explanation": "Amazon Bedrock provides access to a variety of foundation models from different providers through a unified API, allowing developers to easily integrate and experiment with different models."
    },
    {
        "id": 3,
        "question": "What is the primary function of tokenization in the transformer model pipeline?",
        "options": {
            "A": "To convert numerical data into text",
            "B": "To break down text into smaller units (tokens) for processing",
            "C": "To measure the performance of the model",
            "D": "To visualize the model's output"
        },
        "correct": "B",
        "explanation": "Tokenization is the process of breaking down text input into smaller units called tokens, which are then converted into numerical identifiers that can be processed by the model."
    },
    {
        "id": 4,
        "question": "In the context of language models, what does 'context length' refer to?",
        "options": {
            "A": "The maximum length of the generated output",
            "B": "The amount of memory used by the model",
            "C": "The number of previous tokens the model can consider when generating the next token",
            "D": "The time it takes for the model to process input"
        },
        "correct": "C",
        "explanation": "Context length refers to the number of previous tokens (words or subwords) that the model can take into account when generating the next token. It represents the model's ability to maintain coherence over longer pieces of text."
    },
    {
        "id": 5,
        "question": "Which of the following is NOT a common approach for customizing foundation models?",
        "options": {
            "A": "Prompt Engineering",
            "B": "Fine-tuning",
            "C": "Retrieval Augmented Generation (RAG)",
            "D": "Recursive Pattern Matching"
        },
        "correct": "D",
        "explanation": "Recursive Pattern Matching is not a standard approach for customizing foundation models. The common approaches include Prompt Engineering, Fine-tuning, Continued Pre-training, and Retrieval Augmented Generation (RAG)."
    }
]

def display_knowledge_check():
    """Display the knowledge check section"""
    st.header("Knowledge Check")
    st.markdown("""
    Test your understanding of transformer models and AWS AI services with this 
    knowledge check. Select the correct answer for each question and check your results.
    """)
    
    # Start/restart button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Start/Restart Quiz", key="start_quiz"):
            st.session_state.knowledge_check_started = True
            st.session_state.current_question = 0
            st.session_state.answers = {}
            st.session_state.score = 0
    
    if not st.session_state.knowledge_check_started:
        st.info("Click the 'Start Quiz' button to begin the knowledge check.")
        return
    
    # Show current question and progress
    progress = (st.session_state.current_question) / len(knowledge_check_questions)
    st.progress(progress)
    
    if st.session_state.current_question >= len(knowledge_check_questions):
        show_results()
        return
    
    # Display current question
    current_q = knowledge_check_questions[st.session_state.current_question]
    st.subheader(f"Question {current_q['id']}")
    st.markdown(f"**{current_q['question']}**")
    
    # Radio button for options
    answer = st.radio(
        "Select your answer:",
        options=list(current_q["options"].keys()),
        format_func=lambda x: f"{x}: {current_q['options'][x]}",
        key=f"q{current_q['id']}",
        index=None
    )
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    
    with col2:
        if st.button("Next Question", key=f"next_{current_q['id']}"):
            if answer:
                st.session_state.answers[current_q['id']] = answer
                if answer == current_q['correct']:
                    st.session_state.score += 1
                st.session_state.current_question += 1
                st.experimental_rerun()
            else:
                st.warning("Please select an answer before proceeding.")


def show_results():
    """Display the results of the knowledge check"""
    st.subheader("Knowledge Check Results")
    
    total_questions = len(knowledge_check_questions)
    score = st.session_state.score
    percentage = (score / total_questions) * 100
    
    # Display score
    st.markdown(f"### Your Score: {score}/{total_questions} ({percentage:.1f}%)")
    
    # Performance assessment
    if percentage >= 80:
        st.success("🏆 Excellent! You have a strong understanding of transformer models.")
    elif percentage >= 60:
        st.info("👍 Good job! You understand most concepts, but you might want to review some areas.")
    else:
        st.warning("📚 You might need to review the material to strengthen your understanding.")
    
    # Show answers and explanations
    st.subheader("Review Questions and Answers")
    
    for q in knowledge_check_questions:
        q_id = q['id']
        user_answer = st.session_state.answers.get(q_id, "Not answered")
        
        with st.expander(f"Question {q_id}: {q['question']}"):
            st.markdown(f"**Your answer:** {user_answer}: {q['options'].get(user_answer, 'Not answered')}")
            
            if user_answer == q['correct']:
                st.success(f"**Correct answer:** {q['correct']}: {q['options'][q['correct']]}")
            else:
                st.error(f"**Correct answer:** {q['correct']}: {q['options'][q['correct']]}")
            
            st.info(f"**Explanation:** {q['explanation']}")
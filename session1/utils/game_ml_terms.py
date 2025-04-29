import streamlit as st
import random
from utils.common import display_progress, show_tip, reset_game_button

def ml_terms_game():
    st.header("📚 ML Terms")
    st.write("Match the ML terminology to the correct definition")
    
    # Display progress
    display_progress(5)
    
    # Reset game button
    reset_game_button(5)
    
    # Overview of key terms
    st.subheader("Key Machine Learning Terminology")
    
    # Display terminology table
    terms_data = {
        "Term": ["Feature", "Label/Target", "Training Data", "Model", "Inference", "Supervised Learning", "Unsupervised Learning"],
        "Definition": [
            "Input variable used for making predictions",
            "The output variable you're trying to predict",
            "Historical data used to train the ML model",
            "Algorithm that has learned patterns from training data",
            "Using a trained model to make predictions on new data",
            "Learning from labeled examples",
            "Finding patterns in data without labeled examples"
        ]
    }
    
    st.table(terms_data)
    
    # Show tip
    show_tip("Understanding ML terminology is crucial for the AWS AI Practitioner exam. Pay special attention to the differences between features and labels, as well as different learning types.")
    
    # Scenarios
    scenarios = [
        {
            "id": 1,
            "scenario": "A retail company is building a recommendation system. They are using customer age, location, and purchase history as inputs to predict what products a customer might want to buy next.",
            "question": "In this scenario, what would 'customer age, location, and purchase history' be called in ML terminology?",
            "options": ["Features", "Labels", "Training Data", "Models"],
            "answer": "Features",
            "explanation": "Features are the input variables used to make predictions. Customer age, location, and purchase history are all input variables that help predict product recommendations."
        },
        {
            "id": 2,
            "scenario": "A data scientist is building a model to predict customer churn. They've collected data on 10,000 past customers, including whether they churned or not.",
            "question": "What would 'whether they churned or not' be called in ML terminology?",
            "options": ["Feature", "Label/Target", "Inference", "Model"],
            "answer": "Label/Target",
            "explanation": "Label/Target is what you're trying to predict. In this case, whether a customer churned or not is the outcome variable being predicted."
        },
        {
            "id": 3,
            "scenario": "After building an ML model to classify emails as spam or not spam, a company deploys it to analyze incoming emails in real-time.",
            "question": "What is the process of using the trained model to classify new incoming emails called?",
            "options": ["Feature Engineering", "Model Training", "Hyperparameter Tuning", "Inference"],
            "answer": "Inference",
            "explanation": "Inference is the process of using a trained model to make predictions on new data. In this case, classifying new incoming emails using the trained spam detection model."
        },
        {
            "id": 4,
            "scenario": "A developer is working with customer segmentation and needs to group customers based on purchasing behavior without having predefined categories.",
            "question": "Which type of machine learning would be most appropriate for this scenario?",
            "options": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Transfer Learning"],
            "answer": "Unsupervised Learning",
            "explanation": "Unsupervised Learning is used when you want to find patterns or groups in data without predefined categories. Customer segmentation without predefined groups is a classic unsupervised learning task."
        },
        {
            "id": 5,
            "scenario": "A data scientist is preparing data for a machine learning model and decides to convert categorical variables into numerical format, normalize numeric fields, and create new variables based on existing ones.",
            "question": "What is this process called in ML terminology?",
            "options": ["Data Mining", "Feature Engineering", "Data Inference", "Label Encoding"],
            "answer": "Feature Engineering",
            "explanation": "Feature Engineering is the process of transforming raw data into features that better represent the underlying problem, improving model performance. This includes creating new features, normalizing data, and encoding categorical variables."
        }
    ]
    
    st.markdown("---")
    
    # Game scenarios
    for i, scenario in enumerate(scenarios):
        st.markdown(f"""
        <div class="game-card">
            <h3>Scenario {scenario['id']}</h3>
        """, unsafe_allow_html=True)
        
        st.write(scenario["scenario"])
        st.write(f"**Question:** {scenario['question']}")
        
        # Radio button for selection
        user_answer = st.radio(
            "Select the correct answer:", 
            scenario["options"], 
            key=f"terms_{scenario['id']}", index=None
        )
        
        # Submit button
        submit_button = st.button("Submit Answer", key=f"terms_submit_{scenario['id']}")
        
        if submit_button:
            st.session_state.game5_submitted[i] = True
            
            if user_answer == scenario["answer"]:
                st.session_state.game5_score += 1
                st.markdown(f"""
                <div class="correct-answer">
                    <strong>✅ Correct!</strong><br>
                    {scenario['explanation']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="incorrect-answer">
                    <strong>❌ Incorrect.</strong><br>
                    The correct answer is {scenario['answer']}.<br>
                    {scenario['explanation']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Separator between scenarios
        if i < len(scenarios) - 1:
            st.markdown("---")
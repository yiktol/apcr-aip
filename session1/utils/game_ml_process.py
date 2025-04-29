import streamlit as st
import random
from utils.common import display_progress, show_tip, reset_game_button

def ml_process_game():
    st.header("🔄 ML Development Process")
    st.write("Identify the correct phase of the ML development lifecycle for each scenario")
    
    # Display progress
    display_progress(7)
    
    # Reset game button
    reset_game_button(7)
    
    # ML process overview
    st.subheader("Machine Learning Development Lifecycle")
    
    st.image("https://miro.medium.com/max/1400/1*PRkDKYyJ0AwVqyTz7qZnrg.png", caption="ML Development Lifecycle")
    
    st.markdown("""
    The ML development lifecycle typically includes these key phases:
    
    1. **Business Problem Framing**: Define the business problem and how ML can solve it
    2. **Data Collection & Integration**: Gather relevant data from various sources
    3. **Data Preparation & Analysis**: Clean, explore, and prepare data for modeling
    4. **Feature Engineering**: Create meaningful features from raw data
    5. **Model Training & Tuning**: Train models and optimize hyperparameters
    6. **Model Evaluation**: Assess model performance against business goals
    7. **Model Deployment**: Deploy models to production environments
    8. **Monitoring & Debugging**: Track model performance and handle issues
    """)
    
    # Show tip
    show_tip("Understanding the ML development lifecycle is critical. Remember that it's an iterative process - you may need to go back to earlier phases based on results from later phases.")
    
    # Scenarios
    scenarios = [
        {
            "id": 1,
            "scenario": "A data scientist is examining the distribution of values in each column, identifying outliers, and visualizing relationships between variables in a dataset.",
            "options": [
                "Data Collection & Integration", 
                "Data Preparation & Analysis", 
                "Feature Engineering", 
                "Model Training & Tuning"
            ],
            "answer": "Data Preparation & Analysis",
            "explanation": "This describes Data Preparation & Analysis, where data scientists explore and understand the data through statistical analysis and visualization before proceeding to modeling."
        },
        {
            "id": 2,
            "scenario": "A team is creating new variables by combining existing ones, encoding categorical variables, and normalizing numerical features to improve model performance.",
            "options": [
                "Data Preparation & Analysis", 
                "Feature Engineering", 
                "Model Training & Tuning", 
                "Model Evaluation"
            ],
            "answer": "Feature Engineering",
            "explanation": "This describes Feature Engineering, which involves transforming raw data into features that better represent the underlying patterns, including creating new variables, encoding, and normalization."
        },
        {
            "id": 3,
            "scenario": "After deploying a model, a team notices that its accuracy has decreased over time. They set up alerts for when performance metrics drop below certain thresholds.",
            "options": [
                "Model Evaluation", 
                "Model Deployment", 
                "Monitoring & Debugging", 
                "Business Problem Framing"
            ],
            "answer": "Monitoring & Debugging",
            "explanation": "This describes Monitoring & Debugging, which involves tracking model performance in production, detecting issues like model drift, and setting up alerts for when performance degrades."
        },
        {
            "id": 4,
            "scenario": "A team is working with stakeholders to define success metrics and determine whether an ML approach is appropriate for their customer churn problem.",
            "options": [
                "Business Problem Framing", 
                "Data Collection & Integration", 
                "Model Evaluation", 
                "Model Deployment"
            ],
            "answer": "Business Problem Framing",
            "explanation": "This describes Business Problem Framing, which involves defining the business problem, success metrics, and determining whether ML is appropriate for solving the problem."
        },
        {
            "id": 5,
            "scenario": "A data scientist is testing different algorithms, adjusting learning rates, and using cross-validation to find the optimal model configuration.",
            "options": [
                "Feature Engineering", 
                "Model Training & Tuning", 
                "Model Evaluation", 
                "Model Deployment"
            ],
            "answer": "Model Training & Tuning",
            "explanation": "This describes Model Training & Tuning, which involves selecting algorithms, adjusting hyperparameters, and using techniques like cross-validation to optimize model performance."
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
        
        # Radio button for selection
        user_answer = st.radio(
            "Which phase of the ML development lifecycle is being described?", 
            scenario["options"], 
            key=f"ml_process_{scenario['id']}", index=None
        )
        
        # Submit button
        submit_button = st.button("Submit Answer", key=f"ml_process_submit_{scenario['id']}")
        
        if submit_button:
            st.session_state.game7_submitted[i] = True
            
            if user_answer == scenario["answer"]:
                st.session_state.game7_score += 1
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
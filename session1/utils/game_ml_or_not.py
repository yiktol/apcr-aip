import streamlit as st
import random
from utils.common import display_progress, show_tip, reset_game_button

def ml_or_not_game():
    st.header("🎯 ML or Not?")
    st.write("Identify whether each scenario is a good candidate for Machine Learning or not")
    
    # Display progress
    display_progress(3)
    
    # Reset game button
    reset_game_button(3)
    
    # When to use ML
    st.subheader("When to Use Machine Learning:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Good candidates for ML:**
        1. Complex patterns difficult for humans to code
        2. Need to adapt/personalize at scale
        3. Can't code rules explicitly
        4. Have sufficient quality data available
        """)
    
    with col2:
        st.markdown("""
        **Not suitable for ML:**
        1. Simple logic can solve the problem
        2. Need 100% accuracy and explainability
        3. Insufficient or poor quality data
        4. Rules are well-defined and stable
        """)
    
    # Show tip
    show_tip("Machine Learning excels at finding patterns in complex data, but isn't always the best solution. Consider the problem complexity, data availability, and explainability requirements.")
    
    # Scenarios
    scenarios = [
        {
            "id": 1,
            "scenario": "A hospital wants to develop a system that can predict which patients are at high risk for readmission based on their medical history, demographics, and current symptoms.",
            "answer": "Good ML Candidate",
            "explanation": "This is a good ML candidate because it involves complex patterns across multiple variables that would be difficult to code with explicit rules, and historical data is available for training."
        },
        {
            "id": 2,
            "scenario": "A bank needs to calculate interest on loans based on predefined interest rates and loan terms according to regulatory requirements.",
            "answer": "Not an ML Candidate",
            "explanation": "This is not an ML candidate because it involves straightforward calculations with explicit, well-defined rules that can be coded directly."
        },
        {
            "id": 3,
            "scenario": "An e-commerce company wants to provide personalized product recommendations to millions of users based on their browsing history, purchases, and similar customers' behaviors.",
            "answer": "Good ML Candidate",
            "explanation": "This is a good ML candidate because it involves personalization at scale with complex patterns across user behaviors that would be impossible to code manually."
        },
        {
            "id": 4,
            "scenario": "A manufacturing company needs a system to ensure their products meet exact specifications with zero tolerance for error, where all quality parameters are clearly defined.",
            "answer": "Not an ML Candidate",
            "explanation": "This is not an ML candidate because it requires 100% accuracy with clearly defined parameters. Traditional quality control systems with explicit rules would be more appropriate."
        },
        {
            "id": 5,
            "scenario": "A security company wants to develop a system that can identify unusual patterns in network traffic that might indicate a cyber attack, even if the attack method hasn't been seen before.",
            "answer": "Good ML Candidate",
            "explanation": "This is a good ML candidate because it involves detecting complex, evolving patterns that can't be fully specified in advance, making it perfect for anomaly detection with ML."
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
            "Is this a good candidate for Machine Learning?", 
            ["Good ML Candidate", "Not an ML Candidate"], 
            key=f"ml_or_not_{scenario['id']}", index=None
        )
        
        # Submit button
        submit_button = st.button("Submit Answer", key=f"ml_not_submit_{scenario['id']}")
        
        if submit_button:
            st.session_state.game3_submitted[i] = True
            
            if user_answer == scenario["answer"]:
                st.session_state.game3_score += 1
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
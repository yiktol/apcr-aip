import streamlit as st
import random
from utils.common import display_progress, show_tip, reset_game_button

def traditional_vs_ml_game():
    st.header("💻 Traditional Programming vs Machine Learning")
    st.write("Identify whether each approach is Traditional Programming or Machine Learning")
    
    # Display progress
    display_progress(2)
    
    # Reset game button
    reset_game_button(2)
    
    # Visual comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Traditional Programming")
        st.image("https://miro.medium.com/max/1400/1*sXNXYfAqfLUeiDXPCo130w.png", 
                caption="Traditional Programming Flow")
        st.markdown("""
        - Human creates explicit rules
        - Input + Rules → Output
        - Rules are hand-coded
        - Limited ability to handle complexity
        """)
        
    with col2:
        st.subheader("Machine Learning")
        st.image("https://miro.medium.com/max/1400/1*Gf_rJdblU_5KjtPOaTpTUg.png", 
                caption="Machine Learning Flow")
        st.markdown("""
        - System learns rules from examples
        - Input + Output → Rules (training)
        - Rules are derived from data
        - Can handle complex patterns
        """)
    
    # Show tip
    show_tip("In traditional programming, humans define the rules. In ML, the system learns the rules from data.")
    
    # Scenarios
    scenarios = [
        {
            "id": 1,
            "scenario": "A developer writes code with specific IF-THEN statements to categorize emails based on keywords in the subject line.",
            "answer": "Traditional Programming",
            "explanation": "This is traditional programming because explicit rules (IF-THEN statements) are being coded by a human developer."
        },
        {
            "id": 2,
            "scenario": "A system analyzes thousands of past loan applications and their outcomes to create a model that can predict loan default risk for new applicants.",
            "answer": "Machine Learning",
            "explanation": "This is machine learning because the system is learning patterns from historical data (past loan applications) to make predictions on new data."
        },
        {
            "id": 3,
            "scenario": "A developer creates a function that calculates shipping costs based on package weight, dimensions, and delivery distance using a predefined formula.",
            "answer": "Traditional Programming",
            "explanation": "This is traditional programming as it uses a predefined formula with explicit rules coded by developers."
        },
        {
            "id": 4,
            "scenario": "A system is shown millions of images labeled 'cat' or 'not cat' and develops the ability to recognize cats in new, unseen images.",
            "answer": "Machine Learning",
            "explanation": "This is machine learning, specifically supervised learning, where the system learns patterns from labeled examples to classify new inputs."
        },
        {
            "id": 5,
            "scenario": "A developer creates a tax calculation system that applies different tax rates based on income brackets defined in the tax code.",
            "answer": "Traditional Programming",
            "explanation": "This is traditional programming because it applies explicit rules (tax rates and brackets) defined in the code."
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
            "Is this Traditional Programming or Machine Learning?", 
            ["Traditional Programming", "Machine Learning"], 
            key=f"trad_vs_ml_{scenario['id']}", index=None
        )
        
        # Submit button
        submit_button = st.button("Submit Answer", key=f"trad_ml_submit_{scenario['id']}")
        
        if submit_button:
            st.session_state.game2_submitted[i] = True
            
            if user_answer == scenario["answer"]:
                st.session_state.game2_score += 1
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
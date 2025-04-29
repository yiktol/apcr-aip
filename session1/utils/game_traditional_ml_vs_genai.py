import streamlit as st
import random
from utils.common import display_progress, show_tip, reset_game_button

def traditional_ml_vs_genai_game():
    st.header("⚔️ Traditional ML vs Generative AI")
    st.write("Determine whether each scenario is best suited for Traditional Machine Learning or Generative AI")
    
    # Display progress
    display_progress(4)
    
    # Reset game button
    reset_game_button(4)
    
    # Visualization of differences
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Traditional ML")
        st.image("https://miro.medium.com/max/1200/1*c_fiB-YgbnMl6nntYGBMHQ.jpeg", caption="Traditional ML Workflow")
        st.markdown("""
        **Best for:**
        - Structured data analysis
        - Clear classification tasks
        - Regression problems
        - Anomaly detection
        - When interpretability is crucial
        """)
    
    with col2:
        st.subheader("Generative AI")
        st.image("https://miro.medium.com/max/1400/1*U_YpRawOekhqrxj3UFvdIw.jpeg", caption="Generative AI Workflow")
        st.markdown("""
        **Best for:**
        - Content creation
        - Natural language understanding
        - Open-ended reasoning
        - Creative tasks
        - Complex, unstructured data
        """)
    
    # Show tip
    show_tip("Traditional ML is typically better for well-defined problems with structured data where interpretability is important. Generative AI excels at creative, unstructured tasks where generating new content is the goal.")
    
    # Scenarios
    scenarios = [
        {
            "id": 1,
            "scenario": "A financial institution needs to detect fraudulent credit card transactions with clear audit trails showing why each transaction was flagged as suspicious.",
            "answer": "Traditional ML",
            "explanation": "This requires Traditional ML because interpretability and explainability are crucial for financial fraud detection. The system needs to provide clear reasons why a transaction was flagged as fraudulent."
        },
        {
            "id": 2,
            "scenario": "A marketing team needs a tool that can generate personalized email content for different customer segments, adapting tone and style while maintaining brand consistency.",
            "answer": "Generative AI",
            "explanation": "This is ideal for Generative AI because it involves creating new, original content (email copy) tailored to different audiences, which requires understanding context and generating human-like text."
        },
        {
            "id": 3,
            "scenario": "A manufacturing company wants to predict equipment failures before they happen based on sensor data collected from machinery.",
            "answer": "Traditional ML",
            "explanation": "This is best for Traditional ML as it's a classic predictive maintenance problem using structured time-series data, where clear patterns in sensor readings can indicate potential failures."
        },
        {
            "id": 4,
            "scenario": "A software company needs a tool that can automatically generate code snippets based on natural language descriptions of what the code should do.",
            "answer": "Generative AI",
            "explanation": "This requires Generative AI because it involves creating new content (code) based on natural language understanding, requiring the ability to translate concepts into structured code."
        },
        {
            "id": 5,
            "scenario": "A healthcare provider needs to classify medical images to detect signs of specific diseases with high accuracy and provide confidence scores.",
            "answer": "Traditional ML",
            "explanation": "This is best suited for Traditional ML because medical image classification requires high precision, clear confidence metrics, and interpretability for healthcare professionals to trust and use the results."
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
            "Which approach is better suited for this scenario?", 
            ["Traditional ML", "Generative AI"], 
            key=f"trad_vs_gen_{scenario['id']}", index=None
        )
        
        # Submit button
        submit_button = st.button("Submit Answer", key=f"trad_gen_submit_{scenario['id']}")
        
        if submit_button:
            st.session_state.game4_submitted[i] = True
            
            if user_answer == scenario["answer"]:
                st.session_state.game4_score += 1
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
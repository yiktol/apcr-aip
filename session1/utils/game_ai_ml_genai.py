import streamlit as st
import random
from utils.common import display_progress, show_tip, reset_game_button

def ai_ml_genai_game():
    st.header("🤖 AI, ML, or Generative AI?")
    st.write("Determine whether each scenario describes Artificial Intelligence, Machine Learning, or Generative AI")
    
    # Scenarios and answers
    scenarios = [
        {
            "id": 1,
            "scenario": "A system that can understand and respond to natural language questions about company data, providing insights without having been explicitly programmed for each question type.",
            "answer": "Artificial Intelligence",
            "explanation": "This describes AI capabilities to understand and respond to natural language, which is a broader category encompassing various technologies including NLP.",
            "image": "https://d1.awsstatic.com/r2018/h/99Artificial%20Intelligence.a184be3121af4fa750e0b06c5990b513ecb8e3c9.png"
        },
        {
            "id": 2,
            "scenario": "A system that analyzes historical sales data to identify patterns and automatically predicts future sales volumes for inventory management.",
            "answer": "Machine Learning",
            "explanation": "This is ML because it uses historical data to learn patterns and make predictions on new data.",
            "image": "https://d1.awsstatic.com/product-marketing/ML/machine_learning_helps.e4ca5a70c5656b6780f0603924ddd3d0ea67a9c6.png"
        },
        {
            "id": 3,
            "scenario": "A system that can create new, original artwork in the style of Renaissance painters after being trained on thousands of classic paintings.",
            "answer": "Generative AI",
            "explanation": "This describes Generative AI's capability to create new content (artwork) based on patterns learned from training data.",
            "image": "https://d1.awsstatic.com/generative-ai-images/Benefits_Innovate%20and%20build%20faster.82763f46ea8139b03b7ad91123d64548bd24b244.png"
        },
        {
            "id": 4,
            "scenario": "A customer service chatbot that can have realistic conversations with users, answering questions and creating custom responses that weren't explicitly programmed.",
            "answer": "Generative AI",
            "explanation": "This is Generative AI because it's creating new, original text content (responses) that weren't pre-programmed based on the input it receives.",
            "image": "https://d1.awsstatic.com/generative-ai-images/Use%20Cases_GenAI%20Website_1.b8e7fc0179e77a8d2581043b05d9df2f0c0e3b56.jpg"
        },
        {
            "id": 5,
            "scenario": "A system that automatically categorizes customer support tickets into different departments based on the content of the ticket.",
            "answer": "Machine Learning",
            "explanation": "This is ML specifically - it's a classification task that learns from labeled examples to categorize new inputs.",
            "image": "https://d1.awsstatic.com/re19/FargateonEKS/product-page-diagram_Fargate@2x.a04a636ef7364456072619883059329c957cca40.png"
        }
    ]
    
    # Display progress
    display_progress(1)
    
    # Reset game button
    reset_game_button(1)
    
    # Visualize the relationship between AI, ML, and Generative AI
    st.subheader("Understanding the Relationship:")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*n-KSTngf-Pg00r_Y9q52Aw.jpeg", 
                caption="AI, ML, and Generative AI Relationship")
    
    with col2:
        st.markdown("""
        - **Artificial Intelligence (AI)** is the broadest category - systems that can simulate human intelligence
        - **Machine Learning (ML)** is a subset of AI that learns patterns from data
        - **Generative AI** is a subset of ML that can create new, original content
        """)
    
    # Show tip
    show_tip("Remember that Generative AI creates new content, ML identifies patterns in data to make predictions, and AI is the broadest category.")
    st.divider()
    # Game scenarios
    for i, scenario in enumerate(scenarios):
        st.markdown(f"""
        <div class="game-card">
            <h3>Scenario {scenario['id']}</h3>
        """, unsafe_allow_html=True)
        
        # Display scenario image
        # st.image(scenario["image"], width=400)
        
        st.write(scenario["scenario"])
        
        # Radio button for selection
        options = ["Artificial Intelligence", "Machine Learning", "Generative AI"]
        user_answer = st.radio("What type of technology is this?", options, key=f"scenario_{scenario['id']}", index=None)
        
        # Submit button
        submit_button = st.button("Submit Answer", key=f"submit_{scenario['id']}")
        
        if submit_button:
            st.session_state.game1_submitted[i] = True
            
            if user_answer == scenario["answer"]:
                st.session_state.game1_score += 1
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
import streamlit as st
import random
from utils.common import display_progress, show_tip, reset_game_button

def learning_types_game():
    st.header("🧠 ML Learning Types")
    st.write("Match each scenario to the correct type of machine learning")
    
    # Display progress
    display_progress(6)
    
    # Reset game button
    reset_game_button(6)
    
    # Visual overview of learning types
    st.subheader("Types of Machine Learning")
    
    st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*xz5baytHfiB2Ex8iclS4XA.png", caption="Machine Learning Types", width=800)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Supervised Learning")
        # st.image("https://miro.medium.com/max/1400/1*JUliPcZRC_qppP5XqXQCHw.png", caption="Supervised Learning")
        st.markdown("""
        - Learning from labeled data
        - Predicting outcomes
        - Examples: Classification, Regression
        """)
    
    with col2:
        st.markdown("### Unsupervised Learning")
        # st.image("https://miro.medium.com/max/1400/1*c2soUQxTXGNdrHJQMNPJhQ.png", caption="Unsupervised Learning")
        st.markdown("""
        - Finding patterns in unlabeled data
        - No predefined outcomes
        - Examples: Clustering, Dimensionality Reduction
        """)
    
    with col3:
        st.markdown("### Self-Supervised Learning")
        # st.image("https://miro.medium.com/max/1400/1*ySAWadPotWFlDFYPYywVgw.png", caption="Self-Supervised Learning")
        st.markdown("""
        - Creates own supervision from data
        - Predicts parts of data from other parts
        - Used in Generative AI & Foundation Models
        """)
    
    st.markdown("### Reinforcement Learning")
    st.image("https://miro.medium.com/max/1400/1*HvoLc50Dpq1ESKuejhICHg.png", caption="Reinforcement Learning")
    st.markdown("""
    - Learning through trial and error
    - Rewarded for correct actions
    - Examples: Game playing, Robotics
    """)
    
    # Show tip
    show_tip("The learning type determines how a model learns from data. Supervised needs labeled data, unsupervised finds patterns in unlabeled data, self-supervised creates its own supervision signal, and reinforcement learning learns through rewards and penalties.")
    
    # Scenarios
    scenarios = [
        {
            "id": 1,
            "scenario": "A chatbot is trained on a massive dataset of text from the internet, learning to predict the next word in a sequence without explicit labels for each prediction.",
            "answer": "Self-Supervised Learning",
            "explanation": "This is Self-Supervised Learning because the system creates its own supervision signal by masking words and predicting them based on context, which is how large language models for generative AI are typically trained."
        },
        {
            "id": 2,
            "scenario": "A recommendation system groups customers into segments based on their purchasing behavior without any predefined categories.",
            "answer": "Unsupervised Learning",
            "explanation": "This is Unsupervised Learning because it's finding patterns (customer segments) in data without labeled examples of what those segments should be."
        },
        {
            "id": 3,
            "scenario": "A model learns to play chess by playing thousands of games against itself, receiving a positive reward when it wins and a negative reward when it loses.",
            "answer": "Reinforcement Learning",
            "explanation": "This is Reinforcement Learning because the system learns through trial and error with rewards (winning) and penalties (losing), improving its strategy over time."
        },
        {
            "id": 4,
            "scenario": "A spam filter is trained on thousands of emails that have been manually labeled as 'spam' or 'not spam'.",
            "answer": "Supervised Learning",
            "explanation": "This is Supervised Learning because the model is trained on labeled examples (emails already classified as spam or not spam) to predict the classification of new emails."
        },
        {
            "id": 5,
            "scenario": "A system analyzes customer reviews to identify common themes and topics without being told in advance what themes to look for.",
            "answer": "Unsupervised Learning",
            "explanation": "This is Unsupervised Learning because it's discovering patterns (themes and topics) in data without labeled examples of what those themes should be."
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
            "What type of machine learning is being described?", 
            ["Supervised Learning", "Unsupervised Learning", "Self-Supervised Learning", "Reinforcement Learning"], 
            key=f"learning_type_{scenario['id']}", index=None
        )
        
        # Submit button
        submit_button = st.button("Submit Answer", key=f"learning_type_submit_{scenario['id']}")
        
        if submit_button:
            st.session_state.game6_submitted[i] = True
            
            if user_answer == scenario["answer"]:
                st.session_state.game6_score += 1
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
import streamlit as st
import random
from utils.common import display_progress, show_tip, reset_game_button

def aws_services_game():
    st.header("☁️ AWS AI Services")
    st.write("Match each use case to the most appropriate AWS AI service")
    
    # Display progress
    display_progress(8)
    
    # Reset game button
    reset_game_button(8)
    
    # AWS AI services overview
    st.subheader("AWS AI Services Overview")
    
    # Show AWS AI stack
    st.image("https://d1.awsstatic.com/diagrams/ai-ml-stack-v4.81d23a600ab3d467395f988358118dc38ca1b5c4.png", caption="AWS AI/ML Stack")
    
    # Show tip
    show_tip("AWS offers AI services at different levels of abstraction. AI Services (top layer) require no ML expertise, SageMaker (middle layer) simplifies ML workflow, and ML Frameworks (bottom layer) provide maximum flexibility but require more expertise.")
    
    # AWS AI Services Table
    services_data = {
        "Service": ["Amazon Rekognition", "Amazon Textract", "Amazon Comprehend", "Amazon Kendra", "Amazon Personalize", "Amazon SageMaker", "Amazon Fraud Detector", "Amazon Bedrock"],
        "Purpose": [
            "Image and video analysis",
            "Document text extraction",
            "Natural language processing",
            "Intelligent search",
            "Personalized recommendations",
            "Build, train and deploy ML models",
            "Fraud detection",
            "Foundation models for generative AI"
        ]
    }
    
    st.table(services_data)
    
    # Scenarios
    scenarios = [
        {
            "id": 1,
            "scenario": "A retail company wants to automatically moderate user-submitted product images to ensure they don't contain inappropriate content before publishing them on their website.",
            "options": ["Amazon Rekognition", "Amazon Textract", "Amazon Comprehend", "Amazon SageMaker"],
            "answer": "Amazon Rekognition",
            "explanation": "Amazon Rekognition is the appropriate choice as it provides image analysis capabilities including content moderation to detect inappropriate content in images."
        },
        {
            "id": 2,
            "scenario": "A financial institution needs to process thousands of loan applications daily, extracting key information from scanned documents including forms, tables, and handwritten notes.",
            "options": ["Amazon Textract", "Amazon Comprehend", "Amazon Kendra", "Amazon Personalize"],
            "answer": "Amazon Textract",
            "explanation": "Amazon Textract is designed specifically for document processing and can extract text, forms, tables, and even handwriting from scanned documents."
        },
        {
            "id": 3,
            "scenario": "An e-commerce company wants to provide personalized product recommendations to customers based on their browsing history and purchase behavior.",
            "options": ["Amazon Personalize", "Amazon Kendra", "Amazon Fraud Detector", "Amazon Comprehend"],
            "answer": "Amazon Personalize",
            "explanation": "Amazon Personalize is specifically designed for building recommendation systems that provide personalized product recommendations based on user behavior."
        },
        {
            "id": 4,
            "scenario": "A customer service team wants to analyze thousands of support tickets to identify common themes, sentiment, and key entities mentioned by customers.",
            "options": ["Amazon Comprehend", "Amazon Rekognition", "Amazon Textract", "Amazon Fraud Detector"],
            "answer": "Amazon Comprehend",
            "explanation": "Amazon Comprehend provides natural language processing capabilities to analyze text for sentiment, key phrases, entities, and themes, making it perfect for analyzing customer support tickets."
        },
        {
            "id": 5,
            "scenario": "A company is developing an advanced AI chatbot that needs to generate human-like responses, understand complex queries, and maintain context across conversations.",
            "options": ["Amazon Bedrock", "Amazon SageMaker", "Amazon Kendra", "Amazon Comprehend"],
            "answer": "Amazon Bedrock",
            "explanation": "Amazon Bedrock provides access to foundation models for generative AI applications like chatbots that need to generate human-like responses and understand complex queries with context."
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
            "Which AWS AI service would be most appropriate for this use case?", 
            scenario["options"], 
            key=f"aws_services_{scenario['id']}",
            index=None
        )
        
        # Submit button
        submit_button = st.button("Submit Answer", key=f"aws_services_submit_{scenario['id']}")
        
        if submit_button:
            st.session_state.game8_submitted[i] = True
            
            if user_answer == scenario["answer"]:
                st.session_state.game8_score += 1
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
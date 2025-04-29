import streamlit as st

def initialize_knowledge_check():
    if 'knowledge_check_answered' not in st.session_state:
        st.session_state.knowledge_check_answered = [False] * 5
        st.session_state.knowledge_check_correct = [False] * 5
        st.session_state.knowledge_check_progress = 0

def reset_knowledge_check():
    st.session_state.knowledge_check_answered = [False] * 5
    st.session_state.knowledge_check_correct = [False] * 5
    st.session_state.knowledge_check_progress = 0

def display_knowledge_check():
    initialize_knowledge_check()
    
    # Progress tracking
    progress = st.progress(st.session_state.knowledge_check_progress / 5)
    st.write(f"Progress: {st.session_state.knowledge_check_progress}/5 questions answered")
    
    st.write("""
    Test your understanding of AI and ML concepts with these knowledge check questions.
    Select your answers and click 'Submit' for each question to check your understanding.
    """)
    
    # Question 1
    st.subheader("Question 1: Machine Learning Types")
    st.write("""
    A company wants to develop a machine learning model to analyze customer feedback on their products. 
    They have a dataset of 10,000 customer reviews, with each review labeled as either "positive" or "negative". 
    The company will train the machine learning model to predict whether new customer reviews are "positive" or "negative".
    
    Which of the following machine learning types does this describe?
    """)
    
    q1_options = [
        "Supervised Learning",
        "Unsupervised Learning",
        "Self-Supervised Learning",
        "Reinforcement Learning"
    ]
    
    q1_answer = st.radio(
        "Select one option:",
        q1_options,
        key="q1",
        index=None  # No default selection
    )
    
    q1_submitted = st.button("Submit Answer", key="submit_q1")
    
    if q1_submitted and q1_answer:
        st.session_state.knowledge_check_answered[0] = True
        if q1_answer == "Supervised Learning":
            st.session_state.knowledge_check_correct[0] = True
            st.success("Correct! This is a supervised learning problem where you have to choose between two labels (binary classification). The dataset provided is labeled as positive and negative.")
        else:
            st.error("Not quite right. This is a supervised learning problem (binary classification) because the dataset includes labeled examples of positive and negative reviews.")
        
        if not st.session_state.knowledge_check_answered[0]:
            st.session_state.knowledge_check_progress += 1
    
    st.markdown("---")
    
    # Question 2
    st.subheader("Question 2: AI Service Selection")
    st.write("""
    An e-commerce company needs to integrate tailored recommendations to customers based on their browsing and purchase history. 
    The company needs to quickly implement this solution but their team has limited machine learning expertise.
    
    Which of the following solutions would meet the requirements in the most operationally efficient way?
    """)
    
    q2_options = [
        "Set up an EC2 cluster and train the model on the cluster. Use Amazon S3 to store your training data and model artifacts, and deploy the model on an EC2 instance behind an Elastic Load Balancer.",
        "Add the data to Amazon Personalize and then access real time recommendations via the personalization API.",
        "Use Amazon SageMaker AI to create a machine learning pipeline that includes data preparation, model training, and model deployment steps.",
        "Use Amazon Bedrock to generate personalized recommendations that would be shown to customers on the website."
    ]
    
    q2_answer = st.radio(
        "Select one option:",
        q2_options,
        key="q2",
        index=None  # No default selection
    )
    
    q2_submitted = st.button("Submit Answer", key="submit_q2")
    
    if q2_submitted and q2_answer:
        st.session_state.knowledge_check_answered[1] = True
        if q2_answer == "Add the data to Amazon Personalize and then access real time recommendations via the personalization API.":
            st.session_state.knowledge_check_correct[1] = True
            st.success("Correct! Amazon Personalize allows you to quickly implement a customized personalization engine in days without ML expertise required.")
        else:
            st.error("Not quite right. Amazon Personalize is the most efficient solution as it's specifically designed for personalized recommendations and doesn't require ML expertise.")
        
        if not st.session_state.knowledge_check_answered[1]:
            st.session_state.knowledge_check_progress += 1
    
    st.markdown("---")
    
    # Question 3
    st.subheader("Question 3: ML Pipeline")
    st.write("""
    A machine learning engineer wants to implement a machine learning pipeline on AWS to automate the process of 
    training and deploying models. The pipeline should include data preprocessing, model training, and model deployment.
    
    Which AWS service would the machine learning engineer use to orchestrate this machine learning pipeline?
    """)
    
    q3_options = [
        "Amazon Rekognition",
        "Amazon Bedrock",
        "Amazon Q",
        "Amazon SageMaker AI"
    ]
    
    q3_answer = st.radio(
        "Select one option:",
        q3_options,
        key="q3",
        index=None  # No default selection
    )
    
    q3_submitted = st.button("Submit Answer", key="submit_q3")
    
    if q3_submitted and q3_answer:
        st.session_state.knowledge_check_answered[2] = True
        if q3_answer == "Amazon SageMaker AI":
            st.session_state.knowledge_check_correct[2] = True
            st.success("Correct! Amazon SageMaker AI allows you to build, train, and deploy machine learning models for any use case with fully managed infrastructure, tools, and workflows.")
        else:
            st.error("Not quite right. Amazon SageMaker AI is the service designed for building ML pipelines that include data preprocessing, model training, and model deployment.")
        
        if not st.session_state.knowledge_check_answered[2]:
            st.session_state.knowledge_check_progress += 1
    
    st.markdown("---")
    
    # Question 4
    st.subheader("Question 4: Traditional Programming vs Machine Learning")
    st.write("""
    Which statement correctly describes the fundamental difference between traditional programming and machine learning?
    """)
    
    q4_options = [
        "Traditional programming uses complex algorithms while machine learning uses simple algorithms.",
        "Traditional programming derives output based on inputs and rules, while machine learning derives rules from inputs and outputs.",
        "Traditional programming is manual, while machine learning is entirely automated with no human intervention.",
        "Traditional programming works with small datasets, while machine learning only works with big data."
    ]
    
    q4_answer = st.radio(
        "Select one option:",
        q4_options,
        key="q4",
        index=None  # No default selection
    )
    
    q4_submitted = st.button("Submit Answer", key="submit_q4")
    
    if q4_submitted and q4_answer:
        st.session_state.knowledge_check_answered[3] = True
        if q4_answer == "Traditional programming derives output based on inputs and rules, while machine learning derives rules from inputs and outputs.":
            st.session_state.knowledge_check_correct[3] = True
            st.success("Correct! This captures the fundamental difference: traditional programming explicitly defines rules to process inputs, while ML learns the rules from examples of inputs and desired outputs.")
        else:
            st.error("Not quite right. The key difference is that traditional programming requires explicit rules coded by humans, while ML learns the rules from data.")
        
        if not st.session_state.knowledge_check_answered[3]:
            st.session_state.knowledge_check_progress += 1
    
    st.markdown("---")
    
    # Question 5
    st.subheader("Question 5: ML Use Cases")
    st.write("""
    Which of the following scenarios would be BEST suited for a traditional machine learning approach rather than generative AI? (Select all that apply)
    """)
    
    q5_options = {
        "Generating creative marketing content for different customer segments": False,
        "Detecting fraudulent transactions in banking systems": True,
        "Predicting equipment failures based on sensor data": True,
        "Creating conversational customer support chatbots": False,
        "Classifying customer support tickets by priority and department": True
    }
    
    q5_answers = {}
    for option in q5_options.keys():
        q5_answers[option] = st.checkbox(option, key=f"q5_{option}")
    
    q5_submitted = st.button("Submit Answer", key="submit_q5")
    
    if q5_submitted and any(q5_answers.values()):
        st.session_state.knowledge_check_answered[4] = True
        
        # Check if user selected correct options
        correct = True
        for option, is_correct in q5_options.items():
            if q5_answers[option] != is_correct:
                correct = False
                break
        
        st.session_state.knowledge_check_correct[4] = correct
        
        if correct:
            st.success("""
            Correct! The best candidates for traditional ML are:
            - Detecting fraudulent transactions (classification with explainability requirements)
            - Predicting equipment failures (time series prediction with structured data)
            - Classifying customer tickets (multi-class classification with clear categories)
            
            Generative AI would be better for content generation and conversational tasks.
            """)
        else:
            st.error("""
            Not quite right. Traditional ML is best for:
            - Detecting fraudulent transactions (classification with explainability requirements)
            - Predicting equipment failures (time series prediction with structured data)
            - Classifying customer tickets (multi-class classification with clear categories)
            
            Generative AI would be better for content generation and conversational tasks.
            """)
        
        if not st.session_state.knowledge_check_answered[4]:
            st.session_state.knowledge_check_progress += 1
    
    # Update progress
    total_answered = sum(st.session_state.knowledge_check_answered)
    st.session_state.knowledge_check_progress = total_answered
    
    # Overall score
    if total_answered == 5:
        total_correct = sum(st.session_state.knowledge_check_correct)
        st.subheader(f"Your Score: {total_correct}/5")
        
        if total_correct == 5:
            st.balloons()
            st.success("Perfect score! You have a strong understanding of AI/ML fundamentals.")
        elif total_correct >= 3:
            st.success("Good job! You have a solid understanding of AI/ML fundamentals.")
        else:
            st.info("You might want to review some of the topics to strengthen your understanding.")
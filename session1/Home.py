import streamlit as st
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from utils.knowledge_check import display_knowledge_check, reset_knowledge_check
from utils.styles import load_css, custom_header
from utils.common import render_sidebar

# Set page configuration
st.set_page_config(
    page_title="AWS AI Practitioner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
load_css()

# Session Management
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar
with st.sidebar:
    render_sidebar()
    
    # About this App (collapsible)
    with st.expander("ℹ️ About this App", expanded=False):
        st.write("""
        This interactive e-learning app helps you prepare for the AWS AI Practitioner certification.
        
        **Topics covered:**
        - AI, ML and Generative AI concepts
        - Traditional vs ML approaches
        - Machine learning types
        - Common use cases
        - ML process overview
        - AWS AI/ML stack
        - Knowledge check
        """)

# Main Content - Tab-based Navigation
tab_home, tab_ai_ml, tab_prog_ml, tab_when_ml, tab_ml_genai, tab_terms, tab_ml_types, tab_use_cases, tab_process, tab_stack, tab_knowledge = st.tabs([
    "🏠 Home", 
    "🤖 AI vs ML", 
    "💻 Programming vs ML", 
    "🕒 When to use ML", 
    "🔄 ML vs GenAI",
    "📚 Terminology", 
    "🧩 ML Types", 
    "📋 Use Cases", 
    "🔄 ML Process", 
    "🏗️ AWS Stack", 
    "❓ Knowledge Check"
])

# Home - Program Overview
with tab_home:
    st.markdown(custom_header("AWS AI Practitioner Certification Readiness", 1 ), unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Program Overview")
        st.write("""
        Welcome to the AWS AI Practitioner certification preparation program! 
        
        This program will help you understand the fundamentals of Artificial Intelligence, 
        Machine Learning, and Generative AI concepts on AWS.
        
        **Learning Outcomes:**
        - Understand the difference between AI, ML, and Generative AI
        - Learn when to use machine learning vs traditional programming
        - Explore different ML types and common use cases
        - Master the ML development lifecycle
        - Familiarize yourself with the AWS AI/ML stack
        """)
        
        st.subheader("Program Structure")
        program_data = {
            "Session": ["Session 1", "Session 2", "Session 3", "Session 4"],
            "Content": [
                "Kickoff & Fundamentals of AI and ML",
                "Fundamentals of Generative AI",
                "Applications of Foundation Models",
                "Responsible AI & Security, Compliance, and Governance"
            ]
        }
        st.table(pd.DataFrame(program_data))
    
    with col2:
        st.image("assets/images/AWS-Certified-AI-Practitioner_badge.png", width=300)
        st.caption("AWS Certified AI Practitioner ")
        
        progress = st.progress(0)
        st.info("Navigate through the tabs at the top to explore all topics!")

# Difference between AI, ML, and generative AI
with tab_ai_ml:
    st.title("Difference between AI, ML, and Generative AI")
    
    # st.image("assets/ai_ml_genai.png", use_column_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Artificial Intelligence (AI)")
        st.write("""
        AI is the broader concept of machines being able to perform tasks in a way 
        that we would consider "smart" - simulating human intelligence.

        **Key Aspects:**
        - Goal is to create systems that can function intelligently and independently
        - Encompasses many approaches including rule-based systems and ML
        - Can involve reasoning, knowledge representation, planning, and more
        """)
        
        if st.button("Learn more about AI", key="ai_button"):
            st.markdown("[AWS: What is Artificial Intelligence?](https://aws.amazon.com/what-is/artificial-intelligence/)")
        
    with col2:
        st.subheader("Machine Learning (ML)")
        st.write("""
        ML is a subset of AI that focuses on algorithms that can learn from data
        without being explicitly programmed.

        **Key Aspects:**
        - Learns patterns from historical data
        - Makes predictions or decisions on new data
        - Improves with experience (more data)
        - Includes traditional ML and deep learning
        """)
        
        if st.button("Learn more about ML", key="ml_button"):
            st.markdown("[AWS: What is Machine Learning?](https://aws.amazon.com/what-is/machine-learning/)")
            
    with col3:
        st.subheader("Generative AI")
        st.write("""
        Generative AI is a subset of ML focused on creating new content
        like text, images, audio, or code.

        **Key Aspects:**
        - Uses foundation models pre-trained on vast amounts of data
        - Capable of generating novel content
        - Built on deep learning architectures
        - Examples include large language models (LLMs)
        """)
        
        if st.button("Learn more about Generative AI", key="genai_button"):
            st.markdown("[AWS: What is Generative AI?](https://aws.amazon.com/what-is/generative-ai/)")
    
    st.subheader("Deep Learning")
    st.write("""
    Deep Learning is a specialized subset of machine learning that uses artificial neural networks 
    with many layers (hence "deep"). It excels at handling unstructured data like images, audio, and text.
    
    Deep learning is crucial for generative AI as it powers the foundation models that generate new content.
    """)
    
    if st.button("Learn more about Deep Learning"):
        st.markdown("[AWS: What is Deep Learning?](https://aws.amazon.com/what-is/deep-learning/)")
    
    # Interactive example
    st.subheader("Interactive Example: AI, ML, or Generative AI?")
    
    examples = {
        "Speech recognition in a virtual assistant": "AI (specifically ML)",
        "Creating a novel image of a cat in space": "Generative AI",
        "A rule-based expert system": "AI (but not ML)",
        "Predicting customer churn": "ML",
        "Writing a poem in Shakespeare's style": "Generative AI",
        "Classifying emails as spam": "ML"
    }
    
    example = st.selectbox("Select an example:", list(examples.keys()))
    
    if st.button("Check Answer", key="ai_ml_check"):
        st.success(f"Answer: {examples[example]}")

# Traditional programming vs machine learning
with tab_prog_ml:
    st.title("Traditional Programming vs Machine Learning")
    
    # st.image("assets/ml_vs_programming.png", use_column_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Traditional Programming")
        st.write("""
        In traditional programming, developers write explicit rules that tell the computer how to solve a problem.

        **Characteristics:**
        - **Input + Rules = Output**
        - Human programmer defines all rules
        - Rules are coded explicitly
        - Good for well-defined problems
        - Limited by the programmer's ability to define rules
        """)
        
        st.code("""
        # Example: Traditional programming for recommendation
        def recommend_product(customer_data):
            if customer_data['age'] < 30 and customer_data['gender'] == 'M':
                return 'Video Games'
            elif customer_data['purchase_history'].contains('Books'):
                return 'More Books'
            elif customer_data['age'] > 60:
                return 'Health Products'
            else:
                return 'General Merchandise'
        """, language="python")
    
    with col2:
        st.subheader("Machine Learning")
        st.write("""
        In machine learning, algorithms learn patterns from data to derive rules automatically.

        **Characteristics:**
        - **Input + Output = Rules**
        - Algorithm learns from historical data
        - Rules are inferred, not explicitly coded
        - Can handle complex patterns
        - Performance improves with more data
        """)
        
        st.code("""
        # Example: ML approach for recommendation
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        
        # Load historical data
        data = pd.read_csv('customer_purchases.csv')
        
        # Features and labels
        X = data[['age', 'gender', 'purchase_history']]
        y = data['purchased_product']
        
        # Train the model
        model = RandomForestClassifier()
        model.fit(X, y)
        
        # Predict for new customer
        model.predict(new_customer_data)
        """, language="python")
    
    # Interactive example
    st.subheader("Interactive Example: Traditional Programming vs ML")
    
    st.write("Let's look at how the same problem can be approached differently:")
    
    problem = st.selectbox("Select a problem:", [
        "Spam email detection",
        "Product recommendation",
        "Credit approval",
        "Voice recognition"
    ])
    
    approach = st.radio("Choose an approach:", ["Traditional Programming", "Machine Learning"])
    
    examples = {
        "Spam email detection": {
            "Traditional Programming": "Create rules like: If email contains 'viagra' or 'lottery' AND sender is unknown, then classify as spam.",
            "Machine Learning": "Train a model on thousands of emails labeled as 'spam' or 'not spam', allowing it to learn patterns that indicate spam."
        },
        "Product recommendation": {
            "Traditional Programming": "Create rules like: If user bought a phone, recommend phone cases and screen protectors.",
            "Machine Learning": "Learn from purchasing patterns of similar users to recommend products, adapting to changing trends automatically."
        },
        "Credit approval": {
            "Traditional Programming": "If income > X AND debt ratio < Y AND credit history > Z years, then approve credit.",
            "Machine Learning": "Train on historical applications and outcomes to identify complex patterns that predict successful repayment."
        },
        "Voice recognition": {
            "Traditional Programming": "Very difficult with rules. Would require enormous rule sets trying to match sound patterns.",
            "Machine Learning": "Train on thousands of voice samples to recognize patterns in speech, adapting to different accents and speech patterns."
        }
    }
    
    st.write(examples[problem][approach])
    
    if approach == "Machine Learning":
        st.success("ML is especially powerful for this type of problem!")
    else:
        if problem in ["Voice recognition", "Product recommendation"]:
            st.warning("This problem is challenging with traditional programming and better suited for ML!")
        else:
            st.info("This approach can work but may not scale well with complexity.")

# When to use machine learning
with tab_when_ml:
    st.title("When to Use Machine Learning")
    
    st.write("""
    Machine learning is powerful but not always necessary. Here are key scenarios where ML provides significant advantages:
    """)
    
    scenarios = {
        "When you can't code it": {
            "description": "Tasks too complex for deterministic solutions",
            "examples": ["Image recognition", "Speech recognition", "Natural language understanding"],
            "icon": "🧩"
        },
        "When you can't scale it": {
            "description": "Repetitive tasks requiring human-like expertise but at massive scale",
            "examples": ["Content recommendation", "Spam detection", "Fraud detection", "Machine translation"],
            "icon": "📈"
        },
        "When you need to adapt/personalize": {
            "description": "Systems that need to adapt to individual preferences or changing conditions",
            "examples": ["Personalized recommendations", "Adaptive user interfaces", "Dynamic pricing"],
            "icon": "🔄"
        },
        "When you can't track it": {
            "description": "Problems where the environment changes too rapidly for manual updates",
            "examples": ["Autonomous driving", "Real-time bidding systems", "Dynamic resource allocation"],
            "icon": "⚡"
        }
    }
    
    # Display each scenario in expandable sections
    for scenario, details in scenarios.items():
        with st.expander(f"{details['icon']} {scenario}"):
            st.write(details["description"])
            st.write("**Examples:**")
            for example in details["examples"]:
                st.write(f"- {example}")
    
    # Decision tree visualization
    st.subheader("Decision Flow: When to Use ML")
    
    decision_tree = """
    graph TD
        A[Problem] --> B{Can you easily code rules?}
        B -->|Yes| C{Will rules scale?}
        B -->|No| D[Use Machine Learning]
        C -->|Yes| E[Traditional Programming]
        C -->|No| F{Need personalization?}
        F -->|Yes| G[Use Machine Learning]
        F -->|No| H{Environment changes rapidly?}
        H -->|Yes| I[Use Machine Learning]
        H -->|No| J[Traditional Programming]
    """
    
    st.graphviz_chart(decision_tree)
    
    # Interactive example
    st.subheader("Interactive Example: ML or Not?")
    
    scenarios_quiz = [
        "Email spam filter processing millions of emails daily",
        "Calculator app that performs basic arithmetic",
        "System that recognizes handwritten digits",
        "User interface that adapts to user preferences over time",
        "Simple alarm clock application"
    ]
    
    scenario = st.selectbox("Select a scenario:", scenarios_quiz)
    
    answers = {
        "Email spam filter processing millions of emails daily": 
            {"answer": "Use ML", "explanation": "High volume data with complex patterns - perfect for ML."},
        "Calculator app that performs basic arithmetic": 
            {"answer": "Traditional Programming", "explanation": "Simple, rule-based operations with clear algorithms."},
        "System that recognizes handwritten digits": 
            {"answer": "Use ML", "explanation": "Complex pattern recognition task where rules are hard to define."},
        "User interface that adapts to user preferences over time": 
            {"answer": "Use ML", "explanation": "Personalization that improves with more user interaction data."},
        "Simple alarm clock application": 
            {"answer": "Traditional Programming", "explanation": "Well-defined functionality with clear rules."}
    }
    
    user_answer = st.radio("What's your decision?", ["Use ML", "Traditional Programming"])
    
    if st.button("Check", key="when_ml_check"):
        correct_answer = answers[scenario]["answer"]
        explanation = answers[scenario]["explanation"]
        
        if user_answer == correct_answer:
            st.success(f"Correct! {explanation}")
        else:
            st.error(f"Not quite. {explanation}")

# Why Machine Learning vs Generative AI
with tab_ml_genai:
    st.title("Why Traditional ML vs Generative AI")
    
    st.write("""
    While generative AI is powerful and flexible, traditional machine learning approaches 
    still have important advantages in many scenarios. Here are key reasons to choose traditional ML over generative AI in certain situations:
    """)
    
    # Create tabs for each advantage
    adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5 = st.tabs([
        "Transparency", 
        "Explainability", 
        "Robustness", 
        "Data Efficiency",
        "Task Performance"
    ])
    
    with adv_tab1:
        st.subheader("Transparency and Interpretability")
        st.write("""
        Traditional ML models often provide clear insight into their decision-making process.
        
        **Why it matters:**
        - Decision processes are traceable and verifiable
        - Essential for regulated industries like finance and healthcare
        - Makes compliance requirements easier to meet
        - Helps build trust in the system
        """)
        
        st.info("**Example:** A linear regression model for loan approval can clearly show which factors (income, credit history, etc.) influenced the decision and by how much.")
        
    with adv_tab2:
        st.subheader("Explainability and Fairness")
        st.write("""
        Traditional models can be easier to analyze for bias and fairness.
        
        **Why it matters:**
        - Easier to audit for bias in critical sectors
        - More straightforward to explain decisions to stakeholders
        - Transparent processes help ensure ethical deployment
        - Supports legal and regulatory requirements for non-discrimination
        """)
        
        st.info("**Example:** When rejecting a loan application, a traditional ML model can provide specific reasons for the rejection based on its decision factors.")
    
    with adv_tab3:
        st.subheader("Robustness and Consistency")
        st.write("""
        Traditional ML models often deliver more consistent and predictable outputs.
        
        **Why it matters:**
        - Less prone to hallucinations or fabricated outputs
        - More reliable for critical applications
        - Consistent behavior across similar inputs
        - Better suited for safety-critical systems
        """)
        
        st.info("**Example:** In medical diagnosis, traditional models may provide fewer false positives/negatives compared to generative AI models that could occasionally produce convincing but incorrect diagnoses.")
    
    with adv_tab4:
        st.subheader("Data Efficiency")
        st.write("""
        Traditional ML often requires less training data to achieve good performance.
        
        **Why it matters:**
        - Can work effectively with smaller datasets
        - Lower computational requirements
        - More accessible for organizations with limited data
        - Faster to train and deploy
        """)
        
        st.info("**Example:** A decision tree can be effectively trained on a few thousand examples, while LLMs may require billions of tokens to develop robust capabilities.")
    
    with adv_tab5:
        st.subheader("Specific Task Performance")
        st.write("""
        Traditional models built for specific purposes can achieve better results for those tasks.
        
        **Why it matters:**
        - Optimized for particular problems
        - Higher accuracy for narrow, well-defined tasks
        - More efficient resource utilization
        - Easier to benchmark and validate
        """)
        
        st.info("**Example:** A specialized fraud detection model trained specifically on financial transaction data will likely outperform a general-purpose LLM at identifying fraudulent transactions.")
    
    # Comparison chart
    st.subheader("Comparison: Traditional ML vs Generative AI")
    
    comparison_data = {
        "Factor": [
            "Interpretability", 
            "Training Data Required", 
            "Computational Resources", 
            "Specialized Task Performance",
            "Versatility",
            "Creativity",
            "Bias Detection",
            "Consistency"
        ],
        "Traditional ML": [
            "High", 
            "Moderate", 
            "Lower", 
            "Higher for specific tasks",
            "Lower",
            "Limited",
            "Easier",
            "Higher"
        ],
        "Generative AI": [
            "Low", 
            "Massive", 
            "Higher", 
            "Lower for specific tasks",
            "Higher",
            "High",
            "More difficult",
            "Variable"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    st.write("""
    **Key Takeaway:** The choice between traditional ML and generative AI depends on your specific requirements:
    
    - **Traditional ML** is better when you need interpretability, efficiency, and consistency for well-defined tasks
    - **Generative AI** excels at versatility, creativity, and handling tasks requiring contextual understanding
    """)

# Terminology and concepts
with tab_terms:
    st.title("Machine Learning Terminology and Concepts")
    
    st.write("""
    Understanding the key terminology in machine learning is essential for effective communication and implementation.
    Here are the core terms you should know:
    """)
    
    # Create a dataframe for the terminology
    terms_data = {
        "ML Term": ["Label/Target", "Feature", "Feature Engineering", "Feature Selection"],
        "Statistical Definition": ["Dependent variable", "Independent variable", "Data transformation", "Variable/subset selection"],
        "Everyday Definition": ["What you are trying to predict", "Data that helps you make predictions", "Process of reshaping data to get more value out of it", "Process of using the most valuable data"],
        "Example": [
            "Customer churn (Yes/No), House price ($300K)",
            "Age, Income, Purchase history, Email clicks",
            "Creating ratio of clicks to purchases, Converting dates to day of week",
            "Selecting only the features that most strongly correlate with the target"
        ]
    }
    
    terms_df = pd.DataFrame(terms_data)
    st.table(terms_df)
    
    # Example dataset
    st.subheader("Example Dataset with Features and Target")
    
    example_data = {
        "customer_id": [1001, 1002, 1003, 1004, 1005],
        "age": [34, 28, 45, 52, 31],
        "income": [72000, 48000, 92000, 65000, 58000],
        "gender": ["M", "F", "F", "M", "F"],
        "location": ["urban", "suburban", "urban", "rural", "suburban"],
        "purchase_history": [12, 5, 20, 8, 10],
        "website_visits": [25, 18, 40, 12, 30],
        "email_clicks": [12, 5, 4, 3, 10],
        "days_since_last_purchase": [8, 30, 15, 45, 3],
        "target_purchase": [1, 0, 1, 0, 1]
    }
    
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)
    
    st.write("""
    In this example:
    - **Target/Label**: `target_purchase` (whether the customer made a purchase or not)
    - **Features**: All other columns like age, income, gender, etc.
    """)
    
    # Interactive example
    st.subheader("Interactive: Identify Terms")
    
    scenarios = [
        "When building a house price prediction model, the price in dollars is the _____.",
        "In an email spam detection system, words in the email are _____.",
        "Converting timestamps to day of week is an example of _____.",
        "Removing features that don't contribute to prediction accuracy is _____."
    ]
    
    answers = {
        "When building a house price prediction model, the price in dollars is the _____.": "Label/Target",
        "In an email spam detection system, words in the email are _____.": "Features",
        "Converting timestamps to day of week is an example of _____.": "Feature Engineering",
        "Removing features that don't contribute to prediction accuracy is _____.": "Feature Selection"
    }
    
    scenario = st.selectbox("Complete the sentence:", scenarios)
    
    options = ["Label/Target", "Features", "Feature Engineering", "Feature Selection"]
    user_answer = st.radio("Select the correct term:", options)
    
    if st.button("Check", key="terms_check"):
        correct_answer = answers[scenario]
        if user_answer == correct_answer:
            st.success(f"Correct! {scenario.replace('_____.', correct_answer)}")
        else:
            st.error(f"Not quite. The correct answer is {correct_answer}.")

# Machine learning types
with tab_ml_types:
    st.title("Machine Learning Types")
    
    # st.image("assets/ml_types.png", use_column_width=True)
    
    st.write("""
    Machine learning can be categorized into different types based on the learning approach and the kind of problems they solve.
    The three main types are:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Supervised Learning")
        st.write("""
        Learning from labeled training data to make predictions.
        
        **Key Characteristics:**
        - Training data includes correct answers (labels)
        - Algorithm learns to map inputs to outputs
        - Learns from examples and feedback
        
        **Types:**
        - **Classification**: Predicting a category
          - Binary: Yes/No, Spam/Not Spam
          - Multiclass: Apple/Orange/Banana
        - **Regression**: Predicting a numeric value
          - House prices, Temperature, Sales forecast
        
        **Examples:**
        - Spam detection
        - Credit scoring
        - Sales forecasting
        """)
    
    with col2:
        st.subheader("Unsupervised Learning")
        st.write("""
        Discovering patterns in unlabeled data.
        
        **Key Characteristics:**
        - No labels or correct answers provided
        - Finds hidden patterns or structures
        - Self-guided learning
        
        **Types:**
        - **Clustering**: Grouping similar items
        - **Dimensionality Reduction**: Simplifying data
        - **Anomaly Detection**: Finding outliers
        
        **Examples:**
        - Customer segmentation
        - Recommendation systems
        - Anomaly detection
        """)
    
    with col3:
        st.subheader("Reinforcement Learning")
        st.write("""
        Learning through trial, error, and rewards.
        
        **Key Characteristics:**
        - Agent learns from environment
        - Takes actions to maximize rewards
        - Learns optimal behavior through experience
        
        **Components:**
        - Agent: The learner/decision maker
        - Environment: Where the agent operates
        - Actions: What the agent can do
        - Rewards: Feedback from the environment
        
        **Examples:**
        - Game playing (AlphaGo, DeepRacer)
        - Robotics
        - Resource management
        """)
    
    st.subheader("Self-Supervised Learning")
    st.write("""
    A special approach used extensively in generative AI where the model generates its own supervision signal from the data.
    
    **Key Characteristics:**
    - Creates pseudo-labels from the input data itself
    - No human-provided labels needed
    - Can leverage vast amounts of unlabeled data
    
    **How it works:**
    - Part of the input data is hidden or masked
    - Model is trained to predict the masked portion
    - The original, complete data serves as its own supervision
    """)
    
    # Self-supervised learning visualization
    st.code("""
    Original sentence: "The quick brown fox jumped over the lazy dog."
    
    Masked input: "The quick brown fox jumped over the [MASK] dog."
    
    Model task: Predict the masked word ("lazy") using context from the rest of the sentence.
    """)
    
    # Interactive example
    st.subheader("Interactive: Match Learning Types")
    
    examples = [
        "Predicting house prices based on features like size, location, and age",
        "Grouping customers based on purchasing behavior without predefined categories",
        "Teaching a robot to walk by rewarding successful steps",
        "Training a language model by masking words in sentences and asking it to predict them"
    ]
    
    learning_types = ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Self-Supervised Learning"]
    
    example = st.selectbox("Select an example:", examples)
    
    user_answer = st.radio("What type of machine learning is this?", learning_types)
    
    answers = {
        "Predicting house prices based on features like size, location, and age": "Supervised Learning",
        "Grouping customers based on purchasing behavior without predefined categories": "Unsupervised Learning",
        "Teaching a robot to walk by rewarding successful steps": "Reinforcement Learning",
        "Training a language model by masking words in sentences and asking it to predict them": "Self-Supervised Learning"
    }
    
    if st.button("Check", key="ml_types_check"):
        correct_answer = answers[example]
        if user_answer == correct_answer:
            st.success(f"Correct! This is {correct_answer}.")
        else:
            st.error(f"Not quite. This example describes {correct_answer}.")

# Common Use Cases
with tab_use_cases:
    st.title("Common Use Cases for AI/ML")
    
    st.write("""
    Machine learning and AI can be applied across many industries. Here are some common use cases where traditional ML 
    and generative AI provide significant value:
    """)
    
    # Create tabs for different domains
    fin_tab, health_tab, auto_tab, mfg_tab, cyber_tab, pred_tab = st.tabs([
        "Finance", 
        "Healthcare", 
        "Autonomous Vehicles",
        "Manufacturing",
        "Cybersecurity",
        "Predictive Maintenance"
    ])
    
    with fin_tab:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Financial Use Cases")
            st.write("""
            **Fraud Detection**
            
            Traditional ML is preferred because:
            - Interpretability & explainability are crucial for regulations & fairness
            - Clear decision rules are required for audit trails
            - Models need to explain why transactions were flagged as fraudulent
            
            **Credit Risk Assessment**
            
            Traditional models like logistic regression and decision trees are preferred for:
            - Transparent credit scoring with clear decision criteria
            - Regulatory compliance requirements
            - Avoiding discriminatory lending practices
            """)
        
        with col2:
            # Create a simple visualization for fraud detection
            labels = ['Normal Transactions', 'Fraudulent']
            sizes = [98.7, 1.3]
            colors = ['lightblue', 'red']
            
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title('Typical Distribution in Fraud Detection')
            st.pyplot(fig)
            
            st.info("**Key Challenge:** Class imbalance - fraudulent transactions typically represent less than 1% of all transactions")
    
    with health_tab:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Healthcare Diagnostics")
            st.write("""
            Both accuracy and interpretability are critical in healthcare applications. Traditional ML excels when applied to structured data such as:
            
            - Patient records
            - Lab results 
            - Medical imaging data
            
            ML models can provide:
            - Reliable diagnoses & treatment recommendations
            - Decision processes that can be audited by medical experts
            - Support for clinical decision-making
            """)
        
        with col2:
            # Simple visualization of ML in medical imaging
            st.image("https://img.freepik.com/free-vector/doctor-examining-patient-with-x-ray-medical-worker-man-blue-uniform-examines-scan-with-patients-lungs-healthcare-medicine-concept-vector-illustration_169479-788.jpg?w=900", caption="ML assists in medical imaging analysis")
            
            st.info("ML can identify patterns in medical images that may be difficult for human eyes to detect, but the final diagnosis should always involve human expertise.")
    
    with auto_tab:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Autonomous Vehicles")
            st.write("""
            Safety and reliability are paramount in autonomous driving systems. Traditional ML approaches are preferred:
            
            - Computer vision for object detection
            - Rule-based systems for decision making
            - Sensor fusion techniques
            
            These provide:
            - Robust and predictable behavior in real-time environments
            - Ability to make split-second decisions
            - Clear audit trails when accidents occur
            """)
        
        with col2:
            # Simple visualization of autonomous vehicle sensing
            radar_range = np.linspace(0, np.pi, 100)
            lidar_range = np.linspace(0, 2*np.pi, 100)
            
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='polar')
            ax.plot(radar_range, np.ones(100)*5, 'r-', label='Radar')
            ax.plot(lidar_range, np.ones(100)*7, 'b-', label='Lidar')
            ax.set_rticks([2, 4, 6, 8, 10])
            ax.set_rlabel_position(45)
            ax.legend(loc='upper right')
            ax.set_title('Sensor Coverage in Autonomous Vehicles', y=1.08)
            st.pyplot(fig)
    
    with mfg_tab:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Manufacturing Quality Control")
            st.write("""
            Traditional ML excels in manufacturing for detecting defects and anomalies using:
            
            - Computer vision on production lines
            - Sensor data analysis
            - Statistical process control
            
            Benefits include:
            - Precise and consistent quality assessment
            - Real-time detection of manufacturing issues
            - Traceability for quality assurance
            - Process optimization
            """)
        
        with col2:
            # Quality control chart
            x = np.arange(1, 31)
            quality_metric = np.random.normal(10, 1, 30)
            upper_control = np.ones(30) * 12
            lower_control = np.ones(30) * 8
            
            fig, ax = plt.subplots()
            ax.plot(x, quality_metric, 'bo-', label='Quality Metric')
            ax.plot(x, upper_control, 'r--', label='Upper Control Limit')
            ax.plot(x, lower_control, 'r--', label='Lower Control Limit')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Measurement')
            ax.set_title('Quality Control Chart')
            ax.legend()
            st.pyplot(fig)
            
            st.info("ML can detect subtle patterns that indicate potential quality issues before they become visible defects.")
    
    with cyber_tab:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Cybersecurity Threat Detection")
            st.write("""
            Traditional ML is widely used to analyze network traffic, system logs, and other security data to:
            
            - Detect anomalies that may indicate intrusions
            - Identify malware signatures
            - Monitor for unusual user behavior
            
            ML provides:
            - High accuracy and reliability in threat detection
            - Reduced false positives compared to rule-based systems
            - Ability to detect novel attack patterns
            """)
        
        with col2:
            # Network traffic anomaly visualization
            traffic = np.random.normal(100, 15, 100)
            # Insert anomaly
            traffic[70:75] = traffic[70:75] * 3
            
            fig, ax = plt.subplots()
            ax.plot(traffic, 'b-')
            ax.axvspan(70, 75, color='r', alpha=0.3, label='Anomaly')
            ax.set_xlabel('Time')
            ax.set_ylabel('Network Traffic')
            ax.set_title('Network Traffic Anomaly Detection')
            ax.legend()
            st.pyplot(fig)
    
    with pred_tab:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Predictive Maintenance")
            st.write("""
            Traditional ML models excel at predicting when equipment will fail before it happens by analyzing:
            
            - Sensor data from machines
            - Maintenance records
            - Operating conditions
            
            Using techniques like:
            - Time-series forecasting
            - Regression analysis
            - Classification of failure modes
            
            Benefits include:
            - Reduced downtime
            - Lower maintenance costs
            - Extended equipment life
            - Optimized maintenance scheduling
            """)
        
        with col2:
            # Predictive maintenance visualization
            time = np.arange(0, 100)
            vibration = 2 + 0.1 * time + np.random.normal(0, 1, 100)
            threshold = np.ones(100) * 11
            
            fig, ax = plt.subplots()
            ax.plot(time, vibration, 'b-', label='Vibration Level')
            ax.plot(time, threshold, 'r--', label='Failure Threshold')
            ax.axvline(x=90, color='g', linestyle=':', label='Predicted Failure')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Vibration')
            ax.set_title('Predictive Maintenance')
            ax.legend()
            st.pyplot(fig)
            
            st.info("ML can predict equipment failure days or weeks in advance, allowing for planned maintenance rather than emergency repairs.")
    
    # Decision guide
    st.subheader("Choosing Between Traditional ML and Generative AI")
    
    st.write("""
    When deciding which approach to use for a specific use case, consider these factors:
    """)
    
    decision_factors = {
        "Factor": [
            "Interpretability Requirements",
            "Regulatory Compliance",
            "Data Availability",
            "Task Specificity",
            "Performance Requirements",
            "Creative Content Generation",
            "Deployment Environment"
        ],
        "Choose Traditional ML When...": [
            "Decisions must be explainable and transparent",
            "Industry has strict regulatory requirements",
            "Limited specialized data is available",
            "Task is narrowly defined with clear success metrics",
            "Consistent, reliable results are critical",
            "Creative generation is not needed",
            "Edge devices with limited resources"
        ],
        "Choose Generative AI When...": [
            "Black-box solutions are acceptable",
            "Fewer regulatory constraints exist",
            "Large diverse datasets are available",
            "Task requires flexibility across domains",
            "Some variability in results is acceptable",
            "Content creation and variation are desired",
            "Powerful cloud infrastructure is available"
        ]
    }
    
    decision_df = pd.DataFrame(decision_factors)
    st.dataframe(decision_df, use_container_width=True)

# ML process overview
with tab_process:
    st.title("Machine Learning Development Lifecycle")
    
    # st.image("assets/ml_lifecycle.png", use_column_width=True)
    
    st.write("""
    The machine learning development lifecycle is an iterative process that starts with a business problem and ends 
    with a deployed model that provides ongoing value. Let's explore each phase:
    """)
    
    # Create expandable sections for each phase
    with st.expander("1. Business Problem Framing", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            The ML process begins with a clear business problem that needs to be solved. This phase involves:
            
            - **Business Problem Identification**: Understand the core challenge
            - **ML Problem Framing**: Determine if and how ML can address the problem
            - **Success Criteria**: Define how success will be measured
            - **Feasibility Assessment**: Evaluate whether ML is the right approach
            """)
        
        with col2:
            st.info("**Key Question:** Can this business problem be framed as an ML problem?")
    
    with st.expander("2. Data Collection and Preparation"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            Once the ML problem is framed, data needs to be collected and prepared:
            
            - **Data Collection**: Gather relevant data from various sources
            - **Data Integration**: Combine data from different sources
            - **Data Cleaning**: Handle missing values, outliers, and errors
            - **Data Transformation**: Convert data into format suitable for ML
            """)
        
        with col2:
            st.info("**Key Question:** Do we have enough quality data to address our ML problem?")
    
    with st.expander("3. Data Visualization and Analysis"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            This phase involves exploring and understanding the data:
            
            - **Exploratory Data Analysis (EDA)**: Understand data distributions and relationships
            - **Statistical Analysis**: Identify correlations and significant features
            - **Data Visualization**: Create graphs and charts to reveal patterns
            - **Insight Generation**: Develop hypotheses about the data
            """)
        
        with col2:
            st.info("**Key Question:** What patterns or insights can we extract from the data?")
    
    with st.expander("4. Feature Engineering"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            Feature engineering transforms raw data into features that better represent the underlying problem:
            
            - **Feature Creation**: Develop new features from existing data
            - **Feature Transformation**: Apply scaling, encoding, and other transformations
            - **Feature Selection**: Choose the most relevant features for the model
            - **Dimensionality Reduction**: Simplify the feature space if needed
            """)
        
        with col2:
            st.info("**Key Question:** How can we represent the data to best capture the patterns relevant to our problem?")
    
    with st.expander("5. Model Training and Parameter Tuning"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            This phase involves selecting, training, and optimizing models:
            
            - **Algorithm Selection**: Choose appropriate ML algorithms
            - **Model Training**: Train models on prepared data
            - **Hyperparameter Tuning**: Optimize model parameters
            - **Cross-Validation**: Ensure model generalizes well
            """)
        
        with col2:
            st.info("**Key Question:** Which model and parameters best capture the patterns in our data?")
    
    with st.expander("6. Model Evaluation"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            Model evaluation determines if the model meets business requirements:
            
            - **Performance Metrics**: Calculate accuracy, precision, recall, etc.
            - **Business Value Assessment**: Determine if the model provides sufficient business value
            - **Error Analysis**: Understand where and why the model makes mistakes
            - **Comparison**: Compare different models to select the best one
            """)
        
        with col2:
            st.info("**Key Question:** Does the model's performance meet our business goals?")
    
    with st.expander("7. Model Deployment"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            Deployment makes the model available for use in production:
            
            - **Infrastructure Setup**: Prepare the environment for model deployment
            - **Integration**: Connect the model with existing systems
            - **Scaling**: Ensure the model can handle production load
            - **Documentation**: Create documentation for users and developers
            """)
        
        with col2:
            st.info("**Key Question:** How can we make our model available and useful in production?")
    
    with st.expander("8. Monitoring and Debugging"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            Once deployed, the model requires ongoing monitoring:
            
            - **Performance Monitoring**: Track model performance metrics
            - **Drift Detection**: Identify when data or concept drift occurs
            - **Debugging**: Troubleshoot issues as they arise
            - **Feedback Collection**: Gather user feedback for improvements
            """)
        
        with col2:
            st.info("**Key Question:** Is our model continuing to perform well in production?")
    
    with st.expander("9. Iteration and Improvement"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            ML development is iterative and requires continuous improvement:
            
            - **Data Augmentation**: Collect additional data if needed
            - **Feature Augmentation**: Create new features to improve performance
            - **Model Retraining**: Update the model with new data
            - **Process Refinement**: Improve the entire ML pipeline
            """)
        
        with col2:
            st.info("**Key Question:** How can we continuously improve our model's performance and value?")
    
    # MLOps chart
    st.subheader("MLOps: Operationalizing the ML Lifecycle")
    
    st.write("""
    MLOps combines ML and DevOps practices to standardize and streamline the ML lifecycle, 
    making it more reliable, scalable, and maintainable.
    """)
    
    mlops_components = {
        "Component": [
            "Continuous Integration (CI)",
            "Continuous Delivery (CD)",
            "Model Versioning",
            "Automated Testing",
            "Monitoring & Alerting",
            "Reproducibility"
        ],
        "Description": [
            "Automatically build and test code changes",
            "Automate model deployment to production",
            "Track model versions and their performance",
            "Test model quality, data quality, and infrastructure",
            "Track model performance and trigger alerts when issues arise",
            "Ensure experiments can be repeated with the same results"
        ]
    }
    
    mlops_df = pd.DataFrame(mlops_components)
    st.table(mlops_df)
    
    st.info("MLOps helps organizations overcome the challenges of deploying ML in production by bringing software engineering best practices to ML development.")

# AWS AI/ML stack
with tab_stack:
    st.title("Amazon AI/ML Stack")
    
    # st.image("assets/aws_ml_stack.png", use_column_width=True)
    
    st.write("""
    AWS offers a comprehensive stack of AI and ML services to support all aspects of the machine learning 
    lifecycle, from high-level AI services to low-level infrastructure. The stack consists of three main layers:
    """)
    
    # Create tabs for each layer
    layer1, layer2, layer3 = st.tabs([
        "AI Services", 
        "ML Services (SageMaker)", 
        "ML Frameworks & Infrastructure"
    ])
    
    with layer1:
        st.subheader("AI Services")
        st.write("""
        Pre-trained AI services that require no machine learning expertise. These services 
        provide ready-to-use intelligence for specific use cases.

        **Key Characteristics:**
        - No ML expertise required
        - API-driven integration
        - Pre-trained models
        - Specialized for specific tasks
        """)
        
        # Create columns for different AI service categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Vision")
            st.write("""
            **Amazon Rekognition**
            - Image and video analysis
            - Object detection
            - Facial recognition
            - Content moderation
            
            **Amazon Textract**
            - Document text extraction
            - Form data extraction
            - Table data extraction
            """)
        
        with col2:
            st.markdown("#### Language")
            st.write("""
            **Amazon Comprehend**
            - Natural language processing
            - Entity recognition
            - Sentiment analysis
            - Key phrase extraction
            
            **Amazon Translate**
            - Neural machine translation
            - Real-time translation
            - Document translation
            """)
        
        with col3:
            st.markdown("#### Business Solutions")
            st.write("""
            **Amazon Personalize**
            - Real-time personalization
            - Recommendation systems
            
            **Amazon Fraud Detector**
            - Online fraud detection
            - Account takeover prevention
            
            **Amazon Forecast**
            - Time-series forecasting
            - Demand planning
            """)
        
        # Interactive selector
        st.subheader("Explore AWS AI Services")
        
        services = {
            "Amazon Rekognition": {
                "description": "Automate image and video analysis with machine learning",
                "use_cases": ["Media analysis", "Identity verification", "Content moderation"],
                "features": ["Object & scene detection", "Face analysis", "Text detection", "Content moderation"]
            },
            "Amazon Textract": {
                "description": "Extract text and data from documents using ML",
                "use_cases": ["Automated document processing", "Smart search indexes", "Compliance document archives"],
                "features": ["OCR+", "Form extraction", "Table extraction", "Document analysis"]
            },
            "Amazon Comprehend": {
                "description": "Discover insights and relationships in text",
                "use_cases": ["Call center analytics", "Content organization", "Sentiment analysis"],
                "features": ["Entity recognition", "Key phrase extraction", "Sentiment analysis", "Topic modeling"]
            },
            "Amazon Kendra": {
                "description": "Intelligent search service powered by ML",
                "use_cases": ["Enterprise search", "Knowledge management", "Customer support"],
                "features": ["Natural language understanding", "Document connectors", "Relevance tuning", "Incremental learning"]
            },
            "Amazon Personalize": {
                "description": "Create personalized user experiences at scale",
                "use_cases": ["Product recommendations", "Content personalization", "Personalized rankings"],
                "features": ["Real-time personalization", "Easy implementation", "Customized recommendations"]
            },
            "Amazon Fraud Detector": {
                "description": "Detect online fraud faster",
                "use_cases": ["Account registration fraud", "Payment fraud", "Guest checkout fraud"],
                "features": ["Custom fraud detection", "Real-time API", "Amazon fraud expertise"]
            }
        }
        
        selected_service = st.selectbox("Select an AWS AI Service to explore:", list(services.keys()))
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(selected_service)
            st.write(services[selected_service]["description"])
            
            st.markdown("#### Key Features")
            for feature in services[selected_service]["features"]:
                st.write(f"- {feature}")
        
        with col2:
            st.markdown("#### Use Cases")
            for use_case in services[selected_service]["use_cases"]:
                st.write(f"- {use_case}")
            
            st.markdown("#### No ML Expertise Required!")
            st.info("These services are designed to be used via simple API calls, allowing developers without ML expertise to add intelligence to their applications.")
            
    with layer2:
        st.subheader("ML Services (Amazon SageMaker)")
        st.write("""
        Amazon SageMaker is a fully managed service that enables data scientists and developers to quickly and easily build, train, and deploy machine learning models at any scale.
        """)
        
        # Create columns for different SageMaker capabilities
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### Prepare")
            st.write("""
            - **Ground Truth**: Label data
            - **Data Wrangler**: Prepare data
            - **Feature Store**: Store features
            - **Processing**: Process data
            - **Clarify**: Detect bias
            """)
        
        with col2:
            st.markdown("#### Build")
            st.write("""
            - **Studio**: IDE for ML
            - **Notebooks**: Code execution
            - **Algorithms**: Built-in algorithms
            - **JumpStart**: Pre-built solutions
            - **Autopilot**: AutoML
            """)
        
        with col3:
            st.markdown("#### Train & Tune")
            st.write("""
            - **Training**: Distributed training
            - **Hyperparameter Tuning**: Optimize models
            - **Debugger**: Debug training
            - **Experiments**: Track experiments
            - **Distributed Training**: Scale training
            """)
        
        with col4:
            st.markdown("#### Deploy & Manage")
            st.write("""
            - **Endpoints**: Serve models
            - **Batch Transform**: Batch inference
            - **Model Monitor**: Monitor drift
            - **Pipelines**: ML workflows
            - **Model Registry**: Catalog models
            """)
        
        # SageMaker interfaces visualization
        st.subheader("SageMaker Interfaces")
        
        interfaces = {
            "Interface": ["SageMaker Studio", "SageMaker Canvas", "SageMaker APIs", "SageMaker Studio Lab"],
            "Description": [
                "Integrated IDE for ML development",
                "No-code ML for business analysts",
                "Programmatic access for automation",
                "Free ML development environment"
            ],
            "Target Users": [
                "Data Scientists, ML Engineers",
                "Business Analysts, Domain Experts",
                "Developers, ML Engineers",
                "Students, ML Beginners"
            ]
        }
        
        interfaces_df = pd.DataFrame(interfaces)
        st.table(interfaces_df)
        
        # SageMaker value proposition
        st.subheader("Why Use SageMaker?")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("""
            SageMaker eliminates the heavy lifting from each step of the machine learning workflow:
            
            1. **Simplified ML Development**: Integrated tools for the entire ML lifecycle
            2. **Reduced Time to Production**: Accelerate from idea to deployment
            3. **Cost Optimization**: Pay only for what you use
            4. **Scalability**: Scale from experiments to production workloads
            5. **Reduced Operational Overhead**: Fully managed infrastructure
            """)
        
        with col2:
            # Create a simple cost/benefit visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Infrastructure Management', 'ML Development', 'Model Deployment', 'Model Monitoring'],
                y=[80, 30, 75, 60],
                name='Without SageMaker',
                marker_color='indianred'
            ))
            fig.add_trace(go.Bar(
                x=['Infrastructure Management', 'ML Development', 'Model Deployment', 'Model Monitoring'],
                y=[10, 20, 15, 20],
                name='With SageMaker',
                marker_color='lightseagreen'
            ))
            
            fig.update_layout(
                title='Effort Reduction with SageMaker',
                xaxis_title='ML Lifecycle Phase',
                yaxis_title='Relative Effort',
                barmode='group',
                height=400,
                width=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with layer3:
        st.subheader("ML Frameworks & Infrastructure")
        st.write("""
        The foundation of the AWS AI/ML stack, providing flexible, low-level tools for advanced machine learning practitioners.

        **Key Characteristics:**
        - Maximum flexibility and control
        - Support for all major ML frameworks
        - Optimized infrastructure for ML workloads
        - Requires ML expertise to use effectively
        """)
        
        # Split into frameworks and infrastructure
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ML Frameworks")
            st.write("""
            AWS supports all major machine learning frameworks:
            
            - **TensorFlow**: Google's open-source ML library
            - **PyTorch**: Facebook's deep learning framework
            - **MXNet**: A flexible deep learning framework
            - **Scikit-learn**: Python-based ML library
            - **Hugging Face**: Transformers for NLP
            """)
        
        with col2:
            st.markdown("#### ML Infrastructure")
            st.write("""
            AWS provides specialized compute resources for ML:
            
            - **Amazon EC2**: CPU and GPU instances
            - **AWS Trainium**: ML training acceleration
            - **AWS Inferentia**: ML inference acceleration
            - **AWS Deep Learning AMIs**: Pre-configured ML environments
            - **AWS Deep Learning Containers**: Containerized ML environments
            """)
        
        # Hardware acceleration visualization
        st.subheader("ML Hardware Acceleration Options")
        
        hardware = {
            "Hardware": ["CPU", "GPU", "AWS Inferentia", "AWS Trainium", "FPGA"],
            "Best For": [
                "General purpose ML, small models",
                "Deep learning training, model parallelism",
                "Cost-effective inference at scale",
                "Cost-effective large-scale training",
                "Custom acceleration algorithms"
            ],
            "Relative Cost": ["$", "$$$", "$$", "$$", "$$$$"],
            "Relative Performance": ["⭐", "⭐⭐⭐", "⭐⭐⭐⭐ (inference)", "⭐⭐⭐⭐ (training)", "⭐⭐⭐"]
        }
        
        hardware_df = pd.DataFrame(hardware)
        st.table(hardware_df)
        
        # Deployment approaches
        st.subheader("Deployment Approaches")
        
        deployment = {
            "Approach": ["SageMaker", "ECS/EKS with ML containers", "Custom EC2 deployment", "Edge deployment"],
            "Use When": [
                "You want a managed ML platform with minimal operational overhead",
                "You need container orchestration for ML workloads",
                "You need maximum control over infrastructure",
                "You need ML at the edge with limited connectivity"
            ],
            "Flexibility": ["Medium", "High", "Very High", "Medium"],
            "Operational Overhead": ["Low", "Medium", "High", "Medium"]
        }
        
        deployment_df = pd.DataFrame(deployment)
        st.table(deployment_df)
        
        st.info("While this layer offers maximum flexibility, most users are better served by SageMaker unless they have specific requirements that necessitate direct infrastructure control.")

# Knowledge Check
with tab_knowledge:
    st.title("Knowledge Check")
    
    if st.button("Reset Knowledge Check", key="reset_kc"):
        reset_knowledge_check()
        st.rerun()
    
    display_knowledge_check()

# Footer

st.markdown("""
<div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
</div>
""", unsafe_allow_html=True)
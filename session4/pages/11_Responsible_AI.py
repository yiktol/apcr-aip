import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from PIL import Image
import random
import hashlib
import utils.authenticate as authenticate
from utils.styles import load_css

# Set page config
st.set_page_config(
    page_title="Responsible AI Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# Function to generate a synthetic dataset for the examples
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    age = np.random.normal(40, 10, n_samples)
    income = 30000 + age * 1000 + np.random.normal(0, 10000, n_samples)
    
    # Gender with slight income disparity
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.5, 0.5])
    income[gender == 'Female'] = income[gender == 'Female'] * 0.85
    
    # Ethnicity with some disparity
    ethnicity = np.random.choice(['Group A', 'Group B', 'Group C', 'Group D'], 
                               size=n_samples, p=[0.5, 0.2, 0.2, 0.1])
    
    # Credit score slightly biased by demographic factors
    credit_score = (
        700 
        + 0.5 * (age - 40) 
        + 0.002 * (income - 70000)
        + np.random.normal(0, 50, n_samples)
    )
    
    # Add some bias
    credit_score[gender == 'Female'] -= 10
    credit_score[ethnicity == 'Group C'] -= 15
    credit_score[ethnicity == 'Group D'] -= 20
    
    # Ensure valid range
    credit_score = np.clip(credit_score, 300, 850)
    
    # Loan approval slightly biased
    approval_threshold = 650
    approved = credit_score > approval_threshold
    
    # Create some educational features
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                               size=n_samples, p=[0.3, 0.4, 0.2, 0.1])
    
    # Months at current job
    job_months = np.random.poisson(36, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'gender': gender,
        'ethnicity': ethnicity,
        'education': education,
        'job_months': job_months,
        'credit_score': credit_score,
        'approved': approved
    })
    
    return df

# Load sample data
df = generate_synthetic_data()

# Helper functions for each dimension example
def fairness_demo():
    st.markdown("<div class='dimension-card'>", unsafe_allow_html=True)
    st.markdown("## 🎯 Fairness")
    st.markdown("Fairness involves ensuring AI systems don't discriminate against different groups.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1589829545856-d10d557cf95f?q=80&w=1200", 
                 caption="Fairness means equal treatment across different groups", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Why it matters
        - Prevents discrimination
        - Builds trust in AI systems
        - Ensures equitable outcomes
        - Aligns with legal and ethical standards
        """)
    
    st.markdown("### Interactive Example: Loan Approval Bias Detection")
    
    # Show approval rates by gender and ethnicity
    st.write("Let's examine approval rates across different demographic groups:")
    
    tab1, tab2 = st.tabs(["Gender Analysis", "Ethnicity Analysis"])
    
    with tab1:
        gender_approval = df.groupby('gender')['approved'].mean().reset_index()
        fig = px.bar(gender_approval, x='gender', y='approved', 
                     title='Loan Approval Rate by Gender',
                     labels={'approved': 'Approval Rate', 'gender': 'Gender'},
                     color='gender', color_discrete_sequence=['#3498db', '#e74c3c'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        **Observation**: There appears to be a gender disparity in loan approval rates. This could indicate potential bias in the model.
        
        **Mitigation strategies**:
        - Implement fairness constraints during model training
        - Use balanced training data
        - Apply post-processing techniques to ensure equal approval rates
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        ethnicity_approval = df.groupby('ethnicity')['approved'].mean().reset_index()
        fig = px.bar(ethnicity_approval, x='ethnicity', y='approved', 
                     title='Loan Approval Rate by Ethnicity',
                     labels={'approved': 'Approval Rate', 'ethnicity': 'Ethnicity'},
                     color='ethnicity')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        **Observation**: Significant disparities exist in approval rates across ethnic groups.
        
        **Mitigation strategies**:
        - Remove ethnicity as a direct input feature
        - Apply fairness-aware algorithms
        - Audit models regularly for bias
        - Consider demographic parity or equal opportunity constraints
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Generate Fairness Report"):
        with st.spinner("Analyzing fairness metrics..."):
            time.sleep(1)
            
            st.success("Fairness analysis complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Gender Disparity", 
                          value=f"{abs(gender_approval['approved'][0] - gender_approval['approved'][1]):.2%}",
                          delta="-2.5% from last quarter")
            
            with col2:
                max_eth = ethnicity_approval['approved'].max()
                min_eth = ethnicity_approval['approved'].min()
                st.metric(label="Ethnicity Disparity Range", 
                          value=f"{max_eth - min_eth:.2%}",
                          delta="-1.2% from last quarter",
                          delta_color="inverse")
            
            st.markdown("### Recommended Actions:")
            st.info("1. Implement fairness constraints in the model training process")
            st.info("2. Consider resampling techniques to balance representation")
            st.info("3. Perform regular bias audits on model outputs")
    st.markdown("</div>", unsafe_allow_html=True)

def explainability_demo():
    st.markdown("<div class='dimension-card'>", unsafe_allow_html=True)
    st.markdown("## 🔍 Explainability")
    st.markdown("Explainability enables humans to understand how AI systems make decisions.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1434030216411-0b793f4b4173?q=80&w=1200", 
                 caption="Making AI decisions understandable to humans", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Why it matters
        - Builds trust through transparency
        - Helps identify potential biases
        - Allows for meaningful human oversight
        - Enables debugging and improvement
        """)
    
    st.markdown("### Interactive Example: Understanding Loan Approval Decisions")
    
    # Train a simple model
    if st.checkbox("Generate model explanations"):
        with st.spinner("Training model and generating explanations..."):
            # Prepare features and target
            features = ['age', 'income', 'job_months', 'credit_score']
            X = df[features]
            y = df['approved']
            
            # Split and standardize
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Feature importance
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(feature_importance, x='Feature', y='Importance', 
                         title='Feature Importance in Loan Approval Model',
                         color='Importance')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown(f"""
            **Key insights**:
            - Credit score is the most influential factor at {importance[features.index('credit_score')]:.1%}
            - Income contributes {importance[features.index('income')]:.1%} to the decision
            - Age and job stability have smaller but notable impacts
            
            This type of explanation helps users understand what factors are driving the model's decisions.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # SHAP values demonstration
            st.markdown("### Local Explanations for Individual Decisions")
            st.write("SHAP (SHapley Additive exPlanations) values help explain individual predictions:")
            
            # Simple SHAP visualization
            sample_idx = 42
            sample = X_test.iloc[sample_idx:sample_idx+1]
            
            prediction = model.predict_proba(sample)[0, 1]
            
            # Create a basic SHAP-like visualization
            feature_contributions = []
            base_value = 0.5  # Simplified base value
            
            for i, feature in enumerate(features):
                # Simplified feature contribution calculation
                contribution = importance[i] * (sample[feature].values[0] - X_train[feature].mean()) / X_train[feature].std()
                feature_contributions.append({
                    'Feature': feature,
                    'Value': sample[feature].values[0],
                    'Contribution': contribution
                })
            
            contrib_df = pd.DataFrame(feature_contributions)
            contrib_df = contrib_df.sort_values('Contribution', ascending=False)
            
            fig = px.bar(contrib_df, x='Contribution', y='Feature', 
                     title=f'Feature Impact on Prediction (Approval Probability: {prediction:.2f})',
                     orientation='h',
                     color='Contribution',
                     color_continuous_scale=['red', 'lightgrey', 'blue'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Example explanation in natural language
            st.markdown("### Natural Language Explanation")
            
            top_feature = contrib_df.iloc[0]['Feature']
            top_value = contrib_df.iloc[0]['Value']
            
            if top_feature == 'credit_score':
                feature_text = f"credit score of {int(top_value)}"
                comparison = "higher than average" if top_value > X_train['credit_score'].mean() else "lower than average"
            elif top_feature == 'income':
                feature_text = f"income of ${int(top_value):,}"
                comparison = "higher than average" if top_value > X_train['income'].mean() else "lower than average"
            elif top_feature == 'age':
                feature_text = f"age of {int(top_value)}"
                comparison = "higher than average" if top_value > X_train['age'].mean() else "lower than average"
            else:
                feature_text = f"{top_feature} of {int(top_value)}"
                comparison = "higher than average" if top_value > X_train[top_feature].mean() else "lower than average"
            
            st.info(f"""
            This loan application has a {prediction:.0%} probability of approval.
            
            The main factor affecting this decision is the applicant's {feature_text}, which is {comparison}.
            This accounts for approximately {abs(contrib_df.iloc[0]['Contribution']/sum(abs(contrib_df['Contribution']))):.0%} of the model's decision.
            """)
    st.markdown("</div>", unsafe_allow_html=True)

def privacy_security_demo():
    st.markdown("<div class='dimension-card'>", unsafe_allow_html=True)
    st.markdown("## 🔒 Privacy and Security")
    st.markdown("Privacy and security ensure data and models are protected from misuse.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1614064642761-92b4c1dcf540?q=80&w=1200", 
                 caption="Protecting sensitive data is essential for responsible AI", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Why it matters
        - Protects personal information
        - Prevents unauthorized access
        - Maintains regulatory compliance
        - Builds user trust
        """)
    
    st.markdown("### Interactive Example: Data Privacy Techniques")
    
    # Demo basic privacy protection techniques
    technique = st.selectbox("Select a Privacy Protection Technique:", 
                            ["Data Anonymization", "Data Masking", "Differential Privacy"])
    
    sample_data = pd.DataFrame({
        'Name': ['John Smith', 'Jane Doe', 'Alex Johnson', 'Maria Garcia'],
        'SSN': ['123-45-6789', '987-65-4321', '555-12-3456', '888-99-7777'],
        'Email': ['john@example.com', 'jane@example.com', 'alex@example.com', 'maria@example.com'],
        'Age': [32, 45, 27, 39],
        'Income': [75000, 85000, 62000, 91000],
        'Credit Score': [720, 680, 750, 810]
    })
    
    st.write("Original Sensitive Data:")
    st.dataframe(sample_data)
    
    if st.button("Apply Privacy Protection"):
        with st.spinner("Applying privacy protection..."):
            time.sleep(1)
            
            if technique == "Data Anonymization":
                protected_data = sample_data.copy()
                protected_data['Name'] = ['Person A', 'Person B', 'Person C', 'Person D']
                protected_data['SSN'] = ['XXX-XX-XXXX' for _ in range(len(protected_data))]
                protected_data['Email'] = ['user1@anonymous.com', 'user2@anonymous.com', 
                                          'user3@anonymous.com', 'user4@anonymous.com']
                
                st.success("Data anonymized successfully!")
                st.dataframe(protected_data)
                
                st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                st.markdown("""
                **Anonymization** removes or replaces personally identifiable information while preserving 
                the utility of the data for analysis. This helps protect individuals' identities while still 
                allowing the data to be useful for training AI models.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
                
            elif technique == "Data Masking":
                protected_data = sample_data.copy()
                protected_data['SSN'] = protected_data['SSN'].apply(lambda x: x[:3] + '-XX-' + x[-4:])
                protected_data['Email'] = protected_data['Email'].apply(lambda x: x[0] + '*****' + x[x.find('@'):])
                
                st.success("Data masked successfully!")
                st.dataframe(protected_data)
                
                st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                st.markdown("""
                **Data Masking** partially obscures sensitive information while maintaining some of its 
                characteristics. This technique is useful when you need to preserve some information for 
                functionality while still protecting privacy.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
                
            elif technique == "Differential Privacy":
                # Add noise to numerical columns
                protected_data = sample_data.copy()
                epsilon = 1.0  # Privacy parameter (lower = more private, more noise)
                
                # Apply noise to numerical columns
                for col in ['Age', 'Income', 'Credit Score']:
                    # Scale noise based on data range
                    sensitivity = (protected_data[col].max() - protected_data[col].min()) / 10
                    noise_scale = sensitivity / epsilon
                    noise = np.random.laplace(0, noise_scale, size=len(protected_data))
                    protected_data[col] = protected_data[col] + noise
                    # Round to make it look reasonable
                    if col in ['Age', 'Credit Score']:
                        protected_data[col] = protected_data[col].round().astype(int)
                    else:
                        protected_data[col] = protected_data[col].round(-2).astype(int)  # Round to hundreds
                
                st.success("Differential privacy applied successfully!")
                st.dataframe(protected_data)
                
                st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                st.markdown("""
                **Differential Privacy** adds mathematical noise to data in a way that preserves overall statistical 
                patterns while making it difficult to identify individual records. This technique provides strong 
                privacy guarantees while maintaining data utility for machine learning.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### Security Best Practices for AI Systems")
    
    security_practices = {
        "Data Encryption": "Encrypt sensitive data both at rest and in transit",
        "Access Controls": "Implement strict access controls and authentication for AI systems",
        "Secure API Design": "Design APIs with proper authentication and rate limiting",
        "Model Protection": "Protect AI models from extraction attacks and unauthorized access",
        "Regular Auditing": "Conduct regular security audits and penetration testing"
    }
    
    for practice, description in security_practices.items():
        st.info(f"**{practice}**: {description}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def safety_demo():
    st.markdown("<div class='dimension-card'>", unsafe_allow_html=True)
    st.markdown("## 🛡️ Safety")
    st.markdown("Safety focuses on preventing harmful system outputs and misuse.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1507925921958-8a62f3d1a50d?q=80&w=1200", 
                 caption="AI systems must be designed with safety as a priority", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Why it matters
        - Prevents harm to users and society
        - Avoids reinforcement of harmful content
        - Maintains reliability in critical applications
        - Builds trust in AI technologies
        """)
    
    st.markdown("### Interactive Example: Content Moderation")
    
    st.write("""
    This demo simulates a content moderation system that detects potentially harmful content.
    Try entering different text to see how the system responds.
    """)
    
    user_input = st.text_area("Enter text to analyze for safety concerns:", 
                             "Climate change is real and we need to take action.")
    
    # Simple content moderation simulation
    def simple_toxicity_score(text):
        potentially_harmful_words = [
            "hate", "kill", "threat", "violence", "racist", "attack",
            "bomb", "destroy", "harmful", "illegal", "danger"
        ]
        
        text = text.lower()
        
        # Calculate a simple toxicity score based on dangerous word matches
        score = 0
        matches = []
        
        for word in potentially_harmful_words:
            if word in text:
                score += 0.15
                matches.append(word)
        
        # Simulating context awareness by reducing score for educational content
        educational_indicators = ["research", "study", "analysis", "education", "discuss", "learn", "context"]
        
        for word in educational_indicators:
            if word in text:
                score = max(0, score - 0.05)
                
        # Add some randomness to avoid predictability
        score = min(1.0, score + random.uniform(-0.1, 0.1))
        
        # Ensure score is in the range [0, 1]
        score = max(0, min(1, score))
        
        return score, matches
    
    if st.button("Analyze Content"):
        with st.spinner("Analyzing content for safety..."):
            time.sleep(1)
            
            toxicity_score, flagged_terms = simple_toxicity_score(user_input)
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = toxicity_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Content Risk Score"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.7
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            if toxicity_score < 0.3:
                st.success("✅ Content appears safe.")
            elif toxicity_score < 0.7:
                st.warning("⚠️ Content may contain concerning elements and should be reviewed.")
                if flagged_terms:
                    st.write("Flagged terms:", ", ".join(flagged_terms))
            else:
                st.error("🚫 Content appears unsafe and has been flagged.")
                if flagged_terms:
                    st.write("Flagged terms:", ", ".join(flagged_terms))
    
    st.markdown("### Safety Framework Elements")
    
    safety_elements = [
        {
            "name": "Red Teaming",
            "description": "Employing ethical hackers to try to make the system produce harmful outputs."
        },
        {
            "name": "Robust Input Validation",
            "description": "Ensuring inputs are sanitized and validated before processing."
        },
        {
            "name": "Output Filtering",
            "description": "Adding post-processing filters to catch harmful content before display."
        },
        {
            "name": "Safety Monitoring",
            "description": "Continuous monitoring of the system for unexpected or harmful behavior."
        },
        {
            "name": "User Reporting Mechanisms",
            "description": "Providing easy ways for users to report safety issues."
        }
    ]
    
    for element in safety_elements:
        st.markdown(f"**{element['name']}**: {element['description']}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def controllability_demo():
    st.markdown("<div class='dimension-card'>", unsafe_allow_html=True)
    st.markdown("## 🎮 Controllability")
    st.markdown("Controllability ensures humans can monitor and govern AI system behavior.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1200", 
                 caption="Humans should maintain control over AI systems", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Why it matters
        - Ensures human oversight
        - Allows for intervention when necessary
        - Prevents autonomous harmful actions
        - Maintains accountability
        """)
    
    st.markdown("### Interactive Example: AI System Controls")
    
    st.write("""
    This demonstration shows how a responsible AI system provides various control mechanisms
    to human operators. Explore the different controls available in this simulated AI dashboard.
    """)
    
    # Simulated AI system controls
    st.markdown("#### System Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        system_status = st.radio("System Status", ["Active", "Monitoring Only", "Paused"])
    
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
    
    with col3:
        human_review = st.checkbox("Require Human Review", value=True)
    
    st.markdown("#### Decision Authority Settings")
    
    decision_areas = {
        "Content Moderation": st.slider("Content Moderation Autonomy", 0, 10, 3),
        "Resource Allocation": st.slider("Resource Allocation Autonomy", 0, 10, 2),
        "User Interaction": st.slider("User Interaction Autonomy", 0, 10, 5),
        "Data Processing": st.slider("Data Processing Autonomy", 0, 10, 7)
    }
    
    # Visualization of autonomy levels
    autonomy_df = pd.DataFrame({
        'Area': list(decision_areas.keys()),
        'Autonomy': list(decision_areas.values()),
        'Human Oversight': [10 - x for x in decision_areas.values()]
    })
    
    fig = px.bar(autonomy_df, x='Area', y=['Autonomy', 'Human Oversight'],
                title='AI System Autonomy vs. Human Oversight',
                barmode='stack',
                color_discrete_sequence=['#2ecc71', '#3498db'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Emergency override section
    st.markdown("#### Emergency Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("⚠️ Emergency Override")
        if st.button("Activate Emergency Shutdown"):
            st.warning("Emergency shutdown sequence initiated. AI systems will be safely terminated.")
            time.sleep(1)
            st.success("All AI systems successfully shutdown. Manual restart required.")
    
    with col2:
        st.warning("🔄 System Rollback")
        if st.button("Rollback to Safe State"):
            st.info("Preparing to rollback system to last known safe state...")
            time.sleep(1)
            st.success("System successfully rolled back to checkpoint from 24 hours ago.")
    
    st.markdown("#### Control Panel Insights")
    
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown(f"""
    **Current System Configuration**:
    - System is in **{system_status}** mode
    - Confidence threshold set to **{confidence_threshold:.0%}**
    - Human review is **{"Required" if human_review else "Not Required"}**
    
    This configuration ensures appropriate human oversight while allowing the AI system
    to operate efficiently within safe boundaries. The controls demonstrate key principles
    of responsible AI: maintaining human agency, providing override mechanisms, and establishing
    clear authority boundaries.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def veracity_robustness_demo():
    st.markdown("<div class='dimension-card'>", unsafe_allow_html=True)
    st.markdown("## ✅ Veracity and Robustness")
    st.markdown("Veracity and robustness ensure AI systems produce correct outputs even with unexpected inputs.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1569396116180-210c182bedb8?q=80&w=1200", 
                 caption="AI systems must be robust against adversarial inputs", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Why it matters
        - Ensures system reliability
        - Defends against adversarial attacks
        - Maintains performance with unexpected inputs
        - Builds trust through consistent behavior
        """)
    
    st.markdown("### Interactive Example: Adversarial Testing")
    
    st.write("""
    This demonstration shows how AI systems can be tested for robustness against adversarial inputs
    and edge cases. We'll simulate a simple image classification system and test its robustness.
    """)
    
    # Simulate a classification model
    def simulate_classification(confidence_level, noise_level=0):
        # Artificial delay
        time.sleep(0.5)
        
        base_confidence = confidence_level
        
        # Add some randomness
        adjusted_confidence = base_confidence * (1 - noise_level * random.uniform(0, 1))
        
        # Ensure reasonable bounds
        adjusted_confidence = max(0, min(1, adjusted_confidence))
        
        # For high noise, sometimes flip the classification
        if noise_level > 0.7 and random.random() < 0.3:
            return "Cat" if base_confidence > 0.5 else "Dog", adjusted_confidence
        
        # Return class with adjusted confidence
        return "Dog" if adjusted_confidence > 0.5 else "Cat", adjusted_confidence
    
    st.markdown("#### Test Model Robustness")
    
    test_type = st.selectbox("Select Test Scenario:", 
                            ["Normal Input", "Noisy Input", "Adversarial Input", "Edge Case"])
    
    # Explanations for each test type
    test_explanations = {
        "Normal Input": "Standard input the model was trained on",
        "Noisy Input": "Input with random noise that could affect classification",
        "Adversarial Input": "Specifically crafted input designed to fool the model",
        "Edge Case": "Unusual inputs at the boundary of the model's training distribution"
    }
    
    st.info(test_explanations[test_type])
    
    # Set noise level based on test type
    if test_type == "Normal Input":
        noise_level = 0.05
    elif test_type == "Noisy Input":
        noise_level = 0.3
    elif test_type == "Adversarial Input":
        noise_level = 0.8
    else:  # Edge Case
        noise_level = 0.5
    
    # Image selection based on test type
    image_options = {
        "Normal Input": "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=400&h=300&fit=crop",
        "Noisy Input": "https://images.unsplash.com/photo-1561037404-61cd46aa615b?w=400&h=300&fit=crop",
        "Adversarial Input": "https://images.unsplash.com/photo-1543852786-1cf6624b9987?w=400&h=300&fit=crop",
        "Edge Case": "https://images.unsplash.com/photo-1550414913-a6fad9a5e918?w=400&h=300&fit=crop"
    }
    
    # Show selected image
    st.image(image_options[test_type], use_container_width=True)
    
    if st.button("Run Robustness Test"):
        with st.spinner("Testing model robustness..."):
            # Run multiple iterations to test stability
            iterations = 5
            results = []
            
            for i in range(iterations):
                # Base confidence varies by test type
                if test_type == "Normal Input":
                    base_confidence = 0.9
                elif test_type == "Noisy Input":
                    base_confidence = 0.7
                elif test_type == "Adversarial Input":
                    base_confidence = 0.6
                else:  # Edge Case
                    base_confidence = 0.55
                
                # Get classification
                classification, confidence = simulate_classification(base_confidence, noise_level)
                results.append((classification, confidence))
            
            # Show results
            st.success(f"Robustness test completed with {iterations} iterations.")
            
            # Process results
            class_counts = {}
            avg_confidence = 0
            
            for cls, conf in results:
                class_counts[cls] = class_counts.get(cls, 0) + 1
                avg_confidence += conf
            
            avg_confidence /= iterations
            
            # Calculate consistency
            consistency = max(class_counts.values()) / iterations if class_counts else 0
            
            # Visualize results
            st.markdown("#### Test Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Classification Consistency", f"{consistency:.0%}")
                
                # Display prediction distribution
                class_dist = pd.DataFrame({
                    'Class': list(class_counts.keys()),
                    'Count': list(class_counts.values())
                })
                
                fig = px.pie(class_dist, names='Class', values='Count', 
                            title='Prediction Distribution',
                            hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Average Confidence", f"{avg_confidence:.0%}")
                
                # Individual run results
                run_df = pd.DataFrame(results, columns=['Classification', 'Confidence'])
                run_df['Run'] = range(1, iterations + 1)
                
                fig = px.line(run_df, x='Run', y='Confidence', color='Classification',
                            title='Confidence Per Run',
                            markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            st.markdown("#### Robustness Analysis")
            
            if consistency >= 0.9:
                consistency_eval = "Excellent"
            elif consistency >= 0.7:
                consistency_eval = "Good"
            elif consistency >= 0.5:
                consistency_eval = "Fair"
            else:
                consistency_eval = "Poor"
                
            if avg_confidence >= 0.8:
                confidence_eval = "High"
            elif avg_confidence >= 0.6:
                confidence_eval = "Moderate"
            else:
                confidence_eval = "Low"
            
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.markdown(f"""
            **Model Performance Summary**:
            - Classification consistency: **{consistency_eval}** ({consistency:.0%})
            - Prediction confidence: **{confidence_eval}** ({avg_confidence:.0%})
            
            **Analysis**: The model shows {'high' if consistency > 0.7 else 'concerning'} variability 
            under this {test_type.lower()} scenario. {'This indicates good robustness.' if consistency > 0.7 else 'This suggests more robustness improvements are needed.'}
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommendations based on results
            st.markdown("#### Recommended Actions")
            
            if test_type == "Adversarial Input" and consistency < 0.8:
                st.info("🛡️ Implement adversarial training techniques")
                st.info("🔍 Add adversarial detection mechanisms")
            
            if test_type == "Noisy Input" and consistency < 0.8:
                st.info("🔊 Improve noise resistance through data augmentation")
                st.info("⚖️ Consider ensemble methods to increase stability")
            
            if test_type == "Edge Case" and consistency < 0.8:
                st.info("📊 Expand training data to include more edge cases")
                st.info("⚠️ Implement confidence thresholds to detect uncertain predictions")
            
            if avg_confidence < 0.7:
                st.info("❗ Add uncertainty quantification to flag low-confidence predictions")
    
    st.markdown("</div>", unsafe_allow_html=True)

def governance_demo():
    st.markdown("<div class='dimension-card'>", unsafe_allow_html=True)
    st.markdown("## 📋 Governance")
    st.markdown("Governance incorporates best practices throughout the AI supply chain.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?q=80&w=1200", 
                 caption="Proper governance ensures responsible AI across the entire pipeline", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Why it matters
        - Establishes accountability
        - Ensures regulatory compliance
        - Manages risks systematically
        - Creates documented processes
        """)
    
    st.markdown("### Interactive Example: AI Governance Framework")
    
    st.write("""
    This demonstration shows key elements of an AI governance framework and how they
    interact to ensure responsible AI practices throughout an organization.
    """)
    
    # Governance pillars
    pillars = {
        "Strategy & Leadership": [
            "Executive accountability for AI ethics",
            "Organizational principles for responsible AI",
            "Investment in responsible AI capabilities"
        ],
        "Risk Management": [
            "Risk assessment methodologies",
            "Mitigation strategies",
            "Continuous monitoring processes"
        ],
        "People & Training": [
            "Ethics training for AI teams",
            "Clear roles and responsibilities",
            "Cross-functional expertise"
        ],
        "Processes & Tools": [
            "Documentation requirements",
            "Impact assessment workflows",
            "Testing and validation protocols"
        ],
        "Monitoring & Reporting": [
            "Key metrics for responsible AI",
            "Audit procedures",
            "Transparent reporting frameworks"
        ]
    }
    
    # Select governance pillar to explore
    selected_pillar = st.selectbox("Select Governance Pillar to Explore:", list(pillars.keys()))
    
    st.markdown(f"#### {selected_pillar}")
    
    for practice in pillars[selected_pillar]:
        st.markdown(f"- {practice}")
    
    # Governance maturity assessment
    st.markdown("### Governance Maturity Assessment")
    st.write("Evaluate your organization's AI governance maturity:")
    
    assessment_areas = [
        "Leadership commitment to responsible AI",
        "Documented AI ethics principles",
        "AI risk management processes",
        "Cross-functional AI review board",
        "Employee training on AI ethics",
        "Monitoring of AI systems in production",
        "Incident response procedures",
        "Compliance with regulations",
        "Transparent AI documentation"
    ]
    
    maturity_scores = {}
    
    for area in assessment_areas:
        maturity_scores[area] = st.select_slider(
            area,
            options=["Not Started", "Initial", "Developing", "Established", "Leading"],
            value="Developing"
        )
    
    if st.button("Generate Maturity Report"):
        with st.spinner("Analyzing governance maturity..."):
            time.sleep(1)
            
            # Convert ratings to numeric scores
            score_mapping = {
                "Not Started": 0,
                "Initial": 1,
                "Developing": 2, 
                "Established": 3,
                "Leading": 4
            }
            
            numeric_scores = [score_mapping[v] for v in maturity_scores.values()]
            average_score = sum(numeric_scores) / len(numeric_scores)
            
            # Categorize overall maturity
            if average_score < 1:
                maturity_level = "Early Stage"
                recommendation_level = "Fundamental"
            elif average_score < 2:
                maturity_level = "Developing"
                recommendation_level = "Intermediate"
            elif average_score < 3:
                maturity_level = "Established"
                recommendation_level = "Advanced"
            else:
                maturity_level = "Leading"
                recommendation_level = "Excellence"
                
            # Create radar chart
            categories = list(maturity_scores.keys())
            values = [score_mapping[v] for v in maturity_scores.values()]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current State'
            ))
            
            # Add benchmark comparison
            benchmark_values = [3, 3, 2, 2, 3, 2, 2, 3, 2]  # Example industry benchmark
            
            fig.add_trace(go.Scatterpolar(
                r=benchmark_values,
                theta=categories,
                fill='toself',
                name='Industry Benchmark'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 4]
                    )),
                showlegend=True,
                title="AI Governance Maturity Assessment"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"Your organization's AI governance maturity level: **{maturity_level}**")
            
            # Show recommendations based on maturity level
            st.markdown("### Key Recommendations")
            
            if recommendation_level == "Fundamental":
                recommendations = [
                    "Establish executive ownership for responsible AI",
                    "Develop foundational AI ethics principles",
                    "Create basic documentation templates for AI systems",
                    "Implement preliminary risk assessment procedures"
                ]
            elif recommendation_level == "Intermediate":
                recommendations = [
                    "Formalize AI governance structures across departments",
                    "Develop comprehensive training on responsible AI",
                    "Implement regular AI system audits",
                    "Establish incident response procedures"
                ]
            elif recommendation_level == "Advanced":
                recommendations = [
                    "Integrate responsible AI metrics into executive dashboards",
                    "Develop advanced testing for bias and fairness",
                    "Implement continuous monitoring of production AI systems",
                    "Create transparent reporting on AI system performance"
                ]
            else:  # Excellence
                recommendations = [
                    "Lead industry standards development for responsible AI",
                    "Implement cutting-edge explainability techniques",
                    "Develop advanced governance automation tools",
                    "Create center of excellence for responsible AI practices"
                ]
                
            for i, rec in enumerate(recommendations):
                st.info(f"{i+1}. {rec}")
                
            # Show areas for improvement
            lowest_areas = sorted([(area, score) for area, score in maturity_scores.items()], 
                                key=lambda x: score_mapping[x[1]])[:3]
            
            st.markdown("### Priority Areas for Improvement")
            
            for area, score in lowest_areas:
                st.warning(f"**{area}**: Currently at '{score}' level")
    
    st.markdown("</div>", unsafe_allow_html=True)

def transparency_demo():
    st.markdown("<div class='dimension-card'>", unsafe_allow_html=True)
    st.markdown("## 🔎 Transparency")
    st.markdown("Transparency enables stakeholders to make informed choices about their engagement with AI systems.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1586892477838-2b96e85e0f96?q=80&w=1200", 
                 caption="Transparent AI empowers users to understand how systems work", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Why it matters
        - Builds trust with users
        - Enables informed consent
        - Supports accountability
        - Facilitates regulatory compliance
        """)
    
    st.markdown("### Interactive Example: AI System Documentation")
    
    st.write("""
    This demonstration shows how AI system documentation can provide transparency to users
    and stakeholders. Explore the different components of an AI transparency framework.
    """)
    
    # Sample AI system documentation
    st.markdown("#### Sample AI System Documentation")
    
    doc_sections = ["System Purpose", "Data Usage", "Model Information", "Performance Metrics", "Limitations"]
    selected_section = st.selectbox("Explore Documentation Section:", doc_sections)
    
    if selected_section == "System Purpose":
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        ### System Purpose and Scope
        
        **Primary Purpose**: This AI system evaluates loan applications to predict creditworthiness 
        and recommend approval decisions.
        
        **Intended Users**: Financial institution loan officers and credit analysts.
        
        **Use Context**: To be used as a decision support tool, not as the sole decision maker for 
        loan approvals. Human review is required for all decisions.
        
        **Success Criteria**: The system aims to improve efficiency by 30% while maintaining or 
        improving accuracy compared to manual assessments.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif selected_section == "Data Usage":
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        ### Data Usage Information
        
        **Training Data Sources**:
        - Historical loan applications (2015-2022)
        - Credit bureau reports
        - Internal customer transaction history
        
        **Data Preprocessing**:
        - Missing value imputation using [specific method]
        - Outlier treatment by [specific approach]
        - Feature normalization
        
        **Bias Mitigation Efforts**:
        - Demographic parity constraints applied
        - Balanced sampling across demographic groups
        - Regular bias audits conducted
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif selected_section == "Model Information":
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        ### Model Information
        
        **Model Type**: Ensemble of gradient boosted trees and neural network
        
        **Key Features Used**:
        - Credit score
        - Income level
        - Employment history
        - Debt-to-income ratio
        - Payment history
        
        **Update Frequency**: Model is retrained quarterly and validated before deployment
        
        **Human Oversight**: All model predictions above 0.7 or below 0.3 confidence are 
        automatically approved/denied, those between require human review
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif selected_section == "Performance Metrics":
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        ### Performance Metrics
        
        **Overall Accuracy**: 89.3%
        
        **False Positive Rate**: 7.2% (incorrect approvals)
        
        **False Negative Rate**: 4.5% (incorrect denials)
        
        **Demographic Performance**:
        - Group A: 90.1% accuracy
        - Group B: 88.7% accuracy
        - Group C: 87.5% accuracy
        - Group D: 86.9% accuracy
        
        **Confidence Calibration**: The model's confidence scores are well-calibrated,
        with 82% of predictions with >0.8 confidence being correct.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:  # Limitations
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("""
        ### Known Limitations and Restrictions
        
        **Performance Limitations**:
        - Lower accuracy for applicants with limited credit history
        - May not fully capture seasonal income patterns
        - Higher uncertainty for very high net worth individuals
        
        **Excluded Use Cases**:
        - Not designed for commercial loan evaluation
        - Not suitable for microloans under $5,000
        - Should not be used without human review
        
        **Monitoring and Safeguards**:
        - Drift detection alerts if data distribution changes significantly
        - Automatic suspension if demographic disparity exceeds 5%
        - Regular audits of denied applications
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a transparency checklist
    st.markdown("### Transparency Checklist for AI Systems")
    
    checklist_items = {
        "Purpose Clarity": "Clear statement of the system's purpose and use cases",
        "Data Transparency": "Documentation of data sources, collection methods, and processing",
        "Model Documentation": "Description of the model type, features, and training methodology",
        "Performance Disclosure": "Publication of performance metrics including accuracy and limitations",
        "Update Policies": "Clear information about how and when the system is updated",
        "Human Oversight": "Documentation of human involvement in decision processes",
        "Feedback Channels": "Mechanisms for users to report issues or provide feedback",
        "Decision Explanations": "Ability to explain individual decisions made by the system",
        "Demographic Impact": "Assessment and disclosure of impacts across different user groups"
    }
    
    # Example company that implements each practice
    example_companies = {
        "Purpose Clarity": "Google's AI Principles documentation",
        "Data Transparency": "Microsoft's Datasheets for Datasets",
        "Model Documentation": "OpenAI's Model Cards for GPT models",
        "Performance Disclosure": "IBM's AI FactSheets",
        "Update Policies": "Apple's Privacy Labels",
        "Human Oversight": "Meta's Oversight Board",
        "Feedback Channels": "Amazon's AI service customer feedback system",
        "Decision Explanations": "Salesforce's Einstein Explainability",
        "Demographic Impact": "LinkedIn's Fairness Toolkit"
    }
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        for item, description in checklist_items.items():
            st.checkbox(f"{item}: {description}", value=random.choice([True, True, True, False]))
    
    with col2:
        st.markdown("### Industry Examples")
        selected_practice = st.selectbox("Select practice:", list(example_companies.keys()))
        st.info(f"**Example**: {example_companies[selected_practice]}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main application
def main():
    st.title("🤖 Responsible AI Explorer")
    st.markdown("""
    <div class='info-box'>
    Explore the core dimensions of responsible AI through interactive examples. 
    Select a dimension below to learn more and interact with demonstrations.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for each dimension
    tabs = st.tabs([
        "Fairness", "Explainability", "Privacy & Security", "Safety",
        "Controllability", "Veracity & Robustness", "Governance", "Transparency"
    ])
    
    with tabs[0]:
        fairness_demo()
    
    with tabs[1]:
        explainability_demo()
    
    with tabs[2]:
        privacy_security_demo()
    
    with tabs[3]:
        safety_demo()
    
    with tabs[4]:
        controllability_demo()
    
    with tabs[5]:
        veracity_robustness_demo()
    
    with tabs[6]:
        governance_demo()
    
    with tabs[7]:
        transparency_demo()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Made with ❤️ for Responsible AI | © 2023</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import json
import utils.authenticate as authenticate
from utils.styles import load_css, sub_header

# Set page config
st.set_page_config(
    page_title="Responsible AI with AWS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# Standardized color scheme
COLORS = {
    'primary': '#FF9900',      # AWS Orange
    'secondary': '#232F3E',    # AWS Navy
    'success': '#3EB489',      # Green
    'warning': '#F2C94C',      # Yellow
    'danger': '#D13212',       # Red
    'info': '#0073BB',         # Blue
    'light': '#F8F9FA',        # Light gray
    'border': '#E9ECEF'        # Border gray
}

# Generate synthetic data
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    age = np.random.normal(40, 10, n_samples)
    income = 30000 + age * 1000 + np.random.normal(0, 10000, n_samples)
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.5, 0.5])
    income[gender == 'Female'] = income[gender == 'Female'] * 0.85
    ethnicity = np.random.choice(['Group A', 'Group B', 'Group C', 'Group D'], 
                               size=n_samples, p=[0.5, 0.2, 0.2, 0.1])
    
    credit_score = (700 + 0.5 * (age - 40) + 0.002 * (income - 70000) + 
                   np.random.normal(0, 50, n_samples))
    credit_score[gender == 'Female'] -= 10
    credit_score[ethnicity == 'Group C'] -= 15
    credit_score[ethnicity == 'Group D'] -= 20
    credit_score = np.clip(credit_score, 300, 850)
    
    approved = credit_score > 650
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'gender': gender,
        'ethnicity': ethnicity,
        'credit_score': credit_score,
        'approved': approved
    })
    
    return df

df = generate_synthetic_data()


def fairness_demo():
    """Fairness and bias detection with AWS SageMaker Clarify"""
    
    st.markdown(sub_header("Fairness & Bias Detection", "🎯"), unsafe_allow_html=True)
    
    st.write("""
    **Amazon SageMaker Clarify** helps detect bias in ML models and training data.
    It provides metrics like Disparate Impact and Statistical Parity to ensure fairness.
    """)
    
    # AWS Service Info
    with st.expander("🔧 AWS Services for Fairness"):
        st.markdown("""
        **Amazon SageMaker Clarify:**
        - Pre-training bias detection
        - Post-training bias metrics
        - Feature attribution (SHAP values)
        - Continuous monitoring with Model Monitor
        
        **Key Metrics:**
        - Disparate Impact (DI): Ratio of positive outcomes between groups
        - Statistical Parity Difference (SPD): Difference in approval rates
        - Class Imbalance (CI): Balance in training data
        """)
    
    # Analysis tabs
    tab1, tab2 = st.tabs(["Gender Analysis", "Ethnicity Analysis"])
    
    with tab1:
        st.markdown("### Loan Approval by Gender")
        
        gender_approval = df.groupby('gender')['approved'].mean().reset_index()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(gender_approval, x='gender', y='approved',
                         title='Approval Rate by Gender',
                         color='gender',
                         color_discrete_sequence=[COLORS['info'], COLORS['primary']])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            male_rate = gender_approval[gender_approval['gender'] == 'Male']['approved'].values[0]
            female_rate = gender_approval[gender_approval['gender'] == 'Female']['approved'].values[0]
            disparate_impact = female_rate / male_rate if male_rate > 0 else 0
            
            st.metric("Male Approval", f"{male_rate:.1%}")
            st.metric("Female Approval", f"{female_rate:.1%}")
            st.metric("Disparate Impact", f"{disparate_impact:.3f}",
                     help="Should be >= 0.8 for fairness")
        
        # Analysis box
        if disparate_impact >= 0.8:
            st.success(f"✅ Disparate Impact: {disparate_impact:.3f} (Acceptable)")
        else:
            st.warning(f"⚠️ Disparate Impact: {disparate_impact:.3f} (Below 0.8 threshold)")
        
        st.info("""
        **AWS Mitigation Strategies:**
        - Use SageMaker Clarify for pre-training bias detection
        - Apply fairness constraints during model training
        - Set up Model Monitor for continuous bias tracking
        - Use Amazon A2I for human review of borderline cases
        """)
    
    with tab2:
        st.markdown("### Loan Approval by Ethnicity")
        
        ethnicity_approval = df.groupby('ethnicity')['approved'].mean().reset_index()
        
        fig = px.bar(ethnicity_approval, x='ethnicity', y='approved',
                     title='Approval Rate by Ethnicity',
                     color='approved',
                     color_continuous_scale='RdYlGn')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        max_rate = ethnicity_approval['approved'].max()
        min_rate = ethnicity_approval['approved'].min()
        disparity = max_rate - min_rate
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Highest Rate", f"{max_rate:.1%}")
        col2.metric("Lowest Rate", f"{min_rate:.1%}")
        col3.metric("Disparity", f"{disparity:.1%}")
        
        if disparity > 0.1:
            st.warning(f"⚠️ Disparity of {disparity:.1%} detected across ethnic groups")
        else:
            st.success(f"✅ Disparity of {disparity:.1%} is within acceptable range")


def explainability_demo():
    """Explainability with SageMaker Clarify and Bedrock"""
    
    st.markdown(sub_header("Explainability & Transparency", "🔍"), unsafe_allow_html=True)
    
    st.write("""
    **Amazon SageMaker Clarify** provides SHAP values for feature attribution.
    **Amazon Bedrock** can generate natural language explanations of model decisions.
    """)
    
    # AWS Service Info
    with st.expander("🔧 AWS Services for Explainability"):
        st.markdown("""
        **Amazon SageMaker Clarify:**
        - SHAP (Shapley Additive Explanations) values
        - Global feature importance
        - Local explanations for individual predictions
        
        **Amazon Bedrock:**
        - Natural language explanations
        - Chain-of-thought reasoning
        - Citation support
        """)
    
    # Feature importance
    st.markdown("### Feature Importance Analysis")
    
    features_data = pd.DataFrame({
        'Feature': ['Credit Score', 'Income', 'Debt Ratio', 'Employment', 'Age'],
        'Importance': [0.35, 0.22, 0.18, 0.12, 0.13]
    })
    
    fig = px.bar(features_data, x='Importance', y='Feature',
                orientation='h',
                title='Global Feature Importance',
                color='Importance',
                color_continuous_scale='Blues')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("✅ Credit Score is the most important feature (35% importance)")
    
    # Code example
    st.markdown("### Implementation with SageMaker Clarify")
    st.code("""
# Configure SageMaker Clarify for explainability
from sagemaker import clarify

clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Run SHAP analysis
clarify_processor.run_explainability(
    data_config=data_config,
    model_config=model_config,
    explainability_config=clarify.ExplainabilityConfig(
        shap_config=clarify.SHAPConfig(
            baseline=[baseline_data],
            num_samples=100
        )
    )
)
    """, language="python")


def privacy_security_demo():
    """Privacy and security with AWS KMS, Macie, and IAM"""
    
    st.markdown(sub_header("Privacy & Security", "🔒"), unsafe_allow_html=True)
    
    st.write("""
    AWS provides comprehensive security services for ML workloads including
    encryption, access control, and PII detection.
    """)
    
    # AWS Service Info
    with st.expander("🔧 AWS Security Services"):
        st.markdown("""
        **AWS KMS:** Encryption for data at rest and in transit
        
        **Amazon Macie:** Discover and protect sensitive data (PII)
        
        **AWS IAM:** Fine-grained access control for ML resources
        
        **AWS CloudTrail:** Audit logging for all API calls
        
        **Amazon GuardDuty:** Threat detection for ML workloads
        """)
    
    tab1, tab2 = st.tabs(["Data Anonymization", "Access Control"])
    
    with tab1:
        st.markdown("### Data Anonymization Techniques")
        
        # Sample data
        sample_data = pd.DataFrame({
            'Name': ['John Smith', 'Jane Doe', 'Alex Johnson'],
            'SSN': ['123-45-6789', '987-65-4321', '555-12-3456'],
            'Email': ['john@example.com', 'jane@example.com', 'alex@example.com'],
            'Credit_Score': [720, 680, 750]
        })
        
        st.write("**Original Data:**")
        st.dataframe(sample_data, use_container_width=True)
        
        technique = st.selectbox("Select Technique:", 
                                ["Pseudonymization", "Masking", "Generalization"])
        
        if st.button("Apply Anonymization"):
            anonymized = sample_data.copy()
            
            if technique == "Pseudonymization":
                anonymized['Name'] = ['Person A', 'Person B', 'Person C']
                anonymized['SSN'] = ['XXX-XX-XXXX'] * 3
                anonymized['Email'] = ['user1@anon.com', 'user2@anon.com', 'user3@anon.com']
            elif technique == "Masking":
                anonymized['SSN'] = anonymized['SSN'].apply(lambda x: x[:3] + '-XX-' + x[-4:])
                anonymized['Email'] = anonymized['Email'].apply(lambda x: x[0] + '***' + x[x.find('@'):])
            else:  # Generalization
                anonymized['Name'] = ['[REDACTED]'] * 3
                anonymized['SSN'] = ['[REDACTED]'] * 3
                anonymized['Email'] = ['[REDACTED]'] * 3
                anonymized['Credit_Score'] = anonymized['Credit_Score'].apply(
                    lambda x: '700-750' if 700 <= x < 750 else '750-800'
                )
            
            st.write("**Anonymized Data:**")
            st.dataframe(anonymized, use_container_width=True)
            st.success(f"✅ {technique} applied successfully!")
    
    with tab2:
        st.markdown("### IAM Role-Based Access Control")
        
        role = st.selectbox("Select Role:", 
                           ["Data Scientist", "ML Engineer", "Data Analyst", "Auditor"])
        
        roles_info = {
            "Data Scientist": {
                "allowed": ["CreateTrainingJob", "CreateNotebookInstance", "CreateExperiment"],
                "denied": ["DeleteEndpoint (production)", "DeleteModel (production)"]
            },
            "ML Engineer": {
                "allowed": ["CreateEndpoint", "UpdateEndpoint", "CreateModel"],
                "denied": ["DeleteModel (without approval)", "DeleteBucket"]
            },
            "Data Analyst": {
                "allowed": ["DescribeEndpoint", "InvokeEndpoint", "GetMetricData"],
                "denied": ["CreateTrainingJob", "DeleteEndpoint", "PutObject"]
            },
            "Auditor": {
                "allowed": ["DescribeModel", "ListModels", "LookupEvents"],
                "denied": ["CreateTrainingJob", "DeleteModel", "PutObject"]
            }
        }
        
        info = roles_info[role]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**✅ Allowed Permissions:**")
            for perm in info['allowed']:
                st.write(f"- {perm}")
        
        with col2:
            st.error("**🚫 Denied Permissions:**")
            for perm in info['denied']:
                st.write(f"- {perm}")


def safety_guardrails_demo():
    """Safety with Amazon Bedrock Guardrails"""
    
    st.markdown(sub_header("Safety & Guardrails", "🛡️"), unsafe_allow_html=True)
    
    st.write("""
    **Amazon Bedrock Guardrails** implement safeguards for foundation model applications
    including content filtering, PII redaction, and topic restrictions.
    """)
    
    # AWS Service Info
    with st.expander("🔧 Amazon Bedrock Guardrails"):
        st.markdown("""
        **Content Filters:**
        - Hate speech detection
        - Violence detection
        - Sexual content filtering
        - Misconduct detection
        
        **Additional Features:**
        - PII redaction (email, phone, SSN)
        - Denied topics configuration
        - Custom word filters
        - Contextual grounding checks
        """)
    
    # Content testing
    st.markdown("### Test Content Safety")
    
    # Example test cases
    with st.expander("📝 Try These Example Inputs"):
        st.markdown("""
        **Safe Examples:**
        - "I'm interested in learning about machine learning best practices."
        - "How can I improve my Python programming skills?"
        - "Tell me about AWS security features."
        
        **Unsafe Examples (will be flagged):**
        - "I'm interested in learning how to rob a bank"
        - "How to hack into someone's account"
        - "I want to hurt someone"
        - "Teach me how to make illegal substances"
        """)
    
    user_input = st.text_area(
        "Enter text to analyze:",
        "I'm interested in learning about machine learning best practices.",
        height=100
    )
    
    if st.button("Analyze Content", type="primary"):
        with st.spinner("Analyzing with Bedrock Guardrails..."):
            time.sleep(1)
        
        # Simulate analysis with comprehensive harmful content detection
        harmful_patterns = {
            'violence': ['kill', 'murder', 'attack', 'assault', 'harm', 'hurt', 'shoot', 'stab', 'bomb', 'weapon'],
            'illegal': ['rob', 'steal', 'theft', 'fraud', 'scam', 'hack', 'break-in', 'burglar', 'illegal', 'crime'],
            'hate': ['hate', 'racist', 'discriminate', 'slur', 'bigot'],
            'self-harm': ['suicide', 'self-harm', 'cut myself', 'end my life'],
            'sexual': ['explicit', 'pornography', 'sexual assault'],
            'drugs': ['cocaine', 'heroin', 'meth', 'drug deal']
        }
        
        detected = []
        detected_categories = []
        input_lower = user_input.lower()
        
        for category, words in harmful_patterns.items():
            for word in words:
                if word in input_lower:
                    detected.append(word)
                    if category not in detected_categories:
                        detected_categories.append(category)
        
        # Calculate risk score based on detected harmful content
        if detected:
            # Base risk from number of harmful words
            base_risk = min(0.9, len(detected) * 0.25)
            # Additional risk from multiple categories
            category_risk = len(detected_categories) * 0.15
            risk_score = min(1.0, base_risk + category_risk)
        else:
            # Low baseline risk for clean content
            risk_score = random.uniform(0.05, 0.15)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_score:.1%}")
        
        with col2:
            if risk_score < 0.3:
                st.metric("Status", "SAFE", delta="✅")
            elif risk_score < 0.7:
                st.metric("Status", "REVIEW", delta="⚠️")
            else:
                st.metric("Status", "BLOCKED", delta="🚫")
        
        with col3:
            st.metric("Flagged Terms", len(detected))
        
        # Show detected categories
        if detected_categories:
            st.warning(f"**⚠️ Detected Categories:** {', '.join(detected_categories).title()}")
            
            with st.expander("🔍 View Flagged Terms"):
                st.write(f"**Flagged words:** {', '.join(set(detected))}")
        
        # Status messages
        if risk_score < 0.3:
            st.success("✅ Content approved - no safety concerns detected")
        elif risk_score < 0.7:
            st.warning("⚠️ Content requires human review - potential safety concerns detected")
        else:
            st.error("🚫 Content blocked due to safety concerns - violates content policy")
    
    # Configuration
    st.markdown("### Configure Guardrails")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Content Filters:**")
        hate_filter = st.select_slider("Hate Speech", ["NONE", "LOW", "MEDIUM", "HIGH"], value="HIGH")
        violence_filter = st.select_slider("Violence", ["NONE", "LOW", "MEDIUM", "HIGH"], value="HIGH")
    
    with col2:
        st.write("**Additional Protection:**")
        pii_redaction = st.checkbox("Enable PII Redaction", value=True)
        profanity_filter = st.checkbox("Enable Profanity Filter", value=True)
    
    if st.button("Save Configuration"):
        st.success("✅ Guardrail configuration saved!")


def controllability_demo():
    """Controllability with Amazon A2I"""
    
    st.markdown(sub_header("Controllability & Human Oversight", "🎮"), unsafe_allow_html=True)
    
    st.write("""
    **Amazon Augmented AI (A2I)** enables human review workflows for ML predictions,
    ensuring human oversight for critical decisions.
    """)
    
    with st.expander("🔧 AWS Services for Controllability"):
        st.markdown("""
        **Amazon Augmented AI (A2I):**
        - Human review workflows
        - Confidence-based routing
        - Custom review interfaces
        
        **SageMaker Model Monitor:**
        - Data quality monitoring
        - Model drift detection
        - Bias drift tracking
        """)
    
    st.markdown("### Human Review Configuration")
    
    confidence_threshold = st.slider(
        "Confidence Threshold for Human Review",
        0.0, 1.0, 0.7, 0.05,
        help="Predictions below this will be sent for human review"
    )
    
    # Simulate predictions
    np.random.seed(42)
    predictions = pd.DataFrame({
        'ID': [f'PRED-{i:03d}' for i in range(20)],
        'Confidence': np.random.beta(8, 2, 20)
    })
    predictions['Needs_Review'] = predictions['Confidence'] < confidence_threshold
    
    review_count = predictions['Needs_Review'].sum()
    auto_count = (~predictions['Needs_Review']).sum()
    
    col1, col2 = st.columns(2)
    col1.metric("Automatic Decisions", auto_count, f"{auto_count/20:.0%}")
    col2.metric("Human Review Required", review_count, f"{review_count/20:.0%}")
    
    fig = px.scatter(predictions, x=predictions.index, y='Confidence',
                    color='Needs_Review',
                    title='Prediction Confidence Distribution',
                    color_discrete_map={True: COLORS['warning'], False: COLORS['success']})
    fig.add_hline(y=confidence_threshold, line_dash="dash", line_color="red")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"✅ {review_count} predictions will be sent for human review")


def veracity_robustness_demo():
    """Veracity and robustness testing"""
    
    st.markdown(sub_header("Veracity & Robustness", "✅"), unsafe_allow_html=True)
    
    st.write("""
    Test model performance under different conditions to ensure reliability and accuracy.
    """)
    
    with st.expander("🔧 AWS Testing Tools"):
        st.markdown("""
        **SageMaker Model Evaluation:**
        - Automated testing metrics
        - Holdout validation
        - Cross-validation support
        
        **AWS Fault Injection Simulator:**
        - Chaos engineering
        - Stress testing
        - Failure scenario simulation
        """)
    
    st.markdown("### Model Performance Testing")
    
    test_scenarios = pd.DataFrame({
        'Scenario': ['Clean Data', 'Noisy Data', 'Missing Values', 'Outliers'],
        'Accuracy': [0.94, 0.89, 0.91, 0.88]
    })
    
    fig = px.bar(test_scenarios, x='Scenario', y='Accuracy',
                title='Model Performance Across Conditions',
                color='Accuracy',
                color_continuous_scale='RdYlGn',
                range_color=[0.8, 1.0])
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    best = test_scenarios['Accuracy'].max()
    worst = test_scenarios['Accuracy'].min()
    degradation = best - worst
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Accuracy", f"{best:.1%}")
    col2.metric("Worst Accuracy", f"{worst:.1%}")
    col3.metric("Max Degradation", f"{degradation:.1%}")
    
    if degradation < 0.1:
        st.success(f"✅ Model shows good robustness (degradation: {degradation:.1%})")
    else:
        st.warning(f"⚠️ Model degradation of {degradation:.1%} may need attention")


def governance_demo():
    """Governance with SageMaker Model Cards"""
    
    st.markdown(sub_header("Governance & Compliance", "📋"), unsafe_allow_html=True)
    
    st.write("""
    **Amazon SageMaker Model Cards** provide centralized documentation for ML models,
    supporting governance and compliance requirements.
    """)
    
    with st.expander("🔧 AWS Governance Services"):
        st.markdown("""
        **SageMaker Model Cards:**
        - Model documentation
        - Intended use and limitations
        - Performance metrics
        - Training details
        
        **SageMaker Model Registry:**
        - Version control
        - Approval workflows
        - Lineage tracking
        
        **AWS CloudTrail:**
        - API logging
        - Audit trails
        - Compliance support
        """)
    
    st.markdown("### Sample Model Card")
    
    with st.expander("📄 Fraud Detection Model v2.1", expanded=True):
        st.write("**Status:** Production | **Risk:** Medium")
        
        st.write("**Intended Use:**")
        st.write("Real-time fraud detection for credit card transactions")
        
        st.write("**Performance Metrics:**")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [0.94, 0.92, 0.89, 0.905]
        })
        st.dataframe(metrics_df, use_container_width=True)
        
        st.write("**Known Limitations:**")
        st.write("- Not suitable for business accounts")
        st.write("- Requires 3+ months transaction history")
        st.write("- Higher false positives for international transactions")
    
    st.markdown("### Create Model Card")
    st.code("""
# Create SageMaker Model Card
import boto3

sagemaker = boto3.client('sagemaker')

response = sagemaker.create_model_card(
    ModelCardName='fraud-detection-v2',
    Content={
        'model_overview': {
            'model_name': 'fraud-detection',
            'model_version': '2.1.0',
            'problem_type': 'Binary Classification'
        },
        'intended_uses': {
            'purpose_of_model': 'Fraud detection',
            'intended_uses': 'Decision support'
        },
        'training_details': {
            'training_data': 's3://ml-data/fraud/',
            'training_metrics': [
                {'name': 'accuracy', 'value': 0.94}
            ]
        }
    },
    ModelCardStatus='Approved'
)
    """, language="python")


def transparency_demo():
    """Transparency and documentation"""
    
    st.markdown(sub_header("Transparency", "🔎"), unsafe_allow_html=True)
    
    st.write("""
    Transparency enables stakeholders to understand how AI systems work and make informed decisions.
    """)
    
    st.markdown("### AI System Documentation")
    
    doc_sections = {
        "System Purpose": """
**Primary Purpose:** Evaluate loan applications to predict creditworthiness

**Intended Users:** Financial institution loan officers

**Use Context:** Decision support tool with required human review
        """,
        "Data Usage": """
**Training Data:**
- Historical loan applications (2015-2023)
- Credit bureau reports
- Customer transaction history

**Bias Mitigation:**
- Demographic parity constraints
- Balanced sampling
- Regular bias audits
        """,
        "Performance": """
**Overall Accuracy:** 89.3%

**False Positive Rate:** 7.2%

**False Negative Rate:** 4.5%

**Fairness:** Demographic parity within 5%
        """
    }
    
    selected = st.selectbox("Select Documentation Section:", list(doc_sections.keys()))
    
    st.info(doc_sections[selected])
    
    st.success("✅ Comprehensive documentation builds trust and enables informed decisions")


# Main application
def main():
    # Hero section with gradient header
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>
                🤖 Responsible AI with AWS
            </h1>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                Build trustworthy AI systems using AWS AI/ML services and best practices.
                Explore fairness, explainability, privacy, and safety with hands-on examples.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Principles overview
    st.markdown(sub_header("8 Principles of Responsible AI", "🌟", "aws"), unsafe_allow_html=True)
    
    # Quick stats with enhanced cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 0.75rem; text-align: center;
                        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
                        transition: transform 0.3s ease;'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🎯</div>
                <div style='color: white; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem;'>Fairness</div>
                <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>SageMaker Clarify</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 0.75rem; text-align: center;
                        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🔍</div>
                <div style='color: white; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem;'>Explainability</div>
                <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>SHAP & Bedrock</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 0.75rem; text-align: center;
                        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🔒</div>
                <div style='color: white; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem;'>Privacy</div>
                <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>KMS & Macie</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 1.5rem; border-radius: 0.75rem; text-align: center;
                        box-shadow: 0 4px 12px rgba(67, 233, 123, 0.3);'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🛡️</div>
                <div style='color: white; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem;'>Safety</div>
                <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Bedrock Guardrails</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional principles
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1.5rem; border-radius: 0.75rem; text-align: center;
                        box-shadow: 0 4px 12px rgba(250, 112, 154, 0.3);'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🎮</div>
                <div style='color: white; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem;'>Controllability</div>
                <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Amazon A2I</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); 
                        padding: 1.5rem; border-radius: 0.75rem; text-align: center;
                        box-shadow: 0 4px 12px rgba(48, 207, 208, 0.3);'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>✅</div>
                <div style='color: white; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem;'>Veracity</div>
                <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Model Testing</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 1.5rem; border-radius: 0.75rem; text-align: center;
                        box-shadow: 0 4px 12px rgba(168, 237, 234, 0.3);'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>📋</div>
                <div style='color: #232F3E; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem;'>Governance</div>
                <div style='color: rgba(35, 47, 62, 0.7); font-size: 0.9rem;'>Model Cards</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                        padding: 1.5rem; border-radius: 0.75rem; text-align: center;
                        box-shadow: 0 4px 12px rgba(255, 236, 210, 0.3);'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🔎</div>
                <div style='color: #232F3E; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.25rem;'>Transparency</div>
                <div style='color: rgba(35, 47, 62, 0.7); font-size: 0.9rem;'>Documentation</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Introduction section
    st.markdown(sub_header("Why Responsible AI Matters", "🌍"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: #f8f9fb; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid #FF9900; height: 100%;'>
            <h4 style='color: #232F3E; margin-top: 0;'>🎯 Business Impact</h4>
            <p style='color: #5A6D87; margin-bottom: 0;'>
                Responsible AI builds trust with customers, reduces legal risks, 
                and ensures sustainable AI adoption across your organization.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: #f8f9fb; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid #0073BB; height: 100%;'>
            <h4 style='color: #232F3E; margin-top: 0;'>⚖️ Ethical Considerations</h4>
            <p style='color: #5A6D87; margin-bottom: 0;'>
                AI systems impact real people's lives. Fairness, transparency, 
                and accountability are not optional—they're essential.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: #f8f9fb; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid #3EB489; height: 100%;'>
            <h4 style='color: #232F3E; margin-top: 0;'>📜 Regulatory Compliance</h4>
            <p style='color: #5A6D87; margin-bottom: 0;'>
                Meet requirements from GDPR, AI Act, and other regulations 
                with proper documentation and bias mitigation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive demo selector
    st.markdown(sub_header("Explore Responsible AI Principles", "🚀", "aws"), unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1.5rem;'>
        <p style='margin: 0; color: #232F3E; font-size: 1.05rem;'>
            👇 Select a tab below to explore interactive demonstrations of each principle. 
            Each section includes AWS service examples, code samples, and hands-on simulations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content tabs
    tabs = st.tabs([
        "🎯 Fairness",
        "🔍 Explainability",
        "🔒 Privacy & Security",
        "🛡️ Safety & Guardrails",
        "🎮 Controllability",
        "✅ Veracity & Robustness",
        "📋 Governance",
        "🔎 Transparency"
    ])
    
    with tabs[0]:
        fairness_demo()
    
    with tabs[1]:
        explainability_demo()
    
    with tabs[2]:
        privacy_security_demo()
    
    with tabs[3]:
        safety_guardrails_demo()
    
    with tabs[4]:
        controllability_demo()
    
    with tabs[5]:
        veracity_robustness_demo()
    
    with tabs[6]:
        governance_demo()
    
    with tabs[7]:
        transparency_demo()
    
    # Summary section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(sub_header("Key Takeaways", "💡", "minimal"), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### AWS Services for Responsible AI
        
        - **Amazon SageMaker Clarify**: Detect and mitigate bias, explain predictions
        - **Amazon Bedrock Guardrails**: Content filtering, PII redaction, safety controls
        - **Amazon Augmented AI (A2I)**: Human review workflows for critical decisions
        - **Amazon Macie**: Discover and protect sensitive data
        - **AWS KMS**: Encryption for data security
        - **SageMaker Model Cards**: Document models for governance
        - **SageMaker Model Monitor**: Continuous monitoring for drift and bias
        """)
    
    with col2:
        st.markdown("""
        ### Best Practices
        
        1. **Assess bias** in training data before model development
        2. **Implement fairness constraints** during model training
        3. **Use explainability tools** to understand model decisions
        4. **Apply guardrails** to prevent harmful outputs
        5. **Enable human oversight** for high-stakes decisions
        6. **Monitor continuously** for drift and degradation
        7. **Document thoroughly** with model cards
        8. **Test robustness** under various conditions
        """)
    
    # Call to action
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 1rem; margin-top: 2rem;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center;'>
            <h3 style='color: white; margin: 0 0 1rem 0;'>Ready to Build Responsible AI?</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0 0 1rem 0;'>
                Explore AWS AI/ML services and implement responsible AI practices in your projects.
            </p>
            <div style='display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;'>
                <a href='https://aws.amazon.com/sagemaker/clarify/' target='_blank' 
                   style='background: white; color: #667eea; padding: 0.75rem 1.5rem; 
                          border-radius: 0.5rem; text-decoration: none; font-weight: 600;
                          box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                    Learn More About SageMaker Clarify
                </a>
                <a href='https://aws.amazon.com/bedrock/guardrails/' target='_blank' 
                   style='background: rgba(255,255,255,0.2); color: white; padding: 0.75rem 1.5rem; 
                          border-radius: 0.5rem; text-decoration: none; font-weight: 600;
                          border: 2px solid white;'>
                    Explore Bedrock Guardrails
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("© 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.")


if __name__ == "__main__":
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        is_authenticated = authenticate.login()
        if is_authenticated:
            main()

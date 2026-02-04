
import streamlit as st
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from utils.styles import load_css, custom_header, create_footer
import utils.common as common
import utils.authenticate as authenticate

# Set page configuration
st.set_page_config(
    page_title="AWS AI Practitioner - Domain 4 & 5",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

common.initialize_session_state()

# Initialize session state
def initialize_session_state():
    if "knowledge_check_progress" not in st.session_state:
        st.session_state.knowledge_check_progress = 0
    
    if "knowledge_check_score" not in st.session_state:
        st.session_state.knowledge_check_score = 0
        
    if "knowledge_check_answers" not in st.session_state:
        st.session_state.knowledge_check_answers = {}

    if "knowledge_check_feedback" not in st.session_state:
        st.session_state.knowledge_check_feedback = {}

# Reset the session
def reset_session():
    for key in list(st.session_state.keys()):
        if key != "session_id":
            del st.session_state[key]
    initialize_session_state()
    st.rerun()

# Reset knowledge check
def reset_knowledge_check():
    st.session_state.knowledge_check_progress = 0
    st.session_state.knowledge_check_score = 0
    st.session_state.knowledge_check_answers = {}
    st.session_state.knowledge_check_feedback = {}
    st.rerun()

# Home Tab Content
def home_tab():
    st.markdown("# Guidelines for Responsible AI and Security, Compliance, and Governance for AI Solutions")
    
    st.markdown("### Program Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### Session 1")
        st.markdown("- Kickoff")
        st.markdown("- Domain 1: Fundamentals of AI and ML")
    
    with col2:
        st.markdown("#### Session 2")
        st.markdown("- Domain 2: Fundamentals of Generative AI")
    
    with col3:
        st.markdown("#### Session 3")
        st.markdown("- Domain 3: Applications of Foundation Models")
    
    with col4:
        st.markdown("#### Session 4")
        st.markdown("- Domain 4: Guidelines for Responsible AI")
        st.markdown("- Domain 5: Security, Compliance, and Governance for AI Solutions")
    
    st.divider()
    
    st.markdown("### Today's Learning Outcomes")
    st.markdown("""
    During this session, we will cover:
    - **Task Statement 4.1**: Explain the development of AI systems that are responsible
    - **Task Statement 4.2**: Recognize the importance of transparent and explainable models
    - **Task Statement 5.1**: Explain methods to secure AI systems
    - **Task Statement 5.2**: Recognize governance and compliance regulations for AI systems
    """)
    
    st.divider()
    
    st.markdown("### Final Digital Training Curriculum")
    st.markdown("""
    Do your best to finish these courses. Take notes and dive deep on topics as needed:
    
    **AWS Skill Builder Learning Plan Courses:**
    - Amazon Bedrock Getting Started
    - Exam Prep Standard Course: AWS Certified AI Practitioner
    
    **Enhanced Exam Prep Plan (Optional):**
    - Finish – CloudQuest: Generative AI
    - Amazon Bedrock Getting Started
    - Complete Lab – Getting Started with Amazon Comprehend: Custom Classification
    - Exam Prep Enhanced Course: AWS Certified AI Practitioner
    - Complete the labs and Official Pretest
    """)
    
    st.info("Access time for most of the digital resources is limited to the duration of this program. Make sure to complete them before the program ends.")
    
# Responsible AI Tab Content
def responsible_ai_tab():
    st.markdown("# Guidelines for Responsible AI")
    st.markdown("## Task Statement 4.1: Explain the development of AI systems that are responsible")
    
    # What is responsible AI
    st.markdown("### What is Responsible AI?")
    st.markdown("""
    Responsible AI (Artificial Intelligence) refers to the ethical and trustworthy development, 
    deployment, and use of AI systems. It encompasses a set of principles, practices, and governance 
    frameworks aimed at ensuring AI systems are designed and used in a way that benefits society, 
    respects human rights, and mitigates potential risks or harms.
    """)
    
    cols = st.columns(4)
    with cols[0]:
        st.markdown("#### FAIRNESS")
        st.markdown("Considering impacts on different groups of stakeholders")
    
    with cols[1]:
        st.markdown("#### EXPLAINABILITY")
        st.markdown("Understanding and evaluating system outputs")
    
    with cols[2]:
        st.markdown("#### CONTROLLABILITY")
        st.markdown("Having mechanisms to monitor and steer AI system behavior")
    
    with cols[3]:
        st.markdown("#### PRIVACY & SECURITY")
        st.markdown("Appropriately obtaining, using and protecting data and models")
    
    cols = st.columns(4)
    with cols[0]:
        st.markdown("#### GOVERNANCE")
        st.markdown("Incorporating best practices into the AI supply chain, including providers and deployers")
    
    with cols[1]:
        st.markdown("#### TRANSPARENCY")
        st.markdown("Enabling stakeholders to make informed choices about their engagement with an AI system")
    
    with cols[2]:
        st.markdown("#### SAFETY")
        st.markdown("Preventing harmful system output and misuse")
    
    with cols[3]:
        st.markdown("#### VERACITY & ROBUSTNESS")
        st.markdown("Achieving correct system outputs, even with unexpected or adversarial inputs")
    
    # Dataset bias section
    st.markdown("### Dataset Bias")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Dataset bias** refers to the systematic skew or imbalance in the data used to train a machine learning model.
        - This can happen when the data collection process is not representative of the real-world distribution of the problem being modeled.
        
        **Common types of dataset bias include:**
        - **Sampling bias**: The data collected does not represent the true population.
        - **Historical bias**: The data reflects past biases and inequities in society.
        - **Measurement bias**: The data collection process itself introduces systematic errors.
        
        **How to identify imbalance:**
        - Calculate the ratio of the smaller class vs total data
        - Visualize the distribution of data across different groups
        - Use statistical tests to detect skew in representation
        """)
    
    with col2:
        # Create a sample chart showing dataset bias
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['Male', 'Female', 'Non-binary']
        dataset1 = [75, 23, 2]
        dataset2 = [34, 33, 33]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, dataset1, width, label='Biased Dataset')
        ax.bar(x + width/2, dataset2, width, label='Balanced Dataset')
        
        ax.set_ylabel('Percentage')
        ax.set_title('Example of Gender Distribution in Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        st.pyplot(fig)
        st.caption("Example visualization showing gender distribution in biased vs balanced datasets.")
    
    # Bias vs Variance
    st.markdown("### Model Generalization Problems – Bias vs Variance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Underfitting (High Bias)")
        st.markdown("""
        The model is too simple to capture the input/output relationship.
        
        It will have poor training and test performance.
        
        **How to fix:**
        - Increase model complexity
        - Add more relevant features
        - Reduce regularization
        - Use a more sophisticated model architecture
        """)
        
        # Create simple visualization for underfitting
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Generate data points
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        
        # Plot data points
        ax.scatter(x, y, alpha=0.5)
        
        # Underfit line (linear model)
        ax.plot(x, 0.5*x + 2, 'r-', linewidth=2)
        
        ax.set_title("Underfitting Example")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### Overfitting (High Variance)")
        st.markdown("""
        The model picks up the noise instead of the underlying relationship.
        
        We will see good scores in training data but poor in test data.
        
        **How to fix:**
        - Reduce model flexibility
        - Use fewer features
        - Add more training data
        - Increase regularization
        - Early stopping
        """)
        
        # Create simple visualization for overfitting
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Generate data points
        x = np.linspace(0, 10, 20)
        y = np.sin(x) + np.random.normal(0, 0.2, 20)
        
        # Plot data points
        ax.scatter(x, y, alpha=0.5)
        
        # Overfit line (high-degree polynomial)
        p = np.poly1d(np.polyfit(x, y, 15))
        ax.plot(np.linspace(0, 10, 100), p(np.linspace(0, 10, 100)), 'r-', linewidth=2)
        
        ax.set_title("Overfitting Example")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        st.pyplot(fig)
    
    with col3:
        st.markdown("#### Appropriate Fitting (Low Bias, Low Variance)")
        st.markdown("""
        The model captures the underlying pattern without fitting to noise.
        
        It will have good performance on both training and test data.
        
        **How to achieve:**
        - Balance model complexity
        - Use cross-validation
        - Apply appropriate regularization
        - Feature engineering
        - Ensemble methods
        """)
        
        # Create simple visualization for good fit
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Generate data points
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.2, 100)
        
        # Plot data points
        ax.scatter(x, y, alpha=0.5)
        
        # Good fit line (appropriate degree polynomial)
        ax.plot(x, np.sin(x), 'r-', linewidth=2)
        
        ax.set_title("Appropriate Fitting Example")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        st.pyplot(fig)
    
    st.info("Note: The bias discussed in model fitting (bias-variance tradeoff) is different from the dataset bias mentioned earlier. Model bias refers to the error introduced by approximating a complex real-world problem with a simpler model.")
    
    # SageMaker Clarify section
    st.markdown("### Ensuring Responsible AI with AWS Services")
    
    st.subheader("Amazon SageMaker Clarify")
    st.markdown("""
    Amazon SageMaker Clarify provides greater visibility into training data and models to help identify and limit bias and explain predictions:
    
    1. **Identify imbalances in data**: Detect bias during data preparation
    2. **Check your trained model for bias**: Evaluate the degree to which various types of bias are present in your model
    3. **Explain overall model behavior**: Understand the relative importance of each feature to your model's behavior
    4. **Explain individual predictions**: Understand the relative importance of each feature for individual inferences
    5. **Detect drift in bias and model behavior over time**: Provide alerts and detect drift over time due to changing real-world conditions
    6. **Generate automated reports**: Produce reports on bias and explanations to support internal presentations
    """)
    
    with st.expander("SageMaker Clarify Code Example"):
        st.code("""
        # Example of using Amazon SageMaker Clarify to detect bias
        
        import sagemaker
        from sagemaker import clarify
        
        # Setup the SageMaker session
        session = sagemaker.Session()
        
        # Define the Clarify processor
        clarify_processor = clarify.SageMakerClarifyProcessor(
            role=role,
            instance_count=1,
            instance_type='ml.c5.xlarge',
            sagemaker_session=session
        )
        
        # Define bias config
        bias_config = clarify.BiasConfig(
            label_values_or_threshold=[1],
            facet_name='age',
            facet_values_or_threshold=[40],
            group_name='gender',
            group_values=['Male', 'Female']
        )
        
        # Run bias analysis
        clarify_processor.run_bias_analysis(
            data_config=data_config,
            bias_config=bias_config,
            model_config=model_config,
            model_predicted_label_config=predictor_config,
            methods=['npv', 'dp', 'di']
        )
        """)
    
    # Amazon Augmented AI
    st.subheader("Amazon Augmented AI (A2I)")
    st.markdown("""
    Amazon Augmented AI (Amazon A2I) is a service that brings human review of ML predictions to all developers by removing the heavy lifting associated with building human review systems or managing large numbers of human reviewers.
    
    **Key features:**
    - Implement human review of ML predictions
    - Built-in human review workflows for common ML use cases
    - Ability to create custom workflows for any ML model
    - Integration with AWS services like Amazon Rekognition, Amazon Textract, and more
    """)
    
    with st.expander("A2I Use Case Examples"):
        st.markdown("""
        - **Use Amazon A2I with Amazon Textract**: Have humans review important key-value pairs in documents
        - **Use Amazon A2I with Amazon Rekognition**: Have humans review unsafe images for explicit adult or violent content
        - **Use Amazon A2I to review real-time ML inferences**: Review low-confidence predictions in real-time
        - **Use Amazon A2I with Amazon Comprehend**: Review sentiment analysis, text syntax, and entity detection
        - **Use Amazon A2I with Amazon Transcribe**: Review transcriptions of video or audio files
        - **Use Amazon A2I with Amazon Translate**: Review low-confidence translations
        """)
    
    # SageMaker Model Monitor
    st.subheader("Amazon SageMaker Model Monitor")
    st.markdown("""
    Amazon SageMaker Model Monitor automatically monitors machine learning (ML) models in production and notifies you when quality issues arise.
    
    **Key capabilities:**
    - Automatically monitors ML models in production
    - Detects quality issues and drift using rules
    - Alerts users when problems occur
    - Monitors data quality, model quality, bias drift, and feature attribution drift
    """)
    
    # Create a simple diagram to explain Model Monitor
    fig = go.Figure()
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 1],
        y=[0, 0, 0, 1],
        mode='markers+text',
        marker=dict(size=[60, 60, 60, 60],
                   color=['#FF9900', '#232F3E', '#232F3E', '#232F3E']),
        text=['Input Data', 'ML Model', 'Predictions', 'Model Monitor'],
        textposition='bottom center',
        hoverinfo='text'
    ))
    
    # Add arrows
    fig.add_annotation(
        x=0.5, y=0,
        ax=0, ay=0,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#232F3E'
    )
    
    fig.add_annotation(
        x=1.5, y=0,
        ax=1, ay=0,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#232F3E'
    )
    
    # Add monitoring arrows
    fig.add_annotation(
        x=0.5, y=0.5,
        ax=1, ay=1,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor='#FF9900'
    )
    
    fig.add_annotation(
        x=1.5, y=0.5,
        ax=1, ay=1,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor='#FF9900'
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        title="Amazon SageMaker Model Monitor Workflow",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    st.plotly_chart(fig)
    
    # ML governance features
    st.subheader("ML Governance Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Amazon SageMaker Role Manager")
        st.markdown("""
        - Define minimum permissions in minutes
        - Onboard new users faster
        - Generate custom policies based on specific needs
        """)
    
    with col2:
        st.markdown("#### Amazon SageMaker Model Cards")
        st.markdown("""
        - Document, retrieve, and share necessary model information
        - Create a single source of truth for model information
        - Record details such as purpose and performance goals
        """)
    
    with col3:
        st.markdown("#### Amazon SageMaker Model Dashboard")
        st.markdown("""
        - Monitor model performance through a unified view
        - Track resources and model behavior violations in one place
        - Provide automated alerts for deviations
        """)
    
    # Human-centered design
    st.markdown("### Human-centered design")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Build a Diverse and Multidisciplinary Team")
        st.markdown("""
        - Include AI, data, ethics, legal, domain experts
        - Ensure holistic understanding of AI implications
        - Foster collaboration among all stakeholders
        """)
    
    with col2:
        st.markdown("#### Prioritize Education")
        st.markdown("""
        - Raise awareness on responsible AI practices
        - Provide training on bias, privacy, explainability
        - Create internal resources with guidelines and best practices
        """)
    
    with col3:
        st.markdown("#### Balance AI and Human Judgement")
        st.markdown("""
        - Recognize AI limitations like biases, lack of context
        - Leverage human strengths like reasoning, empathy
        - Promote responsible AI deployment
        """)
    
    # Interactive demo - Hypothetical bias detection
    st.markdown("### Interactive Demo: Bias Detection")
    
    st.markdown("This interactive demo simulates bias detection in a loan approval dataset.")
    
    # Create sample data
    demo_data = {
        'Age Group': ['18-25', '26-35', '36-45', '46-55', '56+'],
        'Application Count': [150, 250, 300, 200, 100],
        'Approval Rate (%)': [45, 65, 75, 70, 50],
        'Average Loan Amount ($)': [10000, 25000, 50000, 45000, 30000]
    }
    
    df = pd.DataFrame(demo_data)
    
    # Display the data
    st.write("Sample Loan Application Data:")
    st.dataframe(df)
    
    # Create bias visualization
    st.markdown("#### Visualize potential bias by demographic:")
    
    chart_type = st.selectbox("Select chart type:", ["Approval Rate", "Average Loan Amount"])
    
    if chart_type == "Approval Rate":
        fig = px.bar(
            df, 
            x='Age Group', 
            y='Approval Rate (%)',
            text_auto=True,
            color='Approval Rate (%)',
            color_continuous_scale=['#FF9900', '#232F3E']
        )
        fig.update_layout(title="Loan Approval Rate by Age Group")
    else:
        fig = px.bar(
            df, 
            x='Age Group', 
            y='Average Loan Amount ($)',
            text_auto=True,
            color='Average Loan Amount ($)',
            color_continuous_scale=['#FF9900', '#232F3E']
        )
        fig.update_layout(title="Average Loan Amount by Age Group")
    
    st.plotly_chart(fig)
    
    # Analysis
    st.markdown("#### Analysis:")
    st.markdown("""
    The data suggests potential age-related bias:
    - Middle-aged applicants (36-45) have the highest approval rates (75%)
    - Younger applicants (18-25) have significantly lower approval rates (45%)
    - Loan amounts also show disparity across age groups
    
    **Recommendations for addressing this bias:**
    1. Investigate factors contributing to lower approval rates for younger applicants
    2. Implement fairness-aware algorithms that balance predictive performance with fairness metrics
    3. Use Amazon SageMaker Clarify to continuously monitor and mitigate bias in production models
    4. Consider including human review for edge cases using Amazon Augmented AI
    """)

# Transparent and Explainable Models Tab Content
def transparent_explainable_models_tab():
    st.markdown("# Guidelines for Responsible AI")
    st.markdown("## Task Statement 4.2: Recognize the importance of transparent and explainable models")
    
    # Transparent and Explainable Models overview
    st.markdown("### Transparent and Explainable Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Transparency, Interpretability, and Explainability**
        
        - **Transparency**: Observing a model's internal logic and decision-making process
        - **Interpretability**: Understanding the model's internal logic and how inputs relate to outputs
        - **Explainability**: Explaining model behavior in human terms that non-technical stakeholders can understand
        
        **Why are these concepts important?**
        
        1. **Trust**: Users and stakeholders need to trust AI systems to adopt them
        2. **Compliance**: Regulations increasingly require explainable AI decisions
        3. **Debugging**: Understanding model behavior helps identify and fix issues
        4. **Fairness**: Transparency helps detect and mitigate biases in AI systems
        5. **Accountability**: Clear attribution of responsibility when AI systems make errors
        """)
    
    with col2:
        # Create a hierarchy chart for explainable AI
        fig = go.Figure(go.Sunburst(
            labels=["AI Model", "Transparent", "Interpretable", "Explainable", 
                   "View Logic", "Understand Logic", "Explain to Others"],
            parents=["", "AI Model", "Transparent", "Interpretable", 
                    "Transparent", "Interpretable", "Explainable"],
            values=[10, 8, 6, 4, 8, 6, 4],
            branchvalues="total",
            marker=dict(colors=["#232F3E", "#FF9900", "#1E88E5", "#16DB93", 
                               "#FF9900", "#1E88E5", "#16DB93"]),
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Parent: %{parent}',
        ))
        
        fig.update_layout(
            margin=dict(t=0, l=0, r=0, b=0),
            title="Hierarchy of AI Model Transparency"
        )
        
        st.plotly_chart(fig)
    
    # Types of explainable models
    st.markdown("### Types of Explainable Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Inherently Interpretable Models
        
        These models are transparent by design and their decision-making process is easily understandable:
        
        - **Linear/Logistic Regression**: Coefficients directly represent feature importance
        - **Decision Trees**: Decision paths can be followed and visualized
        - **Rule-based Systems**: Explicit if-then rules determine outcomes
        - **K-Nearest Neighbors**: Predictions based on similar examples
        
        #### Post-hoc Explanation Methods
        
        These methods explain black-box models after they've been trained:
        
        - **LIME** (Local Interpretable Model-agnostic Explanations)
        - **SHAP** (SHapley Additive exPlanations)
        - **Feature Importance Analysis**
        - **Partial Dependence Plots**
        - **Individual Conditional Expectation Plots**
        """)
    
    with col2:
        st.markdown("""
        #### Model-specific Explanation Techniques
        
        Techniques designed for specific types of models:
        
        - **Neural Networks**:
          - Activation visualization
          - Saliency maps
          - Class activation mapping (CAM)
          - Gradient-weighted Class Activation Mapping (Grad-CAM)
        
        - **Tree-based Models**:
          - Tree visualization
          - Feature importance from node splits
          - SHAP tree explainer
        
        - **Foundation Models**:
          - Attention visualization
          - Prompt engineering for self-explanation
          - Chain-of-thought reasoning
        """)
    
    # Transparency tools
    st.markdown("### Transparency Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Amazon SageMaker Model Cards
        
        Model Cards provide documentation to improve transparency by recording:
        
        - Model purpose and intended use cases
        - Training methodology and data sources
        - Performance metrics and limitations
        - Fairness considerations and evaluations
        - Deployment recommendations
        - Risk ratings and mitigations
        
        #### Open-source Models
        
        Open-source models enhance transparency by:
        
        - Allowing inspection of underlying code and architecture
        - Supporting community review and validation
        - Enabling customization and adaptation
        - Providing visibility into training methodologies
        """)
        
        # Example Model Card
        st.markdown("#### Example Model Card Structure")
        
        with st.expander("View Example Model Card"):
            st.markdown("""
            ### Model Card: Loan Approval Prediction Model
            
            **Model Overview**  
            - **Name**: LoanPredictorV2
            - **Version**: 2.3.1
            - **Type**: XGBoost Classifier
            - **Date Created**: March 15, 2025
            - **Last Updated**: May 10, 2025
            
            **Intended Use**  
            - **Primary Use Case**: Predict likelihood of loan repayment
            - **Intended Users**: Loan officers, financial analysts
            - **Out-of-scope Uses**: Automated loan approval without human review
            
            **Training Data**  
            - **Source**: Internal customer records 2020-2024
            - **Size**: 250,000 records
            - **Features**: Age, income, employment history, credit score, loan history
            - **Preprocessing**: Missing value imputation, standardization, one-hot encoding
            
            **Performance Metrics**  
            - **Accuracy**: 87%
            - **Precision**: 83%
            - **Recall**: 79%
            - **F1 Score**: 0.81
            - **AUC-ROC**: 0.89
            
            **Fairness Evaluation**  
            - **Protected Attributes**: Age, gender, race
            - **Disparate Impact Ratio**: Age (0.92), Gender (0.95), Race (0.89)
            - **Equal Opportunity Difference**: Age (0.04), Gender (0.03), Race (0.06)
            
            **Limitations and Risks**  
            - May underperform for applicants with limited credit history
            - Not suitable for small business loans
            - Risk of reinforcing historical lending patterns
            
            **Mitigation Strategies**  
            - Regular bias monitoring with SageMaker Clarify
            - Human review of edge cases
            - Quarterly model retraining with updated data
            """)
    
    with col2:
        st.markdown("""
        #### Documentation Best Practices
        
        Comprehensive documentation should include:
        
        - Data sources and collection methods
        - Data preprocessing and feature engineering steps
        - Model architecture and hyperparameters
        - Evaluation methods and results
        - Known limitations and edge cases
        - Licensing information
        - Model cards and data sheets
        
        #### Explainable AI Tools on AWS
        
        AWS provides several tools to enhance model explainability:
        
        - **SageMaker Clarify**: Detecting and explaining bias
        - **SageMaker Debugger**: Visualizing and monitoring model internals
        - **SageMaker Model Monitor**: Detecting drift in model behavior
        - **SageMaker Feature Store**: Tracking feature lineage
        """)
        
        # Example code for SageMaker Clarify
        st.markdown("#### Example: Using SageMaker Clarify for Model Explainability")
        
        with st.expander("View Code Example"):
            st.code("""
            # Import required libraries
            import sagemaker
            from sagemaker import clarify
            
            # Initialize the SageMaker session
            session = sagemaker.Session()
            
            # Define the Clarify processor
            clarify_processor = clarify.SageMakerClarifyProcessor(
                role=role,
                instance_count=1,
                instance_type='ml.c5.xlarge',
                sagemaker_session=session
            )
            
            # Configure the data for Clarify
            data_config = clarify.DataConfig(
                s3_data_input_path='s3://bucket-name/input-data.csv',
                s3_output_path='s3://bucket-name/clarify-output',
                label='target_column',
                features='features.csv',
                dataset_type='text/csv'
            )
            
            # Configure the model to explain
            model_config = clarify.ModelConfig(
                model_name='my-model',
                instance_type='ml.c5.xlarge',
                instance_count=1,
                content_type='text/csv',
                accept_type='text/csv'
            )
            
            # Define the explainability configuration
            explainability_config = clarify.ExplainabilityConfig(
                shap_config=clarify.SHAPConfig(
                    baseline=[[0, 0, 0, 0]],
                    num_samples=100,
                    agg_method='mean_abs'
                )
            )
            
            # Run the explainability analysis
            clarify_processor.run_explainability(
                data_config=data_config,
                model_config=model_config,
                explainability_config=explainability_config
            )
            """)
            
    # Tradeoffs section
    st.markdown("### Tradeoffs Between Transparency and Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Transparency vs. Model Performance
        
        There's often a tradeoff between model explainability and performance:
        
        - **Simple, interpretable models** (e.g., linear regression, decision trees)
          - Easy to explain
          - May have lower performance on complex tasks
          
        - **Complex, black-box models** (e.g., deep neural networks, ensemble methods)
          - Can achieve higher performance
          - More difficult to explain
          
        The appropriate balance depends on use case requirements:
        
        - **High-stakes decisions** (healthcare, finance, legal) may prioritize explainability
        - **Lower-risk applications** (recommendations, content generation) may prioritize performance
        """)
        
    with col2:
        # Create visualization showing the tradeoff
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Model complexity vs performance and interpretability
        complexity = np.linspace(0, 10, 100)
        performance = 1 - np.exp(-0.3 * complexity) * np.cos(complexity)
        interpretability = np.exp(-0.4 * complexity)
        
        ax.plot(complexity, performance, 'b-', label='Performance')
        ax.plot(complexity, interpretability, 'r-', label='Interpretability')
        
        # Add model examples at different complexity levels
        models = [
            (1, 'Linear Models'),
            (2.5, 'Decision Trees'),
            (4, 'Random Forests'),
            (6, 'Gradient Boosting'),
            (8, 'Deep Learning')
        ]
        
        for x, name in models:
            y_perf = 1 - np.exp(-0.3 * x) * np.cos(x)
            y_inter = np.exp(-0.4 * x)
            ax.plot(x, y_perf, 'bo', alpha=0.7)
            ax.plot(x, y_inter, 'ro', alpha=0.7)
            ax.annotate(name, (x, (y_perf + y_inter)/2), fontsize=8)
        
        ax.set_xlabel('Model Complexity')
        ax.set_ylabel('Score')
        ax.set_title('Tradeoff: Model Performance vs. Interpretability')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
    
    # Regulatory requirements
    st.markdown("### Regulatory Requirements for Explainable AI")
    
    st.markdown("""
    #### Key Regulations Requiring Model Transparency
    
    Many regulations now require some level of AI transparency and explainability:
    
    - **GDPR (General Data Protection Regulation)**
      - Right to explanation of automated decisions
      - Meaningful information about the logic involved
      
    - **EU AI Act (Proposed)**
      - Risk-based approach to AI regulation
      - Higher transparency requirements for high-risk AI systems
      
    - **Financial Services Regulations**
      - Fair Credit Reporting Act (FCRA)
      - Equal Credit Opportunity Act (ECOA)
      - SR 11-7 (Fed guidance on model risk management)
      
    - **Healthcare Regulations**
      - FDA guidance on AI/ML in medical devices
      - Requirements for clinical validation and explainability
    """)
    
    # Interactive demo
    st.markdown("### Interactive Demo: Model Explainability")
    
    st.markdown("This interactive demo illustrates how different features contribute to a loan approval prediction.")
    
    # Create sample feature importance data
    features = ['Credit Score', 'Income', 'Debt-to-Income Ratio', 'Employment History', 'Loan Amount', 'Age', 'Previous Defaults']
    importance_values = [0.32, 0.27, 0.18, 0.12, 0.06, 0.03, 0.02]
    
    # Create a sample prediction
    sample_values = [720, 85000, 0.28, 5, 250000, 35, 0]
    
    # Display the sample applicant data
    st.write("#### Sample Loan Applicant")
    
    applicant_data = pd.DataFrame({
        'Feature': features,
        'Value': sample_values
    })
    
    st.dataframe(applicant_data)
    
    # Create feature importance visualization
    st.write("#### Feature Importance (SHAP Values)")
    
    # Allow user to select model type
    model_type = st.selectbox(
        "Select model type to see how feature importance varies:", 
        ["XGBoost", "Logistic Regression", "Neural Network"]
    )
    
    # Adjust importance values based on model type
    if model_type == "Logistic Regression":
        # Logistic regression tends to rely more heavily on fewer features
        importance_values = [0.42, 0.32, 0.15, 0.05, 0.03, 0.02, 0.01]
    elif model_type == "Neural Network":
        # Neural networks might distribute importance more evenly
        importance_values = [0.28, 0.25, 0.20, 0.15, 0.07, 0.03, 0.02]
    
    # Create horizontal bar chart
    fig = px.bar(
        x=importance_values,
        y=features,
        orientation='h',
        color=importance_values,
        color_continuous_scale=['#232F3E', '#FF9900'],
        labels={'x': 'Importance Value', 'y': 'Feature'},
        title=f"Feature Importance for {model_type}",
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig)
    
    # Display prediction explanation
    approval_score = sum(a*b for a, b in zip(importance_values, [0.9, 0.85, 0.7, 0.95, 0.6, 0.8, 1.0]))
    approval_pct = min(approval_score * 100, 98)  # Cap at 98%
    
    st.write(f"#### Prediction Explanation")
    st.write(f"Approval Likelihood: {approval_pct:.1f}%")
    
    # Create explanation based on top features
    st.markdown(f"""
    **Model Explanation:**
    
    This applicant is likely to be approved because:
    
    1. **Credit Score (720)** is above the average of 680, contributing positively to the prediction
    2. **Income (${sample_values[1]:,})** is well above the required minimum, strongly supporting approval
    3. **Debt-to-Income Ratio ({sample_values[2]})** is below our threshold of 0.36, indicating financial stability
    4. **Employment History ({sample_values[3]} years)** shows stable employment, reducing perceived risk
    
    The model has identified these factors as the most important indicators of loan repayment ability.
    """)
    
    # Best practices for transparent AI
    st.markdown("### Best Practices for Implementing Transparent AI")
    
    st.markdown("""
    #### Recommendations for Building Transparent and Explainable AI Systems
    
    1. **Start with interpretability in mind**
       - Consider explainability requirements from the beginning of the project
       - Select model architectures that balance performance with interpretability
       
    2. **Document thoroughly**
       - Maintain detailed documentation of data, models, and processes
       - Create model cards for all deployed models
       - Record model performance across different demographic groups
       
    3. **Use explainability tools**
       - Implement tools like SHAP, LIME, or SageMaker Clarify
       - Generate feature importance metrics for all models
       - Visualize model decision boundaries where possible
       
    4. **Implement human oversight**
       - Use Amazon A2I for human review of critical or uncertain predictions
       - Establish regular model review processes
       - Have subject matter experts validate model explanations
       
    5. **Provide appropriate explanations to different stakeholders**
       - Technical explanations for data scientists and developers
       - Business-friendly explanations for product managers
       - Simple, understandable explanations for end users
    """)
    
    # Case study
    st.markdown("### Case Study: Transparent AI in Financial Services")
    
    with st.expander("View Case Study"):
        st.markdown("""
        #### Challenge
        
        A major financial institution needed to implement an AI-based credit scoring system that would:
        - Improve prediction accuracy over traditional methods
        - Comply with regulatory requirements for explainability
        - Provide clear reasons for credit decisions to customers
        - Ensure fair treatment across demographic groups
        
        #### Solution
        
        The company implemented a solution using AWS services:
        
        1. **Model Development**
           - Used Amazon SageMaker to build and train gradient boosting models
           - Selected a model architecture that balanced performance with interpretability
           - Integrated SageMaker Clarify to analyze feature importance and detect bias
        
        2. **Transparency Mechanisms**
           - Created detailed model cards documenting model characteristics
           - Implemented SHAP values to explain individual predictions
           - Used SageMaker Model Monitor to detect concept drift and model degradation
        
        3. **Human Oversight**
           - Integrated Amazon A2I for human review of edge cases
           - Established a model governance committee to review model performance
           - Created an appeals process for customers who wished to contest decisions
        
        #### Results
        
        - 15% improvement in prediction accuracy compared to previous methods
        - Successfully passed regulatory compliance audits
        - Reduced customer complaints about credit decisions by 35%
        - Detected and mitigated potential bias in credit scoring for younger applicants
        
        #### Key Learnings
        
        - Explainable AI doesn't necessarily mean sacrificing performance
        - Starting with explainability as a design requirement saves time and resources later
        - Different stakeholders need different levels of explanation
        - Regular monitoring and retraining are essential to maintain model fairness and accuracy
        """)
    
# Methods to Secure AI Systems Tab Content
def secure_ai_systems_tab():
    st.markdown("# Security, Compliance, and Governance for AI Solutions")
    st.markdown("## Task Statement 5.1: Explain methods to secure AI systems")
    
    # Introduction to AI security
    st.markdown("### The Importance of Security in AI Systems")
    
    st.markdown("""
    AI systems, particularly those built with machine learning and generative AI capabilities, 
    introduce unique security concerns beyond traditional applications. Securing AI systems is 
    critical because:
    
    1. They often process sensitive data
    2. They can be targets for adversarial attacks
    3. They may expose intellectual property in model weights and architectures
    4. Compromised AI systems can lead to misleading outputs or harmful decisions
    """)
    
    # Threat landscape for AI systems
    st.markdown("### Threat Landscape for AI Systems")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Common Security Threats to AI Systems
        
        - **Prompt Injection**
          - Manipulating input prompts to generate unintended outputs
          - Bypassing safety filters and content policies
          
        - **Data Poisoning**
          - Introducing malicious data into training datasets
          - Causing model bias or backdoor vulnerabilities
          
        - **Model Extraction**
          - Stealing model parameters through repeated API calls
          - Reconstructing proprietary models through outputs
          
        - **Adversarial Examples**
          - Crafting inputs that cause misclassification
          - Exploiting model vulnerabilities
        """)
        
    with col2:
        st.markdown("""
        #### OWASP Top 10 for Large Language Models
        
        1. **Prompt Injection**: Manipulating model behavior through crafted prompts
        2. **Insecure Output Handling**: Failure to validate model outputs
        3. **Training Data Poisoning**: Compromising model behavior through tainted data
        4. **Model Denial of Service**: Overloading systems with resource-intensive prompts
        5. **Supply Chain Vulnerabilities**: Risks from third-party models and components
        6. **Sensitive Information Disclosure**: Leaking confidential data in responses
        7. **Insecure Plugin Design**: Vulnerabilities in model extensions
        8. **Excessive Agency**: Giving models too much autonomy
        9. **Overreliance**: Excessive trust in model outputs
        10. **Model Theft**: Unauthorized extraction of model weights or architecture
        """)
    
    # Security and Privacy Considerations
    st.markdown("### Security and Privacy Considerations for AI Systems")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Threat Detection")
        st.markdown("""
        - Monitor for potential security threats
        - Analyze network traffic and user behavior
        - Deploy AI-powered threat detection systems
        - Identify unusual patterns in model usage
        """)
        
    with col2:
        st.markdown("#### Vulnerability Management")
        st.markdown("""
        - Identify and address vulnerabilities in AI systems
        - Conduct security assessments and penetration testing
        - Implement robust patch management
        - Regular code reviews and security audits
        """)
        
    with col3:
        st.markdown("#### Infrastructure Protection")
        st.markdown("""
        - Secure the underlying infrastructure
        - Implement strong access controls
        - Network segmentation
        - Encryption mechanisms
        - Ensure resilience against attacks
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Prompt Injection Mitigation")
        st.markdown("""
        - Filter and sanitize input prompts
        - Implement input validation
        - Use guardrails for model responses
        - Monitor for suspicious patterns in prompts
        - Test models with adversarial examples
        """)
        
    with col2:
        st.markdown("#### Data Encryption")
        st.markdown("""
        - Encrypt data at rest and in transit
        - Secure training data and model weights
        - Proper key management
        - Use of secure cryptographic protocols
        - Regular security audits of encryption practices
        """)
    
    # Defense in depth
    st.markdown("### Defense in Depth for AI Systems")
    
    st.markdown("""
    The **Defense in Depth** security strategy uses multiple redundant defenses to protect your AWS accounts, workloads, data, and assets. 
    It helps ensure that if any one security control is compromised or fails, additional layers exist to help isolate threats and prevent, 
    detect, respond, and recover from security events.
    """)
    
    # Create defense in depth visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create concentric circles for defense layers
    circle_colors = ['#232F3E', '#FF9900', '#1E88E5', '#16DB93', '#EFBF38']
    circle_labels = ['Data', 'Application', 'Compute', 'Network', 'Perimeter']
    circle_radii = [1, 2, 3, 4, 5]
    
    for i, (radius, color, label) in enumerate(zip(circle_radii, circle_colors, circle_labels)):
        circle = plt.Circle((0, 0), radius, fill=True, alpha=0.6, color=color, label=label)
        ax.add_patch(circle)
        
        # Add label inside the circle layer
        angle = np.pi/4  # 45 degrees
        x = (radius - 0.5) * np.cos(angle)
        y = (radius - 0.5) * np.sin(angle)
        ax.text(x, y, label, ha='center', va='center', color='white', fontweight='bold')
    
    # Add some defense mechanisms to each layer
    mechanisms = [
        ('Data', 'Encryption, Access Control, Tokenization'),
        ('Application', 'Input Validation, Auth, Monitoring'),
        ('Compute', 'Secure AMIs, Patching, Isolation'),
        ('Network', 'Segmentation, Firewalls, Encryption'),
        ('Perimeter', 'WAF, Shield, CloudFront')
    ]
    
    for label, mechanism in mechanisms:
        idx = circle_labels.index(label)
        radius = circle_radii[idx]
        angle = -np.pi/4  # -45 degrees
        x = (radius - 0.5) * np.cos(angle)
        y = (radius - 0.5) * np.sin(angle)
        ax.text(x, y, mechanism, ha='center', va='center', color='white', fontsize=8)
    
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    st.pyplot(fig)
    
    st.markdown("""
    **Applying Defense in Depth to AI Systems:**
    
    1. **Data Layer**
       - Encrypt sensitive training data and model weights
       - Use fine-grained access controls for data access
       - Implement data tokenization for sensitive information
       
    2. **Application Layer**
       - Validate and sanitize all inputs to AI models
       - Implement authentication and authorization for API access
       - Monitor for abnormal usage patterns
       
    3. **Compute Layer**
       - Use secure compute environments for model training and inference
       - Regular patching and updates of framework dependencies
       - Isolate AI workloads from other systems
       
    4. **Network Layer**
       - Segment networks to isolate AI systems
       - Use encryption for all network communication
       - Implement firewalls to restrict access
       
    5. **Perimeter Layer**
       - Web Application Firewalls to filter malicious requests
       - DDoS protection for AI endpoints
       - Edge caching and content delivery for public-facing AI services
    """)
    
    # AWS Services for AI Security
    st.markdown("### AWS Services for Securing AI Systems")
    
    security_services = [
        {
            "name": "AWS Identity and Access Management (IAM)",
            "description": "Securely control access to AWS services and resources",
            "features": [
                "IAM users, groups, and roles",
                "IAM policies for fine-grained permissions",
                "Multi-factor authentication",
                "Least privilege principle enforcement"
            ],
            "use_case": "Control who can access your AI models and data, and what actions they can perform"
        },
        {
            "name": "AWS Key Management Service (KMS)",
            "description": "Create and control encryption keys to secure data",
            "features": [
                "Centralized key management",
                "Key rotation and lifecycle management",
                "Integration with other AWS services",
                "FIPS 140-2 validated cryptographic modules"
            ],
            "use_case": "Encrypt model weights, training data, and API requests/responses"
        },
        {
            "name": "Amazon Macie",
            "description": "Discover, classify, and protect sensitive data",
            "features": [
                "Automated sensitive data discovery",
                "Data classification",
                "Security assessment for S3 buckets",
                "Integration with AWS Security Hub"
            ],
            "use_case": "Identify and protect sensitive data used in AI model training"
        },
        {
            "name": "Amazon Virtual Private Cloud (VPC)",
            "description": "Provision an isolated network environment",
            "features": [
                "Network isolation",
                "Subnet configuration",
                "Security groups and NACLs",
                "Private connectivity to AWS services"
            ],
            "use_case": "Isolate AI training and inference environments from public internet"
        },
        {
            "name": "AWS PrivateLink",
            "description": "Secure private connectivity to AWS services",
            "features": [
                "Private connectivity to AWS services",
                "No exposure to public internet",
                "Reduced data exposure",
                "Simplified network management"
            ],
            "use_case": "Access Amazon Bedrock or SageMaker endpoints without traversing the public internet"
        },
        {
            "name": "AWS CloudWatch",
            "description": "Monitor AWS resources and applications",
            "features": [
                "Real-time monitoring",
                "Log collection and analysis",
                "Alarms and notifications",
                "Automated actions based on metrics"
            ],
            "use_case": "Monitor AI model performance, detect anomalies, and trigger alerts"
        },
        {
            "name": "AWS CloudTrail",
            "description": "Track user activity and API usage",
            "features": [
                "API call history",
                "Event filtering",
                "CloudTrail Insights for unusual activity",
                "Log file integrity validation"
            ],
            "use_case": "Audit all actions taken with your AI resources and detect unauthorized access"
        },
        {
            "name": "AWS Config",
            "description": "Assess, audit, and evaluate configurations",
            "features": [
                "Resource discovery and tracking",
                "Configuration history",
                "Compliance evaluation",
                "Automated remediation"
            ],
            "use_case": "Ensure your AI infrastructure maintains compliance with security policies"
        },
        {
            "name": "Amazon Inspector",
            "description": "Automated security assessment",
            "features": [
                "Continuous scanning for vulnerabilities",
                "EC2 instances, container images, Lambda functions",
                "Integration with AWS Organizations",
                "Risk prioritization"
            ],
            "use_case": "Identify vulnerabilities in the infrastructure hosting your AI workloads"
        }
    ]
    
    # Display services in expandable sections
    for i, service in enumerate(security_services):
        with st.expander(f"{service['name']} - {service['description']}"):
            st.markdown(f"### {service['name']}")
            st.markdown(f"{service['description']}")
            
            st.markdown("#### Key Features")
            for feature in service['features']:
                st.markdown(f"- {feature}")
            
            st.markdown("#### AI Security Use Case")
            st.markdown(service['use_case'])
            
            # Example code for selected services
            if service['name'] == "AWS Identity and Access Management (IAM)":
                st.markdown("#### Example IAM Policy for Amazon Bedrock Access")
                st.code("""
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "bedrock:InvokeModel",
                                "bedrock:InvokeModelWithResponseStream"
                            ],
                            "Resource": [
                                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2",
                                "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-express-v1"
                            ]
                        },
                        {
                            "Effect": "Deny",
                            "Action": [
                                "bedrock:CreateModelCustomizationJob",
                                "bedrock:StopModelCustomizationJob"
                            ],
                            "Resource": "*"
                        }
                    ]
                }
                """)
            elif service['name'] == "AWS CloudTrail":
                st.markdown("#### Example CloudTrail Event for SageMaker Model Creation")
                st.code("""
                {
                    "eventVersion": "1.08",
                    "userIdentity": {
                        "type": "IAMUser",
                        "principalId": "AIDAEXAMPLE",
                        "arn": "arn:aws:iam::123456789012:user/data-scientist",
                        "accountId": "123456789012",
                        "accessKeyId": "AKIAEXAMPLE",
                        "userName": "data-scientist"
                    },
                    "eventTime": "2025-05-20T15:32:10Z",
                    "eventSource": "sagemaker.amazonaws.com",
                    "eventName": "CreateModel",
                    "awsRegion": "us-east-1",
                    "sourceIPAddress": "192.0.2.1",
                    "userAgent": "aws-cli/2.8.12 Python/3.9.16",
                    "requestParameters": {
                        "modelName": "sentiment-analysis-model",
                        "primaryContainer": {
                            "image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0",
                            "modelDataUrl": "s3://my-bucket/models/sentiment-analysis-v1.tar.gz"
                        },
                        "executionRoleArn": "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
                    },
                    "responseElements": {
                        "modelArn": "arn:aws:sagemaker:us-east-1:123456789012:model/sentiment-analysis-model"
                    },
                    "requestID": "a1b2c3d4-5678-90ab-cdef-EXAMPLE",
                    "eventID": "a1b2c3d4-5678-90ab-cdef-EXAMPLE",
                    "readOnly": false,
                    "eventType": "AwsApiCall",
                    "managementEvent": true,
                    "recipientAccountId": "123456789012"
                }
                """)
    
    # Generative AI Security Scoping Matrix
    st.markdown("### Generative AI Security Scoping Matrix")
    
    st.markdown("""
    The **Generative AI Security Scoping Matrix** is a mental model to classify use-cases and determine 
    appropriate security controls based on how you're using generative AI.
    """)
    
    # Create a table for the scoping matrix
    scopes_data = {
        "Scope": ["Scope 1: Consumer App", "Scope 2: Enterprise App", "Scope 3: Pre-trained Models", "Scope 4: Fine-tuned Models", "Scope 5: Self-trained Models"],
        "Description": [
            "Using 'public' generative AI services",
            "Using an app or SaaS with generative AI features",
            "Building your app on a versioned model",
            "Fine-tuning a model on your data",
            "Training a model from scratch on your data"
        ],
        "Examples": [
            "ChatGPT, Midjourney",
            "Salesforce Einstein GPT, Amazon Q Developer",
            "Amazon Bedrock base models",
            "Amazon Bedrock customized models, SageMaker JumpStart",
            "Amazon SageMaker custom training"
        ],
        "Security Complexity": [1, 2, 3, 4, 5]
    }
    
    df_scopes = pd.DataFrame(scopes_data)
    
    # Create a color scale for the security complexity
    def color_security_complexity(val):
        color = ''
        if val == 1:
            color = 'background-color: #ccffcc'
        elif val == 2:
            color = 'background-color: #e6ffcc'
        elif val == 3:
            color = 'background-color: #ffffcc'
        elif val == 4:
            color = 'background-color: #ffebcc'
        elif val == 5:
            color = 'background-color: #ffcccc'
        return color
    
    # Apply the color styling
    styled_df = df_scopes.style.applymap(color_security_complexity, subset=['Security Complexity'])
    
    st.table(styled_df)
    
    st.markdown("""
    As you move from Scope 1 to Scope 5:
    
    1. Your control over the model increases
    2. Your security responsibilities increase
    3. The complexity of security considerations grows
    
    For each scope, consider these security domains:
    - **Governance & Compliance**
    - **Legal & Privacy**
    - **Risk Management**
    - **Security Controls**
    - **Resilience**
    """)
    
    # Interactive demo
    st.markdown("### Interactive Demo: Securing AI Workloads")
    
    st.markdown("This interactive demo helps you understand the security controls needed for different types of AI workloads.")
    
    # Create a form to select the type of AI workload
    workload_type = st.selectbox(
        "Select your AI workload type:",
        [
            "Public-facing chatbot using Amazon Bedrock API",
            "Internal document analysis tool using fine-tuned models",
            "Healthcare diagnostic system with custom-trained models",
            "Customer service automation with third-party SaaS solution",
            "Content moderation system for user-generated content"
        ]
    )
    
    # Map workload types to recommended security controls
    security_recommendations = {
        "Public-facing chatbot using Amazon Bedrock API": {
            "data_sensitivity": "Medium",
            "authentication": ["Amazon Cognito", "IAM roles"],
            "network": ["API Gateway", "WAF", "CloudFront"],
            "monitoring": ["CloudWatch", "CloudTrail"],
            "compliance": ["Document terms of service", "Clear user consent"],
            "key_risks": ["Prompt injection", "Data leakage", "Overreliance"]
        },
        "Internal document analysis tool using fine-tuned models": {
            "data_sensitivity": "High",
            "authentication": ["IAM roles", "MFA", "Single Sign-On"],
            "network": ["VPC", "PrivateLink", "VPC Endpoints"],
            "monitoring": ["CloudWatch", "CloudTrail", "Security Hub"],
            "compliance": ["Data classification", "Access reviews", "Audit logging"],
            "key_risks": ["Unauthorized access", "Data poisoning", "Model extraction"]
        },
        "Healthcare diagnostic system with custom-trained models": {
            "data_sensitivity": "Very High",
            "authentication": ["IAM roles", "MFA", "Single Sign-On", "Service Control Policies"],
            "network": ["VPC", "PrivateLink", "VPC Endpoints", "Network Firewalls"],
            "monitoring": ["CloudWatch", "CloudTrail", "Security Hub", "GuardDuty", "Macie"],
            "compliance": ["HIPAA controls", "Model documentation", "Human oversight"],
            "key_risks": ["PHI exposure", "Model bias", "Reliability concerns", "Regulatory compliance"]
        },
        "Customer service automation with third-party SaaS solution": {
            "data_sensitivity": "Medium",
            "authentication": ["Identity Federation", "SSO"],
            "network": ["IAM Access Controls", "API Gateway"],
            "monitoring": ["CloudWatch", "SaaS provider monitoring"],
            "compliance": ["Vendor assessment", "Data processing agreements"],
            "key_risks": ["Vendor access to data", "Limited visibility", "Contractual limitations"]
        },
        "Content moderation system for user-generated content": {
            "data_sensitivity": "Medium-High",
            "authentication": ["IAM roles", "API Gateway authorization"],
            "network": ["WAF", "Shield", "CloudFront"],
            "monitoring": ["CloudWatch", "GuardDuty", "Security Hub"],
            "compliance": ["Content moderation policies", "Audit capabilities"],
            "key_risks": ["Moderation evasion", "False positives/negatives", "Adversarial content"]
        }
    }
    
    # Get recommendations for selected workload
    recommendation = security_recommendations[workload_type]
    
    # Display security recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Workload Security Profile")
        st.markdown(f"**Data Sensitivity:** {recommendation['data_sensitivity']}")
        
        st.markdown("**Authentication & Authorization**")
        for auth in recommendation['authentication']:
            st.markdown(f"- {auth}")
            
        st.markdown("**Network Security**")
        for net in recommendation['network']:
            st.markdown(f"- {net}")
    
    with col2:
        st.markdown("#### Monitoring & Governance")
        for monitor in recommendation['monitoring']:
            st.markdown(f"- {monitor}")
            
        st.markdown("**Compliance Considerations**")
        for comp in recommendation['compliance']:
            st.markdown(f"- {comp}")
            
        st.markdown("**Key Risks**")
        for risk in recommendation['key_risks']:
            st.markdown(f"- {risk}")
    
    # Security architecture diagram
    st.markdown("### Sample Security Architecture")
    
    # Create a simple security architecture diagram based on the selected workload
    if workload_type == "Public-facing chatbot using Amazon Bedrock API":
        st.markdown("#### Public-facing Chatbot Architecture")
        st.code("""
        ┌────────────────┐       ┌────────────┐      ┌──────────────┐      ┌─────────────────┐
        │                │       │            │      │              │      │                 │
        │  CloudFront    ├──────►│ API Gateway├─────►│ Lambda       ├─────►│ Amazon Bedrock  │
        │  + WAF         │       │            │      │ (with IAM)   │      │                 │
        └────────────────┘       └────────────┘      └──────────────┘      └─────────────────┘
               ▲                                            │
               │                                            ▼
        ┌──────┴───────┐                           ┌──────────────┐
        │              │                           │              │
        │  End Users   │                           │  CloudWatch  │
        │              │                           │  Logs        │
        └──────────────┘                           └──────────────┘
        """)
    elif workload_type == "Internal document analysis tool using fine-tuned models":
        st.markdown("#### Internal Document Analysis Architecture")
        st.code("""
        ┌──────────────────┐     ┌───────────┐     ┌───────────────┐     ┌───────────────────┐
        │                  │     │           │     │               │     │                   │
        │  Corporate       │     │ ALB in    │     │ EC2/ECS in    │     │ SageMaker         │
        │  Network + SSO   ├────►│ Private   ├────►│ Private       ├────►│ Endpoint in VPC   │
        │                  │     │ Subnet    │     │ Subnet        │     │                   │
        └──────────────────┘     └───────────┘     └───────────────┘     └───────────────────┘
                                                          │                       │
                                                          ▼                       │
        ┌────────────────────┐    ┌───────────────┐    ┌─────────────┐           │
        │                    │    │               │    │             │           │
        │  S3 (Encrypted)    │◄───┤ IAM Roles     │    │ KMS Keys    │◄──────────┘
        │  with VPC Endpoint │    │ with Policies │    │             │
        │                    │    │               │    │             │
        └────────────────────┘    └───────────────┘    └─────────────┘
        """)
    else:
        st.markdown(f"#### Security Architecture for {workload_type}")
        st.info("A specialized security architecture would be developed based on the specific requirements of this workload type.")
    
    # Best practices
    st.markdown("### Best Practices for Secure AI Implementation")
    
    with st.expander("Least Privilege Access Control"):
        st.markdown("""
        **Implement the principle of least privilege:**
        
        - Grant the minimum permissions necessary for a user or service to perform its functions
        - Regularly review and audit permissions to identify and remove unnecessary access
        - Use IAM roles with temporary credentials instead of long-term access keys
        - Apply resource-based policies where appropriate
        
        **Example IAM policy granting least privilege to invoke specific models in Amazon Bedrock:**
        """)
        
        st.code("""
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "bedrock:InvokeModel",
                    "Resource": [
                        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2"
                    ],
                    "Condition": {
                        "StringEquals": {
                            "aws:RequestTag/Environment": "Production"
                        }
                    }
                }
            ]
        }
        """)
    
    with st.expander("Data Protection and Encryption"):
        st.markdown("""
        **Protect data throughout its lifecycle:**
        
        - Encrypt sensitive data at rest using AWS KMS
        - Encrypt data in transit using TLS
        - Implement data classification and handling procedures
        - Use tokenization or anonymization where appropriate
        
        **Example configuration for S3 bucket encryption with training data:**
        """)
        
        st.code("""
        # Using boto3 to create an encrypted S3 bucket for AI training data
        
        import boto3
        
        s3 = boto3.client('s3')
        kms = boto3.client('kms')
        
        # Create a KMS key for encryption
        key_response = kms.create_key(
            Description='Key for AI training data encryption',
            KeyUsage='ENCRYPT_DECRYPT',
            Origin='AWS_KMS'
        )
        
        key_id = key_response['KeyMetadata']['KeyId']
        
        # Create an S3 bucket with encryption enabled
        s3.create_bucket(
            Bucket='ai-training-data-secure',
            CreateBucketConfiguration={
                'LocationConstraint': 'us-west-2'
            }
        )
        
        # Set default encryption on the bucket
        s3.put_bucket_encryption(
            Bucket='ai-training-data-secure',
            ServerSideEncryptionConfiguration={
                'Rules': [
                    {
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'aws:kms',
                            'KMSMasterKeyID': key_id
                        },
                        'BucketKeyEnabled': True
                    }
                ]
            }
        )
        
        # Block public access to the bucket
        s3.put_public_access_block(
            Bucket='ai-training-data-secure',
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': True,
                'RestrictPublicBuckets': True
            }
        )
        """)
    
    with st.expander("Network Security"):
        st.markdown("""
        **Secure network communication for AI workloads:**
        
        - Use VPCs to isolate AI training and inference environments
        - Implement security groups and NACLs to control traffic
        - Use VPC endpoints to access AWS services privately
        - Consider AWS PrivateLink for secure access to services
        
        **Example VPC design for secure AI workloads:**
        """)
        
        st.code("""
        # Using CDK to create a secure VPC for AI workloads
        
        from aws_cdk import (
            aws_ec2 as ec2,
            aws_sagemaker as sagemaker,
            Stack
        )
        from constructs import Construct
        
        class SecureAIInfrastructureStack(Stack):
            def __init__(self, scope: Construct, id: str, **kwargs) -> None:
                super().__init__(scope, id, **kwargs)
                
                # Create a VPC with isolated subnets
                self.vpc = ec2.Vpc(
                    self, "AISecureVPC",
                    max_azs=2,
                    subnet_configuration=[
                        ec2.SubnetConfiguration(
                            name="Private",
                            subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT,
                            cidr_mask=24
                        ),
                        ec2.SubnetConfiguration(
                            name="Isolated",
                            subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                            cidr_mask=24
                        )
                    ]
                )
                
                # Create VPC Endpoints for AWS services
                self.s3_endpoint = self.vpc.add_gateway_endpoint(
                    "S3Endpoint",
                    service=ec2.GatewayVpcEndpointAwsService.S3
                )
                
                self.dynamodb_endpoint = self.vpc.add_gateway_endpoint(
                    "DynamoDBEndpoint",
                    service=ec2.GatewayVpcEndpointAwsService.DYNAMODB
                )
                
                self.sagemaker_endpoint = self.vpc.add_interface_endpoint(
                    "SageMakerEndpoint",
                    service=ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_API
                )
                
                self.sagemaker_runtime_endpoint = self.vpc.add_interface_endpoint(
                    "SageMakerRuntimeEndpoint",
                    service=ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_RUNTIME
                )
        """)
    
    with st.expander("Monitoring and Logging"):
        st.markdown("""
        **Implement comprehensive monitoring for AI systems:**
        
        - Use CloudWatch to monitor model performance and usage metrics
        - Set up CloudTrail to track API calls and user activity
        - Configure alarms for abnormal patterns or potential security events
        - Centralize logs for analysis and correlation
        
        **Example CloudWatch dashboard and alarm configuration:**
        """)
        
        st.code("""
        # Using CloudWatch to monitor SageMaker endpoints and set alarms
        
        import boto3
        from datetime import datetime, timedelta
        
        cloudwatch = boto3.client('cloudwatch')
        
        # Create a dashboard for SageMaker monitoring
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [ "AWS/SageMaker", "Invocations", "EndpointName", "my-model-endpoint" ],
                            [ ".", "InvocationsPerInstance", ".", "." ],
                            [ ".", "ModelLatency", ".", "." ],
                            [ ".", "OverheadLatency", ".", "." ]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": "us-east-1",
                        "period": 300,
                        "stat": "Average",
                        "title": "Endpoint Performance"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [ "AWS/SageMaker", "CPUUtilization", "EndpointName", "my-model-endpoint" ],
                            [ ".", "MemoryUtilization", ".", "." ],
                            [ ".", "DiskUtilization", ".", "." ]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": "us-east-1",
                        "period": 300,
                        "stat": "Average",
                        "title": "Resource Utilization"
                    }
                }
            ]
        }
        
        # Create the dashboard
        cloudwatch.put_dashboard(
            DashboardName='SageMakerModelMonitoring',
            DashboardBody=str(dashboard_body)
        )
        
        # Create an alarm for high model latency
        cloudwatch.put_metric_alarm(
            AlarmName='HighModelLatency',
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=3,
            MetricName='ModelLatency',
            Namespace='AWS/SageMaker',
            Period=300,
            Statistic='Average',
            Threshold=1000,  # 1 second
            ActionsEnabled=True,
            AlarmActions=[
                'arn:aws:sns:us-east-1:123456789012:model-alerts'
            ],
            AlarmDescription='Alarm when model latency exceeds 1 second',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': 'my-model-endpoint'
                }
            ]
        )
        
        # Create an alarm for abnormal number of invocations (potential DoS)
        cloudwatch.put_metric_alarm(
            AlarmName='AbnormalInvocations',
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=1,
            MetricName='Invocations',
            Namespace='AWS/SageMaker',
            Period=60,
            Statistic='Sum',
            Threshold=5000,  # Adjust based on expected traffic
            ActionsEnabled=True,
            AlarmActions=[
                'arn:aws:sns:us-east-1:123456789012:security-alerts'
            ],
            AlarmDescription='Alarm when abnormally high number of invocations occur',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': 'my-model-endpoint'
                }
            ]
        )
        """)
    
    with st.expander("Vulnerability Management"):
        st.markdown("""
        **Maintain a robust vulnerability management program:**
        
        - Use Amazon Inspector to identify vulnerabilities in infrastructure
        - Regularly patch and update AI frameworks and dependencies
        - Conduct security testing of AI applications
        - Implement a responsible disclosure program
        
        **Best practices for AI vulnerability management:**
        
        1. **Keep dependencies updated**
           - Regularly update ML frameworks, libraries, and plugins
           - Use dependency scanning tools to identify vulnerable components
           
        2. **Secure model serving infrastructure**
           - Use Amazon Inspector to scan compute resources
           - Implement a regular patching cadence
           - Use container scanning for containerized deployments
           
        3. **Test AI-specific vulnerabilities**
           - Test models for prompt injection vulnerabilities
           - Evaluate models with adversarial examples
           - Test rate limiting and resource throttling
           
        4. **Monitor for new vulnerabilities**
           - Subscribe to security advisories for ML frameworks
           - Participate in responsible AI communities
           - Implement a process for emergency patching
        """)
    
    # Case study
    st.markdown("### Case Study: Securing a Financial Services AI System")
    
    with st.expander("View Case Study"):
        st.markdown("""
        #### Challenge
        
        A large financial institution needed to develop an AI-based fraud detection system with extremely high security requirements:
        
        - Processing sensitive customer transaction data
        - Subject to strict regulatory requirements
        - High availability and performance needs
        - Protection against sophisticated adversarial attacks
        
        #### Solution
        
        The financial institution implemented a multi-layered security approach on AWS:
        
        1. **Data Protection**
           - Encrypted all data at rest using AWS KMS with customer-managed keys
           - Implemented field-level encryption for PII
           - Applied strict data access controls with IAM policies
           - Used secure enclaves for model training
        
        2. **Network Security**
           - Deployed the entire solution in private subnets within a dedicated VPC
           - Used AWS PrivateLink to access AWS services without traversing the internet
           - Implemented multiple security groups with least-privilege rules
           - Used AWS WAF to protect API endpoints
        
        3. **Monitoring and Detection**
           - Configured CloudTrail to log all API calls
           - Set up CloudWatch alarms for suspicious activities
           - Used Amazon Detective to analyze security events
           - Implemented custom anomaly detection for model inputs and outputs
        
        4. **Model Security**
           - Implemented input validation and sanitization
           - Deployed model monitoring for concept drift
           - Added adversarial robustness testing
           - Created automatic retraining pipelines
        
        5. **Governance**
           - Documented all security controls
           - Conducted regular security assessments
           - Implemented model risk governance
           - Established a security incident response plan
        
        #### Results
        
        - Successfully passed regulatory security audits
        - Zero security incidents in first year of operation
        - 99.99% system availability
        - 35% reduction in fraud losses
        - Maintained model performance while enhancing security
        
        #### Key Learnings
        
        - Security must be built in from the beginning, not added later
        - Defense in depth is essential for high-value AI systems
        - Regular security testing and monitoring are critical
        - Cross-functional collaboration between data science and security teams improves outcomes
        """)

# Governance and Compliance Tab Content
def governance_compliance_tab():
    st.markdown("# Security, Compliance, and Governance for AI Solutions")
    st.markdown("## Task Statement 5.2: Recognize governance and compliance regulations for AI systems")
    
    # AI standards compliance
    st.markdown("### AI Standards Compliance")
    
    st.markdown("""
    AI standards compliance influences how organizations follow established guidelines, rules, and legal 
    requirements that govern the development, deployment, and use of AI technologies. AI compliance differs 
    from traditional software and technology requirements in several key ways:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Complexity and opacity
        
        AI systems, especially large language models (LLMs) and generative AI, can be highly complex with 
        opaque decision-making processes. This makes it challenging to audit and understand how they arrive 
        at outputs, which is crucial for compliance.
        
        #### Dynamism and adaptability
        
        AI systems are often dynamic and can adapt and change over time, even after deployment. 
        This makes it difficult to apply static standards, frameworks, and mandates.
        
        #### Emergent capabilities
        
        Emergent capabilities in AI systems refer to unexpected or unintended capabilities that arise as a result 
        of complex interactions within the AI system, in contrast to capabilities that are explicitly programmed 
        or designed.
        """)
    
    with col2:
        st.markdown("""
        #### Unique risks
        
        AI poses novel risks, such as algorithmic bias, privacy violations, misinformation, and AI-powered 
        automation displacing human workers. Traditional requirements might not adequately address these.
        
        #### Algorithm accountability
        
        Algorithmic bias refers to the systematic errors or unfair prejudices that can be introduced into the 
        outputs of AI and machine learning algorithms. Algorithm accountability refers to the idea that algorithms, 
        especially those used in AI systems, should be transparent, explainable, and subject to oversight 
        and accountability measures.
        """)
    
    # The AWS Shared Responsibility Model
    st.markdown("### The AWS Shared Responsibility Model")
    
    st.markdown("""
    The AWS Shared Responsibility Model clarifies the division of security responsibilities between AWS and its customers.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### AWS Responsibilities: Security OF the cloud
        
        - Physical security of data centers
        - Hardware and software infrastructure
        - Network infrastructure
        - Virtualization infrastructure
        - Service-specific security features
        
        For AI services like Amazon Bedrock, AWS is responsible for:
        - Securing the underlying infrastructure
        - Protecting the base models
        - Implementing service-level security controls
        """)
    
    with col2:
        st.markdown("""
        #### Customer Responsibilities: Security IN the cloud
        
        - Data security and encryption
        - Identity and access management
        - Network and firewall configurations
        - Client and endpoint security
        - Maintaining compliance
        
        For AI services, customers are responsible for:
        - Securing data used with AI services
        - Managing access to AI services
        - Implementing appropriate controls for AI outputs
        - Ensuring compliant usage of AI services
        """)
    
    # Create a visualization of the shared responsibility model
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the height of each section
    customer_height = 0.6
    aws_height = 0.4
    
    # Create the customer responsibility section
    customer = plt.Rectangle((0, aws_height), 1, customer_height, facecolor='#FF9900', alpha=0.8)
    ax.add_patch(customer)
    
    # Create the AWS responsibility section
    aws = plt.Rectangle((0, 0), 1, aws_height, facecolor='#232F3E', alpha=0.8)
    ax.add_patch(aws)
    
    # Add text for customer responsibilities
    ax.text(0.5, aws_height + customer_height/2, 'Customer\nResponsibility\n\nSecurity "IN" the Cloud',
            ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    
    # Add examples of customer responsibilities
    customer_examples = [
        'Data security',
        'IAM',
        'Network config',
        'Client security',
        'AI model governance',
        'AI usage compliance'
    ]
    
    for i, example in enumerate(customer_examples):
        x = 0.14 + (i % 3) * 0.3
        y = aws_height + 0.1 + (i // 3) * 0.3
        ax.text(x, y, example, ha='center', va='center', color='white', fontsize=8)
    
    # Add text for AWS responsibilities
    ax.text(0.5, aws_height/2, 'AWS Responsibility\n\nSecurity "OF" the Cloud',
            ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    
    # Add examples of AWS responsibilities
    aws_examples = [
        'Physical security',
        'Compute',
        'Storage',
        'Networking',
        'Database',
        'Base AI models'
    ]
    
    for i, example in enumerate(aws_examples):
        x = 0.14 + (i % 3) * 0.3
        y = 0.05 + (i // 3) * 0.15
        ax.text(x, y, example, ha='center', va='center', color='white', fontsize=8)
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('AWS Shared Responsibility Model for AI Services')
    
    # Display the plot
    st.pyplot(fig)
    
    # Defense in depth
    st.markdown("### Defense in Depth for AI Governance")
    
    st.markdown("""
    A **defense in depth** security strategy uses multiple redundant defenses to protect your AWS accounts, workloads, 
    data, and assets. This strategy applies to AI governance through layered controls:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Identity and Access Controls")
        st.markdown("""
        - IAM roles with least privilege
        - Multi-factor authentication
        - Resource-based policies
        - Service control policies
        """)
    
    with col2:
        st.markdown("#### Data Protection")
        st.markdown("""
        - Encryption at rest and in transit
        - Data governance policies
        - Privacy-enhancing technologies
        - Data classification
        """)
    
    with col3:
        st.markdown("#### Monitoring and Detection")
        st.markdown("""
        - CloudTrail for API auditing
        - CloudWatch for metrics and alarms
        - Model monitoring for drift
        - Anomaly detection
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### AI-Specific Controls")
        st.markdown("""
        - Input validation and sanitization
        - Output filtering
        - Responsible AI policies
        - Model risk management
        """)
    
    with col2:
        st.markdown("#### Infrastructure Security")
        st.markdown("""
        - VPC isolation
        - Security groups
        - Network ACLs
        - Private endpoints
        """)
    
    with col3:
        st.markdown("#### Governance and Compliance")
        st.markdown("""
        - Policy documentation
        - Regular auditing
        - Risk assessments
        - Compliance monitoring
        """)
    
    # Data governance strategies
    st.markdown("### Data Governance Strategies for AI")
    
    st.markdown("""
    Data governance strategies for AI involve a comprehensive approach to managing the data lifecycle, from 
    data collection and storage, to data usage and security.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Data Quality and Integrity
        
        - Establish data quality standards and processes
        - Implement validation and cleansing techniques
        - Maintain data lineage and provenance
        
        #### Data Protection and Privacy
        
        - Develop and enforce data privacy policies
        - Implement access controls and encryption
        - Establish data breach response procedures
        
        #### Data Lifecycle Management
        
        - Classify and catalog data assets
        - Implement retention and disposition policies
        - Develop backup and recovery strategies
        """)
    
    with col2:
        st.markdown("""
        #### Responsible AI
        
        - Establish responsible AI frameworks
        - Implement bias monitoring processes
        - Educate teams on responsible AI practices
        
        #### Governance Structure and Roles
        
        - Establish a data governance council
        - Define clear roles and responsibilities
        - Provide training on governance best practices
        
        #### Data Sharing and Collaboration
        
        - Develop data sharing agreements
        - Implement data virtualization techniques
        - Foster a culture of data-driven decision-making
        """)
    
    # AWS services for governance and compliance
    st.markdown("### AWS Services for AI Governance and Compliance")
    
    services_data = [
        {
            "name": "AWS Audit Manager",
            "description": "Continuously auditing AWS usage to simplify risk and compliance management",
            "features": [
                "Automates evidence collection",
                "Assesses control effectiveness",
                "Provides assessment reports",
                "Streamlines audit preparation",
                "Offers prebuilt compliance frameworks"
            ],
            "use_case": "Demonstrate compliance with regulations and standards applicable to AI systems"
        },
        {
            "name": "AWS Artifact",
            "description": "On-demand access to security and compliance reports and agreements",
            "features": [
                "Access compliance reports from third-party auditors",
                "Review and accept agreements",
                "Access AWS security and compliance documentation",
                "Download attestations of compliance"
            ],
            "use_case": "Provide documentation demonstrating AWS compliance with regulations relevant to AI deployments"
        },
        {
            "name": "AWS Config",
            "description": "Assess, audit, and evaluate resource configurations",
            "features": [
                "Automatic resource discovery",
                "Configuration change tracking",
                "Compliance evaluation",
                "Automated remediation"
            ],
            "use_case": "Monitor and maintain compliance of AI infrastructure configurations"
        },
        {
            "name": "AWS CloudTrail",
            "description": "Track user activity and API usage",
            "features": [
                "Records API calls",
                "User activity tracking",
                "Event history",
                "Log file integrity validation"
            ],
            "use_case": "Maintain an audit trail of all actions taken on AI resources and models"
        },
        {
            "name": "Amazon CloudWatch",
            "description": "Monitor resources and applications",
            "features": [
                "Resource monitoring",
                "Metric collection",
                "Alarm configuration",
                "Log analysis"
            ],
            "use_case": "Monitor AI model performance and set alerts for compliance violations"
        },
        {
            "name": "AWS Trusted Advisor",
            "description": "Recommendations to follow AWS best practices",
            "features": [
                "Cost optimization",
                "Performance improvement",
                "Security risk identification",
                "Fault tolerance enhancement"
            ],
            "use_case": "Identify security risks and compliance gaps in AI deployments"
        },
        {
            "name": "Amazon SageMaker Model Cards",
            "description": "Document and share model information",
            "features": [
                "Centralized model documentation",
                "Risk rating documentation",
                "Model performance tracking",
                "Intended use documentation"
            ],
            "use_case": "Document model details, intended use, limitations, and risk ratings for governance and compliance"
        },
        {
            "name": "Amazon SageMaker Model Dashboard",
            "description": "Unified view of model performance",
            "features": [
                "Model monitoring",
                "Performance tracking",
                "Drift detection",
                "Alert configuration"
            ],
            "use_case": "Monitor model behavior to ensure ongoing compliance with governance requirements"
        },
        {
            "name": "Amazon SageMaker Role Manager",
            "description": "Simplified permission management",
            "features": [
                "Role-based access control",
                "Custom policy generation",
                "Least privilege enforcement",
                "User onboarding"
            ],
            "use_case": "Implement appropriate access controls for AI resources in compliance with governance requirements"
        }
    ]
    
    # Create tabs for each service
    service_tabs = st.tabs([service["name"] for service in services_data])
    
    # Add content to each tab
    for i, tab in enumerate(service_tabs):
        service = services_data[i]
        with tab:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Placeholder for service icon
                st.markdown(f"### {service['name']}")
            
            with col2:
                st.markdown(f"**{service['description']}**")
                
                st.markdown("##### Key Features:")
                features_list = ""
                for feature in service['features']:
                    features_list += f"- {feature}\n"
                st.markdown(features_list)
                
                st.markdown("##### Governance & Compliance Use Case:")
                st.markdown(service['use_case'])
                
                # Add specific examples for select services
                if service['name'] == "AWS Audit Manager":
                    st.markdown("##### Example Compliance Framework:")
                    st.markdown("""
                    **HIPAA Compliance Framework**
                    
                    AWS Audit Manager provides a prebuilt framework for HIPAA compliance that includes:
                    - Controls mapped to HIPAA requirements
                    - Automated evidence collection
                    - Assessment reports for audit preparation
                    
                    This framework helps healthcare organizations demonstrate compliance when using AI for processing protected health information (PHI).
                    """)
                
                elif service['name'] == "AWS Config":
                    st.markdown("##### Example Compliance Rule:")
                    st.code("""
                    {
                      "ConfigRuleName": "sagemaker-notebook-encryption",
                      "Description": "Checks if SageMaker notebooks have encryption enabled",
                      "Scope": {
                        "ComplianceResourceTypes": [
                          "AWS::SageMaker::NotebookInstance"
                        ]
                      },
                      "Source": {
                        "Owner": "AWS",
                        "SourceIdentifier": "SAGEMAKER_NOTEBOOK_INSTANCE_KMS_KEY_CONFIGURED"
                      }
                    }
                    """)
                
                elif service['name'] == "Amazon SageMaker Model Cards":
                    st.markdown("##### Example Model Card Structure:")
                    st.markdown("""
                    **Model Overview**
                    - Model Name: CustomerSentimentAnalysis
                    - Version: 1.3.1
                    - Model Type: Text Classification
                    - Framework: PyTorch 2.0
                    
                    **Intended Use**
                    - Primary Use Case: Analyze customer feedback sentiment
                    - Intended Users: Customer Support, Product Management
                    - Out-of-scope Uses: Automated decision-making without human review
                    
                    **Risk Rating**
                    - Risk Level: Medium
                    - Rationale: Models human opinions, potential for bias
                    
                    **Ethical Considerations**
                    - Potential biases in training data
                    - Limitations in detecting sarcasm or cultural nuances
                    - Mitigation: Regular bias monitoring and human review
                    
                    **Training Data**
                    - Data Sources: Customer feedback form responses
                    - Time Period: January 2023 - March 2025
                    - Data Processing: PII removed, balanced sentiment classes
                    """)
    
    # Regulations and standards
    st.markdown("### Key Regulations and Standards for AI Systems")
    
    # Create tabbed interface for different regions/regulatory domains
    regulation_tabs = st.tabs(["Global Standards", "US Regulations", "EU Regulations", "Industry-Specific", "Emerging Standards"])
    
    with regulation_tabs[0]:  # Global Standards
        st.markdown("""
        #### ISO/IEC Standards
        
        - **ISO/IEC 42001** - Artificial intelligence management system
        - **ISO/IEC 23894** - Risk management guidelines for AI
        - **ISO/IEC 38507** - Governance implications of AI for organizations
        - **ISO/IEC TR 24028** - Overview of trustworthiness in AI
        
        #### IEEE Standards
        
        - **IEEE 7000** - Model process for addressing ethical concerns
        - **IEEE 7001** - Transparency of autonomous systems
        - **IEEE 7010** - Well-being metrics for autonomous systems
        """)
    
    with regulation_tabs[1]:  # US Regulations
        st.markdown("""
        #### Federal Initiatives
        
        - **US AI Risk Management Framework (NIST)**
          - Voluntary framework to manage risks in AI systems
          - Focuses on governance, mapping, measurement, and management
        
        - **Blueprint for an AI Bill of Rights (White House OSTP)**
          - Non-binding guidance focusing on five principles:
          - Safe and effective systems
          - Algorithmic discrimination protections
          - Data privacy
          - Notice and explanation
          - Human alternatives and fallbacks
        
        #### State Regulations
        
        - **California (SB 1047)** - Requires safety testing for frontier AI models
        - **Colorado (SB23-169)** - Insurance algorithm fairness
        - **Illinois (BIPA)** - Biometric privacy protections applicable to AI
        """)
    
    with regulation_tabs[2]:  # EU Regulations
        st.markdown("""
        #### EU AI Act
        
        The EU AI Act is a comprehensive regulatory framework for AI systems based on a risk-based approach:
        
        - **Unacceptable Risk**: Prohibited AI applications (e.g., social scoring, manipulative AI)
        
        - **High Risk**: AI systems that must comply with strict requirements including:
          - Risk management systems
          - Data governance
          - Technical documentation
          - Record-keeping
          - Transparency
          - Human oversight
          - Accuracy, robustness, and cybersecurity
        
        - **Limited Risk**: Transparency obligations (e.g., chatbots must disclose they are AI)
        
        - **Minimal Risk**: Minimal or no regulations (most AI systems)
        
        #### General Data Protection Regulation (GDPR)
        
        Articles relevant to AI systems include:
        
        - **Article 22**: Right not to be subject to automated decision-making
        - **Article 13-15**: Right to explanation of automated decisions
        - **Article 35**: Data Protection Impact Assessments
        - **Article 25**: Data protection by design and default
        """)
    
    with regulation_tabs[3]:  # Industry-Specific
        st.markdown("""
        #### Financial Services
        
        - **SR 11-7** (Fed guidance on model risk management)
        - **FINRA** guidance on AI use in broker-dealer activities
        - **Consumer Financial Protection Bureau (CFPB)** guidance on algorithmic appraisals
        - **New York DFS** guidance on use of AI in insurance
        
        #### Healthcare
        
        - **FDA** regulatory framework for AI/ML-based Software as a Medical Device (SaMD)
        - **HIPAA** regulations for protected health information used in AI
        - **21st Century Cures Act** provisions on health information technology
        
        #### Human Resources
        
        - **EEOC** guidance on AI in hiring and other employment decisions
        - **New York City Local Law 144** - Bias audits of automated employment decision tools
        - **Illinois AI Video Interview Act** - Requires consent and transparency
        """)
    
    with regulation_tabs[4]:  # Emerging Standards
        st.markdown("""
        #### Emerging Frameworks and Guidelines
        
        - **OECD AI Principles**
          - First intergovernmental standard on AI
          - Focuses on inclusive growth, human-centered values, transparency, robustness, and accountability
        
        - **UN Recommendations on Ethics of AI**
          - Framework addressing ethical issues related to AI
          - Focus on human rights, environmental protection, and sustainable development
        
        - **G20 AI Principles**
          - Based on OECD principles
          - Endorsed by G20 countries to promote responsible AI
        
        - **Industry Initiatives**
          - **Partnership on AI**: Multi-stakeholder initiative to develop best practices
          - **MLOps Foundation**: Standards for operationalizing AI models
          - **Open AI Model Cards**: Industry templates for model documentation
        """)
    
    # Data governance frameworks
    st.markdown("### Data Governance Frameworks for AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### DAMA-DMBOK Framework
        
        The **Data Management Body of Knowledge** provides a comprehensive framework for data governance:
        
        1. **Data Architecture**: Blueprint for data management
        2. **Data Modeling**: Structure and relationships between data
        3. **Data Storage**: Management of stored data
        4. **Data Security**: Protection of data assets
        5. **Data Quality**: Ensuring data is fit for purpose
        6. **Master Data Management**: Consistency of critical data
        7. **Data Warehousing**: Storage for analytics
        8. **Metadata Management**: Information about data
        9. **Document Management**: Content and records
        """)
    
    with col2:
        st.markdown("""
        #### AI-specific Considerations
        
        When applying data governance to AI systems, consider:
        
        1. **Training Data Governance**:
           - Documentation of data sources and lineage
           - Version control of training datasets
           - Assessment of bias and representativeness
        
        2. **Model Governance**:
           - Model inventories and documentation
           - Version control of models
           - Performance monitoring and drift detection
           - Explainability requirements
        
        3. **Output Governance**:
           - Quality control of AI-generated content
           - Guidelines for human review
           - Content filtering and safety measures
           - Feedback loops for continuous improvement
        """)
    
    # Responsible AI governance
    st.markdown("### Responsible AI Governance")
    
    st.markdown("""
    Responsible AI governance incorporates ethical considerations, fairness, transparency, and accountability 
    into the development and deployment of AI systems.
    """)
    
    # Create a tabbed interface for governance frameworks
    governance_tabs = st.tabs(["AWS Responsible AI", "NIST AI RMF", "Implementation Steps"])
    
    with governance_tabs[0]:  # AWS Responsible AI
        st.markdown("""
        #### AWS Responsible AI Principles
        
        1. **Fairness**: Considering impacts on different groups of stakeholders
        
        2. **Explainability**: Understanding and evaluating system outputs
        
        3. **Controllability**: Having mechanisms to monitor and steer AI system behavior
        
        4. **Privacy & Security**: Appropriately obtaining, using and protecting data and models
        
        5. **Governance**: Incorporating best practices into the AI supply chain
        
        6. **Transparency**: Enabling stakeholders to make informed choices
        
        7. **Safety**: Preventing harmful system output and misuse
        
        8. **Veracity & Robustness**: Achieving correct system outputs, even with unexpected inputs
        """)
        
        st.markdown("""
        #### AWS Responsible AI Services
        
        - **Amazon SageMaker Clarify**: Detect bias in ML models and understand model predictions
        - **Amazon SageMaker Model Cards**: Document model details, intended use, and limitations
        - **Amazon SageMaker Model Dashboard**: Monitor model behavior for bias and drift
        - **Amazon Augmented AI (A2I)**: Implement human review of ML predictions
        - **Amazon Bedrock Guardrails**: Define content filtering policies for LLMs
        """)
    
    with governance_tabs[1]:  # NIST AI RMF
        st.markdown("""
        #### NIST AI Risk Management Framework
        
        The NIST AI Risk Management Framework is organized into four core functions:
        
        1. **Govern**:
           - Risk governance and strategy
           - Defining roles and responsibilities
           - Establishing oversight structure
           - Workforce development for AI risk
        
        2. **Map**:
           - Context identification
           - Risk identification
           - Risk classification and prioritization
           - Risk tolerance determination
        
        3. **Measure**:
           - Risk assessment
           - Risk analysis and tracking
           - Monitoring AI systems
           - Documentation of risks
        
        4. **Manage**:
           - Risk treatment and mitigation
           - Risk communication
           - Resilience planning
           - Continuous improvement
        """)
        
        # Create a simple visualization for NIST AI RMF
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create cycle visualization
        angles = np.linspace(0, 2*np.pi, 5, endpoint=True)
        functions = ['Govern', 'Map', 'Measure', 'Manage']
        
        # Plot circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linewidth=2)
        ax.add_patch(circle)
        
        # Plot function names at positions around the circle
        for i, func in enumerate(functions):
            angle = angles[i]
            x = 1.2 * np.cos(angle)
            y = 1.2 * np.sin(angle)
            ax.text(x, y, func, ha='center', va='center', fontsize=14, fontweight='bold', color='#232F3E')
            
            # Create an arrow between functions
            next_angle = angles[(i+1) % 4]
            arrow_x = 0.8 * np.cos(angle + (next_angle - angle)/2)
            arrow_y = 0.8 * np.sin(angle + (next_angle - angle)/2)
            dx = 0.2 * np.cos(next_angle - angle)
            dy = 0.2 * np.sin(next_angle - angle)
            ax.arrow(arrow_x, arrow_y, dx, dy, head_width=0.1, head_length=0.1, fc='#FF9900', ec='#FF9900')
        
        # Add center text
        ax.text(0, 0, "NIST\nAI Risk\nManagement\nFramework", ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Set axis limits and remove ticks
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        st.pyplot(fig)
    
    with governance_tabs[2]:  # Implementation Steps
        st.markdown("""
        #### Implementing Responsible AI Governance
        
        1. **Establish Leadership and Oversight**
           - Form a cross-functional AI governance committee
           - Define roles and responsibilities
           - Develop AI ethics principles and policies
        
        2. **Create a Responsible AI Framework**
           - Define risk assessment methodology
           - Establish development guidelines
           - Create documentation requirements
           - Define testing and validation protocols
        
        3. **Implement Technical Controls**
           - Deploy bias detection and mitigation tools
           - Implement model explainability mechanisms
           - Establish monitoring for drift and performance
           - Create audit trails for model decisions
        
        4. **Operationalize Policies**
           - Train teams on responsible AI practices
           - Integrate governance into development workflows
           - Implement review and approval processes
           - Create reporting and escalation procedures
        
        5. **Continuous Improvement**
           - Regularly assess effectiveness of controls
           - Update policies based on emerging risks
           - Monitor regulatory developments
           - Engage with industry best practices
        """)
        
        st.info("""
        **Implementation Example:** A financial services company implemented responsible AI governance by:
        
        1. Creating a cross-functional AI ethics committee with representatives from legal, compliance, data science, and business units
        2. Developing a tiered risk assessment framework for AI use cases
        3. Implementing technical controls including SageMaker Clarify for bias detection and Model Cards for documentation
        4. Requiring human review of high-risk model decisions using Amazon A2I
        5. Conducting quarterly reviews of model performance and compliance with policies
        """)
    
    # Generative AI Governance Challenges
    st.markdown("### Generative AI Governance Challenges")
    
    st.markdown("""
    Generative AI presents unique governance challenges compared to traditional AI systems:
    """)
    
    challenges = {
        "Content Generation": "Models can create content that is harmful, misleading, or violates copyright",
        "Rapid Evolution": "Foundation models evolve rapidly, making governance frameworks quickly outdated",
        "Black Box Nature": "Complex foundation models are often opaque, making oversight difficult",
        "Unpredictable Outputs": "Models can generate unexpected outputs that bypass safeguards",
        "Copyright and IP": "Unclear legal status of AI-generated content and training data usage",
        "Prompt Engineering": "Security vulnerabilities through adversarial prompting",
        "Supply Chain Risk": "Limited visibility into model development process"
    }
    
    # Create a challenge matrix
    challenges_df = pd.DataFrame({
        "Challenge": list(challenges.keys()),
        "Description": list(challenges.values()),
        "Impact Level": [4, 3, 5, 4, 3, 4, 3]
    })
    
    # Create a color scale for the impact level
    def color_impact_level(val):
        color = ''
        if val <= 2:
            color = 'background-color: #ccffcc'
        elif val <= 3:
            color = 'background-color: #ffffcc'
        elif val <= 4:
            color = 'background-color: #ffebcc'
        else:
            color = 'background-color: #ffcccc'
        return color
    
    # Apply the color styling
    styled_challenges = challenges_df.style.applymap(color_impact_level, subset=['Impact Level'])
    
    st.table(styled_challenges)
    
    # Governance strategies for generative AI
    st.markdown("""
    #### Governance Strategies for Generative AI
    
    1. **Content Filtering and Safety Mechanisms**
       - Implement input filtering and content moderation
       - Use Amazon Bedrock Guardrails to define content policies
       - Establish human review for edge cases
    
    2. **Model Cards and Documentation**
       - Document model capabilities, limitations, and risks
       - Maintain transparency about training data sources
       - Use Amazon SageMaker Model Cards for standardized documentation
    
    3. **Usage Policies and Guidelines**
       - Define acceptable use cases and prohibited uses
       - Establish guidelines for prompt construction
       - Create escalation procedures for policy violations
    
    4. **Monitoring and Auditing**
       - Monitor model outputs for policy compliance
       - Track user interactions for misuse patterns
       - Implement audit trails for model inputs and outputs
    
    5. **Regular Assessment and Updates**
       - Conduct regular red team exercises
       - Update governance policies as model capabilities evolve
       - Stay informed of emerging risks and mitigation techniques
    """)
    
    # Case study
    st.markdown("### Case Study: Implementing AI Governance in Healthcare")
    
    with st.expander("View Case Study"):
        st.markdown("""
        #### Background
        
        A large healthcare provider sought to implement an AI system to assist radiologists in early detection of lung cancer. 
        The system needed to comply with healthcare regulations while ensuring patient safety.
        
        #### Challenge
        
        The organization faced several governance challenges:
        
        - Ensuring compliance with HIPAA and FDA regulations
        - Managing patient data privacy and security
        - Implementing appropriate oversight for clinical decision support
        - Creating audit trails for AI-assisted diagnoses
        - Training staff on responsible AI use
        
        #### Solution: Comprehensive AI Governance Framework
        
        The healthcare provider implemented a multi-faceted governance approach:
        
        1. **Regulatory Compliance**
           - Conducted thorough regulatory analysis with legal and compliance teams
           - Implemented HIPAA-compliant data handling throughout the AI pipeline
           - Engaged with FDA early through the Software as a Medical Device (SaMD) pathway
        
        2. **Data Governance**
           - Established a data governance committee with clinical, technical, and ethics members
           - Implemented strict de-identification protocols for training data
           - Used AWS KMS for encryption of all patient data
           - Deployed Amazon Macie to detect sensitive information
        
        3. **Model Governance**
           - Created detailed model cards documenting model development, testing, and limitations
           - Implemented Amazon SageMaker Model Monitor for drift detection
           - Used SageMaker Clarify to detect potential biases across patient demographics
           - Established quarterly model review process with clinicians
        
        4. **Operational Controls**
           - Deployed the model with human-in-the-loop review using Amazon A2I
           - Created clear escalation paths for uncertain AI predictions
           - Implemented CloudTrail and CloudWatch for comprehensive audit trails
           - Set up automated alerts for anomalous model behavior
        
        5. **Organizational Measures**
           - Established an AI ethics committee with diverse representation
           - Created training programs for radiologists on AI capabilities and limitations
           - Developed clear documentation on the role of AI as a decision support tool
           - Implemented patient consent and education about AI use
        
        #### Results
        
        - Successfully deployed the AI system in compliance with all relevant regulations
        - Obtained necessary regulatory approvals through comprehensive documentation
        - Maintained average detection improvement of 23% while ensuring human oversight
        - Passed external audit of AI governance practices
        - Created a reusable governance framework for future AI initiatives
        
        #### Key Learnings
        
        - Early engagement with regulators provided clarity on compliance requirements
        - Cross-functional governance teams were essential for addressing diverse concerns
        - Regular assessment of model performance in production identified issues not found in testing
        - Treating governance as an ongoing process rather than a one-time effort was critical
        - Transparent communication with patients and providers built trust in the AI system
        """)
    
    # Interactive demo - AI governance assessment
    st.markdown("### Interactive Demo: AI Governance Assessment")
    
    st.markdown("""
    This interactive tool helps you assess the maturity of your AI governance program across key dimensions.
    Rate your organization's current state for each dimension on a scale of 1 (Initial) to 5 (Optimized).
    """)
    
    governance_dimensions = [
        "AI Policies and Standards",
        "Risk Management Framework",
        "Model Documentation",
        "Monitoring and Controls",
        "Explainability and Transparency",
        "Data Governance",
        "Ethical Use Guidelines"
    ]
    
    # Create sliders for each dimension
    dimension_scores = {}
    for dimension in governance_dimensions:
        score = st.slider(
            f"{dimension}",
            min_value=1,
            max_value=5,
            value=3,
            help=f"1: Initial, 2: Developing, 3: Defined, 4: Managed, 5: Optimized"
        )
        dimension_scores[dimension] = score
    
    # Create a radar chart with the scores
    categories = list(dimension_scores.keys())
    values = list(dimension_scores.values())
    
    # Close the loop for the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 153, 0, 0.5)',
        line=dict(color='#FF9900'),
        name='Your Organization'
    ))
    
    # Add an "ideal state" for comparison
    ideal_values = [5] * len(categories)
    fig.add_trace(go.Scatterpolar(
        r=ideal_values,
        theta=categories,
        fill=None,
        line=dict(color='#232F3E', dash='dash'),
        name='Ideal State'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        title="AI Governance Maturity Assessment",
        showlegend=True
    )
    
    st.plotly_chart(fig)
    
    # Calculate overall maturity score
    avg_score = sum(dimension_scores.values()) / len(dimension_scores)
    st.markdown(f"### Overall Maturity: {avg_score:.1f}/5.0")
    
    # Provide recommendations based on the assessment
    st.markdown("### Recommendations for Improvement")
    
    # Find the lowest scoring dimensions
    lowest_dims = sorted(dimension_scores.items(), key=lambda x: x[1])[:3]
    
    for dim, score in lowest_dims:
        st.markdown(f"#### {dim}: {score}/5")
        
        if dim == "AI Policies and Standards":
            st.markdown("""
            - Develop formal AI policy documentation covering responsible use
            - Align policies with relevant regulations and industry standards
            - Implement policy review and approval processes
            - Conduct regular policy reviews as regulations evolve
            """)
        
        elif dim == "Risk Management Framework":
            st.markdown("""
            - Adopt a structured approach like the NIST AI RMF
            - Implement risk assessment for all AI initiatives
            - Define risk thresholds and required mitigations
            - Establish regular risk review cadence
            """)
        
        elif dim == "Model Documentation":
            st.markdown("""
            - Implement standardized model cards for all AI models
            - Document model purpose, limitations, and risks
            - Record training data sources and processing steps
            - Maintain version control for models and documentation
            """)
        
        elif dim == "Monitoring and Controls":
            st.markdown("""
            - Deploy comprehensive model monitoring solutions
            - Set up alerts for drift and performance degradation
            - Implement automated and manual validation processes
            - Establish incident response procedures for AI systems
            """)
        
        elif dim == "Explainability and Transparency":
            st.markdown("""
            - Implement tools like SageMaker Clarify for model explanations
            - Create user-friendly explanations of model decisions
            - Document model limitations and confidence levels
            - Balance transparency requirements with model performance
            """)
        
        elif dim == "Data Governance":
            st.markdown("""
            - Establish data quality standards for AI training data
            - Implement data lineage tracking for AI datasets
            - Create data review and approval processes
            - Conduct regular data quality assessments
            """)
        
        elif dim == "Ethical Use Guidelines":
            st.markdown("""
            - Develop AI ethics principles for your organization
            - Create clear guidelines for acceptable AI use cases
            - Implement ethics review for high-risk applications
            - Provide ethics training for AI practitioners
            """)

# Knowledge Check Tab Content
def knowledge_check_tab():
    st.markdown("# Knowledge Check")
    st.markdown("Test your understanding of security, compliance, and governance for AI solutions.")
    
    # Questions for the knowledge check
    questions = [
        {
            "question": "What is responsible AI primarily concerned with?",
            "type": "single",
            "options": {
                "a": "Maximizing AI system performance regardless of societal impact",
                "b": "Ethical and trustworthy development, deployment, and use of AI systems",
                "c": "Using AI exclusively for approved government applications",
                "d": "Limiting AI capabilities to prevent technological singularity"
            },
            "correct": "b",
            "explanation": "Responsible AI refers to the ethical and trustworthy development, deployment, and use of AI systems. It focuses on ensuring AI systems are designed and used in a way that benefits society, respects human rights, and mitigates potential risks or harms."
        },
        {
            "question": "Which AWS service helps detect bias in machine learning models and explain model predictions?",
            "type": "single",
            "options": {
                "a": "Amazon Comprehend",
                "b": "Amazon Bedrock",
                "c": "Amazon SageMaker Clarify",
                "d": "AWS Glue"
            },
            "correct": "c",
            "explanation": "Amazon SageMaker Clarify provides machine learning developers with greater visibility into their training data and models so they can identify and limit bias and explain predictions. It includes features for both bias detection and model explainability."
        },
        {
            "question": "Which of the following are key components of model bias versus variance trade-off? (Select all that apply)",
            "type": "multiple",
            "options": {
                "a": "Underfitting occurs when a model has high bias and low variance",
                "b": "Overfitting occurs when a model has high variance and low bias",
                "c": "A model with low bias and low variance is the ideal goal",
                "d": "Adding more features always reduces both bias and variance"
            },
            "correct": ["a", "b", "c"],
            "explanation": "The bias-variance tradeoff involves balancing between underfitting (high bias, low variance) and overfitting (low bias, high variance). The ideal model has both low bias and low variance, achieving good performance on both training and test data. Adding more features typically reduces bias but increases variance, not both."
        },
        {
            "question": "What is the primary purpose of Amazon SageMaker Model Cards?",
            "type": "single",
            "options": {
                "a": "To provide computing resources for model training",
                "b": "To document and share model information including purpose, performance, and limitations",
                "c": "To automatically optimize hyperparameters for models",
                "d": "To convert models to different frameworks"
            },
            "correct": "b",
            "explanation": "Amazon SageMaker Model Cards helps you create a single source of truth for model information by centralizing and standardizing ML model documentation throughout the model lifecycle. You can record details such as model purpose, performance goals, and limitations, while SageMaker Model Cards auto-populates training details to accelerate the documentation process."
        },
        {
            "question": "In the context of AI security, what does a 'defense in depth' strategy involve?",
            "type": "single",
            "options": {
                "a": "Using only the most advanced AI models to prevent attacks",
                "b": "Implementing multiple layers of security controls to protect AI systems",
                "c": "Training AI systems to detect and counter cyber threats automatically",
                "d": "Limiting AI functionality to minimize potential security vulnerabilities"
            },
            "correct": "b",
            "explanation": "A defense in depth security strategy uses multiple redundant defenses to protect AWS accounts, workloads, data, and assets. It helps ensure that if any one security control is compromised or fails, additional layers exist to help isolate threats and prevent, detect, respond, and recover from security events."
        },
        {
            "question": "Which of the following are common security threats to AI systems? (Select all that apply)",
            "type": "multiple",
            "options": {
                "a": "Prompt injection",
                "b": "Data poisoning",
                "c": "Model extraction",
                "d": "Cloud migration"
            },
            "correct": ["a", "b", "c"],
            "explanation": "Common security threats to AI systems include prompt injection (manipulating input prompts to generate unintended outputs), data poisoning (introducing malicious data into training datasets), and model extraction (stealing model parameters through repeated API calls). Cloud migration is a process of moving applications to the cloud, not a security threat to AI systems."
        },
        {
            "question": "Which AWS service provides on-demand access to security and compliance reports and agreements?",
            "type": "single",
            "options": {
                "a": "AWS Config",
                "b": "AWS Artifact",
                "c": "AWS Trusted Advisor",
                "d": "AWS CloudTrail"
            },
            "correct": "b",
            "explanation": "AWS Artifact is a service that provides on-demand access to AWS security and compliance reports and select online agreements. It consists of AWS Artifact Agreements and AWS Artifact Reports, which provide compliance reports from third-party auditors who have tested and verified AWS's compliance with various security standards."
        },
        {
            "question": "What is the AWS Shared Responsibility Model in the context of AI services?",
            "type": "single",
            "options": {
                "a": "AWS is responsible for securing everything related to AI services",
                "b": "Customers are solely responsible for securing AI workloads",
                "c": "AWS is responsible for security OF the cloud, while customers are responsible for security IN the cloud",
                "d": "Third-party auditors are responsible for securing AI systems"
            },
            "correct": "c",
            "explanation": "The AWS Shared Responsibility Model states that AWS is responsible for security OF the cloud (infrastructure), while customers are responsible for security IN the cloud (data, access management, etc.). For AI services, AWS secures the underlying infrastructure and base models, while customers are responsible for securing data, managing access, and ensuring compliant usage of AI services."
        },
        {
            "question": "Which of the following are key data governance strategies for AI workloads? (Select all that apply)",
            "type": "multiple",
            "options": {
                "a": "Data quality and integrity",
                "b": "Data protection and privacy",
                "c": "Data lifecycle management",
                "d": "Data maximization"
            },
            "correct": ["a", "b", "c"],
            "explanation": "Key data governance strategies for AI include data quality and integrity (ensuring accurate and consistent data), data protection and privacy (safeguarding sensitive information), and data lifecycle management (classifying, retaining, and archiving data appropriately). Data maximization is not a recognized governance strategy; in fact, data minimization is often recommended as a privacy best practice."
        },
        {
            "question": "What are the unique challenges of AI standards compliance compared to traditional software? (Select all that apply)",
            "type": "multiple",
            "options": {
                "a": "Complexity and opacity of AI systems",
                "b": "Dynamism and adaptability of AI systems",
                "c": "Emergent capabilities in AI systems",
                "d": "Lower cost of AI system development"
            },
            "correct": ["a", "b", "c"],
            "explanation": "AI standards compliance faces unique challenges including: complexity and opacity (AI systems can be highly complex with opaque decision-making), dynamism and adaptability (AI systems can change over time), and emergent capabilities (unexpected capabilities that arise from complex interactions). The cost of AI system development is often higher than traditional software, not lower, due to these complexities and the resources required for training."
        },
        {
            "question": "What AWS service allows you to track user activity and API usage across your AWS environment?",
            "type": "single",
            "options": {
                "a": "Amazon Inspector",
                "b": "AWS CloudTrail",
                "c": "Amazon Macie",
                "d": "AWS Config"
            },
            "correct": "b",
            "explanation": "AWS CloudTrail records API calls for your AWS account, providing a history of user activity and API calls for your applications and resources. This allows you to track changes to resources, identify unusual activity, and maintain an audit trail for compliance purposes."
        },
        {
            "question": "What is the purpose of the NIST AI Risk Management Framework?",
            "type": "single",
            "options": {
                "a": "To prohibit certain high-risk AI applications",
                "b": "To provide a structure for managing risks in AI systems across the AI lifecycle",
                "c": "To certify AI systems for government use",
                "d": "To standardize AI programming languages"
            },
            "correct": "b",
            "explanation": "The NIST AI Risk Management Framework provides a structured approach to managing risks in AI systems across their lifecycle. It is organized into four core functions: Govern (strategy and oversight), Map (context and risk identification), Measure (risk assessment and monitoring), and Manage (risk treatment and improvement)."
        }
    ]
    
    # Display progress
    if st.session_state.knowledge_check_progress > 0:
        progress_text = f"Question {st.session_state.knowledge_check_progress}/{len(questions)}"
        st.progress(st.session_state.knowledge_check_progress/len(questions), text=progress_text)
    
    # Add a restart button
    if st.session_state.knowledge_check_progress > 0:
        if st.button("Restart Knowledge Check",key='restart_kb'):
            reset_knowledge_check()
            return
    
    # Display the current question or final score
    if st.session_state.knowledge_check_progress < len(questions):
        current_q = questions[st.session_state.knowledge_check_progress]
        
        st.markdown(f"### Question {st.session_state.knowledge_check_progress + 1}")
        st.markdown(current_q["question"])
        
        # For single selection questions
        if current_q["type"] == "single":
            answer = st.radio(
                "Select one answer:",
                list(current_q["options"].keys()),
                format_func=lambda x: f"{x}) {current_q['options'][x]}",
                index=None,
                key=f"q{st.session_state.knowledge_check_progress}_radio"
            )
            
            if st.button("Submit Answer"):
                if answer:
                    st.session_state.knowledge_check_answers[st.session_state.knowledge_check_progress] = answer
                    
                    if answer == current_q["correct"]:
                        st.session_state.knowledge_check_score += 1
                        st.session_state.knowledge_check_feedback[st.session_state.knowledge_check_progress] = "Correct! " + current_q["explanation"]
                    else:
                        st.session_state.knowledge_check_feedback[st.session_state.knowledge_check_progress] = f"Incorrect. The correct answer is {current_q['correct']}: {current_q['options'][current_q['correct']]}. " + current_q["explanation"]
                    
                    st.session_state.knowledge_check_progress += 1
                    st.rerun()
                else:
                    st.warning("Please select an answer before submitting.")
        
        # For multiple selection questions
        elif current_q["type"] == "multiple":
            options = {}
            for key in current_q["options"].keys():
                options[key] = st.checkbox(
                    f"{key}) {current_q['options'][key]}",
                    key=f"q{st.session_state.knowledge_check_progress}_{key}"
                )
            
            if st.button("Submit Answer"):
                selected = [k for k, v in options.items() if v]
                
                if selected:
                    st.session_state.knowledge_check_answers[st.session_state.knowledge_check_progress] = selected
                    
                    if set(selected) == set(current_q["correct"]):
                        st.session_state.knowledge_check_score += 1
                        st.session_state.knowledge_check_feedback[st.session_state.knowledge_check_progress] = "Correct! " + current_q["explanation"]
                    else:
                        correct_options = ", ".join([f"{k}: {current_q['options'][k]}" for k in current_q["correct"]])
                        st.session_state.knowledge_check_feedback[st.session_state.knowledge_check_progress] = f"Incorrect. The correct answers are: {correct_options}. " + current_q["explanation"]
                    
                    st.session_state.knowledge_check_progress += 1
                    st.rerun()
                else:
                    st.warning("Please select at least one answer before submitting.")
    
    else:
        # Display final score
        st.markdown("### Knowledge Check Complete!")
        st.markdown(f"Your score: {st.session_state.knowledge_check_score}/{len(questions)}")
        
        # Calculate percentage
        percentage = (st.session_state.knowledge_check_score / len(questions)) * 100
        
        # Display different messages based on score
        if percentage >= 90:
            st.success(f"Excellent! You scored {percentage:.1f}%. You have a strong understanding of AI security, compliance, and governance concepts.")
        elif percentage >= 75:
            st.success(f"Good job! You scored {percentage:.1f}%. You have a solid grasp of the key concepts.")
        elif percentage >= 60:
            st.warning(f"You scored {percentage:.1f}%. You understand some concepts but might want to review the material again.")
        else:
            st.error(f"You scored {percentage:.1f}%. It's recommended to review the material thoroughly before taking the AWS AI Practitioner exam.")
        
        # Show review of all questions
        st.markdown("### Review Your Answers")
        
        for i, question in enumerate(questions):
            with st.expander(f"Question {i+1}: {question['question']}"):
                if i in st.session_state.knowledge_check_answers:
                    if question["type"] == "single":
                        st.markdown(f"Your answer: {st.session_state.knowledge_check_answers[i]}) {question['options'][st.session_state.knowledge_check_answers[i]]}")
                        st.markdown(f"Correct answer: {question['correct']}) {question['options'][question['correct']]}")
                    else:
                        selected = ", ".join([f"{k}) {question['options'][k]}" for k in st.session_state.knowledge_check_answers[i]])
                        correct = ", ".join([f"{k}) {question['options'][k]}" for k in question['correct']])
                        st.markdown(f"Your answer: {selected}")
                        st.markdown(f"Correct answer: {correct}")
                    
                    st.markdown("#### Explanation")
                    st.markdown(question["explanation"])
                else:
                    st.markdown("Question not answered.")
        
        # Button to restart the knowledge check
        if st.button("Restart Knowledge Check"):
            reset_knowledge_check()
    
    # Display feedback for the previous question
    if st.session_state.knowledge_check_progress > 0 and st.session_state.knowledge_check_progress <= len(questions):
        prev_q_index = st.session_state.knowledge_check_progress - 1
        if prev_q_index in st.session_state.knowledge_check_feedback:
            if "Correct" in st.session_state.knowledge_check_feedback[prev_q_index]:
                st.success(st.session_state.knowledge_check_feedback[prev_q_index])
            else:
                st.error(st.session_state.knowledge_check_feedback[prev_q_index])

# Main application function
def main():
    # Initialize session state
    initialize_session_state()
    
    # Apply custom styling
    load_css()
    
    with st.sidebar:
        # Sidebar content
        common.render_sidebar()
        
        # Add About this App section to sidebar (collapsed by default)
        with st.expander("About this App"):
            st.markdown("""
            This interactive learning application covers Domain 4 (Guidelines for Responsible AI) and Domain 5 
            (Security, Compliance, and Governance for AI Solutions) of the AWS AI Practitioner Certification.
            
            **Topics covered:**
            - Responsible AI development
            - Transparent and explainable models
            - Methods to secure AI systems
            - Governance and compliance regulations for AI systems
            
            The app includes interactive examples, code snippets, and knowledge checks to help you prepare 
            for the AWS AI Practitioner certification exam.
            
            © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
        """)
    
    # Display home content as main page
    home_tab()
    
    # Add footer
    create_footer()

# Run the application
if __name__ == "__main__":
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import utils.authenticate as authenticate
from utils.styles import load_css, sub_header, custom_header

# Set page config
st.set_page_config(
    page_title="Dataset Bias Types",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# AWS Color scheme
COLORS = {
    'primary': '#FF9900',
    'secondary': '#232F3E',
    'success': '#3EB489',
    'warning': '#F2C94C',
    'danger': '#D13212',
    'info': '#0073BB',
    'light': '#F8F9FA',
    'border': '#E9ECEF'
}

def generate_sampling_bias_data():
    """Generate data demonstrating sampling bias"""
    np.random.seed(42)
    
    # True population distribution
    true_ages = np.random.normal(45, 15, 1000)
    true_ages = np.clip(true_ages, 18, 80)
    
    # Biased sample (only college students)
    biased_ages = np.random.normal(22, 3, 1000)
    biased_ages = np.clip(biased_ages, 18, 30)
    
    return true_ages, biased_ages

def generate_historical_bias_data():
    """Generate data demonstrating historical bias in hiring"""
    np.random.seed(42)
    
    # Historical hiring data (male-dominated)
    years = np.arange(1980, 2025)
    male_hired = np.array([85, 84, 83, 82, 81, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 
                           60, 58, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 
                           43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29])
    female_hired = 100 - male_hired
    
    return years, male_hired, female_hired

def generate_measurement_bias_data():
    """Generate data demonstrating measurement bias"""
    np.random.seed(42)
    
    # True satisfaction scores
    true_scores = np.random.normal(7, 1.5, 500)
    true_scores = np.clip(true_scores, 1, 10)
    
    # Biased scores (leading questions push scores higher)
    biased_scores = true_scores + np.random.normal(1.5, 0.5, 500)
    biased_scores = np.clip(biased_scores, 1, 10)
    
    return true_scores, biased_scores

def main():
    st.markdown(custom_header("Understanding Dataset Bias Types", 1), unsafe_allow_html=True)
    
    st.markdown("""
    Dataset bias refers to systematic skew or imbalance in data used to train machine learning models.
    When data collection doesn't represent real-world distribution, models can perpetuate or amplify biases,
    leading to unfair or inaccurate predictions.
    """)
    
    # Overview section
    st.markdown(sub_header("What is Dataset Bias?", "📚"), unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Dataset bias** occurs when training data systematically misrepresents the real-world problem.
        This happens when:
        
        - Data collection methods are flawed
        - Historical inequities are reflected in data
        - Measurement processes introduce errors
        - Certain groups are over/under-represented
        
        **Impact on AI Systems:**
        - Models learn and perpetuate biases
        - Unfair predictions for certain groups
        - Reduced accuracy for underrepresented populations
        - Potential legal and ethical issues
        - Loss of trust in AI systems
        """)
    
    with col2:
        st.info("""
        💡 **Key Insight**
        
        Biased data leads to biased models. 
        Even the most sophisticated algorithms 
        cannot overcome fundamentally flawed 
        training data.
        """)
        
        # Create a simple bias flow diagram
        fig = go.Figure()
        
        fig.add_trace(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Biased Data", "ML Model", "Biased Predictions", "Unfair Outcomes"],
                color=[COLORS['danger'], COLORS['warning'], COLORS['danger'], COLORS['danger']]
            ),
            link=dict(
                source=[0, 1, 2],
                target=[1, 2, 3],
                value=[10, 10, 10],
                color=[COLORS['danger'], COLORS['warning'], COLORS['danger']]
            )
        ))
        
        fig.update_layout(
            title="How Dataset Bias Propagates",
            height=250,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Three types of bias
    st.markdown(sub_header("Common Types of Dataset Bias", "🎯", "aws"), unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Sampling Bias",
        "📜 Historical Bias",
        "📏 Measurement Bias",
        "🔍 All Comparisons"
    ])
    
    # Tab 1: Sampling Bias
    with tab1:
        st.markdown("### Sampling Bias")
        
        st.markdown("""
        **Definition:** The data collected does not represent the true population.
        
        **What causes it:**
        - Convenience sampling (surveying only easily accessible people)
        - Self-selection bias (only motivated people respond)
        - Non-response bias (certain groups don't participate)
        - Geographic limitations
        """)
        
        st.markdown("#### Example: Age Distribution Survey")
        
        st.markdown("""
        **Scenario:** A company wants to understand customer preferences across all age groups.
        They survey college students on campus (convenient but biased sample).
        """)
        
        # Generate and visualize sampling bias
        true_ages, biased_ages = generate_sampling_bias_data()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("True Population Distribution", "Biased Sample (College Campus)")
        )
        
        # True population
        fig.add_trace(
            go.Histogram(x=true_ages, name="True Population", 
                        marker_color=COLORS['success'], nbinsx=30),
            row=1, col=1
        )
        
        # Biased sample
        fig.add_trace(
            go.Histogram(x=biased_ages, name="Biased Sample", 
                        marker_color=COLORS['danger'], nbinsx=30),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Age", row=1, col=1)
        fig.update_xaxes(title_text="Age", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Sampling Bias: Age Distribution Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("True Population Mean Age", f"{true_ages.mean():.1f} years")
            st.metric("True Population Std Dev", f"{true_ages.std():.1f} years")
        
        with col2:
            st.metric("Biased Sample Mean Age", f"{biased_ages.mean():.1f} years", 
                     delta=f"{biased_ages.mean() - true_ages.mean():.1f} years",
                     delta_color="inverse")
            st.metric("Biased Sample Std Dev", f"{biased_ages.std():.1f} years",
                     delta=f"{biased_ages.std() - true_ages.std():.1f} years",
                     delta_color="inverse")
        
        st.warning("""
        **Impact:** A model trained on the biased sample would:
        - Underrepresent older adults (45+ years)
        - Make poor predictions for middle-aged and senior customers
        - Miss important preferences of the majority demographic
        - Lead to business decisions that ignore 70% of the market
        """)
        
        st.markdown("#### How to Detect Sampling Bias")
        
        st.markdown("""
        1. **Compare sample demographics to known population statistics**
        2. **Check for underrepresented groups**
        3. **Analyze response rates across different segments**
        4. **Review data collection methodology**
        5. **Use statistical tests (Chi-square, KS test)**
        """)
        
        st.markdown("#### How to Mitigate Sampling Bias")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Collection:**
            - Use stratified sampling
            - Ensure representative sampling
            - Random sampling when possible
            - Oversample underrepresented groups
            """)
        
        with col2:
            st.markdown("""
            **Post-Collection:**
            - Apply sample weighting
            - Use data augmentation
            - Collect additional data for missing groups
            - Document sampling methodology
            """)
    
    # Tab 2: Historical Bias
    with tab2:
        st.markdown("### Historical Bias")
        
        st.markdown("""
        **Definition:** The data reflects past biases and inequities in society.
        
        **What causes it:**
        - Historical discrimination and prejudice
        - Past institutional policies
        - Societal norms that have changed
        - Legacy systems and processes
        """)
        
        st.markdown("#### Example: Gender Bias in Tech Hiring")
        
        st.markdown("""
        **Scenario:** A company trains an AI hiring model on historical employment data
        from a male-dominated tech industry (1980-2024).
        """)
        
        # Generate and visualize historical bias
        years, male_hired, female_hired = generate_historical_bias_data()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years, y=male_hired,
            mode='lines',
            name='Male Hires (%)',
            line=dict(color=COLORS['info'], width=3),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=years, y=female_hired,
            mode='lines',
            name='Female Hires (%)',
            line=dict(color=COLORS['warning'], width=3),
            fill='tozeroy'
        ))
        
        # Add annotation for historical bias period
        fig.add_vrect(
            x0=1980, x1=2000,
            fillcolor=COLORS['danger'], opacity=0.1,
            layer="below", line_width=0,
            annotation_text="High Historical Bias Period",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title="Historical Gender Distribution in Tech Hiring (1980-2024)",
            xaxis_title="Year",
            yaxis_title="Percentage of Hires (%)",
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("1980s Average", "83% Male", "High Bias")
        
        with col2:
            st.metric("2000s Average", "55% Male", "Moderate Bias")
        
        with col3:
            st.metric("2020s Average", "31% Male", "Improving")
        
        st.warning("""
        **Impact:** A model trained on all historical data (1980-2024) would:
        - Learn that male candidates are "preferred" based on historical patterns
        - Perpetuate gender discrimination in hiring decisions
        - Disadvantage qualified female candidates
        - Violate equal employment opportunity laws
        """)
        
        st.markdown("#### Real-World Example: Amazon's Hiring AI (2018)")
        
        with st.expander("📰 Case Study: Amazon's Biased Hiring Algorithm"):
            st.markdown("""
            In 2018, Amazon discovered their AI recruiting tool showed bias against women:
            
            **What Happened:**
            - Model trained on 10 years of historical resumes (mostly from men)
            - Algorithm learned to penalize resumes containing the word "women's"
            - Downgraded graduates of all-women's colleges
            - Favored male candidates based on historical patterns
            
            **Amazon's Response:**
            - Discontinued the tool
            - Acknowledged the bias couldn't be fully removed
            - Highlighted the importance of diverse training data
            
            **Lessons Learned:**
            - Historical data reflects historical biases
            - Technical fixes alone cannot solve social bias problems
            - Human oversight is essential
            - Diverse teams help identify bias early
            """)
        
        st.markdown("#### How to Detect Historical Bias")
        
        st.markdown("""
        1. **Analyze temporal trends in data**
        2. **Compare historical vs current distributions**
        3. **Review data for known societal biases**
        4. **Consult domain experts and affected communities**
        5. **Use fairness metrics across protected attributes**
        """)
        
        st.markdown("#### How to Mitigate Historical Bias")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Strategies:**
            - Use recent data only
            - Remove biased features
            - Reweight historical data
            - Synthetic data generation
            - Counterfactual data augmentation
            """)
        
        with col2:
            st.markdown("""
            **Model Strategies:**
            - Fairness constraints
            - Adversarial debiasing
            - Regular bias audits
            - Human-in-the-loop review
            - Diverse evaluation metrics
            """)
    
    # Tab 3: Measurement Bias
    with tab3:
        st.markdown("### Measurement Bias")
        
        st.markdown("""
        **Definition:** The data collection process itself introduces systematic errors.
        
        **What causes it:**
        - Poorly calibrated measurement devices
        - Leading or loaded questions
        - Observer bias in data labeling
        - Inconsistent measurement protocols
        - Cultural differences in interpretation
        """)
        
        st.markdown("#### Example: Customer Satisfaction Survey")
        
        st.markdown("""
        **Scenario:** Two different survey approaches to measure customer satisfaction:
        
        **Neutral Question:**
        > "How would you rate your experience with our product?" (1-10 scale)
        
        **Leading Question (Biased):**
        > "How much did you enjoy our amazing, industry-leading product?" (1-10 scale)
        """)
        
        # Generate and visualize measurement bias
        true_scores, biased_scores = generate_measurement_bias_data()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Neutral Question Results", "Leading Question Results (Biased)")
        )
        
        # Neutral scores
        fig.add_trace(
            go.Histogram(x=true_scores, name="Neutral", 
                        marker_color=COLORS['success'], nbinsx=20),
            row=1, col=1
        )
        
        # Biased scores
        fig.add_trace(
            go.Histogram(x=biased_scores, name="Leading", 
                        marker_color=COLORS['danger'], nbinsx=20),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Satisfaction Score (1-10)", row=1, col=1)
        fig.update_xaxes(title_text="Satisfaction Score (1-10)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Measurement Bias: Survey Question Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Neutral Question Mean", f"{true_scores.mean():.2f}/10")
            st.metric("Neutral Question Std Dev", f"{true_scores.std():.2f}")
        
        with col2:
            st.metric("Leading Question Mean", f"{biased_scores.mean():.2f}/10",
                     delta=f"+{biased_scores.mean() - true_scores.mean():.2f}",
                     delta_color="inverse")
            st.metric("Leading Question Std Dev", f"{biased_scores.std():.2f}",
                     delta=f"{biased_scores.std() - true_scores.std():.2f}",
                     delta_color="inverse")
        
        st.warning("""
        **Impact:** The leading question artificially inflates satisfaction scores by ~1.5 points:
        - Model trained on biased data overestimates customer satisfaction
        - Business decisions based on false positive feedback
        - Misses opportunities to improve actual customer experience
        - Wastes resources on products that need improvement
        """)
        
        st.markdown("#### Other Examples of Measurement Bias")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Medical Devices:**
            - Pulse oximeters less accurate on darker skin tones
            - Blood pressure cuffs calibrated for average arm sizes
            - Diagnostic tools tested primarily on one demographic
            
            **Facial Recognition:**
            - Training data predominantly lighter-skinned faces
            - Poor lighting conditions in data collection
            - Limited diversity in age, gender, ethnicity
            """)
        
        with col2:
            st.markdown("""
            **Data Labeling:**
            - Annotators' personal biases affect labels
            - Inconsistent labeling guidelines
            - Cultural differences in interpretation
            
            **Sensor Data:**
            - Poorly calibrated equipment
            - Environmental factors not accounted for
            - Systematic measurement errors
            """)
        
        st.markdown("#### How to Detect Measurement Bias")
        
        st.markdown("""
        1. **Review data collection protocols and instruments**
        2. **Compare measurements across different methods**
        3. **Analyze inter-rater reliability for labeled data**
        4. **Check for systematic patterns in errors**
        5. **Validate against ground truth when available**
        """)
        
        st.markdown("#### How to Mitigate Measurement Bias")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Prevention:**
            - Use neutral, unbiased questions
            - Calibrate measurement devices properly
            - Standardize data collection protocols
            - Train data collectors/annotators
            - Use diverse data collection teams
            """)
        
        with col2:
            st.markdown("""
            **Correction:**
            - Apply calibration corrections
            - Use multiple measurement methods
            - Cross-validate with independent sources
            - Statistical bias correction techniques
            - Regular audits of measurement processes
            """)
    
    # Tab 4: All Comparisons
    with tab4:
        st.markdown("### Comparison of All Bias Types")
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Bias Type': ['Sampling Bias', 'Historical Bias', 'Measurement Bias'],
            'Definition': [
                'Data doesn\'t represent true population',
                'Data reflects past societal biases',
                'Collection process introduces errors'
            ],
            'Primary Cause': [
                'Flawed sampling methodology',
                'Historical discrimination/inequity',
                'Measurement instruments/protocols'
            ],
            'Example': [
                'Surveying only college students',
                'Male-dominated hiring history',
                'Leading survey questions'
            ],
            'Detection Method': [
                'Compare to population statistics',
                'Analyze temporal trends',
                'Review collection protocols'
            ],
            'Mitigation Strategy': [
                'Stratified/random sampling',
                'Use recent data, fairness constraints',
                'Standardize protocols, calibrate tools'
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Visual Comparison of Impact")
        
        # Create a radar chart comparing severity across dimensions
        categories = ['Data Quality Impact', 'Fairness Impact', 'Model Performance', 
                     'Detection Difficulty', 'Mitigation Complexity']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[8, 9, 7, 6, 7],
            theta=categories,
            fill='toself',
            name='Sampling Bias',
            line_color=COLORS['danger']
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[7, 10, 8, 8, 9],
            theta=categories,
            fill='toself',
            name='Historical Bias',
            line_color=COLORS['warning']
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[9, 6, 6, 7, 6],
            theta=categories,
            fill='toself',
            name='Measurement Bias',
            line_color=COLORS['info']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title="Bias Type Severity Comparison (Scale: 1-10)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Insights:**
        - **Historical Bias** has the highest fairness impact and is hardest to mitigate
        - **Sampling Bias** significantly affects data quality and model performance
        - **Measurement Bias** is often easier to detect and correct than other types
        - All three types can coexist in the same dataset
        """)
    
    # AWS Tools section
    st.markdown(sub_header("AWS Tools for Detecting and Mitigating Bias", "🛠️", "aws"), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Amazon SageMaker Clarify")
        
        st.markdown("""
        SageMaker Clarify helps detect bias in your data and models:
        
        **Pre-training Bias Detection:**
        - Class Imbalance (CI)
        - Difference in Proportions of Labels (DPL)
        - Kullback-Leibler Divergence (KL)
        - Jensen-Shannon Divergence (JS)
        - Lp-norm (LP)
        - Total Variation Distance (TVD)
        - Kolmogorov-Smirnov (KS)
        - Conditional Demographic Disparity (CDD)
        
        **Post-training Bias Metrics:**
        - Difference in Positive Proportions (DPPL)
        - Disparate Impact (DI)
        - Difference in Conditional Acceptance (DCA)
        - Difference in Conditional Rejection (DCR)
        - Recall Difference (RD)
        - Accuracy Difference (AD)
        """)
        
        with st.expander("📝 Example: Using SageMaker Clarify"):
            st.code("""
import sagemaker
from sagemaker import clarify

# Initialize Clarify processor
clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type='ml.c5.xlarge',
    sagemaker_session=session
)

# Configure bias detection
bias_config = clarify.BiasConfig(
    label_values_or_threshold=[1],
    facet_name='gender',  # Protected attribute
    facet_values_or_threshold=['Female']
)

# Run pre-training bias analysis
clarify_processor.run_pre_training_bias(
    data_config=data_config,
    data_bias_config=bias_config,
    methods='all'
)

# View bias report
bias_report = clarify_processor.latest_bias_report()
print(bias_report)
            """, language="python")
    
    with col2:
        st.markdown("### Amazon SageMaker Data Wrangler")
        
        st.markdown("""
        Data Wrangler helps identify and fix data quality issues:
        
        **Data Quality Insights:**
        - Missing values detection
        - Outlier detection
        - Duplicate detection
        - Data type validation
        - Distribution analysis
        
        **Bias Detection:**
        - Quick Model analysis
        - Target leakage detection
        - Feature correlation analysis
        - Class imbalance visualization
        
        **Data Transformation:**
        - Resampling techniques
        - Feature engineering
        - Data balancing
        - Encoding categorical variables
        """)
        
        st.markdown("### Amazon Augmented AI (A2I)")
        
        st.markdown("""
        A2I enables human review to catch bias that automated tools miss:
        
        **Use Cases:**
        - Review low-confidence predictions
        - Validate fairness for edge cases
        - Audit model decisions on protected groups
        - Continuous monitoring with human oversight
        """)
    
    # Best Practices section
    st.markdown(sub_header("Best Practices for Bias Prevention", "✨", "minimal"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### Data Collection
        
        - Use representative sampling
        - Document data sources
        - Include diverse perspectives
        - Standardize protocols
        - Regular data audits
        - Transparent methodology
        """)
    
    with col2:
        st.markdown("""
        #### Model Development
        
        - Analyze training data for bias
        - Use fairness metrics
        - Test across demographics
        - Apply debiasing techniques
        - Cross-validate thoroughly
        - Document limitations
        """)
    
    with col3:
        st.markdown("""
        #### Production Monitoring
        
        - Continuous bias monitoring
        - Track performance by group
        - Human review processes
        - Regular model retraining
        - Incident response plan
        - Stakeholder feedback loops
        """)
    
    # Interactive Quiz
    st.markdown(sub_header("Knowledge Check", "🎯", "outline"), unsafe_allow_html=True)
    
    st.markdown("Test your understanding of dataset bias types:")
    
    with st.form("bias_quiz"):
        q1 = st.radio(
            "1. A survey only includes responses from people who visited a website. What type of bias is this?",
            ["Sampling Bias", "Historical Bias", "Measurement Bias"],
            index=None
        )
        
        q2 = st.radio(
            "2. A hiring model trained on 20 years of data shows preference for one gender. What type of bias?",
            ["Sampling Bias", "Historical Bias", "Measurement Bias"],
            index=None
        )
        
        q3 = st.radio(
            "3. A survey asks 'How much do you love our product?' instead of 'How do you rate our product?'. What bias?",
            ["Sampling Bias", "Historical Bias", "Measurement Bias"],
            index=None
        )
        
        q4 = st.multiselect(
            "4. Which AWS services can help detect bias? (Select all that apply)",
            ["Amazon SageMaker Clarify", "Amazon S3", "Amazon SageMaker Data Wrangler", 
             "Amazon EC2", "Amazon Augmented AI"],
        )
        
        submitted = st.form_submit_button("Check Answers")
        
        if submitted:
            score = 0
            total = 4
            
            if q1 == "Sampling Bias":
                st.success("✅ Question 1: Correct! Website visitors are a convenience sample, not representative.")
                score += 1
            elif q1:
                st.error("❌ Question 1: Incorrect. This is Sampling Bias - only certain people visit the website.")
            
            if q2 == "Historical Bias":
                st.success("✅ Question 2: Correct! The model learns from historical discrimination patterns.")
                score += 1
            elif q2:
                st.error("❌ Question 2: Incorrect. This is Historical Bias - past societal inequities in the data.")
            
            if q3 == "Measurement Bias":
                st.success("✅ Question 3: Correct! The leading question influences responses.")
                score += 1
            elif q3:
                st.error("❌ Question 3: Incorrect. This is Measurement Bias - the question itself is biased.")
            
            correct_services = {"Amazon SageMaker Clarify", "Amazon SageMaker Data Wrangler", "Amazon Augmented AI"}
            if set(q4) == correct_services:
                st.success("✅ Question 4: Correct! All three services help detect and mitigate bias.")
                score += 1
            elif q4:
                st.error(f"❌ Question 4: Incorrect. The correct services are: {', '.join(correct_services)}")
            
            st.markdown(f"### Your Score: {score}/{total} ({score/total*100:.0f}%)")
            
            if score == total:
                st.balloons()
                st.success("🎉 Perfect score! You have a strong understanding of dataset bias types!")
            elif score >= total * 0.75:
                st.success("👍 Great job! You understand most concepts well.")
            elif score >= total * 0.5:
                st.info("📚 Good effort! Review the material to strengthen your understanding.")
            else:
                st.warning("💪 Keep learning! Review each bias type and try again.")
    
    # Summary section
    st.markdown(sub_header("Summary", "📝"), unsafe_allow_html=True)
    
    st.markdown("""
    ### Key Takeaways
    
    1. **Dataset Bias is Pervasive**: All three types (sampling, historical, measurement) are common in real-world data
    
    2. **Bias Propagates**: Biased data → Biased models → Unfair outcomes → Societal harm
    
    3. **Detection is Critical**: Use statistical analysis, AWS tools, and domain expertise to identify bias
    
    4. **Mitigation Requires Multiple Approaches**: Combine data strategies, model techniques, and human oversight
    
    5. **AWS Provides Tools**: SageMaker Clarify, Data Wrangler, and A2I help detect and mitigate bias
    
    6. **Continuous Monitoring**: Bias detection isn't one-time - monitor throughout the ML lifecycle
    
    7. **Responsible AI**: Understanding and addressing dataset bias is fundamental to building fair, trustworthy AI systems
    """)
    
    st.info("""
    💡 **Remember**: Even with perfect algorithms, biased data will produce biased models. 
    Addressing dataset bias is not just a technical challenge - it's an ethical imperative 
    for responsible AI development.
    """)
    
    # Additional Resources
    with st.expander("📚 Additional Resources"):
        st.markdown("""
        **AWS Documentation:**
        - [Amazon SageMaker Clarify Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-fairness-and-explainability.html)
        - [Detect Bias in ML Models](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-detect-data-bias.html)
        - [Amazon SageMaker Data Wrangler](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler.html)
        
        **Research Papers:**
        - "Fairness and Machine Learning" by Barocas, Hardt, and Narayanan
        - "A Survey on Bias and Fairness in Machine Learning" (ACM Computing Surveys)
        - "Datasheets for Datasets" by Gebru et al.
        
        **Industry Guidelines:**
        - NIST AI Risk Management Framework
        - EU AI Act Requirements
        - IEEE Ethically Aligned Design
        """)

if __name__ == "__main__":
    if 'localhost' in st.context.headers.get("host", ""):
        main()
    else:
        if authenticate.login():
            main()

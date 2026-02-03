import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import base64
from PIL import Image
import io
from utils.styles import load_css
import utils.authenticate as authenticate

# Initialize session state
def initialize_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'timestamp' not in st.session_state:
        st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
        
    if 'sample_data' not in st.session_state:
        # Create sample manufacturing data
        np.random.seed(42)
        n_samples = 1000
        
        # Features
        temperature = np.random.normal(100, 15, n_samples)
        pressure = np.random.normal(50, 10, n_samples)
        vibration = np.random.normal(5, 2, n_samples)
        humidity = np.random.uniform(30, 70, n_samples)
        speed = np.random.normal(120, 20, n_samples)
        
        # Generate defect labels based on feature combinations
        defect = []
        for i in range(n_samples):
            if (temperature[i] > 115 and pressure[i] > 60) or \
               (vibration[i] > 7 and speed[i] < 100) or \
               (humidity[i] < 35 and temperature[i] > 110):
                defect.append(1)  # Defective
            else:
                defect.append(0)  # Non-defective
        
        # Create DataFrame
        st.session_state.sample_data = pd.DataFrame({
            'temperature': temperature,
            'pressure': pressure,
            'vibration': vibration,
            'humidity': humidity,
            'speed': speed,
            'defect': defect
        })
        
        # Train a model
        X = st.session_state.sample_data[['temperature', 'pressure', 'vibration', 'humidity', 'speed']]
        y = st.session_state.sample_data['defect']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = model.predict(X_test)
        st.session_state.feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

def reset_session():
    for key in st.session_state.keys():
        if key not in ["authenticated", "user_cognito_groups", "auth_code","user_info"]:
            del st.session_state[key]
    initialize_session_state()
    st.rerun()

def predict_defect(temperature, pressure, vibration, humidity, speed):
    input_data = pd.DataFrame([[temperature, pressure, vibration, humidity, speed]], 
                             columns=['temperature', 'pressure', 'vibration', 'humidity', 'speed'])
    prediction = st.session_state.model.predict(input_data)[0]
    probability = st.session_state.model.predict_proba(input_data)[0][1]
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    st.session_state.prediction_history.append({
        'timestamp': timestamp,
        'temperature': temperature,
        'pressure': pressure,
        'vibration': vibration,
        'humidity': humidity,
        'speed': speed,
        'prediction': 'Defective' if prediction == 1 else 'Non-Defective',
        'probability': probability
    })
    
    return prediction, probability

def create_gauge_chart(value, title, min_val=0, max_val=1):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#1F77B4" if value < 0.5 else "#FF7F0E"},
            'steps': [
                {'range': [0, 0.5], 'color': '#EBF5FB'},
                {'range': [0.5, 1], 'color': '#FDEBD0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def prediction_tab():
    st.header("🔍 Manufacturing Quality Control Prediction")
    st.markdown("""
    This tool helps identify potential defects in manufacturing by analyzing sensor data. 
    Fill in the values from your production line sensors to predict if a product might be defective.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (°C)", 50.0, 150.0, 100.0, 0.5)
        pressure = st.slider("Pressure (PSI)", 20.0, 80.0, 50.0, 0.5)
        vibration = st.slider("Vibration (mm/s)", 0.0, 10.0, 5.0, 0.1)
    
    with col2:
        humidity = st.slider("Humidity (%)", 20.0, 80.0, 50.0, 1.0)
        speed = st.slider("Production Speed (units/hour)", 60, 180, 120, 1)
    
    if st.button("Predict Quality", use_container_width=True):
        prediction, probability = predict_defect(temperature, pressure, vibration, humidity, speed)
        
        st.markdown("### Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **Potential Defect Detected**")
                st.markdown(f"Confidence: **{probability:.2%}**")
            else:
                st.success("✅ **Quality Check Passed**")
                st.markdown(f"Confidence: **{(1-probability):.2%}**")
            
        with col2:
            gauge_fig = create_gauge_chart(probability, "Defect Probability")
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        st.markdown("### Sensor Data Analysis")
        
        # Show factors influencing prediction
        st.write("Factors influencing this prediction:")
        
        factors = []
        if temperature > 115:
            factors.append(f"❗ Temperature is high ({temperature:.1f}°C)")
        if pressure > 60:
            factors.append(f"❗ Pressure is high ({pressure:.1f} PSI)")
        if vibration > 7:
            factors.append(f"❗ Vibration is high ({vibration:.1f} mm/s)")
        if humidity < 35:
            factors.append(f"❗ Humidity is low ({humidity:.1f}%)")
        if speed < 100:
            factors.append(f"❗ Production speed is low ({speed} units/hour)")
            
        if factors:
            for factor in factors:
                st.warning(factor)
        else:
            st.info("✅ All parameters are within optimal ranges.")
    
    if st.session_state.prediction_history:
        st.markdown("### Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)

def model_performance_tab():
    st.header("📊 Model Performance")
    st.markdown("""
    Explore the performance metrics of our manufacturing quality control machine learning model.
    Understanding these metrics helps to validate the reliability of defect predictions.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📈 Key Metrics", "🧩 Confusion Matrix", "⚖️ Feature Importance"])
    
    with tab1:
        # Calculate metrics
        accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
        precision = precision_score(st.session_state.y_test, st.session_state.y_pred)
        recall = recall_score(st.session_state.y_test, st.session_state.y_pred)
        f1 = f1_score(st.session_state.y_test, st.session_state.y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
            st.metric("Precision", f"{precision:.2%}")
        
        with col2:
            st.metric("Recall", f"{recall:.2%}")
            st.metric("F1 Score", f"{f1:.2%}")
            
        st.markdown("""
        **Understanding these metrics:**
        
        - **Accuracy**: Percentage of correctly predicted defects and non-defects
        - **Precision**: How often a predicted defect is actually a defect
        - **Recall**: How often actual defects are correctly identified
        - **F1 Score**: Harmonic mean of precision and recall
        """)
        
        # ROC Curve
        st.subheader("ROC Curve")
        from sklearn.metrics import roc_curve, auc
        y_scores = st.session_state.model.predict_proba(st.session_state.X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(st.session_state.y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC = {roc_auc:.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig)
        
    with tab2:
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        
        # Visualize confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Non-Defective', 'Defective'])
        ax.set_yticklabels(['Non-Defective', 'Defective'])
        st.pyplot(fig)
        
        st.markdown("""
        **Reading the Confusion Matrix:**
        
        - **Top-Left (True Negatives)**: Correctly identified non-defective products
        - **Bottom-Right (True Positives)**: Correctly identified defective products
        - **Top-Right (False Positives)**: Non-defective products incorrectly classified as defective
        - **Bottom-Left (False Negatives)**: Defective products incorrectly classified as non-defective
        
        In manufacturing quality control, false negatives are particularly costly as they represent defective products that passed quality checks.
        """)
        
    with tab3:
        # Feature Importance
        st.subheader("Feature Importance")
        
        fig = px.bar(
            st.session_state.feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance in Predicting Defects'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig)
        
        st.markdown("""
        Feature importance shows which sensor measurements are most critical for detecting defects. 
        
        Prioritize monitoring and maintenance of sensors that measure the most important features to maintain prediction quality.
        """)

def data_exploration_tab():
    st.header("🔬 Data Exploration")
    st.markdown("""
    Explore the relationships between different manufacturing parameters and how they relate to product quality.
    This visualization helps identify patterns and potential thresholds for quality control.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Distribution", "🔄 Correlations", "🎯 Defect Patterns"])
    
    with tab1:
        st.subheader("Parameter Distributions by Quality Outcome")
        
        feature = st.selectbox("Select Parameter", 
                               ['temperature', 'pressure', 'vibration', 'humidity', 'speed'])
        
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(data=st.session_state.sample_data, x=feature, hue='defect', 
                     element="step", stat="density", common_norm=False)
        plt.title(f"Distribution of {feature.capitalize()} by Quality Outcome")
        plt.xlabel(f"{feature.capitalize()} Value")
        plt.ylabel("Density")
        plt.legend(labels=['Non-Defective', 'Defective'])
        st.pyplot(fig)
        
        # Summary statistics
        st.write("Summary Statistics:")
        summary = st.session_state.sample_data.groupby('defect')[feature].describe().reset_index()
        summary = summary.rename(columns={'defect': 'Quality', 0: 'Non-Defective', 1: 'Defective'})
        summary['Quality'] = summary['Quality'].map({0: 'Non-Defective', 1: 'Defective'})
        st.dataframe(summary)
        
    with tab2:
        st.subheader("Correlation Matrix")
        
        corr = st.session_state.sample_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                    linewidths=0.5, ax=ax)
        plt.title("Correlation Matrix of Manufacturing Parameters")
        st.pyplot(fig)
        
        st.markdown("""
        **Understanding the correlation matrix:**
        
        - Values closer to 1 indicate strong positive correlation
        - Values closer to -1 indicate strong negative correlation
        - Values closer to 0 indicate little or no correlation
        
        Strong correlations between parameters might indicate:
        1. Underlying process relationships
        2. Potential sensor redundancy
        3. Compound factors affecting quality
        """)
        
    with tab3:
        st.subheader("Defect Patterns")
        
        x_axis = st.selectbox("X-Axis Parameter", 
                             ['temperature', 'pressure', 'vibration', 'humidity', 'speed'], 
                             key="x_axis")
        
        y_axis = st.selectbox("Y-Axis Parameter", 
                             ['pressure', 'temperature', 'vibration', 'humidity', 'speed'], 
                             index=1, 
                             key="y_axis")
        
        fig = px.scatter(
            st.session_state.sample_data, 
            x=x_axis, 
            y=y_axis, 
            color='defect',
            color_discrete_sequence=["blue", "red"],
            labels={"defect": "Quality"},
            category_orders={"defect": [0, 1]},
            title=f"Defect Patterns: {x_axis.capitalize()} vs {y_axis.capitalize()}",
            hover_data=['temperature', 'pressure', 'vibration', 'humidity', 'speed']
        )
        
        fig.update_layout(
            legend=dict(
                title="Quality",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                itemsizing="constant",
                itemclick="toggleothers"
            ),
            legend_title_text="Quality",
            coloraxis_colorbar=dict(
                title="Quality",
                tickvals=[0, 1],
                ticktext=["Non-Defective", "Defective"]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add best-fit decision boundary
        st.checkbox("Show Decision Boundary", value=False, key="show_boundary")
        
        if st.session_state.show_boundary:
            # Create a mesh grid for decision boundary
            x_min, x_max = st.session_state.sample_data[x_axis].min() - 1, st.session_state.sample_data[x_axis].max() + 1
            y_min, y_max = st.session_state.sample_data[y_axis].min() - 1, st.session_state.sample_data[y_axis].max() + 1
            h = 0.1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Create temp dataframe with default values for all features
            mesh_defaults = {
                'temperature': st.session_state.sample_data['temperature'].median(),
                'pressure': st.session_state.sample_data['pressure'].median(),
                'vibration': st.session_state.sample_data['vibration'].median(),
                'humidity': st.session_state.sample_data['humidity'].median(),
                'speed': st.session_state.sample_data['speed'].median()
            }
            
            # Update with our selected values
            grid_df = pd.DataFrame({
                x_axis: xx.ravel(),
                y_axis: yy.ravel()
            })
            
            for feature in ['temperature', 'pressure', 'vibration', 'humidity', 'speed']:
                if feature != x_axis and feature != y_axis:
                    grid_df[feature] = mesh_defaults[feature]
            
            # Make prediction on the grid
            Z = st.session_state.model.predict_proba(grid_df[['temperature', 'pressure', 'vibration', 'humidity', 'speed']])[:, 1]
            Z = Z.reshape(xx.shape)
            
            # Create the decision boundary plot
            fig_boundary = go.Figure()
            
            # Add contour
            fig_boundary.add_trace(
                go.Contour(
                    z=Z,
                    x=np.arange(x_min, x_max, h),
                    y=np.arange(y_min, y_max, h),
                    colorscale='RdBu_r',
                    opacity=0.6,
                    showscale=True,
                    contours=dict(
                        start=0,
                        end=1,
                        size=0.05
                    )
                )
            )
            
            # Add data points
            fig_boundary.add_trace(
                go.Scatter(
                    x=st.session_state.sample_data[x_axis][st.session_state.sample_data['defect'] == 0],
                    y=st.session_state.sample_data[y_axis][st.session_state.sample_data['defect'] == 0],
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name='Non-Defective'
                )
            )
            
            fig_boundary.add_trace(
                go.Scatter(
                    x=st.session_state.sample_data[x_axis][st.session_state.sample_data['defect'] == 1],
                    y=st.session_state.sample_data[y_axis][st.session_state.sample_data['defect'] == 1],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='Defective'
                )
            )
            
            fig_boundary.update_layout(
                title=f"Decision Boundary: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                xaxis_title=x_axis.capitalize(),
                yaxis_title=y_axis.capitalize(),
                height=600
            )
            
            st.plotly_chart(fig_boundary, use_container_width=True)
            
            st.markdown("""
            **How to read the decision boundary:**
            
            The color gradient represents the model's predicted probability of a defect:
            - **Blue areas**: Parameters likely resulting in non-defective products
            - **Red areas**: Parameters likely resulting in defective products
            - **The boundary between colors**: Critical thresholds where the model changes its prediction
            
            This visualization helps identify safe operating zones for your manufacturing process.
            """)

def concepts_tab():
    st.header("🧠 ML in Manufacturing Concepts")
    st.markdown("""
    Learn about key concepts in applying machine learning to manufacturing quality control.
    """)
    
    concept = st.selectbox(
        "Select a concept to explore:", 
        [
            "Traditional ML vs. Generative AI in Manufacturing",
            "ML Development Lifecycle for Manufacturing",
            "Common Use Cases in Manufacturing",
            "Data Collection Strategies",
            "Responsible AI in Manufacturing"
        ]
    )
    
    if concept == "Traditional ML vs. Generative AI in Manufacturing":
        st.subheader("Traditional ML vs. Generative AI in Manufacturing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Traditional ML Approaches")
            st.markdown("""
            **Strengths for Manufacturing:**
            - **Interpretability & Transparency**: Clear decision rules critical for regulatory compliance and safety
            - **Consistent Results**: Predictable outputs essential for quality-critical components
            - **Data Efficiency**: Works well with smaller, structured sensor datasets
            - **Focused Performance**: Specialized for specific defect detection tasks
            
            **Common Applications:**
            - Defect classification systems
            - Predictive maintenance
            - Process optimization
            - Quality control thresholds
            """)
            
        with col2:
            st.markdown("#### Generative AI Applications")
            st.markdown("""
            **Emerging Manufacturing Uses:**
            - **Design Generation**: Creating new product designs based on constraints
            - **Recipe Formulation**: Generating new material formulations
            - **Simulation**: Creating synthetic data for rare defect types
            - **Documentation**: Automatic generation of reports and manuals
            
            **Limitations:**
            - Less interpretable decisions
            - Higher data and computing requirements
            - May produce inconsistent results
            """)
            
        st.info("""
        **Key Takeaway**: In manufacturing quality control, traditional ML models are typically preferred for 
        their interpretability, consistency, and ability to work with structured sensor data. Generative AI 
        finds applications in complementary areas like design innovation and documentation.
        """)
            
    elif concept == "ML Development Lifecycle for Manufacturing":
        st.subheader("ML Development Lifecycle for Manufacturing")
        
        lifecycle_steps = [
            {
                "title": "1. Business Problem Framing",
                "desc": "Identify specific quality issues to address (e.g., reducing defect rates, improving consistency)",
                "icon": "🎯"
            },
            {
                "title": "2. Data Collection",
                "desc": "Gather sensor data, QA records, and production parameters",
                "icon": "📥"
            },
            {
                "title": "3. Data Preparation",
                "desc": "Clean sensor data, handle missing values, normalize readings",
                "icon": "🧹"
            },
            {
                "title": "4. Feature Engineering",
                "desc": "Create relevant features like vibration patterns, temperature deltas, or production timing",
                "icon": "⚙️"
            },
            {
                "title": "5. Model Training",
                "desc": "Train classification models to identify defect patterns",
                "icon": "🏋️"
            },
            {
                "title": "6. Model Evaluation",
                "desc": "Assess performance with emphasis on minimizing false negatives (missed defects)",
                "icon": "📊"
            },
            {
                "title": "7. Deployment",
                "desc": "Integrate with production systems for real-time quality monitoring",
                "icon": "🚀"
            },
            {
                "title": "8. Monitoring & Maintenance",
                "desc": "Track model performance, retrain with new defect patterns, adjust for process drift",
                "icon": "🔍"
            }
        ]
        
        for step in lifecycle_steps:
            with st.container():
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.markdown(f"<h1 style='text-align: center; color: #FF9900;'>{step['icon']}</h1>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{step['title']}**")
                    st.markdown(step['desc'])
                st.divider()
        
        st.markdown("""
        **Manufacturing-Specific Considerations:**
        
        * **Data Imbalance**: Defective parts are typically rare compared to good parts
        * **Concept Drift**: Manufacturing processes change over time due to tool wear or material variations
        * **Real-time Requirements**: Models often need to make predictions within production cycle times
        * **Interpretability**: Quality engineers need to understand why a part was flagged as defective
        """)
        
        # Add a simple diagram
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*zWBYt9DQQEf8K5S1t0yDEA.png", 
                caption="ML Development Lifecycle Diagram")
        
    elif concept == "Common Use Cases in Manufacturing":
        st.subheader("Common ML Use Cases in Manufacturing")
        
        use_cases = [
            {
                "title": "Visual Defect Detection",
                "desc": "Computer vision models that detect surface defects, misalignments, or missing components",
                "ml_type": "Traditional ML (Computer Vision)",
                "typical_accuracy": "95-99%"
            },
            {
                "title": "Predictive Maintenance",
                "desc": "Forecasting when equipment will fail based on sensor data and maintenance history",
                "ml_type": "Traditional ML (Time Series Analysis)",
                "typical_accuracy": "80-90%"
            },
            {
                "title": "Process Parameter Optimization",
                "desc": "Recommending optimal machine settings to maximize quality and minimize waste",
                "ml_type": "Traditional ML (Regression/Optimization)",
                "typical_accuracy": "Varies by process"
            },
            {
                "title": "Anomaly Detection",
                "desc": "Identifying unusual patterns in manufacturing data that may indicate emerging issues",
                "ml_type": "Traditional ML (Unsupervised Learning)",
                "typical_accuracy": "75-85%"
            },
            {
                "title": "Supply Chain Forecasting",
                "desc": "Predicting demand and supply chain disruptions to optimize inventory",
                "ml_type": "Traditional ML (Time Series Forecasting)",
                "typical_accuracy": "80-95%"
            }
        ]
        
        for idx, case in enumerate(use_cases):
            with st.expander(f"{case['title']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Description**: {case['desc']}")
                    st.markdown(f"**ML Type**: {case['ml_type']}")
                    st.markdown(f"**Typical Accuracy Range**: {case['typical_accuracy']}")
                
                with col2:
                    # Use placeholder images for each use case
                    placeholder_images = [
                        "https://www.einfochips.com/blog/wp-content/uploads/2022/03/OpenCv_defect_detection_use_case.jpg",
                        "https://www.seebo.com/wp-content/uploads/2021/01/predictive-maintenance-ml.jpg",
                        "https://www.altair.com/images/solutions/manufacturing-process-optimization.jpg",
                        "https://tshuxing.files.wordpress.com/2019/02/vibracao.png",
                        "https://www.mdpi.com/logistics/logistics-05-00034/article_deploy/html/images/logistics-05-00034-g001.png"
                    ]
                    st.image(placeholder_images[idx], use_column_width=True)
    
    elif concept == "Data Collection Strategies":
        st.subheader("Data Collection Strategies for Manufacturing ML")
        
        st.markdown("""
        Effective machine learning for quality control depends on robust data collection strategies. 
        Here are key approaches used in manufacturing environments:
        """)
        
        tab1, tab2, tab3 = st.tabs(["📡 Sensor Data", "📊 Historical Records", "🔄 Data Integration"])
        
        with tab1:
            st.markdown("""
            ### Sensor Data Collection
            
            **Types of Sensors Commonly Used:**
            - Temperature sensors
            - Pressure sensors
            - Vibration analysis equipment
            - Vision systems (cameras)
            - Acoustic sensors
            - Humidity/moisture sensors
            
            **Best Practices:**
            - Install sensors at critical points in the production line
            - Ensure appropriate sampling frequency (high enough to catch defects, manageable for storage)
            - Implement edge processing for data reduction when appropriate
            - Maintain sensor calibration schedules
            
            **Challenges:**
            - Sensor drift and calibration
            - Noisy data in factory environments
            - Bandwidth and storage limitations
            - Sensor failure detection
            """)
            
        with tab2:
            st.markdown("""
            ### Historical Records Integration
            
            **Key Historical Data Sources:**
            - Quality inspection results
            - Maintenance records
            - Production batch information
            - Customer returns and complaints
            - Material supplier information
            
            **Integration Approaches:**
            - Create unified data schemas across systems
            - Establish time correlation between records
            - Implement data governance practices
            
            **Benefits:**
            - Provides context for sensor readings
            - Helps identify root causes beyond immediate measurements
            - Enables long-term trend analysis
            """)
            
        with tab3:
            st.markdown("""
            ### Data Integration Architecture
            
            **Modern Manufacturing Data Stack:**
            
            1. **Edge Layer**
               - Real-time data collection from sensors
               - Initial data filtering and aggregation
               - Time synchronization
            
            2. **Platform Layer**
               - Data lake for storage of raw sensor data
               - Data warehouse for structured manufacturing data
               - Data transformation pipelines
            
            3. **Analytics Layer**
               - Machine learning models for defect prediction
               - Dashboards for quality monitoring
               - Automated alerting systems
            
            **Implementation Tools:**
            - AWS IoT Core for sensor data ingestion
            - AWS S3 for data lake storage
            - Amazon SageMaker for model development
            - AWS QuickSight for visualization
            """)
            
            st.image("https://d1.awsstatic.com/IoT/diagrams/IoT_Core_How-It-Works_Diagram.2f74f0cad1c542c7f0ff60a4daa2e99cbe639e99.png", 
                    caption="AWS IoT Data Collection Architecture")
        
    elif concept == "Responsible AI in Manufacturing":
        st.subheader("Responsible AI in Manufacturing")
        
        st.markdown("""
        Implementing AI responsibly in manufacturing environments is crucial for safety, 
        compliance, and building trust with stakeholders.
        """)
        
        principles = [
            {
                "title": "Safety First",
                "desc": "AI systems should enhance—never compromise—the safety of manufacturing environments",
                "example": "Using explainable AI models for critical safety decisions rather than black-box approaches",
                "icon": "🛡️"
            },
            {
                "title": "Data Privacy",
                "desc": "Protect proprietary manufacturing processes and employee data",
                "example": "Implementing data anonymization for operator-linked quality records",
                "icon": "🔒"
            },
            {
                "title": "Fairness",
                "desc": "Ensure AI systems don't introduce or amplify biases in quality assessment",
                "example": "Regular auditing to verify that defect detection is consistent across production shifts",
                "icon": "⚖️"
            },
            {
                "title": "Transparency",
                "desc": "Make AI decision processes understandable to operators and engineers",
                "example": "Visual interfaces showing which sensor readings triggered a quality alert",
                "icon": "👁️"
            },
            {
                "title": "Human Oversight",
                "desc": "Maintain human supervision and override capabilities",
                "example": "Engineer review process for uncommon defect types before scrapping expensive components",
                "icon": "👤"
            }
        ]
        
        for principle in principles:
            with st.container():
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.markdown(f"<h1 style='text-align: center; color: #232F3E;'>{principle['icon']}</h1>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{principle['title']}**")
                    st.markdown(principle['desc'])
                    st.markdown(f"*Example:* {principle['example']}")
                st.divider()
                
        st.warning("""
        **Regulatory Considerations**
        
        Manufacturing AI implementations may need to comply with:
        - Industry-specific quality standards (ISO 9001, IATF 16949)
        - Safety regulations (OSHA in the US)
        - Emerging AI regulations in different jurisdictions
        """)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Use centralized CSS
    load_css()
    
    # Title
    st.title("🏭 Manufacturing Quality Control ML")
    st.markdown("""
    Explore how machine learning can improve manufacturing quality control by predicting potential defects
    based on production line sensor data.
    """)
    
    # Sidebar
    with st.sidebar:
        st.subheader("Session Management")
        
        st.info(f"**Session ID**: {st.session_state.session_id[:8]}...")
        st.markdown(f"**Started**: {st.session_state.timestamp}")
        if st.button('Reset Session'):
            reset_session()
        
        
        st.divider()
        
        with st.expander("About this App", expanded=False):
            st.markdown("""
            This application demonstrates how machine learning can be applied to manufacturing 
            quality control. Topics covered:
            
            * Defect prediction based on sensor data
            * Model performance evaluation
            * Data exploration for quality insights
            * ML concepts for manufacturing
            
            Built with Streamlit and scikit-learn.
            """)
        

    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Prediction", 
        "📊 Model Performance", 
        "🔬 Data Exploration",
        "🧠 ML Concepts"
    ])
    
    with tab1:
        prediction_tab()
        
    with tab2:
        model_performance_tab()
        
    with tab3:
        data_exploration_tab()
        
    with tab4:
        concepts_tab()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: gray; font-size: 12px;">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Page setup
    st.set_page_config(
        page_title="Manufacturing Quality Control ML",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )    
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()
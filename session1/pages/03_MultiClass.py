
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import uuid
import joblib
from datetime import datetime
import base64
from io import BytesIO
from utils.common import render_sidebar
from utils.styles import load_css
import utils.authenticate as authenticate


# Set page config
st.set_page_config(
    page_title="Supervised Learning: MultiClass",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "initialized" not in st.session_state:
        try:
            with st.spinner("Loading model and data..."):
                # Load and prepare data
                iris = load_iris()
                X = pd.DataFrame(iris.data, columns=iris.feature_names)
                y = pd.Series(iris.target)
                
                # Store data in session state
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.feature_names = iris.feature_names
                st.session_state.target_names = iris.target_names
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Train a model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                st.session_state.model = model
                
                # Make predictions for evaluation
                y_pred = model.predict(X_test)
                st.session_state.y_pred = y_pred
                
                # Calculate metrics
                st.session_state.accuracy = accuracy_score(y_test, y_pred)
                st.session_state.conf_matrix = confusion_matrix(y_test, y_pred)
                st.session_state.class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # Generate feature importance
                st.session_state.feature_importance = pd.DataFrame({
                    'Feature': iris.feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Mark initialization as complete
                st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            st.stop()


# Function to create an image from a matplotlib figure
def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return img_str

# Custom styling functions
def custom_header(text, level=1):
    if level == 1:
        return f"<h1 style='color:#FF9900; font-family:Amazon Ember, Arial, sans-serif;'>{text}</h1>"
    elif level == 2:
        return f"<h2 style='color:#232F3E; font-family:Amazon Ember, Arial, sans-serif;'>{text}</h2>"
    elif level == 3:
        return f"<h3 style='color:#232F3E; font-family:Amazon Ember, Arial, sans-serif;'>{text}</h3>"
    else:
        return f"<h4 style='color:#232F3E; font-family:Amazon Ember, Arial, sans-serif;'>{text}</h4>"

# Topic 1: Introduction to ML Prediction Tab
def show_prediction_tab():
    st.markdown("<h2>Make Your Own Iris Flower Prediction</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This interactive tool demonstrates supervised learning with multiclass classification. 
    Enter the measurements below to classify an iris flower into one of three species:
    **Setosa**, **Versicolor**, or **Virginica**.
    
    The model uses a Random Forest classifier trained on the famous Iris dataset.
    """)
    
    # Create a form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider(
                "Sepal Length (cm)",
                min_value=4.0,
                max_value=8.0,
                value=5.4,
                step=0.1
            )
            
            sepal_width = st.slider(
                "Sepal Width (cm)",
                min_value=2.0,
                max_value=4.5,
                value=3.4,
                step=0.1
            )
        
        with col2:
            petal_length = st.slider(
                "Petal Length (cm)",
                min_value=1.0,
                max_value=7.0,
                value=4.5,
                step=0.1
            )
            
            petal_width = st.slider(
                "Petal Width (cm)",
                min_value=0.1,
                max_value=2.5,
                value=1.5,
                step=0.1
            )
        
        submitted = st.form_submit_button("Predict Species")
    
    if submitted:
        # Make prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = st.session_state.model.predict(input_data)
        probabilities = st.session_state.model.predict_proba(input_data)
        
        # Display prediction
        species = st.session_state.target_names[prediction[0]]
        
        st.success(f"Predicted Species: **{species.capitalize()}**")
        
        # Show probability breakdown
        st.markdown("### Prediction Probability Breakdown")
        
        prob_df = pd.DataFrame({
            'Species': st.session_state.target_names,
            'Probability': probabilities[0] * 100
        })
        
        # Create a bar chart for probabilities
        fig = px.bar(
            prob_df, 
            x='Species', 
            y='Probability', 
            text_auto='.1f',
            text='Probability',
            color='Species',
            color_discrete_sequence=['#FF9900', '#146EB4', '#232F3E'],
            labels={'Probability': 'Probability (%)'}
        )
        fig.update_traces(texttemplate='%{text:.2}%', textposition='outside')
        fig.update_layout(
            title='Species Probability Distribution',
            xaxis_title='Iris Species',
            yaxis_title='Probability (%)',
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Provide context for the prediction
        st.markdown("#### How Your Input Values Compare to Species Averages")
        
        # Group data by species and calculate mean
        avg_by_species = pd.DataFrame(st.session_state.X).join(
            pd.Series(st.session_state.y).rename('species')
        )
        avg_by_species['species'] = avg_by_species['species'].map({
            0: st.session_state.target_names[0],
            1: st.session_state.target_names[1],
            2: st.session_state.target_names[2]
        })
        
        avg_by_species = avg_by_species.groupby('species').mean().reset_index()
        
        # Create comparison dataset
        user_data = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Your Input': [sepal_length, sepal_width, petal_length, petal_width],
            'Setosa Avg': avg_by_species.iloc[0, 1:].values,
            'Versicolor Avg': avg_by_species.iloc[1, 1:].values,
            'Virginica Avg': avg_by_species.iloc[2, 1:].values
        })
        
        st.table(user_data.set_index('Feature').style.format("{:.2f}"))

# Topic 2: Model Performance Tab
def show_model_performance():
    st.markdown("<h2>Model Performance Analysis</h2>", unsafe_allow_html=True)
        
    st.markdown("""
    This section demonstrates how to evaluate a supervised learning model. For our Iris flower classifier,
    we'll explore various metrics such as accuracy, confusion matrix, and classification report. These are
    standard evaluation techniques for multiclass classification problems.
    """)
    
    # Accuracy
    st.markdown("### Model Accuracy")
    st.metric(
        label="Test Set Accuracy", 
        value=f"{st.session_state.accuracy:.2%}",
        help="Accuracy is the proportion of correct predictions among the total number of predictions."
    )
    
    # Tabs for different evaluation metrics
    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["📊 Confusion Matrix", "📈 Classification Report", "🔍 Feature Importance"])
    
    with eval_tab1:
        st.markdown("##### Confusion Matrix")
        st.markdown("""
        A confusion matrix shows the counts of correct and incorrect predictions for each class,
        helping us understand where our model is making mistakes.
        """)
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            st.session_state.conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=st.session_state.target_names,
            yticklabels=st.session_state.target_names,
            ax=ax
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        
        st.markdown("""
        **How to read the confusion matrix:**
        - Diagonal elements (top-left to bottom-right) represent correct predictions
        - Off-diagonal elements represent incorrect predictions
        - Each row represents the actual class, while each column represents the predicted class
        """)
    
    with eval_tab2:
        st.markdown("##### Classification Report")
        st.markdown("""
        The classification report provides precision, recall, and F1-score metrics for each class,
        giving a more detailed view of model performance.
        """)
        
        # Convert classification report to DataFrame for better display
        report_df = pd.DataFrame(st.session_state.class_report).transpose()
        
        # Only keep rows for actual classes (not 'accuracy', 'macro avg', etc.)
        class_indices = [0, 1, 2]
        class_rows = [st.session_state.target_names[i] for i in class_indices]
        
        # Filter and rename rows
        report_df = report_df.iloc[class_indices].copy()
        report_df.index = class_rows
        report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
        
        # Round values for better display
        report_df = report_df.round(3)
        
        # Display the report
        st.table(report_df)
        
        # Add explanations for metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Precision:** Ability of the model not to label as positive a sample that is negative")
            st.markdown("**Recall:** Ability of the model to find all the positive samples")
        with col2:
            st.markdown("**F1-score:** Harmonic mean of precision and recall")
            st.markdown("**Support:** Number of actual occurrences of the class in the test set")
    
    with eval_tab3:
        st.markdown("##### Feature Importance")
        st.markdown("""
        Feature importance shows which features (measurements) have the most influence on the model's predictions.
        """)
        
        # Plot feature importance
        fig = px.bar(
            st.session_state.feature_importance, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues',
        )
        fig.update_layout(
            title='Feature Importance in Random Forest Model',
            xaxis_title='Importance Score',
            yaxis_title='Feature'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Key insights:**
        - Features with higher importance scores have more influence on predictions
        - In this case, petal dimensions tend to be more important than sepal dimensions for classifying iris species
        - This helps us understand which measurements are most critical for identification
        """)

# Topic 3: Data Exploration Tab
def show_data_exploration():
    st.markdown("<h2>Exploring Training Data</h2>", unsafe_allow_html=True)
    st.markdown("""
    Understanding your training data is a crucial first step in any supervised learning project. 
    This section shows how we can explore and visualize the Iris dataset to gain insights before training.
    """)
    
    # Dataset overview
    st.markdown("### Dataset Overview")
    iris_df = pd.DataFrame(
        st.session_state.X, 
        columns=st.session_state.feature_names
    )
    iris_df['species'] = pd.Categorical.from_codes(
        st.session_state.y, 
        categories=st.session_state.target_names
    )
    
    with st.expander("View Dataset Sample"):
        st.dataframe(iris_df.head(10))
    
    # Dataset statistics
    st.markdown("### Dataset Statistics")
    stats_tab1, stats_tab2 = st.tabs(["📊 Summary Statistics", "🔢 Class Distribution"])
    
    with stats_tab1:
        st.markdown("##### Summary Statistics by Species")
        # Group by species and calculate stats
        stats_by_species = iris_df.groupby('species').describe().transpose()
        st.dataframe(stats_by_species)
    
    with stats_tab2:
        st.markdown("##### Class Distribution")
        species_counts = iris_df['species'].value_counts().reset_index()
        species_counts.columns = ['Species', 'Count']
        
        fig = px.pie(
            species_counts, 
            values='Count', 
            names='Species', 
            title='Distribution of Iris Species',
            color_discrete_sequence=['#FF9900', '#146EB4', '#232F3E']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Note:** Having a balanced dataset (roughly equal number of samples per class) is beneficial 
        for model training, as it helps prevent bias toward majority classes.
        """)
    
    # Feature visualizations
    st.markdown("### Feature Visualizations")
    viz_tab1, viz_tab2 = st.tabs(["📈 Feature Distributions", "🔄 Feature Relationships"])
    
    with viz_tab1:
        st.markdown("##### Feature Distributions by Species")
        # Select feature to visualize
        feature_to_viz = st.selectbox(
            "Select feature to visualize:", 
            options=st.session_state.feature_names
        )
        
        # Create histogram for the selected feature
        fig = px.histogram(
            iris_df, 
            x=feature_to_viz, 
            color='species',
            marginal='box',
            opacity=0.7,
            barmode='overlay',
            color_discrete_sequence=['#FF9900', '#146EB4', '#232F3E'],
            title=f'Distribution of {feature_to_viz} by Species'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Analysis tips:**
        - Look for clear separation between species distributions
        - Features with less overlap between classes are better predictors
        - Box plots show median, quartiles, and potential outliers
        """)
    
    with viz_tab2:
        st.markdown("##### Feature Relationships")
        
        # Select features for scatter plot
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox(
                "X-axis feature:",
                options=st.session_state.feature_names,
                index=0
            )
        with col2:
            y_feature = st.selectbox(
                "Y-axis feature:",
                options=st.session_state.feature_names,
                index=2
            )
        
        # Create scatter plot
        fig = px.scatter(
            iris_df, 
            x=x_feature, 
            y=y_feature, 
            color='species',
            color_discrete_sequence=['#FF9900', '#146EB4', '#232F3E'],
            title=f'{x_feature} vs {y_feature} by Species'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Why this matters:**
        - Visual separation of clusters indicates features useful for classification
        - Feature pairs with clear separation are good candidates for decision boundaries
        - This helps understand which measurements are most useful for distinguishing species
        """)
        
        st.info("""
        💡 **Key Insight:** The petal dimensions (length and width) generally show better separation
        between species than sepal dimensions, making them stronger predictors for the model.
        This aligns with the feature importance results from our trained model.
        """)

# Topic 4: Learning ML Concepts Tab
def show_ml_concepts():
    st.markdown("<h2>Understanding Supervised Learning</h2>", unsafe_allow_html=True)
    st.markdown("""
    This section explains key supervised learning concepts as they apply to our Iris classification example.
    Understanding these concepts is crucial for AWS AI Practitioners.
    """)
    
    concepts_tab1, concepts_tab2, concepts_tab3, concepts_tab4 = st.tabs([
        "🎓 What is Supervised Learning?", 
        "🔄 ML Training Process", 
        "📊 Classification vs Regression", 
        "🧩 ML Model Types"
    ])
    
    with concepts_tab1:
        st.markdown("### What is Supervised Learning?")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            **Supervised learning** is a machine learning approach where the model learns from labeled training data.
            
            **Key characteristics:**
            - Training data includes both input features and correct output labels
            - Model learns to map inputs to outputs based on example data
            - Goal is to learn a general rule that can map new inputs to outputs
            
            **In our Iris example:**
            - Input features: sepal length, sepal width, petal length, petal width
            - Output labels: species (setosa, versicolor, or virginica)
            - The model learns patterns in measurements that identify each species
            """)
        
        with col2:
            # Illustration of supervised learning concept
            st.markdown("""
            #### How Supervised Learning Works:
            """)
            
            st.markdown("""
            1️⃣ **Training Phase:**  
            Input + Known Label → Model Learning
            
            2️⃣ **Testing Phase:**  
            New Input → Model Prediction → Compare with Actual
            
            3️⃣ **Production Phase:**  
            Unknown Input → Model → Prediction
            """)
    
    with concepts_tab2:
        st.markdown("### ML Training Process")
        
        st.markdown("""
        The supervised learning process follows these key steps:
        """)
        
        # Steps in the ML process
        steps = [
            {
                "title": "1. Data Collection & Preparation",
                "description": "Gather labeled data and prepare it for training (cleaning, formatting, etc.)",
                "image": "https://img.icons8.com/color/96/000000/data-configuration.png"
            },
            {
                "title": "2. Data Splitting",
                "description": "Divide data into training set (to learn from) and test set (to evaluate)",
                "image": "https://img.icons8.com/color/96/000000/split.png"
            },
            {
                "title": "3. Model Selection & Training",
                "description": "Choose algorithm type and train model on training data",
                "image": "https://img.icons8.com/color/96/000000/artificial-intelligence.png"
            },
            {
                "title": "4. Model Evaluation",
                "description": "Test model on unseen data and measure performance metrics",
                "image": "https://img.icons8.com/color/96/000000/combo-chart--v1.png"
            },
            {
                "title": "5. Model Tuning",
                "description": "Adjust parameters and features to improve performance",
                "image": "https://img.icons8.com/color/96/000000/settings--v1.png"
            },
            {
                "title": "6. Deployment",
                "description": "Deploy model to production environment for real-world use",
                "image": "https://img.icons8.com/color/96/000000/launch-box.png"
            }
        ]
        
        # Display steps in columns
        cols = st.columns(3)
        for i, step in enumerate(steps):
            col_index = i % 3
            with cols[col_index]:
                st.markdown(f"<div style='text-align: center;'><img src='{step['image']}' width='50'></div>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center; color: #232F3E;'>{step['title']}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>{step['description']}</p>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color:#FFEBCC; padding:15px; border-radius:10px; margin-top:20px;'>
        <strong>AWS ML Lifecycle:</strong> The AWS machine learning development lifecycle follows a similar pattern, 
        with tools like SageMaker to streamline each step.
        </div>
        """, unsafe_allow_html=True)
    
    with concepts_tab3:
        st.markdown("### Classification vs Regression")
        
        st.markdown("""
        Supervised learning tasks can be divided into two main categories: classification and regression.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📊 Classification
            
            **Predicts discrete categories or classes**
            
            **Types:**
            - Binary Classification (2 classes)
            - Multiclass Classification (3+ classes)
            
            **Examples:**
            - Email spam detection (spam/not spam)
            - Medical diagnosis (disease present/absent)
            - **Iris flower classification (our example)**
            
            **Evaluation metrics:**
            - Accuracy, Precision, Recall
            - F1-score, Confusion Matrix
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Regression
            
            **Predicts continuous values**
            
            **Types:**
            - Linear Regression
            - Multiple Regression
            - Polynomial Regression
            
            **Examples:**
            - House price prediction
            - Temperature forecasting
            - Age estimation from biological markers
            
            **Evaluation metrics:**
            - Mean Squared Error (MSE)
            - Root Mean Squared Error (RMSE)
            - R-squared (R²) value
            """)
    
    with concepts_tab4:
        st.markdown("### ML Model Types")
        
        st.markdown("""
        Several algorithms can be used for supervised learning. Here are some common ones:
        """)
        
        models = [
            {
                "name": "Random Forest",
                "description": "Ensemble of decision trees, trained on different parts of the same dataset. Used in our Iris example.",
                "use_case": "Good for classification and regression, handles large datasets and many features.",
                "icon": "🌲"
            },
            {
                "name": "Logistic Regression",
                "description": "Models the probability of a class or event using a logistic function.",
                "use_case": "Binary classification problems, interpretable results, probability estimation.",
                "icon": "📉"
            },
            {
                "name": "Support Vector Machines",
                "description": "Finds an optimal hyperplane to separate classes in high-dimensional space.",
                "use_case": "Classification tasks, works well with small datasets, effective in high dimensions.",
                "icon": "📊"
            },
            {
                "name": "Neural Networks",
                "description": "Layers of interconnected nodes that process and transform data.",
                "use_case": "Complex patterns, image recognition, language processing, high-dimensional data.",
                "icon": "🧠"
            },
            {
                "name": "k-Nearest Neighbors",
                "description": "Classifies based on majority vote of k nearest training examples.",
                "use_case": "Simple classification and regression, works with little tuning, intuitive approach.",
                "icon": "👥"
            }
        ]
        
        # Display model information in an expandable container
        for model in models:
            with st.expander(f"{model['icon']} {model['name']}"):
                st.markdown(f"**Description:** {model['description']}")
                st.markdown(f"**Common Use Cases:** {model['use_case']}")
        
        st.info("""
        💡 **AWS Perspective:** Amazon SageMaker provides built-in algorithms for all these model types, 
        allowing easy deployment of supervised learning solutions without managing infrastructure.
        """)

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Check if initialization is complete
    if "initialized" not in st.session_state or not st.session_state.initialized:
        st.warning("Initializing application... Please refresh the page if this message persists.")
        st.stop()
    
    # Custom CSS
    load_css()
    
    # Header
    st.markdown("<h1>🌺 Iris Flower Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>The Iris flower prediction task serves as a canonical example of multi-class classification in machine learning, where models learn to distinguish between three species of Iris flowers (setosa, versicolor, and virginica) based on four features—sepal length, sepal width, petal length, and petal width—demonstrating how algorithms can effectively categorize instances into multiple distinct classes.</div>""", unsafe_allow_html=True)

    
    # Sidebar
    with st.sidebar:

        # Session info
        render_sidebar()
            
        # About accordion
        with st.expander("About this App"):
            st.markdown("""
            This application demonstrates supervised machine learning concepts for AWS AI Practitioners 
            using the Iris flower dataset. 
            
            **Topics covered:**
            - Machine learning prediction with multiclass classification
            - Model performance evaluation
            - Data exploration and visualization
            - Supervised learning concepts
            
            Built with Streamlit and scikit-learn.
            """)

        
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Make a Prediction", 
        "📊 Model Performance", 
        "🔍 Data Exploration", 
        "🎓 ML Concepts"
    ])
    
    with tab1:
        show_prediction_tab()
    
    with tab2:
        show_model_performance()
    
    with tab3:
        show_data_exploration()
    
    with tab4:
        show_ml_concepts()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666666; padding: 10px;'>© 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
        unsafe_allow_html=True
    )


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

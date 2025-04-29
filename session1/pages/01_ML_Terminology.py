import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
from datetime import datetime
import json
import uuid
import os

# Configuration
AWS_COLORS = {
    'primary': '#232F3E',      # AWS Dark Blue
    'secondary': '#FF9900',    # AWS Orange
    'tertiary': '#1A2A3A',     # Darker Blue
    'light': '#EAEDED',        # Light Gray
    'success': '#1E8E3E',      # Green
    'warning': '#FFC107',      # Amber
    'error': '#D13212',        # Red
    'info': '#0073BB',         # AWS Light Blue
}

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_responses' not in st.session_state:
        st.session_state.quiz_responses = {}
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

# CSS for styling
def load_css():
    """Load custom CSS for styling the app"""
    st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
        color: #232F3E;
    }
    .stApp {
        font-family: 'Amazon Ember', Arial, sans-serif;
    }
    h1, h2, h3 {
        color: #232F3E;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #EAEDED;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding: 10px 16px;
        color: #232F3E;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF9900 !important;
        color: #232F3E !important;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .st-eb {
        background-color: rgb(255, 255, 255);
    }
    .st-c2 {
        background-color: rgba(0, 0, 0, 0.75)
    }
    .st-eq {
        background-color: rgba(0, 0, 0, 0);;
    }
    .st-emotion-cache-pkbazv {
        color: #232F3E;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #EAEDED;
        color: #232F3E;
        text-align: center;
        padding: 10px 0;
        font-size: 12px;
    }
    .stButton>button {
        background-color: #FF9900;
        color: #232F3E;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #EC7211;
    }
    .card {
        background-color: #EAEDED;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .metric-container {
        background-color: #232F3E;
        color: white;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 16px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #FF9900;
    }
    .metric-label {
        font-size: 14px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a custom dataset
@st.cache_data
def create_custom_dataset(n_samples=100):
    """Create a simple dataset for house prices with clear features."""
    np.random.seed(42)
    
    # Features
    square_feet = np.random.randint(1000, 4000, size=n_samples)  # Square footage of house
    num_bedrooms = np.random.randint(1, 6, size=n_samples)      # Number of bedrooms
    num_bathrooms = np.random.randint(1, 4, size=n_samples)     # Number of bathrooms
    age_of_house = np.random.randint(0, 50, size=n_samples)     # Age of house in years
    
    # Target variable (house price) with some noise
    price = (
        150000 +                               # Base price
        100 * square_feet +                   # $100 per square foot
        15000 * num_bedrooms +                # $15,000 per bedroom
        20000 * num_bathrooms +               # $20,000 per bathroom
        -1000 * age_of_house +                # -$1,000 per year of age
        np.random.normal(0, 20000, n_samples) # Random noise
    )
    
    # Create a DataFrame
    df = pd.DataFrame({
        'SquareFeet': square_feet,
        'Bedrooms': num_bedrooms,
        'Bathrooms': num_bathrooms,
        'HouseAge': age_of_house,
        'Price': price
    })
    
    return df

# Load built-in datasets
@st.cache_data
def load_datasets():
    """Load and prepare various datasets for demonstration"""
    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['target_name'] = [iris.target_names[t] for t in iris.target]
    
    # Load California Housing dataset
    california = fetch_california_housing()
    california_df = pd.DataFrame(california.data, columns=california.feature_names)
    california_df['target'] = california.target
    
    # Load Diabetes dataset
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['target'] = diabetes.target
    
    # Load Wine dataset
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    wine_df['target_name'] = [wine.target_names[t] for t in wine.target]
    
    # Create custom dataset
    custom_df = create_custom_dataset()
    
    return {
        'House Price (Custom)': custom_df,
        'California Housing': california_df,
        'Iris Classification': iris_df,
        'Wine Classification': wine_df,
        'Diabetes Regression': diabetes_df
    }

# Dictionary of ML terminology
def get_ml_terminology():
    """Return a dictionary of ML terminology with explanations"""
    return {
        'Dataset': {
            'definition': 'A collection of data used for machine learning.',
            'example': 'The California Housing dataset contains information about housing districts in California.',
            'visual_type': 'table'
        },
        'Features': {
            'definition': 'Individual measurable properties or characteristics of the phenomena being observed.',
            'example': 'In a housing dataset, features might include median income, house age, average rooms, etc.',
            'visual_type': 'column_highlight'
        },
        'Target (Label/Output/Dependent Variable)': {
            'definition': 'The variable you want to predict or classify.',
            'example': 'In the California Housing dataset, the target is the median house value.',
            'visual_type': 'column_highlight'
        },
        'Observation (Instance/Sample/Example)': {
            'definition': 'A single data point or record in the dataset.',
            'example': 'In the California Housing dataset, each row represents a housing district with its features.',
            'visual_type': 'row_highlight'
        },
        'Feature Matrix (X)': {
            'definition': 'The entire set of input features, typically represented as a 2D matrix where rows are observations and columns are features.',
            'example': 'All columns except "target" in the California Housing dataset form the feature matrix.',
            'visual_type': 'matrix'
        },
        'Target Vector (y)': {
            'definition': 'The collection of all target values, typically represented as a 1D vector.',
            'example': 'The "target" column in the California Housing dataset is the target vector.',
            'visual_type': 'vector'
        },
        'Training Set': {
            'definition': 'A subset of the data used to train the model.',
            'example': '70-80% of the housing data used for the model to learn patterns.',
            'visual_type': 'split'
        },
        'Testing Set': {
            'definition': 'A subset of the data used to evaluate the model\'s performance.',
            'example': '20-30% of the housing data kept separate to test the model.',
            'visual_type': 'split'
        },
        'Validation Set': {
            'definition': 'A subset of the training data used to tune hyperparameters and prevent overfitting.',
            'example': 'A portion of the training data used to validate model performance during training.',
            'visual_type': 'split'
        },
        'Classification': {
            'definition': 'Predicting categorical class labels or discrete outcomes.',
            'example': 'Predicting whether an email is spam or not spam.',
            'visual_type': 'classification'
        },
        'Regression': {
            'definition': 'Predicting continuous numerical values.',
            'example': 'Predicting house prices based on features.',
            'visual_type': 'regression'
        },
        'Model': {
            'definition': 'An algorithm or mathematical construct that learns patterns from data to make predictions.',
            'example': 'Linear regression, random forest, or neural networks are examples of models.',
            'visual_type': 'model'
        },
        'Prediction': {
            'definition': 'The output value(s) produced by a trained model when given new input data.',
            'example': 'The estimated median house value based on district features.',
            'visual_type': 'prediction'
        },
        'Accuracy': {
            'definition': 'The proportion of correct predictions made by the model.',
            'example': 'If a model correctly classifies 90 out of 100 emails as spam or not spam, the accuracy is 90%.',
            'visual_type': 'metrics'
        },
        'Overfitting': {
            'definition': 'When a model learns the training data too well, including the noise, leading to poor performance on new data.',
            'example': 'A model that perfectly predicts house values in the training data but fails on new districts.',
            'visual_type': 'overfitting'
        },
        'Underfitting': {
            'definition': 'When a model is too simple to capture the underlying pattern in the data.',
            'example': 'Using a linear model to predict house values when the relationship is non-linear.',
            'visual_type': 'underfitting'
        }
    }

# Define the quiz questions
def get_quiz_questions():
    """Return a list of quiz questions with options and answers"""
    return [
        {
            "question": "What is the primary difference between regression and classification?",
            "type": "radio",
            "options": [
                "Regression predicts categorical outcomes, classification predicts continuous values.",
                "Regression predicts continuous values, classification predicts categorical outcomes.",
                "Regression works with structured data, classification works with unstructured data.",
                "Regression can only be performed using linear models, classification can use any model."
            ],
            "correct_answer": 1,
            "explanation": "Regression predicts continuous numerical values (like house prices), while classification predicts categorical outcomes or class labels (like spam/not spam)."
        },
        {
            "question": "Which of these are considered features in a machine learning dataset? (Select all that apply)",
            "type": "checkbox",
            "options": [
                "Input variables used for making predictions",
                "The outcome variable we're trying to predict",
                "Columns in a dataset that describe the properties of observations",
                "The algorithm used to make predictions"
            ],
            "correct_answer": [0, 2],
            "explanation": "Features are the input variables (columns in a dataset) that describe the properties of observations and are used by the model to make predictions."
        },
        {
            "question": "What is overfitting?",
            "type": "radio",
            "options": [
                "When a model performs well on both training and test data",
                "When a model is too simple to capture patterns in the data",
                "When a model learns the training data too well and performs poorly on new data",
                "When a model is trained on too many examples"
            ],
            "correct_answer": 2,
            "explanation": "Overfitting occurs when a model learns the training data too well, capturing noise rather than just the underlying pattern, which leads to poor performance on new, unseen data."
        },
        {
            "question": "What is the purpose of splitting data into training and testing sets?",
            "type": "radio",
            "options": [
                "To make the training process faster",
                "To evaluate model performance on unseen data",
                "To reduce the amount of data needed for training",
                "To make the dataset more balanced"
            ],
            "correct_answer": 1,
            "explanation": "We split data into training and testing sets to evaluate how well the model will perform on new, unseen data. The test set serves as a proxy for real-world data the model will encounter."
        },
        {
            "question": "Which of these are examples of regression problems? (Select all that apply)",
            "type": "checkbox",
            "options": [
                "Predicting house prices based on square footage and location",
                "Classifying emails as spam or not spam",
                "Forecasting stock prices for the next month",
                "Predicting a patient's age based on medical records"
            ],
            "correct_answer": [0, 2, 3],
            "explanation": "Regression problems involve predicting continuous numerical values. House prices, stock prices, and patient age are all continuous values, making these regression problems. Classifying emails is a classification problem since the output is categorical (spam/not spam)."
        },
        {
            "question": "What is a validation set used for?",
            "type": "radio",
            "options": [
                "To train the model faster",
                "To tune hyperparameters and prevent overfitting",
                "To make the final model evaluation",
                "To increase the size of the training data"
            ],
            "correct_answer": 1,
            "explanation": "A validation set is used during the training process to tune hyperparameters and prevent overfitting. It helps optimize the model before final testing on the test set."
        },
        {
            "question": "Which of these metrics are typically used for classification tasks? (Select all that apply)",
            "type": "checkbox",
            "options": [
                "Mean Squared Error (MSE)",
                "Accuracy",
                "Precision and Recall",
                "R-squared"
            ],
            "correct_answer": [1, 2],
            "explanation": "Accuracy, precision, and recall are metrics used for classification tasks. Mean Squared Error (MSE) and R-squared are typically used for regression tasks."
        },
        {
            "question": "What is underfitting in machine learning?",
            "type": "radio",
            "options": [
                "When a model is too complex and captures noise in the training data",
                "When a model is too simple to capture the underlying pattern in the data",
                "When a model uses too many features",
                "When a training dataset is too small"
            ],
            "correct_answer": 1,
            "explanation": "Underfitting occurs when a model is too simple to capture the underlying pattern in the data. This results in poor performance on both training and testing data."
        },
        {
            "question": "Which of these are components of a dataset? (Select all that apply)",
            "type": "checkbox",
            "options": [
                "Features",
                "Target variable",
                "Observations",
                "Algorithms"
            ],
            "correct_answer": [0, 1, 2],
            "explanation": "A dataset consists of features (input variables), a target variable (what we want to predict), and observations (individual data points/rows). Algorithms are used to process the dataset but aren't part of the dataset itself."
        },
        {
            "question": "What does the term 'bias-variance tradeoff' refer to?",
            "type": "radio",
            "options": [
                "The tradeoff between model complexity and computational efficiency",
                "The tradeoff between underfitting and overfitting",
                "The tradeoff between training set size and testing set size",
                "The tradeoff between feature engineering and feature selection"
            ],
            "correct_answer": 1,
            "explanation": "The bias-variance tradeoff refers to the tradeoff between underfitting (high bias) and overfitting (high variance). Finding the right balance is crucial for building models that generalize well to new data."
        }
    ]

# Initialize quiz state
def init_quiz_state():
    """Initialize quiz state variables"""
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_responses' not in st.session_state:
        st.session_state.quiz_responses = {}
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'questions_attempted' not in st.session_state:
        st.session_state.questions_attempted = set()
    if 'question_submitted' not in st.session_state:
        st.session_state.question_submitted = {}

# Reset quiz progress
def reset_quiz_progress():
    """Reset all quiz-related session state variables"""
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.quiz_responses = {}
    st.session_state.current_question = 0
    st.session_state.questions_attempted = set()
    st.session_state.question_submitted = {}

# Navigate to next question
def next_question():
    current_q = st.session_state.current_question
    quiz_questions = get_quiz_questions()
    if current_q < len(quiz_questions) - 1:
        st.session_state.current_question = current_q + 1

# Navigate to previous question
def prev_question():
    current_q = st.session_state.current_question
    if current_q > 0:
        st.session_state.current_question = current_q - 1

# Submit quiz
def submit_quiz():
    st.session_state.quiz_submitted = True
    # Calculate score
    correct_answers = sum(1 for resp in st.session_state.quiz_responses.values() if resp["correct"])
    st.session_state.quiz_score = correct_answers

# Submit question answer
def submit_question_answer(question_idx):
    st.session_state.question_submitted[question_idx] = True
    st.session_state.questions_attempted.add(question_idx)

# Visual explanations based on term type
def render_term_visualization(term_info, selected_term, current_df):
    """Render visualizations based on the selected ML term"""
    if term_info['visual_type'] == 'table':
        render_table_visualization(current_df)
    elif term_info['visual_type'] == 'column_highlight':
        render_column_highlight(selected_term, current_df)
    elif term_info['visual_type'] == 'row_highlight':
        render_row_highlight(current_df)
    elif term_info['visual_type'] == 'matrix':
        render_matrix_visualization(current_df)
    elif term_info['visual_type'] == 'vector':
        render_vector_visualization(current_df)
    elif term_info['visual_type'] == 'split':
        render_split_visualization(selected_term)
    elif term_info['visual_type'] == 'classification':
        render_classification_visualization(current_df, selected_term)
    elif term_info['visual_type'] == 'regression':
        render_regression_visualization(current_df)
    elif term_info['visual_type'] == 'model':
        render_model_visualization(current_df)
    elif term_info['visual_type'] == 'prediction':
        render_prediction_visualization(current_df)
    elif term_info['visual_type'] == 'metrics':
        render_metrics_visualization(current_df)
    elif term_info['visual_type'] == 'overfitting':
        render_overfitting_visualization()
    elif term_info['visual_type'] == 'underfitting':
        render_underfitting_visualization()

def render_table_visualization(current_df):
    """Render table visualization for dataset term"""
    st.subheader("Visual Explanation: Dataset")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        A **dataset** is a collection of observations (rows) with their features and target values.
        
        Key components of a dataset:
        - **Features**: The input variables (columns)
        - **Target**: The output variable to predict
        - **Observations**: Individual data points (rows)
        """)
    
    with col2:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(current_df.columns),
                fill_color=AWS_COLORS['primary'],
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[current_df[col] for col in current_df.columns],
                fill_color=AWS_COLORS['light'],
                align='center'
            )
        )])
        
        fig.update_layout(
            title='Complete Dataset',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_column_highlight(selected_term, current_df):
    """Render column highlight visualization for features or target"""
    st.subheader(f"Visual Explanation: {'Features' if 'Features' in selected_term else 'Target'}")
    
    if 'Features' in selected_term:
        highlight_cols = current_df.columns[:-1]  # All columns except the last (assuming last column is target)
        non_highlight_cols = current_df.columns[-1:]
        title = "Features are the input variables (highlighted columns)"
    else:  # Target
        non_highlight_cols = current_df.columns[:-1]
        highlight_cols = current_df.columns[-1:]
        title = "Target is the output variable to predict (highlighted column)"
    
    # Create a DataFrame with highlighted columns
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(current_df.columns),
            fill_color=[AWS_COLORS['secondary'] if col in highlight_cols else AWS_COLORS['primary'] for col in current_df.columns],
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[current_df[col] for col in current_df.columns],
            fill_color=[AWS_COLORS['light'] if col in highlight_cols else '#f0f0f0' for col in current_df.columns],
            align='center'
        )
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_row_highlight(current_df):
    """Render row highlight visualization for observation term"""
    st.subheader("Visual Explanation: Observation")
    
    # Choose a random row to highlight
    highlight_row = np.random.randint(0, len(current_df))
    
    # Display the dataframe with the highlighted row
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(current_df.columns),
            fill_color=AWS_COLORS['primary'],
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[current_df[col] for col in current_df.columns],
            fill_color=[[AWS_COLORS['secondary'] if i == highlight_row else AWS_COLORS['light'] for i in range(len(current_df))] for _ in range(len(current_df.columns))],
            align='center'
        )
    )])
    
    fig.update_layout(
        title='An Observation is a single data point (highlighted row)',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show the observation as a "record"
    st.markdown("### Observation as a Record")
    
    observation = current_df.iloc[highlight_row].to_dict()
    
    cols = st.columns(len(observation))
    for idx, (feature, value) in enumerate(observation.items()):
        with cols[idx]:
            if feature == current_df.columns[-1]:
                st.markdown(f"""
                <div class="card" style="background-color: {AWS_COLORS['secondary']}; color: white;">
                    <p style="font-size: 14px;">Target: {feature}</p>
                    <p style="font-size: 20px; font-weight: bold;">{f"{value:.2f}" if isinstance(value, (float, np.floating)) else value}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card">
                    <p style="font-size: 14px;">Feature: {feature}</p>
                    <p style="font-size: 20px; font-weight: bold;">{f"{value:.2f}" if isinstance(value, (float, np.floating)) else value}</p>
                </div>
                """, unsafe_allow_html=True)

def render_matrix_visualization(current_df):
    """Render matrix visualization for feature matrix term"""
    st.subheader("Visual Explanation: Feature Matrix (X)")
    
    # Assuming the last column is the target
    X = current_df.iloc[:, :-1]
    y = current_df.iloc[:, -1]
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        The **Feature Matrix (X)** contains all input features for all observations.
        
        It's a 2D matrix where:
        - Each **row** is an observation
        - Each **column** is a feature
        
        In mathematical notation, X often represents the feature matrix.
        """)
    
    with col2:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(X.columns),
                fill_color=AWS_COLORS['primary'],
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[X[col] for col in X.columns],
                fill_color=AWS_COLORS['light'],
                align='center'
            )
        )])
        
        fig.update_layout(
            title='Feature Matrix (X): All input features for all observations',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show matrix shape
    st.markdown(f"""
    <div class="card">
        <p style="font-size: 18px; font-weight: bold;">Feature Matrix Shape:</p>
        <p>{X.shape[0]} rows (observations) × {X.shape[1]} columns (features)</p>
    </div>
    """, unsafe_allow_html=True)

def render_vector_visualization(current_df):
    """Render vector visualization for target vector term"""
    st.subheader("Visual Explanation: Target Vector (y)")
    
    # Assuming the last column is the target
    y = current_df.iloc[:, -1]
    target_name = current_df.columns[-1]
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown(f"""
        The **Target Vector (y)** contains all target values.
        
        It's a 1D vector where:
        - Each element corresponds to an observation's target value
        - In this case, the target is **{target_name}**
        
        In mathematical notation, y often represents the target vector.
        """)
    
    with col2:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[target_name],
                fill_color=AWS_COLORS['secondary'],
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[y],
                fill_color=AWS_COLORS['light'],
                align='center'
            )
        )])
        
        fig.update_layout(
            title='Target Vector (y): All target values',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show target distribution
    st.subheader("Target Distribution")
    
    if len(y.unique()) < 10:  # Categorical target
        fig = px.histogram(current_df, x=target_name, color=target_name if 'target_name' in current_df.columns else None,
                         color_discrete_sequence=[AWS_COLORS['primary'], AWS_COLORS['secondary'], AWS_COLORS['info']])
    else:  # Continuous target
        fig = px.histogram(current_df, x=target_name, color_discrete_sequence=[AWS_COLORS['secondary']])
        
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(
            title=target_name,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Count',
            gridcolor='lightgray'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_split_visualization(selected_term):
    """Render split visualization for train/test/validation term"""
    st.subheader("Visual Explanation: Data Splitting")
    
    # Create a sample split
    train_size = 0.7 if 'Training' in selected_term else 0.6
    test_size = 0.3 if 'Testing' in selected_term else 0.2
    val_size = 0.0 if 'Validation' not in selected_term else 0.2
    
    # Generate indices for splitting
    n_samples = 100
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_end = int(train_size * n_samples)
    test_start = n_samples - int(test_size * n_samples)
    
    train_indices = indices[:train_end]
    test_indices = indices[test_start:]
    val_indices = indices[train_end:test_start] if val_size > 0 else []
    
    # Create visual representation of split
    split_df = pd.DataFrame({
        'index': range(n_samples),
        'set': ['Training' if i in train_indices else 
                'Testing' if i in test_indices else 
                'Validation' for i in range(n_samples)]
    })
    
    colors = {'Training': AWS_COLORS['success'], 'Testing': AWS_COLORS['secondary'], 'Validation': AWS_COLORS['info']}
    
    fig = px.scatter(split_df, x='index', y=[1] * n_samples, 
                   color='set', color_discrete_map=colors,
                   title=f"Data Splitting: {int(train_size*100)}% Training, {int(test_size*100)}% Testing{f', {int(val_size*100)}% Validation' if val_size > 0 else ''}",
                   labels={'index': 'Data points', 'y': ''})
    
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(height=300, showlegend=True, plot_bgcolor='white')
    fig.update_yaxes(showticklabels=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the purpose
    if 'Training' in selected_term:
        st.markdown("""
        <div class="card">
            <h3>Training Set</h3>
            <p>The <b>Training Set</b> is used to train the model. This is where the model learns patterns from the data.</p>
            <ul>
                <li>Usually the largest portion (70-80%) of the data</li>
                <li>The model sees these examples during training</li>
                <li>The model adjusts its parameters based on this data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif 'Testing' in selected_term:
        st.markdown("""
        <div class="card">
            <h3>Testing Set</h3>
            <p>The <b>Testing Set</b> is used to evaluate the model's performance on unseen data.</p>
            <ul>
                <li>Usually 20-30% of the data</li>
                <li>The model never sees this data during training</li>
                <li>Used to estimate how well the model will perform on new, unseen data</li>
                <li>Helps detect overfitting (when a model performs well on training data but poorly on test data)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:  # Validation
        st.markdown("""
        <div class="card">
            <h3>Validation Set</h3>
            <p>The <b>Validation Set</b> is used during the training process to tune hyperparameters and prevent overfitting.</p>
            <ul>
                <li>A portion of the training data set aside</li>
                <li>Used to evaluate the model during training, not after</li>
                <li>Helps in selecting the best model configuration</li>
                <li>Different from the test set, which is only used after training is complete</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_classification_visualization(current_df, selected_term):
    """Render classification visualization"""
    st.subheader("Visual Explanation: Classification")
    
    # Use a classification dataset
    if 'target_name' in current_df.columns:
        df = current_df
    else:
        datasets = load_datasets()
        df = datasets['Iris Classification']
    
    # Select two features for visualization
    if 'Iris' in selected_term or df.equals(datasets['Iris Classification']):
        feature1, feature2 = 'sepal length (cm)', 'sepal width (cm)'
        target = 'target_name'
    elif 'Wine' in selected_term or df.equals(datasets['Wine Classification']):
        feature1, feature2 = df.columns[0], df.columns[1]
        target = 'target_name'
    else:
        # Generic features
        feature1, feature2 = df.columns[0], df.columns[1]
        target = df.columns[-1]
    
    # Create scatter plot
    fig = px.scatter(df, x=feature1, y=feature2, color=target,
                   title="Classification: Predicting Categorical Outcomes",
                   labels={feature1: feature1, feature2: feature2, target: "Class"},
                   color_discrete_sequence=[AWS_COLORS['primary'], AWS_COLORS['secondary'], AWS_COLORS['info']])
    
    # Add decision boundary (simulated for visualization)
    if len(df[target].unique()) <= 3:  # Only if we have a reasonable number of classes
        X = df[[feature1, feature2]].values
        y = df[target].astype('category').cat.codes.values
        
        # Train a simple model to get a decision boundary
        model = RandomForestClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                           np.arange(y_min, y_max, 0.1))
        
        # Predict on the mesh grid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Add contour lines to show decision boundaries
        fig.add_trace(
            go.Contour(
                x=np.arange(x_min, x_max, 0.1),
                y=np.arange(y_min, y_max, 0.1),
                z=Z,
                showscale=False,
                colorscale='Viridis',
                opacity=0.4,
                line=dict(width=0),
                contours=dict(showlabels=False)
            )
        )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="card">
        <h3>Classification</h3>
        <p><b>Classification</b> is a type of supervised learning where the model predicts categorical outcomes.</p>
        <p><b>Examples:</b></p>
        <ul>
            <li>Email spam detection (spam/not spam)</li>
            <li>Disease diagnosis (positive/negative)</li>
            <li>Image recognition (cat/dog/bird/etc.)</li>
        </ul>
        <p><b>In the plot above:</b></p>
        <ul>
            <li>Each point represents an observation</li>
            <li>Colors represent different classes</li>
            <li>The model learns to separate the classes based on features</li>
            <li>The contour lines show the decision boundaries</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_regression_visualization(current_df):
    """Render regression visualization"""
    st.subheader("Visual Explanation: Regression")
    
    # Use a regression dataset
    if 'Price' in current_df.columns or 'California' in current_df.columns.astype(str).tolist():
        df = current_df
        if 'California' in current_df.columns.astype(str).tolist():
            feature = 'MedInc'  # Median Income
            target = 'target'   # Median house value
        else:
            feature = 'SquareFeet'
            target = 'Price'
    else:
        datasets = load_datasets()
        df = datasets['Diabetes Regression']
        feature = df.columns[0]  # First feature
        target = 'target'
    
    # Create scatter plot
    fig = px.scatter(df, x=feature, y=target,
                   title="Regression: Predicting Continuous Values",
                   labels={feature: feature, target: target},
                   color_discrete_sequence=[AWS_COLORS['info']])
    
    # Add regression line
    X = df[feature].values.reshape(-1, 1)
    y = df[target].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    fig.add_trace(
        go.Scatter(
            x=df[feature],
            y=y_pred,
            mode='lines',
            name='Regression Line',
            line=dict(color=AWS_COLORS['secondary'], width=3)
        )
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show equation
    slope = model.coef_[0]
    intercept = model.intercept_
    
    st.markdown(f"""
    <div class="card">
        <h3>Regression</h3>
        <p><b>Regression</b> is a type of supervised learning where the model predicts continuous values.</p>
        <p><b>Examples:</b></p>
        <ul>
            <li>House price prediction</li>
            <li>Stock price forecasting</li>
            <li>Age estimation</li>
            <li>Sales forecasting</li>
        </ul>
        <p><b>In the plot above:</b></p>
        <ul>
            <li>Each point represents an observation</li>
            <li>The line shows the predicted relationship between the feature and target</li>
        </ul>
        <p><b>Regression Line Equation:</b></p>
        <p style="font-weight: bold;">{target} = {slope:.2f} × {feature} + {intercept:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

def render_model_visualization(current_df):
    """Render model visualization"""
    st.subheader("Visual Explanation: Model")
    
    # Use the California Housing dataset
    if 'California' in current_df.columns.astype(str).tolist():
        df = current_df
        target_col = 'target'
    else:
        datasets = load_datasets()
        df = datasets['California Housing']
        target_col = 'target'
    
    # Split data
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Extract coefficients
    coeffs = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    
    # Create visualization
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Model</h3>
            <p>A <b>Model</b> is an algorithm that learns patterns from data to make predictions.</p>
            <p>Models can be:</p>
            <ul>
                <li>Simple (linear regression)</li>
                <li>Complex (neural networks)</li>
                <li>Interpretable or black-box</li>
            </ul>
            <p>A trained model has learned parameters (coefficients, weights, etc.) 
            that determine how it maps inputs to outputs.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Plot coefficients
        fig = px.bar(coeffs, x='Feature', y='Coefficient',
                   title="Linear Regression Model Coefficients",
                   labels={'Coefficient': 'Impact on House Value'},
                   color='Coefficient',
                   color_continuous_scale=[AWS_COLORS['error'], AWS_COLORS['light'], AWS_COLORS['success']])
        
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-label">Mean Squared Error</p>
            <p class="metric-value">{mse:.2f}</p>
        </div>
        
        <div class="metric-container">
            <p class="metric-label">R² Score</p>
            <p class="metric-value">{r2:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Plot actual vs predicted
        fig = px.scatter(x=y_test, y=y_pred, 
                       labels={'x': 'Actual Value', 'y': 'Predicted Value'},
                       title='Model Predictions: Actual vs Predicted',
                       color_discrete_sequence=[AWS_COLORS['info']])
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                line=dict(color=AWS_COLORS['secondary'], dash='dash'),
                name='Perfect Predictions'
            )
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_prediction_visualization(current_df):
    """Render prediction visualization"""
    st.subheader("Visual Explanation: Prediction")
    
    # Use California Housing dataset
    if 'California' in current_df.columns.astype(str).tolist():
        df = current_df
    else:
        datasets = load_datasets()
        df = datasets['California Housing']
    
    # Train a simple model
    X = df.drop('target', axis=1)
    y = df['target']
    
    model = LinearRegression().fit(X, y)
    
    # Create input form with California Housing features
    st.markdown("""
    <div class="card">
        <h3>Prediction</h3>
        <p>Prediction is the output produced by a model when given input features.
        Try making a prediction by adjusting the input features below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        median_income = st.slider("Median Income (tens of thousands)", 
                               float(X['MedInc'].min()), 
                               float(X['MedInc'].max()), 
                               3.5,
                               key='median_income')
        
        house_age = st.slider("Housing Median Age", 
                           float(X['HouseAge'].min()), 
                           float(X['HouseAge'].max()), 
                           28.0,
                           key='house_age')
        
        avg_rooms = st.slider("Average Rooms per Household", 
                           float(X['AveRooms'].min()), 
                           float(X['AveRooms'].max()), 
                           5.0,
                           key='avg_rooms')
        
        avg_bedrooms = st.slider("Average Bedrooms per Household", 
                              float(X['AveBedrms'].min()), 
                              float(X['AveBedrms'].max()), 
                              1.0,
                              key='avg_bedrooms')
    
    with col2:
        population = st.slider("Block Population", 
                            float(X['Population'].min()), 
                            float(X['Population'].max()), 
                            1500.0,
                            key='population')
        
        avg_occupancy = st.slider("Average Household Occupancy", 
                               float(X['AveOccup'].min()), 
                               float(X['AveOccup'].max()), 
                               3.0,
                               key='avg_occupancy')
        
        latitude = st.slider("Latitude", 
                          float(X['Latitude'].min()), 
                          float(X['Latitude'].max()), 
                          35.0,
                          key='latitude')
        
        longitude = st.slider("Longitude", 
                           float(X['Longitude'].min()), 
                           float(X['Longitude'].max()), 
                           -120.0,
                           key='longitude')
    

    # Make prediction
    new_district = pd.DataFrame({
        'MedInc': [median_income],
        'HouseAge': [house_age],
        'AveRooms': [avg_rooms],
        'AveBedrms': [avg_bedrooms],
        'Population': [population],
        'AveOccup': [avg_occupancy],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })
    
    predicted_value = model.predict(new_district)[0]
    
    # Display prediction with nice formatting
    st.markdown("### Model Prediction")
    st.markdown(f"""
    <div style='background-color: {AWS_COLORS['secondary']}; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2 style='margin: 0; color: {AWS_COLORS['primary']};'>${predicted_value * 100000:,.2f}</h2>
        <p style='color: white;'>Predicted Median House Value</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show how the prediction was made
    st.markdown("### How the Prediction Works")
    
    # Calculate contribution of each feature
    intercept = model.intercept_
    contributions = {}
    
    for i, feature in enumerate(X.columns):
        contribution = model.coef_[i] * new_district[feature].values[0]
        contributions[feature] = contribution
    
    # Create waterfall chart data
    waterfall_labels = list(contributions.keys()) + ['Intercept', 'Predicted Value']
    waterfall_values = list(contributions.values()) + [intercept, 0]
    
    # Calculate the cumulative sum for the measure
    measure = ['relative'] * len(contributions) + ['absolute', 'total']
    
    fig = go.Figure(go.Waterfall(
        name="Prediction Breakdown",
        orientation="v",
        measure=measure,
        x=waterfall_labels,
        y=waterfall_values,
        connector={"line": {"color": AWS_COLORS['primary']}},
        increasing={"marker": {"color": AWS_COLORS['success']}},
        decreasing={"marker": {"color": AWS_COLORS['error']}},
        totals={"marker": {"color": AWS_COLORS['secondary']}}
    ))
    
    fig.update_layout(
        title="Contribution of Each Feature to the Prediction",
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a more readable explanation
    st.markdown("""
    <div class="card">
        <h4>Explanation of Feature Contributions:</h4>
        <p>The prediction is calculated as:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Format the contributions for better readability
    st.markdown(f"""
    <div class="card">
        <p><b>Base value (Intercept):</b> ${intercept * 100000:,.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    for feature, contribution in contributions.items():
        contribution_value = contribution * 100000  # Convert to dollar value
        icon = "📈" if contribution > 0 else "📉"
        color = AWS_COLORS['success'] if contribution > 0 else AWS_COLORS['error']
        st.markdown(f"""
        <div class="card" style="border-left: 5px solid {color};">
            <p>{icon} <b>{feature}:</b> ${contribution_value:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card" style="background-color: {AWS_COLORS['secondary']}; color: white;">
        <p style="font-size: 18px;"><b>Total: ${predicted_value * 100000:,.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

def render_metrics_visualization(current_df):
    """Render metrics visualization"""
    st.subheader("Visual Explanation: Model Evaluation Metrics")
    
    # Use a classification dataset for simplicity
    if 'target_name' in current_df.columns:
        df = current_df
    else:
        datasets = load_datasets()
        df = datasets['Iris Classification']
    
    # Split data
    X = df.drop(['target', 'target_name'] if 'target_name' in df.columns else ['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Display metrics
    st.markdown("""
    <div class="card">
        <h3>Evaluation Metrics</h3>
        <p><b>Evaluation metrics</b> measure how well a model performs. Different metrics are used for different types of problems.</p>
        <p>Common metrics include:</p>
        <ul>
            <li><b>Accuracy</b>: Proportion of correct predictions (classification)</li>
            <li><b>Precision</b>: Proportion of positive identifications that were actually correct</li>
            <li><b>Recall</b>: Proportion of actual positives that were identified correctly</li>
            <li><b>F1 Score</b>: Harmonic mean of precision and recall</li>
            <li><b>Mean Squared Error (MSE)</b>: Average squared difference between predictions and actual values (regression)</li>
            <li><b>R²</b>: Proportion of variance explained by the model (regression)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-label">Accuracy</p>
            <p class="metric-value">{accuracy:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'target_name' in df.columns:
            st.markdown("""
            <div class="card">
                <p><b>Model correctly classified:</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            for i, class_name in enumerate(np.unique(df['target_name'])):
                correct = cm[i, i]
                total = np.sum(cm[i, :])
                percentage = correct/total * 100
                
                st.markdown(f"""
                <div class="card" style="background-color: {AWS_COLORS['light']}; margin-bottom: 5px; padding: 8px;">
                    <p style="margin-bottom: 5px;">
                        <b>{class_name}:</b> {correct}/{total} ({percentage:.0f}%)
                    </p>
                    <div style="width: 100%; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="width: {percentage}%; background-color: {AWS_COLORS['secondary']}; height: 10px; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Create confusion matrix plot
        fig = px.imshow(cm,
                       labels=dict(x="Predicted Label", y="True Label", color="Count"),
                       x=[f'Class {i}' for i in range(len(cm))],
                       y=[f'Class {i}' for i in range(len(cm))],
                       text_auto=True,
                       title="Confusion Matrix",
                       color_continuous_scale=[AWS_COLORS['light'], AWS_COLORS['secondary']])
                       
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="card">
        <h3>Understanding the Confusion Matrix</h3>
        <p>The confusion matrix shows:</p>
        <ul>
            <li><b>True Positives (TP)</b>: Correctly predicted positive cases (diagonal elements)</li>
            <li><b>False Positives (FP)</b>: Incorrectly predicted as positive (off-diagonal columns)</li>
            <li><b>False Negatives (FN)</b>: Incorrectly predicted as negative (off-diagonal rows)</li>
            <li><b>True Negatives (TN)</b>: Correctly predicted negative cases</li>
        </ul>
        <p>From these, we can calculate metrics like precision (TP/(TP+FP)) and recall (TP/(TP+FN)).</p>
    </div>
    """, unsafe_allow_html=True)

def render_overfitting_visualization():
    """Render overfitting visualization"""
    st.subheader("Visual Explanation: Overfitting")
    
    # Generate a synthetic dataset to demonstrate overfitting
    np.random.seed(0)
    n_samples = 30
    x = np.linspace(0, 10, n_samples)
    y = 0.5 * x + np.sin(x) + np.random.normal(0, 0.5, n_samples)
    
    # Create DataFrame
    df_overfit = pd.DataFrame({'x': x, 'y': y})
    
    # Function to fit polynomial models of different degrees
    def fit_polynomial(x, y, degree):
        model = np.poly1d(np.polyfit(x, y, degree))
        x_line = np.linspace(min(x), max(x), 100)
        y_line = model(x_line)
        return x_line, y_line, model
    
    # Fit models of different complexity
    x_line_simple, y_line_simple, model_simple = fit_polynomial(x, y, 1)    # Underfitting
    x_line_good, y_line_good, model_good = fit_polynomial(x, y, 3)         # Good fit
    x_line_complex, y_line_complex, model_complex = fit_polynomial(x, y, 15)  # Overfitting
    
    # Calculate train and test errors
    # Split data for demonstration
    x_train, x_test = x[:20], x[20:]
    y_train, y_test = y[:20], y[20:]
    
    models = {
        'Underfit (Linear)': model_simple,
        'Good Fit': model_good,
        'Overfit': model_complex
    }
    
    train_errors = {}
    test_errors = {}
    
    for name, model in models.items():
        train_pred = model(x_train)
        test_pred = model(x_test)
        train_mse = np.mean((train_pred - y_train) ** 2)
        test_mse = np.mean((test_pred - y_test) ** 2)
        train_errors[name] = train_mse
        test_errors[name] = test_mse
    
    # Visualization
    st.markdown("""
    <div class="card">
        <h3>Overfitting</h3>
        <p><b>Overfitting</b> occurs when a model learns the training data too well, capturing noise rather than just the underlying pattern.</p>
        <p>When a model overfits:</p>
        <ul>
            <li>It performs very well on training data</li>
            <li>It performs poorly on new, unseen data</li>
            <li>It has "memorized" the training examples instead of learning generalizable patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create plots
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x_train, y=y_train,
        mode='markers',
        name='Training Data',
        marker=dict(color=AWS_COLORS['info'], size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_test, y=y_test,
        mode='markers',
        name='Testing Data',
        marker=dict(color=AWS_COLORS['success'], size=10)
    ))
    
    # Add model lines
    fig.add_trace(go.Scatter(
        x=x_line_simple, y=y_line_simple,
        mode='lines',
        name='Underfitting (Linear)',
        line=dict(color=AWS_COLORS['primary'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line_good, y=y_line_good,
        mode='lines',
        name='Good Fit',
        line=dict(color='purple', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line_complex, y=y_line_complex,
        mode='lines',
        name='Overfitting',
        line=dict(color=AWS_COLORS['secondary'], width=2)
    ))
    
    fig.update_layout(
        title='Demonstration of Overfitting',
        xaxis_title='x',
        yaxis_title='y',
        legend_title='Legend',
        height=500,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show errors comparison
    st.subheader("Model Performance Comparison")
    
    # Create DataFrame for errors
    error_df = pd.DataFrame({
        'Model': list(train_errors.keys()),
        'Training Error': list(train_errors.values()),
        'Testing Error': list(test_errors.values())
    })
    
    # Plot errors
    fig = go.Figure(data=[
        go.Bar(name='Training Error', x=error_df['Model'], y=error_df['Training Error'], marker_color=AWS_COLORS['info']),
        go.Bar(name='Testing Error', x=error_df['Model'], y=error_df['Testing Error'], marker_color=AWS_COLORS['error'])
    ])
    
    fig.update_layout(
        title='Training vs. Testing Error',
        xaxis_title='Model',
        yaxis_title='Mean Squared Error',
        barmode='group',
        height=400,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div class="card">
        <h3>Key Observations</h3>
        <div style="border-left: 4px solid {AWS_COLORS['primary']}; padding-left: 10px; margin-bottom: 15px;">
            <p><b>1. The Underfit model</b> (blue line) is too simple and misses the pattern in both training and testing data.</p>
            <ul>
                <li>Training Error: {train_errors['Underfit (Linear)']:.4f}</li>
                <li>Testing Error: {test_errors['Underfit (Linear)']:.4f}</li>
            </ul>
        </div>
        <div style="border-left: 4px solid purple; padding-left: 10px; margin-bottom: 15px;">
            <p><b>2. The Good Fit model</b> (purple line) captures the general pattern without following noise.</p>
            <ul>
                <li>Training Error: {train_errors['Good Fit']:.4f}</li>
                <li>Testing Error: {test_errors['Good Fit']:.4f}</li>
            </ul>
        </div> 
        <div style="border-left: 4px solid {AWS_COLORS['secondary']}; padding-left: 10px; margin-bottom: 15px;">
            <p><b>3. The Overfit model</b> (orange line) perfectly follows the training data, including noise.</p>
            <ul>
                <li>Training Error: {train_errors['Overfit']:.4f}</li>
                <li>Testing Error: {test_errors['Overfit']:.4f}</li>
            </ul>
        </div>
        <p>Notice how the overfit model has the lowest training error but highest testing error - this is the hallmark of overfitting.</p>
    </div>
    """, unsafe_allow_html=True)

def render_underfitting_visualization():
    """Render underfitting visualization"""
    st.subheader("Visual Explanation: Underfitting")
    
    # Generate a synthetic dataset to demonstrate underfitting
    np.random.seed(0)
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    y = 0.5 * x**2 + np.random.normal(0, 5, n_samples)  # Quadratic relationship with noise
    
    # Create DataFrame
    df_underfit = pd.DataFrame({'x': x, 'y': y})
    
    # Function to fit polynomial models of different degrees
    def fit_polynomial(x, y, degree):
        model = np.poly1d(np.polyfit(x, y, degree))
        x_line = np.linspace(min(x), max(x), 100)
        y_line = model(x_line)
        return x_line, y_line
    
    # Fit models of different complexity
    x_line_simple, y_line_simple = fit_polynomial(x, y, 1)    # Underfitting
    x_line_good, y_line_good = fit_polynomial(x, y, 2)        # Good fit
    x_line_complex, y_line_complex = fit_polynomial(x, y, 10)  # Overfitting
    
    # Visualization
    st.markdown("""
    <div class="card">
        <h3>Underfitting</h3>
        <p><b>Underfitting</b> occurs when a model is too simple to capture the underlying pattern in the data.</p>
        <p>When a model underfits:</p>
        <ul>
            <li>It performs poorly on both training and testing data</li>
            <li>It fails to capture important relationships in the data</li>
            <li>It has high bias (preconceived notion about the data)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create plots
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Data Points',
        marker=dict(color=AWS_COLORS['light'], size=8, opacity=0.6, line=dict(color=AWS_COLORS['primary'], width=1))
    ))
    
    # Add model lines
    fig.add_trace(go.Scatter(
        x=x_line_simple, y=y_line_simple,
        mode='lines',
        name='Underfit (Linear)',
        line=dict(color=AWS_COLORS['error'], width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line_good, y=y_line_good,
        mode='lines',
        name='Good Fit (Quadratic)',
        line=dict(color=AWS_COLORS['success'], width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line_complex, y=y_line_complex,
        mode='lines',
        name='Overfit (Degree 10)',
        line=dict(color=AWS_COLORS['secondary'], width=3)
    ))
    
    fig.update_layout(
        title='Demonstration of Underfitting vs. Good Fit vs. Overfitting',
        xaxis_title='x',
        yaxis_title='y',
        legend_title='Models',
        height=500,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="card">
        <h3>Balancing Complexity: The Bias-Variance Tradeoff</h3>
        <p>Machine learning involves finding the right balance between:</p>
        <div style="border-left: 4px solid {AWS_COLORS['error']}; padding-left: 10px; margin-bottom: 15px;">
            <p><b>Bias (Underfitting)</b>: When a model makes strong assumptions and is too simple</p>
        </div>
        <div style="border-left: 4px solid {AWS_COLORS['secondary']}; padding-left: 10px; margin-bottom: 15px;">
            <p><b>Variance (Overfitting)</b>: When a model is too complex and captures noise</p>
        </div>
        <p>The ideal model should:</p>
        <ul>
            <li>Be complex enough to capture the underlying patterns</li>
            <li>Be simple enough to generalize well to new data</li>
            <li>Ignore the noise in the training data</li>
        </ul>
        <p>Techniques to prevent underfitting:</p>
        <ul>
            <li>Use more complex models</li>
            <li>Add more relevant features</li>
            <li>Reduce regularization</li>
            <li>Try different algorithms</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a visual to explain bias-variance tradeoff
    bias_variance_img = "https://miro.medium.com/max/1400/1*WQXwEzJJHK7Q6RQFnQMJGw.png"
    st.image(bias_variance_img, caption="The Bias-Variance Tradeoff", width=700)

# Render quiz questions
def render_quiz(quiz_questions):
    """Display quiz questions and handle responses"""
    
    # Initialize quiz state
    init_quiz_state()
    
    current_q = st.session_state.current_question
    
    # Reset quiz button
    col_reset, col_progress = st.columns([1, 3])
    with col_reset:
        if st.button("🔄 Reset Quiz"):
            reset_quiz_progress()
            st.rerun()
    
    with col_progress:
        # Show question counter
        st.markdown(f"""
        <div style="text-align: right; color: {AWS_COLORS['primary']};">
            Question {current_q + 1} of {len(quiz_questions)}
        </div>
        """, unsafe_allow_html=True)
    
        # Progress bar
        progress_value = (current_q) / (len(quiz_questions) - 1)
        st.progress(progress_value)
    
    # Display current question
    question = quiz_questions[current_q]
    st.markdown(f"""
    <div class="card">
        <h3>{question['question']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Track if an answer has been provided for this question
    answer_provided = False
    is_correct = False
    response_value = None
    
    # Check if we already have a response for this question
    question_key = f"q{current_q}"
    has_previous_response = question_key in st.session_state.quiz_responses
    question_submitted = current_q in st.session_state.question_submitted
    
    # Handle different question types
    if question["type"] == "radio":
        # Set default index
        default_index = None
        if has_previous_response:
            default_index = st.session_state.quiz_responses[question_key]["response"]
        
        # Create a key specific to this question
        radio_key = f"radio_q{current_q}"
        
        # Get or initialize the session state for this radio button
        if radio_key not in st.session_state:
            st.session_state[radio_key] = default_index
        
        answer_index = st.radio(
            "Select one answer:",
            options=range(len(question["options"])),
            format_func=lambda i: question["options"][i],
            key=radio_key,
            index=st.session_state[radio_key]
        )
        
        # Check if an answer was selected
        if answer_index is not None:
            answer_provided = True
            option_index = answer_index
            response_value = option_index
            is_correct = option_index == question["correct_answer"]
            
    elif question["type"] == "checkbox":
        # Create a key specific to this question for each checkbox
        selected_indices = []
        
        # If we already have responses, use them to set the defaults
        default_values = []
        if has_previous_response:
            default_values = st.session_state.quiz_responses[question_key]["response"]
        
        # For each option, create a checkbox
        for i, option in enumerate(question["options"]):
            checkbox_key = f"cb_q{current_q}_{i}"
            
            # Initialize session state for this checkbox if needed
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = i in default_values
            
            # Display the checkbox
            if st.checkbox(option, key=checkbox_key, value=st.session_state[checkbox_key]):
                selected_indices.append(i)
        
        # Check if any option was selected
        if selected_indices:
            answer_provided = True
            response_value = selected_indices
            is_correct = set(selected_indices) == set(question["correct_answer"])
    
    # Submit button for the current question
    submit_button_key = f"submit_q{current_q}"
    submit_answer = st.button("Submit Answer", key=submit_button_key, disabled=question_submitted)
    
    # Store the response if an answer was provided and submitted
    if answer_provided and (submit_answer or question_submitted):
        # If just submitted now, update the state
        if submit_answer:
            submit_question_answer(current_q)
            
        # Store the response
        st.session_state.quiz_responses[question_key] = {
            "response": response_value,
            "correct": is_correct
        }
        
        # Display feedback only after submission
        if question_submitted:
            if is_correct:
                st.success("✅ Correct!")
            else:
                st.error("❌ Incorrect.")
                if question["type"] == "radio":
                    st.info(f"Correct Answer: {question['options'][question['correct_answer']]}")
                else:  # checkbox
                    correct_options = [question["options"][i] for i in question["correct_answer"]]
                    st.info(f"Correct Answer(s): {', '.join(correct_options)}")
            
            st.info(f"Explanation: {question['explanation']}")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if current_q > 0:
            if st.button("⬅️ Previous Question"):
                prev_question()
                st.rerun()
    
    with col3:
        if current_q < len(quiz_questions) - 1:
            if st.button("Next Question ➡️"):
                next_question()
                st.rerun()
        else:
            # Only show submit quiz button if all questions have been attempted
            if len(st.session_state.questions_attempted) == len(quiz_questions):
                if st.button("📝 Submit Quiz"):
                    submit_quiz()
                    st.rerun()
            else:
                st.warning("Please attempt all questions before submitting.")
    
    # Show quiz progress
    st.divider()
    st.subheader("Quiz Progress")
    
    total_questions = len(quiz_questions)
    attempted_questions = len(st.session_state.questions_attempted)
    correct_answers = sum(1 for resp in st.session_state.quiz_responses.values() if resp["correct"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{attempted_questions}/{total_questions}</p>
            <p class="metric-label">Questions Attempted</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{correct_answers}</p>
            <p class="metric-label">Correct Answers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if attempted_questions > 0:
            accuracy = (correct_answers / attempted_questions) * 100
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">{accuracy:.1f}%</p>
                <p class="metric-label">Current Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">-</p>
                <p class="metric-label">Current Accuracy</p>
            </div>
            """, unsafe_allow_html=True)

# Display quiz results
def show_quiz_results():
    """Show quiz results after submission"""
    score = st.session_state.quiz_score
    total = len(get_quiz_questions())
    percentage = (score / total) * 100
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2>Quiz Results</h2>
        <div class="metric-container">
            <p class="metric-value">{score}/{total} ({percentage:.0f}%)</p>
            <p class="metric-label">Correct Answers</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display a message based on score
    if percentage >= 80:
        st.success("🎉 Great job! You have a strong understanding of machine learning terminology.")
    elif percentage >= 60:
        st.info("👍 Good effort! You're on your way to mastering machine learning concepts.")
    else:
        st.warning("📚 Keep learning! Review the terms in the Explorer tab for better understanding.")
    
    # Show detailed question breakdown
    st.subheader("Detailed Question Breakdown")
    quiz_questions = get_quiz_questions()
    
    for i, question in enumerate(quiz_questions):
        question_key = f"q{i}"
        if question_key in st.session_state.quiz_responses:
            response = st.session_state.quiz_responses[question_key]
            is_correct = response["correct"]
            color = AWS_COLORS['success'] if is_correct else AWS_COLORS['error']
            icon = "✅" if is_correct else "❌"
            
            with st.expander(f"Question {i+1}: {icon} {question['question'][:80]}..."):
                st.markdown(f"**Question:** {question['question']}")
                
                if question["type"] == "radio":
                    user_answer = question["options"][response["response"]]
                    correct_answer = question["options"][question["correct_answer"]]
                    
                    st.markdown(f"**Your answer:** {user_answer}")
                    st.markdown(f"**Correct answer:** {correct_answer}")
                
                elif question["type"] == "checkbox":
                    user_selections = [question["options"][idx] for idx in response["response"]]
                    correct_selections = [question["options"][idx] for idx in question["correct_answer"]]
                    
                    st.markdown(f"**You selected:** {', '.join(user_selections)}")
                    st.markdown(f"**Correct selections:** {', '.join(correct_selections)}")
                
                st.markdown(f"**Explanation:** {question['explanation']}")
    
    # Option to retake quiz
    if st.button("🔄 Retake Quiz"):
        reset_quiz_progress()
        st.rerun()


# Custom house price prediction example
def render_house_price_prediction(california_df):
    """Render the house price prediction example"""
    st.markdown("""
    <div class="card">
        <h3>Apply ML Terminology to House Price Prediction</h3>
        <p>Now that you understand the key terminology, let's apply it to a house price prediction example using the California Housing dataset.
        Adjust the features below and see how they affect the predicted house value.</p>
    </div>
    """, unsafe_allow_html=True)

    # Get California Housing dataset
    X = california_df.drop('target', axis=1)
    y = california_df['target']

    # Train a model for demonstration
    model = LinearRegression().fit(X, y)

    # Create input form with California Housing features
    col1, col2 = st.columns(2)

    with col1:
        median_income = st.slider("Median Income (tens of thousands)", 
                               float(X['MedInc'].min()), 
                               float(X['MedInc'].max()), 
                               3.5)
        
        house_age = st.slider("Housing Median Age", 
                           float(X['HouseAge'].min()), 
                           float(X['HouseAge'].max()), 
                           28.0)
        
        avg_rooms = st.slider("Average Rooms per Household", 
                           float(X['AveRooms'].min()), 
                           float(X['AveRooms'].max()), 
                           5.0)
        
        avg_bedrooms = st.slider("Average Bedrooms per Household", 
                              float(X['AveBedrms'].min()), 
                              float(X['AveBedrms'].max()), 
                              1.0)

    with col2:
        population = st.slider("Block Population", 
                            float(X['Population'].min()), 
                            float(X['Population'].max()), 
                            1500.0)
        
        avg_occupancy = st.slider("Average Household Occupancy", 
                               float(X['AveOccup'].min()), 
                               float(X['AveOccup'].max()), 
                               3.0)
        
        latitude = st.slider("Latitude", 
                          float(X['Latitude'].min()), 
                          float(X['Latitude'].max()), 
                          35.0)
        
        longitude = st.slider("Longitude", 
                           float(X['Longitude'].min()), 
                           float(X['Longitude'].max()), 
                           -120.0)

    # Create new observation
    new_district = pd.DataFrame({
        'MedInc': [median_income],
        'HouseAge': [house_age],
        'AveRooms': [avg_rooms],
        'AveBedrms': [avg_bedrooms],
        'Population': [population],
        'AveOccup': [avg_occupancy],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    # Make prediction
    predicted_value = model.predict(new_district)[0]

    # Display results with ML terminology
    st.subheader("ML Terminology Applied to This Example")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>Dataset</h4>
            <p>The California Housing dataset with information on housing districts</p> 
            <h4>Features (X)</h4>
            <ul>
                <li>Median Income: ${median_income * 10000:.2f}</li>
                <li>House Age: {house_age:.1f} years</li>
                <li>Average Rooms: {avg_rooms:.2f} per household</li>
                <li>Average Bedrooms: {avg_bedrooms:.2f} per household</li>
                <li>Population: {population:.0f}</li>
                <li>Average Occupancy: {avg_occupancy:.2f} persons per household</li>
                <li>Location: ({latitude:.2f}, {longitude:.2f})</li>
            </ul>
            <h4>Observation</h4>
            <p>This single housing district we're analyzing</p>
            <h4>Model</h4>
            <p>Linear Regression that learned patterns from housing data</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h4>Target (y)</h4>
            <p>Median House Value</p>
            <h4>Prediction</h4>
            <p style="font-size: 20px; font-weight: bold; color: {AWS_COLORS['secondary']};">${predicted_value * 100000:,.2f}</p>
            <h4>Feature Importance</h4>
            <ul>
                <li>Median Income: ${model.coef_[0] * 100000:.2f} per unit</li>
                <li>House Age: ${model.coef_[1] * 100000:.2f} per year</li>
                <li>Average Rooms: ${model.coef_[2] * 100000:.2f} per room</li>
                <li>Average Bedrooms: ${model.coef_[3] * 100000:.2f} per bedroom</li>
                <li>Population: ${model.coef_[4] * 100000:.2f} per person</li>
                <li>Average Occupancy: ${model.coef_[5] * 100000:.2f} per person</li>
                <li>Latitude: ${model.coef_[6] * 100000:.2f} per degree</li>
                <li>Longitude: ${model.coef_[7] * 100000:.2f} per degree</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Add a visual representation of the prediction
    fig = go.Figure()

    # Add base price (intercept)
    fig.add_trace(go.Bar(
        x=['Base Value'],
        y=[model.intercept_ * 100000],  # Convert to dollars
        name='Base Value',
        marker_color=AWS_COLORS['success']
    ))

    # Add feature contributions
    features = X.columns
    contributions = [model.coef_[i] * new_district[feature].values[0] * 100000 for i, feature in enumerate(features)]
    colors = [AWS_COLORS['info'], AWS_COLORS['secondary'], AWS_COLORS['tertiary'], AWS_COLORS['error'], 
            AWS_COLORS['success'], AWS_COLORS['warning'], AWS_COLORS['primary'], AWS_COLORS['light']]

    for i, feature in enumerate(features):
        fig.add_trace(go.Bar(
            x=[feature],
            y=[contributions[i]],
            name=feature,
            marker_color=colors[i % len(colors)]
        ))

    # Add total prediction
    fig.add_trace(go.Bar(
        x=['Predicted Value'],
        y=[predicted_value * 100000],  # Convert to dollars
        name='Total',
        marker_color=AWS_COLORS['secondary']
    ))

    fig.update_layout(
        title='House Value Prediction Breakdown',
        xaxis_title='Components',
        yaxis_title='Value ($)',
        barmode='group',
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Additional explanation
    st.markdown("""
    <div class="card">
        <p>This example illustrates how a machine learning model uses features (inputs) to make predictions about a target variable (output).
        The model has learned patterns from many examples in the California Housing dataset to understand how features like median income
        and house age affect median house values in different districts.</p>
        <p>Try adjusting the sliders to see how different feature values affect the predicted house price!</p>
    </div>
    """, unsafe_allow_html=True)

# Glossary section
def render_glossary():
    """Render glossary of ML terms"""
   
    ml_terminology = get_ml_terminology()
    
    glossary_data = []
    for term, info in ml_terminology.items():
        glossary_data.append({
            'Term': term,
            'Definition': info['definition']
        })

    glossary_df = pd.DataFrame(glossary_data)
    
    # Search functionality
    search = st.text_input("🔍 Search for a term", "")
    
    if search:
        filtered_df = glossary_df[
            glossary_df['Term'].str.lower().str.contains(search.lower()) | 
            glossary_df['Definition'].str.lower().str.contains(search.lower())
        ]
        if not filtered_df.empty:
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.info("No matching terms found.")
    else:
        st.dataframe(glossary_df, use_container_width=True, hide_index=True)

# Reset session state
def reset_session():
    """Reset all session state variables"""
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# Sidebar function
def render_sidebar():
    """Render the sidebar with dataset selection and session management"""
    
    # Dataset selection for the Explorer tab
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = 'California Housing'
    
    if 'selected_term' not in st.session_state:
        st.session_state.selected_term = 'Dataset'
    
    # Session management
    st.sidebar.divider()
    st.sidebar.markdown("### Session Management")
    if st.sidebar.button("🔄 Reset Session"):
        reset_session()
    
    st.sidebar.divider()
    with st.sidebar.expander("📚 About this App", expanded=False):
        # About section
        st.markdown("""
        This app helps you learn machine learning terminology through interactive visualizations and examples.
        
        """)

# Main app
def main():
    """Main function to run the Streamlit app"""
    # Page config
    st.set_page_config(
        page_title="ML Terminology",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Load CSS
    load_css()
    
    # Load datasets
    datasets = load_datasets()
    
    # Sidebar
    render_sidebar()
    
    st.title("📊 Machine Learning Terminology")
    st.markdown('Learners will be able to accurately identify, define, and apply fundamental machine learning terminology by interacting with visualizations, analyzing real datasets, and demonstrating knowledge through practical examples and assessments.')
    
    # Main content with tabs
    tabs = st.tabs([
        "🏠 Home",
        "🔍 ML Terminology",
        "🧠 Knowledge Check",
        "🧩 Interactive Example",
        "📚 Glossary"
    ])
    
    # Home tab
    with tabs[0]:       
        st.markdown("""
        <div class="card">
            <h2>Welcome to the ML Terminology Explorer!</h2>
            <p>This interactive application helps you understand key machine learning concepts through:</p>
            <ul>
                <li>Visual explanations of terminology</li>
                <li>Interactive examples with real datasets</li>
                <li>Knowledge checks to test your understanding</li>
            </ul>
            <p>Use the tabs above to navigate through the different sections of the app.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>🔍 Explorer</h3>
                <p>Explore key machine learning terms with interactive visualizations.</p>
                <ul>
                    <li>Select a term from the dropdown</li>
                    <li>See its definition and example</li>
                    <li>Interact with dynamic visualizations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
                <h3>🧠 Knowledge Check</h3>
                <p>Test your understanding of ML concepts with quiz questions.</p>
                <ul>
                    <li>Answer multiple-choice questions</li>
                    <li>Get immediate feedback</li>
                    <li>See detailed explanations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="card">
                <h3>🧩 Interactive Example</h3>
                <p>Apply ML terminology to a real house price prediction example.</p>
                <ul>
                    <li>Adjust input features</li>
                    <li>See how predictions change</li>
                    <li>Understand model behavior</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Featured terms
        st.subheader("Featured Terminology")
        
        featured_terms = ["Dataset", "Features", "Model", "Overfitting"]
        featured_cols = st.columns(len(featured_terms))
        
        ml_terminology = get_ml_terminology()
        
        for i, term in enumerate(featured_terms):
            with featured_cols[i]:
                st.markdown(f"""
                <div class="card">
                    <h4>{term}</h4>
                    <p>{ml_terminology[term]['definition']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Explorer tab
    with tabs[1]:
        st.header("🔍 ML Terminology Explorer")
        
        # Select dataset and term
        col1, col2 = st.columns(2)
        with col1:
            selected_dataset = st.selectbox("Select a dataset:", list(datasets.keys()), index=1)
            st.session_state.selected_dataset = selected_dataset
        with col2:
            ml_terminology = get_ml_terminology()
            selected_term = st.selectbox("Select a term to explore:", list(ml_terminology.keys()), index=0)
            st.session_state.selected_term = selected_term
        
        current_df = datasets[selected_dataset]
        
        # Display sample data
        with st.expander("Sample Data", expanded=True):
            st.dataframe(current_df.head(5), use_container_width=True)
        
        # Display term info
        st.header(selected_term)
        term_info = ml_terminology[selected_term]
        
        st.markdown(f"""
        <div class="card">
            <h3>Definition</h3>
            <p>{term_info['definition']}</p>
            <h3>Example</h3>
            <p>{term_info['example']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Render visualization based on term
        render_term_visualization(term_info, selected_term, current_df)
    
    # Knowledge Check tab
    with tabs[2]:
        st.header("🧠 Knowledge Check")
        
        st.markdown("""
        <div class="card">
            <p>Test your understanding of machine learning terminology with these questions.
            Select the correct answer for each question to see explanations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        quiz_questions = get_quiz_questions()
        
        if st.session_state.quiz_submitted:
            show_quiz_results()
        else:
            render_quiz(quiz_questions)
    
    # Interactive Example tab
    with tabs[3]:
        st.header("🧩 Interactive Example")
        california_df = datasets['California Housing']
        render_house_price_prediction(california_df)
    
    # Glossary tab
    with tabs[4]:
        st.header("📚 Machine Learning Terminology Glossary")
        render_glossary()
    
    # Footer
    st.markdown("""
    <div class="footer">
        © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

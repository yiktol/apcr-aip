import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import time
import uuid
import warnings
from datetime import datetime
from utils.common import render_sidebar
from utils.styles import load_css
import utils.authenticate as authenticate
# Suppress warnings
warnings.filterwarnings("ignore")

# AWS color scheme
AWS_COLORS = {
    'primary': '#FF9900',      # AWS Orange
    'secondary': '#232F3E',    # AWS Navy
    'tertiary': '#1A476F',     # AWS Blue
    'background': '#FFFFFF',   # White
    'text': '#16191F',         # Dark gray
    'success': '#008296',      # Teal
    'warning': '#D13212',      # Red
    'info': '#1E88E5'          # Blue
}

# ----- Session Management -----

def initialize_session_state():
    """Initialize the session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())
    
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = datetime.now()
        
    # Initialize model results if needed
    if 'heart_results' not in st.session_state:
        st.session_state['heart_results'] = None
        
    if 'diabetes_results' not in st.session_state:
        st.session_state['diabetes_results'] = None

def reset_session():
    """Reset all session state variables"""
    for key in st.session_state.keys():
        if key not in ["authenticated", "user_cognito_groups", "auth_code","user_info"]:
            del st.session_state[key]
    
    # Reinitialize essential variables
    st.session_state['start_time'] = datetime.now()
    st.session_state['heart_results'] = None
    st.session_state['diabetes_results'] = None
    st.success("Session has been reset!")

# ----- Data Loading and Processing Functions -----

@st.cache_data
def load_heart_disease_data():
    """
    Load the heart disease dataset from UCI repository
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        
        # Clean the data
        df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
        df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
        
        # Replace '?' with NaN and handle missing values
        df = df.replace('?', np.nan)
        
        # Convert target to binary (0 = no disease, 1 = disease)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a mock dataset if loading fails
        return create_mock_heart_disease_data()

# @st.cache_data
def load_diabetes_data():
    """
    Load the Pima Indians Diabetes dataset
    """
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        column_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        df = pd.read_csv(url, names=column_names)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a mock dataset if loading fails
        return create_mock_diabetes_data()

def create_mock_heart_disease_data():
    """
    Create a mock heart disease dataset if the original cannot be loaded
    """
    np.random.seed(42)
    n = 303  # Same size as original dataset
    
    # Generate synthetic data
    age = np.random.randint(25, 80, n)
    sex = np.random.randint(0, 2, n)
    cp = np.random.randint(0, 4, n)  # Chest pain type
    trestbps = np.random.randint(90, 200, n)  # Resting blood pressure
    chol = np.random.randint(120, 400, n)  # Cholesterol
    fbs = np.random.randint(0, 2, n)  # Fasting blood sugar
    restecg = np.random.randint(0, 3, n)  # Rest ECG
    thalach = np.random.randint(80, 220, n)  # Max heart rate
    exang = np.random.randint(0, 2, n)  # Exercise induced angina
    oldpeak = np.round(np.random.uniform(0, 6, n), 1)  # ST depression
    slope = np.random.randint(0, 3, n)  # ST slope
    ca = np.random.randint(0, 4, n)  # Major vessels
    thal = np.random.randint(1, 4, n)  # Thalassemia
    
    # Create target variable with correlation to features
    target = np.zeros(n, dtype=int)
    for i in range(n):
        # Simple model: higher risk with age, chest pain, and cholesterol
        risk = (
            0.01 * age[i] +
            0.5 * cp[i] +
            0.005 * chol[i] -
            0.005 * thalach[i] +
            0.5 * exang[i] +
            oldpeak[i] +
            0.5 * ca[i]
        )
        
        # Apply sigmoid to get probability and convert to binary
        prob = 1 / (1 + np.exp(-risk + 5))  # +5 to center the distribution
        target[i] = 1 if np.random.random() < prob else 0
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal, 'target': target
    })
    
    return df

def create_mock_diabetes_data():
    """
    Create a mock diabetes dataset if the original cannot be loaded
    """
    np.random.seed(42)
    n = 768  # Same size as original dataset
    
    # Generate synthetic data
    pregnancies = np.random.randint(0, 18, n)
    glucose = np.random.randint(50, 200, n)
    blood_pressure = np.random.randint(30, 120, n)
    skin_thickness = np.random.randint(5, 60, n)
    insulin = np.random.randint(10, 400, n)
    bmi = np.round(np.random.uniform(15, 50, n), 1)
    diabetes_pedigree = np.round(np.random.uniform(0.05, 2.5, n), 3)
    age = np.random.randint(21, 81, n)
    
    # Create target variable with correlation to features
    outcome = np.zeros(n, dtype=int)
    for i in range(n):
        # Simple model: higher risk with glucose, BMI, and age
        risk = (
            0.01 * glucose[i] +
            0.05 * bmi[i] +
            0.01 * age[i] +
            0.2 * diabetes_pedigree[i]
        )
        
        # Apply sigmoid to get probability and convert to binary
        prob = 1 / (1 + np.exp(-risk + 5))  # +5 to center the distribution
        outcome[i] = 1 if np.random.random() < prob else 0
    
    # Create DataFrame
    df = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age,
        'Outcome': outcome
    })
    
    return df

@st.cache_data
def get_heart_disease_description():
    """Return information about heart disease features"""
    return {
        "age": "Age in years",
        "sex": "Sex (1 = male; 0 = female)",
        "cp": "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure in mm Hg",
        "chol": "Serum cholesterol in mg/dl",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
        "restecg": "Resting ECG results (0-2)",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise induced angina (1 = yes; 0 = no)",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "Slope of the peak exercise ST segment (0-2)",
        "ca": "Number of major vessels (0-3) colored by fluoroscopy",
        "thal": "Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)",
        "target": "Heart disease (1 = present; 0 = absent)"
    }

@st.cache_data
def get_diabetes_description():
    """Return information about diabetes features"""
    return {
        "Pregnancies": "Number of times pregnant",
        "Glucose": "Plasma glucose concentration (mg/dL)",
        "BloodPressure": "Diastolic blood pressure (mm Hg)",
        "SkinThickness": "Triceps skin fold thickness (mm)",
        "Insulin": "2-Hour serum insulin (mu U/ml)",
        "BMI": "Body mass index (weight in kg/(height in m)²)",
        "DiabetesPedigreeFunction": "Diabetes pedigree function (genetic score)",
        "Age": "Age in years",
        "Outcome": "Class variable (0 = no diabetes, 1 = diabetes)"
    }

def preprocess_data(df, dataset_type):
    """
    Preprocess the data for modeling
    """
    # Handle missing values first
    imputer = SimpleImputer(strategy='median')
    
    if dataset_type == 'heart_disease':
        # Split features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Impute missing values
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
    else:  # diabetes
        # Split features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Replace zeros with NaN for certain features
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            X[col] = X[col].replace(0, np.nan)
        
        # Impute missing values
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X, X_scaled, y, X_train, X_test, y_train, y_test, scaler

@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    """
    Train various machine learning models and return their performances
    """
    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # Train each model and collect results
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    return results

def get_feature_importance(model_name, model, X):
    """
    Get feature importance based on the model type
    """
    feature_importance = None
    
    if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        # Tree-based models have feature_importance_ attribute
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    elif model_name == "Logistic Regression":
        # Linear models have coef_ attribute
        if hasattr(model, 'coef_'):
            feature_importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
    elif model_name == "SVM" and hasattr(model, 'coef_'):
        # Linear SVM has coef_
        feature_importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
    
    return feature_importance

def calculate_shap_values(model, X_sample, model_name):
    """
    Calculate SHAP values for model interpretation
    """
    try:
        # For different model types, use appropriate explainer
        if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take the positive class
        elif model_name in ["Logistic Regression", "SVM"]:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        else:
            # For other models, use KernelExplainer but limit to fewer samples for performance
            background = shap.kmeans(X_sample, 10)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample[:50])[1]
        
        return shap_values, explainer
    except Exception as e:
        st.warning(f"SHAP calculation failed: {e}. Using default feature importance instead.")
        return None, None

def get_normal_ranges(dataset_type):
    """
    Return normal ranges for health metrics
    """
    if dataset_type == 'heart_disease':
        return {
            'age': (18, 100),
            'sex': (0, 1),
            'cp': (0, 3),
            'trestbps': (90, 120),  # Normal resting blood pressure
            'chol': (125, 200),     # Normal cholesterol
            'fbs': (0, 1),
            'restecg': (0, 2),
            'thalach': (85, 185),   # Normal max heart rate
            'exang': (0, 1),
            'oldpeak': (0, 4),
            'slope': (0, 2),
            'ca': (0, 3),
            'thal': (1, 3)
        }
    else:  # diabetes
        return {
            'Pregnancies': (0, 20),
            'Glucose': (70, 99),      # Normal fasting glucose
            'BloodPressure': (60, 80), # Normal diastolic BP
            'SkinThickness': (10, 40), # Normal triceps skin fold
            'Insulin': (16, 166),      # Normal insulin levels
            'BMI': (18.5, 24.9),       # Normal BMI
            'DiabetesPedigreeFunction': (0.1, 2.5),
            'Age': (18, 100)
        }

# ----- UI Component Functions -----

def create_header():
    """Create the application header with AWS styling"""
    st.markdown(
        f"""
        <style>
        .header-container {{
            background-color: {AWS_COLORS['secondary']};
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }}
        .header-title {{
            color: {AWS_COLORS['primary']};
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        .header-subtitle {{
            color: white;
            font-size: 1.2rem;
            font-style: italic;
        }}
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 16px;
            background-color: #f8f9fa;
            border-radius: 4px 4px 0 0;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {AWS_COLORS['primary']} !important;
            color: white !important;
        }}        
        
        </style>

        """, 
        unsafe_allow_html=True
    )

def create_footer():
    """Create the application footer with AWS copyright"""
    st.markdown(
        f"""
        <style>
        .footer-container {{
            background-color: {AWS_COLORS['secondary']};
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-top: 2rem;
            text-align: center;
        }}
        .footer-text {{
            color: white;
            font-size: 0.8rem;
        }}
        </style>
        <div class="footer-container">
            <div class="footer-text">© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

def setup_sidebar():
    """Configure the sidebar with navigation and controls"""
    with st.sidebar:
        
        # st.header("Navigation")
        
        # Navigation radio buttons with emojis
        # app_mode = st.radio("Go to", 
        #     ["🏠 Introduction", 
        #      "❤️ Heart Disease Prediction", 
        #      "🩸 Diabetes Prediction", 
        #      "📊 About ML in Healthcare"])
        
        # # Strip the emojis for internal logic
        # clean_mode = app_mode.split(" ", 1)[1] if " " in app_mode else app_mode
        
        # Session info
        render_sidebar()
        
        # About this App section (collapsed by default)
        with st.expander("About this App"):
            st.markdown("""
            ### Healthcare ML Diagnostics
            
            This application demonstrates the power of machine learning in healthcare diagnostics.
            
            **Topics covered:**
            - Heart disease risk prediction
            - Diabetes risk assessment
            - ML model evaluation and interpretation
            - Healthcare data visualization
            - SHAP value analysis for model interpretability
            """)
        
        
        
    # return clean_mode

# ----- Disease Prediction Page Functions -----

def show_heart_disease_prediction(results, X, scaler, feature_desc):
    """Display the heart disease prediction interface"""
    st.subheader("📋 Patient Data Input")
    st.write("Enter patient information to predict heart disease risk")
    
    # Select best model (by AUC)
    best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
    
    # Option to choose model
    model_name = st.selectbox("Select model for prediction", 
                            list(results.keys()), 
                            index=list(results.keys()).index(best_model_name))
    
    model = results[model_name]['model']
    
    # Get normal ranges
    normal_ranges = get_normal_ranges('heart_disease')
    
    # Create form for input
    with st.form(key="heart_disease_form"):
        st.markdown(f"""
        <style>
        .form-header {{
            background-color: {AWS_COLORS['tertiary']};
            padding: 0.5rem;
            border-radius: 0.3rem;
            color: white;
            margin-bottom: 0.5rem;
        }}
        </style>
        <div class="form-header">Patient Information</div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 
                                min_value=18, max_value=100, 
                                value=50, 
                                help=feature_desc['age'])
            
            sex = st.radio("Sex", 
                        [0, 1], 
                        format_func=lambda x: "Female" if x == 0 else "Male",
                        horizontal=True,
                        help=feature_desc['sex'])
            
            cp = st.selectbox("Chest Pain Type", 
                           [0, 1, 2, 3], 
                           format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                                "Non-anginal Pain", "Asymptomatic"][x],
                           help=feature_desc['cp'])
            
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                    min_value=80, max_value=220, 
                                    value=120,
                                    help=feature_desc['trestbps'])
            
            chol = st.number_input("Serum Cholesterol (mg/dl)", 
                                min_value=100, max_value=600, 
                                value=200,
                                help=feature_desc['chol'])
        
        with col2:
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", 
                        [0, 1], 
                        format_func=lambda x: "No" if x == 0 else "Yes",
                        horizontal=True,
                        help=feature_desc['fbs'])
            
            restecg = st.selectbox("Resting ECG Results", 
                                [0, 1, 2], 
                                format_func=lambda x: ["Normal", "ST-T Wave Abnormality", 
                                                     "Left Ventricular Hypertrophy"][x],
                                help=feature_desc['restecg'])
            
            thalach = st.number_input("Maximum Heart Rate", 
                                    min_value=60, max_value=220, 
                                    value=150,
                                    help=feature_desc['thalach'])
            
            exang = st.radio("Exercise-Induced Angina", 
                           [0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes",
                           horizontal=True,
                           help=feature_desc['exang'])
            
            oldpeak = st.number_input("ST Depression Induced by Exercise", 
                                    min_value=0.0, max_value=10.0, 
                                    value=1.0, step=0.1,
                                    help=feature_desc['oldpeak'])
        
        with col3:
            slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                              [0, 1, 2], 
                              format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                              help=feature_desc['slope'])
            
            ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", 
                           [0, 1, 2, 3],
                           help=feature_desc['ca'])
            
            thal = st.selectbox("Thalassemia", 
                             [1, 2, 3], 
                             format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1],
                             help=feature_desc['thal'])
            
        submitted = st.form_submit_button(label="Predict Heart Disease Risk", 
                                         use_container_width=True,
                                         type="primary")
    
    # When form is submitted
    if submitted:
        # Create input array for prediction
        input_data = np.array([
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
            exang, oldpeak, slope, ca, thal
        ]).reshape(1, -1)
        
        # Create a DataFrame for visualization
        input_df = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
            'ca': ca, 'thal': thal
        }])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_scaled)[0, 1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Display prediction with AWS styling
        st.markdown(f"""
        <style>
        .prediction-header {{
            background-color: {AWS_COLORS['secondary']};
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            font-size: 1.5rem;
            text-align: center;
        }}
        </style>
        <div class="prediction-header">Prediction Result</div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.error("⚠️ High Risk of Heart Disease")
                risk_level = "High Risk"
            else:
                st.success("✅ Low Risk of Heart Disease")
                risk_level = "Low Risk"
            
            risk_percentage = f"{prediction_proba:.1%}"
            st.metric("Risk Probability", risk_percentage)
            st.caption(f"Using {model_name} model")
        
        with col2:
            # Create gauge chart for risk visualization with AWS colors
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Heart Disease Risk", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': AWS_COLORS['tertiary']},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': AWS_COLORS['secondary'],
                    'steps': [
                        {'range': [0, 25], 'color': '#7CC097'},  # Green
                        {'range': [25, 50], 'color': '#F2CD5D'},  # Yellow
                        {'range': [50, 75], 'color': '#F8AA4B'},  # Orange
                        {'range': [75, 100], 'color': '#FB5A5A'},  # Red
                    ],
                    'threshold': {
                        'line': {'color': AWS_COLORS['primary'], 'width': 4},
                        'thickness': 0.75,
                        'value': prediction_proba * 100
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': AWS_COLORS['tertiary'], 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show risk factors
        st.subheader("Patient Risk Factor Analysis")
        
        # Get feature importance for this model
        feature_importance = get_feature_importance(model_name, model, X)
        
        # Highlight abnormal values
        abnormal_features = []
        for feature, value in input_df.iloc[0].items():
            if feature in normal_ranges:
                low, high = normal_ranges[feature]
                if value < low or value > high:
                    abnormal_features.append(feature)
        
        if abnormal_features:
            st.markdown(f"""
            <style>
            .warning-box {{
                background-color: #FFF9E6;
                border-left: 5px solid {AWS_COLORS['primary']};
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 0.3rem;
            }}
            </style>
            <div class="warning-box">
            <strong>⚠️ The following values are outside normal ranges:</strong><br>
            {"<br>".join([f"<b>{f}</b>: {input_df[f].iloc[0]}" for f in abnormal_features])}
            </div>
            """, unsafe_allow_html=True)
        
        # Show top features contributing to prediction
        if feature_importance is not None:
            st.subheader("Top Contributing Factors")
            
            # Sort features by importance
            sorted_importance = feature_importance.sort_values(ascending=False)
            top_features = sorted_importance.index[:5].tolist()  # Top 5 features
            
            # Create contribution chart
            contrib_df = pd.DataFrame({
                'Feature': [feature_desc.get(f, f) for f in top_features],
                'Importance': sorted_importance[:5],
                'Value': [input_df[f].iloc[0] for f in top_features]
            })
            
            fig = px.bar(contrib_df, x='Importance', y='Feature', 
                       text='Value',
                       orientation='h',
                       title="Feature Importance for Heart Disease Risk",
                       color='Importance',
                       color_continuous_scale=['#00A1B4', AWS_COLORS['primary'], '#DD3C51']
                       )
            
            fig.update_layout(
                height=400,
                coloraxis_colorbar=dict(title="Importance"),
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Health recommendations with AWS styling
        st.markdown(f"""
        <style>
        .recommendations {{
            background-color: #F3FFFC;
            border-left: 5px solid {AWS_COLORS['success']};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.3rem;
        }}
        .recommendations h4 {{
            color: {AWS_COLORS['success']};
            margin-top: 0;
        }}
        </style>
        <div class="recommendations">
        <h4>Recommendations</h4>
        """, unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            Based on the high risk assessment:
            
            - **Consult a cardiologist** for comprehensive evaluation
            - **Monitor blood pressure** and heart rate regularly
            - **Consider stress test or ECG** as recommended by your doctor
            - **Review medication** and treatment options
            - **Lifestyle changes** may include:
              - Heart-healthy diet low in saturated fats
              - Regular exercise appropriate for your condition
              - Smoking cessation if applicable
              - Stress reduction techniques
            """)
        else:
            st.markdown("""
            While your current risk appears low:
            
            - **Continue regular check-ups** with your healthcare provider
            - **Maintain heart-healthy habits**:
              - Regular physical activity (aim for 150 minutes/week)
              - Balanced diet rich in fruits, vegetables, and whole grains
              - Limited alcohol intake
              - Adequate sleep (7-8 hours)
            - **Monitor your numbers** (blood pressure, cholesterol) annually
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Disclaimer with AWS styling
        st.markdown(f"""
        <style>
        .disclaimer {{
            background-color: #F0F7FF;
            border: 1px solid {AWS_COLORS['info']};
            padding: 0.7rem;
            margin: 1rem 0;
            border-radius: 0.3rem;
            font-size: 0.9rem;
            text-align: center;
        }}
        </style>
        <div class="disclaimer">
        <strong>Disclaimer:</strong> This prediction is for educational purposes only and should not replace professional medical advice.
        </div>
        """, unsafe_allow_html=True)

def show_diabetes_prediction(results, X, scaler, feature_desc):
    """Display the diabetes prediction interface"""
    st.subheader("📋 Patient Data Input")
    st.write("Enter patient information to predict diabetes risk")
    
    # Select best model (by AUC)
    best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
    
    # Option to choose model
    model_name = st.selectbox("Select model for prediction", 
                            list(results.keys()), 
                            index=list(results.keys()).index(best_model_name))
    
    model = results[model_name]['model']
    
    # Get normal ranges
    normal_ranges = get_normal_ranges('diabetes')
    
    # Create form for input
    with st.form(key="diabetes_form"):
        st.markdown(f"""
        <style>
        .form-header {{
            background-color: {AWS_COLORS['tertiary']};
            padding: 0.5rem;
            border-radius: 0.3rem;
            color: white;
            margin-bottom: 0.5rem;
        }}
        </style>
        <div class="form-header">Patient Information</div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", 
                                        min_value=0, max_value=20, 
                                        value=2, 
                                        help=feature_desc['Pregnancies'])
            
            glucose = st.number_input("Plasma Glucose Concentration (mg/dL)", 
                                    min_value=40, max_value=300, 
                                    value=120,
                                    help=feature_desc['Glucose'])
            
            blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", 
                                          min_value=40, max_value=130, 
                                          value=70,
                                          help=feature_desc['BloodPressure'])
            
            skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", 
                                          min_value=5, max_value=100, 
                                          value=20,
                                          help=feature_desc['SkinThickness'])
        
        with col2:
            insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", 
                                    min_value=0, max_value=700, 
                                    value=80,
                                    help=feature_desc['Insulin'])
            
            bmi = st.number_input("Body Mass Index (kg/m²)", 
                                min_value=15.0, max_value=60.0, 
                                value=25.0, step=0.1,
                                help=feature_desc['BMI'])
            
            dpf = st.number_input("Diabetes Pedigree Function", 
                                min_value=0.05, max_value=2.5, 
                                value=0.5, step=0.01,
                                help=feature_desc['DiabetesPedigreeFunction'])
            
            age = st.number_input("Age (years)", 
                                min_value=18, max_value=100, 
                                value=35,
                                help=feature_desc['Age'])
        
        submit_button = st.form_submit_button(label="Predict Diabetes Risk", 
                                            use_container_width=True,
                                            type="primary")
    
    # When form is submitted
    if submit_button:
        # Create input array for prediction
        input_data = np.array([
            pregnancies, glucose, blood_pressure, skin_thickness, 
            insulin, bmi, dpf, age
        ]).reshape(1, -1)
        
        # Create a DataFrame for visualization
        input_df = pd.DataFrame([{
            'Pregnancies': pregnancies, 'Glucose': glucose, 
            'BloodPressure': blood_pressure, 'SkinThickness': skin_thickness,
            'Insulin': insulin, 'BMI': bmi, 
            'DiabetesPedigreeFunction': dpf, 'Age': age
        }])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_scaled)[0, 1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Display prediction with AWS styling
        st.markdown(f"""
        <style>
        .prediction-header {{
            background-color: {AWS_COLORS['secondary']};
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            font-size: 1.5rem;
            text-align: center;
        }}
        </style>
        <div class="prediction-header">Prediction Result</div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.error("⚠️ High Risk of Diabetes")
                risk_level = "High Risk"
            else:
                st.success("✅ Low Risk of Diabetes")
                risk_level = "Low Risk"
            
            risk_percentage = f"{prediction_proba:.1%}"
            st.metric("Risk Probability", risk_percentage)
            st.caption(f"Using {model_name} model")
        
        with col2:
            # Create gauge chart for risk visualization with AWS colors
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Diabetes Risk", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': AWS_COLORS['tertiary']},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': AWS_COLORS['secondary'],
                    'steps': [
                        {'range': [0, 25], 'color': '#7CC097'},  # Green
                        {'range': [25, 50], 'color': '#F2CD5D'},  # Yellow
                        {'range': [50, 75], 'color': '#F8AA4B'},  # Orange
                        {'range': [75, 100], 'color': '#FB5A5A'},  # Red
                    ],
                    'threshold': {
                        'line': {'color': AWS_COLORS['primary'], 'width': 4},
                        'thickness': 0.75,
                        'value': prediction_proba * 100
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': AWS_COLORS['tertiary'], 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show risk factors
        st.subheader("Patient Risk Factor Analysis")
        
        # Get feature importance for this model
        feature_importance = get_feature_importance(model_name, model, X)
        
        # Highlight abnormal values
        abnormal_features = []
        for feature, value in input_df.iloc[0].items():
            if feature in normal_ranges:
                low, high = normal_ranges[feature]
                if value < low or value > high:
                    abnormal_features.append(feature)
        
        if abnormal_features:
            st.markdown(f"""
            <style>
            .warning-box {{
                background-color: #FFF9E6;
                border-left: 5px solid {AWS_COLORS['primary']};
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 0.3rem;
            }}
            </style>
            <div class="warning-box">
            <strong>⚠️ The following values are outside normal ranges:</strong><br>
            {"<br>".join([f"<b>{f}</b>: {input_df[f].iloc[0]}" for f in abnormal_features])}
            </div>
            """, unsafe_allow_html=True)
        
        # Show top features contributing to prediction
        if feature_importance is not None:
            st.subheader("Top Contributing Factors")
            
            # Sort features by importance
            sorted_importance = feature_importance.sort_values(ascending=False)
            top_features = sorted_importance.index[:5].tolist()  # Top 5 features
            
            # Create contribution chart
            contrib_df = pd.DataFrame({
                'Feature': [feature_desc.get(f, f) for f in top_features],
                'Importance': sorted_importance[:5],
                'Value': [input_df[f].iloc[0] for f in top_features]
            })
            
            fig = px.bar(contrib_df, x='Importance', y='Feature', 
                       text='Value',
                       orientation='h',
                       title="Feature Importance for Diabetes Risk",
                       color='Importance',
                       color_continuous_scale=['#00A1B4', AWS_COLORS['primary'], '#DD3C51']
                       )
            
            fig.update_layout(
                height=400,
                coloraxis_colorbar=dict(title="Importance"),
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Health recommendations with AWS styling
        st.markdown(f"""
        <style>
        .recommendations {{
            background-color: #F3FFFC;
            border-left: 5px solid {AWS_COLORS['success']};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.3rem;
        }}
        .recommendations h4 {{
            color: {AWS_COLORS['success']};
            margin-top: 0;
        }}
        </style>
        <div class="recommendations">
        <h4>Recommendations</h4>
        """, unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            Based on the high risk assessment:
            
            - **Consult a healthcare provider** for proper diabetes testing
            - **Monitor blood glucose levels** regularly
            - **Review diet** with focus on:
              - Limiting simple carbohydrates and sugars
              - Increasing fiber intake
              - Controlling portion sizes
            - **Increase physical activity** (aim for at least 150 minutes/week)
            - **Maintain healthy weight** through diet and exercise
            - **Regular check-ups** to monitor kidney function, eye health, and circulation
            """)
        else:
            st.markdown("""
            While your current risk appears low:
            
            - **Maintain a healthy lifestyle**:
              - Balanced diet rich in vegetables, fruits, and whole grains
              - Regular physical activity
              - Healthy weight management
            - **Regular check-ups** including blood glucose screening
            - **Stay well-hydrated** and limit sugary beverages
            - **Learn about diabetes symptoms** for early detection
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Disclaimer with AWS styling
        st.markdown(f"""
        <style>
        .disclaimer {{
            background-color: #F0F7FF;
            border: 1px solid {AWS_COLORS['info']};
            padding: 0.7rem;
            margin: 1rem 0;
            border-radius: 0.3rem;
            font-size: 0.9rem;
            text-align: center;
        }}
        </style>
        <div class="disclaimer">
        <strong>Disclaimer:</strong> This prediction is for educational purposes only and should not replace professional medical advice.
        </div>
        """, unsafe_allow_html=True)

# ----- Data Exploration Functions -----

def explore_heart_data(df, feature_desc):
    """Display exploratory visualizations for heart disease data"""
    st.subheader("Heart Disease Dataset Exploration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <style>
        .stats-box {{
            background-color: #F7F9FA;
            border-left: 5px solid {AWS_COLORS['tertiary']};
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.3rem;
        }}
        </style>
        <div class="stats-box">
        <strong>Dataset Statistics</strong><br>
        Total patients: {len(df)}<br>
        Heart disease cases: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)<br>
        Healthy cases: {len(df) - df['target'].sum()} ({(1-df['target'].mean())*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Feature Descriptions")
        
        # Create an expander for each feature
        for feature, description in feature_desc.items():
            with st.expander(f"{feature}"):
                st.write(description)
                
                # Show basic statistics for numeric features
                if feature != 'target' and df[feature].dtype in [np.int64, np.float64]:
                    st.caption(f"Min: {df[feature].min()}, Max: {df[feature].max()}, Mean: {df[feature].mean():.2f}")
    
    with col2:
        # Show distribution of heart disease by age with AWS colors
        fig = px.histogram(df, x="age", color="target", 
                         barmode="group", 
                         color_discrete_map={0: AWS_COLORS['tertiary'], 1: AWS_COLORS['primary']},
                         labels={"target": "Heart Disease", "age": "Age"},
                         title="Heart Disease Distribution by Age")
        
        fig.update_layout(height=400, legend=dict(
            title="Diagnosis",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig, use_column_width=True)
    
    # Interactive feature selection
    st.subheader("Explore Relationships Between Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature correlation heatmap with AWS colors
        st.markdown("#### Feature Correlations")
        
        # Calculate correlation matrix
        corr = df.corr()
        
        # Create heatmap with AWS-inspired colorscale
        fig = px.imshow(corr, 
                       text_auto=".2f", 
                       aspect="auto", 
                       color_continuous_scale=["#00A1B4", "#FFFFFF", AWS_COLORS['primary']])
        
        fig.update_layout(title="Correlation Matrix", height=550)
        st.plotly_chart(fig, use_column_width=True)
    
    with col2:
        # Comparison of two features
        st.markdown("#### Feature Comparison")
        
        feature1 = st.selectbox("Select First Feature", options=df.columns[:-1], index=0)
        feature2 = st.selectbox("Select Second Feature", options=df.columns[:-1], index=4)
        
        # Create scatter plot with AWS colors
        fig = px.scatter(df, x=feature1, y=feature2, color="target",
                       color_discrete_map={0: AWS_COLORS['tertiary'], 1: AWS_COLORS['primary']},
                       labels={"target": "Heart Disease"},
                       title=f"Relationship between {feature1} and {feature2}")
        
        fig.update_layout(legend=dict(
            title="Diagnosis",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        
        st.plotly_chart(fig, use_column_width=True)
    
    # Distribution by categorical features
    st.subheader("Disease Distribution by Categorical Features")
    
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    # Add interactive element - choose a feature to display
    selected_cat = st.selectbox("Select Categorical Feature", options=categorical_features)
    
    # Special formatting for categorical features
    if selected_cat == "sex":
        category_names = ["Female", "Male"]
    elif selected_cat == "cp":
        category_names = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    elif selected_cat == "fbs":
        category_names = ["≤ 120 mg/dl", "> 120 mg/dl"]
    elif selected_cat == "restecg":
        category_names = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    elif selected_cat == "exang":
        category_names = ["No", "Yes"]
    elif selected_cat == "slope":
        category_names = ["Upsloping", "Flat", "Downsloping"]
    elif selected_cat == "thal":
        category_names = ["Normal", "Fixed Defect", "Reversible Defect"]
    else:
        category_names = [str(i) for i in range(5)]
    
    # Convert to integer first to handle potential float values
    df[selected_cat] = df[selected_cat].astype(int)
    counts = df.groupby([selected_cat, 'target']).size().unstack(fill_value=0)
    
    # Normalize to get percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    
    # Create a bar chart with AWS colors
    fig = go.Figure()
    
    for target, color in zip([0, 1], [AWS_COLORS['tertiary'], AWS_COLORS['primary']]):
        if target in percentages.columns:
            fig.add_trace(go.Bar(
                x=[category_names[i] if i < len(category_names) else str(i) for i in percentages.index],
                y=percentages[target],
                name="No Disease" if target == 0 else "Heart Disease",
                marker_color=color
            ))
    
    fig.update_layout(
        barmode='group',
        title=f"Heart Disease Distribution by {selected_cat}",
        xaxis_title=selected_cat,
        yaxis_title="Percentage",
        legend_title="Diagnosis",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive demo section with sample patients
    st.subheader("Interactive Demo: Explore Sample Patient Profiles")
    
    st.markdown("""
    Below are three patient profiles. Click on a profile to see their risk assessment.
    This demonstrates how different combinations of factors influence heart disease risk.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    # Sample patient profiles
    patients = [
        {
            "name": "Patient A (Low Risk)",
            "data": {
                "age": 42, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
                "fbs": 0, "restecg": 0, "thalach": 165, "exang": 0,
                "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 1
            },
            "description": "Female, 42, normal blood pressure and cholesterol, no chest pain",
            "color": "#7CC097"  # Green
        },
        {
            "name": "Patient B (Medium Risk)",
            "data": {
                "age": 55, "sex": 1, "cp": 1, "trestbps": 140, "chol": 240,
                "fbs": 0, "restecg": 1, "thalach": 140, "exang": 0,
                "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
            },
            "description": "Male, 55, elevated blood pressure and cholesterol, atypical angina",
            "color": "#F8AA4B"  # Orange
        },
        {
            "name": "Patient C (High Risk)",
            "data": {
                "age": 68, "sex": 1, "cp": 3, "trestbps": 160, "chol": 290,
                "fbs": 1, "restecg": 2, "thalach": 120, "exang": 1,
                "oldpeak": 2.8, "slope": 2, "ca": 3, "thal": 3
            },
            "description": "Male, 68, high blood pressure, high cholesterol, asymptomatic chest pain",
            "color": "#FB5A5A"  # Red
        }
    ]
    
    selected_patient = None
    
    with col1:
        p = patients[0]
        if st.button(p["name"], key="patient_a1", 
                    use_container_width=True, 
                    type="secondary"):
            selected_patient = p
        st.markdown(f"<div style='color:{p['color']};'>{p['description']}</div>", unsafe_allow_html=True)
    
    with col2:
        p = patients[1]
        if st.button(p["name"], key="patient_b1", 
                    use_container_width=True, 
                    type="secondary"):
            selected_patient = p
        st.markdown(f"<div style='color:{p['color']};'>{p['description']}</div>", unsafe_allow_html=True)
    
    with col3:
        p = patients[2]
        if st.button(p["name"], key="patient_c1", 
                    use_container_width=True, 
                    type="secondary"):
            selected_patient = p
        st.markdown(f"<div style='color:{p['color']};'>{p['description']}</div>", unsafe_allow_html=True)
    
    # Display patient assessment if selected
    if selected_patient:
        st.markdown(f"""
        <style>
        .patient-profile {{
            background-color: #F7F9FA;
            border-left: 5px solid {selected_patient['color']};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.3rem;
        }}
        </style>
        <div class="patient-profile">
        <h4>{selected_patient['name']}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the patient's data in a nice format
        cols = st.columns(4)
        i = 0
        for feature, value in selected_patient['data'].items():
            with cols[i % 4]:
                if feature == "sex":
                    display_value = "Female" if value == 0 else "Male"
                elif feature == "cp":
                    cp_types = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
                    display_value = cp_types[value]
                elif feature == "fbs":
                    display_value = "Yes" if value == 1 else "No"
                else:
                    display_value = value
                
                st.metric(feature_desc.get(feature, feature), display_value)
            i += 1
        
        # If we have model data, make a prediction
        if 'heart_results' in st.session_state and st.session_state['heart_results']:
            st.markdown("### Risk Assessment")
            
            # Create input array for prediction
            model = st.session_state['heart_results']['Logistic Regression']['model']
            scaler = st.session_state['heart_scaler']
            
            input_data = np.array(list(selected_patient['data'].values())).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction_proba = model.predict_proba(input_scaled)[0, 1]
            risk_percentage = f"{prediction_proba:.1%}"
            
            # Show prediction with gauge
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric("Risk Probability", risk_percentage)
            
            with col2:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Heart Disease Risk", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': selected_patient['color']},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': AWS_COLORS['secondary'],
                        'steps': [
                            {'range': [0, 25], 'color': '#7CC097'},  # Green
                            {'range': [25, 50], 'color': '#F2CD5D'},  # Yellow
                            {'range': [50, 75], 'color': '#F8AA4B'},  # Orange
                            {'range': [75, 100], 'color': '#FB5A5A'},  # Red
                        ],
                        'threshold': {
                            'line': {'color': AWS_COLORS['primary'], 'width': 4},
                            'thickness': 0.75,
                            'value': prediction_proba * 100
                        }
                    }
                ))
                
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': AWS_COLORS['tertiary'], 'family': "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)

def explore_diabetes_data(df, feature_desc):
    """Display exploratory visualizations for diabetes data"""
    st.subheader("Diabetes Dataset Exploration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <style>
        .stats-box {{
            background-color: #F7F9FA;
            border-left: 5px solid {AWS_COLORS['tertiary']};
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.3rem;
        }}
        </style>
        <div class="stats-box">
        <strong>Dataset Statistics</strong><br>
        Total patients: {len(df)}<br>
        Diabetes cases: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)<br>
        Non-diabetic cases: {len(df) - df['Outcome'].sum()} ({(1-df['Outcome'].mean())*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Feature Descriptions")
        
        # Create an expander for each feature
        for feature, description in feature_desc.items():
            with st.expander(f"{feature}"):
                st.write(description)
                
                # Show basic statistics for numeric features
                if feature != 'Outcome' and df[feature].dtype in [np.int64, np.float64]:
                    st.caption(f"Min: {df[feature].min()}, Max: {df[feature].max()}, Mean: {df[feature].mean():.2f}")
    
    with col2:
        # Show distribution of diabetes by age with AWS colors
        fig = px.histogram(df, x="Age", color="Outcome", 
                         barmode="group", 
                         color_discrete_map={0: AWS_COLORS['tertiary'], 1: AWS_COLORS['primary']},
                         labels={"Outcome": "Diabetes", "Age": "Age"},
                         title="Diabetes Distribution by Age")
        
        fig.update_layout(height=400, legend=dict(
            title="Diagnosis",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive feature selection
    st.subheader("Explore Relationships Between Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature correlation heatmap with AWS colors
        st.markdown("#### Feature Correlations")
        
        # Calculate correlation matrix
        corr = df.corr()
        
        # Create heatmap with AWS-inspired colorscale
        fig = px.imshow(corr, 
                       text_auto=".2f", 
                       aspect="auto", 
                       color_continuous_scale=["#00A1B4", "#FFFFFF", AWS_COLORS['primary']])
        
        fig.update_layout(title="Correlation Matrix", height=550)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Comparison of two features
        st.markdown("#### Feature Comparison")
        
        feature1 = st.selectbox("Select First Feature", options=df.columns[:-1], index=1)  # Default to Glucose
        feature2 = st.selectbox("Select Second Feature", options=df.columns[:-1], index=5)  # Default to BMI
        
        # Create scatter plot with AWS colors
        fig = px.scatter(df, x=feature1, y=feature2, color="Outcome",
                       color_discrete_map={0: AWS_COLORS['tertiary'], 1: AWS_COLORS['primary']},
                       labels={"Outcome": "Diabetes"},
                       title=f"Relationship between {feature1} and {feature2}")
        
        fig.update_layout(legend=dict(
            title="Diagnosis",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distribution comparison
    st.subheader("Feature Distribution by Diabetes Status")
    
    selected_num = st.selectbox("Select Feature to Explore", 
                             options=['Glucose', 'BMI', 'Age', 'BloodPressure', 
                                     'Insulin', 'DiabetesPedigreeFunction'])
    
    # Create boxplots with AWS colors
    fig = px.box(df, x="Outcome", y=selected_num, 
               color="Outcome", 
               color_discrete_map={0: AWS_COLORS['tertiary'], 1: AWS_COLORS['primary']},
               labels={"Outcome": "Diabetes"},
               title=f"Distribution of {selected_num} by Diabetes Status",
               category_orders={"Outcome": [0, 1]})
    
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Diabetes', 'Diabetes']),
        legend=dict(
            title="Diagnosis",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive visualization - parallel coordinates
    st.subheader("Multi-Feature Visualization")
    
    st.markdown("""
    The parallel coordinates plot below shows relationships between multiple features simultaneously.
    Each line represents a patient, colored by diabetes status. This helps visualize patterns across multiple dimensions.
    """)
    
    # Create a parallel coordinates plot to visualize multiple features at once
    dimensions = [dict(range=[df[col].min(), df[col].max()],
                      label=col, values=df[col]) for col in df.columns[:-1]]
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=df['Outcome'],
                    colorscale=[[0, AWS_COLORS['tertiary']], [1, AWS_COLORS['primary']]],
                    showscale=True,
                    colorbar=dict(title="Diabetes")),
            dimensions=dimensions
        )
    )
    
    fig.update_layout(
        title="Parallel Coordinates Plot of Features",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive demo section with sample patients
    st.subheader("Interactive Demo: Explore Sample Patient Profiles")
    
    st.markdown("""
    Below are three patient profiles. Click on a profile to see their diabetes risk assessment.
    This demonstrates how different combinations of factors influence diabetes risk.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    # Sample patient profiles
    patients = [
        {
            "name": "Patient A (Low Risk)",
            "data": {
                "Pregnancies": 1, "Glucose": 85, "BloodPressure": 70,
                "SkinThickness": 25, "Insulin": 45, "BMI": 22.5,
                "DiabetesPedigreeFunction": 0.2, "Age": 28
            },
            "description": "28yo, normal glucose and BMI, no family history",
            "color": "#7CC097"  # Green
        },
        {
            "name": "Patient B (Medium Risk)",
            "data": {
                "Pregnancies": 3, "Glucose": 125, "BloodPressure": 80,
                "SkinThickness": 30, "Insulin": 95, "BMI": 28.5,
                "DiabetesPedigreeFunction": 0.5, "Age": 45
            },
            "description": "45yo, elevated glucose, overweight",
            "color": "#F8AA4B"  # Orange
        },
        {
            "name": "Patient C (High Risk)",
            "data": {
                "Pregnancies": 6, "Glucose": 170, "BloodPressure": 90,
                "SkinThickness": 35, "Insulin": 200, "BMI": 35.6,
                "DiabetesPedigreeFunction": 1.2, "Age": 58
            },
            "description": "58yo, high glucose, obese, family history",
            "color": "#FB5A5A"  # Red
        }
    ]
    
    selected_patient = None
    
    with col1:
        p = patients[0]
        if st.button(p["name"], key="patient_a", 
                    use_container_width=True, 
                    type="secondary"):
            selected_patient = p
        st.markdown(f"<div style='color:{p['color']};'>{p['description']}</div>", unsafe_allow_html=True)
    
    with col2:
        p = patients[1]
        if st.button(p["name"], key="patient_b", 
                    use_container_width=True, 
                    type="secondary"):
            selected_patient = p
        st.markdown(f"<div style='color:{p['color']};'>{p['description']}</div>", unsafe_allow_html=True)
    
    with col3:
        p = patients[2]
        if st.button(p["name"], key="patient_c", 
                    use_container_width=True, 
                    type="secondary"):
            selected_patient = p
        st.markdown(f"<div style='color:{p['color']};'>{p['description']}</div>", unsafe_allow_html=True)
    
    # Display patient assessment if selected
    if selected_patient:
        st.markdown(f"""
        <style>
        .patient-profile {{
            background-color: #F7F9FA;
            border-left: 5px solid {selected_patient['color']};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.3rem;
        }}
        </style>
        <div class="patient-profile">
        <h4>{selected_patient['name']}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the patient's data in a nice format
        cols = st.columns(4)
        i = 0
        for feature, value in selected_patient['data'].items():
            with cols[i % 4]:
                st.metric(feature_desc.get(feature, feature), value)
            i += 1
        
        # If we have model data, make a prediction
        if 'diabetes_results' in st.session_state and st.session_state['diabetes_results']:
            st.markdown("### Risk Assessment")
            
            # Create input array for prediction
            model = st.session_state['diabetes_results']['Logistic Regression']['model']
            scaler = st.session_state['diabetes_scaler']
            
            input_data = np.array(list(selected_patient['data'].values())).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction_proba = model.predict_proba(input_scaled)[0, 1]
            risk_percentage = f"{prediction_proba:.1%}"
            
            # Show prediction with gauge
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric("Risk Probability", risk_percentage)
            
            with col2:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Diabetes Risk", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': selected_patient['color']},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': AWS_COLORS['secondary'],
                        'steps': [
                            {'range': [0, 25], 'color': '#7CC097'},  # Green
                            {'range': [25, 50], 'color': '#F2CD5D'},  # Yellow
                            {'range': [50, 75], 'color': '#F8AA4B'},  # Orange
                            {'range': [75, 100], 'color': '#FB5A5A'},  # Red
                        ],
                        'threshold': {
                            'line': {'color': AWS_COLORS['primary'], 'width': 4},
                            'thickness': 0.75,
                            'value': prediction_proba * 100
                        }
                    }
                ))
                
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': AWS_COLORS['tertiary'], 'family': "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ----- Model Performance Functions -----

def show_model_performance(results, condition, X, X_test, y_test, feature_names):
    """Display model performance metrics and visualizations"""
    st.subheader("Model Performance Comparison")
    
    # Create a comparison table
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1 Score': [results[model]['f1'] for model in results],
        'ROC AUC': [results[model]['auc'] for model in results]
    })
    
    # Sort by AUC
    comparison_df = comparison_df.sort_values('ROC AUC', ascending=False).reset_index(drop=True)
    
    # Format percentages
    formatted_df = comparison_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.1%}")
    
    # Add styling for the best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_row = formatted_df['Model'] == best_model_name
    
    # Create a visual performance comparison chart
    fig = go.Figure()
    
    for i, model in enumerate(comparison_df['Model']):
        fig.add_trace(go.Bar(
            name=model,
            x=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            y=[results[model][metric.lower()] for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']],
            marker_color=AWS_COLORS['primary'] if model == best_model_name else AWS_COLORS['tertiary'],
            opacity=1.0 if model == best_model_name else 0.7
        ))
    
    fig.update_layout(
        barmode='group',
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(tickformat='.0%'),
        legend_title="Models",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display the table with AWS styling
        st.markdown(f"""
        <style>
        .best-model {{
            background-color: #FFF9E6 !important;
            font-weight: bold;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Display best model callout
        st.markdown(f"""
        <div style="background-color:#F3FFFC; padding:10px; border-radius:5px; margin-bottom:10px; 
                   border-left:5px solid {AWS_COLORS['success']}">
        <strong>✅ Best Model:</strong> {best_model_name}<br>
        <strong>AUC:</strong> {comparison_df.iloc[0]['ROC AUC']:.1%}
        </div>
        """, unsafe_allow_html=True)
        
        # Show the table
        st.dataframe(formatted_df.set_index('Model'), use_container_width=True)
    
    # Metrics explanation
    with st.expander("Understanding Performance Metrics"):
        st.markdown(f"""
        <style>
        .metric-explanation {{
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            background-color: #F7F9FA;
        }}
        </style>
        <div class="metric-explanation">
        <ul>
        <li><strong>Accuracy</strong>: Overall proportion of correct predictions (TP + TN) / Total</li>
        <li><strong>Precision</strong>: Proportion of true positives among positive predictions TP / (TP + FP)</li>
        <li><strong>Recall (Sensitivity)</strong>: Proportion of true positives correctly identified TP / (TP + FN)</li>
        <li><strong>F1 Score</strong>: Harmonic mean of precision and recall, balances both metrics</li>
        <li><strong>ROC AUC</strong>: Area under the Receiver Operating Characteristic curve, measures the model's ability to distinguish between classes</li>
        </ul>
        
        <p>For clinical applications, high sensitivity (recall) is often prioritized to minimize false negatives (missed diagnoses).</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Let user select a model to analyze
    st.subheader("Detailed Model Analysis")
    selected_model = st.selectbox("Select Model for Detailed Analysis", 
                                list(results.keys()), 
                                index=list(results.keys()).index(best_model_name))
    
    model_result = results[selected_model]
    model = model_result['model']
    
    # Create tabs for different analyses
    tabs = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance", "SHAP Analysis"])
    
    # Confusion Matrix tab
    with tabs[0]:
        cm = confusion_matrix(y_test, model_result['y_pred'])
        
        # Extract the values from the confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create a heatmap with AWS colors
            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=["No Disease", "Disease"],
                y=["No Disease", "Disease"],
                color_continuous_scale=["#EBF5FB", AWS_COLORS['tertiary']]
            )
            
            fig.update_layout(
                title=f"Confusion Matrix - {selected_model}",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Add interpretation of confusion matrix
            st.markdown(f"""
            <style>
            .cm-interpretation {{
                background-color: #F7F9FA;
                padding: 15px;
                border-radius: 5px;
            }}
            </style>
            <div class="cm-interpretation">
            <h4>Understanding the Confusion Matrix</h4>
            <ul>
            <li><strong>True Negatives (TN):</strong> {tn} - Correctly predicted as negative</li>
            <li><strong>False Positives (FP):</strong> {fp} - Incorrectly predicted as positive</li>
            <li><strong>False Negatives (FN):</strong> {fn} - Incorrectly predicted as negative</li>
            <li><strong>True Positives (TP):</strong> {tp} - Correctly predicted as positive</li>
            </ul>
            
            <p><strong>Clinical Implications:</strong></p>
            <p>False Negatives (missed diagnoses) can be particularly problematic in healthcare settings, as they may lead to delayed treatment.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ROC Curve tab
    with tabs[1]:
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, model_result['y_proba'])
        auc_score = roc_auc_score(y_test, model_result['y_proba'])
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create ROC curve with AWS colors
            fig = go.Figure()
            
            # Add ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {auc_score:.3f})',
                line=dict(color=AWS_COLORS['primary'], width=2)
            ))
            
            # Add diagonal line (random classifier)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color=AWS_COLORS['secondary'], width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'ROC Curve - {selected_model}',
                xaxis=dict(title='False Positive Rate'),
                yaxis=dict(title='True Positive Rate'),
                legend=dict(x=0.01, y=0.99),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <style>
            .roc-explanation {{
                background-color: #F7F9FA;
                padding: 15px;
                border-radius: 5px;
            }}
            </style>
            <div class="roc-explanation">
            <h4>Understanding ROC Curve</h4>
            <p>The ROC curve shows the trade-off between:</p>
            <ul>
            <li><strong>True Positive Rate (Sensitivity)</strong>: Proportion of actual positives correctly identified</li>
            <li><strong>False Positive Rate (1-Specificity)</strong>: Proportion of actual negatives incorrectly classified</li>
            </ul>
            
            <p><strong>Interpreting AUC:</strong></p>
            <ul>
            <li><strong>AUC = 1.0</strong>: Perfect classifier</li>
            <li><strong>AUC = 0.5</strong>: No better than random guessing</li>
            <li><strong>AUC > 0.8</strong>: Generally considered good</li>
            </ul>
            
            <p>This model's AUC: <strong>{auc_score:.3f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature Importance tab
    with tabs[2]:
        feature_importance = get_feature_importance(selected_model, model, X)
        
        if feature_importance is not None:
            # Create a DataFrame for the feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_importance.index,
                'Importance': feature_importance.values
            }).sort_values('Importance', ascending=False)
            
            # Create bar chart with AWS colors
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Feature Importance - {selected_model}',
                color='Importance',
                color_continuous_scale=['#00A1B4', AWS_COLORS['primary'], '#DD3C51']
            )
            
            fig.update_layout(
                height=500,
                coloraxis_colorbar=dict(title="Importance"),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"""
                <style>
                .feature-explanation {{
                    background-color: #F7F9FA;
                    padding: 15px;
                    border-radius: 5px;
                }}
                </style>
                <div class="feature-explanation">
                <h4>Top 5 Features for {condition.title()}</h4>
                <ol>
                """, unsafe_allow_html=True)
                
                # Show top 5 features with descriptions
                for i, (feature, importance) in enumerate(zip(importance_df['Feature'].head(5), importance_df['Importance'].head(5))):
                    feature_desc_text = ""
                    if condition == "heart disease":
                        feature_desc_dict = get_heart_disease_description()
                        if feature in feature_desc_dict:
                            feature_desc_text = feature_desc_dict[feature]
                    else:
                        feature_desc_dict = get_diabetes_description()
                        if feature in feature_desc_dict:
                            feature_desc_text = feature_desc_dict[feature]
                    
                    st.markdown(f"<li><strong>{feature}</strong>: {feature_desc_text}<br>Importance: {importance:.4f}</li>", unsafe_allow_html=True)
                
                st.markdown("""
                </ol>
                
                <p><strong>Note:</strong> Feature importance shows how influential each factor is in the model's predictions. Higher values indicate greater impact on the outcome.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a clinical interpretation
                st.markdown(f"""
                <div class="feature-explanation" style="margin-top: 15px;">
                <h4>Clinical Relevance</h4>
                <p>Understanding feature importance helps clinicians focus on the most relevant factors when assessing patient risk.</p>
                
                <p>For example, if glucose levels are highly important in the diabetes model, clinicians might prioritize regular glucose monitoring for at-risk patients.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"Feature importance visualization not available for {selected_model}")
    
    # SHAP Analysis tab
    with tabs[3]:
        # Try to calculate SHAP values (with error handling)
        try:
            # Take a small sample for SHAP analysis (for performance reasons)
            X_sample = X_test[:100]
            
            st.markdown("""
            SHAP (SHapley Additive exPlanations) values help explain how each feature contributes to predictions.
            This provides deeper insights into the model's decision-making process.
            """)
            
            with st.spinner("Calculating SHAP values (this may take a moment)..."):
                shap_values, explainer = calculate_shap_values(model, X_sample, selected_model)
            
            if shap_values is not None:
                # Create a DataFrame for visualization
                feature_names_list = list(feature_names)
                shap_df = pd.DataFrame()
                
                # Collect SHAP values for each feature
                for i, feature in enumerate(feature_names_list):
                    if i < shap_values.shape[1]:  # Ensure we don't go out of bounds
                        shap_df[feature] = shap_values[:, i]
                
                # Calculate mean absolute SHAP values for each feature
                mean_shap = shap_df.abs().mean().sort_values(ascending=False)
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Create bar chart for mean SHAP values with AWS colors
                    fig = px.bar(
                        x=mean_shap.values[:10],  # Top 10 features
                        y=mean_shap.index[:10],
                        orientation='h',
                        title="Mean |SHAP| Value (Feature Impact)",
                        labels={'x': 'mean |SHAP|', 'y': 'Feature'},
                        color=mean_shap.values[:10],
                        color_continuous_scale=['#00A1B4', AWS_COLORS['primary'], '#DD3C51']
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"""
                    <style>
                    .shap-explanation {{
                        background-color: #F7F9FA;
                        padding: 15px;
                        border-radius: 5px;
                    }}
                    </style>
                    <div class="shap-explanation">
                    <h4>Understanding SHAP Values</h4>
                    <p>SHAP values measure each feature's contribution to a prediction relative to the baseline prediction.</p>
                    
                    <ul>
                    <li><strong>Positive values:</strong> Push prediction higher</li>
                    <li><strong>Negative values:</strong> Push prediction lower</li>
                    <li><strong>Magnitude:</strong> Shows strength of impact</li>
                    </ul>
                    
                    <p>Unlike simple feature importance, SHAP values can show when a feature increases or decreases risk for specific patients.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create a long-form DataFrame for the summary plot
                summary_data = []
                
                # Get top features by mean SHAP
                top_features = mean_shap.index[:5].tolist()  # Top 5
                
                for feature in top_features:
                    for i in range(len(X_sample)):
                        summary_data.append({
                            'Feature': feature,
                            'SHAP Value': shap_df[feature][i],
                            'Feature Value': X_sample[:, feature_names_list.index(feature)][i]
                        })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Create the summary plot with AWS colors
                fig = px.strip(
                    summary_df,
                    x='SHAP Value',
                    y='Feature',
                    color='Feature Value',
                    color_continuous_scale=['#00A1B4', '#FFFFFF', AWS_COLORS['primary']],
                    stripmode='overlay',
                    title="SHAP Summary Plot"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # SHAP explanation with AWS styling
                st.markdown(f"""
                <div style="background-color:#F0F7FF; padding:15px; border-radius:5px; 
                           border-left:5px solid {AWS_COLORS['info']}">
                <h4>How to interpret SHAP values:</h4>
                <ul>
                <li><strong>Positive SHAP values</strong> (right side) increase the likelihood of the condition</li>
                <li><strong>Negative SHAP values</strong> (left side) decrease the likelihood</li>
                <li><strong>Color</strong> represents the feature value (blue = low, red = high)</li>
                </ul>
                
                <p>For example, high glucose levels (blue dots) typically have positive SHAP values, indicating they increase the predicted risk of diabetes.</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not calculate SHAP values: {e}")

# ----- Page Content Functions -----

def show_introduction():
    """Display the introduction page content"""
    st.markdown(f"""
    <style>
    .intro-header {{
        color: {AWS_COLORS['primary']};
        font-size: 2rem;
        margin-bottom: 1rem;
    }}
    .intro-subheader {{
        color: {AWS_COLORS['tertiary']};
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }}
    </style>
    <h1 class="intro-header">Machine Learning in Healthcare Diagnostics</h1>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Machine learning is transforming healthcare by enabling earlier detection, more accurate diagnosis, 
        and personalized treatment plans. This application demonstrates two common use cases:
        
        ### 1. Heart Disease Prediction
        
        Early detection of heart disease risk factors can significantly improve patient outcomes.
        Our model analyzes clinical parameters like:
        - Age and demographic information
        - Cholesterol levels
        - Blood pressure measurements
        - ECG results
        - Exercise test data
        
        ### 2. Diabetes Prediction
        
        Diabetes affects millions worldwide, and early intervention can prevent complications.
        The diabetes prediction model considers:
        - Glucose levels
        - BMI (Body Mass Index)
        - Blood pressure
        - Age and other personal factors
        - Family history through diabetes pedigree function
        """)
    
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*JmTyrCfYsNqf9yUeCxbWtw@2x.jpeg", 
                caption="Healthcare ML Applications", use_column_width=True)
    
    # Interactive demo section
    st.markdown(f"""
    <h2 class="intro-subheader">Interactive Demo</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Try out this quick demo to see how machine learning can predict health outcomes based on a few key metrics.
    Adjust the sliders below to see how different factors affect heart disease risk.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 25, 80, 55)
        cholesterol = st.slider("Cholesterol (mg/dL)", 120, 400, 250)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 90, 200, 130)
        max_heart_rate = st.slider("Maximum Heart Rate", 80, 220, 155)
    
    with col2:
        # Simple logistic regression model coefficients (simplified for demo)
        age_coef = 0.03
        chol_coef = 0.005
        bp_coef = 0.02
        heart_rate_coef = -0.02
        intercept = -4
        
        # Calculate risk score
        risk_score = (
            intercept +
            age * age_coef +
            cholesterol * chol_coef +
            blood_pressure * bp_coef +
            max_heart_rate * heart_rate_coef
        )
        
        # Convert to probability
        risk_probability = 1 / (1 + np.exp(-risk_score))
        
        # Create gauge chart with AWS colors
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': AWS_COLORS['tertiary']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': AWS_COLORS['secondary'],
                'steps': [
                    {'range': [0, 25], 'color': '#7CC097'},  # Green
                    {'range': [25, 50], 'color': '#F2CD5D'},  # Yellow
                    {'range': [50, 75], 'color': '#F8AA4B'},  # Orange
                    {'range': [75, 100], 'color': '#FB5A5A'},  # Red
                ],
                'threshold': {
                    'line': {'color': AWS_COLORS['primary'], 'width': 4},
                    'thickness': 0.75,
                    'value': risk_probability * 100
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': AWS_COLORS['tertiary'], 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors explanation
        risk_factors = []
        if age > 60:
            risk_factors.append("Age above 60")
        if cholesterol > 240:
            risk_factors.append("High cholesterol")
        if blood_pressure > 140:
            risk_factors.append("High blood pressure")
        if max_heart_rate < 120:
            risk_factors.append("Low maximum heart rate")
        
        if risk_factors:
            st.markdown(f"""
            <div style="background-color:#FFF9E6; padding:10px; border-radius:5px; 
                       border-left:5px solid {AWS_COLORS['primary']}">
            <strong>Key Risk Factors:</strong><br>
            {"<br>".join([f"• {factor}" for factor in risk_factors])}
            </div>
            """, unsafe_allow_html=True)
    
    # How to use this app section
    st.markdown(f"""
    <h2 class="intro-subheader">How to Use This Application</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color:#F7F9FA; padding:15px; border-radius:5px; height:200px;">
        <h3 style="color:{AWS_COLORS['tertiary']};">1. Explore Data</h3>
        <p>Browse through the datasets to understand key risk factors for heart disease and diabetes.</p>
        <p>Examine correlations between different health metrics and disease outcomes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color:#F7F9FA; padding:15px; border-radius:5px; height:200px;">
        <h3 style="color:{AWS_COLORS['tertiary']};">2. Try Predictions</h3>
        <p>Enter patient information to get disease risk predictions.</p>
        <p>Adjust values to see how different factors affect risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color:#F7F9FA; padding:15px; border-radius:5px; height:200px;">
        <h3 style="color:{AWS_COLORS['tertiary']};">3. Analyze Models</h3>
        <p>Compare performance of different machine learning models.</p>
        <p>Study feature importance and SHAP values to understand what drives predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background-color:#F0F7FF; padding:15px; border-radius:5px; 
                margin-top:20px; border-left:5px solid {AWS_COLORS['info']}">
    <strong>Remember:</strong> This is a demonstration tool. Real medical decisions should involve healthcare professionals.
    </div>
    """, unsafe_allow_html=True)

def show_about_ml_healthcare():
    """Display information about ML in healthcare"""
    st.markdown(f"""
    <style>
    .about-header {{
        color: {AWS_COLORS['primary']};
        font-size: 2rem;
        margin-bottom: 1rem;
    }}
    .about-subheader {{
        color: {AWS_COLORS['tertiary']};
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }}
    </style>
    <h1 class="about-header">Machine Learning in Healthcare</h1>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Transforming Medical Diagnostics

        Machine learning is revolutionizing healthcare by enhancing diagnostic accuracy, improving treatment plans, and increasing operational efficiency. These technologies are becoming essential tools for healthcare providers seeking to deliver personalized and precise care.

        #### Key Applications in Healthcare

        1. **Disease Diagnosis and Prediction**
           - Early detection of diseases like cancer, diabetes, and heart disease
           - Identification of high-risk patients for preventive interventions
           - Analysis of medical images (X-rays, MRIs, CT scans) to detect abnormalities

        2. **Treatment Optimization**
           - Personalized treatment recommendations based on patient characteristics
           - Medication effectiveness prediction
           - Adverse event and readmission risk assessment

        3. **Healthcare Operations**
           - Resource allocation and scheduling optimization
           - Fraud detection in healthcare claims
           - Patient flow management and hospital capacity planning
        """)
    
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/0*fh8Qgq2cAwdEpJjR", 
                caption="AI in Healthcare", use_column_width=True)
        
        st.markdown(f"""
        <div style="background-color:#F7F9FA; padding:15px; border-radius:5px; margin-top:20px;">
        <h4 style="color:{AWS_COLORS['tertiary']};">ML Technologies in Medical Use Today</h4>
        <ul>
        <li><strong>Amazon Comprehend Medical</strong>: NLP for medical text analysis</li>
        <li><strong>IBM Watson for Oncology</strong>: Treatment recommendation</li>
        <li><strong>Google Health</strong>: Medical imaging analysis</li>
        <li><strong>PathAI</strong>: Pathology diagnosis assistance</li>
        <li><strong>Aidoc</strong>: Radiology diagnostic support</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <h2 class="about-subheader">Types of Machine Learning in Healthcare</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color:#F7F9FA; padding:15px; border-radius:5px; height:280px;">
        <h4 style="color:{AWS_COLORS['tertiary']};">Supervised Learning</h4>
        
        <p><strong>Applications:</strong></p>
        <ul>
        <li>Disease classification</li>
        <li>Mortality risk prediction</li>
        <li>Treatment outcome prediction</li>
        </ul>
        
        <p><strong>Examples:</strong></p>
        <ul>
        <li>Logistic regression for disease risk</li>
        <li>Random forests for readmission prediction</li>
        <li>Neural networks for image analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color:#F7F9FA; padding:15px; border-radius:5px; height:280px;">
        <h4 style="color:{AWS_COLORS['tertiary']};">Unsupervised Learning</h4>
        
        <p><strong>Applications:</strong></p>
        <ul>
        <li>Patient segmentation</li>
        <li>Anomaly detection in medical data</li>
        <li>Disease subtype discovery</li>
        </ul>
        
        <p><strong>Examples:</strong></p>
        <ul>
        <li>Clustering for patient stratification</li>
        <li>Dimensionality reduction for genomic data</li>
        <li>Association rules for comorbidity patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color:#F7F9FA; padding:15px; border-radius:5px; height:280px;">
        <h4 style="color:{AWS_COLORS['tertiary']};">Reinforcement Learning</h4>
        
        <p><strong>Applications:</strong></p>
        <ul>
        <li>Treatment optimization</li>
        <li>Adaptive clinical trials</li>
        <li>Personalized dosing regimens</li>
        </ul>
        
        <p><strong>Examples:</strong></p>
        <ul>
        <li>Dynamic treatment regimes</li>
        <li>Automated insulin delivery systems</li>
        <li>Robotic surgery assistance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <h2 class="about-subheader">Ethical Considerations and Challenges</h2>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background-color:#F0F7FF; padding:20px; border-radius:5px; 
               border-left:5px solid {AWS_COLORS['info']}">
    <p>While ML offers tremendous potential, its implementation in healthcare faces important challenges:</p>
    
    <ol>
    <li><strong>Data Privacy and Security</strong>: Patient data requires strict protection under regulations like HIPAA</li>
    <li><strong>Algorithm Bias</strong>: ML models can perpetuate or amplify existing healthcare disparities</li>
    <li><strong>Transparency and Explainability</strong>: Healthcare professionals need to understand AI recommendations</li>
    <li><strong>Regulatory Approval</strong>: Medical ML applications require rigorous validation and approval</li>
    <li><strong>Integration with Workflow</strong>: Technology must enhance, not burden, clinical workflow</li>
    <li><strong>Human Oversight</strong>: ML should augment, not replace, healthcare professional judgment</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <h2 class="about-subheader">Future Directions</h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color:#F7F9FA; padding:15px; border-radius:5px;">
        <h4 style="color:{AWS_COLORS['tertiary']};">Emerging Trends</h4>
        
        <ul>
        <li><strong>Federated Learning</strong>: Training models across institutions without sharing raw data</li>
        <li><strong>Multimodal Models</strong>: Integrating diverse data types (images, text, genomics)</li>
        <li><strong>Continuous Learning Systems</strong>: Models that adapt to new medical evidence</li>
        <li><strong>Edge Computing</strong>: Bringing ML capabilities to medical devices</li>
        <li><strong>Explainable AI</strong>: More transparent algorithms for clinical decision support</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color:#F7F9FA; padding:15px; border-radius:5px;">
        <h4 style="color:{AWS_COLORS['tertiary']};">On the Horizon</h4>
        
        <ul>
        <li><strong>Digital Twin Technology</strong>: Patient-specific models for treatment simulation</li>
        <li><strong>Ambient Clinical Intelligence</strong>: AI assistants during patient-doctor interactions</li>
        <li><strong>Real-time Monitoring</strong>: ML-powered wearables and IoT devices</li>
        <li><strong>Precision Prevention</strong>: Personalized health maintenance programs</li>
        <li><strong>Cross-modal Transfer Learning</strong>: Applying learning from one medical domain to another</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive visualization of ML adoption in healthcare
    st.markdown(f"""
    <h2 class="about-subheader">ML Adoption in Healthcare</h2>
    """, unsafe_allow_html=True)
    
    # Sample data for ML adoption
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    adoption_rates = [12, 18, 27, 35, 42, 53, 61, 68]  # Percentages
    
    # Create interactive chart
    fig = px.line(
        x=years, y=adoption_rates,
        labels={"x": "Year", "y": "Adoption Rate (%)"},
        title="ML Adoption in Healthcare Organizations",
        markers=True
    )
    
    fig.update_traces(line=dict(color=AWS_COLORS['primary'], width=3))
    
    fig.add_shape(
        type="rect",
        x0=2023.5, x1=2025.5, y0=0, y1=70,
        fillcolor="rgba(255,153,0,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        layer="below"
    )
    
    fig.add_annotation(
        x=2024.5, y=65,
        text="Projected",
        showarrow=False,
        font=dict(color=AWS_COLORS['tertiary'])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div style="background-color:#F3FFFC; padding:15px; border-radius:5px; 
               border-left:5px solid {AWS_COLORS['success']};">
    <p><strong>Note:</strong> Despite these advancements, the healthcare industry emphasizes that machine learning tools are designed to support, 
    not replace, the clinical judgment of healthcare professionals. The human element of care remains essential.</p>
    </div>
    """, unsafe_allow_html=True)

# ----- Main Application Function -----

def main():
    """Main application function"""

    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    setup_sidebar()
    
    # Create header
    # create_header()
    st.markdown("<h1>🏥 Healthcare Diagnostics with Machine Learning</h1>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Explore how ML can assist in medical predictions and decision support</div>", unsafe_allow_html=True)
    load_css()
    
    # Load datasets
    heart_df = load_heart_disease_data()
    diabetes_df = load_diabetes_data()
    
    # Get feature descriptions
    heart_desc = get_heart_disease_description()
    diabetes_desc = get_diabetes_description()
    
    main_tabs = st.tabs(["🏠 Introduction", "❤️ Heart Disease Prediction", "🩸 Diabetes Prediction", "📊 About ML in Healthcare"])
    
    
    # Main content based on selected mode
    with main_tabs[0]:
        show_introduction()
    
    # Heart Disease Prediction
    with main_tabs[1]:
        st.markdown("""<div class='info-box'>
                    Heart disease prediction in machine learning employs algorithms to analyze patient data including demographic information, vital signs, and clinical measurements to identify patterns and risk factors that indicate the likelihood of cardiovascular disease, enabling early intervention and personalized treatment strategies.     
                    </div>""",unsafe_allow_html=True)
        
        tabs = st.tabs(["🔮 Make Prediction", "📊 Explore Data", "📈 Model Performance"])
        
        # Process data and train models if not already in session state
        if 'heart_results' not in st.session_state or st.session_state['heart_results'] is None:
            with st.spinner("Training heart disease models (this may take a moment)..."):
                X_heart, X_heart_scaled, y_heart, X_heart_train, X_heart_test, y_heart_train, y_heart_test, heart_scaler = preprocess_data(heart_df, 'heart_disease')
                heart_results = train_models(X_heart_train, y_heart_train, X_heart_test, y_heart_test)
                
                # Store in session state
                st.session_state['heart_results'] = heart_results
                st.session_state['heart_data'] = (X_heart, y_heart)
                st.session_state['heart_test_data'] = (X_heart_test, y_heart_test)
                st.session_state['heart_scaler'] = heart_scaler
        else:
            # Retrieve from session state
            heart_results = st.session_state['heart_results']
            X_heart, y_heart = st.session_state['heart_data']
            X_heart_test, y_heart_test = st.session_state['heart_test_data']
            heart_scaler = st.session_state['heart_scaler']
        
        # Make Prediction tab
        with tabs[0]:
            show_heart_disease_prediction(heart_results, X_heart, heart_scaler, heart_desc)
        
        # Explore Data tab
        with tabs[1]:
            explore_heart_data(heart_df, heart_desc)
        
        # Model Performance tab
        with tabs[2]:
            show_model_performance(heart_results, "heart disease", X_heart, X_heart_test, y_heart_test, X_heart.columns)
    
    # Diabetes Prediction
    with main_tabs[2]:
        st.markdown("""<div class='info-box'>
                    Diabetes prediction in machine learning utilizes algorithms to analyze patient health metrics, including glucose levels, BMI, age, family history, and lifestyle factors, to identify patterns and risk indicators that accurately forecast an individual's likelihood of developing diabetes, enabling timely preventive interventions and personalized healthcare recommendations.  
                    </div>""",unsafe_allow_html=True)
        tabs = st.tabs(["🔮 Make Prediction", "📊 Explore Data", "📈 Model Performance"])
        
        # Process data and train models if not already in session state
        if 'diabetes_results' not in st.session_state or st.session_state['diabetes_results'] is None:
            with st.spinner("Training diabetes models (this may take a moment)..."):
                X_diabetes, X_diabetes_scaled, y_diabetes, X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test, diabetes_scaler = preprocess_data(diabetes_df, 'diabetes')
                diabetes_results = train_models(X_diabetes_train, y_diabetes_train, X_diabetes_test, y_diabetes_test)
                
                # Store in session state
                st.session_state['diabetes_results'] = diabetes_results
                st.session_state['diabetes_data'] = (X_diabetes, y_diabetes)
                st.session_state['diabetes_test_data'] = (X_diabetes_test, y_diabetes_test)
                st.session_state['diabetes_scaler'] = diabetes_scaler
        else:
            # Retrieve from session state
            diabetes_results = st.session_state['diabetes_results']
            X_diabetes, y_diabetes = st.session_state['diabetes_data']
            X_diabetes_test, y_diabetes_test = st.session_state['diabetes_test_data']
            diabetes_scaler = st.session_state['diabetes_scaler']
        
        # Make Prediction tab
        with tabs[0]:
            show_diabetes_prediction(diabetes_results, X_diabetes, diabetes_scaler, diabetes_desc)
        
        # Explore Data tab
        with tabs[1]:
            explore_diabetes_data(diabetes_df, diabetes_desc)
        
        # Model Performance tab
        with tabs[2]:
            show_model_performance(diabetes_results, "diabetes", X_diabetes, X_diabetes_test, y_diabetes_test, X_diabetes.columns)
    
    # About ML in Healthcare
    with main_tabs[3]:
        show_about_ml_healthcare()
    
    # Create footer
    create_footer()


if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="AWS Healthcare ML Diagnostics",
        page_icon="🏥",
        layout="wide"
    )
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

def generate_customer_data(n=500):
    """Generate synthetic customer data for segmentation"""
    np.random.seed(42)
    
    # Generate age between 18-75
    age = np.random.normal(42, 12, n).astype(int)
    age = np.clip(age, 18, 75)
    
    # Generate income between 20000-150000 with some correlation to age
    base_income = 20000 + 30000 * np.random.random(n)
    age_factor = (age - 18) * 1000 * (0.5 + np.random.random(n))
    income = base_income + age_factor
    income = np.clip(income, 20000, 150000).astype(int)
    
    # Generate spending score (1-100)
    # Higher income tends to correlate with higher spending, but with variation
    spending_score = 20 + 60 * (income / 150000) + np.random.normal(0, 20, n)
    spending_score = np.clip(spending_score, 1, 100).astype(int)
    
    # Website visits per month (0-30)
    website_visits = np.random.poisson(10, n)
    website_visits = np.clip(website_visits, 0, 30)
    
    # Email click rate (0-100%)
    email_clicks = np.random.beta(2, 5, n) * 100
    
    # Days since last purchase (0-365)
    days_since_last_purchase = np.random.exponential(50, n).astype(int)
    days_since_last_purchase = np.clip(days_since_last_purchase, 0, 365)
    
    # Gender (categorical)
    gender = np.random.choice(['M', 'F'], n)
    
    # Location (categorical)
    location = np.random.choice(['urban', 'suburban', 'rural'], n, p=[0.5, 0.3, 0.2])
    
    # Purchase history (number of previous purchases)
    purchase_history = np.random.poisson(5, n)
    
    # Customer ID
    customer_id = np.arange(1001, 1001 + n)
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': customer_id,
        'age': age,
        'income': income,
        'gender': gender,
        'location': location,
        'purchase_history': purchase_history,
        'website_visits': website_visits,
        'email_clicks': email_clicks,
        'days_since_last_purchase': days_since_last_purchase,
        'spending_score': spending_score
    })
    
    return data

@st.cache_data
def get_customer_data():
    """Return cached customer data or generate new data"""
    return generate_customer_data()

def preprocess_data(data):
    """Preprocess customer data for clustering"""
    # Create a copy to avoid modifying original
    df = data.copy()
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['gender', 'location'])
    
    # Select features for clustering
    features = [
        'age', 'income', 'purchase_history', 'website_visits', 
        'email_clicks', 'days_since_last_purchase', 'spending_score',
        'gender_F', 'gender_M', 'location_rural', 'location_suburban', 'location_urban'
    ]
    
    X = df_encoded[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, features, scaler

def create_customer_profile(
    age, income, gender, location, purchase_history, 
    website_visits, email_clicks, days_since_last_purchase, spending_score
):
    """Create a single customer profile for prediction"""
    # Create DataFrame with one row
    customer = pd.DataFrame({
        'age': [age],
        'income': [income],
        'gender': [gender],
        'location': [location],
        'purchase_history': [purchase_history],
        'website_visits': [website_visits],
        'email_clicks': [email_clicks],
        'days_since_last_purchase': [days_since_last_purchase],
        'spending_score': [spending_score]
    })
    
    return customer
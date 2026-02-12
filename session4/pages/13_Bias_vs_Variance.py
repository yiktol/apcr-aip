import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import utils.authenticate as authenticate
from utils.styles import load_css, sub_header, custom_header

# Set page config
st.set_page_config(
    page_title="Bias vs Variance - Model Generalization",
    page_icon="⚖️",
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

def generate_data(n_samples=100, noise_level=0.3, seed=42):
    """Generate synthetic data with a non-linear pattern"""
    np.random.seed(seed)
    X = np.linspace(0, 10, n_samples)
    # True function: sine wave with some complexity
    y_true = np.sin(X) + 0.3 * np.sin(3 * X)
    # Add noise
    y = y_true + np.random.normal(0, noise_level, n_samples)
    return X, y, y_true

def fit_polynomial_model(X, y, degree):
    """Fit a polynomial regression model"""
    X_reshaped = X.reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_reshaped)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Generate predictions
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    y_pred = model.predict(X_plot_poly)
    
    # Calculate metrics
    y_train_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)
    
    return X_plot.flatten(), y_pred, mse, r2

def create_comparison_plot(X, y, y_true, models_data, title):
    """Create a comparison plot showing data and fitted models"""
    fig = go.Figure()
    
    # Add true function
    fig.add_trace(go.Scatter(
        x=X, y=y_true,
        mode='lines',
        name='True Function',
        line=dict(color='green', width=3, dash='dash'),
        opacity=0.7
    ))
    
    # Add data points
    fig.add_trace(go.Scatter(
        x=X, y=y,
        mode='markers',
        name='Training Data',
        marker=dict(color=COLORS['info'], size=8, opacity=0.6)
    ))
    
    # Add fitted models
    colors = [COLORS['danger'], COLORS['success'], COLORS['warning']]
    for i, (name, X_plot, y_pred) in enumerate(models_data):
        fig.add_trace(go.Scatter(
            x=X_plot, y=y_pred,
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=3)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def main():
    st.markdown(custom_header("Model Generalization: Bias vs Variance", 1), unsafe_allow_html=True)
    
    st.markdown("""
    Understanding the bias-variance tradeoff is crucial for building models that generalize well to new data.
    This interactive page demonstrates the concepts of underfitting, overfitting, and appropriate fitting.
    """)
    
    # Concept explanation
    st.markdown(sub_header("Understanding Bias and Variance", "📚"), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Bias
        
        **Bias** refers to the error introduced by approximating a real-world problem with a simplified model.
        
        - **High Bias**: Model is too simple, makes strong assumptions
        - **Result**: Underfitting - poor performance on both training and test data
        - **Example**: Using a linear model for non-linear data
        
        **Characteristics of High Bias:**
        - Oversimplified model
        - Poor training accuracy
        - Poor test accuracy
        - Model doesn't capture underlying patterns
        """)
    
    with col2:
        st.markdown("""
        ### Variance
        
        **Variance** refers to the model's sensitivity to small fluctuations in the training data.
        
        - **High Variance**: Model is too complex, learns noise
        - **Result**: Overfitting - good training performance, poor test performance
        - **Example**: Using a very high-degree polynomial
        
        **Characteristics of High Variance:**
        - Overly complex model
        - Excellent training accuracy
        - Poor test accuracy
        - Model memorizes training data
        """)
    
    # Visual representation
    st.markdown(sub_header("The Bias-Variance Tradeoff", "⚖️"), unsafe_allow_html=True)
    
    # Create tradeoff visualization
    complexity = np.linspace(0, 10, 100)
    bias = 10 / (1 + complexity)
    variance = complexity / 10
    total_error = bias + variance + 0.5
    
    fig_tradeoff = go.Figure()
    
    fig_tradeoff.add_trace(go.Scatter(
        x=complexity, y=bias,
        mode='lines',
        name='Bias',
        line=dict(color=COLORS['danger'], width=3)
    ))
    
    fig_tradeoff.add_trace(go.Scatter(
        x=complexity, y=variance,
        mode='lines',
        name='Variance',
        line=dict(color=COLORS['warning'], width=3)
    ))
    
    fig_tradeoff.add_trace(go.Scatter(
        x=complexity, y=total_error,
        mode='lines',
        name='Total Error',
        line=dict(color=COLORS['secondary'], width=4, dash='dash')
    ))
    
    # Mark optimal point
    optimal_idx = np.argmin(total_error)
    fig_tradeoff.add_trace(go.Scatter(
        x=[complexity[optimal_idx]],
        y=[total_error[optimal_idx]],
        mode='markers',
        name='Optimal Complexity',
        marker=dict(color=COLORS['success'], size=15, symbol='star')
    ))
    
    fig_tradeoff.update_layout(
        title="Bias-Variance Tradeoff",
        xaxis_title="Model Complexity",
        yaxis_title="Error",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_tradeoff, use_container_width=True)
    
    st.info("""
    **Key Insight**: The optimal model complexity balances bias and variance to minimize total error.
    Too simple → high bias, too complex → high variance.
    """)
    
    # Interactive demonstration
    st.markdown(sub_header("Interactive Demonstration", "🎮", "aws"), unsafe_allow_html=True)
    
    st.markdown("""
    Adjust the parameters below to see how different model complexities affect fitting:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = st.slider("Number of samples", 50, 200, 100, 10)
    
    with col2:
        noise_level = st.slider("Noise level", 0.1, 1.0, 0.3, 0.1)
    
    with col3:
        seed = st.slider("Random seed", 0, 100, 42, 1)
    
    # Generate data
    X, y, y_true = generate_data(n_samples, noise_level, seed)
    
    # Fit models with different complexities
    st.markdown(sub_header("Model Comparison", "📊", "minimal"), unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Underfitting (High Bias)",
        "🟢 Appropriate Fitting",
        "🟡 Overfitting (High Variance)",
        "📈 All Together"
    ])
    
    # Underfitting
    with tab1:
        st.markdown("### Underfitting: Model Too Simple")
        
        degree_underfit = 1
        X_plot_under, y_pred_under, mse_under, r2_under = fit_polynomial_model(X, y, degree_underfit)
        
        fig_under = go.Figure()
        
        # True function
        fig_under.add_trace(go.Scatter(
            x=X, y=y_true,
            mode='lines',
            name='True Function',
            line=dict(color='green', width=3, dash='dash')
        ))
        
        # Data points
        fig_under.add_trace(go.Scatter(
            x=X, y=y,
            mode='markers',
            name='Training Data',
            marker=dict(color=COLORS['info'], size=8, opacity=0.6)
        ))
        
        # Underfit model
        fig_under.add_trace(go.Scatter(
            x=X_plot_under, y=y_pred_under,
            mode='lines',
            name=f'Linear Model (degree={degree_underfit})',
            line=dict(color=COLORS['danger'], width=4)
        ))
        
        fig_under.update_layout(
            title="Underfitting Example: Linear Model on Non-linear Data",
            xaxis_title="X",
            yaxis_title="Y",
            height=500
        )
        
        st.plotly_chart(fig_under, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Complexity", f"Degree {degree_underfit}", "Too Low")
        with col2:
            st.metric("Training MSE", f"{mse_under:.4f}", "High")
        with col3:
            st.metric("R² Score", f"{r2_under:.4f}", "Poor")
        
        st.markdown("""
        **Observations:**
        - The linear model cannot capture the non-linear pattern in the data
        - High training error indicates the model is too simple
        - The model has **high bias** - it makes strong assumptions that don't match reality
        
        **How to fix underfitting:**
        - ✅ Increase model complexity (higher degree polynomial, more layers in neural network)
        - ✅ Add more relevant features
        - ✅ Reduce regularization
        - ✅ Train for more epochs (for iterative models)
        """)
    
    # Appropriate fitting
    with tab2:
        st.markdown("### Appropriate Fitting: Balanced Model")
        
        degree_good = 5
        X_plot_good, y_pred_good, mse_good, r2_good = fit_polynomial_model(X, y, degree_good)
        
        fig_good = go.Figure()
        
        # True function
        fig_good.add_trace(go.Scatter(
            x=X, y=y_true,
            mode='lines',
            name='True Function',
            line=dict(color='green', width=3, dash='dash')
        ))
        
        # Data points
        fig_good.add_trace(go.Scatter(
            x=X, y=y,
            mode='markers',
            name='Training Data',
            marker=dict(color=COLORS['info'], size=8, opacity=0.6)
        ))
        
        # Good fit model
        fig_good.add_trace(go.Scatter(
            x=X_plot_good, y=y_pred_good,
            mode='lines',
            name=f'Polynomial Model (degree={degree_good})',
            line=dict(color=COLORS['success'], width=4)
        ))
        
        fig_good.update_layout(
            title="Appropriate Fitting: Model Captures Pattern Without Overfitting",
            xaxis_title="X",
            yaxis_title="Y",
            height=500
        )
        
        st.plotly_chart(fig_good, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Complexity", f"Degree {degree_good}", "Optimal")
        with col2:
            st.metric("Training MSE", f"{mse_good:.4f}", "Good")
        with col3:
            st.metric("R² Score", f"{r2_good:.4f}", "Good")
        
        st.markdown("""
        **Observations:**
        - The model captures the underlying pattern without fitting to noise
        - Training error is low and reasonable
        - The model balances **bias and variance** effectively
        - Predictions closely follow the true function
        
        **Characteristics of good fit:**
        - ✅ Low bias - captures underlying patterns
        - ✅ Low variance - doesn't overfit to noise
        - ✅ Good generalization to new data
        - ✅ Reasonable training and test performance
        """)
    
    # Overfitting
    with tab3:
        st.markdown("### Overfitting: Model Too Complex")
        
        degree_overfit = 20
        X_plot_over, y_pred_over, mse_over, r2_over = fit_polynomial_model(X, y, degree_overfit)
        
        fig_over = go.Figure()
        
        # True function
        fig_over.add_trace(go.Scatter(
            x=X, y=y_true,
            mode='lines',
            name='True Function',
            line=dict(color='green', width=3, dash='dash')
        ))
        
        # Data points
        fig_over.add_trace(go.Scatter(
            x=X, y=y,
            mode='markers',
            name='Training Data',
            marker=dict(color=COLORS['info'], size=8, opacity=0.6)
        ))
        
        # Overfit model
        fig_over.add_trace(go.Scatter(
            x=X_plot_over, y=y_pred_over,
            mode='lines',
            name=f'Polynomial Model (degree={degree_overfit})',
            line=dict(color=COLORS['warning'], width=4)
        ))
        
        fig_over.update_layout(
            title="Overfitting Example: High-Degree Polynomial Fits Noise",
            xaxis_title="X",
            yaxis_title="Y",
            height=500
        )
        
        st.plotly_chart(fig_over, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Complexity", f"Degree {degree_overfit}", "Too High")
        with col2:
            st.metric("Training MSE", f"{mse_over:.4f}", "Very Low")
        with col3:
            st.metric("R² Score", f"{r2_over:.4f}", "Excellent (Suspicious!)")
        
        st.markdown("""
        **Observations:**
        - The model fits the training data almost perfectly, including noise
        - Wild oscillations between data points indicate overfitting
        - The model has **high variance** - it's too sensitive to training data
        - Would perform poorly on new, unseen data
        
        **How to fix overfitting:**
        - ✅ Reduce model complexity (lower degree, fewer layers)
        - ✅ Add more training data
        - ✅ Use regularization (L1, L2, dropout)
        - ✅ Apply early stopping
        - ✅ Use cross-validation
        - ✅ Feature selection/reduction
        """)
    
    # All together
    with tab4:
        st.markdown("### Comparison: All Three Models")
        
        models_data = [
            (f"Underfitting (degree={degree_underfit})", X_plot_under, y_pred_under),
            (f"Appropriate (degree={degree_good})", X_plot_good, y_pred_good),
            (f"Overfitting (degree={degree_overfit})", X_plot_over, y_pred_over)
        ]
        
        fig_all = create_comparison_plot(X, y, y_true, models_data, "Model Comparison: Underfitting vs Appropriate vs Overfitting")
        st.plotly_chart(fig_all, use_container_width=True)
        
        # Metrics comparison
        st.markdown("### Performance Metrics Comparison")
        
        metrics_df = pd.DataFrame({
            'Model': ['Underfitting', 'Appropriate Fitting', 'Overfitting'],
            'Degree': [degree_underfit, degree_good, degree_overfit],
            'Training MSE': [mse_under, mse_good, mse_over],
            'R² Score': [r2_under, r2_good, r2_over],
            'Bias': ['High', 'Low', 'Low'],
            'Variance': ['Low', 'Low', 'High']
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        st.success("""
        **Key Takeaway**: The appropriate model (degree 5) balances bias and variance, 
        achieving good performance without overfitting to noise. This is the model we would 
        choose for deployment.
        """)
    
    # Custom model explorer
    st.markdown(sub_header("Custom Model Explorer", "🔬", "outline"), unsafe_allow_html=True)
    
    st.markdown("Try different polynomial degrees to see how model complexity affects fitting:")
    
    custom_degree = st.slider("Select polynomial degree", 1, 25, 5, 1)
    
    X_plot_custom, y_pred_custom, mse_custom, r2_custom = fit_polynomial_model(X, y, custom_degree)
    
    fig_custom = go.Figure()
    
    # True function
    fig_custom.add_trace(go.Scatter(
        x=X, y=y_true,
        mode='lines',
        name='True Function',
        line=dict(color='green', width=3, dash='dash')
    ))
    
    # Data points
    fig_custom.add_trace(go.Scatter(
        x=X, y=y,
        mode='markers',
        name='Training Data',
        marker=dict(color=COLORS['info'], size=8, opacity=0.6)
    ))
    
    # Custom model
    fig_custom.add_trace(go.Scatter(
        x=X_plot_custom, y=y_pred_custom,
        mode='lines',
        name=f'Your Model (degree={custom_degree})',
        line=dict(color=COLORS['primary'], width=4)
    ))
    
    fig_custom.update_layout(
        title=f"Your Custom Model: Polynomial Degree {custom_degree}",
        xaxis_title="X",
        yaxis_title="Y",
        height=500
    )
    
    st.plotly_chart(fig_custom, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Polynomial Degree", custom_degree)
    with col2:
        st.metric("Training MSE", f"{mse_custom:.4f}")
    with col3:
        st.metric("R² Score", f"{r2_custom:.4f}")
    
    # Provide feedback
    if custom_degree <= 2:
        st.warning("⚠️ Your model may be underfitting. Consider increasing complexity.")
    elif custom_degree >= 15:
        st.warning("⚠️ Your model may be overfitting. Consider reducing complexity.")
    else:
        st.success("✅ Your model complexity looks reasonable!")
    
    # Real-world implications
    st.markdown(sub_header("Real-World Implications", "🌍"), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Impact on AI Systems
        
        Understanding bias-variance tradeoff is critical for:
        
        **1. Model Selection**
        - Choose appropriate model complexity for your data
        - Balance interpretability with performance
        
        **2. Responsible AI**
        - Overfit models may amplify biases in training data
        - Underfit models may miss important patterns
        
        **3. Production Deployment**
        - Models must generalize to real-world data
        - Monitor for performance degradation over time
        
        **4. Regulatory Compliance**
        - Explainable models often preferred
        - Need to justify model complexity choices
        """)
    
    with col2:
        st.markdown("""
        ### AWS Tools for Model Optimization
        
        **Amazon SageMaker Automatic Model Tuning**
        - Automatically finds optimal hyperparameters
        - Reduces risk of overfitting/underfitting
        
        **Amazon SageMaker Debugger**
        - Monitors training metrics in real-time
        - Detects overfitting early
        
        **Amazon SageMaker Model Monitor**
        - Tracks model performance in production
        - Alerts when model degrades
        
        **Amazon SageMaker Clarify**
        - Detects bias in models
        - Ensures fair predictions
        """)
    
    # Best practices
    st.markdown(sub_header("Best Practices", "✨", "aws"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### During Training
        
        - Use train/validation/test splits
        - Apply cross-validation
        - Monitor both training and validation metrics
        - Use early stopping
        - Implement regularization
        - Start simple, increase complexity gradually
        """)
    
    with col2:
        st.markdown("""
        #### Model Selection
        
        - Compare multiple model complexities
        - Use validation set for selection
        - Consider ensemble methods
        - Balance performance with interpretability
        - Document model selection rationale
        - Test on held-out data
        """)
    
    with col3:
        st.markdown("""
        #### In Production
        
        - Monitor model performance continuously
        - Track prediction distributions
        - Detect data drift
        - Retrain periodically
        - A/B test model updates
        - Maintain model versioning
        """)
    
    # Summary
    st.markdown(sub_header("Summary", "📝", "minimal"), unsafe_allow_html=True)
    
    st.markdown("""
    ### Key Takeaways
    
    1. **Bias-Variance Tradeoff**: Finding the right model complexity is crucial for good generalization
    
    2. **Underfitting (High Bias)**: Model too simple → poor performance on both training and test data
    
    3. **Overfitting (High Variance)**: Model too complex → excellent training performance, poor test performance
    
    4. **Appropriate Fitting**: Balanced model → good performance on both training and test data
    
    5. **Responsible AI**: Understanding these concepts helps build fair, reliable, and trustworthy AI systems
    
    6. **AWS Tools**: Leverage SageMaker services to optimize models and monitor performance
    """)
    
    st.info("""
    💡 **Remember**: The goal is not to achieve perfect training accuracy, but to build models 
    that generalize well to new, unseen data while maintaining fairness and reliability.
    """)

if __name__ == "__main__":
    if 'localhost' in st.context.headers.get("host", ""):
        main()
    else:
        if authenticate.login():
            main()

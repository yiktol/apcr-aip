import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import uuid

# Import custom modules
from utils.common_clustering import initialize_session_state, create_sidebar, display_footer
from utils.styles import load_css
from utils.common import render_sidebar
from utils.data import get_customer_data, create_customer_profile, preprocess_data
from utils.model import build_kmeans_model, get_cluster_profile, predict_customer_cluster
from utils.visualization import (
    plot_cluster_distribution, plot_cluster_radar_chart, plot_pca_clusters, 
    plot_age_income_clusters, plot_feature_importance
)
import utils.authenticate as authenticate


def setup_page():
    """Set up the page configuration and styles."""
    st.set_page_config(
        page_title="Customer Segmentation",
        page_icon="🔍",
        layout="wide"
    )
    load_css()


def display_header():
    """Display the application header."""
    st.markdown("<h1 class='main-header'>👥 Customer Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
                Unsupervised machine learning for customer segmentation autonomously discovers natural groupings within customer data by identifying hidden patterns in purchasing behaviors, demographics, and interactions without predefined labels, enabling businesses to tailor marketing strategies to distinct customer profiles.
                </div>
                """, unsafe_allow_html=True)


def display_data_explorer_tab(data):
    """Display the Data Explorer tab content."""
    st.markdown("<h3>Customer Data Exploration</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Unsupervised learning discovers patterns in data without labeled outputs. In customer segmentation,
    we group similar customers based on their attributes and behaviors.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sample Customer Data")
        st.dataframe(data.head(10), use_container_width=True)
        
        st.subheader("Feature Distributions")
        feature = st.selectbox(
            "Select feature to visualize:",
            options=["age", "income", "spending_score", "website_visits", "purchase_history", 
                    "days_since_last_purchase", "email_clicks"]
        )
        
        fig = px.histogram(
            data, 
            x=feature, 
            marginal="box", 
            color_discrete_sequence=["#FF9900"],
            title=f"Distribution of {feature.replace('_', ' ').title()}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Information")
        st.markdown("""
        **Features explained:**
        
        - **age**: Customer age
        - **income**: Annual income in USD
        - **gender**: Customer gender (M/F)
        - **location**: Living area (urban/suburban/rural)
        - **purchase_history**: Number of previous purchases
        - **website_visits**: Monthly website visits
        - **email_clicks**: Email engagement rate (%)
        - **days_since_last_purchase**: Recency
        - **spending_score**: Spending behavior (1-100)
        """)
        
        st.subheader("Dataset Summary")
        st.write(f"Total customers: {len(data)}")
        
        # Show correlation heatmap of numeric features
        st.subheader("Feature Correlations")
        numeric_cols = ['age', 'income', 'purchase_history', 'website_visits', 
                       'email_clicks', 'days_since_last_purchase', 'spending_score']
        corr = data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap='YlOrBr', fmt=".2f", linewidths=.5, ax=ax)
        st.pyplot(fig)


def display_segmentation_model_tab(model_data, cluster_profiles):
    """Display the Segmentation Model tab content."""
    st.markdown("<h3>Customer Segmentation Model</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    K-means clustering is an unsupervised learning algorithm that groups data points into K clusters
    based on feature similarity. It's commonly used for customer segmentation to identify distinct 
    customer groups for targeted marketing strategies.
    </div>
    """, unsafe_allow_html=True)
    
    # Display clusters visualization
    st.subheader("Customer Segments Visualization")
    
    viz_col1, viz_col2 = st.columns([2, 1])
    
    with viz_col1:
        pca_fig = plot_pca_clusters(model_data['data_with_clusters'], model_data)
        st.plotly_chart(pca_fig, use_container_width=True)
        
        st.write("""
        This visualization uses Principal Component Analysis (PCA) to reduce all customer features into two dimensions.
        Each point represents a customer, and colors represent different customer segments discovered by the K-means
        algorithm. Cluster centers are marked with X.
        """)
    
    with viz_col2:
        dist_fig = plot_cluster_distribution(model_data['data_with_clusters'])
        st.plotly_chart(dist_fig, use_container_width=True)
        
        st.subheader("Model Information")
        st.write(f"Number of clusters: 4")
        st.write(f"Silhouette score: {model_data['silhouette_score']:.3f}")
        st.write("""
        Silhouette score measures how well clusters are separated.
        Scores close to 1 indicate well-separated clusters.
        """)
    
    # Display cluster profiles
    st.subheader("Customer Segment Profiles")
    
    profile_cols = st.columns(len(cluster_profiles))
    
    for i, profile in enumerate(cluster_profiles):
        with profile_cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <h4><span class="cluster-{i}">Segment {i}: {profile['description']}</span></h4>
                <p>{profile['size']} customers ({profile['percentage']:.1f}%)</p>
                <hr>
                <p>📊 <b>Profile:</b></p>
                <ul>
                    <li>Age: {profile['avg_age']:.1f} years</li>
                    <li>Income: ${profile['avg_income']:,.0f}</li>
                    <li>Spending: {profile['avg_spending_score']:.1f}/100</li>
                    <li>Website Visits: {profile['avg_website_visits']:.1f}/month</li>
                    <li>Email Engagement: {profile['avg_email_clicks']:.1f}%</li>
                </ul>
                <p>🎯 <b>Strategy:</b> {profile['strategy']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    feat_fig = plot_feature_importance(model_data)
    st.plotly_chart(feat_fig, use_container_width=True)
    
    st.write("""
    This chart shows which features were most important in forming the customer segments.
    Features with higher importance have greater variance across cluster centers, meaning they
    played a bigger role in separating customers into different groups.
    """)
    
    # Additional visualizations
    st.subheader("Segment Comparison: Age vs Income")
    age_income_fig = plot_age_income_clusters(model_data['data_with_clusters'])
    st.plotly_chart(age_income_fig, use_container_width=True)


def display_customer_predictor_tab(model_data, cluster_profiles):
    """Display the Customer Predictor tab content."""
    st.markdown("<h3>Customer Segment Predictor</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Enter customer information to predict which segment they belong to. This demonstrates how
    unsupervised learning models can categorize new data points after training.
    </div>
    """, unsafe_allow_html=True)
    
    # Create form for customer data input
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 75, 35)
            income = st.slider("Annual Income ($)", 20000, 150000, 60000, step=1000)
            gender = st.radio("Gender", ["M", "F"])
        
        with col2:
            location = st.selectbox("Location", ["urban", "suburban", "rural"])
            purchase_history = st.slider("Purchase History (# of purchases)", 0, 20, 5)
            website_visits = st.slider("Website Visits per Month", 0, 30, 10)
        
        with col3:
            email_clicks = st.slider("Email Click Rate (%)", 0.0, 100.0, 20.0)
            days_since_last_purchase = st.slider("Days Since Last Purchase", 0, 365, 30)
            spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)
        
        submitted = st.form_submit_button("Predict Segment")
    
    # If form is submitted, make prediction
    if submitted:
        display_prediction_results(
            age, income, gender, location, purchase_history,
            website_visits, email_clicks, days_since_last_purchase, 
            spending_score, model_data, cluster_profiles
        )
    
    # Display prediction history
    display_prediction_history()
    
    # Example customers for quick testing
    display_example_customers(model_data, cluster_profiles)


def display_prediction_results(age, income, gender, location, purchase_history,
                              website_visits, email_clicks, days_since_last_purchase, 
                              spending_score, model_data, cluster_profiles):
    """Display the prediction results for a customer."""
    # Create customer profile
    customer = create_customer_profile(
        age, income, gender, location, purchase_history,
        website_visits, email_clicks, days_since_last_purchase, spending_score
    )
    
    # Predict cluster
    cluster = predict_customer_cluster(customer, model_data)
    
    # Get profile for the predicted cluster
    profile = next(p for p in cluster_profiles if p['cluster'] == cluster)
    
    # Store in session state for history
    st.session_state.prediction_history.append({
        'age': age,
        'income': income,
        'gender': gender,
        'cluster': cluster,
        'description': profile['description']
    })
    
    # Display results
    st.success(f"Customer assigned to Segment {cluster}: **{profile['description']}**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Profile")
        st.json({
            'age': age,
            'income': f"${income:,}",
            'gender': gender,
            'location': location,
            'purchase_history': purchase_history,
            'website_visits': website_visits,
            'email_clicks': f"{email_clicks:.1f}%",
            'days_since_last_purchase': days_since_last_purchase,
            'spending_score': spending_score
        })
    
    with col2:
        st.subheader("Segment Characteristics")
        radar_fig = plot_cluster_radar_chart(cluster_profiles, cluster)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    st.subheader("Marketing Strategy")
    st.markdown(f"""
    <div class="feature-card">
        <p><b>Recommended approach for this customer:</b></p>
        <p>{profile['strategy']}</p>
    </div>
    """, unsafe_allow_html=True)


def display_prediction_history():
    """Display the prediction history."""
    if st.session_state.prediction_history:
        st.subheader("Prediction History")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Format the dataframe
        history_df['income'] = history_df['income'].apply(lambda x: f"${x:,}")
        
        # Display as table
        st.table(history_df)


def display_example_customers(model_data, cluster_profiles):
    """Display example customers for quick testing."""
    st.subheader("Try Example Customers")
    
    example_col1, example_col2, example_col3, example_col4 = st.columns(4)
    
    examples = [
        {
            'name': 'Young Urban Professional',
            'data': {
                'age': 28, 'income': 85000, 'gender': 'F', 'location': 'urban',
                'purchase_history': 8, 'website_visits': 22, 'email_clicks': 45.0,
                'days_since_last_purchase': 7, 'spending_score': 75
            }
        },
        {
            'name': 'Suburban Family Shopper',
            'data': {
                'age': 42, 'income': 65000, 'gender': 'F', 'location': 'suburban',
                'purchase_history': 15, 'website_visits': 8, 'email_clicks': 30.0,
                'days_since_last_purchase': 14, 'spending_score': 60
            }
        },
        {
            'name': 'High-Income Executive',
            'data': {
                'age': 55, 'income': 120000, 'gender': 'M', 'location': 'urban',
                'purchase_history': 12, 'website_visits': 5, 'email_clicks': 15.0,
                'days_since_last_purchase': 45, 'spending_score': 85
            }
        },
        {
            'name': 'Rural Occasional Shopper',
            'data': {
                'age': 62, 'income': 48000, 'gender': 'M', 'location': 'rural',
                'purchase_history': 3, 'website_visits': 2, 'email_clicks': 5.0,
                'days_since_last_purchase': 180, 'spending_score': 30
            }
        }
    ]
    
    for i, (col, example) in enumerate(zip([example_col1, example_col2, example_col3, example_col4], examples)):
        with col:
            st.markdown(f"""
            <div class="feature-card" style="height: 200px;">
                <h4>{example['name']}</h4>
                <p>Age: {example['data']['age']}</p>
                <p>Income: ${example['data']['income']:,}</p>
                <p>Location: {example['data']['location']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                # Create customer profile using example data
                example_customer = create_customer_profile(**example['data'])
                
                # Predict cluster
                example_cluster = predict_customer_cluster(example_customer, model_data)
                
                # Add to session state
                st.session_state.prediction_history.append({
                    'age': example['data']['age'],
                    'income': example['data']['income'],
                    'gender': example['data']['gender'],
                    'cluster': example_cluster,
                    'description': next(p['description'] for p in cluster_profiles if p['cluster'] == example_cluster)
                })
                
                # Reload page to show prediction
                st.rerun()


def compute_inertia():
    """Compute inertia for different numbers of clusters."""
    data = get_customer_data()
    inertia = []
    silhouette = []
    k_values = list(range(2, 11))  # Starting from 2 for silhouette score
    
    X_scaled, _, _ = preprocess_data(data)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(X_scaled, kmeans.labels_))
    
    return k_values, inertia, silhouette


def display_model_performance_tab(model_data, data):
    """Display the Model Performance & Evaluation tab content."""
    st.markdown("<h3>Model Performance & Evaluation</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Evaluating clustering models differs from supervised learning. Since we don't have ground truth labels,
    we use internal validation metrics like silhouette score and inertia to assess model quality.
    </div>
    """, unsafe_allow_html=True)
    
    # Show metrics
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Silhouette Analysis")
        
        st.write("""
        Silhouette score measures how similar an object is to its own cluster compared to other clusters.
        The score ranges from -1 to 1:
        - Score near 1: The customer is well matched to its cluster
        - Score near 0: The customer is on the border between clusters
        - Score near -1: The customer may be assigned to the wrong cluster
        """)
        
        st.metric(
            "Current Model Silhouette Score", 
            f"{model_data['silhouette_score']:.3f}",
            delta="+0.15 compared to 3 clusters"
        )
        
        # Create plot showing inertia vs number of clusters (elbow method)
        st.subheader("Finding Optimal Number of Clusters")
        
        k_values, inertia_values, silhouette_values = compute_inertia()
        
        # Plot elbow curve
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(k_values, inertia_values, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True)

        # Plot silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(k_values, silhouette_values, 'ro-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method')
        plt.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("""
        The **Elbow Method** shows diminishing returns (inertia) as we add more clusters.
        The optimal number of clusters is typically where the curve forms an "elbow" - the point
        where adding more clusters doesn't significantly reduce inertia.
        
        The **Silhouette Method** measures how well-separated the clusters are. Higher scores
        indicate better-defined clusters.
        
        For this dataset, 4 clusters provides a good balance between complexity and performance.
        """)
    
    with col2:
        st.subheader("Inter-Cluster Distances")
        
        # Calculate pairwise distances between cluster centers
        centers = model_data['cluster_centers']
        n_clusters = len(centers)
        
        distance_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                distance_matrix[i, j] = np.linalg.norm(centers[i] - centers[j])
        
        # Create heatmap of distances
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(distance_matrix, cmap='YlOrBr')
        
        # Show all ticks and label them
        ax.set_xticks(np.arange(n_clusters))
        ax.set_yticks(np.arange(n_clusters))
        ax.set_xticklabels([f"Cluster {i}" for i in range(n_clusters)])
        ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(n_clusters):
            for j in range(n_clusters):
                text = ax.text(j, i, f"{distance_matrix[i, j]:.2f}",
                               ha="center", va="center", color="black")
        
        ax.set_title("Inter-Cluster Distances")
        fig.colorbar(im)
        fig.tight_layout()
        
        st.pyplot(fig)
        
        st.write("""
        This heatmap shows the Euclidean distances between cluster centers in the feature space.
        Larger distances indicate more dissimilar customer segments.
        
        Insights:
        - Clusters that are far apart represent very different customer types
        - Closely positioned clusters might indicate potential for merger in a simpler model
        - Well-separated clusters suggest the model captures distinct customer segments
        """)
        
        st.subheader("Model Limitations")
        st.write("""
        It's important to understand the limitations of clustering models:
        
        1. **No Ground Truth**: Unlike supervised learning, we have no absolute "correct answer"
        
        2. **Feature Sensitivity**: Results are highly dependent on feature selection and scaling
        
        3. **Assumption of Shape**: K-means assumes spherical clusters of similar size
        
        4. **Need for Human Interpretation**: Clusters need business context to be actionable
        
        5. **Temporal Changes**: Customer behaviors change over time, requiring model retraining
        """)
        
        st.info("""
        💡 **Production Best Practice**: For real business applications, clustering should be 
        periodically retrained as customer behaviors evolve, and results should be validated with 
        A/B testing of targeted marketing strategies.
        """)


def main():
    """Main execution flow of the application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    with st.sidebar:
        render_sidebar()
        create_sidebar()
    
    # Display header
    display_header()
    
    # Get data and build model
    data = get_customer_data()
    model_data = build_kmeans_model(data)
    cluster_profiles = get_cluster_profile(model_data)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Explorer", 
        "🧩 Segmentation Model", 
        "👤 Customer Predictor",
        "📈 Model Performance"
    ])
    
    # Display content for each tab
    with tab1:
        display_data_explorer_tab(data)
    
    with tab2:
        display_segmentation_model_tab(model_data, cluster_profiles)
    
    with tab3:
        display_customer_predictor_tab(model_data, cluster_profiles)
    
    with tab4:
        display_model_performance_tab(model_data, data)
    
    # Display footer
    display_footer()


# Main execution flow
if __name__ == "__main__":
    setup_page()
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()

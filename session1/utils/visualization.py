import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

def plot_cluster_distribution(data_with_clusters):
    """Create pie chart showing cluster distribution"""
    cluster_counts = data_with_clusters['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    fig = px.pie(
        cluster_counts, 
        values='Count', 
        names='Cluster',
        title='Customer Segment Distribution',
        color_discrete_sequence=['#FF9900', '#232F3E', '#1DC7EA', '#87D068']
    )
    fig.update_traces(textinfo='percent+label')
    
    return fig

def plot_cluster_radar_chart(cluster_profiles, selected_cluster):
    """Create radar chart for selected cluster's attributes"""
    profile = next(p for p in cluster_profiles if p['cluster'] == selected_cluster)
    
    # Normalized values for radar chart (0-1 scale)
    max_values = {
        'Income': 150000, 
        'Spending': 100,
        'Website Visits': 30,
        'Email Engagement': 100,
        'Purchase Frequency': 365  # Inverse of days since purchase
    }
    
    categories = ['Income', 'Spending', 'Website Visits', 'Email Engagement', 'Purchase Frequency']
    values = [
        profile['avg_income'] / max_values['Income'],
        profile['avg_spending_score'] / max_values['Spending'],
        profile['avg_website_visits'] / max_values['Website Visits'],
        profile['avg_email_clicks'] / max_values['Email Engagement'],
        1 - (profile['avg_days_since_purchase'] / max_values['Purchase Frequency'])  # Inverse
    ]
    
    # Add first point at the end to close the loop
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    # Add radar chart
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=f"Cluster {selected_cluster}",
        line_color=['#FF9900', '#232F3E', '#1DC7EA', '#87D068'][selected_cluster % 4]
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f"Cluster {selected_cluster} Profile: {profile['description']}",
        showlegend=False
    )
    
    return fig

def plot_pca_clusters(data_with_clusters, model_data):
    """Create PCA plot showing clusters in 2D"""
    X_scaled = model_data['scaler'].transform(
        data_with_clusters[model_data['features']]
    )
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': data_with_clusters['cluster'].astype(str)
    })
    
    # Calculate cluster centers in PCA space
    centers = pca.transform(model_data['cluster_centers'])
    centers_df = pd.DataFrame({
        'PC1': centers[:, 0],
        'PC2': centers[:, 1],
        'Cluster': [str(i) for i in range(len(centers))]
    })
    
    # Create Plotly scatter plot
    fig = px.scatter(
        pca_df, x='PC1', y='PC2', color='Cluster',
        title='Customer Segments Visualization (PCA)',
        color_discrete_map={
            '0': '#FF9900', 
            '1': '#232F3E',
            '2': '#1DC7EA',
            '3': '#87D068'
        }
    )
    
    # Add cluster centers
    for i, row in centers_df.iterrows():
        fig.add_annotation(
            x=row['PC1'], y=row['PC2'],
            text=f"Cluster {row['Cluster']}",
            showarrow=True,
            arrowhead=1,
            ax=0, ay=-40
        )
        
        fig.add_traces(
            go.Scatter(
                x=[row['PC1']], 
                y=[row['PC2']], 
                mode='markers',
                marker=dict(
                    color='black',
                    size=15,
                    symbol='x'
                ),
                name=f"Center {row['Cluster']}"
            )
        )
    
    fig.update_layout(
        legend_title="Customer Segment",
        xaxis_title=f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
    )
    
    return fig

def plot_age_income_clusters(data_with_clusters):
    """Create scatter plot of age vs income colored by cluster"""
    fig = px.scatter(
        data_with_clusters,
        x='age',
        y='income',
        color='cluster',
        hover_data=['spending_score', 'purchase_history', 'website_visits'],
        title='Customer Segmentation: Age vs Income',
        color_discrete_map={
            0: '#FF9900', 
            1: '#232F3E',
            2: '#1DC7EA',
            3: '#87D068'
        }
    )
    
    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Income",
        legend_title="Customer Segment"
    )
    
    return fig

def plot_feature_importance(model_data):
    """Create bar chart showing feature importance"""
    # Calculate variance of cluster centers for each feature
    cluster_centers = model_data['cluster_centers']
    feature_importance = np.var(cluster_centers, axis=0)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': model_data['features'],
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        importance_df,
        x='Feature',
        y='Importance',
        title='Feature Importance in Cluster Formation',
        color_discrete_sequence=['#FF9900']
    )
    
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Importance (Variance)",
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig
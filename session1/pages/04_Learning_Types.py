import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, silhouette_score
import utils.authenticate as authenticate
from utils.common import render_sidebar
from utils.styles import load_css, sub_header, AWS_COLORS

# Page config
st.set_page_config(
    page_title="Learning Types",
    page_icon="🎓",
    layout="wide"
)

def init_session_state():
    """Initialize session state variables"""
    if 'learning_type_selected' not in st.session_state:
        st.session_state.learning_type_selected = None
    if 'interactive_step' not in st.session_state:
        st.session_state.interactive_step = 0
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False

@st.cache_data
def generate_supervised_data(n_samples=300):
    """Generate data for supervised learning demonstration"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.1,
        random_state=42
    )
    return X, y

@st.cache_data
def generate_unsupervised_data(n_samples=300):
    """Generate data for unsupervised learning demonstration"""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )
    return X, y

@st.cache_data
def generate_semi_supervised_data(n_samples=300, labeled_ratio=0.1):
    """Generate data for semi-supervised learning demonstration"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.05,
        random_state=42
    )
    
    # Create semi-supervised scenario: only label a small portion
    n_labeled = int(n_samples * labeled_ratio)
    y_semi = y.copy()
    unlabeled_indices = np.random.choice(n_samples, n_samples - n_labeled, replace=False)
    y_semi[unlabeled_indices] = -1  # -1 indicates unlabeled
    
    return X, y, y_semi

def plot_supervised_learning(X, y, show_model=False):
    """Create visualization for supervised learning"""
    fig = go.Figure()
    
    # Plot data points
    colors = ['red' if label == 0 else 'blue' for label in y]
    labels = ['Class 0 (Spam)' if label == 0 else 'Class 1 (Not Spam)' for label in y]
    
    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(
            color=colors,
            size=8,
            line=dict(color='white', width=1)
        ),
        text=labels,
        name='Data Points',
        showlegend=False
    ))
    
    if show_model:
        # Train a simple model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Create decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            showscale=False,
            opacity=0.3,
            colorscale=[[0, 'rgba(255,0,0,0.3)'], [1, 'rgba(0,0,255,0.3)']],
            name='Decision Boundary'
        ))
    
    fig.update_layout(
        title='Supervised Learning: Email Classification',
        xaxis_title='Feature 1 (e.g., Word Frequency)',
        yaxis_title='Feature 2 (e.g., Link Count)',
        height=500,
        plot_bgcolor='white',
        showlegend=True
    )
    
    return fig

def plot_unsupervised_learning(X, y, show_clusters=False):
    """Create visualization for unsupervised learning"""
    fig = go.Figure()
    
    if not show_clusters:
        # Show unlabeled data
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                color='gray',
                size=8,
                line=dict(color='white', width=1)
            ),
            name='Unlabeled Data',
            text=['Unknown Group'] * len(X)
        ))
    else:
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        colors_map = {0: 'red', 1: 'blue', 2: 'green'}
        colors = [colors_map[c] for c in clusters]
        
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                color=colors,
                size=8,
                line=dict(color='white', width=1)
            ),
            name='Clustered Data',
            text=[f'Cluster {c}' for c in clusters]
        ))
        
        # Add cluster centers
        centers = kmeans.cluster_centers_
        fig.add_trace(go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode='markers',
            marker=dict(
                color='black',
                size=15,
                symbol='x',
                line=dict(color='white', width=2)
            ),
            name='Cluster Centers'
        ))
    
    fig.update_layout(
        title='Unsupervised Learning: Customer Segmentation',
        xaxis_title='Feature 1 (e.g., Purchase Frequency)',
        yaxis_title='Feature 2 (e.g., Average Spend)',
        height=500,
        plot_bgcolor='white',
        showlegend=True
    )
    
    return fig

def plot_semi_supervised_learning(X, y_true, y_semi, show_propagation=False):
    """Create visualization for semi-supervised learning"""
    fig = go.Figure()
    
    # Separate labeled and unlabeled data
    labeled_mask = y_semi != -1
    unlabeled_mask = y_semi == -1
    
    if not show_propagation:
        # Show initial state: few labeled, many unlabeled
        if labeled_mask.any():
            colors_labeled = ['red' if label == 0 else 'blue' for label in y_semi[labeled_mask]]
            fig.add_trace(go.Scatter(
                x=X[labeled_mask, 0],
                y=X[labeled_mask, 1],
                mode='markers',
                marker=dict(
                    color=colors_labeled,
                    size=12,
                    line=dict(color='white', width=2),
                    symbol='star'
                ),
                name='Labeled Data',
                text=[f'Class {int(label)}' for label in y_semi[labeled_mask]]
            ))
        
        if unlabeled_mask.any():
            fig.add_trace(go.Scatter(
                x=X[unlabeled_mask, 0],
                y=X[unlabeled_mask, 1],
                mode='markers',
                marker=dict(
                    color='lightgray',
                    size=8,
                    line=dict(color='gray', width=1)
                ),
                name='Unlabeled Data',
                text=['Unknown'] * unlabeled_mask.sum()
            ))
    else:
        # Show after label propagation
        model = LabelPropagation(kernel='knn', n_neighbors=7)
        model.fit(X, y_semi)
        y_pred = model.predict(X)
        
        colors = ['red' if label == 0 else 'blue' for label in y_pred]
        symbols = ['star' if labeled else 'circle' for labeled in labeled_mask]
        sizes = [12 if labeled else 8 for labeled in labeled_mask]
        
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                color=colors,
                size=sizes,
                symbol=symbols,
                line=dict(color='white', width=1)
            ),
            name='All Data',
            text=[f'Class {int(label)} ({"Labeled" if labeled else "Propagated"})' 
                  for label, labeled in zip(y_pred, labeled_mask)]
        ))
    
    fig.update_layout(
        title='Semi-Supervised Learning: Medical Diagnosis',
        xaxis_title='Feature 1 (e.g., Blood Pressure)',
        yaxis_title='Feature 2 (e.g., Cholesterol Level)',
        height=500,
        plot_bgcolor='white',
        showlegend=True
    )
    
    return fig

def display_comparison_table():
    """Display comparison table of learning types"""
    comparison_data = {
        'Aspect': [
            'Training Data',
            'Labels Required',
            'Goal',
            'Common Use Cases',
            'Algorithms',
            'Evaluation',
            'Cost',
            'Accuracy'
        ],
        'Supervised': [
            'Labeled data',
            'Yes, all data labeled',
            'Predict labels for new data',
            'Classification, Regression',
            'Linear Regression, Decision Trees, Neural Networks',
            'Accuracy, Precision, Recall',
            'High (labeling cost)',
            'Generally highest'
        ],
        'Unsupervised': [
            'Unlabeled data',
            'No labels needed',
            'Find patterns/structure',
            'Clustering, Dimensionality Reduction',
            'K-Means, DBSCAN, PCA',
            'Silhouette Score, Elbow Method',
            'Low (no labeling)',
            'No ground truth'
        ],
        'Semi-Supervised': [
            'Mostly unlabeled + few labeled',
            'Partial labels (10-30%)',
            'Leverage both labeled & unlabeled',
            'When labeling is expensive',
            'Label Propagation, Self-Training',
            'Accuracy on labeled subset',
            'Medium (some labeling)',
            'Between supervised & unsupervised'
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    return df

def interactive_supervised_demo():
    """Interactive demonstration of supervised learning"""
    st.markdown(sub_header("🎯 Interactive Demo: Supervised Learning", "🎯"), unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <strong>Scenario:</strong> Email Spam Detection<br><br>
        You have a dataset of emails, each labeled as "Spam" or "Not Spam". 
        The model learns from these labeled examples to classify new emails.
        </div>
        """, unsafe_allow_html=True)
        
        n_samples = st.slider("Number of training samples:", 100, 500, 300, 50)
        show_model = st.checkbox("Show trained model decision boundary", value=False)
        
        if show_model:
            st.success("✅ Model trained! The shaded regions show how the model classifies different areas.")
        else:
            st.info("👆 Check the box above to train the model and see the decision boundary.")
    
    with col2:
        X, y = generate_supervised_data(n_samples)
        fig = plot_supervised_learning(X, y, show_model)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show metrics if model is trained
    if show_model:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Samples", len(X_train))
        col2.metric("Test Samples", len(X_test))
        col3.metric("Accuracy", f"{accuracy:.2%}")

def interactive_unsupervised_demo():
    """Interactive demonstration of unsupervised learning"""
    st.markdown(sub_header("🔍 Interactive Demo: Unsupervised Learning", "🔍"), unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <strong>Scenario:</strong> Customer Segmentation<br><br>
        You have customer data but no predefined groups. The algorithm discovers 
        natural groupings based on purchasing behavior and demographics.
        </div>
        """, unsafe_allow_html=True)
        
        n_samples = st.slider("Number of customers:", 100, 500, 300, 50, key='unsup_samples')
        show_clusters = st.checkbox("Discover customer segments", value=False)
        
        if show_clusters:
            st.success("✅ Segments discovered! Customers are grouped by similar characteristics.")
        else:
            st.info("👆 Check the box above to run clustering and discover customer segments.")
    
    with col2:
        X, y = generate_unsupervised_data(n_samples)
        fig = plot_unsupervised_learning(X, y, show_clusters)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show metrics if clustering is performed
    if show_clusters:
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, clusters)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(X))
        col2.metric("Segments Found", 3)
        col3.metric("Silhouette Score", f"{silhouette:.3f}")
        
        st.info("💡 Silhouette Score measures cluster quality. Values closer to 1 indicate well-separated clusters.")

def interactive_semi_supervised_demo():
    """Interactive demonstration of semi-supervised learning"""
    st.markdown(sub_header("🎓 Interactive Demo: Semi-Supervised Learning", "🎓"), unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <strong>Scenario:</strong> Medical Diagnosis<br><br>
        You have many patient records, but only a few have confirmed diagnoses (expensive tests). 
        Semi-supervised learning uses the few labeled cases to help classify the unlabeled ones.
        </div>
        """, unsafe_allow_html=True)
        
        n_samples = st.slider("Number of patient records:", 100, 500, 300, 50, key='semi_samples')
        labeled_ratio = st.slider("Percentage of labeled data:", 5, 30, 10, 5) / 100
        show_propagation = st.checkbox("Propagate labels to unlabeled data", value=False)
        
        if show_propagation:
            st.success(f"✅ Labels propagated! Using {int(labeled_ratio*100)}% labeled data to classify the rest.")
        else:
            st.info("👆 Check the box above to see how labels spread to unlabeled data.")
    
    with col2:
        X, y_true, y_semi = generate_semi_supervised_data(n_samples, labeled_ratio)
        fig = plot_semi_supervised_learning(X, y_true, y_semi, show_propagation)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show metrics
    n_labeled = (y_semi != -1).sum()
    n_unlabeled = (y_semi == -1).sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Labeled Records", n_labeled)
    col2.metric("Unlabeled Records", n_unlabeled)
    
    if show_propagation:
        model = LabelPropagation(kernel='knn', n_neighbors=7)
        model.fit(X, y_semi)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        col3.metric("Accuracy", f"{accuracy:.2%}")
    else:
        col3.metric("Labeling Cost Saved", f"{int((1-labeled_ratio)*100)}%")

def display_quiz():
    """Display knowledge check quiz"""
    st.markdown(sub_header("📝 Knowledge Check", "📝"), unsafe_allow_html=True)
    
    questions = [
        {
            'question': 'Which learning type requires all training data to be labeled?',
            'options': ['Supervised Learning', 'Unsupervised Learning', 'Semi-Supervised Learning', 'Reinforcement Learning'],
            'correct': 0,
            'explanation': 'Supervised learning requires all training data to have labels so the model can learn the relationship between features and labels.'
        },
        {
            'question': 'What is the main goal of unsupervised learning?',
            'options': ['Predict future values', 'Find hidden patterns and structures', 'Classify data into predefined categories', 'Maximize rewards'],
            'correct': 1,
            'explanation': 'Unsupervised learning aims to discover hidden patterns, structures, or groupings in data without predefined labels.'
        },
        {
            'question': 'When is semi-supervised learning most useful?',
            'options': ['When all data is labeled', 'When no data is labeled', 'When labeling data is expensive or time-consuming', 'When you need real-time predictions'],
            'correct': 2,
            'explanation': 'Semi-supervised learning is ideal when you have a small amount of labeled data and a large amount of unlabeled data, typically because labeling is expensive.'
        },
        {
            'question': 'Which of these is an example of unsupervised learning?',
            'options': ['Email spam detection', 'Customer segmentation', 'House price prediction', 'Image classification'],
            'correct': 1,
            'explanation': 'Customer segmentation is unsupervised learning because it discovers natural groupings in customer data without predefined labels.'
        },
        {
            'question': 'What advantage does semi-supervised learning have over supervised learning?',
            'options': ['Always more accurate', 'Requires less labeled data', 'Faster training time', 'Works with any algorithm'],
            'correct': 1,
            'explanation': 'Semi-supervised learning can achieve good performance with much less labeled data, reducing the cost and time of data labeling.'
        }
    ]
    
    score = 0
    total = len(questions)
    
    for i, q in enumerate(questions):
        st.markdown(f"**Question {i+1}:** {q['question']}")
        
        answer = st.radio(
            f"Select your answer:",
            q['options'],
            key=f'quiz_q{i}',
            index=None
        )
        
        if answer is not None:
            st.session_state.quiz_answers[i] = q['options'].index(answer)
            
            if st.session_state.quiz_answers[i] == q['correct']:
                st.success(f"✅ Correct! {q['explanation']}")
                score += 1
            else:
                st.error(f"❌ Incorrect. {q['explanation']}")
        
        st.markdown("---")
    
    if len(st.session_state.quiz_answers) == total:
        st.markdown(f"""
        <div class='card' style='background-color: {AWS_COLORS['hover']}; text-align: center;'>
            <h3>Your Score: {score}/{total} ({score/total*100:.0f}%)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if score == total:
            st.balloons()
            st.success("🎉 Perfect score! You've mastered the learning types!")
        elif score >= total * 0.7:
            st.success("👍 Great job! You have a solid understanding of learning types.")
        else:
            st.info("📚 Review the material above and try again to improve your score.")

def main():
    """Main application function"""
    load_css()
    init_session_state()
    
    # Header
    st.markdown("<h1>🎓 Machine Learning: Types of Learning</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    Machine learning algorithms can be categorized into three main types based on how they learn from data:
    <strong>Supervised</strong>, <strong>Unsupervised</strong>, and <strong>Semi-Supervised</strong> learning.
    Each type has unique characteristics, use cases, and advantages.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎯 Supervised Learning",
        "🔍 Unsupervised Learning",
        "🎓 Semi-Supervised Learning",
        "📝 Knowledge Check"
    ])
    
    with tab1:
        st.markdown(sub_header("Comparison of Learning Types"), unsafe_allow_html=True)
        
        # Display comparison table
        df = display_comparison_table()
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visual comparison
        st.markdown(sub_header("Visual Comparison", "📈"), unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center;'>
                <h3>🎯 Supervised</h3>
                <p style='font-size: 3rem; margin: 1rem 0;'>100%</p>
                <p>Labeled Data</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Key Characteristics:**
            - All data is labeled
            - Learn input → output mapping
            - High accuracy when well-trained
            - Expensive data preparation
            
            **Examples:**
            - Spam detection
            - Image classification
            - Price prediction
            """)
        
        with col2:
            st.markdown("""
            <div class='card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; text-align: center;'>
                <h3>🔍 Unsupervised</h3>
                <p style='font-size: 3rem; margin: 1rem 0;'>0%</p>
                <p>Labeled Data</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Key Characteristics:**
            - No labels required
            - Discover hidden patterns
            - No ground truth for evaluation
            - Low data preparation cost
            
            **Examples:**
            - Customer segmentation
            - Anomaly detection
            - Topic modeling
            """)
        
        with col3:
            st.markdown("""
            <div class='card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; text-align: center;'>
                <h3>🎓 Semi-Supervised</h3>
                <p style='font-size: 3rem; margin: 1rem 0;'>10-30%</p>
                <p>Labeled Data</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Key Characteristics:**
            - Mix of labeled & unlabeled
            - Leverages both data types
            - Cost-effective approach
            - Good accuracy with less labels
            
            **Examples:**
            - Medical diagnosis
            - Web page classification
            - Speech recognition
            """)
        
        # When to use what
        st.markdown(sub_header("When to Use Each Type?", "🤔"), unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
        <h4>✅ Use Supervised Learning when:</h4>
        <ul>
            <li>You have plenty of labeled data</li>
            <li>You need high accuracy predictions</li>
            <li>The task has clear input-output relationships</li>
            <li>You can afford the labeling cost</li>
        </ul>
        </div>
        
        <div class='card'>
        <h4>✅ Use Unsupervised Learning when:</h4>
        <ul>
            <li>You have no labeled data</li>
            <li>You want to explore data structure</li>
            <li>You need to find hidden patterns</li>
            <li>Labeling is impossible or impractical</li>
        </ul>
        </div>
        
        <div class='card'>
        <h4>✅ Use Semi-Supervised Learning when:</h4>
        <ul>
            <li>You have limited labeled data</li>
            <li>Labeling is expensive or time-consuming</li>
            <li>You have abundant unlabeled data</li>
            <li>You want to balance cost and accuracy</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        interactive_supervised_demo()
    
    with tab3:
        interactive_unsupervised_demo()
    
    with tab4:
        interactive_semi_supervised_demo()
    
    with tab5:
        display_quiz()
    
    # Sidebar
    with st.sidebar:
        render_sidebar()
        
        with st.expander("📚 Learning Resources", expanded=False):
            st.markdown("""
            **Supervised Learning:**
            - Classification algorithms
            - Regression techniques
            - Model evaluation metrics
            
            **Unsupervised Learning:**
            - Clustering methods
            - Dimensionality reduction
            - Anomaly detection
            
            **Semi-Supervised Learning:**
            - Label propagation
            - Self-training methods
            - Co-training approaches
            """)

# Main execution
if __name__ == "__main__":
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        if authenticate.login():
            main()

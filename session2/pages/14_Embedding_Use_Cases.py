
import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import boto3
import math
import textwrap
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
import utils.authenticate as authenticate
import utils.common as common
from utils.styles import load_css

# Configure page settings
st.set_page_config(
    page_title="AWS Bedrock Embedding Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

load_css()


class EmbeddingHelper:
    """Helper class for embedding operations"""
    
    def __init__(self):
        self.bedrock = self._get_bedrock_client()
    
    def _get_bedrock_client(self):
        """Initialize Bedrock client"""
        return boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
    
    def get_embedding(self, text):
        """Get embedding for given text"""
        try:
            model_id = 'amazon.titan-embed-g1-text-02'
            input_data = {'inputText': text}
            
            response = self.bedrock.invoke_model(
                body=json.dumps(input_data),
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body['embedding']
        except Exception as e:
            st.error(f"Error getting embedding: {str(e)}")
            return None
    
    def calculate_distance(self, v1, v2):
        """Calculate Euclidean distance between two vectors"""
        return math.dist(v1, v2)
    
    def calculate_cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors"""
        return dot(v1, v2) / (norm(v1) * norm(v2))

def create_sidebar():
    """Create sidebar with app information"""
    with st.sidebar:
        
        common.render_sidebar()
        
        with st.expander("About this App", expanded=False):
            st.markdown("""
            ### 🧠 AWS Bedrock Embedding Playground
            
            This application demonstrates various machine learning techniques using 
            AWS Bedrock embeddings:
            
            **📊 Topics Covered:**
            - **Clustering**: Group similar items using K-means
            - **Search & Recommendation**: Find relevant content
            - **Anomaly Detection**: Identify outliers in data
            - **Classification**: Categorize text into classes
            
            **🛠️ Technology Stack:**
            - AWS Bedrock (Titan Embeddings)
            - Streamlit
            - scikit-learn
            - NumPy & Pandas
            
            **🎯 Use Cases:**
            - Document similarity
            - Content recommendation
            - Fraud detection
            - Sentiment analysis
            """)

def clustering_tab(helper):
    """Clustering functionality tab"""
    st.header("🔍 Clustering Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Clustering** groups similar items together without prior knowledge of categories. 
        Using K-means algorithm, we can separate physicists from musicians based on their 
        semantic similarity in the embedding space.
        """)
        
        with st.form("clustering_form"):
            names = st.multiselect(
                "Select Names for Clustering:",
                ['Albert Einstein', 'Bob Dylan', 'Elvis Presley', 'Isaac Newton', 
                 'Michael Jackson', 'Niels Bohr', 'Taylor Swift', 'Hank Williams', 
                 'Werner Heisenberg', 'Stevie Wonder', 'Marie Curie', 'Ernest Rutherford'],
                default=['Albert Einstein', 'Bob Dylan', 'Elvis Presley', 'Isaac Newton', 
                        'Michael Jackson', 'Niels Bohr']
            )
            
            n_clusters = st.slider("Number of Clusters", 2, 5, 2)
            submit = st.form_submit_button("🔄 Perform Clustering", type="primary")
        
        if submit and names:
            with st.spinner("Generating embeddings and clustering..."):
                # Get embeddings
                embeddings = []
                for name in names:
                    embedding = helper.get_embedding(name)
                    if embedding:
                        embeddings.append(embedding)
                
                if embeddings:
                    # Create DataFrame
                    df = pd.DataFrame({
                        'names': names[:len(embeddings)], 
                        'embeddings': embeddings
                    })
                    
                    # Perform clustering
                    matrix = np.vstack(df.embeddings.values)
                    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
                    df['cluster'] = kmeans.fit_predict(matrix)
                    
                    # Display results
                    st.success("Clustering completed!")
                    st.dataframe(df[['names', 'cluster']], use_container_width=True)
                    
                    # Visualization
                    if len(embeddings) > 1:
                        # Fix: Remove n_iter from __init__ and use max_iter instead
                        tsne = TSNE(random_state=0, perplexity=min(6, len(embeddings)-1))
                        tsne_results = tsne.fit_transform(matrix.astype(np.float32))
                        
                        df['tsne1'] = tsne_results[:, 0]
                        df['tsne2'] = tsne_results[:, 1]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='cluster', 
                                      s=100, ax=ax)
                        
                        for idx, row in df.iterrows():
                            ax.annotate(row['names'], (row['tsne1'], row['tsne2']), 
                                      fontsize=8, ha='center')
                        
                        plt.title('Clustering Visualization (t-SNE)')
                        st.pyplot(fig)
    
    with col2:
        st.code("""
# Clustering Example
import numpy as np
from sklearn.cluster import KMeans

# Get embeddings for names
embeddings = []
for name in names:
    embedding = get_embedding(name)
    embeddings.append(embedding)

# Perform K-means clustering
matrix = np.vstack(embeddings)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(matrix)

# Results show grouping by profession
print(f"Cluster assignments: {clusters}")
        """, language="python")

def search_tab(helper):
    """Search and recommendation functionality tab"""
    st.header("🔎 Search & Recommendation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Semantic Search** finds the most relevant document by comparing embeddings. 
        The system calculates distances between query and documents, returning the closest match.
        """)
        
        with st.form("search_form"):
            query = st.text_area("Enter your search query:", 
                               value="Isaac Newton", height=68)
            
            st.subheader("Document Collection:")
            doc1 = st.text_area("Document 1:", 
                              value="The theory of general relativity says that the observed gravitational effect between masses results from their warping of spacetime.")
            doc2 = st.text_area("Document 2:", 
                              value="Quantum mechanics allows the calculation of properties and behaviour of physical systems.")
            doc3 = st.text_area("Document 3:", 
                              value="Every particle attracts every other particle in the universe with a force proportional to the product of their masses.")
            doc4 = st.text_area("Document 4:", 
                              value="The electromagnetic spectrum is the range of frequencies of electromagnetic radiation.")
            
            submit = st.form_submit_button("🔍 Search Documents", type="primary")
        
        if submit and query.strip():
            with st.spinner("Searching documents..."):
                documents = [doc1, doc2, doc3, doc4]
                document_embeddings = []
                
                # Get embeddings for documents
                for doc in documents:
                    if doc.strip():
                        embedding = helper.get_embedding(doc)
                        if embedding:
                            document_embeddings.append(embedding)
                
                # Get query embedding
                query_embedding = helper.get_embedding(query)
                
                if query_embedding and document_embeddings:
                    # Calculate distances
                    distances = []
                    for doc_embedding in document_embeddings:
                        distance = helper.calculate_distance(query_embedding, doc_embedding)
                        distances.append(distance)
                    
                    # Find best match
                    best_match_idx = np.argmin(distances)
                    
                    st.success("Search completed!")
                    st.info(f"**Best Match:** Document {best_match_idx + 1}")
                    st.write(f"**Content:** {documents[best_match_idx]}")
                    
                    # Display distance table
                    df = pd.DataFrame({
                        'Document': [f'Document {i+1}' for i in range(len(distances))],
                        'Distance': distances,
                        'Similarity Rank': np.argsort(distances) + 1
                    })
                    st.dataframe(df, use_container_width=True)
    
    with col2:
        st.code("""
# Search Example
import math

def search_documents(query, documents):
    # Get embeddings
    query_embedding = get_embedding(query)
    doc_embeddings = [get_embedding(doc) 
                     for doc in documents]
    
    # Calculate distances
    distances = []
    for doc_emb in doc_embeddings:
        distance = math.dist(query_embedding, doc_emb)
        distances.append(distance)
    
    # Return best match
    best_idx = distances.index(min(distances))
    return documents[best_idx]

# Usage
result = search_documents(
    "Isaac Newton", 
    document_list
)
        """, language="python")

def anomaly_detection_tab(helper):
    """Anomaly detection functionality tab"""
    st.header("🚨 Anomaly Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Anomaly Detection** identifies outliers by measuring distance from the center of mass. 
        Items furthest from the group center are considered anomalies.
        """)
        
        with st.form("anomaly_form"):
            names = st.multiselect(
                "Select Names for Analysis:",
                ['Albert Einstein', 'Isaac Newton', 'Stephen Hawking', 'Galileo Galilei',
                 'Niels Bohr', 'Werner Heisenberg', 'Marie Curie', 'Ernest Rutherford',
                 'Michael Faraday', 'Richard Feynman', 'Lady Gaga', 'Erwin Schrödinger',
                 'Max Planck', 'Enrico Fermi', 'Taylor Swift', 'Lord Kelvin'],
                default=['Albert Einstein', 'Isaac Newton', 'Lady Gaga', 'Taylor Swift',
                        'Marie Curie', 'Werner Heisenberg']
            )
            
            detection_method = st.selectbox(
                "Detection Method:",
                ["By Count", "By Percentage", "By Distance Threshold"]
            )
            
            if detection_method == "By Count":
                threshold = st.slider("Number of Outliers", 1, 10, 2)
            elif detection_method == "By Percentage":
                threshold = st.slider("Percentage of Outliers", 1, 50, 10)
            else:
                threshold = st.slider("Distance Threshold (%)", 1, 100, 60)
            
            submit = st.form_submit_button("🔍 Detect Anomalies", type="primary")
        
        if submit and names:
            with st.spinner("Analyzing for anomalies..."):
                # Get embeddings
                dataset = []
                for name in names:
                    embedding = helper.get_embedding(name)
                    if embedding:
                        dataset.append({'name': name, 'embedding': embedding})
                
                if dataset:
                    # Calculate center of mass
                    embeddings = [item['embedding'] for item in dataset]
                    center = np.mean(embeddings, axis=0)
                    
                    # Calculate distances from center
                    for item in dataset:
                        item['distance'] = helper.calculate_distance(item['embedding'], center)
                    
                    # Sort by distance (descending)
                    dataset.sort(key=lambda x: x['distance'], reverse=True)
                    
                    # Determine outliers based on method
                    if detection_method == "By Count":
                        outliers = dataset[:threshold]
                    elif detection_method == "By Percentage":
                        count = max(1, int(len(dataset) * threshold / 100))
                        outliers = dataset[:count]
                    else:  # By Distance
                        max_distance = dataset[0]['distance']
                        min_threshold = threshold * max_distance / 100
                        outliers = [item for item in dataset if item['distance'] >= min_threshold]
                    
                    st.success("Anomaly detection completed!")
                    
                    if outliers:
                        outlier_df = pd.DataFrame({
                            'Name': [item['name'] for item in outliers],
                            'Distance from Center': [f"{item['distance']:.4f}" for item in outliers]
                        })
                        st.dataframe(outlier_df, use_container_width=True)
                    else:
                        st.info("No anomalies detected with current threshold.")
                    
                    # Show all distances for reference
                    with st.expander("View All Distances"):
                        all_df = pd.DataFrame({
                            'Name': [item['name'] for item in dataset],
                            'Distance': [f"{item['distance']:.4f}" for item in dataset],
                            'Status': ['Outlier' if item in outliers else 'Normal' 
                                     for item in dataset]
                        })
                        st.dataframe(all_df, use_container_width=True)
    
    with col2:
        st.code("""
# Anomaly Detection Example
import numpy as np

def find_outliers(dataset, count):
    # Calculate center of mass
    embeddings = [item['embedding'] 
                 for item in dataset]
    center = np.mean(embeddings, axis=0)
    
    # Calculate distances from center
    for item in dataset:
        distance = math.dist(
            item['embedding'], center
        )
        item['distance'] = distance
    
    # Sort by distance (desc) and return top N
    dataset.sort(
        key=lambda x: x['distance'], 
        reverse=True
    )
    return dataset[:count]

# Usage
outliers = find_outliers(data, 2)
        """, language="python")

def classification_tab(helper):
    """Classification functionality tab"""
    st.header("📊 Text Classification")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Classification** assigns text to predefined categories by finding the closest class 
        based on embedding similarity. Useful for sentiment analysis, topic categorization, etc.
        """)
        
        example_type = st.selectbox(
            "Choose Classification Example:",
            ["Student Talent Classification", "Sentiment Analysis"]
        )
        
        if example_type == "Student Talent Classification":
            with st.form("classification_form1"):
                query = st.text_area(
                    "Enter text to classify:",
                    value="Ellison sends a spell to prevent Professor Wang from entering the classroom",
                    height=80
                )
                
                st.subheader("Class Definitions:")
                class1 = st.text_input("Athletics Class:", 
                                      value="all students with a talent in sports")
                class2 = st.text_input("Musician Class:", 
                                      value="all students with a talent in music")
                class3 = st.text_input("Magician Class:", 
                                      value="all students with a talent in witch craft")
                
                submit = st.form_submit_button("🎯 Classify Text", type="primary")
                
                classes = [
                    {'name': 'athletics', 'description': class1},
                    {'name': 'musician', 'description': class2},
                    {'name': 'magician', 'description': class3}
                ]
        
        else:  # Sentiment Analysis
            with st.form("classification_form2"):
                query = st.text_area(
                    "Enter customer feedback:",
                    value="Steve helped me solve the problem in just a few minutes. Thank you for the great work!",
                    height=80
                )
                
                st.subheader("Sentiment Classes:")
                pos_class = st.text_input("Positive Class:", 
                                        value="customer demonstrated positive sentiment in the response")
                neg_class = st.text_input("Negative Class:", 
                                        value="customer demonstrated negative sentiment in the response")
                
                submit = st.form_submit_button("🎯 Analyze Sentiment", type="primary")
                
                classes = [
                    {'name': 'positive', 'description': pos_class},
                    {'name': 'negative', 'description': neg_class}
                ]
        
        if submit and query.strip():
            with st.spinner("Classifying text..."):
                # Get embeddings for classes
                for cls in classes:
                    embedding = helper.get_embedding(cls['description'])
                    if embedding:
                        cls['embedding'] = embedding
                
                # Get query embedding
                query_embedding = helper.get_embedding(query)
                
                if query_embedding and all('embedding' in cls for cls in classes):
                    # Calculate distances to each class
                    distances = []
                    for cls in classes:
                        distance = helper.calculate_distance(cls['embedding'], query_embedding)
                        cls['distance'] = distance
                        distances.append(distance)
                    
                    # Find best match
                    best_class = min(classes, key=lambda x: x['distance'])
                    
                    st.success("Classification completed!")
                    st.info(f"**Predicted Class:** {best_class['name'].title()}")
                    st.write(f"**Confidence Score:** {1/(1+best_class['distance']):.3f}")
                    
                    # Display distance table
                    df = pd.DataFrame({
                        'Class': [cls['name'].title() for cls in classes],
                        'Distance': [f"{cls['distance']:.4f}" for cls in classes],
                        'Confidence': [f"{1/(1+cls['distance']):.3f}" for cls in classes]
                    })
                    st.dataframe(df, use_container_width=True)
    
    with col2:
        st.code("""
# Classification Example
def classify_text(text, classes):
    # Get embeddings for classes
    for cls in classes:
        cls['embedding'] = get_embedding(
            cls['description']
        )
    
    # Get text embedding
    text_embedding = get_embedding(text)
    
    # Calculate distances
    for cls in classes:
        distance = math.dist(
            cls['embedding'], 
            text_embedding
        )
        cls['distance'] = distance
    
    # Return closest class
    best_class = min(classes, 
                    key=lambda x: x['distance'])
    return best_class['name']

# Usage
result = classify_text(
    "Great product!", 
    sentiment_classes
)
        """, language="python")

def main():
    """Main application function"""
    # Create helper instance
    common.initialize_session_state()
    
    helper = EmbeddingHelper()
    
    # Create sidebar
    create_sidebar()
    
    # Main header
    st.markdown("""
        <h1>🧠 Embedding Use Cases</h1>
        <div class="info-box">
        Explore Machine Learning with Semantic Embeddings
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Clustering", 
        "🔎 Search & Recommendation", 
        "🚨 Anomaly Detection", 
        "📊 Classification"
    ])
    
    with tab1:
        clustering_tab(helper)
    
    with tab2:
        search_tab(helper)
    
    with tab3:
        anomaly_detection_tab(helper)
    
    with tab4:
        classification_tab(helper)
    
    # Footer
    st.markdown("""
    <div class="footer">
        © 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:

        if 'localhost' in st.context.headers["host"]:
            main()
        else:
            # First check authentication
            is_authenticated = authenticate.login()
            
            # If authenticated, show the main app content
            if is_authenticated:
                main()

    except Exception as e:
        logger.critical(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
        
        # Provide debugging information in an expander
        with st.expander("Error Details"):
            st.code(str(e))
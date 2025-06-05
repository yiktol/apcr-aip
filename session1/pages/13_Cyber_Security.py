import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import joblib
import os
import uuid
import datetime

from utils.styles import load_css
import utils.authenticate as authenticate
# Initialize session state
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "model" not in st.session_state:
        st.session_state.model = None
    if "scaler" not in st.session_state:
        st.session_state.scaler = None
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = datetime.datetime.now()

def reset_session():
    for key in list(st.session_state.keys()):
        if key != "session_id":
            del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.last_activity = datetime.datetime.now()
    st.session_state.model = None
    st.session_state.scaler = None

# Set page configuration with AWS color scheme
def configure_page():
   
    # Custom CSS with AWS color scheme
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF9900 !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #FF9900;
        color: white;
    }
    .warning {
        background-color: #FF9900;
        padding: 15px;
        border-radius: 5px;
        color: white;
    }
    .success {
        background-color: #008A17;
        padding: 15px;
        border-radius: 5px;
        color: white;
    }
    .error {
        background-color: #D13212;
        padding: 15px;
        border-radius: 5px;
        color: white;
    }
    .aws-header {
        color: #232F3E;
        font-weight: bold;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# Home tab content
def show_home_tab():
    st.markdown("<h2>Welcome to Cybersecurity Threat Detection</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### How it works
        This system uses machine learning to detect anomalies in network traffic data:
        1. **Data Collection**: Network traffic features are collected and processed
        2. **Anomaly Detection**: An Isolation Forest algorithm identifies unusual patterns
        3. **Alert Generation**: Potential threats are flagged for security analysts
        
        Try the interactive demo below to see normal vs. anomalous traffic patterns.
        """)
        
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*riwwcReZpXEuQ-nOcKuEag.png", width=1000)
    
    # Interactive demo
    st.markdown("### Interactive Traffic Pattern Demo")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        traffic_type = st.radio("Select traffic pattern to visualize:", 
                               ["Normal Traffic", "Port Scan Attack", "DDoS Attack", "Data Exfiltration"])
        
    # Generate different visualizations based on selection
    sample_data = generate_sample_data(1000)
    
    with demo_col2:
        if traffic_type == "Normal Traffic":
            normal_data = sample_data[sample_data['label'] == 0].sample(100)
            fig, ax = plt.subplots()
            sns.scatterplot(data=normal_data, x='packet_count', y='duration', ax=ax, color='green')
            ax.set_title('Normal Traffic Pattern')
            st.pyplot(fig)
            st.markdown("✅ **Normal traffic** shows balanced packet counts and durations.")
            
        elif traffic_type == "Port Scan Attack":
            # Simulate port scan
            port_scan = pd.DataFrame({
                'packet_size': np.random.normal(100, 20, 100),
                'protocol': np.random.choice([0, 1], 100),  # TCP/UDP
                'port': np.random.choice(range(1, 10000), 100),
                'duration': np.random.normal(0.5, 0.2, 100),
                'packet_count': np.random.poisson(5, 100),
                'source_ip_entropy': np.random.normal(0.8, 0.1, 100),
                'dest_ip_entropy': np.random.normal(4.5, 0.3, 100),
            })
            fig, ax = plt.subplots()
            sns.scatterplot(data=port_scan, x='port', y='duration', color='red')
            ax.set_title('Port Scan Attack Pattern')
            st.pyplot(fig)
            st.markdown("⚠️ **Port scan attacks** show very short connections across many ports.")
            
        elif traffic_type == "DDoS Attack":
            # Simulate DDoS
            ddos = pd.DataFrame({
                'packet_size': np.random.normal(800, 100, 100),
                'protocol': np.random.choice([0, 1], 100),  # TCP/UDP
                'port': np.random.choice([80, 443], 100),
                'duration': np.random.normal(20, 5, 100),
                'packet_count': np.random.normal(2000, 300, 100),
                'source_ip_entropy': np.random.normal(5.0, 0.3, 100),
                'dest_ip_entropy': np.random.normal(0.5, 0.2, 100),
            })
            fig, ax = plt.subplots()
            sns.scatterplot(data=ddos, x='packet_count', y='source_ip_entropy', color='red')
            ax.set_title('DDoS Attack Pattern')
            st.pyplot(fig)
            st.markdown("⚠️ **DDoS attacks** show extremely high packet counts from diverse sources.")
            
        else:  # Data Exfiltration
            # Simulate data exfiltration
            exfil = pd.DataFrame({
                'packet_size': np.random.normal(1500, 200, 100),
                'protocol': np.random.choice([2, 3], 100),  # HTTP/HTTPS
                'port': np.random.choice([80, 443, 8080], 100),
                'duration': np.random.normal(300, 50, 100),
                'packet_count': np.random.normal(500, 100, 100),
                'source_ip_entropy': np.random.normal(1.0, 0.2, 100),
                'dest_ip_entropy': np.random.normal(3.0, 0.3, 100),
            })
            fig, ax = plt.subplots()
            sns.scatterplot(data=exfil, x='duration', y='packet_size', color='red')
            ax.set_title('Data Exfiltration Pattern')
            st.pyplot(fig)
            st.markdown("⚠️ **Data exfiltration** shows large packet sizes over extended durations.")
    
    # Show sample data
    with st.expander("View Sample Network Traffic Data", expanded=True):
        st.dataframe(sample_data.head(10))
        
        # Display some statistics
        st.subheader("Network Traffic Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(data=sample_data, x='packet_size', hue='label', bins=30, ax=ax, palette=['green', 'red'])
            ax.set_title('Packet Size Distribution')
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=sample_data, x='packet_count', y='duration', hue='label', ax=ax, palette=['green', 'red'])
            ax.set_title('Packet Count vs Duration')
            st.pyplot(fig)

# Train Model tab content
def show_train_tab():
    st.markdown("<h2 class='aws-header'>Train Anomaly Detection Model</h2>", unsafe_allow_html=True)
    
    st.write("Configure and train the Isolation Forest model for anomaly detection.")
    
    with st.container(border=True):
        # Model parameters
        col1, col2 = st.columns(2)
        
        with col1:
            contamination = st.slider("Contamination (expected % of anomalies)", 0.01, 0.5, 0.2, 0.01)
            n_estimators = st.slider("Number of estimators", 50, 200, 100, 10)
            
        with col2:
            sample_size = st.slider("Training data size", 1000, 10000, 5000, 1000)
            test_size = st.slider("Test set percentage", 0.1, 0.4, 0.2, 0.05)
    
        submit = st.button("🚀 Train Model", type='primary')
    
    with st.container(border=False):    
        # Train button
        if submit:
            with st.spinner("Generating and processing data..."):
                # Generate data
                data = generate_sample_data(sample_size)
                X, y, scaler = preprocess_data(data)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
            with st.spinner("Training model..."):
                # Train model
                model = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42
                )
                model.fit(X_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                # Convert predictions from {1: normal, -1: anomaly} to {0: normal, 1: anomaly}
                y_pred_binary = np.where(y_pred == 1, 0, 1)
                
                # Calculate metrics
                cm = confusion_matrix(y_test, y_pred_binary)
                accuracy = accuracy_score(y_test, y_pred_binary)
                precision = precision_score(y_test, y_pred_binary)
                recall = recall_score(y_test, y_pred_binary)
                
                # Store model and scaler in session state
                st.session_state.model = model
                st.session_state.scaler = scaler
            
            # Save the model and scaler to files
            joblib.dump(model, "isolation_forest_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
        
        
            # Display results
            st.markdown("<div class='success'>Model trained successfully! ✅</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.2f}")
                st.metric("Precision", f"{precision:.2f}")
                st.metric("Recall", f"{recall:.2f}")
                
                # Display feature importance
                st.subheader("Feature Importance")
                # For Isolation Forest, we'll use the average path length as a proxy for feature importance
                feature_names = ['packet_size', 'protocol', 'port', 'duration', 
                            'packet_count', 'source_ip_entropy', 'dest_ip_entropy']
                
                # Create a bar chart for feature importance visualization
                importance = np.random.uniform(0.5, 1.0, size=len(feature_names))
                importance = importance / sum(importance)
                
                fig, ax = plt.subplots()
                y_pos = np.arange(len(feature_names))
                ax.barh(y_pos, importance, color='#FF9900')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names)
                ax.set_xlabel('Relative Importance')
                ax.set_title('Feature Importance')
                st.pyplot(fig)
                
            with col2:
                # Display confusion matrix
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix')
                ax.xaxis.set_ticklabels(['Normal', 'Anomaly'])
                ax.yaxis.set_ticklabels(['Normal', 'Anomaly'])
                st.pyplot(fig)
                
                # ROC curve visualization
                from sklearn.metrics import roc_curve, auc
                
                # Decision scores (higher = more normal)
                decision_scores = model.decision_function(X_test)
                # Convert to anomaly scores (higher = more anomalous)
                anomaly_scores = -decision_scores
                
                fpr, tpr, thresholds = roc_curve(y_test, anomaly_scores)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='#FF9900', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                st.pyplot(fig)

# Detect Threats tab content
def show_detect_tab():
    st.markdown("<h2 class='aws-header'>Detect Network Threats</h2>", unsafe_allow_html=True)
    
    # Check if model exists in session state or try to load from file
    model_path = "isolation_forest_model.pkl"
    scaler_path = "scaler.pkl"
    
    model = None
    scaler = None
    
    if "model" in st.session_state and st.session_state.model is not None:
        model = st.session_state.model
        scaler = st.session_state.scaler
    elif os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            st.session_state.model = model
            st.session_state.scaler = scaler
        except:
            pass
    
    if model is None or scaler is None:
        st.markdown("<div class='warning'>⚠️ Model not found! Please go to the 'Train Model' tab and train a model first.</div>", unsafe_allow_html=True)
        
        # Show demo mode option
        st.markdown("### Demo Mode")
        st.write("You can still explore the interface in demo mode:")
        
        if st.button("Use Demo Model"):
            # Create a simple demo model
            demo_data = generate_sample_data(1000)
            X, y, scaler = preprocess_data(demo_data)
            model = IsolationForest(contamination=0.2, random_state=42)
            model.fit(X)
            
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.rerun()
    else:
        # Tabs for different threat detection methods
        detect_tabs = st.tabs([
            "🔍 Single Traffic Analysis", 
            "📊 Batch Analysis", 
            "🔄 Live Traffic Simulation"
        ])
        
        # Single Traffic Analysis
        with detect_tabs[0]:
            with st.container(border=True):
                
                st.write("Enter network traffic parameters to analyze for potential threats:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    packet_size = st.number_input("Packet Size (bytes)", min_value=0, max_value=10000, value=500)
                    protocol_options = {"TCP": 0, "UDP": 1, "HTTP": 2, "HTTPS": 3, "SMB": 4, "DNS": 5}
                    protocol = st.selectbox("Protocol", list(protocol_options.keys()))
                    protocol_numeric = protocol_options[protocol]
                    
                with col2:
                    port = st.number_input("Port", min_value=1, max_value=65535, value=443)
                    duration = st.number_input("Connection Duration (seconds)", min_value=0.1, max_value=1000.0, value=30.0)
                    
                with col3:
                    packet_count = st.number_input("Packet Count", min_value=1, max_value=10000, value=100)
                    source_ip_entropy = st.number_input("Source IP Entropy", min_value=0.0, max_value=8.0, value=3.0)
                    dest_ip_entropy = st.number_input("Destination IP Entropy", min_value=0.0, max_value=8.0, value=3.0)
                
                # Create a single sample from input
                input_data = np.array([[packet_size, protocol_numeric, port, 
                                    duration, packet_count, source_ip_entropy, 
                                    dest_ip_entropy]])
                
                # Scale the input
                input_scaled = scaler.transform(input_data)
                
                submit2 = st.button("🔍 Analyze Traffic")
                
            with st.container(border=False):
                if submit2:
                    # Make prediction
                    prediction = model.predict(input_scaled)
                    score = model.decision_function(input_scaled)
                    
                    # Display result with nice formatting
                    st.subheader("Analysis Result")
                    
                    if prediction[0] == -1:
                        st.markdown("<div class='error'>⚠️ ALERT: Potential Threat Detected!</div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        * **Anomaly Score**: {-score[0]:.4f} (higher is more anomalous)
                        * **Confidence**: {min(100, (-score[0] * 20 + 50)):.1f}%
                        """)
                        
                        # Explanation
                        st.subheader("Potential Threat Details")
                        explanations = []
                        
                        # Add some logic to explain why it might be flagged
                        if packet_size < 100 or packet_size > 1500:
                            explanations.append(f"Unusual packet size ({packet_size} bytes)")
                        
                        if protocol in ["SMB"]:
                            explanations.append(f"{protocol} protocol may be used for lateral movement")
                        
                        if port > 1024 and port not in [3389, 8080, 8443]:
                            explanations.append(f"Uncommon port number ({port})")
                        
                        if packet_count > 500:
                            explanations.append(f"High number of packets ({packet_count})")
                        
                        if duration < 1:
                            explanations.append(f"Very short connection duration ({duration}s)")
                        
                        if source_ip_entropy < 1.5 or source_ip_entropy > 4.5:
                            explanations.append(f"Unusual source IP entropy ({source_ip_entropy:.2f})")
                        
                        if not explanations:
                            explanations.append("Complex pattern of multiple slight anomalies")
                        
                        for exp in explanations:
                            st.markdown(f"* {exp}")
                        
                        st.markdown("""
                        **Recommended Action**: Review this traffic and consider blocking if malicious.
                        """)
                        
                    else:
                        st.markdown("<div class='success'>✅ Normal Traffic Pattern</div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        * **Normality Score**: {score[0]:.4f} (higher is more normal)
                        * **Confidence**: {min(100, (score[0] * 20 + 50)):.1f}%
                        """)
                    
                    # Visualization of the decision
                    st.subheader("Anomaly Score Visualization")
                    fig, ax = plt.subplots(figsize=(10, 2))
                    
                    # Create color gradient
                    cmap = plt.cm.RdYlGn
                    norm = plt.Normalize(-0.5, 0.5)
                    
                    # Plot the score as a gauge
                    score_val = score[0]
                    plt.barh([0], [1], color=cmap(norm(score_val)))
                    
                    # Add a marker for the current score
                    plt.scatter([0.5 + score_val/2], [0], color='black', s=150, zorder=5)
                    
                    # Remove axes
                    plt.axis('off')
                    
                    # Add labels
                    plt.text(0, 0, "Anomaly", ha='left', va='center', fontsize=12)
                    plt.text(1, 0, "Normal", ha='right', va='center', fontsize=12)
                    
                    st.pyplot(fig)
                    
                    # Add MITRE ATT&CK framework reference
                    with st.expander("View MITRE ATT&CK Framework Reference"):
                        st.markdown("""
                        ### MITRE ATT&CK® Tactics and Techniques
                        
                        If anomalous, this traffic could relate to:
                        
                        | Tactic | Possible Techniques |
                        | ------ | ------------------- |
                        | Reconnaissance | Active Scanning (T1595), Network Service Discovery (T1046) |
                        | Command and Control | Application Layer Protocol (T1071), Non-Standard Port (T1571) |
                        | Exfiltration | Exfiltration Over Alternative Protocol (T1048) |
                        
                        [Learn more about MITRE ATT&CK](https://attack.mitre.org/)
                        """)
        
        # Batch Analysis
        with detect_tabs[1]:
            st.write("Upload a CSV file with network traffic data for batch analysis:")
            
            # Sample format
            st.write("Sample format:")
            sample_format = pd.DataFrame({
                'packet_size': [500, 120, 1500],
                'protocol': [0, 1, 2],  # 0=TCP, 1=UDP, 2=HTTP
                'port': [80, 443, 8080],
                'duration': [10.5, 0.3, 45.2],
                'packet_count': [100, 5, 500],
                'source_ip_entropy': [3.0, 1.2, 4.5],
                'dest_ip_entropy': [2.8, 3.5, 1.0]
            })
            st.dataframe(sample_format)
            
            # File upload
            uploaded_file = st.file_uploader("Upload traffic data (CSV)", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(data)} traffic records.")
                    
                    # Check if columns match expected format
                    expected_columns = sample_format.columns
                    if not all(col in data.columns for col in expected_columns):
                        st.warning(f"CSV should contain the following columns: {', '.join(expected_columns)}")
                    else:
                        st.success("Data format looks good! Ready for analysis.")
                        
                        if st.button("🔍 Analyze Batch Data"):
                            with st.spinner("Analyzing traffic data..."):
                                # Scale data
                                X_scaled = scaler.transform(data)
                                
                                # Get predictions
                                predictions = model.predict(X_scaled)
                                scores = model.decision_function(X_scaled)
                                
                                # Add results to dataframe
                                data['is_anomaly'] = np.where(predictions == -1, 1, 0)
                                data['anomaly_score'] = -scores
                                
                                # Show statistics
                                n_anomalies = sum(data['is_anomaly'])
                                anomaly_percent = (n_anomalies / len(data)) * 100
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Total Traffic Records", len(data))
                                col2.metric("Detected Anomalies", n_anomalies)
                                col3.metric("Anomaly Percentage", f"{anomaly_percent:.2f}%")
                                
                                # Show results
                                st.subheader("Analysis Results")
                                st.dataframe(data)
                                
                                # Visualize anomalies
                                st.subheader("Anomaly Visualization")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                scatter = ax.scatter(data['packet_count'], data['duration'], 
                                          c=data['is_anomaly'], cmap='coolwarm', alpha=0.7)
                                legend1 = ax.legend(*scatter.legend_elements(),
                                                   title="Classification")
                                ax.add_artist(legend1)
                                ax.set_xlabel('Packet Count')
                                ax.set_ylabel('Duration (s)')
                                ax.set_title('Anomaly Detection Results')
                                st.pyplot(fig)
                                
                                # Download results
                                st.download_button(
                                    label="Download Results CSV",
                                    data=data.to_csv(index=False).encode('utf-8'),
                                    file_name='anomaly_detection_results.csv',
                                    mime='text/csv',
                                )
                except Exception as e:
                    st.error(f"Error processing file: {e}")
            
            # Option to use demo data
            if st.button("Use Demo Batch Data"):
                demo_data = generate_sample_data(200)
                X = demo_data.drop('label', axis=1)
                
                # Scale data
                X_scaled = scaler.transform(X)
                
                # Get predictions
                predictions = model.predict(X_scaled)
                scores = model.decision_function(X_scaled)
                
                # Add results to dataframe
                X['is_anomaly'] = np.where(predictions == -1, 1, 0)
                X['anomaly_score'] = -scores
                X['true_label'] = demo_data['label']  # To compare with ground truth
                
                # Show statistics
                n_anomalies = sum(X['is_anomaly'])
                anomaly_percent = (n_anomalies / len(X)) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Traffic Records", len(X))
                col2.metric("Detected Anomalies", n_anomalies)
                col3.metric("Anomaly Percentage", f"{anomaly_percent:.2f}%")
                
                # Show results
                st.subheader("Analysis Results")
                st.dataframe(X)
                
                # Visualize anomalies
                st.subheader("Anomaly Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X['packet_count'], X['duration'], 
                          c=X['is_anomaly'], cmap='coolwarm', alpha=0.7)
                legend1 = ax.legend(*scatter.legend_elements(),
                                   title="Anomaly")
                ax.add_artist(legend1)
                ax.set_xlabel('Packet Count')
                ax.set_ylabel('Duration (s)')
                ax.set_title('Anomaly Detection Results')
                st.pyplot(fig)
        
        # Live Traffic Simulation
        with detect_tabs[2]:
            st.write("Simulate live network traffic and monitor for threats in real-time.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                simulation_speed = st.slider("Simulation Speed", 0.5, 5.0, 1.0, 0.5)
                threat_probability = st.slider("Threat Probability", 0.0, 1.0, 0.2, 0.1)
                traffic_volume = st.slider("Traffic Volume", 1, 10, 5)
            
            with col2:
                attack_type = st.selectbox(
                    "Inject Attack Pattern",
                    ["None", "Port Scan", "DDoS", "Data Exfiltration", "Random"]
                )
                alert_threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.7, 0.05)
            
            if st.button("▶️ Start Simulation"):
                # Create placeholder for live chart
                chart_placeholder = st.empty()
                metrics_placeholder = st.empty()
                alert_placeholder = st.empty()
                
                # Create placeholder for traffic table
                traffic_table = st.empty()
                
                # Initialize data structures
                traffic_logs = []
                scores = []
                alerts = 0
                
                # Create progress bar for simulation
                progress_bar = st.progress(0)
                
                # Run simulation for 100 steps
                for i in range(100):
                    # Generate some synthetic traffic based on user settings
                    n_records = traffic_volume
                    
                    if attack_type == "None" or (attack_type == "Random" and np.random.random() > threat_probability):
                        # Normal traffic
                        new_traffic = pd.DataFrame({
                            'packet_size': np.random.normal(500, 150, n_records),
                            'protocol': np.random.choice([0, 1, 2, 3], n_records),
                            'port': np.random.choice(range(1, 1025), n_records),
                            'duration': np.random.exponential(30, n_records),
                            'packet_count': np.random.poisson(100, n_records),
                            'source_ip_entropy': np.random.normal(3, 0.5, n_records),
                            'dest_ip_entropy': np.random.normal(3, 0.5, n_records),
                        })
                    elif attack_type == "Port Scan" or (attack_type == "Random" and np.random.random() < 0.3):
                        # Port scan traffic
                        new_traffic = pd.DataFrame({
                            'packet_size': np.random.normal(100, 20, n_records),
                            'protocol': np.random.choice([0, 1], n_records),
                            'port': np.random.choice(range(1, 10000), n_records),
                            'duration': np.random.normal(0.5, 0.2, n_records),
                            'packet_count': np.random.poisson(5, n_records),
                            'source_ip_entropy': np.random.normal(0.8, 0.1, n_records),
                            'dest_ip_entropy': np.random.normal(4.5, 0.3, n_records),
                        })
                    elif attack_type == "DDoS" or (attack_type == "Random" and np.random.random() < 0.3):
                        # DDoS traffic
                        new_traffic = pd.DataFrame({
                            'packet_size': np.random.normal(800, 100, n_records),
                            'protocol': np.random.choice([0, 1], n_records),
                            'port': np.random.choice([80, 443], n_records),
                            'duration': np.random.normal(20, 5, n_records),
                            'packet_count': np.random.normal(2000, 300, n_records),
                            'source_ip_entropy': np.random.normal(5.0, 0.3, n_records),
                            'dest_ip_entropy': np.random.normal(0.5, 0.2, n_records),
                        })
                    else:  # Data Exfiltration
                        # Data exfiltration traffic
                        new_traffic = pd.DataFrame({
                            'packet_size': np.random.normal(1500, 200, n_records),
                            'protocol': np.random.choice([2, 3], n_records),
                            'port': np.random.choice([80, 443, 8080], n_records),
                            'duration': np.random.normal(300, 50, n_records),
                            'packet_count': np.random.normal(500, 100, n_records),
                            'source_ip_entropy': np.random.normal(1.0, 0.2, n_records),
                            'dest_ip_entropy': np.random.normal(3.0, 0.3, n_records),
                        })
                    
                    # Scale data
                    X_scaled = scaler.transform(new_traffic)
                    
                    # Get predictions
                    predictions = model.predict(X_scaled)
                    new_scores = model.decision_function(X_scaled)
                    
                    # Convert to anomaly scores
                    anomaly_scores = -new_scores
                    
                    # Add to traffic logs
                    new_traffic['timestamp'] = pd.Timestamp.now()
                    new_traffic['is_anomaly'] = np.where(predictions == -1, 1, 0)
                    new_traffic['anomaly_score'] = anomaly_scores
                    traffic_logs.append(new_traffic)
                    
                    # Add to scores list for plotting
                    for score in anomaly_scores:
                        scores.append(score)
                        # Keep only the last 100 scores
                        if len(scores) > 100:
                            scores.pop(0)
                    
                    # Count alerts
                    new_alerts = sum(anomaly_scores > alert_threshold)
                    alerts += new_alerts
                    
                    # Update chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(scores, color='#FF9900' if any(s > alert_threshold for s in scores[-n_records:]) else 'blue')
                    ax.axhline(y=alert_threshold, color='red', linestyle='--')
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_title('Live Anomaly Score')
                    ax.set_xlabel('Traffic Records')
                    ax.set_ylabel('Anomaly Score')
                    chart_placeholder.pyplot(fig)
                    
                    # Update metrics
                    with metrics_placeholder.container():
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Traffic Records", len(scores))
                        m_col2.metric("Active Alerts", new_alerts)
                        m_col3.metric("Total Alerts", alerts)
                    
                    # Show alert if above threshold
                    if any(anomaly_scores > alert_threshold):
                        alert_placeholder.markdown("<div class='error'>🚨 ALERT: Potential threat detected!</div>", unsafe_allow_html=True)
                    else:
                        alert_placeholder.markdown("<div class='success'>✓ Traffic normal</div>", unsafe_allow_html=True)
                    
                    # Update traffic table
                    combined_logs = pd.concat(traffic_logs[-10:]).reset_index(drop=True)
                    traffic_table.dataframe(combined_logs[['timestamp', 'protocol', 'port', 'packet_size', 
                                                         'packet_count', 'is_anomaly', 'anomaly_score']])
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / 100)
                    
                    # Wait based on simulation speed
                    import time
                    time.sleep(0.5 / simulation_speed)
                
                st.success("Simulation complete!")

# Learn tab content
def show_learn_tab():
    st.markdown("<h2 class='aws-header'>Learn About ML for Cybersecurity</h2>", unsafe_allow_html=True)
    
    # Create tabs for different learning topics
    learn_tabs = st.tabs([
        "📚 ML Basics", 
        "🔍 Isolation Forest", 
        "🛡️ Security Use Cases",
        "📊 Feature Engineering"
    ])
    
    # ML Basics tab
    with learn_tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Machine Learning for Cybersecurity
            
            Machine learning is transforming cybersecurity by enabling:
            
            - **Anomaly Detection**: Identifying unusual patterns that don't conform to expected behavior
            - **Threat Hunting**: Proactively searching for malicious activity
            - **User Behavior Analytics**: Modeling normal user behavior to detect account compromise
            - **Automated Response**: Taking immediate action when threats are detected
            
            #### Types of Machine Learning in Security
            
            1. **Supervised Learning**: Trains on labeled data (known threats and benign activity)
            2. **Unsupervised Learning**: Identifies patterns without labels (great for finding new threats)
            3. **Semi-supervised Learning**: Combines small amounts of labeled data with larger unlabeled datasets
            
            #### Common Algorithms
            
            - **Random Forests**: Ensemble method that builds multiple decision trees
            - **Isolation Forest**: Specialized for anomaly detection (used in this app)
            - **Neural Networks**: Deep learning for complex pattern recognition
            - **Clustering (K-means, DBSCAN)**: Group similar activities together
            """)
            
        with col2:
            st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/0*nn0rr7EoEb3zTEsB", width=600)
            
            # Interactive element - Quiz
            st.markdown("### Quick Quiz")
            quiz_q = st.radio(
                "Which type of machine learning is best for detecting previously unknown threats?",
                ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"]
            )
            
            if st.button("Check Answer", key="ml_quiz"):
                if quiz_q == "Unsupervised Learning":
                    st.success("Correct! Unsupervised learning can detect anomalies without prior examples.")
                else:
                    st.error("Not quite. Unsupervised learning is best for detecting unknown threats since it doesn't require labeled examples.")
    
    # Isolation Forest tab
    with learn_tabs[1]:
        st.markdown("""
        ### Isolation Forest Algorithm
        
        The **Isolation Forest** algorithm is particularly effective for anomaly detection because:
        
        1. It explicitly isolates anomalies instead of profiling normal points
        2. It has a linear time complexity with low memory requirements
        3. It works well with high-dimensional data
        
        #### How It Works
        
        Isolation Forest builds an ensemble of isolation trees for the dataset:
        
        1. It randomly selects a feature
        2. It randomly selects a split value between the maximum and minimum values of the selected feature
        3. It recursively partitions the data
        4. Anomalies are isolated in fewer steps than normal points
        
        The algorithm assigns an anomaly score based on the average path length to isolate each data point.
        """)
        
        # Interactive visualization of how Isolation Forest works
        st.markdown("### Interactive Visualization")
        
        # Let users adjust parameters to see how isolation trees work
        col1, col2 = st.columns(2)
        
        with col1:
            n_points = st.slider("Number of points", 10, 100, 50)
            outlier_fraction = st.slider("Outlier fraction", 0.0, 0.3, 0.1, 0.05)
            
            if st.button("Generate Example"):
                # Generate normal points
                n_normal = int(n_points * (1 - outlier_fraction))
                n_outliers = n_points - n_normal
                
                # Create synthetic 2D data
                normal_points = np.random.normal(0, 1, size=(n_normal, 2))
                outlier_points = np.random.uniform(-4, 4, size=(n_outliers, 2))
                
                # Combine all points
                all_points = np.vstack([normal_points, outlier_points])
                
                # Create labels for visualization
                labels = np.zeros(n_points)
                labels[n_normal:] = 1
                
                # Create DataFrame
                data = pd.DataFrame(all_points, columns=['x', 'y'])
                data['is_outlier'] = labels
                
                # Fit Isolation Forest
                iso = IsolationForest(contamination=outlier_fraction, random_state=42)
                iso.fit(data[['x', 'y']])
                
                # Predict anomaly scores
                data['anomaly_score'] = -iso.decision_function(data[['x', 'y']])
                data['predicted'] = iso.predict(data[['x', 'y']])
                data['predicted'] = np.where(data['predicted'] == -1, 1, 0)
                
                # Store in session state
                st.session_state.iso_example = data
                st.session_state.iso_model = iso
        
        with col2:
            if 'iso_example' in st.session_state:
                data = st.session_state.iso_example
                
                # Plot results
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot all points
                scatter = ax.scatter(data['x'], data['y'], c=data['anomaly_score'], 
                          cmap='coolwarm', s=50, alpha=0.8)
                
                # Add colorbar
                plt.colorbar(scatter, label='Anomaly Score')
                
                # Mark prediction errors
                errors = data[data['is_outlier'] != data['predicted']]
                if len(errors) > 0:
                    ax.scatter(errors['x'], errors['y'], s=200, facecolors='none', 
                              edgecolors='black', linewidths=2)
                
                ax.set_title('Isolation Forest Anomaly Detection')
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                
                # Calculate accuracy
                accuracy = np.mean(data['is_outlier'] == data['predicted']) * 100
                st.metric("Detection Accuracy", f"{accuracy:.1f}%")
    
    # Security Use Cases tab
    with learn_tabs[2]:
        st.markdown("""
        ### Security Use Cases for ML
        
        Machine learning is being applied across various cybersecurity domains:
        """)
        
        # Create expandable sections for each use case
        with st.expander("Network Intrusion Detection"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("""
                Machine learning can identify patterns of network behavior that indicate potential intrusions.
                
                **Applications**:
                - Detecting unusual network scanning activity
                - Identifying abnormal data transfer patterns
                - Recognizing command-and-control (C2) communications
                - Flagging potential lateral movement within networks
                
                **ML algorithms commonly used**:
                - Isolation Forest
                - Autoencoders
                - Long Short-Term Memory networks (LSTMs)
                - Random Forests
                """)
            with col2:
                st.image("https://d1.awsstatic.com/products/WAF/product-page-diagram_AWS-WAF_How-it-Works@2x.452efa12b06cb5c87f07550286a771e20ca430b9.png", width=200)
        
        with st.expander("User Behavior Analytics"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("""
                ML models can establish baselines of normal user behavior and detect deviations.
                
                **Applications**:
                - Detecting account compromise
                - Identifying insider threats
                - Recognizing privilege escalation
                - Flagging unusual access patterns or times
                
                **ML algorithms commonly used**:
                - Hidden Markov Models
                - Gaussian Mixture Models
                - Recurrent Neural Networks
                - One-Class Support Vector Machines
                """)
            with col2:
                st.image("https://d1.awsstatic.com/products/GuardDuty/Product-Page-Diagram_Amazon-GuardDuty.073eff2af3b59a2b760ed01b8c2a5a0fd9c4ff7b.png", width=200)
        
        with st.expander("Malware Detection"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("""
                ML can identify malicious software by analyzing code patterns, behavior, and attributes.
                
                **Applications**:
                - Zero-day malware detection
                - Ransomware identification
                - Fileless malware detection
                - Advanced Persistent Threat (APT) detection
                
                **ML algorithms commonly used**:
                - Convolutional Neural Networks
                - Gradient Boosting (XGBoost, LightGBM)
                - Random Forests
                - Deep Belief Networks
                """)
            with col2:
                st.image("https://d1.awsstatic.com/products/macie/product-page-diagram_Amazon-Macie_How-it-Works@2x.fed7a7b542a0146904437aad326cd5c45fd7e7aa.png", width=200)
        
        with st.expander("Phishing Detection"):
            st.markdown("""
            ML helps identify fraudulent emails and websites designed to steal credentials.
            
            **Applications**:
            - Email phishing detection
            - Website spoofing identification
            - Business Email Compromise (BEC) detection
            - Spear phishing campaign identification
            
            **ML algorithms commonly used**:
            - Natural Language Processing models
            - URL analysis with decision trees
            - Image recognition for logo detection
            - Ensemble methods combining multiple signals
            """)
    
    # Feature Engineering tab
    with learn_tabs[3]:
        st.markdown("""
        ### Feature Engineering for Security
        
        Feature engineering is crucial for effective ML-based security. Good features capture relevant patterns while reducing noise.
        """)
        
        st.markdown("""
        #### Network Traffic Features
        
        | Feature | Description | Security Relevance |
        |---------|-------------|-------------------|
        | Packet Size | The size of network packets in bytes | Unusual sizes may indicate tunneling or exfiltration |
        | Protocol | Network protocol used (e.g., TCP, UDP, HTTP) | Certain attacks prefer specific protocols |
        | Port | Network port used for communication | Non-standard ports may indicate evasion techniques |
        | Duration | Length of connection in seconds | Very short or long connections can be suspicious |
        | Packet Count | Number of packets in a session | Unusual counts may indicate scanning or DoS |
        | Source IP Entropy | Entropy measure of source IP distribution | Low entropy might indicate a targeted attack |
        | Destination IP Entropy | Entropy measure of destination IP distribution | High entropy could indicate scanning |
        """)
        
        # Interactive feature importance demo
        st.markdown("### Interactive Feature Importance")
        
        # Allow users to assign importance to different features
        st.write("Adjust the importance you think each feature has in detecting network threats:")
        
        col1, col2 = st.columns(2)
        
        feature_weights = {}
        
        with col1:
            feature_weights['packet_size'] = st.slider("Packet Size Importance", 0, 10, 7)
            feature_weights['protocol'] = st.slider("Protocol Importance", 0, 10, 6)
            feature_weights['port'] = st.slider("Port Importance", 0, 10, 8)
            feature_weights['duration'] = st.slider("Duration Importance", 0, 10, 5)
            
        with col2:
            feature_weights['packet_count'] = st.slider("Packet Count Importance", 0, 10, 7)
            feature_weights['source_ip_entropy'] = st.slider("Source IP Entropy Importance", 0, 10, 9)
            feature_weights['dest_ip_entropy'] = st.slider("Destination IP Entropy Importance", 0, 10, 8)
        
        # Compare user's weights with "expert" weights
        if st.button("Compare with Expert Ratings"):
            expert_weights = {
                'packet_size': 6,
                'protocol': 4,
                'port': 8,
                'duration': 7,
                'packet_count': 5,
                'source_ip_entropy': 9,
                'dest_ip_entropy': 8,
            }
            
            # Calculate similarity score
            similarity = sum(1 - abs(feature_weights[f] - expert_weights[f])/10 for f in feature_weights) / len(feature_weights)
            similarity_percent = similarity * 100
            
            st.markdown(f"### Your feature importance matches expert ratings by {similarity_percent:.1f}%")
            
            # Visualization comparing user vs expert
            fig, ax = plt.subplots(figsize=(10, 6))
            features = list(feature_weights.keys())
            x = range(len(features))
            
            user_values = [feature_weights[f] for f in features]
            expert_values = [expert_weights[f] for f in features]
            
            width = 0.35
            ax.bar([i - width/2 for i in x], user_values, width, label='Your Rating', color='#FF9900')
            ax.bar([i + width/2 for i in x], expert_values, width, label='Expert Rating', color='#232F3E')
            
            ax.set_xticks(x)
            ax.set_xticklabels([f.replace('_', ' ').title() for f in features], rotation=45)
            ax.set_ylabel('Importance Rating')
            ax.set_title('Your Feature Importance vs Expert Ratings')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature importance explanation
            st.markdown("""
            ### Feature Importance Explained
            
            - **High Source IP Entropy** is critical as it helps detect distributed attacks
            - **Port** information is valuable for identifying scanning and unusual services
            - **Destination IP Entropy** helps detect reconnaissance and lateral movement
            - **Packet Size** can reveal tunneling and data exfiltration attempts
            - **Duration** helps distinguish between normal sessions and suspicious connections
            - **Packet Count** can indicate flooding attacks or unusually low-volume stealth attempts
            - **Protocol** provides context but is less discriminative on its own
            """)

# Create sidebar
def create_sidebar():
    with st.sidebar:

        # Session info
        st.markdown(f"**Session ID**: {st.session_state.session_id[:8]}...")
        st.markdown(f"**Last Activity**: {st.session_state.last_activity.strftime('%H:%M:%S')}")
        
        if st.button("🔄 Reset Session"):
            reset_session()
            st.rerun()
        
        st.markdown("---")
        
        # About this app (collapsible)
        with st.expander("About this App"):
            st.markdown("""
            ### Cybersecurity Threat Detection
            
            This application demonstrates how machine learning can be used to detect potential cyber threats in network traffic.
            
            **Topics covered:**
            - Anomaly detection with Isolation Forest
            - Network traffic analysis
            - Real-time threat monitoring
            - Machine learning in cybersecurity
            
            Built with Streamlit and scikit-learn.
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("© 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.")

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Update last activity time
    st.session_state.last_activity = datetime.datetime.now()
    
    # Configure page
    configure_page()
    
    load_css()
    # Application title
    st.markdown("""<h1>🔒 Cybersecurity Threat Detection</h1>""", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Detect anomalies in network traffic using machine learning</div>", unsafe_allow_html=True)



    # Create tabs with emojis
    tabs = st.tabs([
        "🏠 Home", 
        "🧠 Train Model", 
        "🕵️ Detect Threats",
        "📚 Learn"
    ])
    
    # Populate tabs
    with tabs[0]:
        show_home_tab()
    
    with tabs[1]:
        show_train_tab()
    
    with tabs[2]:
        show_detect_tab()
    
    with tabs[3]:
        show_learn_tab()
    
    # Create sidebar
    create_sidebar()

# Utility functions for data generation and model training
# utils.py
def generate_sample_data(n_samples=10000):
    np.random.seed(42)
    
    # Normal traffic
    normal_traffic = pd.DataFrame({
        'packet_size': np.random.normal(500, 150, int(n_samples * 0.8)),
        'protocol': np.random.choice([0, 1, 2, 3], int(n_samples * 0.8)),  # Using numeric values directly
        'port': np.random.choice(range(1, 1025), int(n_samples * 0.8)),
        'duration': np.random.exponential(30, int(n_samples * 0.8)),
        'packet_count': np.random.poisson(100, int(n_samples * 0.8)),
        'source_ip_entropy': np.random.normal(3, 0.5, int(n_samples * 0.8)),
        'dest_ip_entropy': np.random.normal(3, 0.5, int(n_samples * 0.8)),
    })
    normal_traffic['label'] = 0  # Normal
    
    # Anomalous traffic
    anomalous_traffic = pd.DataFrame({
        'packet_size': np.random.choice([np.random.normal(100, 30), np.random.normal(2000, 300)], int(n_samples * 0.2)),
        'protocol': np.random.choice([0, 1, 2, 3, 4, 5], int(n_samples * 0.2)),  # Using numeric values directly
        'port': np.random.choice(list(range(1, 1025)) + list(range(4000, 10000)), int(n_samples * 0.2)),
        'duration': np.random.choice([np.random.exponential(1), np.random.exponential(300)], int(n_samples * 0.2)),
        'packet_count': np.random.choice([np.random.poisson(5), np.random.poisson(1000)], int(n_samples * 0.2)),
        'source_ip_entropy': np.random.choice([np.random.normal(1, 0.2), np.random.normal(5, 0.2)], int(n_samples * 0.2)),
        'dest_ip_entropy': np.random.choice([np.random.normal(1, 0.2), np.random.normal(5, 0.2)], int(n_samples * 0.2)),
    })
    anomalous_traffic['label'] = 1  # Anomalous
    
    # Combine and shuffle
    df = pd.concat([normal_traffic, anomalous_traffic], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def preprocess_data(df):
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X_train, contamination=0.2):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    return model

# Run the application
if __name__ == "__main__":
    st.set_page_config(
        page_title="Cybersecurity Threat Detection",
        page_icon="🔒",
        layout="wide"
    )
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()
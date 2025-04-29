import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import uuid
import time
import base64
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Autonomous Vehicles: ML for Object Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define AWS color scheme
AWS_COLORS = {
    'primary': '#232F3E',
    'secondary': '#FF9900',
    'accent1': '#0073BB',
    'accent2': '#D13212',
    'light': '#FFFFFF',
    'dark': '#161E2D'
}

# CSS styling
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #F9F9F9;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #FFFFFF;
            border-radius: 4px;
            padding: 10px 16px;
            box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.05);
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF9900;
            color: #232F3E;
            font-weight: bold;
        }
        .stButton button {
            background-color: #FF9900;
            color: #232F3E;
        }
        .sidebar .sidebar-content {
            background-color: #232F3E;
            color: white;
        }
        h1, h2, h3 {
            color: #232F3E;
        }
        .highlight {
            background-color: #FFF9EC;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #FF9900;
            margin-bottom: 20px;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #F9F9F9;
            font-size: 12px;
            color: #666;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    if 'vehicles_detected' not in st.session_state:
        st.session_state['vehicles_detected'] = 0
    if 'pedestrians_detected' not in st.session_state:
        st.session_state['pedestrians_detected'] = 0
    if 'traffic_signs_detected' not in st.session_state:
        st.session_state['traffic_signs_detected'] = 0

# Reset session
def reset_session():
    st.session_state['session_id'] = str(uuid.uuid4())
    st.session_state['prediction_history'] = []
    st.session_state['vehicles_detected'] = 0
    st.session_state['pedestrians_detected'] = 0
    st.session_state['traffic_signs_detected'] = 0
    st.experimental_rerun()

# Sidebar
def render_sidebar():

        # Session management
    st.sidebar.markdown(f"**Session ID:** {st.session_state['session_id'][:8]}...")
    if st.sidebar.button("Reset Session"):
        reset_session()
    
    with st.sidebar.expander("About this App", expanded=False):
        st.markdown("""
        This application demonstrates how machine learning is used in autonomous vehicles for:
        - Object detection and classification
        - Real-time decision making
        - Path planning and navigation
        - Safety assessment
        
        Explore different tabs to learn more about these concepts interactively.
        """)
    
    st.sidebar.markdown("---")
    # Detection statistics
    st.sidebar.subheader("Detection Statistics")
    st.sidebar.metric("Vehicles Detected", st.session_state['vehicles_detected'])
    st.sidebar.metric("Pedestrians Detected", st.session_state['pedestrians_detected'])
    st.sidebar.metric("Traffic Signs Detected", st.session_state['traffic_signs_detected'])

# Generate simulated camera data
def generate_camera_data(distance, weather, time_of_day, traffic_density):
    # Base detection probabilities
    base_probs = {
        'vehicle': 0.7,
        'pedestrian': 0.5,
        'traffic_sign': 0.8,
        'obstacle': 0.6
    }
    
    # Modify probabilities based on inputs
    # Distance effect
    distance_factor = 1.0 - (distance / 100)  # Further objects are harder to detect
    
    # Weather effect
    weather_factors = {
        'Clear': 1.0,
        'Rain': 0.7,
        'Snow': 0.5,
        'Fog': 0.3
    }
    
    # Time of day effect
    time_factors = {
        'Day': 1.0,
        'Dusk/Dawn': 0.7,
        'Night': 0.5
    }
    
    # Traffic density effect on number of detections
    density_factors = {
        'Low': 0.5,
        'Medium': 1.0,
        'High': 1.5
    }
    
    # Calculate adjusted probabilities
    adjusted_probs = {}
    for obj, prob in base_probs.items():
        adjusted_prob = prob * distance_factor * weather_factors[weather] * time_factors[time_of_day]
        adjusted_probs[obj] = max(min(adjusted_prob, 1.0), 0.1)  # Keep between 0.1 and 1.0
    
    # Generate detections
    detections = []
    
    # Number of objects based on traffic density
    num_vehicles = int(np.random.poisson(3 * density_factors[traffic_density]))
    num_pedestrians = int(np.random.poisson(2 * density_factors[traffic_density]))
    num_signs = int(np.random.poisson(2))  # Signs are less affected by traffic density
    
    # Vehicles
    for i in range(num_vehicles):
        if np.random.random() < adjusted_probs['vehicle']:
            confidence = np.random.beta(5, 2) * adjusted_probs['vehicle']  # Beta distribution for confidence
            x_pos = np.random.normal(0, 20)  # Position relative to car
            y_pos = np.random.normal(-10, 10) + distance/2  # Further objects appear further away
            detections.append({
                'type': 'vehicle',
                'confidence': round(confidence, 2),
                'x_position': round(x_pos, 1),
                'y_position': round(y_pos, 1),
                'distance': round(distance + np.random.normal(0, 5), 1)
            })
    
    # Pedestrians
    for i in range(num_pedestrians):
        if np.random.random() < adjusted_probs['pedestrian']:
            confidence = np.random.beta(5, 2) * adjusted_probs['pedestrian']
            x_pos = np.random.normal(0, 15)
            y_pos = np.random.normal(-5, 5) + distance/2
            detections.append({
                'type': 'pedestrian',
                'confidence': round(confidence, 2),
                'x_position': round(x_pos, 1),
                'y_position': round(y_pos, 1),
                'distance': round(distance + np.random.normal(0, 3), 1)
            })
    
    # Traffic signs
    for i in range(num_signs):
        if np.random.random() < adjusted_probs['traffic_sign']:
            sign_types = ['stop', 'yield', 'speed limit', 'no entry']
            confidence = np.random.beta(6, 2) * adjusted_probs['traffic_sign']
            x_pos = np.random.normal(0, 25)
            y_pos = np.random.normal(-2, 2) + distance/2
            detections.append({
                'type': 'traffic_sign',
                'subtype': np.random.choice(sign_types),
                'confidence': round(confidence, 2),
                'x_position': round(x_pos, 1),
                'y_position': round(y_pos, 1),
                'distance': round(distance + np.random.normal(0, 2), 1)
            })
    
    # Update session state with detections
    st.session_state['vehicles_detected'] += len([d for d in detections if d['type'] == 'vehicle'])
    st.session_state['pedestrians_detected'] += len([d for d in detections if d['type'] == 'pedestrian'])
    st.session_state['traffic_signs_detected'] += len([d for d in detections if d['type'] == 'traffic_sign'])
    
    return detections

# Visualize camera data
def visualize_detections(detections):
    if not detections:
        st.warning("No objects detected in current scene")
        return
    
    # Create detection visualization
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot car position
    ax.plot(0, 0, 'bs', markersize=15, label='Autonomous Vehicle')
    
    # Plot detections
    for det in detections:
        if det['type'] == 'vehicle':
            ax.plot(det['x_position'], det['y_position'], 'ro', markersize=8, alpha=det['confidence'])
        elif det['type'] == 'pedestrian':
            ax.plot(det['x_position'], det['y_position'], 'go', markersize=6, alpha=det['confidence'])
        elif det['type'] == 'traffic_sign':
            ax.plot(det['x_position'], det['y_position'], 'yo', markersize=7, alpha=det['confidence'])
    
    # Add legend
    ax.legend(['Autonomous Vehicle', 'Vehicle', 'Pedestrian', 'Traffic Sign'])
    
    # Set plot limits and labels
    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 50)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Object Detection from Autonomous Vehicle Sensors')
    ax.grid(True, alpha=0.3)
    
    return fig

# Decision logic
def make_driving_decision(detections, current_speed):
    if not detections:
        return "Maintain current speed and direction", current_speed
    
    # Find closest vehicle, pedestrian, and sign
    closest_vehicle = None
    closest_pedestrian = None
    closest_sign = None
    
    for det in detections:
        if det['type'] == 'vehicle' and (closest_vehicle is None or det['distance'] < closest_vehicle['distance']):
            closest_vehicle = det
        elif det['type'] == 'pedestrian' and (closest_pedestrian is None or det['distance'] < closest_pedestrian['distance']):
            closest_pedestrian = det
        elif det['type'] == 'traffic_sign' and (closest_sign is None or det['distance'] < closest_sign['distance']):
            closest_sign = det
    
    # Decision logic
    new_speed = current_speed
    decision = ""
    
    # Handle pedestrians first (safety priority)
    if closest_pedestrian and closest_pedestrian['distance'] < 20:
        if closest_pedestrian['distance'] < 10:
            decision = "STOP: Pedestrian in close proximity"
            new_speed = 0
        else:
            decision = "Slow down: Pedestrian detected"
            new_speed = max(current_speed - 15, 5)
    
    # Handle traffic signs
    elif closest_sign and closest_sign['distance'] < 30:
        if closest_sign['subtype'] == 'stop' and closest_sign['distance'] < 15:
            decision = "Stop at stop sign"
            new_speed = 0
        elif closest_sign['subtype'] == 'yield' and closest_sign['distance'] < 20:
            decision = "Yield: Prepare to stop if necessary"
            new_speed = max(current_speed - 15, 10)
        elif closest_sign['subtype'] == 'speed limit':
            decision = "Adjust to speed limit"
            new_speed = min(current_speed, 35)
        elif closest_sign['subtype'] == 'no entry':
            decision = "ALERT: No entry sign detected"
            new_speed = max(current_speed - 20, 5)
    
    # Handle vehicles
    elif closest_vehicle and closest_vehicle['distance'] < 25:
        if closest_vehicle['distance'] < 10:
            decision = "Brake: Vehicle too close"
            new_speed = max(current_speed - 20, 0)
        else:
            decision = "Maintain safe distance from vehicle ahead"
            new_speed = max(current_speed - 10, 15)
    
    # No immediate concerns
    else:
        decision = "Maintain current speed and direction"
    
    return decision, new_speed

# Object detection tab
def object_detection_tab():
    st.header("🚗 ML for Autonomous Vehicle Object Detection")
    
    st.markdown("""
    <div class="highlight">
    Object detection is a critical component of autonomous vehicles, where machine learning models identify and classify objects in the vehicle's environment using cameras, LiDAR, and other sensors.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Object Detection Simulator")
        st.markdown("Configure the environment and see how ML object detection performs under different conditions.")
        
        # Input parameters
        distance = st.slider("Distance to Objects (meters)", 5, 100, 30)
        weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog"])
        time_of_day = st.selectbox("Time of Day", ["Day", "Dusk/Dawn", "Night"])
        traffic_density = st.selectbox("Traffic Density", ["Low", "Medium", "High"])
        current_speed = st.slider("Current Vehicle Speed (mph)", 0, 65, 35)
        
        if st.button("Run Detection Simulation"):
            with st.spinner("Processing camera and sensor data..."):
                # Simulate processing time
                time.sleep(1.5)
                
                # Generate detections
                detections = generate_camera_data(
                    distance=distance,
                    weather=weather,
                    time_of_day=time_of_day,
                    traffic_density=traffic_density
                )
                
                # Store in session state for other tabs
                st.session_state['latest_detections'] = detections
                st.session_state['current_speed'] = current_speed
                
                # Make driving decision
                decision, new_speed = make_driving_decision(detections, current_speed)
                st.session_state['latest_decision'] = decision
                st.session_state['new_speed'] = new_speed
                
                # Display results
                st.success("Detection complete!")
                
                # Add to history
                st.session_state['prediction_history'].append({
                    'timestamp': time.strftime("%H:%M:%S"),
                    'weather': weather,
                    'time_of_day': time_of_day,
                    'detections': len(detections),
                    'decision': decision
                })
    
    with col2:
        st.subheader("Detection Analysis")
        
        if 'latest_detections' in st.session_state:
            detections = st.session_state['latest_detections']
            
            # Show detection counts
            st.metric("Objects Detected", len(detections))
            
            # Detection breakdown
            detection_types = {}
            for det in detections:
                if det['type'] not in detection_types:
                    detection_types[det['type']] = 0
                detection_types[det['type']] += 1
            
            if detection_types:
                fig = px.pie(
                    values=list(detection_types.values()),
                    names=list(detection_types.keys()),
                    title="Detection Breakdown",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_traces(textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show decision
            if 'latest_decision' in st.session_state:
                st.info(f"**ML Decision:** {st.session_state['latest_decision']}")
                speed_change = st.session_state['new_speed'] - st.session_state['current_speed']
                if speed_change < 0:
                    st.warning(f"Speed adjustment: {speed_change} mph")
                elif speed_change > 0:
                    st.success(f"Speed adjustment: +{speed_change} mph")
                else:
                    st.info("Speed maintained")
        else:
            st.info("Run the detection simulation to see results")
    
    # Visualization
    st.subheader("Visual Detection Map")
    if 'latest_detections' in st.session_state:
        detection_fig = visualize_detections(st.session_state['latest_detections'])
        if detection_fig:
            st.pyplot(detection_fig)
        
        # Details table
        st.subheader("Detected Objects Details")
        if st.session_state['latest_detections']:
            detections_df = pd.DataFrame(st.session_state['latest_detections'])
            st.dataframe(detections_df, use_container_width=True)
        else:
            st.write("No objects detected")
    else:
        st.info("Run the detection simulation to see visualization")

# Model performance tab
def model_performance_tab():
    st.header("📊 Model Performance and Metrics")
    
    st.markdown("""
    <div class="highlight">
    The performance of ML models for autonomous vehicles is critical to ensure safety and reliability.
    Different conditions affect model performance in various ways.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Detection Performance by Condition")
        
        # Sample performance data
        perf_data = {
            'Condition': ['Clear Day', 'Clear Night', 'Rain Day', 'Rain Night', 
                         'Snow Day', 'Snow Night', 'Fog Day', 'Fog Night'],
            'Vehicle Detection': [0.97, 0.89, 0.88, 0.79, 0.82, 0.73, 0.76, 0.62],
            'Pedestrian Detection': [0.95, 0.85, 0.84, 0.72, 0.78, 0.68, 0.70, 0.58],
            'Sign Detection': [0.98, 0.92, 0.90, 0.84, 0.86, 0.78, 0.80, 0.72]
        }
        
        perf_df = pd.DataFrame(perf_data)
        
        # Performance chart
        performance_metric = st.selectbox(
            "Select Detection Type", 
            ["Vehicle Detection", "Pedestrian Detection", "Sign Detection"]
        )
        
        fig = px.bar(
            perf_df, 
            x='Condition', 
            y=performance_metric, 
            color=performance_metric,
            color_continuous_scale='RdYlGn',
            title=f"{performance_metric} Accuracy by Condition"
        )
        fig.update_layout(yaxis_range=[0.5, 1.0])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Key Insights:
        - Weather conditions significantly impact detection accuracy
        - Night time decreases performance across all models
        - Combining poor weather with night time creates the most challenging conditions
        - Vehicle detection generally performs better than pedestrian detection in adverse conditions
        """)
    
    with col2:
        st.subheader("Detection Distance Limitations")
        
        # Sample distance limitation data
        distance_data = {
            'Condition': ['Clear', 'Rain', 'Snow', 'Fog'],
            'Camera': [100, 70, 50, 30],
            'LiDAR': [200, 180, 150, 100],
            'Radar': [250, 220, 200, 180]
        }
        
        distance_df = pd.DataFrame(distance_data)
        
        # Melt the dataframe for better plotting
        distance_df_melted = pd.melt(
            distance_df, 
            id_vars='Condition', 
            var_name='Sensor', 
            value_name='Detection Distance (m)'
        )
        
        fig = px.bar(
            distance_df_melted, 
            x='Condition', 
            y='Detection Distance (m)', 
            color='Sensor',
            barmode='group',
            title="Maximum Detection Distance by Condition and Sensor Type"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Sensor Fusion:
        ML models for autonomous vehicles combine data from multiple sensors to overcome individual sensor limitations:
        
        - **Cameras:** High resolution visual data, but affected by lighting and weather
        - **LiDAR:** Precise 3D mapping, less affected by lighting but degraded by precipitation
        - **Radar:** Works well in adverse weather but with lower resolution
        - **Ultrasonic:** Short-range but precise object detection
        
        By combining these sensors, ML models can maintain safety even when individual sensors are compromised.
        """)
    
    # Confusion matrix section
    st.subheader("Model Evaluation Metrics")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Confusion matrix data
        classes = ['Car', 'Truck', 'Bike', 'Pedestrian', 'Sign', 'Obstacle']
        conf_matrix = np.array([
            [952, 28, 5, 2, 1, 12],
            [31, 897, 3, 0, 1, 18],
            [8, 2, 824, 42, 0, 4],
            [3, 0, 37, 908, 0, 2],
            [2, 1, 0, 0, 937, 10],
            [15, 21, 7, 3, 8, 896]
        ])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=classes,
            yticklabels=classes,
            ax=ax
        )
        plt.title('Confusion Matrix for Object Detection Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
    
    with col2:
        # Performance metrics
        metrics_data = {
            'Class': classes,
            'Precision': [0.94, 0.92, 0.89, 0.95, 0.99, 0.91],
            'Recall': [0.95, 0.94, 0.93, 0.96, 0.98, 0.94],
            'F1-Score': [0.945, 0.930, 0.909, 0.955, 0.985, 0.925],
            'Support': [1000, 950, 880, 950, 950, 950]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("""
        ### Understanding Metrics:
        
        - **Precision:** Proportion of positive identifications that were actually correct
        - **Recall:** Proportion of actual positives that were identified correctly
        - **F1-Score:** Harmonic mean of precision and recall
        - **Support:** Number of actual occurrences of the class in the test set
        
        For autonomous vehicles, high recall for critical objects like pedestrians and vehicles is particularly important for safety.
        """)

# Data exploration tab
def data_exploration_tab():
    st.header("🔍 Training Data for Autonomous Vehicle ML")
    
    st.markdown("""
    <div class="highlight">
    Autonomous vehicles require massive datasets for training accurate ML models. The quality and diversity of this data directly impacts model performance and safety.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Data Collection Methods")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://images.unsplash.com/photo-1617704548623-340376564e68?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80", 
                 caption="Real-world test drives")
        st.markdown("""
        **Real-world Test Drives**
        - Fleet vehicles with sensors
        - Millions of miles of driving data
        - Various road types, weather, traffic
        - Captures unexpected scenarios
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1614064641938-3bbee52942c7?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80", 
                 caption="Simulation environments")
        st.markdown("""
        **Simulation Environments**
        - Virtual testing environments
        - Controllable conditions
        - Scaling rare event testing
        - Testing edge cases safely
        """)
    
    with col3:
        st.image("https://images.unsplash.com/photo-1580927752452-89d86da3fa0a?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80", 
                 caption="Synthetic data generation")
        st.markdown("""
        **Synthetic Data Generation**
        - AI-generated scenarios
        - Balancing dataset classes
        - Augmenting rare events
        - Reducing data collection costs
        """)
    
    st.markdown("---")
    
    # Dataset statistics
    st.subheader("Training Dataset Statistics")
    
    # Example dataset composition
    dataset_composition = {
        'Category': ['Urban Roads', 'Highways', 'Rural Roads', 'Intersections', 'Parking Lots', 'Construction Zones'],
        'Images (millions)': [45, 32, 18, 25, 10, 8],
        'Video Hours': [8500, 6200, 3400, 4800, 1900, 1500],
        'Annotated Objects (millions)': [320, 180, 90, 210, 75, 60]
    }
    
    df_composition = pd.DataFrame(dataset_composition)
    
    # Plot composition
    fig = px.bar(
        df_composition, 
        x='Category', 
        y='Images (millions)',
        color='Category',
        title="Training Dataset Composition"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        weather_dist = {
            'Weather': ['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog', 'Other'],
            'Percentage': [62, 18, 12, 5, 2, 1]
        }
        df_weather = pd.DataFrame(weather_dist)
        
        fig = px.pie(
            df_weather,
            values='Percentage',
            names='Weather',
            title='Weather Condition Distribution in Dataset',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        time_dist = {
            'Time': ['Daytime', 'Dawn/Dusk', 'Night'],
            'Percentage': [70, 15, 15]
        }
        df_time = pd.DataFrame(time_dist)
        
        fig = px.pie(
            df_time,
            values='Percentage',
            names='Time',
            title='Lighting Condition Distribution in Dataset',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Data Annotation Process")
    
    st.markdown("""
    ### How ML Training Data is Annotated:
    
    1. **Object Bounding Box Annotation**
       - Drawing boxes around vehicles, pedestrians, signs, etc.
       - Labeling object class and attributes
       
    2. **Semantic Segmentation**
       - Pixel-level classification of roads, sidewalks, vegetation
       - Essential for understanding drivable surfaces
       
    3. **Instance Segmentation**
       - Distinguishing individual objects of the same class
       - Important for tracking multiple vehicles/pedestrians
       
    4. **3D Point Cloud Annotation**
       - Labeling LiDAR data points
       - Creating 3D bounding boxes
       
    5. **Temporal Annotation**
       - Tracking objects across video frames
       - Understanding object movement and behavior
    """)
    
    # Annotation quality metrics
    annotation_metrics = {
        'Metric': ['Inter-annotator Agreement', 'Annotation Precision', 'Label Consistency', 'Temporal Coherence', 'Edge Case Coverage'],
        'Score (%)': [94, 96, 93, 91, 87]
    }
    
    df_metrics = pd.DataFrame(annotation_metrics)
    
    fig = px.bar(
        df_metrics,
        x='Metric',
        y='Score (%)',
        color='Score (%)',
        color_continuous_scale='Viridis',
        title="Annotation Quality Metrics"
    )
    
    fig.update_layout(yaxis_range=[80, 100])
    st.plotly_chart(fig, use_container_width=True)

# Advanced concepts tab
def advanced_concepts_tab():
    st.header("🧠 Advanced ML Techniques for Autonomous Vehicles")
    
    st.markdown("""
    <div class="highlight">
    Beyond basic object detection, autonomous vehicles employ sophisticated ML techniques to navigate safely in complex environments.
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs within the advanced concepts tab
    adv_tab1, adv_tab2, adv_tab3 = st.tabs([
        "Path Planning & Decision Making", 
        "Reinforcement Learning", 
        "Uncertainty Handling"
    ])
    
    with adv_tab1:
        st.subheader("Path Planning & Decision Making")
        
        st.markdown("""
        ### How ML Models Plan Safe Paths:
        
        ML models in autonomous vehicles need to plan paths that are:
        - Safe for passengers and others
        - Compliant with traffic rules
        - Efficient to reach the destination
        - Comfortable for passengers
        """)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Example path planning visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Road boundaries
            ax.plot([-50, 50], [0, 0], 'k-', linewidth=2)
            ax.plot([-50, 50], [10, 10], 'k-', linewidth=2)
            ax.plot([-50, 50], [20, 20], 'k--', linewidth=1, color='gray')
            
            # Obstacles
            ax.plot([-20, -20], [2, 8], 'rs', markersize=12)  # Vehicle 1
            ax.plot([10, 10], [12, 18], 'rs', markersize=12)  # Vehicle 2
            ax.plot([30], [5], 'go', markersize=8)  # Pedestrian
            
            # Path options
            x = np.linspace(-40, 40, 100)
            y1 = 5 + 0.5 * np.sin(x/10)  # Path 1
            y2 = 15 + 0.5 * np.sin(x/10)  # Path 2
            
            ax.plot(x, y1, 'b-', alpha=0.7, linewidth=3, label='Path Option 1')
            ax.plot(x, y2, 'g-', alpha=0.7, linewidth=3, label='Path Option 2')
            
            # Ego vehicle
            ax.plot([-40], [5], 'bs', markersize=15, label='Autonomous Vehicle')
            
            ax.set_xlim(-50, 50)
            ax.set_ylim(-5, 25)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title('ML-based Path Planning Example')
            ax.legend()
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### Decision Factors:
            
            ML models prioritize different factors when planning paths:
            
            1. **Safety Margin** - Distance kept from obstacles
            2. **Rule Compliance** - Following traffic laws
            3. **Comfort** - Smooth acceleration and turns
            4. **Progress** - Efficient route to destination
            5. **Predictability** - Behaving in expected ways
            
            Different models may weight these factors differently based on context, but safety is always the top priority.
            """)
            
            # Example decision weights
            decision_weights = {
                'Factor': ['Safety', 'Rule Compliance', 'Comfort', 'Progress', 'Predictability'],
                'Weight': [0.45, 0.25, 0.10, 0.10, 0.10]
            }
            
            df_weights = pd.DataFrame(decision_weights)
            
            fig = px.bar(
                df_weights, 
                x='Factor', 
                y='Weight', 
                color='Weight',
                color_continuous_scale='Blues',
                title="Decision Factor Weights"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### ML Path Planning Techniques:
        
        1. **Search-based Planning**
           - A* algorithm for finding optimal paths
           - Rapidly exploring random trees (RRT)
        
        2. **Sampling-based Planning**
           - Probabilistic roadmaps
           - Monte Carlo tree search
        
        3. **Optimization-based Planning**
           - Model predictive control (MPC)
           - Optimal control theory
        
        4. **Behavior Prediction Integration**
           - Predicting other road users' intentions
           - Adjusting plans based on predicted behavior
        """)
    
    with adv_tab2:
        st.subheader("Reinforcement Learning for Autonomous Vehicles")
        
        st.markdown("""
        Reinforcement Learning (RL) is particularly valuable for autonomous vehicles as it allows the AI to learn optimal driving policies through experience and reward systems.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### How RL Works for Autonomous Driving:
            
            1. **Agent**: The autonomous vehicle control system
            
            2. **Environment**: Roads, other vehicles, pedestrians, weather
            
            3. **State**: Current position, speed, sensor readings
            
            4. **Actions**: Steering, acceleration, braking
            
            5. **Reward**: Safety, progress toward destination, comfort, efficiency
            
            6. **Policy**: Learned strategy mapping states to actions
            
            The RL model learns to maximize cumulative rewards by taking optimal actions in different states.
            """)
            
            st.image("https://miro.medium.com/max/1400/1*wMmktnYIzC7-UrZZiRhTBw.png", caption="Reinforcement Learning Process")
        
        with col2:
            st.markdown("""
            ### Applications of RL in Autonomous Vehicles:
            
            - **Adaptive Cruise Control**
              - Learning optimal following distances
              - Responding to various traffic patterns
            
            - **Lane Changing Decisions**
              - When to change lanes safely
              - How to execute smooth transitions
            
            - **Negotiating Complex Intersections**
              - Learning right-of-way protocols
              - Managing unstructured intersections
            
            - **Dealing with Edge Cases**
              - Unusual road conditions
              - Rare traffic situations
            
            - **Energy Management**
              - Optimizing battery use in electric vehicles
              - Maximizing range and efficiency
            """)
            
            # Example RL learning curve
            steps = np.arange(0, 1000)
            rewards = 100 * (1 - np.exp(-steps/200)) + 5 * np.sin(steps/30) + np.random.normal(0, 5, size=1000)
            
            fig = px.line(
                x=steps, 
                y=rewards,
                labels={'x': 'Training Episodes', 'y': 'Average Reward'},
                title="Reinforcement Learning Progress"
            )
            
            fig.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Expert Performance")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Challenges in RL for Autonomous Vehicles:
        
        1. **Safety During Learning**
           - Cannot learn unsafe behaviors in real traffic
           - Need for simulation before deployment
           
        2. **Transfer from Simulation to Reality**
           - Bridging sim-to-real gap
           - Domain adaptation techniques
           
        3. **Sample Efficiency**
           - Need many experiences to learn
           - Improving learning with fewer trials
           
        4. **Interpretability**
           - Understanding why the model makes specific decisions
           - Satisfying regulatory requirements
        """)
    
    with adv_tab3:
        st.subheader("Handling Uncertainty in ML Models")
        
        st.markdown("""
        Autonomous vehicles must operate in an uncertain world. Advanced ML models explicitly account for uncertainty in their predictions to make safer decisions.
        """)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Sources of Uncertainty:
            
            1. **Sensor Uncertainty**
               - Noise in camera, LiDAR, radar measurements
               - Occlusions and limited sensor range
            
            2. **Prediction Uncertainty**
               - Future positions of other road users
               - Intent of pedestrians and other drivers
            
            3. **Map and Localization Uncertainty**
               - GPS inaccuracies
               - Outdated map information
            
            4. **Model Uncertainty**
               - ML model's confidence in its predictions
               - Out-of-distribution scenarios
            """)
            
            # Example visualization of uncertainty
            st.image("https://autonomoustuff.com/wp-content/uploads/2019/11/Applied-Autonomy-Development-Pathing-software-1920x1080-1-1024x576.jpg", 
                     caption="Visualization of path uncertainty")
        
        with col2:
            st.markdown("""
            ### ML Techniques for Handling Uncertainty:
            
            1. **Bayesian Neural Networks**
               - Estimate probability distributions over outputs
               - Provide confidence intervals
            
            2. **Monte Carlo Dropout**
               - Run multiple forward passes
               - Get uncertainty estimates
            
            3. **Ensemble Methods**
               - Use multiple diverse models
               - Aggregate their predictions
            
            4. **Evidential Deep Learning**
               - Direct modeling of uncertainty
               - Separating aleatoric and epistemic uncertainty
            
            5. **Probabilistic Motion Planning**
               - Multiple possible future trajectories
               - Risk assessment for each option
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### Example: Uncertainty in Pedestrian Trajectory Prediction
        """)
        
        # Create visualization of pedestrian trajectory prediction with uncertainty
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Road
        ax.plot([-50, 50], [0, 0], 'k-', linewidth=2)
        ax.plot([-50, 50], [15, 15], 'k-', linewidth=2)
        
        # Car
        ax.plot([0], [5], 'bs', markersize=15, label='Autonomous Vehicle')
        
        # Pedestrian
        ax.plot([20], [12], 'ro', markersize=10, label='Pedestrian')
        
        # Pedestrian potential trajectories with uncertainty
        x_ped = np.linspace(20, -10, 30)
        
        # Trajectory 1 (crossing road)
        y_ped1 = 12 - 0.8 * (x_ped - 20)
        ax.plot(x_ped, y_ped1, 'r-', alpha=0.7, label='Potential Path 1')
        
        # Uncertainty for trajectory 1
        y_uncertainty1 = np.array([0.2 * np.abs(x - 20) for x in x_ped])
        ax.fill_between(x_ped, y_ped1 - y_uncertainty1, y_ped1 + y_uncertainty1, color='red', alpha=0.2)
        
        # Trajectory 2 (walking along sidewalk)
        y_ped2 = np.ones_like(x_ped) * 12
        ax.plot(x_ped, y_ped2, 'b-', alpha=0.7, label='Potential Path 2')
        
        # Uncertainty for trajectory 2
        y_uncertainty2 = np.array([0.1 * np.abs(x - 20) for x in x_ped])
        ax.fill_between(x_ped, y_ped2 - y_uncertainty2, y_ped2 + y_uncertainty2, color='blue', alpha=0.2)
        
        # Plot settings
        ax.set_xlim(-30, 30)
        ax.set_ylim(-5, 20)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Uncertain Pedestrian Trajectory Prediction')
        
        st.pyplot(fig)
        
        st.markdown("""
        The uncertainty in pedestrian behavior prediction increases with prediction time horizon. 
        Safe autonomous vehicles plan for all plausible trajectories, slowing down when uncertainty is high.
        
        ### Benefits of Uncertainty-Aware ML Models:
        
        1. **Safer Decision Making**
           - Conservative actions when uncertainty is high
           - Confident actions when certainty is high
           
        2. **Better Edge Case Handling**
           - Identifying out-of-distribution inputs
           - Falling back to safe defaults
           
        3. **Explainable Decisions**
           - Quantifying confidence in predictions
           - Justifying actions based on uncertainty
           
        4. **More Human-Like Driving**
           - Slowing down in unfamiliar situations
           - Maintaining appropriate caution
        """)

# Main function
def main():
    # Initialize session state
    init_session_state()
    
    # Apply CSS
    local_css()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("Autonomous Vehicles: ML for Object Detection & Navigation")
    st.markdown("Explore how machine learning enables autonomous vehicles to understand and navigate their environment safely.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚗 Object Detection Simulator", 
        "📊 Model Performance", 
        "🔍 Training Data Exploration", 
        "🧠 Advanced ML Concepts"
    ])
    
    with tab1:
        object_detection_tab()
        
    with tab2:
        model_performance_tab()
        
    with tab3:
        data_exploration_tab()
        
    with tab4:
        advanced_concepts_tab()
    
    # Footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

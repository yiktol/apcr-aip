
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime, timedelta

# Define the CSS
def load_css():
    st.markdown("""
    <style>
    /* AWS-Themed Streamlit Configuration - Modern & Engaging */
    @import url('https://fonts.googleapis.com/css2?family=Amazon+Ember:wght@300;400;500;700&display=swap');

    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa;
        font-family: 'Amazon Ember', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #232f3e;
        padding: 1rem;
    }

    [data-testid="stSidebar"] {
        background-color: #263850;
        background-image: linear-gradient(180deg, #263850 0%, #1e2e40 100%);
        color: #ffffff;
        border-right: none;
        border-radius: 0 12px 12px 0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }

    [data-testid="stSidebarNav"] {
        background-color: transparent;
    }

    [data-testid="stSidebarNav"] a {
        color: #ffffff;
        font-weight: 500;
        margin-bottom: 0.5rem;
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebarNav"] a:hover {
        color: #ffffff;
        background-color: rgba(255, 153, 0, 0.2);
        transform: translateX(3px);
    }

    h1 {
        color: #232f3e;
        font-weight: 700;
        font-size: 2.25rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(255, 153, 0, 0.3);
        padding-bottom: 0.5rem;
    }

    h2 {
        color: #232f3e;
        font-weight: 600;
        font-size: 1.75rem;
    }

    h3 {
        color: #232f3e;
        font-weight: 500;
        font-size: 1.5rem;
    }

    p {
        font-size: 1rem;
        line-height: 1.6;
        color: #323f4b;
    }

    code {
        background-color: #f0f3f7;
        color: #d13212;
        padding: 0.2em 0.4em;
        border-radius: 4px;
        font-size: 0.85em;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.05);
    }

    [data-testid="stButton"] button {
        background-color: #ff9900;
        background-image: linear-gradient(135deg, #ff9900 0%, #ffb144 100%);
        color: #232f3e;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
        box-shadow: 0 2px 5px rgba(255, 153, 0, 0.3);
    }

    [data-testid="stButton"] button:hover {
        background-color: #ec7211;
        background-image: linear-gradient(135deg, #ec7211 0%, #ff9900 100%);
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 153, 0, 0.4);
    }

    [data-testid="stDownloadButton"] button {
        background-color: #0073bb;
        background-image: linear-gradient(135deg, #0073bb 0%, #0195eb 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0, 115, 187, 0.3);
        transition: all 0.3s;
    }

    [data-testid="stExpander"] {
        border: 1px solid #e9eff3;
        border-radius: 12px;
        background-color: #ffffff;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
        transition: all 0.3s;
    }

    [data-testid="stExpander"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-1px);
    }

    [data-testid="stTabs"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }

    [data-testid="stTabs"] > div > div > div {
        background-color: #f8f9fa;
        border-bottom: 1px solid #e9eff3;
    }

    [data-testid="stTabs"] > div > div > div > button {
        color: #232f3e;
        font-weight: 500;
        padding: 0.8rem 1rem;
        margin-right: 0.3rem;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s;
    }

    [data-testid="stTabs"] > div > div > div > button[aria-selected="true"] {
        color: #ff9900;
        border-bottom: 3px solid #ff9900;
        font-weight: 600;
    }

    [data-testid="stTabs"] > div > div > div:nth-child(2) {
        background-color: #ffffff;
        border: 1px solid #e9eff3;
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 1.5rem;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e9eff3;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.3s;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
    }

    [data-testid="stAlert"][kind="info"] {
        background-color: #f0f7ff;
        border-left: 4px solid #0073bb;
    }

    [data-testid="stAlert"][kind="success"] {
        background-color: #f0fff4;
        border-left: 4px solid #1d8102;
    }

    [data-testid="stAlert"][kind="warning"] {
        background-color: #fffaf0;
        border-left: 4px solid #ff9900;
    }

    [data-testid="stAlert"][kind="error"] {
        background-color: #fff5f5;
        border-left: 4px solid #d13212;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #e9eff3;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #d9e2ec;
        border-radius: 12px;
        padding: 1.2rem;
        transition: all 0.3s;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #ff9900;
        background-color: rgba(255, 153, 0, 0.02);
    }

    hr {
        border: none;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(35, 47, 62, 0.1), rgba(0,0,0,0));
        margin: 1.5rem 0;
    }

    footer {
        color: #687078;
        font-size: 0.8rem;
        border-top: 1px solid #e9eff3;
        padding-top: 1rem;
        margin-top: 2rem;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .element-animation {
        animation: fadeIn 0.6s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the CSS
load_css()

# Main app
def main():
    # Sidebar
    st.sidebar.title("AWS-Styled Demo")
    
    demo_option = st.sidebar.selectbox(
        "Select Demo Section",
        ["Home", "Input Components", "Data Display", "Media Elements", "Layouts & Containers", "Alerts & Messages"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    st.sidebar.checkbox("Enable dark mode (not implemented)")
    st.sidebar.slider("Animation speed", 0, 10, 5)
    st.sidebar.selectbox("Theme variant", ["Default", "High Contrast", "Soft"])
    
    # Count visits using session state
    if 'visits' not in st.session_state:
        st.session_state.visits = 0
    st.session_state.visits += 1
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"👀 Visit count: {st.session_state.visits}")
    st.sidebar.markdown(f"🕒 Current time: {datetime.now().strftime('%H:%M:%S')}")

    # Main content
    if demo_option == "Home":
        show_home()
    elif demo_option == "Input Components":
        show_input_components()
    elif demo_option == "Data Display":
        show_data_display()
    elif demo_option == "Media Elements":
        show_media_elements()
    elif demo_option == "Layouts & Containers":
        show_layouts_containers()
    elif demo_option == "Alerts & Messages":
        show_alerts_messages()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <footer>
        AWS-Styled Streamlit Demo • Created with ❤️ using Streamlit • 
        <a href="https://github.com/example/aws-streamlit-demo">GitHub</a>
    </footer>
    """, unsafe_allow_html=True)

def show_home():
    st.markdown("""
    <div class="element-animation">
        <h1>AWS-Styled Modern Streamlit Demo</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This demo shows off the AWS-styled modern CSS configuration for Streamlit. Explore the various components and see how they're styled with a modern, engaging design that maintains AWS branding.
    
    Use the sidebar to navigate through different sections of the demo.
    """)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Cloud Storage", value="1.2 TB", delta="12% this month")
    with col2:
        st.metric(label="EC2 Instances", value="24", delta="-3 since yesterday", delta_color="inverse")
    with col3:
        st.metric(label="Monthly Cost", value="$432.10", delta="-8.1%", delta_color="inverse") 

    st.markdown("### Key features of this styling:")
    
    st.markdown("""
    - 🎨 Modern AWS color scheme
    - 🔄 Interactive hover effects and animations
    - 📱 Responsive design for all screen sizes
    - 🧩 Consistent styling across all components
    - ✨ Rounded corners and subtle shadows for visual depth
    """)
    
    # Button Demo
    st.subheader("Button Demo")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Standard Button"):
            st.success("Standard button clicked!")
            
    with col2:
        csv = pd.DataFrame({
            'Item': ['AWS EC2', 'AWS S3', 'AWS Lambda'],
            'Usage': [65, 27, 13]
        }).to_csv().encode('utf-8')
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='aws_services.csv',
            mime='text/csv',
        )

def show_input_components():
    st.markdown("""
    <div class="element-animation">
        <h1>Input Components</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("This section demonstrates various input components with the AWS styling.")
    
    # Text inputs
    st.subheader("Text Inputs")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", placeholder="Enter your name")
    with col2:
        email = st.text_input("Email", placeholder="Enter your email")
    
    # Number inputs
    st.subheader("Number Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    with col2:
        instances = st.number_input("EC2 Instances", min_value=1, max_value=100, value=5)
    with col3:
        storage = st.number_input("Storage (GB)", min_value=1, max_value=10000, value=500)
    
    # Selections
    st.subheader("Selection Widgets")
    col1, col2 = st.columns(2)
    with col1:
        service = st.selectbox(
            "Select AWS Service",
            ["EC2", "S3", "Lambda", "DynamoDB", "CloudFront", "RDS"]
        )
    with col2:
        regions = st.multiselect(
            "Select AWS Regions",
            ["us-east-1", "us-west-1", "eu-west-1", "ap-southeast-1", "sa-east-1"],
            ["us-east-1"]
        )
    
    # Sliders
    st.subheader("Sliders")
    col1, col2 = st.columns(2)
    with col1:
        cpu = st.slider("CPU Utilization (%)", 0, 100, 45)
    with col2:
        budget = st.slider("Monthly Budget ($)", 100, 10000, 1500)
    
    # Date & Time
    st.subheader("Date & Time")
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Launch Date", datetime.now())
    with col2:
        time = st.time_input("Maintenance Window Start Time", datetime.now().time())
    
    # Checkbox & Radio
    st.subheader("Checkbox & Radio")
    col1, col2 = st.columns(2)
    with col1:
        options = st.checkbox("Enable advanced options")
        backup = st.checkbox("Enable automatic backups")
        monitoring = st.checkbox("Enable detailed monitoring")
    
    with col2:
        tier = st.radio(
            "Select service tier",
            ["Free Tier", "Developer", "Business", "Enterprise"]
        )
    
    # File uploader
    st.subheader("File Uploader")
    uploaded_file = st.file_uploader("Upload a configuration file", type=["json", "yaml", "txt"])
    if uploaded_file is not None:
        st.success(f"File {uploaded_file.name} successfully uploaded!")

def show_data_display():
    st.markdown("""
    <div class="element-animation">
        <h1>Data Display Components</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("This section shows data visualization and display components.")
    
    # Create sample data
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'EC2 Usage': np.random.randint(10, 100, 10),
        'S3 Storage (GB)': np.random.randint(200, 500, 10),
        'Lambda Invocations': np.random.randint(1000, 10000, 10),
        'Cost ($)': np.random.uniform(100, 500, 10).round(2)
    })
    
    # Simple table
    st.subheader("Tables")
    st.table(data.head(3))
    
    # Interactive dataframe
    st.subheader("Interactive DataFrames")
    st.dataframe(
        data,
        column_config={
            "date": "Date",
            "EC2 Usage": st.column_config.NumberColumn(
                "EC2 Usage (%)",
                help="Average EC2 usage percentage",
                format="%d%%"
            ),
            "S3 Storage (GB)": st.column_config.NumberColumn(
                "S3 Storage",
                help="Total S3 storage used",
                format="%d GB"
            ),
            "Lambda Invocations": st.column_config.NumberColumn(
                "Lambda",
                help="Number of Lambda function invocations",
                format="%d calls"
            ),
            "Cost ($)": st.column_config.NumberColumn(
                "Cost",
                help="Total daily cost",
                format="$%.2f"
            )
        },
        hide_index=True,
    )
    
    # Metrics
    st.subheader("Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Average CPU Utilization",
            value="43%",
            delta="2.5%"
        )
    with col2:
        st.metric(
            label="Network Throughput",
            value="1.2 GB/s",
            delta="-0.1 GB/s",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label="Uptime",
            value="99.99%",
            delta="0.01%"
        )
    
    # Charts
    st.subheader("Charts")
    tab1, tab2, tab3 = st.tabs(["Line Chart", "Bar Chart", "Area Chart"])
    
    with tab1:
        chart_data = data[['date', 'EC2 Usage', 'S3 Storage (GB)']].rename(columns={'date': 'index'}).set_index('index')
        st.line_chart(chart_data)
    
    with tab2:
        chart = alt.Chart(data).mark_bar().encode(
            x='date:T',
            y='Cost ($):Q',
            tooltip=['date', 'Cost ($)']
        ).properties(
            title='Daily Cost'
        )
        st.altair_chart(chart, use_container_width=True)
        
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(data['date'], data['Lambda Invocations'], alpha=0.5, color='#ff9900')
        ax.plot(data['date'], data['Lambda Invocations'], color='#232f3e', linewidth=2)
        ax.set_title('Lambda Invocations')
        ax.set_ylabel('Number of Invocations')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

def show_media_elements():
    st.markdown("""
    <div class="element-animation">
        <h1>Media Elements</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("This section demonstrates media elements and their styling.")
    
    # Image
    st.subheader("Images")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("AWS Architecture Diagram:")
        st.image("https://d1.awsstatic.com/product-marketing/CloudWatch/product-page-diagram-CloudWatch_HIW-v2.6d4f7f51256a0877b94244166575c2557614b4ca.png", 
                 caption="AWS CloudWatch Architecture")
    
    with col2:
        st.markdown("AWS Logo:")
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg", 
                 caption="AWS Logo")
    
    # Video
    st.subheader("Video")
    st.video("https://www.youtube.com/watch?v=IT0zCNEYO84")
    
    # Audio
    st.subheader("Audio")
    st.audio("https://www2.cs.uic.edu/~i101/SoundFiles/StarWars3.wav")
    
    # Progress
    st.subheader("Progress Bar")
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(0, 101, 25):
        my_bar.progress(percent_complete, text=f"{progress_text} ({percent_complete}%)")
        time.sleep(0.1)
    my_bar.progress(100, text="Operation complete!")

def show_layouts_containers():
    st.markdown("""
    <div class="element-animation">
        <h1>Layouts & Containers</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("This section demonstrates various layout options and containers.")
    
    # Columns
    st.subheader("Columns Layout")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Column 1**")
        st.write("This is the first column content.")
        st.button("Column 1 Button")
    
    with col2:
        st.markdown("**Column 2**")
        st.write("This is the second column content.")
        st.button("Column 2 Button")
    
    with col3:
        st.markdown("**Column 3**")
        st.write("This is the third column content.")
        st.button("Column 3 Button")
    
    # Expander
    st.subheader("Expanders")
    
    with st.expander("Click to expand details about EC2"):
        st.write("""
            Amazon Elastic Compute Cloud (Amazon EC2) is a web service that provides secure, 
            resizable compute capacity in the cloud. It is designed to make web-scale cloud 
            computing easier for developers.
        """)
        st.image("https://d1.awsstatic.com/product-marketing/EC2/product-page-diagram_Amazon-EC2-Regular_HIW.095e3a17a816da96e9c505a039da998e92d1e9be.png", 
                 caption="EC2 Architecture")
    
    with st.expander("Click to expand details about S3"):
        st.write("""
            Amazon Simple Storage Service (Amazon S3) is an object storage service that offers 
            industry-leading scalability, data availability, security, and performance.
        """)
        st.image("https://d1.awsstatic.com/s3-pdx/product-marketing/S3/S3_HIW_Diagram.cf4c2bd7a506b7228109bc7c210545217ac669da.png", 
                 caption="S3 Architecture")
    
    # Tabs
    st.subheader("Tabs")
    tab1, tab2, tab3 = st.tabs(["Compute", "Storage", "Database"])
    
    with tab1:
        st.markdown("### AWS Compute Services")
        st.write("""
            - **EC2**: Virtual servers in the cloud
            - **Lambda**: Run code without thinking about servers
            - **Elastic Beanstalk**: Easy-to-use service for deploying and scaling web applications
        """)
    
    with tab2:
        st.markdown("### AWS Storage Services")
        st.write("""
            - **S3**: Object storage built to store and retrieve any amount of data
            - **EBS**: Persistent block storage for EC2 instances
            - **EFS**: Scalable file storage for use with EC2 instances
        """)
    
    with tab3:
        st.markdown("### AWS Database Services")
        st.write("""
            - **RDS**: Relational Database Service for MySQL, PostgreSQL, etc.
            - **DynamoDB**: Fast and flexible NoSQL database service
            - **Aurora**: MySQL and PostgreSQL-compatible relational database
        """)
    
    # Container with custom HTML
    st.subheader("Custom Container")
    st.markdown("""
    <div style="background-color:#ffffff; padding:1.2rem; border-radius:12px; box-shadow: 0 3px 10px rgba(0,0,0,0.05); border: 1px solid #e9eff3;">
        <h4 style="color:#232f3e; margin-top:0;">AWS Best Practices</h4>
        <ul>
            <li>Use IAM roles to manage permissions</li>
            <li>Implement Multi-Factor Authentication (MFA)</li>
            <li>Set up monitoring and logging with CloudWatch</li>
            <li>Use resource tagging for better organization</li>
            <li>Implement automated backups for critical data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def show_alerts_messages():
    st.markdown("""
    <div class="element-animation">
        <h1>Alerts & Messages</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("This section demonstrates various alert types and message displays.")
    
    # Alert boxes
    st.subheader("Alert Types")
    
    st.info("**Info:** Your EC2 instances are running normally. 30 days remaining in free tier.", icon="ℹ️")
    
    st.success("**Success:** Your S3 bucket was created successfully! Files can now be uploaded.", icon="✅")
    
    st.warning(
        "**Warning:** Your AWS Lambda function is approaching its memory limit. Consider upgrading.", 
        icon="⚠️"
    )
    
    st.error(
        "**Error:** Failed to connect to DynamoDB. Check your permissions and network settings.", 
        icon="🚨"
    )
    
    # Exceptions
    st.subheader("Exception Handling")
    
    with st.expander("Show example exception"):
        try:
            division_result = 1 / 0
        except Exception as e:
            st.exception(e)
    
    # Messages with emojis
    st.subheader("Messages with Emojis")
    
    st.markdown("""
    🚀 **Getting Started with AWS**  
    Start building on AWS today with these simple steps.
    
    ⚙️ **Configuration Required**  
    Please configure your AWS credentials to continue.
    
    💰 **Cost Alert**  
    Your AWS bill is projected to exceed your budget this month.
    
    🔄 **Syncing**  
    Your files are being synced to S3. Please wait...
    """)
    
    # Toast messages (simulated)
    st.subheader("Toast Messages (Simulated)")
    
    if st.button("Show Success Toast"):
        st.success("Operation completed successfully!")
        
    if st.button("Show Error Toast"):
        st.error("An error occurred during the operation.")
    
    # Code block
    st.subheader("Code Blocks")
    
    st.code('''
# AWS SDK for Python (Boto3) example
import boto3

# Create an S3 client
s3 = boto3.client('s3')

# List all S3 buckets
response = s3.list_buckets()

# Print bucket names
print('Existing buckets:')
for bucket in response['Buckets']:
    print(f'  {bucket["Name"]}')
''', language='python')

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import json
import time
import uuid
from datetime import datetime, timedelta
import random
import networkx as nx
from PIL import Image
import io
import base64
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import plotly.express as px
from streamlit.components.v1 import html
import utils.authenticate as authenticate

# Set page configuration
st.set_page_config(
    page_title="ML Governance with Amazon SageMaker",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    
    if 'selected_role' not in st.session_state:
        st.session_state.selected_role = "DataScientist"
        
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "fraud_detection_model"
        
    if 'model_cards' not in st.session_state:
        st.session_state.model_cards = generate_model_cards()
    
    if 'model_dashboard_data' not in st.session_state:
        st.session_state.model_dashboard_data = generate_dashboard_data()
    
    if 'roles_data' not in st.session_state:
        st.session_state.roles_data = generate_roles_data()


def reset_session():
    """
    Reset the session state
    """
    # Keep only user_id and reset all other state
    user_id = st.session_state.user_id
    st.session_state.clear()
    st.session_state.user_id = user_id


# Data generation functions
def generate_model_cards():
    """
    Generate example model cards for demonstration
    """
    model_cards = {
        "fraud_detection_model": {
            "name": "Credit Card Fraud Detection Model",
            "version": "1.2",
            "created_by": "Jane Davis",
            "created_date": "2023-10-15",
            "approved_by": "Michael Chen",
            "approval_date": "2023-10-22",
            "status": "Approved",
            "description": "XGBoost model to detect fraudulent credit card transactions",
            "intended_uses": [
                "Real-time transaction screening",
                "Post-transaction fraud investigation",
                "Risk scoring for new accounts"
            ],
            "limitations": [
                "Not designed for business account transactions",
                "Less effective for fraud patterns not seen in training data",
                "May have higher false positive rate for international transactions"
            ],
            "ethical_considerations": [
                "Fairness across demographic groups",
                "Potential for disparate impact on certain user segments",
                "Privacy considerations for transaction data"
            ],
            "metrics": {
                "accuracy": 0.994,
                "precision": 0.967,
                "recall": 0.891,
                "f1_score": 0.928,
                "auc": 0.987,
                "fairness_metrics": {
                    "statistical_parity_difference": 0.017,
                    "disparate_impact": 0.982
                }
            },
            "training_data": {
                "source": "s3://example-bucket/fraud-detection/training/",
                "timeframe": "Jan 2022 to Aug 2023",
                "rows": 3500000,
                "features": 243,
                "preprocessing": [
                    "Normalization",
                    "Missing value imputation",
                    "Feature encoding",
                    "SMOTE for class imbalance"
                ]
            },
            "evaluation_data": {
                "source": "s3://example-bucket/fraud-detection/evaluation/",
                "timeframe": "Sep 2023",
                "rows": 450000
            },
            "deployment": {
                "endpoint": "fraud-detection-prod",
                "instance_type": "ml.c5.2xlarge",
                "auto_scaling": True,
                "monitoring": {
                    "data_quality": True,
                    "model_quality": True,
                    "bias_drift": True,
                    "feature_attribution": True
                }
            },
            "risk_rating": "Medium",
            "risk_mitigations": [
                "Human review for transactions with scores in uncertain range (0.4-0.7)",
                "Regular model retraining with recent data",
                "Monitoring for demographic bias"
            ],
            "lineage": {
                "base_model": "fraud_detection_model_v1.1",
                "training_job": "fraud-detection-train-2023-10-12",
                "git_repository": "https://github.com/example-org/fraud-models",
                "commit_id": "a1b2c3d4e5f6",
                "container": "123456789012.dkr.ecr.us-west-2.amazonaws.com/fraud-model:1.2"
            }
        },
        "customer_churn_model": {
            "name": "Customer Churn Prediction Model",
            "version": "2.4",
            "created_by": "Robert Smith",
            "created_date": "2023-09-05",
            "approved_by": "Emma Wilson",
            "approval_date": "2023-09-12",
            "status": "Approved",
            "description": "Random Forest model to predict customer churn likelihood",
            "intended_uses": [
                "Customer retention campaigns",
                "Proactive customer service interventions",
                "Product improvement prioritization"
            ],
            "limitations": [
                "Optimized for consumer accounts only",
                "Requires at least 3 months of customer history",
                "Not calibrated for new product offerings"
            ],
            "ethical_considerations": [
                "Fairness across customer segments",
                "Avoiding exploitation of vulnerable customers",
                "Transparent use of customer data"
            ],
            "metrics": {
                "accuracy": 0.876,
                "precision": 0.823,
                "recall": 0.791,
                "f1_score": 0.807,
                "auc": 0.914,
                "fairness_metrics": {
                    "statistical_parity_difference": 0.035,
                    "disparate_impact": 0.924
                }
            },
            "training_data": {
                "source": "s3://example-bucket/churn-prediction/training/",
                "timeframe": "Jan 2021 to June 2023",
                "rows": 1200000,
                "features": 187,
                "preprocessing": [
                    "StandardScaler",
                    "Label encoding",
                    "Feature selection with RFE",
                    "Target encoding for categorical variables"
                ]
            },
            "evaluation_data": {
                "source": "s3://example-bucket/churn-prediction/evaluation/",
                "timeframe": "July 2023 to Aug 2023",
                "rows": 150000
            },
            "deployment": {
                "endpoint": "churn-prediction-prod",
                "instance_type": "ml.m5.xlarge",
                "auto_scaling": True,
                "monitoring": {
                    "data_quality": True,
                    "model_quality": True,
                    "bias_drift": False,
                    "feature_attribution": True
                }
            },
            "risk_rating": "Low",
            "risk_mitigations": [
                "Regular retraining schedule (quarterly)",
                "Monitoring for concept drift",
                "A/B testing for interventions"
            ],
            "lineage": {
                "base_model": "customer_churn_model_v2.3",
                "training_job": "churn-train-2023-08-28",
                "git_repository": "https://github.com/example-org/customer-models",
                "commit_id": "f7e6d5c4b3a2",
                "container": "123456789012.dkr.ecr.us-west-2.amazonaws.com/churn-model:2.4"
            }
        },
        "product_recommendation_model": {
            "name": "Product Recommendation Engine",
            "version": "3.7",
            "created_by": "Priya Patel",
            "created_date": "2023-11-03",
            "approved_by": "David Johnson",
            "approval_date": "2023-11-10",
            "status": "Approved",
            "description": "Matrix factorization model for personalized product recommendations",
            "intended_uses": [
                "Homepage personalization",
                "Email marketing campaigns",
                "In-app recommendations",
                "Post-purchase recommendations"
            ],
            "limitations": [
                "Cold start problem for new users and products",
                "Limited context awareness",
                "No consideration of inventory availability"
            ],
            "ethical_considerations": [
                "Filter bubble effects on user experience",
                "Potential reinforcement of biases in purchase patterns",
                "Transparency in recommendation criteria"
            ],
            "metrics": {
                "ndcg@10": 0.842,
                "precision@5": 0.734,
                "recall@20": 0.687,
                "mean_reciprocal_rank": 0.628,
                "fairness_metrics": {
                    "exposure_parity_difference": 0.042,
                    "recommendation_diversity": 0.876
                }
            },
            "training_data": {
                "source": "s3://example-bucket/recommendations/training/",
                "timeframe": "Last 12 months",
                "interactions": 78000000,
                "rows": 78000000,
                "users": 2300000,
                "items": 145000,
                "preprocessing": [
                    "Interaction weighting by recency",
                    "Negative sampling",
                    "Temporal splitting for validation"
                ]
            },
            "evaluation_data": {
                "source": "s3://example-bucket/recommendations/evaluation/",
                "timeframe": "Last 2 weeks",
                "interactions": 5200000,
                "rows": 5200000
            },
            "deployment": {
                "endpoint": "product-recommendations-prod",
                "instance_type": "ml.g4dn.2xlarge",
                "auto_scaling": True,
                "monitoring": {
                    "data_quality": True,
                    "model_quality": True,
                    "bias_drift": True,
                    "feature_attribution": False
                }
            },
            "risk_rating": "Low",
            "risk_mitigations": [
                "Regular retraining with fresh data",
                "Diversity boosting in recommendation lists",
                "A/B testing for algorithm changes"
            ],
            "lineage": {
                "base_model": "product_recommendation_model_v3.6",
                "training_job": "recs-train-2023-10-30",
                "git_repository": "https://github.com/example-org/recommendation-engine",
                "commit_id": "1a2b3c4d5e6f7g8h",
                "container": "123456789012.dkr.ecr.us-west-2.amazonaws.com/recs-model:3.7"
            }
        }
    }
    
    return model_cards


def generate_dashboard_data():
    """
    Generate example data for model dashboard
    """
    # Generate dates for the past 30 days
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Create data for multiple models
    models = {
        "fraud_detection_model": {
            "name": "Credit Card Fraud Detection",
            "endpoint": "fraud-detection-prod",
            "version": "1.2",
            "status": "Healthy",
            "instance_type": "ml.c5.2xlarge",
            "creation_date": "2023-10-22",
            "last_updated": "2023-12-03",
            "invocations": [random.randint(85000, 110000) for _ in range(30)],
            "latency": [random.uniform(15, 25) for _ in range(30)],
            "cpu_utilization": [random.uniform(35, 65) for _ in range(30)],
            "memory_utilization": [random.uniform(40, 75) for _ in range(30)],
            "accuracy": [random.uniform(0.984, 0.996) for _ in range(30)],
            "data_drift": [random.uniform(0.01, 0.08) for _ in range(30)],
            "model_drift": [random.uniform(0.02, 0.09) for _ in range(30)],
            "drift_status": "No significant drift",
            "alerts": [
                {
                    "date": "2023-12-01",
                    "type": "Data Quality",
                    "message": "Missing values in feature 'transaction_amount' exceeded threshold (1.2%)",
                    "severity": "Low"
                }
            ]
        },
        "customer_churn_model": {
            "name": "Customer Churn Prediction",
            "endpoint": "churn-prediction-prod",
            "version": "2.4",
            "status": "Warning",
            "instance_type": "ml.m5.xlarge",
            "creation_date": "2023-09-12",
            "last_updated": "2023-12-05",
            "invocations": [random.randint(25000, 40000) for _ in range(30)],
            "latency": [random.uniform(8, 18) for _ in range(30)],
            "cpu_utilization": [random.uniform(25, 55) for _ in range(30)],
            "memory_utilization": [random.uniform(30, 65) for _ in range(30)],
            "accuracy": [random.uniform(0.86, 0.89) - (i * 0.001) for i in range(30)],  # Declining accuracy
            "data_drift": [random.uniform(0.03, 0.06) + (i * 0.003) for i in range(30)],  # Increasing drift
            "model_drift": [random.uniform(0.04, 0.07) + (i * 0.004) for i in range(30)],  # Increasing drift
            "drift_status": "Moderate concept drift detected",
            "alerts": [
                {
                    "date": "2023-12-04",
                    "type": "Model Quality",
                    "message": "Accuracy declined from 0.876 to 0.853 in the last 7 days",
                    "severity": "Medium"
                },
                {
                    "date": "2023-11-28",
                    "type": "Data Drift",
                    "message": "Feature 'subscription_type' distribution changed significantly (KL divergence: 0.23)",
                    "severity": "Medium"
                }
            ]
        },
        "product_recommendation_model": {
            "name": "Product Recommendation Engine",
            "endpoint": "product-recommendations-prod",
            "version": "3.7",
            "status": "Healthy",
            "instance_type": "ml.g4dn.2xlarge",
            "creation_date": "2023-11-10",
            "last_updated": "2023-11-30",
            "invocations": [random.randint(150000, 250000) for _ in range(30)],
            "latency": [random.uniform(30, 45) for _ in range(30)],
            "cpu_utilization": [random.uniform(45, 80) for _ in range(30)],
            "memory_utilization": [random.uniform(50, 85) for _ in range(30)],
            "ndcg_score": [random.uniform(0.83, 0.86) for _ in range(30)],
            "data_drift": [random.uniform(0.02, 0.07) for _ in range(30)],
            "model_drift": [random.uniform(0.01, 0.06) for _ in range(30)],
            "drift_status": "No significant drift",
            "alerts": []
        }
    }
    
    # Create dataframes for each model
    dashboard_data = {}
    
    for model_id, model_data in models.items():
        # Time series data
        ts_data = pd.DataFrame({
            'date': dates,
            'invocations': model_data['invocations'],
            'latency': model_data['latency'],
            'cpu_utilization': model_data['cpu_utilization'],
            'memory_utilization': model_data['memory_utilization'],
            'data_drift': model_data['data_drift'],
            'model_drift': model_data['model_drift'],
        })
        
        if 'accuracy' in model_data:
            ts_data['accuracy'] = model_data['accuracy']
        
        if 'ndcg_score' in model_data:
            ts_data['ndcg_score'] = model_data['ndcg_score']
        
        # Static information
        static_info = {
            'name': model_data['name'],
            'endpoint': model_data['endpoint'],
            'version': model_data['version'],
            'status': model_data['status'],
            'instance_type': model_data['instance_type'],
            'creation_date': model_data['creation_date'],
            'last_updated': model_data['last_updated'],
            'drift_status': model_data['drift_status'],
            'alerts': model_data['alerts']
        }
        
        dashboard_data[model_id] = {
            'time_series': ts_data,
            'info': static_info
        }
    
    return dashboard_data


def generate_roles_data():
    """
    Generate example roles and permissions data for SageMaker Role Manager
    """
    roles = {
        "DataScientist": {
            "description": "Build, train, and deploy machine learning models",
            "permissions": [
                {"name": "Train models", "access": "Full", "description": "Create and run training jobs", "actions": ["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob"]},
                {"name": "Deploy models", "access": "Full", "description": "Create endpoints and deploy models", "actions": ["sagemaker:CreateEndpoint", "sagemaker:UpdateEndpoint"]},
                {"name": "Access data", "access": "Limited", "description": "Access training and validation data", "actions": ["s3:GetObject", "s3:ListBucket"]},
                {"name": "Notebook instances", "access": "Full", "description": "Create and manage notebook instances", "actions": ["sagemaker:CreateNotebookInstance", "sagemaker:StartNotebookInstance"]},
                {"name": "Feature Store", "access": "Full", "description": "Create and use feature groups", "actions": ["sagemaker:CreateFeatureGroup", "sagemaker:BatchGetRecord"]},
                {"name": "Experiments", "access": "Full", "description": "Track and compare experiments", "actions": ["sagemaker:CreateExperiment", "sagemaker:CreateTrial"]},
                {"name": "SageMaker Canvas", "access": "Read-only", "description": "View Canvas applications", "actions": ["sagemaker:DescribeApp"]},
                {"name": "Model Registry", "access": "Limited", "description": "Register and version models", "actions": ["sagemaker:RegisterModel", "sagemaker:DescribeModelPackage"]},
                {"name": "ML Lineage", "access": "Full", "description": "Track artifacts and lineage", "actions": ["sagemaker:AddTags", "sagemaker:CreateArtifact"]},
            ],
            "persona": "Data scientists who need to develop and deploy ML models",
            "aws_managed_policies": ["AmazonSageMakerFullAccess"],
            "resource_scope": ["arn:aws:sagemaker:*:*:training-job/*", "arn:aws:sagemaker:*:*:endpoint/*"],
            "data_access": {
                "s3_buckets": ["sagemaker-{region}-{account_id}", "company-ml-training-data"],
                "conditions": {
                    "s3:ResourceTag/Project": ["allowed-projects"],
                    "s3:ResourceTag/Environment": ["dev", "test"]
                }
            }
        },
        "MLOpsEngineer": {
            "description": "Build and manage ML infrastructure and pipelines",
            "permissions": [
                {"name": "CI/CD pipelines", "access": "Full", "description": "Create and manage ML pipelines", "actions": ["sagemaker:CreatePipeline", "sagemaker:UpdatePipeline"]},
                {"name": "Model deployment", "access": "Full", "description": "Deploy and configure endpoints", "actions": ["sagemaker:CreateEndpoint", "sagemaker:UpdateEndpointWeightsAndCapacities"]},
                {"name": "Model monitoring", "access": "Full", "description": "Configure and manage monitors", "actions": ["sagemaker:CreateMonitoringSchedule", "sagemaker:DescribeMonitoringSchedule"]},
                {"name": "Model registry", "access": "Full", "description": "Manage model registry", "actions": ["sagemaker:CreateModel", "sagemaker:CreateModelPackageGroup"]},
                {"name": "Infrastructure", "access": "Full", "description": "Manage SageMaker resources", "actions": ["sagemaker:CreateDomain", "sagemaker:CreateApp"]},
                {"name": "Training jobs", "access": "Limited", "description": "View training jobs", "actions": ["sagemaker:DescribeTrainingJob", "sagemaker:ListTrainingJobs"]},
                {"name": "Access data", "access": "Limited", "description": "Access deployment configurations", "actions": ["s3:GetObject", "s3:ListBucket"]},
                {"name": "CloudFormation", "access": "Full", "description": "Manage infrastructure as code", "actions": ["cloudformation:CreateStack", "cloudformation:UpdateStack"]},
                {"name": "EventBridge", "access": "Full", "description": "Configure event-driven automation", "actions": ["events:PutRule", "events:PutTargets"]},
            ],
            "persona": "MLOps engineers who manage ML infrastructure and automation",
            "aws_managed_policies": ["AmazonSageMakerFullAccess", "AWSCloudFormationFullAccess"],
            "resource_scope": ["arn:aws:sagemaker:*:*:pipeline/*", "arn:aws:sagemaker:*:*:endpoint/*", "arn:aws:sagemaker:*:*:model-package-group/*"],
            "data_access": {
                "s3_buckets": ["sagemaker-{region}-{account_id}", "company-ml-models", "company-ml-configs"],
                "conditions": {
                    "s3:ResourceTag/Environment": ["dev", "test", "prod"]
                }
            }
        },
        "ModelGovernanceOfficer": {
            "description": "Ensure ML governance, compliance, and responsible AI practices",
            "permissions": [
                {"name": "Model Cards", "access": "Full", "description": "Create and review model cards", "actions": ["sagemaker:CreateModelCard", "sagemaker:UpdateModelCard"]},
                {"name": "Model registry", "access": "Read-only", "description": "Review registered models", "actions": ["sagemaker:DescribeModelPackage", "sagemaker:ListModelPackages"]},
                {"name": "Model dashboard", "access": "Full", "description": "Access model monitoring dashboards", "actions": ["sagemaker:DescribeModelDashboard", "sagemaker:ListMonitoringAlerts"]},
                {"name": "Model bias", "access": "Full", "description": "Analyze model bias reports", "actions": ["sagemaker:CreateMonitoringSchedule", "sagemaker:DescribeMonitoringSchedule"]},
                {"name": "Model explainability", "access": "Full", "description": "Review model explanations", "actions": ["sagemaker:SendShapValues", "sagemaker:InvokeEndpoint"]},
                {"name": "Model lineage", "access": "Full", "description": "Track model and data lineage", "actions": ["sagemaker:QueryLineage", "sagemaker:ListArtifacts"]},
                {"name": "Access data", "access": "Limited", "description": "Access model evaluation data", "actions": ["s3:GetObject", "s3:ListBucket"]},
                {"name": "Tags and labels", "access": "Full", "description": "Manage governance tags", "actions": ["sagemaker:AddTags", "sagemaker:ListTags"]},
                {"name": "Security scanning", "access": "Full", "description": "Run security and compliance scans", "actions": ["sagemaker:CreateProcessingJob", "sagemaker:DescribeProcessingJob"]},
            ],
            "persona": "Governance officers who ensure responsible ML and compliance",
            "aws_managed_policies": ["AmazonSageMakerReadOnly", "AmazonS3ReadOnlyAccess"],
            "resource_scope": ["arn:aws:sagemaker:*:*:model-card/*", "arn:aws:sagemaker:*:*:model-package/*"],
            "data_access": {
                "s3_buckets": ["company-ml-audit", "company-ml-compliance"],
                "conditions": {
                    "s3:ResourceTag/Compliance": ["required"],
                    "s3:ResourceTag/Sensitivity": ["high", "medium", "low"]
                }
            }
        },
        "BusinessAnalyst": {
            "description": "Analyze ML results and derive business insights",
            "permissions": [
                {"name": "Canvas", "access": "Full", "description": "Use no-code ML building", "actions": ["sagemaker:CreateApp", "sagemaker:DescribeApp"]},
                {"name": "Model deployment", "access": "None", "description": "No deployment permissions", "actions": []},
                {"name": "Model invocation", "access": "Limited", "description": "Use deployed endpoints", "actions": ["sagemaker:InvokeEndpoint"]},
                {"name": "Access data", "access": "Limited", "description": "Access business data", "actions": ["s3:GetObject", "s3:ListBucket"]},
                {"name": "QuickSight", "access": "Full", "description": "Create ML-powered dashboards", "actions": ["quicksight:CreateAnalysis", "quicksight:UpdateDashboard"]},
                {"name": "Athena", "access": "Full", "description": "Query data for analysis", "actions": ["athena:StartQueryExecution", "athena:GetQueryResults"]},
                {"name": "Model tracking", "access": "Read-only", "description": "View model metrics", "actions": ["sagemaker:DescribeModelQualityJobDefinition"]},
                {"name": "Feature Store", "access": "Read-only", "description": "Query features", "actions": ["sagemaker:BatchGetRecord", "sagemaker:GetRecord"]},
                {"name": "SageMaker Studio", "access": "Limited", "description": "Access limited Studio apps", "actions": ["sagemaker:CreatePresignedDomainUrl"]},
            ],
            "persona": "Business analysts who use ML insights but don't build models",
            "aws_managed_policies": ["AmazonSageMakerCanvasFullAccess", "AmazonQuickSightUserAccess"],
            "resource_scope": ["arn:aws:sagemaker:*:*:app/*", "arn:aws:sagemaker:*:*:endpoint/*/invocations"],
            "data_access": {
                "s3_buckets": ["company-business-data", "company-analytics"],
                "conditions": {
                    "s3:ResourceTag/Department": ["marketing", "sales", "finance"],
                    "s3:ResourceTag/Sensitivity": ["low", "medium"]
                }
            }
        }
    }
    
    return roles


def load_lottie_url(url: str):
    """
    Load lottie animation from URL
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


# Visualization functions
def create_model_card_visualization(model_card):
    """
    Create a visualization of a model card
    """
    # Create tabs for different sections of the model card
    tabs = st.tabs([
        "📋 Overview",
        "📊 Performance",
        "🧠 Training",
        "⚠️ Risks",
        "🔄 Lineage"
    ])
    
    # Overview tab
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(model_card["name"])
            st.markdown(f"**Version:** {model_card['version']}")
            st.markdown(f"**Status:** {model_card['status']}")
            st.markdown(f"**Description:** {model_card['description']}")
            
            st.markdown("### Intended Uses")
            for use in model_card["intended_uses"]:
                st.markdown(f"- {use}")
                
            st.markdown("### Limitations")
            for limitation in model_card["limitations"]:
                st.markdown(f"- {limitation}")
        
        with col2:
            # Create approval card
            st.markdown("""
            <div style="border: 1px solid #E9ECEF; border-radius: 10px; padding: 15px; margin-bottom: 20px; background-color: #F8F9FA;">
                <h4 style="color: #232F3E;">Approval Information</h4>
                <p><strong>Created by:</strong> {created_by}<br>
                <strong>Created date:</strong> {created_date}<br>
                <strong>Approved by:</strong> {approved_by}<br>
                <strong>Approval date:</strong> {approval_date}</p>
                <div style="text-align: center; margin-top: 10px;">
                    <span style="background-color: #59BA47; color: white; padding: 5px 15px; border-radius: 15px; font-weight: bold;">{status}</span>
                </div>
            </div>
            """.format(
                created_by=model_card["created_by"],
                created_date=model_card["created_date"],
                approved_by=model_card["approved_by"],
                approval_date=model_card["approval_date"],
                status=model_card["status"]
            ), unsafe_allow_html=True)
            
            # Create risk rating
            risk_color = "#59BA47" if model_card["risk_rating"] == "Low" else ("#FF9900" if model_card["risk_rating"] == "Medium" else "#D13212")
            st.markdown(f"""
            <div style="border: 1px solid #E9ECEF; border-radius: 10px; padding: 15px; margin-bottom: 20px; background-color: #F8F9FA;">
                <h4 style="color: #232F3E;">Risk Assessment</h4>
                <div style="text-align: center; margin: 10px 0;">
                    <span style="background-color: {risk_color}; color: white; padding: 8px 20px; border-radius: 15px; font-weight: bold; font-size: 18px;">{model_card["risk_rating"]} Risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance tab
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Key metrics
            st.subheader("Performance Metrics")
            metrics = model_card["metrics"]
            
            # Create metrics visualization
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys())[:5],  # Get only main metrics, not nested ones
                'Value': [metrics[k] for k in list(metrics.keys())[:5]]
            })
            
            # Filter out fairness_metrics
            metrics_df = metrics_df[metrics_df['Metric'] != 'fairness_metrics']
            
            # Create bar chart
            chart = px.bar(
                metrics_df, 
                x='Metric', 
                y='Value',
                text='Value',
                color_discrete_sequence=['#00A1C9'],
                labels={'Value': 'Score', 'Metric': ''}
            )
            
            # Format
            chart.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            chart.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=40),
                yaxis_range=[0, 1.0]
            )
            
            st.plotly_chart(chart, use_column_width=True)
        
        with col2:
            # Fairness metrics if available
            if "fairness_metrics" in metrics:
                st.subheader("Fairness Metrics")
                
                fairness = metrics["fairness_metrics"]
                
                # Create gauge charts for fairness metrics
                for metric_name, value in fairness.items():
                    # Determine color and status based on value
                    if metric_name == "disparate_impact":
                        # For disparate impact, closer to 1.0 is better
                        if value > 0.8 and value < 1.2:
                            color = "#59BA47"  # green
                            status = "Good"
                        elif (value > 0.7 and value < 0.8) or (value > 1.2 and value < 1.3):
                            color = "#FF9900"  # orange
                            status = "Caution"
                        else:
                            color = "#D13212"  # red
                            status = "Concern"
                            
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=value,
                            title={'text': metric_name.replace('_', ' ').title()},
                            gauge={
                                'axis': {'range': [0, 2], 'tickvals': [0, 0.8, 1.0, 1.2, 2]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 0.8], 'color': "#D13212"},
                                    {'range': [0.8, 1.2], 'color': "#59BA47"},
                                    {'range': [1.2, 2], 'color': "#D13212"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 2},
                                    'thickness': 0.8,
                                    'value': value
                                }
                            }
                        ))
                    else:
                        # For other metrics, closer to 0 is better
                        if value < 0.05:
                            color = "#59BA47"  # green
                            status = "Good"
                        elif value < 0.1:
                            color = "#FF9900"  # orange
                            status = "Caution"
                        else:
                            color = "#D13212"  # red
                            status = "Concern"
                            
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=value,
                            title={'text': metric_name.replace('_', ' ').title()},
                            gauge={
                                'axis': {'range': [0, 0.5], 'tickvals': [0, 0.05, 0.1, 0.2, 0.5]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 0.05], 'color': "#59BA47"},
                                    {'range': [0.05, 0.1], 'color': "#FF9900"},
                                    {'range': [0.1, 0.5], 'color': "#D13212"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 2},
                                    'thickness': 0.8,
                                    'value': value
                                }
                            }
                        ))
                        
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_column_width=True)
                    
                    # Add explanation of metric status
                    st.markdown(f"**Status: {status}** - {get_fairness_explanation(metric_name, status)}")
            
            # If no fairness metrics
            else:
                st.info("No fairness metrics available for this model")
    
    # Training tab
    with tabs[2]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Training Data")
            st.markdown(f"**Source:** `{model_card['training_data']['source']}`")
            st.markdown(f"**Timeframe:** {model_card['training_data']['timeframe']}")
            st.markdown(f"**Rows:** {model_card['training_data']['rows']:,}")
            st.markdown(f"**Rows:** {model_card.get('training_data', {}).get('rows', 'N/A'):,}")
            if 'features' in model_card['training_data']:
                st.markdown(f"**Features:** {model_card['training_data']['features']}")
            
            st.subheader("Preprocessing Steps")
            for step in model_card["training_data"]["preprocessing"]:
                st.markdown(f"- {step}")
        
        with col2:
            st.subheader("Evaluation Data")
            st.markdown(f"**Source:** `{model_card['evaluation_data']['source']}`")
            st.markdown(f"**Timeframe:** {model_card['evaluation_data']['timeframe']}")
            st.markdown(f"**Rows:** {model_card['evaluation_data']['rows']:,}")
            st.markdown(f"**Rows:** {model_card.get('training_data', {}).get('rows', 'N/A'):,}")
            
            st.subheader("Deployment Configuration")
            st.markdown(f"**Endpoint:** `{model_card['deployment']['endpoint']}`")
            st.markdown(f"**Instance Type:** {model_card['deployment']['instance_type']}")
            st.markdown(f"**Auto Scaling:** {'Enabled' if model_card['deployment']['auto_scaling'] else 'Disabled'}")
            
            # Monitoring configuration as a visual
            st.subheader("Monitoring Configuration")
            
            monitoring = model_card['deployment']['monitoring']
            
            # Create monitoring status indicators
            cols = st.columns(2)
            
            for i, (monitor_type, enabled) in enumerate(monitoring.items()):
                with cols[i % 2]:
                    status = "✓ Enabled" if enabled else "✗ Disabled"
                    color = "#59BA47" if enabled else "#D13212"
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #E9ECEF; border-radius: 8px; padding: 10px; margin-bottom: 10px; display: flex; align-items: center;">
                        <div style="background-color: {color}; width: 12px; height: 12px; border-radius: 50%; margin-right: 10px;"></div>
                        <div>
                            <span style="font-weight: bold;">{monitor_type.replace('_', ' ').title()}</span><br>
                            <span>{status}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Risks tab
    with tabs[3]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Ethical Considerations")
            for consideration in model_card["ethical_considerations"]:
                st.markdown(f"- {consideration}")
                
            st.subheader("Limitations")
            for limitation in model_card["limitations"]:
                st.markdown(f"- {limitation}")
        
        with col2:
            st.subheader("Risk Mitigations")
            for mitigation in model_card["risk_mitigations"]:
                st.markdown(f"- {mitigation}")
    
    # Lineage tab
    with tabs[4]:
        # Create a graph visualization of the model lineage
        G = nx.DiGraph()
        
        # Add nodes
        G.add_node(model_card["lineage"]["base_model"], type="model", color="#00A1C9")
        G.add_node(model_card["name"], type="model", color="#FF9900")
        G.add_node(model_card["lineage"]["training_job"], type="job", color="#59BA47")
        G.add_node(model_card["lineage"]["git_repository"], type="repo", color="#232F3E")
        G.add_node(model_card["lineage"]["container"], type="container", color="#545B64")
        G.add_node(model_card["training_data"]["source"], type="data", color="#D13212")
        
        # Add edges
        G.add_edge(model_card["lineage"]["base_model"], model_card["name"])
        G.add_edge(model_card["lineage"]["training_job"], model_card["name"])
        G.add_edge(model_card["lineage"]["git_repository"], model_card["lineage"]["training_job"])
        G.add_edge(model_card["lineage"]["container"], model_card["lineage"]["training_job"])
        G.add_edge(model_card["training_data"]["source"], model_card["lineage"]["training_job"])
        
        # Create position layout
        pos = {
            model_card["lineage"]["base_model"]: (0, 1),
            model_card["name"]: (2, 1),
            model_card["lineage"]["training_job"]: (1, 0),
            model_card["lineage"]["git_repository"]: (0, -1),
            model_card["lineage"]["container"]: (1, -1),
            model_card["training_data"]["source"]: (2, -1)
        }
        
        # Draw the graph
        plt.figure(figsize=(10, 6))
        
        # Node colors
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray', arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="white")
        
        # Add a legend
        legend_labels = {
            "model": "Model Version", 
            "job": "Training Job",
            "repo": "Git Repository",
            "container": "Container Image",
            "data": "Training Data"
        }
        
        # Create legend patches
        patches = []
        for node_type, color in [
            ("model", "#00A1C9"), 
            ("model (current)", "#FF9900"),
            ("job", "#59BA47"), 
            ("repo", "#232F3E"),
            ("container", "#545B64"), 
            ("data", "#D13212")
        ]:
            patches.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, 
                                     label=legend_labels.get(node_type.split()[0], node_type)))
            
        plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        # Remove axis
        plt.axis('off')
        plt.tight_layout()
        
        # Display the graph
        st.pyplot(plt)
        
        # Additional lineage information
        st.markdown(f"**Base Model:** {model_card['lineage']['base_model']}")
        st.markdown(f"**Training Job:** {model_card['lineage']['training_job']}")
        st.markdown(f"**Git Repository:** {model_card['lineage']['git_repository']}")
        st.markdown(f"**Commit ID:** {model_card['lineage']['commit_id']}")
        st.markdown(f"**Container Image:** {model_card['lineage']['container']}")


def create_dashboard_visualization(dashboard_data, model_id):
    """
    Create a visualization of the model dashboard for a specific model
    """
    model_data = dashboard_data[model_id]
    info = model_data['info']
    ts_data = model_data['time_series']
    
    # Display model information header
    status_color = "#59BA47" if info['status'] == "Healthy" else ("#FF9900" if info['status'] == "Warning" else "#D13212")
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <div style="font-size: 24px; font-weight: bold; margin-right: 15px;">{info['name']}</div>
        <div style="background-color: {status_color}; color: white; padding: 5px 10px; border-radius: 15px; font-weight: bold;">{info['status']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"**Endpoint:** `{info['endpoint']}`")
    with col2:
        st.markdown(f"**Version:** {info['version']}")
    with col3:
        st.markdown(f"**Instance Type:** {info['instance_type']}")
    with col4:
        st.markdown(f"**Last Updated:** {info['last_updated']}")
    
    # Create tabs for different dashboard views
    tabs = st.tabs([
        "📈 Performance Metrics", 
        "🔍 Drift Analysis", 
        "⚙️ Resource Utilization", 
        "🚨 Alerts"
    ])
    
    # Performance Metrics tab
    with tabs[0]:
        # Create time series chart of model performance
        st.subheader("Model Performance Over Time")
        
        fig = go.Figure()
        
        # Add appropriate performance metric based on model type
        if 'accuracy' in ts_data.columns:
            fig.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#00A1C9', width=2)
            ))
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=ts_data['date'].min(),
                y0=0.85,
                x1=ts_data['date'].max(),
                y1=0.85,
                line=dict(
                    color="#D13212",
                    width=2,
                    dash="dash",
                )
            )
            
            fig.add_annotation(
                x=ts_data['date'].max(),
                y=0.85,
                text="Alert Threshold",
                showarrow=True,
                arrowhead=1,
                arrowcolor="#D13212",
                ax=-80,
                ay=20
            )
            
            y_title = "Accuracy"
            
        elif 'ndcg_score' in ts_data.columns:
            fig.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['ndcg_score'],
                mode='lines+markers',
                name='NDCG@10',
                line=dict(color='#00A1C9', width=2)
            ))
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=ts_data['date'].min(),
                y0=0.80,
                x1=ts_data['date'].max(),
                y1=0.80,
                line=dict(
                    color="#D13212",
                    width=2,
                    dash="dash",
                )
            )
            
            fig.add_annotation(
                x=ts_data['date'].max(),
                y=0.80,
                text="Alert Threshold",
                showarrow=True,
                arrowhead=1,
                arrowcolor="#D13212",
                ax=-80,
                ay=20
            )
            
            y_title = "NDCG@10"
            
        fig.update_layout(
            title=f"30-Day {y_title} Trend",
            xaxis_title="Date",
            yaxis_title=y_title,
            height=400,
            margin=dict(l=20, r=20, t=40, b=40),
            hovermode="x unified",
            yaxis=dict(range=[0.8, 1.0]),
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        st.plotly_chart(fig, use_column_width=True)
        
        # Invocation metrics
        st.subheader("Model Usage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Invocations chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['invocations'],
                mode='lines',
                name='Invocations',
                line=dict(color='#FF9900', width=2),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title="Daily Invocations",
                xaxis_title="Date",
                yaxis_title="Count",
                height=300,
                margin=dict(l=20, r=20, t=40, b=40),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_column_width=True)
            
        with col2:
            # Latency chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['latency'],
                mode='lines',
                name='Latency',
                line=dict(color='#59BA47', width=2)
            ))
            
            # Add p95 latency (simulated as 20% higher)
            fig.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['latency'] * 1.2,
                mode='lines',
                name='p95 Latency',
                line=dict(color='#FF9900', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Model Latency (ms)",
                xaxis_title="Date",
                yaxis_title="Milliseconds",
                height=300,
                margin=dict(l=20, r=20, t=40, b=40),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_column_width=True)
    
    # Drift Analysis tab
    with tabs[1]:
        st.subheader("Drift Detection")
        
        # Status indicator for drift
        drift_color = "#59BA47" if "No" in info['drift_status'] else ("#FF9900" if "Moderate" in info['drift_status'] else "#D13212")
        
        st.markdown(f"""
        <div style="background-color: {drift_color}; color: white; padding: 10px 15px; border-radius: 5px; margin-bottom: 20px;">
            <strong>Current Status:</strong> {info['drift_status']}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data drift chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['data_drift'],
                mode='lines+markers',
                name='Data Drift',
                line=dict(color='#00A1C9', width=2)
            ))
            
            # Add threshold lines
            fig.add_shape(
                type="line",
                x0=ts_data['date'].min(),
                y0=0.05,
                x1=ts_data['date'].max(),
                y1=0.05,
                line=dict(color="#FF9900", width=2, dash="dash")
            )
            
            fig.add_shape(
                type="line",
                x0=ts_data['date'].min(),
                y0=0.1,
                x1=ts_data['date'].max(),
                y1=0.1,
                line=dict(color="#D13212", width=2, dash="dash")
            )
            
            fig.update_layout(
                title="Data Drift Score (KL Divergence)",
                xaxis_title="Date",
                yaxis_title="Drift Score",
                height=350,
                margin=dict(l=20, r=20, t=40, b=40),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_column_width=True)
            
        with col2:
            # Model drift chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['model_drift'],
                mode='lines+markers',
                name='Model Drift',
                line=dict(color='#FF9900', width=2)
            ))
            
            # Add threshold lines
            fig.add_shape(
                type="line",
                x0=ts_data['date'].min(),
                y0=0.05,
                x1=ts_data['date'].max(),
                y1=0.05,
                line=dict(color="#FF9900", width=2, dash="dash")
            )
            
            fig.add_shape(
                type="line",
                x0=ts_data['date'].min(),
                y0=0.1,
                x1=ts_data['date'].max(),
                y1=0.1,
                line=dict(color="#D13212", width=2, dash="dash")
            )
            
            fig.update_layout(
                title="Model Drift Score (Performance Degradation)",
                xaxis_title="Date",
                yaxis_title="Drift Score",
                height=350,
                margin=dict(l=20, r=20, t=40, b=40),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_column_width=True)
        
        # Feature importance drift visualization (mock)
        st.subheader("Feature Importance Drift")
        
        # Create sample feature importance data
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        baseline_importance = [0.35, 0.25, 0.18, 0.12, 0.10]
        current_importance = [0.30, 0.22, 0.23, 0.15, 0.10]
        
        # Create a grouped bar chart
        feature_drift_data = pd.DataFrame({
            'Feature': feature_names * 2,
            'Importance': baseline_importance + current_importance,
            'Time': ['Baseline'] * 5 + ['Current'] * 5
        })
        
        fig = px.bar(
            feature_drift_data,
            x='Feature',
            y='Importance',
            color='Time',
            barmode='group',
            color_discrete_map={'Baseline': '#00A1C9', 'Current': '#FF9900'},
            labels={'Importance': 'Feature Importance', 'Feature': ''},
            title='Feature Importance Comparison'
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_column_width=True)
    
    # Resource Utilization tab
    with tabs[2]:
        st.subheader("Endpoint Resource Utilization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU utilization
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['cpu_utilization'],
                mode='lines',
                name='CPU Utilization',
                line=dict(color='#00A1C9', width=2),
                fill='tozeroy'
            ))
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=ts_data['date'].min(),
                y0=80,
                x1=ts_data['date'].max(),
                y1=80,
                line=dict(color="#D13212", width=2, dash="dash")
            )
            
            fig.update_layout(
                title="CPU Utilization (%)",
                xaxis_title="Date",
                yaxis_title="Percentage",
                height=300,
                margin=dict(l=20, r=20, t=40, b=40),
                hovermode="x unified",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_column_width=True)
            
        with col2:
            # Memory utilization
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['memory_utilization'],
                mode='lines',
                name='Memory Utilization',
                line=dict(color='#FF9900', width=2),
                fill='tozeroy'
            ))
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=ts_data['date'].min(),
                y0=85,
                x1=ts_data['date'].max(),
                y1=85,
                line=dict(color="#D13212", width=2, dash="dash")
            )
            
            fig.update_layout(
                title="Memory Utilization (%)",
                xaxis_title="Date",
                yaxis_title="Percentage",
                height=300,
                margin=dict(l=20, r=20, t=40, b=40),
                hovermode="x unified",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_column_width=True)
        
        # Cost analysis (mock)
        st.subheader("Cost Analysis")
        
        # Create sample cost data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Base cost pattern with some daily variation
        base_cost = [
            random.uniform(20, 25) if i < 23 else random.uniform(35, 42)  # Cost increased in last week
            for i in range(30)
        ]
        
        cost_df = pd.DataFrame({
            'date': dates,
            'compute_cost': [c * 0.7 for c in base_cost],  # 70% compute
            'storage_cost': [c * 0.2 for c in base_cost],  # 20% storage
            'data_transfer': [c * 0.1 for c in base_cost]   # 10% data transfer
        })
        
        # Create stacked area chart
        fig = px.area(
            cost_df, 
            x='date', 
            y=['compute_cost', 'storage_cost', 'data_transfer'],
            labels={'value': 'Cost (USD)', 'date': 'Date', 'variable': 'Cost Component'},
            color_discrete_map={
                'compute_cost': '#00A1C9',
                'storage_cost': '#FF9900',
                'data_transfer': '#59BA47'
            },
            title='Daily Endpoint Cost Breakdown'
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=40),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title_text=''
            )
        )
        
        # Rename legend items
        new_names = {'compute_cost': 'Compute', 'storage_cost': 'Storage', 'data_transfer': 'Data Transfer'}
        for i, trace in enumerate(fig.data):
            trace.name = new_names[trace.name]
            
        st.plotly_chart(fig, use_column_width=True)
            
    # Alerts tab
    with tabs[3]:
        st.subheader("Model Alerts and Notifications")
        
        # Display alerts if any
        if info['alerts']:
            for alert in info['alerts']:
                # Set color based on severity
                if alert['severity'] == 'Low':
                    color = '#59BA47'
                elif alert['severity'] == 'Medium':
                    color = '#FF9900'
                else:
                    color = '#D13212'
                    
                st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding: 10px 15px; margin-bottom: 15px; background-color: #F8F9FA; border-radius: 0 5px 5px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <div><strong>{alert['type']} Alert</strong></div>
                        <div>Date: {alert['date']}</div>
                    </div>
                    <p style="margin: 0;">{alert['message']}</p>
                    <div style="text-align: right; margin-top: 8px;">
                        <span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px;">{alert['severity']} Severity</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.markdown("""
            <div style="border: 1px solid #E9ECEF; padding: 20px; text-align: center; border-radius: 5px; background-color: #F8F9FA;">
                <div style="font-size: 36px; margin-bottom: 10px;">✓</div>
                <div style="font-weight: bold; margin-bottom: 5px;">No Alerts</div>
                <div style="color: #545B64;">This model has no active alerts or notifications</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Alert settings (mock)
        st.subheader("Alert Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Metric Thresholds")
            
            # Show some example threshold settings
            thresholds = [
                {"metric": "Data Quality Score", "warning": "< 0.95", "critical": "< 0.9"},
                {"metric": "Accuracy/NDCG", "warning": "< 0.9", "critical": "< 0.85"},
                {"metric": "Drift Score", "warning": "> 0.05", "critical": "> 0.1"},
                {"metric": "CPU Utilization", "warning": "> 70%", "critical": "> 85%"}
            ]
            
            # Create a table
            threshold_df = pd.DataFrame(thresholds)
            st.table(threshold_df)
            
        with col2:
            st.markdown("#### Notification Channels")
            
            # Mock notification settings
            st.markdown("""
            <div style="margin-bottom: 10px; padding: 8px; border: 1px solid #E9ECEF; border-radius: 5px; display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: #59BA47; border-radius: 50%; margin-right: 10px;"></div>
                <div>
                    <div style="font-weight: bold;">Email Notifications</div>
                    <div style="font-size: 12px; color: #545B64;">ml-alerts@example.com</div>
                </div>
            </div>
            <div style="margin-bottom: 10px; padding: 8px; border: 1px solid #E9ECEF; border-radius: 5px; display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: #59BA47; border-radius: 50%; margin-right: 10px;"></div>
                <div>
                    <div style="font-weight: bold;">Slack Channel</div>
                    <div style="font-size: 12px; color: #545B64;">#ml-model-alerts</div>
                </div>
            </div>
            <div style="margin-bottom: 10px; padding: 8px; border: 1px solid #E9ECEF; border-radius: 5px; display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: #FF9900; border-radius: 50%; margin-right: 10px;"></div>
                <div>
                    <div style="font-weight: bold;">AWS SNS Topic</div>
                    <div style="font-size: 12px; color: #545B64;">arn:aws:sns:us-west-2:123456789012:ModelAlerts</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def create_role_manager_visualization(role_data):
    """
    Create a visualization of the SageMaker Role Manager
    """
    # Display selected role information
    role = role_data
    
    # Role header
    st.markdown(f"""
    <div style="background-color: #232F3E; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <div style="font-size: 24px; font-weight: bold;">{st.session_state.selected_role} Role</div>
        <div style="margin-top: 5px;">{role["description"]}</div>
        <div style="margin-top: 10px; font-style: italic;">Persona: {role["persona"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different views
    tabs = st.tabs([
        "🔑 Permissions", 
        "📂 Resource Access",
        "🔄 Policies"
    ])
    
    # Permissions tab
    with tabs[0]:
        st.subheader("Role Permissions")
        
        # Group permissions by access level
        full_access = [p for p in role["permissions"] if p["access"] == "Full"]
        limited_access = [p for p in role["permissions"] if p["access"] == "Limited"]
        read_only = [p for p in role["permissions"] if p["access"] == "Read-only"]
        no_access = [p for p in role["permissions"] if p["access"] == "None"]
        
        # Create columns for different access levels
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Full Access")
            for permission in full_access:
                st.markdown(f"""
                <div style="border-left: 4px solid #59BA47; padding: 10px; margin-bottom: 10px; background-color: #F8F9FA;">
                    <div style="font-weight: bold;">{permission["name"]}</div>
                    <div style="font-size: 13px; color: #545B64;">{permission["description"]}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("#### No Access")
            if no_access:
                for permission in no_access:
                    st.markdown(f"""
                    <div style="border-left: 4px solid #D13212; padding: 10px; margin-bottom: 10px; background-color: #F8F9FA;">
                        <div style="font-weight: bold;">{permission["name"]}</div>
                        <div style="font-size: 13px; color: #545B64;">{permission["description"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("*No explicit denials configured*")
            
        with col2:
            st.markdown("#### Limited Access")
            for permission in limited_access:
                st.markdown(f"""
                <div style="border-left: 4px solid #FF9900; padding: 10px; margin-bottom: 10px; background-color: #F8F9FA;">
                    <div style="font-weight: bold;">{permission["name"]}</div>
                    <div style="font-size: 13px; color: #545B64;">{permission["description"]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("#### Read-Only Access")
            for permission in read_only:
                st.markdown(f"""
                <div style="border-left: 4px solid #00A1C9; padding: 10px; margin-bottom: 10px; background-color: #F8F9FA;">
                    <div style="font-weight: bold;">{permission["name"]}</div>
                    <div style="font-size: 13px; color: #545B64;">{permission["description"]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Visualize permissions as a radar chart
        st.subheader("Permission Overview")
        
        # Convert access levels to numeric values
        access_values = {
            "Full": 1.0,
            "Limited": 0.6,
            "Read-only": 0.3,
            "None": 0.0
        }
        
        # Group permissions by category for radar chart
        categories = []
        values = []
        
        for permission in role["permissions"]:
            categories.append(permission["name"])
            values.append(access_values[permission["access"]])
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Access Level',
            line_color='#00A1C9'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=500,
            margin=dict(l=80, r=80, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_column_width=True)
    
    # Resource Access tab
    with tabs[1]:
        st.subheader("Resource Access Scope")
        
        # Display resource ARN patterns
        st.markdown("#### Resource ARN Patterns")
        for resource_arn in role["resource_scope"]:
            st.code(resource_arn)
        
        # Display data access information
        st.subheader("Data Access")
        
        # S3 buckets
        st.markdown("#### S3 Buckets")
        for bucket in role["data_access"]["s3_buckets"]:
            st.markdown(f"""
            <div style="border: 1px solid #E9ECEF; padding: 8px 12px; margin-bottom: 8px; border-radius: 5px; background-color: #F8F9FA;">
                <span style="font-family: monospace; color: #545B64;">s3://</span><span style="font-family: monospace;">{bucket}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Access conditions
        st.markdown("#### Access Conditions")
        
        for tag_key, tag_values in role["data_access"]["conditions"].items():
            st.markdown(f"**{tag_key}:**")
            
            tag_html = ""
            for value in tag_values:
                tag_html += f'<span style="background-color: #E9ECEF; padding: 4px 8px; border-radius: 15px; margin-right: 5px; margin-bottom: 5px; display: inline-block;">{value}</span>'
            
            st.markdown(tag_html, unsafe_allow_html=True)
        
        # Visualize data access permissions
        st.subheader("Data Access Visualization")
        
        # Create a visual representation of data access patterns
        G = nx.Graph()
        
        # Add role node
        G.add_node(st.session_state.selected_role, type="role", color="#232F3E")
        
        # Add bucket nodes
        for bucket in role["data_access"]["s3_buckets"]:
            G.add_node(bucket, type="bucket", color="#FF9900")
            G.add_edge(st.session_state.selected_role, bucket)
        
        # Add condition nodes
        for tag_key, tag_values in role["data_access"]["conditions"].items():
            condition_name = f"{tag_key}"
            G.add_node(condition_name, type="condition", color="#00A1C9")
            G.add_edge(st.session_state.selected_role, condition_name)
            
            # Add tag values
            for value in tag_values:
                value_name = f"{value}"
                G.add_node(value_name, type="value", color="#59BA47")
                G.add_edge(condition_name, value_name)
        
        # Create position layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Adjust positions for better layout
        # Move the role to the center-top
        role_pos = pos[st.session_state.selected_role]
        pos[st.session_state.selected_role] = [0, 0.8]
        
        # Draw the graph
        plt.figure(figsize=(10, 8))
        
        # Node colors and shapes
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        node_shapes = {
            "role": "s",  # Square for role
            "bucket": "o",  # Circle for buckets
            "condition": "^",  # Triangle for conditions
            "value": "d"  # Diamond for values
        }
        
        # Draw nodes by type
        for node_type, shape in node_shapes.items():
            nodes = [n for n in G.nodes() if G.nodes[n].get('type') == node_type]
            if not nodes:
                continue
                
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodes,
                node_color=[G.nodes[n]['color'] for n in nodes],
                node_shape=shape,
                node_size=800 if node_type == "role" else 500,
                alpha=0.8
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            width=1.5,
            alpha=0.7,
            edge_color="#aaaaaa"
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=10,
            font_weight="bold",
            font_color="white"
        )
        
        # Add a legend
        legend_labels = {
            "role": "Role", 
            "bucket": "S3 Bucket",
            "condition": "Condition Tag",
            "value": "Tag Value"
        }
        
        # Create legend patches
        patches = []
        for node_type, color in [
            ("role", "#232F3E"),
            ("bucket", "#FF9900"),
            ("condition", "#00A1C9"),
            ("value", "#59BA47")
        ]:
            shape = node_shapes[node_type]
            patches.append(plt.Line2D([0], [0], marker=shape, color='w', 
                                     markerfacecolor=color, markersize=10, 
                                     label=legend_labels[node_type]))
            
        plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        
        # Remove axis
        plt.axis('off')
        plt.tight_layout()
        
        # Display the graph
        st.pyplot(plt)
    
    # Policies tab
    with tabs[2]:
        st.subheader("AWS Managed Policies")
        
        for policy in role["aws_managed_policies"]:
            st.markdown(f"""
            <div style="border: 1px solid #E9ECEF; padding: 12px; margin-bottom: 10px; border-radius: 5px; background-color: #F8F9FA;">
                <div style="font-weight: bold;">{policy}</div>
                <div style="font-size: 12px; color: #545B64; margin-top: 5px;">AWS Managed Policy</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Custom Policy Document")
        
        # Create a sample policy document based on the permissions
        policy_document = {
            "Version": "2012-10-17",
            "Statement": []
        }
        
        # Add permissions based on access level
        for permission in role["permissions"]:
            if permission["access"] in ["Full", "Limited", "Read-only"] and permission["actions"]:
                statement = {
                    "Sid": f"{permission['name'].replace(' ', '')}",
                    "Effect": "Allow",
                    "Action": permission["actions"],
                    "Resource": "*"
                }
                policy_document["Statement"].append(statement)
        
        # Add resource conditions
        if role["resource_scope"]:
            policy_document["Statement"].append({
                "Sid": "ResourceScope",
                "Effect": "Allow",
                "Action": ["sagemaker:*"],
                "Resource": role["resource_scope"]
            })
        
        # Format and display policy JSON
        policy_json = json.dumps(policy_document, indent=4)
        st.code(policy_json, language="json")
        
        # Policy creation button (mock)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.button("Create IAM Role", type="primary")
        
        with col2:
            st.download_button(
                label="Download Policy JSON",
                data=policy_json,
                file_name=f"{st.session_state.selected_role}_policy.json",
                mime="application/json"
            )


def get_fairness_explanation(metric_name, status):
    """
    Get explanation text for fairness metric status
    """
    if metric_name == "disparate_impact":
        if status == "Good":
            return "Model predictions are similarly distributed across demographic groups"
        elif status == "Caution":
            return "Model shows some imbalance in predictions across groups"
        else:
            return "Model shows significant bias in predictions across groups"
    else:  # statistical_parity_difference or similar
        if status == "Good":
            return "Minimal difference in predictions between demographic groups"
        elif status == "Caution":
            return "Moderate difference in predictions between demographic groups"
        else:
            return "Large difference in predictions between demographic groups"


# Main application
def main():

    
    # Initialize session state
    initialize_session_state()
    
    # AWS Color Scheme
    AWS_COLORS = {
        "orange": "#FF9900",
        "teal": "#00A1C9", 
        "blue": "#232F3E",
        "gray": "#E9ECEF",
        "light_gray": "#F8F9FA",
        "white": "#FFFFFF",
        "dark_gray": "#545B64",
        "green": "#59BA47",
        "red": "#D13212"
    }
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main {
            background-color: #F8F9FA;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #F8F9FA;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            white-space: pre-wrap;
            border-radius: 6px;
            font-weight: 600;
            background-color: #FFFFFF;
            color: #232F3E;
            border: 1px solid #E9ECEF;
            padding: 5px 15px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF9900 !important;
            color: #FFFFFF !important;
            border: 1px solid #FF9900 !important;
        }
        .stButton button {
            background-color: #FF9900;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 8px 16px;
        }
        .stButton button:hover {
            background-color: #EC7211;
        }
        .info-box {
            background-color: #E6F2FF;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #00A1C9;
        }
        .warning-box {
            background-color: #FFF8E6;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #FF9900;
        }
        .code-box {
            background-color: #232F3E;
            color: #FFFFFF;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
            margin: 15px 0;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 0;
            right: 0;
            left: 0;
            background-color: #232F3E;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 12px;
        }
        h1, h2, h3 {
            color: #232F3E;
        }
        .status-card {
            border: 1px solid #E9ECEF;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            background-color: #FFFFFF;
            transition: transform 0.2s;
        }
        .status-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .status-1 {
            border-left: 5px solid #59BA47;
        }
        .status-2 {
            border-left: 5px solid #D13212;
        }
        .status-3 {
            border-left: 5px solid #FF9900;
        }
        .metrics-table th {
            font-weight: normal;
            color: #545B64;
        }
        .metrics-table td {
            font-weight: bold;
        }
        .role-selection {
            padding: 12px;
            border-radius: 5px;
            border: 1px solid #E9ECEF;
            background-color: white;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .role-selection:hover {
            border-color: #00A1C9;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .role-selected {
            border-color: #FF9900;
            border-width: 2px;
            box-shadow: 0 2px 8px rgba(255, 153, 0, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar content
    with st.sidebar:
        st.title("Session Management")
        st.info(f"User ID: {st.session_state.user_id}")
        
        if st.button("Reset Session"):
            reset_session()
            st.rerun()
        
        st.divider()
        # Information about the application
        with st.expander("About this application", expanded=False):
            st.markdown("""
                This e-learning application demonstrates ML Governance with Amazon SageMaker.
                Learn about SageMaker Role Manager, Model Cards, and the Model Dashboard.
            """)
            
            # Load lottie animation
            lottie_url = "https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json"
            lottie_json = load_lottie_url(lottie_url)
            if lottie_json:
                st_lottie(lottie_json, height=200, key="sidebar_animation")
            
            # Additional resources section
            st.subheader("Additional Resources")
            st.markdown("""
                - [SageMaker Role Manager Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/role-manager.html)
                - [SageMaker Model Cards Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-cards.html)
                - [SageMaker Model Dashboard Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-dashboard.html)
                - [ML Governance Best Practices](https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/ml-governance/)
            """)
    
    # Main app header
    st.title("ML Governance with Amazon SageMaker")
    st.markdown("Learn how SageMaker helps organizations implement ML governance through role-based access control, model documentation, and continuous monitoring.")
    
    # Tab-based navigation with emoji
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Overview", 
        "🔑 SageMaker Role Manager",
        "📋 SageMaker Model Cards",
        "📊 SageMaker Model Dashboard"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("ML Governance Overview")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ML governance refers to the framework of processes, policies, and tools that ensure machine learning systems are developed and deployed responsibly, securely, and in compliance with regulations.
            
            Amazon SageMaker provides comprehensive governance capabilities to help organizations:
            
            - **Control access** to ML resources through fine-grained permissions
            - **Document models** with standardized model cards
            - **Monitor model performance** across the organization
            - **Implement accountability** through audit trails and lineage tracking
            - **Ensure compliance** with industry regulations and internal standards
            """)
            
            st.markdown("""
            <div class="info-box">
            <h3>Why ML Governance Matters</h3>
            <p>As machine learning becomes more widespread in business-critical applications, governance becomes essential to:</p>
            <ul>
                <li><strong>Mitigate risks</strong> associated with model failures and biased outputs</li>
                <li><strong>Meet regulatory requirements</strong> in industries like finance, healthcare, and insurance</li>
                <li><strong>Build trust</strong> with customers, employees, and stakeholders</li>
                <li><strong>Scale ML responsibly</strong> across the organization</li>
                <li><strong>Track AI system lineage</strong> for auditing and reproducibility</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Governance diagram
            st.image("https://d1.awsstatic.com/aws-mls-platform/SageMaker%20assets/mlops-infinity-loop.485d11146c76f95123a2def677139b8d23e5c247.png", 
                    caption="ML Governance within the ML lifecycle", use_column_width=True)
            
            # Add a key metrics card
            st.markdown("""
            <div class="card" style="margin-top: 20px;">
                <h4>ML Governance Impact</h4>
                <div style="display: flex; justify-content: space-between; text-align: center; margin-top: 15px;">
                    <div>
                        <div style="font-size: 24px; font-weight: bold; color: #00A1C9;">92%</div>
                        <div style="font-size: 14px;">Faster Compliance</div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: bold; color: #FF9900;">78%</div>
                        <div style="font-size: 14px;">Risk Reduction</div>
                    </div>
                    <div>
                        <div style="font-size: 24px; font-weight: bold; color: #59BA47;">3.5x</div>
                        <div style="font-size: 14px;">ML Team Productivity</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Key SageMaker Governance Tools")
        
        # Three columns for three tools
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>🔑 SageMaker Role Manager</h3>
                <p>Simplifies creation and management of IAM roles with appropriate permissions for ML workflows.</p>
                <ul>
                    <li>Create purpose-built ML roles</li>
                    <li>Enforce least-privilege access</li>
                    <li>Implement data access controls</li>
                    <li>Support diverse ML personas</li>
                </ul>
                <div style="text-align: center; margin-top: 15px;">
                    <button style="background-color: #FF9900; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;" 
                            onclick="document.querySelector('[data-baseweb=\\'tab\\']', '[id=\\'tabs-1-tab-1\\']').click()">
                        Explore Role Manager
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
                <h3>📋 SageMaker Model Cards</h3>
                <p>Standardizes documentation of essential information about machine learning models.</p>
                <ul>
                    <li>Document model details</li>
                    <li>Track performance metrics</li>
                    <li>Record intended uses & limitations</li>
                    <li>Assess model risks & mitigations</li>
                </ul>
                <div style="text-align: center; margin-top: 15px;">
                    <button style="background-color: #FF9900; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;"
                            onclick="document.querySelector('[data-baseweb=\\'tab\\']', '[id=\\'tabs-1-tab-2\\']').click()">
                        Explore Model Cards
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="card">
                <h3>📊 SageMaker Model Dashboard</h3>
                <p>Provides a centralized interface for monitoring and managing models across your organization.</p>
                <ul>
                    <li>Monitor model performance</li>
                    <li>Track data & concept drift</li>
                    <li>Receive alerts for anomalies</li>
                    <li>Manage model inventory</li>
                </ul>
                <div style="text-align: center; margin-top: 15px;">
                    <button style="background-color: #FF9900; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;"
                            onclick="document.querySelector('[data-baseweb=\\'tab\\']', '[id=\\'tabs-1-tab-3\\']').click()">
                        Explore Model Dashboard
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Implementing ML Governance with SageMaker")
        
        # Governance lifecycle visualization
        lifecycle = [
            {
                "stage": "Define Governance Framework",
                "description": "Establish ML policies, standards, and controls",
                "tools": ["SageMaker Role Manager", "SageMaker Projects"],
                "status": "status-1"
            },
            {
                "stage": "Implement Access Controls",
                "description": "Create custom roles for different ML personas",
                "tools": ["SageMaker Role Manager", "IAM"],
                "status": "status-1"
            },
            {
                "stage": "Document Models",
                "description": "Create comprehensive model cards for all models",
                "tools": ["SageMaker Model Cards", "SageMaker Model Registry"],
                "status": "status-3"
            },
            {
                "stage": "Monitor Model Performance",
                "description": "Track metrics, detect drift, and alert on anomalies",
                "tools": ["SageMaker Model Dashboard", "SageMaker Model Monitor"],
                "status": "status-3"
            },
            {
                "stage": "Audit & Report",
                "description": "Generate compliance reports and conduct audits",
                "tools": ["SageMaker Model Dashboard", "CloudTrail", "CloudWatch"],
                "status": "status-2"
            }
        ]
        
        # Create timeline visualization
        col1, col2, col3, col4, col5 = st.columns(5)
        columns = [col1, col2, col3, col4, col5]
        
        for i, (col, stage) in enumerate(zip(columns, lifecycle)):
            with col:
                status_color = "#59BA47" if stage["status"] == "status-1" else ("#FF9900" if stage["status"] == "status-3" else "#D13212")
                
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="background-color: {status_color}; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-weight: bold;">{i+1}</div>
                    <div style="border-top: 2px solid {status_color}; margin: 10px 0;"></div>
                    <div style="font-weight: bold;">{stage["stage"]}</div>
                    <div style="font-size: 12px; color: #545B64; margin-top: 5px; height: 60px;">{stage["description"]}</div>
                    <div style="margin-top: 10px;">
                """, unsafe_allow_html=True)
                
                for tool in stage["tools"]:
                    tool_color = "#00A1C9" if "Role" in tool else ("#FF9900" if "Model" in tool else "#59BA47")
                    st.markdown(f"""
                        <span style="background-color: {tool_color}; color: white; font-size: 10px; padding: 2px 8px; border-radius: 10px; display: inline-block; margin: 2px;">{tool}</span>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        # Case study
        st.markdown("""
        <div class="warning-box">
        <h3>Case Study: Financial Services ML Governance</h3>
        <p>A major financial institution implemented SageMaker governance tools to manage their ML models:</p>
        <ul>
            <li><strong>Challenge:</strong> Meeting regulatory requirements while scaling ML across the organization</li>
            <li><strong>Solution:</strong> Implemented SageMaker Role Manager to create persona-based access controls, documented all models with Model Cards, and used Model Dashboard for continuous monitoring</li>
            <li><strong>Results:</strong> 70% faster compliance reporting, 92% reduction in security incidents, and successful regulatory audits</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 2: SAGEMAKER ROLE MANAGER
    with tab2:
        st.header("Amazon SageMaker Role Manager")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Amazon SageMaker Role Manager simplifies the process of creating and managing IAM roles for ML workflows. 
            It helps organizations enforce least privilege access by creating purpose-built roles for different ML personas.
            
            **Key capabilities:**
            - Create pre-configured IAM roles for specific ML workflows
            - Define least-privilege permissions for data scientists and ML engineers
            - Manage role-based access to SageMaker resources and data
            - Enable self-service creation of properly scoped ML roles
            - Maintain security best practices by enforcing permission boundaries
            """)
            
            st.markdown("""
            <div class="info-box">
            <h3>Common ML Personas</h3>
            <p>SageMaker Role Manager supports the following common ML personas:</p>
            <ul>
                <li><strong>Data Scientists</strong>: Build, train, and deploy machine learning models</li>
                <li><strong>MLOps Engineers</strong>: Build and manage ML infrastructure and pipelines</li>
                <li><strong>ML Governance Officers</strong>: Ensure compliance and responsible AI practices</li>
                <li><strong>Business Analysts</strong>: Use ML insights for business decision-making</li>
            </ul>
            <p>Each persona requires different permissions to SageMaker resources.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Role Manager diagram
            st.image("https://d1.awsstatic.com/ml-governance/persona-mgmt.daf5b956b050647c2c6b7ef6b5fc7e3c35c61651.png", 
                    use_column_width=True)
            st.caption("SageMaker Role Manager supports different ML personas")
        
        st.subheader("Select ML Persona")
        
        # Create role selection cards
        roles_data = st.session_state.roles_data
        
        # Display the roles in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        # Define role-specific colors
        role_colors = {
            "DataScientist": "#00A1C9",
            "MLOpsEngineer": "#FF9900",
            "ModelGovernanceOfficer": "#59BA47",
            "BusinessAnalyst": "#545B64"
        }
        
        # Display role cards
        with col1:
            selected = st.session_state.selected_role == "DataScientist"
            st.markdown(f"""
            <div class="role-selection {'role-selected' if selected else ''}" onclick="this.classList.toggle('role-selected');" 
                 data-role="DataScientist" style="border-top: 4px solid {role_colors['DataScientist']};">
                <h4 style="margin-top: 0;">Data Scientist</h4>
                <p style="font-size: 12px; color: #545B64;">{roles_data['DataScientist']['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Data Scientist", key="ds_button"):
                st.session_state.selected_role = "DataScientist"
                st.rerun()
        
        with col2:
            selected = st.session_state.selected_role == "MLOpsEngineer"
            st.markdown(f"""
            <div class="role-selection {'role-selected' if selected else ''}" onclick="this.classList.toggle('role-selected');"
                 data-role="MLOpsEngineer" style="border-top: 4px solid {role_colors['MLOpsEngineer']};">
                <h4 style="margin-top: 0;">MLOps Engineer</h4>
                <p style="font-size: 12px; color: #545B64;">{roles_data['MLOpsEngineer']['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select MLOps Engineer", key="mlops_button"):
                st.session_state.selected_role = "MLOpsEngineer"
                st.rerun()
        
        with col3:
            selected = st.session_state.selected_role == "ModelGovernanceOfficer"
            st.markdown(f"""
            <div class="role-selection {'role-selected' if selected else ''}" onclick="this.classList.toggle('role-selected');"
                 data-role="ModelGovernanceOfficer" style="border-top: 4px solid {role_colors['ModelGovernanceOfficer']};">
                <h4 style="margin-top: 0;">Governance Officer</h4>
                <p style="font-size: 12px; color: #545B64;">{roles_data['ModelGovernanceOfficer']['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Governance Officer", key="gov_button"):
                st.session_state.selected_role = "ModelGovernanceOfficer"
                st.rerun()
        
        with col4:
            selected = st.session_state.selected_role == "BusinessAnalyst"
            st.markdown(f"""
            <div class="role-selection {'role-selected' if selected else ''}" onclick="this.classList.toggle('role-selected');"
                 data-role="BusinessAnalyst" style="border-top: 4px solid {role_colors['BusinessAnalyst']};">
                <h4 style="margin-top: 0;">Business Analyst</h4>
                <p style="font-size: 12px; color: #545B64;">{roles_data['BusinessAnalyst']['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select Business Analyst", key="ba_button"):
                st.session_state.selected_role = "BusinessAnalyst"
                st.rerun()
        
        st.divider()
        
        # Create role manager visualization
        create_role_manager_visualization(roles_data[st.session_state.selected_role])
        
        # Example code for creating a role
        st.subheader("Creating a Role with SageMaker Role Manager")
        
        st.markdown("""
        Role Manager simplifies creating IAM roles with the right permissions for ML workflows.
        Below is an example of how to create a role using the AWS CLI:
        """)
        
        st.code(f"""
# Create a role for {st.session_state.selected_role} persona
aws sagemaker create-role-from-persona \\
    --persona {st.session_state.selected_role} \\
    --role-name "{st.session_state.selected_role}Role" \\
    --tags Key=Department,Value=ML Key=Environment,Value=Development \\
    --s3-resource-configs BucketName=company-ml-training,Access=Full \\
    --resource-config ResourceTypes=notebook,EndpointConfig,TrainingJob \\
    --custom-permissions-boundary arn:aws:iam::123456789012:policy/MLPermissionsBoundary
        """)
        
        st.markdown("""
        <div class="warning-box">
        <h3>Best Practices for Role-Based Access Control in ML</h3>
        <ul>
            <li><strong>Least privilege:</strong> Grant only the permissions required for each persona</li>
            <li><strong>Separation of duties:</strong> Ensure no single role can create, approve, and deploy models</li>
            <li><strong>Permission boundaries:</strong> Set maximum permissions for self-service role creation</li>
            <li><strong>Resource tagging:</strong> Control access based on project, environment, or sensitivity tags</li>
            <li><strong>Regular audits:</strong> Review and update roles as ML workflows evolve</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 3: SAGEMAKER MODEL CARDS
    with tab3:
        st.header("Amazon SageMaker Model Cards")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Amazon SageMaker Model Cards provide a standardized way to document essential information about your machine learning models.
            Model cards create a central source of truth for model information, helping teams track model details, performance, and lineage.
            
            **Key benefits:**
            - Document model details, intended uses, and limitations
            - Track performance metrics across different model versions
            - Record risk ratings and mitigation strategies
            - Support governance, compliance, and audit requirements
            - Facilitate collaboration and knowledge sharing
            """)
            
            st.markdown("""
            <div class="info-box">
            <h3>What's in a Model Card?</h3>
            <p>SageMaker Model Cards typically include:</p>
            <ul>
                <li><strong>Basic information:</strong> Model name, version, owner, and approval status</li>
                <li><strong>Model details:</strong> Framework, architecture, and training methodology</li>
                <li><strong>Intended uses:</strong> Approved use cases and applications</li>
                <li><strong>Performance metrics:</strong> Accuracy, precision, recall, and other metrics</li>
                <li><strong>Limitations:</strong> Known constraints and scenarios where the model may underperform</li>
                <li><strong>Ethical considerations:</strong> Potential biases and fairness concerns</li>
                <li><strong>Risk rating:</strong> Overall risk assessment and mitigation strategies</li>
                <li><strong>Lineage:</strong> Training datasets, development history, and dependencies</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Model Card diagram
            st.image("https://d1.awsstatic.com/ml-governance/governance_model-cards_how-it-works-diagram.3df069ff7225d3a0ffadf524a87c84dcb88c097d.png", 
                    caption="SageMaker Model Cards provide standardized documentation", use_column_width=True)
        
        st.subheader("Select a Model Card")
        
        # Create model selection cards
        model_cards = st.session_state.model_cards
        
        # Display the models in a grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected = st.session_state.selected_model == "fraud_detection_model"
            model = model_cards["fraud_detection_model"]
            risk_color = "#59BA47" if model["risk_rating"] == "Low" else ("#FF9900" if model["risk_rating"] == "Medium" else "#D13212")
            
            st.markdown(f"""
            <div class="role-selection {'role-selected' if selected else ''}" style="border-top: 4px solid {risk_color};">
                <h4 style="margin-top: 0;">{model["name"]}</h4>
                <p style="font-size: 12px; color: #545B64;">{model["description"]}</p>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #545B64; margin-top: 10px;">
                    <div>Version: {model["version"]}</div>
                    <div>
                        <span style="background-color: {risk_color}; color: white; padding: 2px 8px; border-radius: 10px;">{model["risk_rating"]} Risk</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Fraud Detection Model", key="fraud_button"):
                st.session_state.selected_model = "fraud_detection_model"
                st.rerun()
        
        with col2:
            selected = st.session_state.selected_model == "customer_churn_model"
            model = model_cards["customer_churn_model"]
            risk_color = "#59BA47" if model["risk_rating"] == "Low" else ("#FF9900" if model["risk_rating"] == "Medium" else "#D13212")
            
            st.markdown(f"""
            <div class="role-selection {'role-selected' if selected else ''}" style="border-top: 4px solid {risk_color};">
                <h4 style="margin-top: 0;">{model["name"]}</h4>
                <p style="font-size: 12px; color: #545B64;">{model["description"]}</p>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #545B64; margin-top: 10px;">
                    <div>Version: {model["version"]}</div>
                    <div>
                        <span style="background-color: {risk_color}; color: white; padding: 2px 8px; border-radius: 10px;">{model["risk_rating"]} Risk</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Customer Churn Model", key="churn_button"):
                st.session_state.selected_model = "customer_churn_model"
                st.rerun()
        
        with col3:
            selected = st.session_state.selected_model == "product_recommendation_model"
            model = model_cards["product_recommendation_model"]
            risk_color = "#59BA47" if model["risk_rating"] == "Low" else ("#FF9900" if model["risk_rating"] == "Medium" else "#D13212")
            
            st.markdown(f"""
            <div class="role-selection {'role-selected' if selected else ''}" style="border-top: 4px solid {risk_color};">
                <h4 style="margin-top: 0;">{model["name"]}</h4>
                <p style="font-size: 12px; color: #545B64;">{model["description"]}</p>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #545B64; margin-top: 10px;">
                    <div>Version: {model["version"]}</div>
                    <div>
                        <span style="background-color: {risk_color}; color: white; padding: 2px 8px; border-radius: 10px;">{model["risk_rating"]} Risk</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Recommendation Model", key="rec_button"):
                st.session_state.selected_model = "product_recommendation_model"
                st.rerun()
        
        st.divider()
        
        # Display the selected model card
        st.subheader(f"Model Card: {model_cards[st.session_state.selected_model]['name']}")
        
        # Create model card visualization
        create_model_card_visualization(model_cards[st.session_state.selected_model])
        
        # Example code for creating a model card
        st.subheader("Creating a Model Card with SageMaker")
        
        model_name = model_cards[st.session_state.selected_model]["name"]
        
        st.markdown(f"""
        Here's how you can create a model card programmatically for the {model_name}:
        """)
        
        st.code(f"""
# Create a Model Card using the SageMaker Python SDK
import boto3
import json
from datetime import datetime

# Initialize the SageMaker client
sagemaker_client = boto3.client('sagemaker')

# Create model card content
model_card_content = {{
    "model_overview": {{
        "model_name": "{model_name}",
        "model_version": "{model_cards[st.session_state.selected_model]['version']}",
        "model_description": "{model_cards[st.session_state.selected_model]['description']}"
    }},
    "intended_uses": {{
        "intended_uses": {json.dumps(model_cards[st.session_state.selected_model]['intended_uses'])},
        "limitations": {json.dumps(model_cards[st.session_state.selected_model]['limitations'])}
    }},
    "ethical_considerations": {{
        "ethical_considerations": {json.dumps(model_cards[st.session_state.selected_model]['ethical_considerations'])}
    }},
    "risk_rating": "{model_cards[st.session_state.selected_model]['risk_rating']}",
    "training_details": {{
        "data_source": "{model_cards[st.session_state.selected_model]['training_data']['source']}",
        "training_period": "{model_cards[st.session_state.selected_model]['training_data']['timeframe']}"
    }},
    "evaluation_details": {{
        "metrics": {json.dumps(model_cards[st.session_state.selected_model]['metrics'])}
    }}
}}

# Create the model card
response = sagemaker_client.create_model_card(
    ModelCardName="{model_name.replace(' ', '_').lower()}",
    ModelCardStatus="Approved",
    Content=json.dumps(model_card_content),
    Tags=[
        {{
            'Key': 'Department',
            'Value': 'RiskManagement'
        }},
        {{
            'Key': 'Environment',
            'Value': 'Production'
        }}
    ]
)

print(f"Created model card: {{response['ModelCardArn']}}")
        """)
        
        st.markdown("""
        <div class="warning-box">
        <h3>Best Practices for Model Documentation</h3>
        <ul>
            <li><strong>Create documentation early:</strong> Start documenting as soon as model development begins</li>
            <li><strong>Use standardized templates:</strong> Maintain consistent documentation across all models</li>
            <li><strong>Document limitations clearly:</strong> Be transparent about scenarios where the model may not perform well</li>
            <li><strong>Track model lineage:</strong> Document all data sources and model dependencies</li>
            <li><strong>Include ethical assessments:</strong> Evaluate and document potential biases and fairness concerns</li>
            <li><strong>Keep documentation updated:</strong> Update model cards when models are retrained or deployed to new environments</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 4: SAGEMAKER MODEL DASHBOARD
    with tab4:
        st.header("Amazon SageMaker Model Dashboard")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Amazon SageMaker Model Dashboard provides a centralized interface for monitoring and managing all ML models deployed across your organization.
            It gives you visibility into model performance, data quality, and potential drift.
            
            **Key capabilities:**
            - View all models deployed in SageMaker endpoints
            - Monitor model performance metrics and drift detection
            - Track endpoint health and resource utilization
            - Receive alerts for drift detection and performance anomalies
            - Integrate with Model Cards for comprehensive model information
            """)
            
            st.markdown("""
            <div class="info-box">
            <h3>What Can You Monitor?</h3>
            <p>SageMaker Model Dashboard provides insights into:</p>
            <ul>
                <li><strong>Data Quality:</strong> Monitor for missing values, type mismatches, and distribution shifts</li>
                <li><strong>Model Quality:</strong> Track accuracy, precision, recall, and other performance metrics</li>
                <li><strong>Bias Drift:</strong> Detect changes in model fairness across demographic groups</li>
                <li><strong>Feature Attribution:</strong> Monitor changes in feature importance over time</li>
                <li><strong>Operational Metrics:</strong> Track invocations, latency, and resource utilization</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Model Dashboard diagram
            st.image("https://d1.awsstatic.com/ml-governance/governance_model-dashboard_how-it-works-diagram.6ede2e9669f88c29eb59e6c66b9f3a28516e3fe9.png", 
                    use_column_width=True)
            st.caption("SageMaker Model Dashboard provides centralized model monitoring")
        
        st.subheader("Model Dashboard Overview")
        
        # Create a summary of all models
        model_data = st.session_state.model_dashboard_data
        
        # Create model cards in a grid
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, (model_id, model) in enumerate(model_data.items()):
            info = model['info']
            with cols[i % 3]:
                # Set status color
                status_color = "#59BA47" if info['status'] == "Healthy" else ("#FF9900" if info['status'] == "Warning" else "#D13212")
                
                # Create alert badge if there are alerts
                alert_badge = f"""<span style="background-color: #D13212; color: white; padding: 1px 8px; border-radius: 10px; font-size: 12px; margin-left: 10px;">{len(info['alerts'])} Alerts</span>""" if info['alerts'] else ""
                
                st.markdown(f"""
                <div class="card" style="border-top: 4px solid {status_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div style="font-weight: bold; font-size: 16px;">{info['name']}</div>
                        <div>
                            <span style="background-color: {status_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px;">{info['status']}</span>{alert_badge}
                        </div>
                    </div>
                    <div style="margin-bottom: 15px; font-size: 13px;">
                        <strong>Endpoint:</strong> {info['endpoint']}<br>
                        <strong>Version:</strong> {info['version']}<br>
                        <strong>Updated:</strong> {info['last_updated']}
                    </div>
                    <div style="margin-bottom: 10px; font-size: 13px; color: #545B64;">
                        <strong>Drift Status:</strong> {info['drift_status']}
                    </div>
                    <div style="text-align: right;">
                        <button style="background-color: #FF9900; color: white; border: none; padding: 5px 10px; border-radius: 5px; font-size: 12px; cursor: pointer;" onclick="document.querySelector('[data-baseweb=\\'tab\\']', '[id=\\'tabs-1-tab-3\\']').click()"> View Details</button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if model_id == st.session_state.selected_model:
                    if st.button(f"View {info['name']} Dashboard", key=f"view_{model_id}"):
                        st.session_state.selected_model = model_id
                        st.rerun()
        
        st.divider()
        
        # Display detailed dashboard for the selected model
        st.subheader(f"Model Dashboard: {model_data[st.session_state.selected_model]['info']['name']}")
        
        # Create dashboard visualization
        create_dashboard_visualization(model_data, st.session_state.selected_model)
        
        # Example code for setting up model monitoring
        st.subheader("Setting Up Model Monitoring")
        
        st.markdown("""
        Here's how you can set up comprehensive model monitoring using SageMaker Model Monitor:
        """)
        
        st.code("""
# Set up SageMaker Model Monitoring
import boto3
import sagemaker
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Configure data capture for the endpoint
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=f's3://{bucket}/model-monitor/data-capture'
)

# Deploy model with data capture enabled
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='my-monitored-endpoint',
    data_capture_config=data_capture_config
)

# Create model quality monitor
model_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

# Generate baseline statistics
model_monitor.suggest_baseline(
    baseline_dataset=f's3://{bucket}/model-monitor/baseline/training-dataset.csv',
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=f's3://{bucket}/model-monitor/baseline-stats',
    wait=True
)

# Create monitoring schedule
model_monitor.create_monitoring_schedule(
    monitor_schedule_name='my-model-monitor-schedule',
    endpoint_input=predictor.endpoint_name,
    statistics=model_monitor.baseline_statistics(),
    constraints=model_monitor.suggested_constraints(),
    schedule_expression='cron(0 * ? * * *)',  # Run hourly
    enable_cloudwatch_metrics=True
)

print(f"Model monitoring scheduled for endpoint {predictor.endpoint_name}")
        """)
        
        st.markdown("""
        <div class="warning-box">
        <h3>Best Practices for Model Monitoring</h3>
        <ul>
            <li><strong>Establish meaningful baselines:</strong> Use representative training data to create baseline statistics</li>
            <li><strong>Monitor gradually:</strong> Start with data quality monitoring, then add more complex monitoring types</li>
            <li><strong>Set appropriate thresholds:</strong> Balance sensitivity to detect issues without generating too many false alarms</li>
            <li><strong>Create automated workflows:</strong> Set up automated responses to detected drift (alerts, retraining)</li>
            <li><strong>Implement human review:</strong> Establish processes for humans to review and act on monitoring alerts</li>
            <li><strong>Use dashboard insights:</strong> Regularly review dashboard data to identify trends before they become issues</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Add footer
    st.markdown("""
    <div class="footer">
    © 2025, Amazon Web Services, Inc. or its affiliates. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # First check authentication
    is_authenticated = authenticate.login()
    
    # If authenticated, show the main app content
    if is_authenticated:
        main()

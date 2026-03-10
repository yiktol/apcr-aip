# Technology Stack

## Core Technologies

- **Frontend Framework**: Streamlit 1.52.0
- **Python Version**: 3.12+
- **Deployment**: AWS CloudFormation with ALB/ASG

## AWS Services

- Amazon Bedrock (48 foundation models across 11 providers)
- Amazon SageMaker
- Amazon Cognito (authentication with Managed Login v2 and custom branding)
- Amazon S3, KMS, Macie, IAM
- Amazon A2I, CloudTrail, CloudWatch
- Amazon CloudFront (CDN with WAF)
- Amazon Route 53 (DNS management)
- AWS Certificate Manager (SSL/TLS certificates)

## Key Python Libraries

### Data & ML
- pandas 2.1.1, numpy 1.26.0
- scikit-learn >=1.3.0
- xgboost 2.0.3
- transformers >=4.30.0
- torch
- shap >=0.44.0

### Visualization
- matplotlib 3.8.0
- seaborn 0.13.0
- plotly 5.18.0

### AWS & AI
- boto3, botocore
- langchain_aws, langchain_core, langchain_community
- strands-agents

### Other
- Pillow 10.1.0
- chromadb (vector database)
- pypdf (document processing)

## Common Commands

### Local Development

**Session 0 (Static HTML):**
```bash
cd session0
python3 -m http.server 8080
```

**Sessions 1-5 (Streamlit):**
```bash
cd sessionX
./setup.sh --port 808X
```

Or manually:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Home.py --server.port 808X
```

## Port Configuration

- Session 0: 8080 (HTTP server)
- Session 1: 8081
- Session 2: 8082
- Session 3: 8083
- Session 4: 8084
- Session 5: 8085

## Environment Variables

Required for production:
- `COGNITO_DOMAIN`
- `COGNITO_APP_CLIENT_ID`
- `COGNITO_APP_CLIENT_SECRET`
- `COGNITO_REDIRECT_URI_1`

## Deployment

### CloudFormation Infrastructure

The `deploy/` directory contains CloudFormation templates for full infrastructure deployment:

```bash
cd deploy
./deploy.sh
```

This deploys:
- VPC with multi-AZ subnets
- Cognito User Pool with Managed Login v2 and AWS-branded styling
- CloudFront distribution with WAF
- Application Load Balancer and Auto Scaling Group
- Route 53 DNS records

Templates are uploaded to S3 and deployed via nested stacks in the `ap-southeast-1` region.

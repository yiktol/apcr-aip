# AWS AI Practitioner Certification Readiness

An interactive e-learning platform built to help users prepare for the AWS Certified AI Practitioner exam.

## Overview

This application provides hands-on learning experiences covering:
- AI, ML, and Generative AI fundamentals
- AWS AI/ML services and best practices
- Foundation models and transformer architectures
- Prompt engineering and model customization
- Responsible AI, security, and governance
- Real-world use cases and applications

## Project Structure

The project is organized into 6 sessions:

### Session 0: Landing Page
- Static HTML landing page with navigation
- Modern responsive design
- Links to all training sessions
- Error pages (403, 404, 502, 503, 504)

### Session 1: Kickoff & Fundamentals of AI and ML
- ML terminology and concepts
- Binary and multi-class classification
- Regression and clustering
- Reinforcement learning
- Real-world use cases (fraud detection, healthcare, cybersecurity)
- Interactive games and visualizations

### Session 2: Fundamentals of Generative AI
- Use cases and concerns of GenAI
- Transformer architecture
- Word embeddings and vector representations
- Titan Text and Multimodal Embeddings
- Similarity metrics
- Context and attention mechanisms

### Session 3: Applications of Foundation Models
- Elements of prompts
- Prompt engineering techniques
- Chain-of-Thought and Tree-of-Thought
- LLM security and vulnerabilities
- Model evaluation
- Fine-tuning dataset guide
- Retrieval Augmented Generation (RAG)
- Bedrock Agents and Guardrails
- Prompt templates

### Session 4: Responsible AI & Security, Compliance, and Governance
- Protecting against vulnerabilities
- Governance with SageMaker
- Responsible AI principles (8 dimensions)
- Bias vs Variance demonstrations
- Dataset bias types
- Bedrock Guardrails
- Mitigating bias in images

### Session 5: Practical Applications
- Market research assistant
- Product description generator
- Image generation
- Sales assistant
- ECS Fargate deployment infrastructure

## Technology Stack

### Core Technologies
- **Frontend**: Streamlit 1.52.0
- **Python**: 3.12
- **AWS Services**: Bedrock, SageMaker, Cognito, S3, KMS, Macie, IAM, A2I, CloudTrail, CloudWatch

### Key Libraries
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- boto3, botocore
- transformers, torch
- langchain_aws

## Quick Start

### Prerequisites
- Python 3.12+
- AWS Account with Bedrock access
- AWS CLI configured
- Docker (optional, for containerized deployment)

### Local Development

**Session 0 (Static HTML):**
```bash
cd apcr-aip/session0
python3 -m http.server 8080
# Access at http://localhost:8080
```

**Sessions 1-5 (Streamlit Apps):**
```bash
cd apcr-aip/sessionX
./setup.sh --port 808X
# Or manually:
# python -m venv .venv
# source .venv/bin/activate  # On Windows: .venv\Scripts\activate
# pip install -r requirements.txt
# streamlit run Home.py --server.port 808X
```

### Docker Deployment

```bash
cd apcr-aip/sessionX
docker build -t session-name .
docker run -p 8501:8501 session-name
```

### AWS ECS Deployment (Session 5)

```bash
cd apcr-aip/session5/infrastructure
# Update parameters.json with your VPC and subnet IDs
./scripts/deploy.sh
```

## Features

### Interactive Learning
- Real-time visualizations with Plotly and Matplotlib
- Knowledge checks and quizzes
- Hands-on AWS service demonstrations
- Interactive parameter tuning

### AWS Integration
- 48 foundation models across 11 providers
- Amazon Bedrock for generative AI
- SageMaker for ML workflows
- Cognito for authentication
- Multi-metric model evaluation

### Responsible AI
- Bias detection and mitigation
- Model explainability (SHAP values)
- Content safety and guardrails
- Privacy and security best practices
- Governance and compliance tracking

## Documentation

- **PROJECT_CHANGELOG.md** - Complete project history and updates
- **Session 5 README.md** - ECS deployment guide
- **.kiro/steering/** - Development guidelines and standards

## Port Configuration

Default ports by session:
- Session 0: 8080 (HTTP server)
- Session 1: 8081
- Session 2: 8082
- Session 3: 8083
- Session 4: 8084
- Session 5: 8501 (Docker default)

## Environment Variables

Required for production deployment:
- `COGNITO_DOMAIN`
- `COGNITO_APP_CLIENT_ID`
- `COGNITO_APP_CLIENT_SECRET`
- `COGNITO_REDIRECT_URI_1`

## Contributing

This is an educational project. Contributions are welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License (2025-2026)

See LICENSE and NOTICE files for details.

## Disclaimer

This project is not affiliated with, endorsed by, or sponsored by Amazon Web Services, Inc. or its affiliates. AWS, Amazon Bedrock, Amazon SageMaker, and related service names are trademarks of Amazon.com, Inc. or its affiliates.

This is an educational project designed to help users prepare for the AWS Certified AI Practitioner exam. It is not an official AWS training resource.

## Support

For issues or questions:
1. Check the documentation in PROJECT_CHANGELOG.md
2. Review session-specific README files
3. Check CloudFormation events (for deployment issues)
4. Review ECS task logs (for runtime issues)

## Acknowledgments

Built with:
- Streamlit for the interactive UI
- AWS Bedrock for foundation models
- AWS SageMaker for ML capabilities
- Open source libraries (see NOTICE file)

---

**Ready to start your AWS AI Practitioner certification journey!** 🚀
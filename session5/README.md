# ECS Bedrock Application Deployment

This repository contains CloudFormation templates and deployment scripts for deploying an ECS-based application that interfaces with AWS Bedrock and S3 services in the Singapore region (ap-southeast-1).

## Architecture Overview

The solution deploys:
- **ECS Fargate Cluster**: Serverless container orchestration
- **Application Load Balancer (ALB)**: Distributes traffic across containers
- **ECS Service**: Manages container lifecycle and scaling
- **Task Definition**: Defines container configuration
- **IAM Roles**: Provides permissions for Bedrock and S3 access
- **Auto Scaling**: Automatically scales based on CPU utilization
- **CloudWatch Logs**: Centralized logging

## Architecture Diagram

```
Internet → ALB (Public Subnets) → ECS Service (Private Subnets) → Bedrock/S3
                                         ↓
                                   ECR Repository
```

## Prerequisites

1. **AWS Account**: With appropriate permissions
2. **AWS CLI**: Version 2.x or higher
3. **Existing VPC**: With public and private subnets
4. **ECR Repository**: With Docker image pushed
5. **Docker Image**: Application listening on port 8501

## Required IAM Permissions

The deployment user needs permissions for:
- CloudFormation
- ECS
- EC2 (Security Groups, VPC)
- Elastic Load Balancing
- IAM (Roles and Policies)
- CloudWatch Logs
- Application Auto Scaling

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| VpcId | ID of existing VPC | Required |
| PublicSubnetIds | Comma-separated list of public subnet IDs | Required |
| PrivateSubnetIds | Comma-separated list of private subnet IDs | Required |
| ECRRepositoryURI | Full URI of ECR repository with tag | Required |
| ContainerCpu | CPU units (256, 512, 1024, 2048, 4096) | 512 |
| ContainerMemory | Memory in MB | 1024 |
| DesiredCount | Number of tasks to run | 2 |
| Environment | Environment name | production |

## Quick Start

1. Clone this repository
2. Configure AWS CLI with Singapore region
3. Update `parameters.json` with your values
4. Run deployment script

```bash
./deploy.sh
```

## File Structure

```
.
├── README.md
├── ecs-bedrock-deployment.yaml
├── parameters.json
├── deploy.sh
├── validate.sh
├── update-stack.sh
├── delete-stack.sh
└── scripts/
    ├── check-prerequisites.sh
    └── get-stack-outputs.sh
```

## Security Considerations

- ECS tasks run in private subnets
- ALB in public subnets with security groups
- IAM roles follow least privilege principle
- Containers access Bedrock and S3 via VPC endpoints (recommended)
- CloudWatch Logs for audit trail

## Monitoring

- CloudWatch Container Insights enabled
- ALB access logs (optional configuration)
- ECS task logs in CloudWatch
- Auto-scaling metrics

## Cost Optimization

- Fargate Spot instances (optional)
- Auto-scaling based on demand
- Right-sized container resources
- CloudWatch log retention policy (30 days)

## Support

For issues or questions:
1. Check CloudFormation events
2. Review ECS task logs
3. Verify IAM permissions
4. Check security group rules

## License

MIT License
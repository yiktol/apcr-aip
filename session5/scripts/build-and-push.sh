#!/bin/bash
set -e

# Configuration
AWS_REGION=${AWS_REGION:-ap-southeast-1}
AccountId=$(aws sts get-caller-identity --query "Account" --output text)
STACK_NAME=${STACK_NAME:-apcr-aip-session5}
IMAGE_TAG=${IMAGE_TAG:-latest}
ECRRepository=${ECRRepository:-apcr/aip-5}

# Check if ECR repository exists
echo "Checking if ECR repository exists..."
REPO_EXISTS=$(aws ecr describe-repositories \
    --repository-names "$ECRRepository" \
    --region "$AWS_REGION" \
    --query "repositories[0].repositoryUri" \
    --output text 2>/dev/null || echo "")

if [ -z "$REPO_EXISTS" ]; then
    echo "ECR repository '$ECRRepository' does not exist. Creating..."
    aws ecr create-repository \
        --repository-name "$ECRRepository" \
        --region "$AWS_REGION" \
        --output text
    echo "ECR repository '$ECRRepository' created successfully."
else
    echo "ECR repository '$ECRRepository' already exists."
fi

# Get ECR repository URI
ECR_URI="$AccountId.dkr.ecr.$AWS_REGION.amazonaws.com/$ECRRepository"

if [ -z "$ECR_URI" ]; then
  echo "Error: Could not retrieve ECR repository URI"
  exit 1
fi

echo "ECR Repository URI: $ECR_URI"

# Login to ECR
echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI

# Build Docker image
echo "Building Docker image..."
docker build -t $STACK_NAME:$IMAGE_TAG .

# Tag image for ECR
echo "Tagging image for ECR..."
docker tag $STACK_NAME:$IMAGE_TAG $ECR_URI:$IMAGE_TAG

# Push image to ECR
echo "Pushing image to ECR..."
docker push $ECR_URI:$IMAGE_TAG

echo "Successfully pushed $ECR_URI:$IMAGE_TAG"

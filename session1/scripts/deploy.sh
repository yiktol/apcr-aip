#!/bin/bash

# ECS Bedrock Application Deployment Script
# Region: Singapore (ap-southeast-1)

set -e

# Configuration
STACK_NAME="genai-solution-app"
TEMPLATE_FILE="template.yaml"
ACCOUNTID=$(aws sts get-caller-identity --query "Account" --output text)
REGION="ap-southeast-1"
PROFILE="default"  # Change if using named profile
ECRRepositoryURI="$ACCOUNTID.dkr.ecr.$REGION.amazonaws.com/aipapcr/aip-5"
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo "Deploying AWS infrastructure..."

# Get VPC and subnet information
VPC_ID=$(aws cloudformation describe-stacks --stack-name vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`VpcId`].OutputValue' \
  --output text)

SUBNET_A=$(aws cloudformation describe-stacks  --stack-name vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`PrivateSubnetOneId`].OutputValue' \
  --output text)
SUBNET_B=$(aws cloudformation describe-stacks  --stack-name vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`PrivateSubnetTwoId`].OutputValue' \
  --output text)
PUBLIC_SUBNET_C=$(aws cloudformation describe-stacks  --stack-name vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnetOneId`].OutputValue' \
  --output text)
PUBLIC_SUBNET_D=$(aws cloudformation describe-stacks  --stack-name vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnetTwoId`].OutputValue' \
  --output text)




# Check prerequisites
check_prerequisites() {
    print_message "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity --profile ${PROFILE} --region ${REGION} &> /dev/null; then
        print_error "AWS credentials not configured or invalid"
        exit 1
    fi
    
    # Check template file
    if [ ! -f "infrastructure/${TEMPLATE_FILE}" ]; then
        print_error "Template file infrastructure/${TEMPLATE_FILE} not found"
        exit 1
    fi
    
    
    print_message "Prerequisites check passed"
}

# Validate template
validate_template() {
    print_message "Validating CloudFormation template..."
    
    aws cloudformation validate-template \
        --template-body file://infrastructure/${TEMPLATE_FILE} \
        --profile ${PROFILE} \
        --region ${REGION} > /dev/null
    
    if [ $? -eq 0 ]; then
        print_message "Template validation successful"
    else
        print_error "Template validation failed"
        exit 1
    fi
}

# Check if stack exists
stack_exists() {
    aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --profile ${PROFILE} \
        --region ${REGION} &> /dev/null
    
    return $?
}

# Deploy stack
deploy_stack() {
    if stack_exists; then
        print_warning "Stack ${STACK_NAME} already exists. Use update-stack.sh to update."
        exit 1
    fi
    
    print_message "Creating stack ${STACK_NAME}..."
    
    aws cloudformation deploy \
        --stack-name ${STACK_NAME} \
        --template-file infrastructure/${TEMPLATE_FILE} \
        --parameter-overrides \
                VpcId=$VPC_ID \
                PrivateSubnetIds=$SUBNET_A,$SUBNET_B \
                PublicSubnetIds=$PUBLIC_SUBNET_C,$PUBLIC_SUBNET_D \
                ECRRepositoryURI=$ECRRepositoryURI \
        --capabilities CAPABILITY_NAMED_IAM \
        --profile ${PROFILE} \
        --region ${REGION} 

    
    if [ $? -eq 0 ]; then
        print_message "Stack creation initiated"
        print_message "Waiting for stack creation to complete..."
        
        aws cloudformation wait stack-create-complete \
            --stack-name ${STACK_NAME} \
            --profile ${PROFILE} \
            --region ${REGION}
        
        if [ $? -eq 0 ]; then
            print_message "Stack created successfully!"
            
            # Get outputs
            print_message "Stack Outputs:"
            aws cloudformation describe-stacks \
                --stack-name ${STACK_NAME} \
                --profile ${PROFILE} \
                --region ${REGION} \
                --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
                --output table
        else
            print_error "Stack creation failed"
            exit 1
        fi
    else
        print_error "Failed to initiate stack creation"
        exit 1
    fi
}

# Main execution
main() {
    print_message "Starting ECS Bedrock Application Deployment"
    print_message "Region: ${REGION}"
    print_message "Stack Name: ${STACK_NAME}"
    echo ""
    
    check_prerequisites
    validate_template
    deploy_stack
    
    echo ""
    print_message "Deployment completed successfully!"
    print_message "Application URL will be available at the LoadBalancerURL output"
}

# Run main function
main
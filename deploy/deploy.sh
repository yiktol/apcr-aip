#!/bin/bash

################################################################################
# CloudFormation Deployment Script
# Description: Deploy GenAI infrastructure with monitoring and error handling
################################################################################

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGION='ap-southeast-1'
STACK_NAME='genai'
TEMPLATE_PREFIX='templates/demo'
LOG_FILE="deployment-$(date +%Y%m%d-%H%M%S).log"

################################################################################
# Helper Functions
################################################################################

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1" | tee -a "$LOG_FILE"
}

check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    log_success "AWS CLI found"
}

check_aws_credentials() {
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        exit 1
    fi
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local user_arn=$(aws sts get-caller-identity --query Arn --output text)
    log_success "AWS credentials valid - Account: $account_id"
    log "User/Role: $user_arn"
}

get_ssm_parameter() {
    local param_name=$1
    local value=$(aws ssm get-parameter --name "$param_name" --query 'Parameter.Value' --output text 2>/dev/null)
    
    if [ -z "$value" ]; then
        log_error "Failed to retrieve SSM parameter: $param_name"
        exit 1
    fi
    
    echo "$value"
}

check_stack_exists() {
    local stack_name=$1
    aws cloudformation describe-stacks --stack-name "$stack_name" --region "$REGION" &> /dev/null
    return $?
}

wait_for_stack() {
    local stack_name=$1
    local operation=$2
    
    log "Waiting for stack $operation to complete..."
    
    local start_time=$(date +%s)
    local timeout=3600  # 1 hour timeout
    
    while true; do
        local status=$(aws cloudformation describe-stacks \
            --stack-name "$stack_name" \
            --region "$REGION" \
            --query 'Stacks[0].StackStatus' \
            --output text 2>/dev/null || echo "UNKNOWN")
        
        case $status in
            *COMPLETE)
                local end_time=$(date +%s)
                local duration=$((end_time - start_time))
                log_success "Stack $operation completed in ${duration}s - Status: $status"
                return 0
                ;;
            *FAILED|*ROLLBACK*)
                log_error "Stack $operation failed - Status: $status"
                show_stack_events "$stack_name" 10
                return 1
                ;;
            *IN_PROGRESS)
                echo -ne "\r${BLUE}Status: $status${NC} ($(( $(date +%s) - start_time ))s elapsed)"
                ;;
            UNKNOWN)
                log_error "Unable to get stack status"
                return 1
                ;;
        esac
        
        # Check timeout
        if [ $(($(date +%s) - start_time)) -gt $timeout ]; then
            log_error "Operation timed out after ${timeout}s"
            return 1
        fi
        
        sleep 10
    done
}

show_stack_events() {
    local stack_name=$1
    local count=${2:-10}
    
    log "Recent stack events:"
    aws cloudformation describe-stack-events \
        --stack-name "$stack_name" \
        --region "$REGION" \
        --max-items "$count" \
        --query 'StackEvents[*].[Timestamp,ResourceStatus,ResourceType,LogicalResourceId,ResourceStatusReason]' \
        --output table 2>/dev/null || log_warning "Could not retrieve stack events"
}

upload_templates() {
    local bucket=$1
    local script_dir="$(dirname "$(readlink -f "$0")")"
    
    log "Uploading CloudFormation templates from $script_dir to s3://$bucket/$TEMPLATE_PREFIX/"
    
    # Upload all YAML files from the script's directory
    if aws s3 cp "$script_dir/" "s3://$bucket/$TEMPLATE_PREFIX/" \
        --recursive \
        --exclude "*" \
        --include "*.yaml" \
        --region "$REGION" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Templates uploaded successfully"
    else
        log_error "Failed to upload templates"
        exit 1
    fi
}

validate_template() {
    local template_url=$1
    
    log "Validating CloudFormation template..."
    
    if aws cloudformation validate-template \
        --template-url "$template_url" \
        --region "$REGION" &> /dev/null; then
        log_success "Template validation passed"
    else
        log_error "Template validation failed"
        exit 1
    fi
}

deploy_stack() {
    local stack_name=$1
    local template_url=$2
    local bucket=$3
    
    if check_stack_exists "$stack_name"; then
        log "Stack '$stack_name' exists. Updating..."
        
        if aws cloudformation update-stack \
            --region "$REGION" \
            --stack-name "$stack_name" \
            --template-url "$template_url" \
            --parameters \
                ParameterKey=Prefix,ParameterValue="$TEMPLATE_PREFIX" \
                ParameterKey=BucketRegion,ParameterValue="$REGION" \
            --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_NAMED_IAM \
            2>&1 | tee -a "$LOG_FILE"; then
            
            wait_for_stack "$stack_name" "update"
        else
            if grep -q "No updates are to be performed" "$LOG_FILE"; then
                log_warning "No updates needed for stack"
                return 0
            else
                log_error "Stack update failed"
                return 1
            fi
        fi
    else
        log "Stack '$stack_name' does not exist. Creating..."
        
        if aws cloudformation create-stack \
            --region "$REGION" \
            --stack-name "$stack_name" \
            --template-url "$template_url" \
            --parameters \
                ParameterKey=Prefix,ParameterValue="$TEMPLATE_PREFIX" \
                ParameterKey=BucketRegion,ParameterValue="$REGION" \
            --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_NAMED_IAM \
            2>&1 | tee -a "$LOG_FILE"; then
            
            wait_for_stack "$stack_name" "create"
        else
            log_error "Stack creation failed"
            return 1
        fi
    fi
}

show_stack_outputs() {
    local stack_name=$1
    
    log "Stack Outputs:"
    aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue,Description]' \
        --output table 2>/dev/null || log_warning "No outputs available"
}

show_nested_stacks() {
    local stack_name=$1
    
    log "Nested Stacks:"
    aws cloudformation list-stack-resources \
        --stack-name "$stack_name" \
        --region "$REGION" \
        --query 'StackResourceSummaries[?ResourceType==`AWS::CloudFormation::Stack`].[LogicalResourceId,ResourceStatus,PhysicalResourceId]' \
        --output table 2>/dev/null || log_warning "No nested stacks found"
}

cleanup_on_error() {
    log_error "Deployment failed. Check $LOG_FILE for details."
    exit 1
}

################################################################################
# Main Deployment Flow
################################################################################

main() {
    log "=========================================="
    log "GenAI Infrastructure Deployment"
    log "=========================================="
    log "Region: $REGION"
    log "Stack Name: $STACK_NAME"
    log "Log File: $LOG_FILE"
    log "=========================================="
    
    # Pre-flight checks
    log "Running pre-flight checks..."
    check_aws_cli
    check_aws_credentials
    
    # Get S3 bucket from SSM
    log "Retrieving configuration from SSM Parameter Store..."
    BUCKET=$(get_ssm_parameter "/genai/cognito/BucketName")
    log_success "S3 Bucket: $BUCKET"
    
    # Verify bucket exists
    if ! aws s3 ls "s3://$BUCKET" --region "$REGION" &> /dev/null; then
        log_error "S3 bucket '$BUCKET' does not exist or is not accessible"
        exit 1
    fi
    log_success "S3 bucket accessible"
    
    # Upload templates
    upload_templates "$BUCKET"
    
    # Construct template URL
    TEMPLATE_URL="https://$BUCKET.s3.$REGION.amazonaws.com/$TEMPLATE_PREFIX/main.yaml"
    log "Template URL: $TEMPLATE_URL"
    
    # Validate template
    validate_template "$TEMPLATE_URL"
    
    # Deploy stack
    log "=========================================="
    log "Starting deployment..."
    log "=========================================="
    
    if deploy_stack "$STACK_NAME" "$TEMPLATE_URL" "$BUCKET"; then
        log_success "=========================================="
        log_success "Deployment completed successfully!"
        log_success "=========================================="
        
        # Show outputs
        show_stack_outputs "$STACK_NAME"
        show_nested_stacks "$STACK_NAME"
        
        log ""
        log "View stack in AWS Console:"
        log "https://$REGION.console.aws.amazon.com/cloudformation/home?region=$REGION#/stacks/stackinfo?stackId=$STACK_NAME"
        
        exit 0
    else
        cleanup_on_error
    fi
}

# Trap errors
trap cleanup_on_error ERR

# Run main function
main "$@"

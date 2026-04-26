#!/bin/bash

################################################################################
# CloudFormation Deployment Script
# Description: Deploy GenAI Essentials infrastructure with Lambda@Edge auth
#
# Deployment order:
#   1. Lambda@Edge auth stack (us-east-1) — reads genai/essentials for credentials
#   2. Store Lambda@Edge ARNs in SSM (ap-southeast-1)
#   3. Main stack (ap-southeast-1) — includes CloudFront, VPC, ALB, Cognito (nested)
#
# Note: Cognito is deployed as a nested stack inside main.yaml, not standalone.
#
# Usage:
#   ./deploy.sh              # Deploy everything (edge + ssm + main)
#   ./deploy.sh edge         # Deploy Lambda@Edge stack only
#   ./deploy.sh ssm          # Store Lambda@Edge ARNs in SSM only
#   ./deploy.sh main         # Deploy main stack only
################################################################################

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
REGION='ap-southeast-1'
EDGE_REGION='us-east-1'
STACK_NAME='genai'
EDGE_STACK_NAME='genai-ess-edge-auth'
TEMPLATE_PREFIX='templates/demo'
PARAMETER_PREFIX='/genai/cognito'
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
    local region=${2:-$REGION}
    local value=$(aws ssm get-parameter --name "$param_name" --region "$region" --query 'Parameter.Value' --output text 2>/dev/null)
    
    if [ -z "$value" ]; then
        log_error "Failed to retrieve SSM parameter: $param_name"
        exit 1
    fi
    
    echo "$value"
}

put_ssm_parameter() {
    local param_name=$1
    local param_value=$2
    local region=${3:-$REGION}
    
    aws ssm put-parameter \
        --name "$param_name" \
        --value "$param_value" \
        --type String \
        --overwrite \
        --region "$region" &> /dev/null
    
    if [ $? -eq 0 ]; then
        log_success "Stored SSM parameter: $param_name"
    else
        log_error "Failed to store SSM parameter: $param_name"
        exit 1
    fi
}

check_stack_exists() {
    local stack_name=$1
    local region=${2:-$REGION}
    aws cloudformation describe-stacks --stack-name "$stack_name" --region "$region" &> /dev/null
    return $?
}

wait_for_stack() {
    local stack_name=$1
    local operation=$2
    local region=${3:-$REGION}
    
    log "Waiting for stack $operation to complete..."
    
    local start_time=$(date +%s)
    local timeout=3600
    
    while true; do
        local status=$(aws cloudformation describe-stacks \
            --stack-name "$stack_name" \
            --region "$region" \
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
                show_stack_events "$stack_name" 10 "$region"
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
    local region=${3:-$REGION}
    
    log "Recent stack events:"
    aws cloudformation describe-stack-events \
        --stack-name "$stack_name" \
        --region "$region" \
        --max-items "$count" \
        --query 'StackEvents[*].[Timestamp,ResourceStatus,ResourceType,LogicalResourceId,ResourceStatusReason]' \
        --output table 2>/dev/null || log_warning "Could not retrieve stack events"
}

upload_templates() {
    local bucket=$1
    local script_dir="$(dirname "$(readlink -f "$0")")"
    
    log "Uploading CloudFormation templates from $script_dir to s3://$bucket/$TEMPLATE_PREFIX/"
    
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
    local region=${2:-$REGION}
    
    log "Validating CloudFormation template..."
    
    if aws cloudformation validate-template \
        --template-url "$template_url" \
        --region "$region" &> /dev/null; then
        log_success "Template validation passed"
    else
        log_error "Template validation failed"
        exit 1
    fi
}

show_stack_outputs() {
    local stack_name=$1
    local region=${2:-$REGION}
    
    log "Stack Outputs:"
    aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --region "$region" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue,Description]' \
        --output table 2>/dev/null || log_warning "No outputs available"
}

show_nested_stacks() {
    local stack_name=$1
    local region=${2:-$REGION}
    
    log "Nested Stacks:"
    aws cloudformation list-stack-resources \
        --stack-name "$stack_name" \
        --region "$region" \
        --query 'StackResourceSummaries[?ResourceType==`AWS::CloudFormation::Stack`].[LogicalResourceId,ResourceStatus,PhysicalResourceId]' \
        --output table 2>/dev/null || log_warning "No nested stacks found"
}

cleanup_on_error() {
    log_error "Deployment failed. Check $LOG_FILE for details."
    exit 1
}

################################################################################
# Step 1: Deploy Lambda@Edge Auth Stack (us-east-1)
################################################################################

deploy_edge_stack() {
    log "=========================================="
    log "Step 1: Deploy Lambda@Edge Auth Stack"
    log "Region: $EDGE_REGION (required for Lambda@Edge)"
    log "=========================================="
    
    local script_dir="$(dirname "$(readlink -f "$0")")"
    local template_file="$script_dir/lambda-edge-auth.yaml"
    
    if [ ! -f "$template_file" ]; then
        log_error "Template not found: $template_file"
        exit 1
    fi
    
    local bucket
    bucket=$(get_ssm_parameter "${PARAMETER_PREFIX}/BucketName")
    
    log "Uploading Lambda@Edge template to S3..."
    aws s3 cp "$template_file" "s3://$bucket/$TEMPLATE_PREFIX/lambda-edge-auth.yaml" \
        --region "$REGION" 2>&1 | tee -a "$LOG_FILE"
    log_success "Uploaded lambda-edge-auth.yaml"
    
    local template_url="https://$bucket.s3.$REGION.amazonaws.com/$TEMPLATE_PREFIX/lambda-edge-auth.yaml"
    
    if check_stack_exists "$EDGE_STACK_NAME" "$EDGE_REGION"; then
        log "Stack '$EDGE_STACK_NAME' exists in $EDGE_REGION. Updating..."
        
        local update_output
        update_output=$(aws cloudformation update-stack \
            --region "$EDGE_REGION" \
            --stack-name "$EDGE_STACK_NAME" \
            --template-url "$template_url" \
            --parameters \
                ParameterKey=ParameterPrefix,ParameterValue="$PARAMETER_PREFIX" \
            --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
            2>&1) || true
        
        if echo "$update_output" | grep -q "No updates are to be performed"; then
            log_warning "No updates needed for Lambda@Edge stack"
        else
            echo "$update_output" | tee -a "$LOG_FILE"
            wait_for_stack "$EDGE_STACK_NAME" "update" "$EDGE_REGION"
        fi
    else
        log "Stack '$EDGE_STACK_NAME' does not exist. Creating in $EDGE_REGION..."
        
        aws cloudformation create-stack \
            --region "$EDGE_REGION" \
            --stack-name "$EDGE_STACK_NAME" \
            --template-url "$template_url" \
            --parameters \
                ParameterKey=ParameterPrefix,ParameterValue="$PARAMETER_PREFIX" \
            --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
            2>&1 | tee -a "$LOG_FILE"
        
        wait_for_stack "$EDGE_STACK_NAME" "create" "$EDGE_REGION"
    fi
    
    show_stack_outputs "$EDGE_STACK_NAME" "$EDGE_REGION"
    log_success "Lambda@Edge stack deployed successfully"
}

################################################################################
# Step 3: Store Lambda@Edge ARNs in SSM (ap-southeast-1)
################################################################################

store_edge_arns_in_ssm() {
    log "=========================================="
    log "Step 2: Store Lambda@Edge ARNs in SSM"
    log "=========================================="
    
    local auth_check_arn
    auth_check_arn=$(aws cloudformation describe-stacks \
        --stack-name "$EDGE_STACK_NAME" \
        --region "$EDGE_REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`AuthCheckFunctionArn`].OutputValue' \
        --output text 2>/dev/null)
    
    local auth_callback_arn
    auth_callback_arn=$(aws cloudformation describe-stacks \
        --stack-name "$EDGE_STACK_NAME" \
        --region "$EDGE_REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`AuthCallbackFunctionArn`].OutputValue' \
        --output text 2>/dev/null)
    
    if [ -z "$auth_check_arn" ] || [ -z "$auth_callback_arn" ]; then
        log_error "Failed to retrieve Lambda@Edge ARNs from stack outputs"
        log_error "AuthCheckArn: $auth_check_arn"
        log_error "AuthCallbackArn: $auth_callback_arn"
        exit 1
    fi
    
    log "AuthCheckVersionArn: $auth_check_arn"
    log "AuthCallbackVersionArn: $auth_callback_arn"
    
    put_ssm_parameter "${PARAMETER_PREFIX}/AuthCheckVersionArn" "$auth_check_arn" "$REGION"
    put_ssm_parameter "${PARAMETER_PREFIX}/AuthCallbackVersionArn" "$auth_callback_arn" "$REGION"
    
    log_success "Lambda@Edge ARNs stored in SSM Parameter Store"
}

################################################################################
# Step 4: Deploy Main Stack (ap-southeast-1)
################################################################################

deploy_main_stack() {
    log "=========================================="
    log "Step 3: Deploy Main Stack"
    log "Region: $REGION"
    log "=========================================="
    
    local bucket
    bucket=$(get_ssm_parameter "${PARAMETER_PREFIX}/BucketName")
    log_success "S3 Bucket: $bucket"
    
    if ! aws s3 ls "s3://$bucket" --region "$REGION" &> /dev/null; then
        log_error "S3 bucket '$bucket' does not exist or is not accessible"
        exit 1
    fi
    log_success "S3 bucket accessible"
    
    upload_templates "$bucket"
    
    local template_url="https://$bucket.s3.$REGION.amazonaws.com/$TEMPLATE_PREFIX/main.yaml"
    log "Template URL: $template_url"
    
    validate_template "$template_url"
    
    if check_stack_exists "$STACK_NAME"; then
        log "Stack '$STACK_NAME' exists. Updating..."
        
        local update_output
        update_output=$(aws cloudformation update-stack \
            --region "$REGION" \
            --stack-name "$STACK_NAME" \
            --template-url "$template_url" \
            --parameters \
                ParameterKey=Prefix,ParameterValue="$TEMPLATE_PREFIX" \
                ParameterKey=BucketRegion,ParameterValue="$REGION" \
                ParameterKey=DeployNonce,ParameterValue="$(date +%s)" \
            --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_NAMED_IAM \
            2>&1) || true
        
        if echo "$update_output" | grep -q "No updates are to be performed"; then
            log_warning "No updates needed for main stack"
        else
            echo "$update_output" | tee -a "$LOG_FILE"
            wait_for_stack "$STACK_NAME" "update"
        fi
    else
        log "Stack '$STACK_NAME' does not exist. Creating..."
        
        aws cloudformation create-stack \
            --region "$REGION" \
            --stack-name "$STACK_NAME" \
            --template-url "$template_url" \
            --parameters \
                ParameterKey=Prefix,ParameterValue="$TEMPLATE_PREFIX" \
                ParameterKey=BucketRegion,ParameterValue="$REGION" \
                ParameterKey=DeployNonce,ParameterValue="$(date +%s)" \
            --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_NAMED_IAM \
            2>&1 | tee -a "$LOG_FILE"
        
        wait_for_stack "$STACK_NAME" "create"
    fi
    
    show_stack_outputs "$STACK_NAME"
    show_nested_stacks "$STACK_NAME"
    log_success "Main stack deployed successfully"
}

################################################################################
# Main Deployment Flow
################################################################################

main() {
    local target=${1:-all}
    
    log "=========================================="
    log "GenAI Essentials Infrastructure Deployment"
    log "=========================================="
    log "Target: $target"
    log "Main Region: $REGION"
    log "Edge Region: $EDGE_REGION"
    log "Main Stack: $STACK_NAME"
    log "Edge Stack: $EDGE_STACK_NAME"
    log "Log File: $LOG_FILE"
    log "=========================================="
    
    log "Running pre-flight checks..."
    check_aws_cli
    check_aws_credentials
    
    case $target in
        edge)
            deploy_edge_stack
            ;;
        ssm)
            store_edge_arns_in_ssm
            ;;
        main)
            deploy_main_stack
            ;;
        all)
            deploy_edge_stack
            store_edge_arns_in_ssm
            deploy_main_stack
            ;;
        *)
            log_error "Unknown target: $target"
            log "Usage: $0 [edge|ssm|main|all]"
            exit 1
            ;;
    esac
    
    log ""
    log_success "=========================================="
    log_success "Deployment completed successfully!"
    log_success "=========================================="
    
    if [ "$target" = "all" ] || [ "$target" = "main" ]; then
        log ""
        log "View stack in AWS Console:"
        log "https://$REGION.console.aws.amazon.com/cloudformation/home?region=$REGION#/stacks/stackinfo?stackId=$STACK_NAME"
    fi
    
    if [ "$target" = "all" ] || [ "$target" = "edge" ]; then
        log ""
        log "View Lambda@Edge stack in AWS Console:"
        log "https://$EDGE_REGION.console.aws.amazon.com/cloudformation/home?region=$EDGE_REGION#/stacks/stackinfo?stackId=$EDGE_STACK_NAME"
    fi
}

trap cleanup_on_error ERR

main "$@"

#!/bin/bash

################################################################################
# CloudFormation Cleanup Script
# Description: Delete GenAI infrastructure stack and associated resources
################################################################################

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration (must match deploy.sh)
REGION='ap-southeast-1'
STACK_NAME='genai'
TEMPLATE_PREFIX='templates/demo'
LOG_FILE="cleanup-$(date +%Y%m%d-%H%M%S).log"

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
    log_success "AWS credentials valid - Account: $account_id"
}

check_stack_exists() {
    local stack_name=$1
    aws cloudformation describe-stacks --stack-name "$stack_name" --region "$REGION" &> /dev/null
    return $?
}

show_stack_resources() {
    local stack_name=$1

    log "Resources in stack '$stack_name':"
    aws cloudformation list-stack-resources \
        --stack-name "$stack_name" \
        --region "$REGION" \
        --query 'StackResourceSummaries[*].[ResourceType,LogicalResourceId,ResourceStatus]' \
        --output table 2>/dev/null || log_warning "Could not retrieve stack resources"
}

empty_s3_buckets() {
    local stack_name=$1

    log "Checking for S3 buckets in stack..."
    local buckets=$(aws cloudformation list-stack-resources \
        --stack-name "$stack_name" \
        --region "$REGION" \
        --query 'StackResourceSummaries[?ResourceType==`AWS::S3::Bucket`].PhysicalResourceId' \
        --output text 2>/dev/null)

    if [ -z "$buckets" ]; then
        log "No S3 buckets found in stack"
        return 0
    fi

    for bucket in $buckets; do
        log "Emptying S3 bucket: $bucket"
        aws s3 rm "s3://$bucket" --recursive --region "$REGION" 2>&1 | tee -a "$LOG_FILE" || true

        # Delete versioned objects if versioning is enabled
        local version_count=$(aws s3api list-object-versions \
            --bucket "$bucket" \
            --region "$REGION" \
            --query 'length(Versions || `[]`)' \
            --output text 2>/dev/null || echo "0")

        if [ "$version_count" -gt 0 ] 2>/dev/null; then
            log "Deleting $version_count versioned objects in: $bucket"
            local versions=$(aws s3api list-object-versions \
                --bucket "$bucket" \
                --region "$REGION" \
                --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
                --output json 2>/dev/null)
            aws s3api delete-objects \
                --bucket "$bucket" \
                --region "$REGION" \
                --delete "$versions" 2>&1 | tee -a "$LOG_FILE" || true
        fi

        # Delete delete markers
        local marker_count=$(aws s3api list-object-versions \
            --bucket "$bucket" \
            --region "$REGION" \
            --query 'length(DeleteMarkers || `[]`)' \
            --output text 2>/dev/null || echo "0")

        if [ "$marker_count" -gt 0 ] 2>/dev/null; then
            log "Deleting $marker_count delete markers in: $bucket"
            local markers=$(aws s3api list-object-versions \
                --bucket "$bucket" \
                --region "$REGION" \
                --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' \
                --output json 2>/dev/null)
            aws s3api delete-objects \
                --bucket "$bucket" \
                --region "$REGION" \
                --delete "$markers" 2>&1 | tee -a "$LOG_FILE" || true
        fi

        log_success "Bucket emptied: $bucket"
    done
}

empty_nested_s3_buckets() {
    local stack_name=$1

    log "Checking nested stacks for S3 buckets..."
    local nested_stacks=$(aws cloudformation list-stack-resources \
        --stack-name "$stack_name" \
        --region "$REGION" \
        --query 'StackResourceSummaries[?ResourceType==`AWS::CloudFormation::Stack`].PhysicalResourceId' \
        --output text 2>/dev/null)

    for nested_stack in $nested_stacks; do
        empty_s3_buckets "$nested_stack"
    done
}

delete_vpc_endpoints() {
    local stack_name=$1

    log "Looking up VPC from nested VpcStack..."

    # Get the VpcStack physical resource ID (nested stack ARN)
    local vpc_stack_id=$(aws cloudformation list-stack-resources \
        --stack-name "$stack_name" \
        --region "$REGION" \
        --query 'StackResourceSummaries[?LogicalResourceId==`VpcStack`].PhysicalResourceId' \
        --output text 2>/dev/null)

    if [ -z "$vpc_stack_id" ]; then
        log_warning "VpcStack not found in stack — skipping VPC endpoint cleanup"
        return 0
    fi

    # Get the VPC ID from the nested stack outputs
    local vpc_id=$(aws cloudformation describe-stacks \
        --stack-name "$vpc_stack_id" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`VpcId`].OutputValue' \
        --output text 2>/dev/null)

    if [ -z "$vpc_id" ]; then
        log_warning "Could not resolve VPC ID from VpcStack — skipping VPC endpoint cleanup"
        return 0
    fi

    log "Found VPC: $vpc_id"
    log "Listing VPC endpoints..."

    local endpoint_ids=$(aws ec2 describe-vpc-endpoints \
        --region "$REGION" \
        --filters "Name=vpc-id,Values=$vpc_id" \
        --query 'VpcEndpoints[*].VpcEndpointId' \
        --output text 2>/dev/null)

    if [ -z "$endpoint_ids" ]; then
        log "No VPC endpoints found for $vpc_id"
        return 0
    fi

    for endpoint_id in $endpoint_ids; do
        log "Deleting VPC endpoint: $endpoint_id"
        if aws ec2 delete-vpc-endpoints \
            --region "$REGION" \
            --vpc-endpoint-ids "$endpoint_id" 2>&1 | tee -a "$LOG_FILE"; then
            log_success "Deleted VPC endpoint: $endpoint_id"
        else
            log_warning "Failed to delete VPC endpoint: $endpoint_id (may already be deleted)"
        fi
    done

    log_success "VPC endpoint cleanup complete"
}

delete_cloudwatch_log_groups_post() {
    local lambda_names=$1

    log "Cleaning up CloudWatch log groups created by the project..."

    # Delete /aws/lambda/<function-name> log groups for pre-collected Lambda functions
    for func_name in $lambda_names; do
        local log_group="/aws/lambda/$func_name"
        local found=$(aws logs describe-log-groups \
            --log-group-name-prefix "$log_group" \
            --region "$REGION" \
            --query "logGroups[?logGroupName==\`$log_group\`].logGroupName" \
            --output text 2>/dev/null)

        if [ -n "$found" ]; then
            log "Deleting log group: $log_group"
            aws logs delete-log-group --log-group-name "$log_group" --region "$REGION" 2>&1 | tee -a "$LOG_FILE" || true
            log_success "Deleted: $log_group"
        fi
    done

    # Delete project-related log groups by known prefixes
    local prefixes=("/aws/fis/" "/genai/" "/aws/cloudfront/")
    for prefix in "${prefixes[@]}"; do
        local groups=$(aws logs describe-log-groups \
            --log-group-name-prefix "$prefix" \
            --region "$REGION" \
            --query 'logGroups[*].logGroupName' \
            --output text 2>/dev/null)

        for group in $groups; do
            log "Deleting log group: $group"
            aws logs delete-log-group --log-group-name "$group" --region "$REGION" 2>&1 | tee -a "$LOG_FILE" || true
            log_success "Deleted: $group"
        done
    done

    log_success "CloudWatch log group cleanup complete"
}

delete_uploaded_templates() {
    log "Retrieving S3 bucket name from SSM..."
    local bucket=$(aws ssm get-parameter \
        --name "/genai/cognito/BucketName" \
        --query 'Parameter.Value' \
        --output text 2>/dev/null || true)

    if [ -n "$bucket" ]; then
        log "Cleaning up uploaded templates from s3://$bucket/$TEMPLATE_PREFIX/"
        aws s3 rm "s3://$bucket/$TEMPLATE_PREFIX/" --recursive --region "$REGION" 2>&1 | tee -a "$LOG_FILE" || true
        log_success "Templates cleaned up"
    else
        log_warning "Could not retrieve bucket name from SSM — skipping template cleanup"
    fi
}

wait_for_delete() {
    local stack_name=$1

    log "Waiting for stack deletion to complete..."

    local start_time=$(date +%s)
    local timeout=3600

    while true; do
        local status=$(aws cloudformation describe-stacks \
            --stack-name "$stack_name" \
            --region "$REGION" \
            --query 'Stacks[0].StackStatus' \
            --output text 2>/dev/null || echo "DELETE_COMPLETE")

        case $status in
            DELETE_COMPLETE)
                local end_time=$(date +%s)
                local duration=$((end_time - start_time))
                log_success "Stack deleted successfully in ${duration}s"
                return 0
                ;;
            DELETE_FAILED)
                log_error "Stack deletion failed"
                aws cloudformation describe-stack-events \
                    --stack-name "$stack_name" \
                    --region "$REGION" \
                    --max-items 10 \
                    --query 'StackEvents[?ResourceStatus==`DELETE_FAILED`].[ResourceType,LogicalResourceId,ResourceStatusReason]' \
                    --output table 2>/dev/null || true
                return 1
                ;;
            DELETE_IN_PROGRESS)
                echo -ne "\r${BLUE}Status: $status${NC} ($(( $(date +%s) - start_time ))s elapsed)"
                ;;
            *)
                log_warning "Unexpected status: $status"
                ;;
        esac

        if [ $(($(date +%s) - start_time)) -gt $timeout ]; then
            log_error "Deletion timed out after ${timeout}s"
            return 1
        fi

        sleep 10
    done
}

################################################################################
# Main Cleanup Flow
################################################################################

main() {
    log "=========================================="
    log "GenAI Infrastructure Cleanup"
    log "=========================================="
    log "Region: $REGION"
    log "Stack Name: $STACK_NAME"
    log "Log File: $LOG_FILE"
    log "=========================================="

    # Pre-flight checks
    check_aws_cli
    check_aws_credentials

    # Verify stack exists
    if ! check_stack_exists "$STACK_NAME"; then
        log_warning "Stack '$STACK_NAME' does not exist. Nothing to clean up."
        exit 0
    fi

    # Show what will be deleted
    show_stack_resources "$STACK_NAME"

    # Confirmation prompt
    echo ""
    log_warning "This will permanently delete the '$STACK_NAME' stack and ALL its resources."
    read -p "Are you sure you want to proceed? (yes/no): " confirm
    echo ""

    if [ "$confirm" != "yes" ]; then
        log "Cleanup cancelled."
        exit 0
    fi

    # Collect Lambda function names before deleting the stack (needed for log group cleanup)
    log "Collecting Lambda function names for post-deletion log cleanup..."
    LAMBDA_FUNCTIONS_TO_CLEAN=""
    root_lambdas=$(aws cloudformation list-stack-resources \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'StackResourceSummaries[?ResourceType==`AWS::Lambda::Function`].PhysicalResourceId' \
        --output text 2>/dev/null || true)
    LAMBDA_FUNCTIONS_TO_CLEAN="$root_lambdas"

    nested_stacks=$(aws cloudformation list-stack-resources \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'StackResourceSummaries[?ResourceType==`AWS::CloudFormation::Stack`].PhysicalResourceId' \
        --output text 2>/dev/null || true)
    for ns in $nested_stacks; do
        nested_lambdas=$(aws cloudformation list-stack-resources \
            --stack-name "$ns" \
            --region "$REGION" \
            --query 'StackResourceSummaries[?ResourceType==`AWS::Lambda::Function`].PhysicalResourceId' \
            --output text 2>/dev/null || true)
        LAMBDA_FUNCTIONS_TO_CLEAN="$LAMBDA_FUNCTIONS_TO_CLEAN $nested_lambdas"
    done

    # Empty S3 buckets (required before stack deletion)
    empty_s3_buckets "$STACK_NAME"
    empty_nested_s3_buckets "$STACK_NAME"

    # Delete VPC endpoints (can block VPC deletion if left behind)
    delete_vpc_endpoints "$STACK_NAME"

    # Delete the CloudFormation stack
    log "=========================================="
    log "Deleting stack '$STACK_NAME'..."
    log "=========================================="

    if aws cloudformation delete-stack \
        --stack-name "$STACK_NAME" \
        --region "$REGION" 2>&1 | tee -a "$LOG_FILE"; then

        if wait_for_delete "$STACK_NAME"; then
            # Clean up CloudWatch log groups (auto-created, not managed by CloudFormation)
            delete_cloudwatch_log_groups_post "$LAMBDA_FUNCTIONS_TO_CLEAN"

            # Clean up uploaded templates from S3
            delete_uploaded_templates

            log_success "=========================================="
            log_success "Cleanup completed successfully!"
            log_success "=========================================="
            exit 0
        else
            log_error "Stack deletion encountered issues. Check the AWS Console for details:"
            log "https://$REGION.console.aws.amazon.com/cloudformation/home?region=$REGION#/stacks"
            exit 1
        fi
    else
        log_error "Failed to initiate stack deletion"
        exit 1
    fi
}

# Run
main "$@"

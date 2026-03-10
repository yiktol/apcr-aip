#!/bin/bash

################################################################################
# CloudFormation Template Validation Script
# Description: Validates all CloudFormation templates in the deploy folder
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_LOG="validation-$(date +%Y%m%d-%H%M%S).log"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$VALIDATION_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1" | tee -a "$VALIDATION_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1" | tee -a "$VALIDATION_LOG"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1" | tee -a "$VALIDATION_LOG"
}

validate_template() {
    local template_file=$1
    local template_name=$(basename "$template_file")
    
    log "Validating: $template_name"
    
    if aws cloudformation validate-template \
        --template-body "file://$template_file" \
        --output json > /tmp/validation_output.json 2>&1; then
        
        log_success "$template_name is valid"
        
        # Extract and display key information
        local description=$(jq -r '.Description // "No description"' /tmp/validation_output.json)
        local param_count=$(jq '.Parameters | length' /tmp/validation_output.json)
        local capabilities=$(jq -r '.Capabilities[]? // empty' /tmp/validation_output.json | tr '\n' ', ' | sed 's/,$//')
        
        echo "  Description: $description" | tee -a "$VALIDATION_LOG"
        echo "  Parameters: $param_count" | tee -a "$VALIDATION_LOG"
        
        if [ -n "$capabilities" ]; then
            echo "  Capabilities Required: $capabilities" | tee -a "$VALIDATION_LOG"
        fi
        
        # List parameters
        if [ "$param_count" -gt 0 ]; then
            echo "  Parameter Details:" | tee -a "$VALIDATION_LOG"
            jq -r '.Parameters[] | "    - \(.ParameterKey): \(.Description // "No description")"' /tmp/validation_output.json | tee -a "$VALIDATION_LOG"
        fi
        
        echo "" | tee -a "$VALIDATION_LOG"
        return 0
    else
        log_error "$template_name validation failed"
        cat /tmp/validation_output.json | tee -a "$VALIDATION_LOG"
        echo "" | tee -a "$VALIDATION_LOG"
        return 1
    fi
}

check_yaml_syntax() {
    local template_file=$1
    local template_name=$(basename "$template_file")
    
    # Basic YAML syntax check using Python
    if python3 -c "import yaml; yaml.safe_load(open('$template_file'))" 2>/dev/null; then
        return 0
    else
        log_error "$template_name has YAML syntax errors"
        python3 -c "import yaml; yaml.safe_load(open('$template_file'))" 2>&1 | tee -a "$VALIDATION_LOG"
        return 1
    fi
}

main() {
    log "=========================================="
    log "CloudFormation Template Validation"
    log "=========================================="
    log "Directory: $SCRIPT_DIR"
    log "Log File: $VALIDATION_LOG"
    log "=========================================="
    echo ""
    
    # Check if AWS CLI is available
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check if Python is available for YAML syntax check
    if ! command -v python3 &> /dev/null; then
        log_warning "Python3 not found, skipping YAML syntax checks"
        SKIP_YAML_CHECK=true
    fi
    
    # Find all YAML templates
    local templates=()
    while IFS= read -r -d '' file; do
        templates+=("$file")
    done < <(find "$SCRIPT_DIR" -maxdepth 1 -name "*.yaml" -print0 | sort -z)
    
    if [ ${#templates[@]} -eq 0 ]; then
        log_warning "No YAML templates found in $SCRIPT_DIR"
        exit 0
    fi
    
    log "Found ${#templates[@]} template(s) to validate"
    echo ""
    
    local valid_count=0
    local invalid_count=0
    local failed_templates=()
    
    # Validate each template
    for template in "${templates[@]}"; do
        # YAML syntax check
        if [ "$SKIP_YAML_CHECK" != "true" ]; then
            if ! check_yaml_syntax "$template"; then
                ((invalid_count++))
                failed_templates+=("$(basename "$template")")
                continue
            fi
        fi
        
        # CloudFormation validation
        if validate_template "$template"; then
            ((valid_count++))
        else
            ((invalid_count++))
            failed_templates+=("$(basename "$template")")
        fi
    done
    
    # Summary
    log "=========================================="
    log "Validation Summary"
    log "=========================================="
    log_success "Valid templates: $valid_count"
    
    if [ $invalid_count -gt 0 ]; then
        log_error "Invalid templates: $invalid_count"
        log "Failed templates:"
        for template in "${failed_templates[@]}"; do
            log_error "  - $template"
        done
        log "=========================================="
        log_error "Validation completed with errors"
        exit 1
    else
        log "=========================================="
        log_success "All templates validated successfully!"
        exit 0
    fi
}

main "$@"

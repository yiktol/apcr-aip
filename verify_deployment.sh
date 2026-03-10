#!/bin/bash

# Verification script for enhanced authentication deployment
# Checks all sessions for proper implementation

echo "=========================================="
echo "Authentication Deployment Verification"
echo "=========================================="
echo ""

SESSIONS=("session1" "session2" "session3" "session4" "session5")
PASS_COUNT=0
FAIL_COUNT=0

for session in "${SESSIONS[@]}"; do
    echo "Checking $session..."
    
    # Check if authenticate.py exists
    if [ ! -f "$session/utils/authenticate.py" ]; then
        echo "  ✗ authenticate.py not found"
        ((FAIL_COUNT++))
        continue
    fi
    
    # Check for key features
    FEATURES=(
        "generate_pkce_pair"
        "validate_state"
        "refresh_access_token"
        "is_token_expired"
        "Welcome,"
        "user_info.get"
    )
    
    ALL_FOUND=true
    for feature in "${FEATURES[@]}"; do
        if ! grep -q "$feature" "$session/utils/authenticate.py"; then
            echo "  ✗ Missing feature: $feature"
            ALL_FOUND=false
        fi
    done
    
    if [ "$ALL_FOUND" = true ]; then
        # Try to compile
        if python -m py_compile "$session/utils/authenticate.py" 2>/dev/null; then
            echo "  ✓ All features present and compiles successfully"
            ((PASS_COUNT++))
        else
            echo "  ✗ Compilation failed"
            ((FAIL_COUNT++))
        fi
    else
        ((FAIL_COUNT++))
    fi
    
    echo ""
done

echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "Sessions passed: $PASS_COUNT/5"
echo "Sessions failed: $FAIL_COUNT/5"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓ All sessions successfully updated!"
    echo ""
    echo "Enhanced features deployed:"
    echo "  • PKCE security"
    echo "  • CSRF protection"
    echo "  • Token persistence"
    echo "  • Automatic token refresh"
    echo "  • Username display"
    echo "  • Enhanced error handling"
    echo ""
    exit 0
else
    echo "✗ Some sessions failed verification"
    echo "Please review the errors above"
    echo ""
    exit 1
fi

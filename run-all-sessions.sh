#!/bin/bash

# Run All Sessions Script
# Runs sessions 1-5 sequentially, monitoring each for successful startup

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SESSIONS=(1 2 3 4 5)
BASE_PORT=8081
STARTUP_TIMEOUT=30
CHECK_INTERVAL=2

# Log function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Wait for Streamlit to start and get its PID
wait_for_streamlit() {
    local port=$1
    local session=$2
    local elapsed=0
    
    log "Waiting for session $session to start on port $port..."
    
    while [ $elapsed -lt $STARTUP_TIMEOUT ]; do
        if check_port $port; then
            # Additional check: try to curl the health endpoint
            if curl -s http://localhost:$port/_stcore/health >/dev/null 2>&1; then
                # Get the actual Streamlit PID from the session's pid file
                if [ -f "session$session/pid" ]; then
                    local streamlit_pid=$(cat "session$session/pid")
                    log_success "Session $session is running on port $port (Streamlit PID: $streamlit_pid)"
                else
                    log_success "Session $session is running on port $port"
                fi
                return 0
            fi
        fi
        
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
        echo -n "."
    done
    
    echo ""
    log_error "Timeout waiting for session $session to start"
    return 1
}

# Check if session directory exists
check_session_dir() {
    local session=$1
    if [ ! -d "session$session" ]; then
        log_error "Session $session directory not found"
        return 1
    fi
    return 0
}

# Setup and run a session
run_session() {
    local session=$1
    local port=$((BASE_PORT + session - 1))
    
    log "=========================================="
    log "Starting Session $session"
    log "=========================================="
    
    # Check if session directory exists
    if ! check_session_dir $session; then
        return 1
    fi
    
    # Navigate to session directory
    cd "session$session"
    
    # Check if port is already in use
    if check_port $port; then
        log_warning "Port $port is already in use. Skipping session $session setup."
        cd ..
        return 0
    fi
    
    # Check if setup.sh exists
    if [ -f "setup.sh" ]; then
        log "Running setup.sh for session $session..."
        chmod +x setup.sh
        ./setup.sh --port $port
        
        # Wait for Streamlit to start
        if wait_for_streamlit $port $session; then
            log_success "Session $session started successfully"
        else
            log_error "Failed to start session $session"
            cd ..
            return 1
        fi
    else
        log_error "setup.sh not found in session$session"
        cd ..
        return 1
    fi
    
    cd ..
    return 0
}

# Cleanup function
cleanup() {
    if [ "$1" = "EXIT" ]; then
        # Normal exit, don't kill processes
        return 0
    fi
    
    log_warning "Interrupt received. Cleaning up..."
    for session in "${SESSIONS[@]}"; do
        if [ -f "session$session/pid" ]; then
            local pid=$(cat "session$session/pid")
            if ps -p $pid > /dev/null 2>&1; then
                log "Stopping session $session (PID: $pid)..."
                kill $pid 2>/dev/null || true
                sleep 1
                # Force kill if still running
                if ps -p $pid > /dev/null 2>&1; then
                    kill -9 $pid 2>/dev/null || true
                fi
            fi
        fi
    done
}

# Trap cleanup on interrupt, but not on normal exit
trap 'cleanup INT' INT TERM

# Main execution
main() {
    log "=========================================="
    log "AWS AI Practitioner - Run All Sessions"
    log "=========================================="
    log ""
    
    # Check if we're in the right directory
    if [ ! -d "session1" ]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run each session
    for session in "${SESSIONS[@]}"; do
        if ! run_session $session; then
            log_error "Failed to start session $session. Stopping."
            exit 1
        fi
        log ""
        sleep 2
    done
    
    log_success "=========================================="
    log_success "All sessions started successfully!"
    log_success "=========================================="
    log ""
    log "Session URLs:"
    for session in "${SESSIONS[@]}"; do
        local port=$((BASE_PORT + session - 1))
        log "  Session $session: http://localhost:$port"
    done
    log ""
    log_success "All sessions are now running in the background."
    log "To stop all sessions, run: pkill -f streamlit"
}

# Run main function
main

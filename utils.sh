#!/bin/bash
# utils.sh — shared shell utilities for ML-Inference-Server-Template

# --- Colours ---
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'; DIM='\033[2m'

info()    { echo -e "${GREEN}[run]${NC} $1"; }
warning() { echo -e "${YELLOW}[warning]${NC} $1"; }
error()   { echo -e "${RED}[error]${NC} $1"; exit 1; }
section() { echo -e "\n${BOLD}${CYAN}=== $1 ===${NC}\n"; }

get_config() {
    python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print($1)"
}

wait_for_server() {
    local port="$1" pid="$2" timeout="$3" logfile="$4"
    for i in $(seq 1 "$timeout"); do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            info "Server is up."
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            error "Server process died before becoming healthy. Check $logfile."
        fi
        sleep 1
        if [ "$i" -eq "$timeout" ]; then
            error "Server failed to respond after ${timeout}s. Check $logfile."
        fi
    done
}

stop_server() {
    local pid="$1" port="$2"
    if kill -0 "$pid" 2>/dev/null; then
        info "Shutting down server (PID $pid)..."
        kill "$pid"
        wait "$pid" 2>/dev/null || true
    fi
    # Force-release port if anything is still bound to it
    local lingering
    lingering=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$lingering" ]; then
        info "Force-killing process(es) still on port $port: $lingering"
        echo "$lingering" | xargs kill -9 2>/dev/null || true
    fi
    # Wait for the OS to release the port (up to 10 s)
    for i in $(seq 1 10); do
        lsof -ti :"$port" > /dev/null 2>&1 || { info "Port $port is free."; return 0; }
        sleep 1
    done
    warning "Port $port may still be in use after cleanup."
}

install_requirements_pretty() {
    local REQUIREMENTS="$1"
    info "Installing/updating dependencies (may take a few minutes)..."
    pip install --upgrade pip --quiet
    TOTAL=$(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "$REQUIREMENTS" || true)
    CURRENT=0
    PIP_ERR=$(mktemp)
    while IFS= read -r package || [ -n "$package" ]; do
        [[ -z "$package" || "$package" =~ ^# ]] && continue
        CURRENT=$((CURRENT + 1))
        printf "\n  [%d/%d] Checking %s...\n" "$CURRENT" "$TOTAL" "$package"
        [[ "$package" =~ vllm ]] && \
            printf "    (vllm also installs torch — this may take a while)\n"
        if ! pip install --upgrade "$package" 2>"$PIP_ERR" | \
             grep -E '^(Downloading|Collecting|Installing|Using cached)' | \
             while IFS= read -r line; do printf "\r    %-70.70s" "$line"; done; then
            printf "\n"
            warning "Failed to install: $package"
            cat "$PIP_ERR" >&2
            rm -f "$PIP_ERR"; exit 1
        fi
        printf "\r    %-70.70s\n" "Done."
    done < "$REQUIREMENTS"
    rm -f "$PIP_ERR"
    info "All packages installed."
}

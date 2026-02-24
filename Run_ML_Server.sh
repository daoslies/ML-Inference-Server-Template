#!/bin/bash
set -e

# -------------------------------------------------------
# Run_ML_Server.sh — single entrypoint for ml_server_template
#
# Usage:
#   bash Run_ML_Server.sh           # setup (if needed) → test → serve
#   bash Run_ML_Server.sh --skip-tests  # setup (if needed) → serve directly
#
# On first run: creates venv, installs deps, runs tests, starts server.
# On subsequent runs: activates existing venv, runs tests, starts server.
#
# To force a clean reinstall:
#   rm -rf ~/venvs/ml_server_template && bash Run_ML_Server.sh
# -------------------------------------------------------

VENV_DIR="$HOME/venvs/ml_server_template"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10
SKIP_TESTS=false

# Parse flags
for arg in "$@"; do
    [[ "$arg" == "--skip-tests" ]] && SKIP_TESTS=true
done

# --- Colours ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${GREEN}[run]${NC} $1"; }
warning() { echo -e "${YELLOW}[warning]${NC} $1"; }
error()   { echo -e "${RED}[error]${NC} $1"; exit 1; }
section() { echo -e "\n${BOLD}${CYAN}=== $1 ===${NC}\n"; }

# --- Install from requirements.txt with progress counter --- ## a little extra, but it's neater.

REQUIREMENTS="$(dirname "$0")/requirements.txt"

install_package() {
    local package="$1"
    local extra=""

    if [[ "$package" =~ vllm ]]; then
        extra=" (vllm is also installing torch. torch is a big boi, may take a mo)"
    fi

    pip install "$package" 2>"$PIP_ERR" | while IFS= read -r line; do
        if [[ "$line" =~ ^(Downloading|Collecting|Installing|Using\ cached) ]]; then
            if [ -n "$extra" ]; then
                printf "\n    %-70.70s\n" "$extra"
                extra=""
            fi
            printf "\r    %-70.70s" "$line"
        fi
    done

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        printf "\n"
        warning "Failed to install: $package"
        warning "Error output:"
        cat "$PIP_ERR" >&2
        rm -f "$PIP_ERR"
        exit 1
    fi
}

install_requirements() {
    [ -f "$REQUIREMENTS" ] || error "requirements.txt not found at $REQUIREMENTS"

    TOTAL=$(grep -vc '^\s*#\|^\s*$' "$REQUIREMENTS" || true)
    CURRENT=0
    PIP_ERR=$(mktemp)

    while IFS= read -r package || [ -n "$package" ]; do
        [ -z "$package" ] && continue
        [[ "$package" =~ ^# ]] && continue

        CURRENT=$((CURRENT + 1))
        printf "\n  [%d/%d] Installing %s...\n" "$CURRENT" "$TOTAL" "$package"
        install_package "$package"
        printf "\r    %-70.70s\n" "Done."
    done < "$REQUIREMENTS"

    rm -f "$PIP_ERR"
    echo ""
    info "All packages installed."
}

# --- Find Python ---
PYTHON=$(command -v python3 || command -v python || true)
[ -z "$PYTHON" ] && error "Python not found. Please install Python 3.10+."

VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MAJOR=$(echo $VERSION | cut -d. -f1)
MINOR=$(echo $VERSION | cut -d. -f2)

if [ "$MAJOR" -lt "$PYTHON_MIN_MAJOR" ] || ([ "$MAJOR" -eq "$PYTHON_MIN_MAJOR" ] && [ "$MINOR" -lt "$PYTHON_MIN_MINOR" ]); then
    error "Python 3.10+ required, found $VERSION"
fi

SENTINEL="$VENV_DIR/.install_complete"

# -------------------------------------------------------
# PHASE 1: Setup
# -------------------------------------------------------

section "SETUP"

if [ ! -f "$SENTINEL" ]; then
    [ -d "$VENV_DIR" ] && { warning "Incomplete install detected — cleaning up..."; rm -rf "$VENV_DIR"; }
    echo ""
    echo -e "  ${YELLOW}First run detected.${NC} This will:"
    echo -e "  • Create a Python venv at ${GREEN}$VENV_DIR${NC}"
    echo -e "  • Install: flask, pyyaml, torch, vllm"
    echo ""
    read -r -p "  Proceed? [y/N] " confirm
    echo ""
    [[ "$confirm" =~ ^([Yy]([Ee][Ss])?)$ ]] || { info "Aborted. Nothing was installed."; exit 0; }
    info "Setting up environment..."

    SETUP_START=$SECONDS

    if command -v nvidia-smi &> /dev/null; then
        info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        warning "No NVIDIA GPU detected — inference will be slow."
    fi

    info "Creating venv at $VENV_DIR..."
    mkdir -p "$(dirname $VENV_DIR)"
    $PYTHON -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    info "Installing dependencies (one-time, may take a few minutes)..."
    pip install --upgrade pip --quiet
    install_requirements

    # Disable Ray telemetry ## there's a subcomponent of vllm that calls home with usage stats unless you tell it not to.
    info "Disabling vllm telemetry..."
    ray disable-usage-stats 2>/dev/null || true

    # Mark install complete
    touch "$SENTINEL"
    SETUP_ELAPSED=$((SECONDS-SETUP_START))
    info "Setup complete in ${SETUP_ELAPSED}s."
else
    source "$VENV_DIR/bin/activate"
    info "Environment ready."
fi

# -------------------------------------------------------
# PHASE 2: Test (unless --skip-tests)
# -------------------------------------------------------

TEST_SERVER_PID=""

cleanup_test_server() {
    if [ -n "$PORT" ]; then
        PIDS=$(lsof -ti :$PORT)
        if [ -n "$PIDS" ]; then
            info "Shutting down server(s) on port $PORT (PID(s): $PIDS)..."
            kill $PIDS
            wait $PIDS 2>/dev/null || true
        fi
    elif [ -n "$TEST_SERVER_PID" ] && kill -0 "$TEST_SERVER_PID" 2>/dev/null; then
        info "Shutting down background server (PID $TEST_SERVER_PID)..."
        kill "$TEST_SERVER_PID"
        wait "$TEST_SERVER_PID" 2>/dev/null || true
    fi
}

if [ "$SKIP_TESTS" = false ]; then
    trap cleanup_test_server EXIT
    section "TESTS"

    PORT=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['server'].get('port', 27776))")

    info "Starting server in background for tests (log → server_test.log)..."
    python server.py > server_test.log 2>&1 &
    TEST_SERVER_PID=$!

    info "Waiting for server on port $PORT..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            info "Server is up."
            break
        fi
        if ! kill -0 "$TEST_SERVER_PID" 2>/dev/null; then
            error "Server process died before becoming healthy. Check server_test.log."
        fi
        sleep 1
        [ "$i" -eq 30 ] && error "Server failed to respond after 30s. Check server_test.log."
    done

    info "Running test suite..."
    echo ""
    pytest tests.py -v
    TEST_EXIT=$?
    echo ""

    if [ "$TEST_EXIT" -eq 0 ]; then
        # Output introductory text from the model
        TEST_MODEL=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('tests', c.get('defaults', {})).get('test_model', 'Qwen/Qwen3-0.6B'))")
        INTRO_PROMPT="Say hello, world!"
        info "Loading test model ($TEST_MODEL) for introductory output..."
        curl -sf -X POST "http://localhost:$PORT/load_model" \
            -H "Content-Type: application/json" \
            -d '{"model_path":"'$TEST_MODEL'"}' > /dev/null || info "Model already loaded or failed to load."
        info "Requesting introductory output from the model..."
        sleep 1  # Ensure server is ready
        INTRO_OUTPUT=$(curl -sv -X POST "http://localhost:$PORT/inference" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"'$INTRO_PROMPT'","max_tokens":10,"temperature":0.0}' 2>&1)
        CURL_EXIT=$?
        echo "[debug] CURL EXIT CODE: $CURL_EXIT"
        echo "[debug] INTRO_OUTPUT (raw): $INTRO_OUTPUT"
        if [ $CURL_EXIT -ne 0 ]; then
            warning "curl failed with exit code $CURL_EXIT. See above for details."
        fi
        if [ -n "$INTRO_OUTPUT" ]; then
            RESPONSE=$(echo "$INTRO_OUTPUT" | python -c "import sys, json; print(json.load(sys.stdin).get('results',{}).get('response',''))" 2>/dev/null)
            echo "[debug] RESPONSE: $RESPONSE"
            if [ -n "$RESPONSE" ]; then
                echo -e "\n${CYAN}${BOLD}Model introductory output:${NC} $RESPONSE\n"
            else
                warning "Could not parse model response. Raw output: $INTRO_OUTPUT"
            fi
        else
            warning "Could not get introductory output from the model."
        fi
    fi

    info "Shutting down test server..."
    kill "$TEST_SERVER_PID"
    wait "$TEST_SERVER_PID" 2>/dev/null || true
    TEST_SERVER_PID=""   # Clear so trap doesn't double-kill

    trap - EXIT

    if [ "$TEST_EXIT" -ne 0 ]; then
        echo -e "${RED}${BOLD}Tests failed. Server will not start.${NC}"
        echo -e "  Server logs from the test run are in ${YELLOW}server_test.log${NC}"
        exit "$TEST_EXIT"
    fi

    echo -e "${GREEN}${BOLD}All tests passed.${NC}"
fi

# -------------------------------------------------------
# PHASE 3: Serve
# -------------------------------------------------------

section "SERVER"
info "Starting server..."
python server.py
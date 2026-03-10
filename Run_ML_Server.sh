#!/bin/bash
set -e

# -------------------------------------------------------
# Run_ML_Server.sh — single entrypoint for ml_server_template
#
# Usage:
#   bash Run_ML_Server.sh            # setup (if needed) → test → serve
#   bash Run_ML_Server.sh --skip-tests  # setup (if needed) → serve directly
#
# On first run: creates venv, installs deps, runs tests, starts server.
# On subsequent runs: activates existing venv, runs tests, starts server.
#
# To force a clean reinstall:
#   rm -rf ~/venvs/ml_server_template && bash Run_ML_Server.sh
# -------------------------------------------------------

VENV_DIR="$HOME/venvs/ml_server_template"
REQUIREMENTS="$(dirname "$0")/requirements.txt"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10
SKIP_TESTS=false

for arg in "$@"; do
    [[ "$arg" == "--skip-tests" ]] && SKIP_TESTS=true
done

# --- Colours ---
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${GREEN}[run]${NC} $1"; }
warning() { echo -e "${YELLOW}[warning]${NC} $1"; }
error()   { echo -e "${RED}[error]${NC} $1"; exit 1; }
section() { echo -e "\n${BOLD}${CYAN}=== $1 ===${NC}\n"; }

# --- Read config values once, reused throughout ---
get_config() {
    python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print($1)"
}

# --- Wait for the server on a given port, with timeout ---
# Usage: wait_for_server <port> <pid> <timeout_secs> <log_file>
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

# --- Stop a server and release its port ---
# Usage: stop_server <pid> <port>
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

# -------------------------------------------------------
# PHASE 1: Setup
# -------------------------------------------------------
section "SETUP"

PYTHON=$(command -v python3 || command -v python || true)
[ -z "$PYTHON" ] && error "Python not found. Please install Python 3.10+."

read -r MAJOR MINOR < <($PYTHON -c "import sys; print(sys.version_info.major, sys.version_info.minor)")
if [ "$MAJOR" -lt "$PYTHON_MIN_MAJOR" ] || \
   ([ "$MAJOR" -eq "$PYTHON_MIN_MAJOR" ] && [ "$MINOR" -lt "$PYTHON_MIN_MINOR" ]); then
    error "Python 3.10+ required, found ${MAJOR}.${MINOR}"
fi

[ -f "$REQUIREMENTS" ] || error "requirements.txt not found at $REQUIREMENTS"

SENTINEL="$VENV_DIR/.install_complete"

if [ ! -f "$SENTINEL" ]; then
    [ -d "$VENV_DIR" ] && { warning "Incomplete install detected — cleaning up..."; rm -rf "$VENV_DIR"; }

    echo -e "\n  ${YELLOW}First run detected.${NC} This will:"
    echo -e "  • Create a Python venv at ${GREEN}$VENV_DIR${NC}"
    echo -e "  • Install all packages from requirements.txt\n"
    read -r -p "  Proceed? [y/N] " confirm; echo ""
    [[ "$confirm" =~ ^([Yy]([Ee][Ss])?)$ ]] || { info "Aborted. Nothing was installed."; exit 0; }

    if command -v nvidia-smi &> /dev/null; then
        info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        warning "No NVIDIA GPU detected — inference will be slow."
    fi

    info "Creating venv at $VENV_DIR..."
    mkdir -p "$(dirname "$VENV_DIR")"
    $PYTHON -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    info "Installing dependencies (one-time, may take a few minutes)..."
    pip install --upgrade pip --quiet

    TOTAL=$(grep -vc '^\s*#\|^\s*$' "$REQUIREMENTS" || true)
    CURRENT=0
    PIP_ERR=$(mktemp)

    while IFS= read -r package || [ -n "$package" ]; do
        [[ -z "$package" || "$package" =~ ^# ]] && continue
        CURRENT=$((CURRENT + 1))
        printf "\n  [%d/%d] Installing %s...\n" "$CURRENT" "$TOTAL" "$package"
        [[ "$package" =~ vllm ]] && \
            printf "    (vllm also installs torch — this may take a while)\n"

        if ! pip install "$package" 2>"$PIP_ERR" | \
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

    info "Disabling vllm/Ray telemetry..."
    ray disable-usage-stats 2>/dev/null || true

    touch "$SENTINEL"
    info "Setup complete."
else
    source "$VENV_DIR/bin/activate"
    info "Environment ready. Checking for updated requirements..."
    pip install --upgrade --requirement "$REQUIREMENTS" --quiet
    info "Requirements are up to date."
fi

# -------------------------------------------------------
# PHASE 2: Tests + introductory output
# -------------------------------------------------------
PORT=$(get_config "c['server'].get('port', 27776)")
TEST_MODEL=$(get_config "c.get('tests', c.get('defaults', {})).get('test_model', 'Qwen/Qwen3-0.6B')")
INTRO_PROMPT="Hello there, a pleasure to make your acquaintance!"

# Helper: fire the intro inference request against the already-running server.
run_intro_output() {
    section "INTRODUCTORY OUTPUT"
    info "Requesting introductory output..."
    local response
    response=$(curl -sf --max-time 30 -X POST "http://localhost:$PORT/inference" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":\"$INTRO_PROMPT\",\"max_tokens\":10,\"temperature\":0.0}" 2>/dev/null \
        | python -c "import sys,json; print(json.load(sys.stdin).get('results',{}).get('response',''))" \
        2>/dev/null || true)
    if [ -n "$response" ]; then
        echo -e "\n${CYAN}${BOLD}Model introductory output:${NC} $response\n"
    else
        warning "Could not get introductory output — server will still start."
    fi
}

if [ "$SKIP_TESTS" = false ]; then
    section "TESTS"

    info "Starting server in background for tests (log → server_test.log)..."
    python server.py > server_test.log 2>&1 &
    TEST_SERVER_PID=$!

    trap 'stop_server "$TEST_SERVER_PID" "$PORT"' EXIT

    wait_for_server "$PORT" "$TEST_SERVER_PID" 30 "server_test.log"

    info "Running test suite..."
    echo ""
    pytest tests.py -v
    TEST_EXIT=$?
    echo ""

    if [ "$TEST_EXIT" -ne 0 ]; then
        echo -e "${RED}${BOLD}Tests failed. Server will not start.${NC}"
        echo -e "  Server logs from the test run are in ${YELLOW}server_test.log${NC}"
        exit "$TEST_EXIT"
    fi
    echo -e "${GREEN}${BOLD}All tests passed.${NC}"

    # Reuse the live test server — model is already loaded, no need to spin up another.
    run_intro_output

    trap - EXIT
    stop_server "$TEST_SERVER_PID" "$PORT"
else
    # No test server to piggyback on; spin up a minimal one just for the intro output.
    section "INTRODUCTORY OUTPUT"

    info "Starting server in background (log → server_intro.log)..."
    python server.py > server_intro.log 2>&1 &
    INTRO_SERVER_PID=$!

    trap 'stop_server "$INTRO_SERVER_PID" "$PORT"' EXIT

    wait_for_server "$PORT" "$INTRO_SERVER_PID" 30 "server_intro.log"

    info "Loading test model: $TEST_MODEL ..."
    curl -sf -X POST "http://localhost:$PORT/load_model" \
        -H "Content-Type: application/json" \
        -d "{\"model_path\":\"$TEST_MODEL\"}" > /dev/null \
        || warning "load_model request failed — model may already be loaded."

    info "Waiting for model to finish loading (up to 60 s)..."
    for i in $(seq 1 60); do
        MODEL_PATH=$(curl -sf "http://localhost:$PORT/health" \
            | python -c "import sys,json; print(json.load(sys.stdin).get('model_path',''))" 2>/dev/null || true)
        if [ -n "$MODEL_PATH" ] && [ "$MODEL_PATH" != "None" ]; then
            info "Model loaded: $MODEL_PATH"; break
        fi
        if ! kill -0 "$INTRO_SERVER_PID" 2>/dev/null; then
            warning "Intro server died before model finished loading. Check server_intro.log."; break
        fi
        sleep 1
        [ "$i" -eq 60 ] && warning "Model failed to load after 60 s. Check server_intro.log." && break
    done

    run_intro_output

    trap - EXIT
    stop_server "$INTRO_SERVER_PID" "$PORT"
fi

# -------------------------------------------------------
# PHASE 3: Serve (via tmux)
# -------------------------------------------------------
section "SERVER"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ML_PY="$PROJECT_DIR/ml.py"

# --- Ensure tmux is available ---
if ! command -v tmux &> /dev/null; then
    warning "tmux not found — attempting to install..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y tmux || error "Failed to install tmux via apt. Please install it manually."
    elif command -v brew &> /dev/null; then
        brew install tmux || error "Failed to install tmux via brew. Please install it manually."
    else
        error "Could not install tmux automatically. Please install it and re-run."
    fi
    info "tmux installed."
fi

# --- Ensure ml.py exists ---
[ -f "$ML_PY" ] || error "ml.py not found at $ML_PY — please add it to the project directory."

TMUX_SESSION="ml_server"

# Kill any leftover session with the same name
( tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true )

# Write a minimal per-pane rc file so 'ml' works without touching .bashrc.
# It activates the venv and defines ml as a function scoped to this session.
ML_RC=$(mktemp /tmp/ml_rc.XXXXXX)
cat > "$ML_RC" << EOF
# Auto-sourced by ml_server tmux session — not a permanent change to your shell.
source "$VENV_DIR/bin/activate"
ml() { python "$ML_PY" "\$@"; }
export -f ml

# Welcome message
echo ""
echo -e "\033[1m\033[0;36m  ML Inference Server — interactive shell\033[0m"
echo ""
echo -e "  \033[1mCommands:\033[0m"
echo -e "    ml health"
echo -e "    ml load <model_path>"
echo -e "    ml infer \"<prompt>\"  [--max-tokens N]  [--temperature F]"
echo -e "    ml batch \"<p1>\" \"<p2>\" ...  [--max-tokens N]"
echo -e "    ml extract \"<text>\" <prompts_path>"
echo -e "    ml shell   \033[2m# interactive REPL\033[0m"
echo ""
EOF

# Build the session:
#   Pane 0 (left)  — server logs
#   Pane 1 (right) — ml interactive shell
tmux new-session -d -s "$TMUX_SESSION" -x "220" -y "50"

# Left pane: run the server
tmux send-keys -t "$TMUX_SESSION:0.0" \
    "source '$VENV_DIR/bin/activate' && python3 '$PROJECT_DIR/server.py'" Enter

# Split vertically (left/right), start the ml shell in the right pane
tmux split-window -h -t "$TMUX_SESSION:0"
tmux send-keys -t "$TMUX_SESSION:0.1" "source '$ML_RC' && rm -f '$ML_RC'" Enter

# Give the left pane 40% of the width
tmux resize-pane -t "$TMUX_SESSION:0.0" -x "40%"

# Focus the right (ml shell) pane so the user lands there
tmux select-pane -t "$TMUX_SESSION:0.1"

info "Attaching to tmux session '$TMUX_SESSION'..."
info "  Left pane : server logs"
info "  Right pane: ml shell (ready to use)"
echo ""
echo -e "  ${DIM}Tip: Ctrl-B D to detach and leave the server running.${NC}"
echo -e "       Ctrl-B X to kill the session (and server)."
echo -e "       bash Kill_ML_Server.sh to kill from outside tmux."
echo ""

exec tmux attach-session -t "$TMUX_SESSION"
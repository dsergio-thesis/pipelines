
#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# AstroOS / pipelines installer (conda + pyproject.toml)
#
# Features:
# - Run from repo root named "pipelines"
# - Creates/uses a conda env named "<project>-<version>-py311"
# - Default: installs ALL extras ([dev,ml])
# - Flags:
#     --reinstall          Force reinstall of editable project + deps inside env
#     --cuda               Install torch/torchvision from PyTorch CUDA index (Linux only, default: cu121)
#     --cuda=cuXXX         Choose CUDA channel (e.g. cu118, cu121, cu124)
#     --no-extras          Install only base deps (no extras)
#     --extras=LIST        Override extras (comma-separated) e.g. --extras=dev,ml or --extras=dev
# - Writes ASTROOS_ROOT/PATH to first writable rc file:
#     ~/.zshrc, ~/.bashrc, ~/.bash_profile
#   fallback: ~/.astroos_env (prints instructions)
# - Creates CLI router: astroos <cmd> ... dispatches to $ASTROOS_ROOT/bin/<cmd>(.sh)
# - Adds Makefile targets (if Makefile doesn't exist):
#     make clean-env   (removes the conda env)
#     make reinstall   (runs install.sh --reinstall)
# -------------------------------

# ---- must be run from pipelines/ ----
if [ "$(basename "$PWD")" != "pipelines" ]; then
  echo "Error: Please run install script from the 'pipelines' directory."
  exit 1
fi

REPO_ROOT="$PWD"
BIN_DIR="$REPO_ROOT/bin"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Error: missing '$1'"; exit 1; }; }

need_cmd tar

# ---- parse args ----
REINSTALL=0
CUDA_MODE=0
CUDA_CHANNEL="cu121"
NO_EXTRAS=0
EXTRAS_OVERRIDE=""

usage() {
  cat <<EOF
Usage: ./install.sh [options]

Options:
  --reinstall          Force reinstall of editable project + deps in the env
  --cuda               Install torch/torchvision from PyTorch CUDA wheels (Linux only, default channel: cu121)
  --cuda=cuXXX         Specify CUDA wheel channel (e.g. --cuda=cu118, --cuda=cu121, --cuda=cu124)
  --no-extras          Install only base deps (no extras)
  --extras=LIST        Override extras (comma-separated) e.g. --extras=dev,ml or --extras=dev
  -h, --help           Show help

Defaults:
  Installs with all extras: [dev,ml]
EOF
}

for arg in "$@"; do
  case "$arg" in
    --reinstall) REINSTALL=1 ;;
    --cuda) CUDA_MODE=1 ;;
    --cuda=*) CUDA_MODE=1; CUDA_CHANNEL="${arg#*=}" ;;
    --no-extras) NO_EXTRAS=1 ;;
    --extras=*) EXTRAS_OVERRIDE="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Error: unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

# ---- choose conda implementation ----
if command -v mamba >/dev/null 2>&1; then
  CONDA_CMD="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
else
  echo "Error: neither 'conda' nor 'mamba' found on PATH."
  echo "Install Miniconda/Mambaforge first."
  exit 1
fi

# ---- parse project name/version from pyproject.toml ----
if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
  echo "Error: pyproject.toml not found in $REPO_ROOT"
  exit 1
fi

PROJECT_NAME="$(awk '
  $0 ~ /^\[project\]/ {inproj=1; next}
  $0 ~ /^\[/ && $0 !~ /^\[project\]/ {inproj=0}
  inproj && $0 ~ /^name[[:space:]]*=/ {
    gsub(/.*=[[:space:]]*"/,""); gsub(/".*/,""); print; exit
  }' pyproject.toml)"

PROJECT_VERSION="$(awk '
  $0 ~ /^\[project\]/ {inproj=1; next}
  $0 ~ /^\[/ && $0 !~ /^\[project\]/ {inproj=0}
  inproj && $0 ~ /^version[[:space:]]*=/ {
    gsub(/.*=[[:space:]]*"/,""); gsub(/".*/,""); print; exit
  }' pyproject.toml)"

if [ -z "${PROJECT_NAME:-}" ] || [ -z "${PROJECT_VERSION:-}" ]; then
  echo "Error: could not parse [project].name and/or [project].version from pyproject.toml"
  exit 1
fi

PY_VER="3.11"
ENV_NAME="${PROJECT_NAME}-${PROJECT_VERSION}-py311"

echo "Project:     $PROJECT_NAME"
echo "Version:     $PROJECT_VERSION"
echo "Conda env:   $ENV_NAME"
echo "Conda tool:  $CONDA_CMD"
echo "Reinstall:   $REINSTALL"
echo "CUDA mode:   $CUDA_MODE (${CUDA_CHANNEL})"
echo "No extras:   $NO_EXTRAS"
echo "Extras ovrd: ${EXTRAS_OVERRIDE:-<none>}"

# ---- create env if missing ----
if ! $CONDA_CMD env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env '$ENV_NAME' (python=$PY_VER)..."
  $CONDA_CMD create -y -n "$ENV_NAME" "python=$PY_VER" pip
else
  echo "Conda env '$ENV_NAME' already exists."
fi

# ---- decide extras ----
DEFAULT_EXTRAS="dev,ml"

if [ "$NO_EXTRAS" -eq 1 ]; then
  EXTRAS_SPEC=""
elif [ -n "$EXTRAS_OVERRIDE" ]; then
  EXTRAS_SPEC="$EXTRAS_OVERRIDE"
else
  EXTRAS_SPEC="$DEFAULT_EXTRAS"
fi

# Build pip target (IMPORTANT: extras attach directly to path: /path/to/repo[dev,ml])
if [ -n "$EXTRAS_SPEC" ]; then
  PIP_TARGET="${REPO_ROOT}[${EXTRAS_SPEC}]"
else
  PIP_TARGET="$REPO_ROOT"
fi

# ---- upgrade pip ----
echo "Upgrading pip in env..."
$CONDA_CMD run -n "$ENV_NAME" python -m pip install -U pip

# ---- CUDA torch handling (Linux only) ----
# If --cuda is set:
# 1) Install torch/torchvision from CUDA index
# 2) Then install project WITHOUT 'ml' extra to avoid pip swapping CUDA wheels
UNAME_S="$(uname -s || true)"

strip_extra() {
  # strip_extra "dev,ml" "ml"  -> "dev"
  local list="$1"
  local drop="$2"
  echo "$list" | awk -F',' -v drop="$drop" '{
    out=""; for(i=1;i<=NF;i++){ if($i!=drop && $i!=""){ out=(out==""?$i:out","$i) } }
    print out
  }'
}

if [ "$CUDA_MODE" -eq 1 ]; then
  if [ "$UNAME_S" != "Linux" ]; then
    echo "Warning: --cuda requested but OS is '$UNAME_S'. Skipping CUDA torch install."
  else
    echo "Installing torch/torchvision from PyTorch CUDA index ($CUDA_CHANNEL)..."
    CUDA_INDEX_URL="https://download.pytorch.org/whl/${CUDA_CHANNEL}"
    $CONDA_CMD run -n "$ENV_NAME" python -m pip install -U torch torchvision --index-url "$CUDA_INDEX_URL"

    # Remove ml extra from project install if present
    if [ -n "$EXTRAS_SPEC" ]; then
      EXTRAS_STRIPPED="$(strip_extra "$EXTRAS_SPEC" "ml")"
      if [ -n "$EXTRAS_STRIPPED" ]; then
        PIP_TARGET="${REPO_ROOT}[${EXTRAS_STRIPPED}]"
      else
        PIP_TARGET="$REPO_ROOT"
      fi
      echo "CUDA mode: installing project without 'ml' extra (torch already installed): $PIP_TARGET"
    fi
  fi
fi

# ---- install project into env ----
echo "Installing project from pyproject.toml: $PIP_TARGET"

if [ "$REINSTALL" -eq 1 ]; then
  echo "Reinstall mode: force reinstalling editable project + deps..."
  $CONDA_CMD run -n "$ENV_NAME" python -m pip install -e "$PIP_TARGET" --upgrade --force-reinstall
else
  $CONDA_CMD run -n "$ENV_NAME" python -m pip install -e "$PIP_TARGET"
fi

# ---- persist ASTROOS_ROOT and PATH (portable) ----
ASTRO_LINE="export ASTROOS_ROOT=\"$REPO_ROOT\""
PATH_LINE_1='export PATH="$ASTROOS_ROOT/bin:$PATH"'
PATH_LINE_2='export PATH="$HOME/.local/bin:$PATH"'

append_if_missing() {
  local file="$1"
  local line="$2"
  grep -Fq "$line" "$file" 2>/dev/null || echo "$line" >> "$file"
}

try_configure_rc() {
  local file="$1"

  if [ ! -e "$file" ]; then
    touch "$file" 2>/dev/null || return 1
  fi
  [ -w "$file" ] || return 1

  echo "" >> "$file"
  append_if_missing "$file" "# AstroOS"
  append_if_missing "$file" "$ASTRO_LINE"
  append_if_missing "$file" "$PATH_LINE_1"
  append_if_missing "$file" "$PATH_LINE_2"
  return 0
}

RC_CHOSEN=""
for candidate in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do
  if try_configure_rc "$candidate"; then
    RC_CHOSEN="$candidate"
    break
  fi
done

if [ -n "$RC_CHOSEN" ]; then
  echo "Configured AstroOS env vars in: $RC_CHOSEN"
else
  ENV_FALLBACK="$HOME/.astroos_env"
  {
    echo ""
    echo "# AstroOS"
    echo "$ASTRO_LINE"
    echo "$PATH_LINE_1"
    echo "$PATH_LINE_2"
  } >> "$ENV_FALLBACK"
  echo "⚠️ Could not write to ~/.zshrc, ~/.bashrc, or ~/.bash_profile."
  echo "Wrote environment settings to: $ENV_FALLBACK"
  echo "Add this line to a shell startup file you can edit:"
  echo "  source \"$ENV_FALLBACK\""
fi

# also set for current session
export ASTROOS_ROOT="$REPO_ROOT"
export PATH="$ASTROOS_ROOT/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"

# ---- create CLI router script in repo bin/ ----
mkdir -p "$BIN_DIR"

CLI_NAME="astroos"
CLI_PATH="$BIN_DIR/$CLI_NAME"

cat > "$CLI_PATH" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [ -z "${ASTROOS_ROOT:-}" ]; then
  echo "Error: ASTROOS_ROOT is not set. Run the installer or set it in your shell rc."
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: astroos <command> [args...]"
  echo ""
  echo "Available commands:"
  ls -1 "$ASTROOS_ROOT/bin" 2>/dev/null | sed 's/\.sh$//' | grep -vE '^(astroos|install\.sh)$' || true
  exit 1
fi

CMD="$1"; shift
TARGET="$ASTROOS_ROOT/bin/$CMD"
TARGET_SH="$ASTROOS_ROOT/bin/$CMD.sh"

if [ -x "$TARGET" ]; then
  exec "$TARGET" "$@"
elif [ -x "$TARGET_SH" ]; then
  exec "$TARGET_SH" "$@"
else
  echo "Error: unknown command '$CMD'"
  echo "Looked for:"
  echo "  $TARGET"
  echo "  $TARGET_SH"
  exit 1
fi
EOF

chmod +x "$CLI_PATH"

# ---- create a global symlink (no sudo) ----
mkdir -p "$HOME/.local/bin"
ln -sf "$CLI_PATH" "$HOME/.local/bin/$CLI_NAME"

# ---- write a Makefile with clean-env target (if not present) ----
MAKEFILE_PATH="$REPO_ROOT/Makefile"
if [ ! -f "$MAKEFILE_PATH" ]; then
  cat > "$MAKEFILE_PATH" <<EOF
# Auto-generated by install.sh (safe to edit)

ENV_NAME := $ENV_NAME

.PHONY: clean-env reinstall

clean-env:
	@echo "Removing conda env: \$(ENV_NAME)"
	@conda env remove -n "\$(ENV_NAME)" -y || mamba env remove -n "\$(ENV_NAME)" -y

reinstall:
	@./install.sh --reinstall
EOF
  echo "Created Makefile with: make clean-env, make reinstall"
else
  echo "Makefile already exists; not overwriting."
  echo "Add this manually if you want:"
  echo "  make clean-env  (remove conda env: $ENV_NAME)"
fi

echo ""
echo "✅ Install complete."
echo ""
echo "To activate the env:"
echo "  conda activate $ENV_NAME"
echo ""
if [ -n "$RC_CHOSEN" ]; then
  echo "Reload your shell or run:"
  echo "  source \"$RC_CHOSEN\""
else
  echo "Reload your shell after adding the fallback source line."
fi
echo ""
echo "Commands:"
echo "  astroos"
echo "  astroos <command> --help"
echo ""
echo "Make targets:"
echo "  make clean-env"
echo "  make reinstall"

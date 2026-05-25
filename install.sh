
#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# Installer 
# -------------------------------

# ---- must be run from pipelines/ ----
if [ "$(basename "$PWD")" != "pipelines" ]; then
  echo "Error: Please run install script from the 'pipelines' directory."
  exit 1
fi

# if env.sh doesn't exist, stop with error
if [ ! -f "env.sh" ]; then
  echo "Error: env.sh not found. Please run this script from the root of the repo."
  exit 1
fi

source "env.sh"

REPO_ROOT="$PWD"
BIN_DIR="$REPO_ROOT/bin"

usage() {
  cat <<EOF
Usage: ./install.sh [options]

Options:
  -h, --help           Show help

EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    *)
      echo "Error: unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

# ---- persist ASTROOS_ROOT and PATH ----
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

# ---- CLI router script in repo bin/ ----
mkdir -p "$BIN_DIR"

CLI_NAME="rad"
CLI_PATH="$BIN_DIR/$CLI_NAME"

cat > "$CLI_PATH" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [ -z "${ASTROOS_ROOT:-}" ]; then
  echo "Error: ASTROOS_ROOT is not set. Run the installer or set it in your shell rc."
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: rad <command> [args...]"
  echo ""
  echo "Available commands:"
  ls -1 "$ASTROOS_ROOT/scripts" 2>/dev/null | sed 's/\.sh$//' || true
  exit 1
fi

CMD="$1"; shift
TARGET="$ASTROOS_ROOT/scripts/$CMD"
TARGET_SH="$ASTROOS_ROOT/scripts/$CMD.sh"

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

# --- global symlink ----
mkdir -p "$HOME/.local/bin"
ln -sf "$CLI_PATH" "$HOME/.local/bin/$CLI_NAME"

echo ""
echo "✅ Install complete."
if [ -n "$RC_CHOSEN" ]; then
  echo "Reload your shell or run:"
  echo "  source \"$RC_CHOSEN\""
else
  echo "Reload your shell after adding the fallback source line."
fi
echo ""
echo "Commands:"
echo "  rad"
echo "  rad <command> --help"
echo ""

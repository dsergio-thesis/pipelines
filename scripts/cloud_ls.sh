
#!/usr/bin/env bash
set -euo pipefail

echo "Google Drive lsl: "
rclone lsl "gdrive:Thesis" | tail -n 5


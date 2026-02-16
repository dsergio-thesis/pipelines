
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./fetch_from_gdrive.sh <dataset-name>
#
# Example:
#   ./fetch_from_gdrive.sh lsst-1
#
# This will:
#   1. Find the newest matching tarball in gdrive:Thesis
#   2. Download it
#   3. Extract into data/
#   4. Remove the tarball

# ----------------------------
# Argument validation
# ----------------------------
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <dataset-name>"
  exit 1
fi

# Must run from pipelines/
if [[ "$(basename "$PWD")" != "pipelines" ]]; then
  echo "Error: This script must be run from the 'pipelines' directory."
  exit 1
fi

DATASET="$1"
DEST_DIR="data"
mkdir -p "$DEST_DIR"

REMOTE_DIR="gdrive:Thesis"

echo "Searching for latest tarball for dataset '$DATASET' in $REMOTE_DIR ..."

# Find newest matching tarball
LATEST_FILE=$(rclone lsf "$REMOTE_DIR" \
  | grep "^${DATASET}-.*\.tar\.gz$" \
  | sort \
  | tail -n 1)

if [[ -z "$LATEST_FILE" ]]; then
  echo "Error: No tarball found for dataset '$DATASET' in $REMOTE_DIR"
  exit 1
fi

echo "Found: $LATEST_FILE"

LOCAL_TARBALL="$DEST_DIR/$LATEST_FILE"

echo "Downloading -> $LOCAL_TARBALL"

rclone copyto "$REMOTE_DIR/$LATEST_FILE" "$LOCAL_TARBALL" \
  --progress \
  --transfers 1 \
  --checkers 4 \
  --drive-chunk-size 64M \
  --retries 10 \
  --low-level-retries 20

echo "Download complete."
echo "Size:"
ls -lh "$LOCAL_TARBALL"

echo "Extracting into $DEST_DIR ..."

tar -xzf "$LOCAL_TARBALL" -C "$DEST_DIR"

echo "Extraction complete."

# Optional: remove tarball after success
rm -f "$LOCAL_TARBALL"
echo "Removed tarball."

echo "Done."

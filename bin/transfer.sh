
#!/usr/bin/env bash
set -euo pipefail

# read the directory name from the command line argument
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <dataset-dir-name-under-data>"
  exit 1
fi

# must be run from pipelines/
if [[ "$(basename "$PWD")" != "pipelines" ]]; then
  echo "Error: This script must be run from the 'pipelines' directory."
  exit 1
fi

DATASET="$1"
SRC_DIR="data/$DATASET"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Error: Directory '$SRC_DIR' does not exist."
  exit 1
fi

# put tarball somewhere that won't pollute git and is easy to clean
OUT_DIR="${TMPDIR:-/tmp}/pipelines_exports"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
TARBALL="$OUT_DIR/${DATASET}-${TS}.tar.gz"

echo "Compressing '$SRC_DIR' -> '$TARBALL' ..."
# -C keeps paths clean; tar the folder name itself
tar -czf "$TARBALL" -C data "$DATASET"

echo "Created: $TARBALL"
echo "Size:"
ls -lh "$TARBALL"

# upload to Drive (explicit destination filename)
DEST="gdrive:Thesis/${DATASET}-${TS}.tar.gz"
echo "Uploading -> $DEST"

rclone copyto "$TARBALL" "$DEST" \
  --progress \
  --transfers 1 \
  --checkers 4 \
  --drive-chunk-size 64M \
  --retries 10 \
  --low-level-retries 20

echo "Upload done. Verifying listing:"
rclone lsl "gdrive:Thesis" | tail -n 5

# remove local tarball after success
rm -f "$TARBALL"
echo "Done."

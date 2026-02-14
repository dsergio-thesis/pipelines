
#!/usr/bin/env bash
set -euo pipefail

# Ensure ASTROOS_ROOT is set
if [ -z "${ASTROOS_ROOT:-}" ]; then
  echo "Error: ASTROOS_ROOT is not set."
  exit 1
fi

DATA_DIR="${ASTROOS_ROOT}/data"
ARCHIVE="lsst-1.tar.gz"

mkdir -p "$DATA_DIR"

echo "Downloading archive..."
rclone copy "gdrive:Thesis/${ARCHIVE}" "$DATA_DIR" --progress

echo "Extracting..."
tar -xzf "${DATA_DIR}/${ARCHIVE}" -C "$DATA_DIR"

echo "Cleaning up..."
rm "${DATA_DIR}/${ARCHIVE}"

echo "Done."

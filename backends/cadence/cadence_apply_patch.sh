#!/bin/sh

set -e

CADENCE_DIR="backends/cadence"
GFLAGS_DIR="third-party/gflags"
PATCH_NAME="cadence_patch.patch"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../" && pwd)
TARGET_DIR="$PROJECT_ROOT/$GFLAGS_DIR"
PATCH_FILE="$PROJECT_ROOT/$CADENCE_DIR/$PATCH_NAME"

if [ ! -f "$PATCH_FILE" ]; then
    echo "Error: Patch file not found at $PATCH_FILE"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory does not exist at $TARGET_DIR"
    exit 1
fi

cd "$TARGET_DIR"

echo "Applying patch..."

if git apply --reverse --check "$PATCH_FILE" > /dev/null 2>&1; then
    echo "Patch already applied. Skipping."
elif git apply --check "$PATCH_FILE" > /dev/null 2>&1; then
    git apply "$PATCH_FILE"
    echo "Patch applied successfully using git apply."
else
    echo "Error: Patch does not apply cleanly (possible conflicts)."
    exit 1
fi

#!/usr/bin/env bash

SCRIPT_DIR_PATH="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

red=$(tput setaf 1)
green=$(tput setaf 2)
reset=$(tput sgr0)

EXECUTORCH_ROOT_PATH=$(realpath "$SCRIPT_DIR_PATH/../../")
CADENCE_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/cadence"
FUSION_DIR_PATH="$CADENCE_DIR_PATH/fusion_g3/third-party/nnlib/nnlib-FusionG3"
FACTO_DIR_PATH="$CADENCE_DIR_PATH/utils/FACTO"

cd "$EXECUTORCH_ROOT_PATH"

## HiFi

targets=(hifi4 hifi5)

for target in ${targets[@]};
do
    TARGET_DIR_PATH="$CADENCE_DIR_PATH/hifi/third-party/nnlib/nnlib-$target"

    rm -rf "$TARGET_DIR_PATH"
    mkdir -p "$TARGET_DIR_PATH"

    echo "${green}ExecuTorch: Cloning $target nnlib"
    git clone "https://github.com/foss-xtensa/nnlib-$target.git" $TARGET_DIR_PATH
    cd $TARGET_DIR_PATH
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "${red}ExecuTorch: Failed to clone $target nnlib."
        exit 1
    fi
done

## Fusion G3
if [ -d "$FUSION_DIR_PATH/.git" ]; then
    echo "${green}ExecuTorch: fusion g3 already cloned. Skipping...${reset}"
    # Change into the directory to ensure it's on the correct commit even if already cloned
    cd "$FUSION_DIR_PATH" || exit 1
    git checkout 11230f47b587b074ba0881deb28beb85db566ac2
else
    rm -rf "$FUSION_DIR_PATH"
    echo "${green}ExecuTorch: Cloning fusion g3${reset}"
    git clone "https://github.com/foss-xtensa/nnlib-FusionG3.git" "$FUSION_DIR_PATH"
    if [ $? -ne 0 ]; then
        echo "${red}ExecuTorch: Failed to clone fusion g3.${reset}"
        exit 1
    fi
    
    cd "$FUSION_DIR_PATH" || exit 1
    git checkout 11230f47b587b074ba0881deb28beb85db566ac2
fi

## FACTO
pip install -e "$FACTO_DIR_PATH"

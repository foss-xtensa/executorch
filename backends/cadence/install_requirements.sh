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
HIFI4_DIR_PATH="$CADENCE_DIR_PATH/hifi/third-party/nnlib/nnlib-hifi4"
HIFI5_DIR_PATH="$CADENCE_DIR_PATH/hifi/third-party/nnlib/nnlib-hifi5"
FUSION_DIR_PATH="$CADENCE_DIR_PATH/fusion_g3/third-party/nnlib/nnlib-FusionG3"
FACTO_DIR_PATH="$CADENCE_DIR_PATH/utils/FACTO"

cd "$EXECUTORCH_ROOT_PATH"

## HiFi 4
if [ -d "$HIFI4_DIR_PATH/.git" ]; then
    echo "${green}ExecuTorch: hifi4 nnlib already cloned. Skipping...${reset}"
else
    rm -rf "$HIFI4_DIR_PATH"
    echo "${green}ExecuTorch: Cloning hifi4 nnlib${reset}"
    git clone "https://github.com/foss-xtensa/nnlib-hifi4.git" "$HIFI4_DIR_PATH"
    if [ $? -ne 0 ]; then
        echo "${red}ExecuTorch: Failed to clone hifi4 nnlib.${reset}"
        exit 1
    fi
fi

## HiFi 5
if [ -d "$HIFI5_DIR_PATH/.git" ]; then
    echo "${green}ExecuTorch: hifi5 nnlib already cloned. Skipping...${reset}"
else
    rm -rf "$HIFI5_DIR_PATH"
    echo "${green}ExecuTorch: Cloning hifi5 nnlib${reset}"
    git clone "https://github.com/foss-xtensa/nnlib-hifi5.git" "$HIFI5_DIR_PATH"
    if [ $? -ne 0 ]; then
        echo "${red}ExecuTorch: Failed to clone hifi5 nnlib.${reset}"
        exit 1
    fi
fi

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

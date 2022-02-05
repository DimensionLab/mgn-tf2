#!/bin/bash
# Dataset download script
# 
# Note: For now, there is only one dataset. In future, this will be obsolete.
# 
# Usage:
#     sh download_dataset.sh

#set -e

DATASET_NAME="flag_simple"
CURRENT_DIR=$(pwd)
DATASETS_DIR=""

if [ "$(uname)" == "Darwin" ]; then
    # MacOS       
    DATASETS_DIR=${CURRENT_DIR}/datasets
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    DATASETS_DIR=${CURRENT_DIR}/datasets
else
    # Windows
    DATASETS_DIR="${CURRENT_DIR}/datasets"
fi

if [ DATASETS_DIR == "" ]; then
    echo "Directory where datasets will be stored is not set. Exiting."
    exit 0
fi

echo "Datasets directory: ${DATASETS_DIR}"
OUTPUT_DIR="${DATASETS_DIR}/${DATASET_NAME}"

mkdir -p OUTPUT_DIR

BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/${DATASET_NAME}/"

if [[ -f "${OUTPUT_DIR}/meta.json" ]]; then
    echo "Metadata for $DATASET_NAME exists."
else
    curl "${BASE_URL}meta.json" -o "${OUTPUT_DIR}/meta.json"
fi

for split in train valid test
  do
    if [[ -f "${OUTPUT_DIR}/${split}.tfrecord" ]]; then
            echo "$split.tfrecord exists."
    else
      curl "${BASE_URL}${split}.tfrecord" -o "${OUTPUT_DIR}/${split}.tfrecord"
    fi
done


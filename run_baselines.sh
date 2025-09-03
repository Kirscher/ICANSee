#!/bin/bash

set -e

CONFIGS=("resnet50" "resnet50_oversampling" "resnet50_undersampling" "resnet50_focal" "resnet50_gastronet")

for CONFIG in "${CONFIGS[@]}"; do
  echo "Running baseline: ${CONFIG}"

  # Extract config values using yq
  MODEL=$(yq -r '.model' configs/${CONFIG}.yaml)
  BATCH_SIZE=$(yq -r '.batch_size' configs/${CONFIG}.yaml)
  EPOCHS=$(yq -r '.epochs' configs/${CONFIG}.yaml)
  LR=$(yq -r '.lr' configs/${CONFIG}.yaml)
  VAL_SPLIT=$(yq -r '.val_split' configs/${CONFIG}.yaml)
  SAMPLING=$(yq -r '.sampling // empty' configs/${CONFIG}.yaml)
  GASTRONET=$(yq -r '.gastronet // empty' configs/${CONFIG}.yaml)
  LOSS=$(yq -r '.loss // "bce"' configs/${CONFIG}.yaml)
  FOCAL_ALPHA=$(yq -r '.focal_alpha // 0.25' configs/${CONFIG}.yaml)
  FOCAL_GAMMA=$(yq -r '.focal_gamma // 2.0' configs/${CONFIG}.yaml)

  # Base command
  CMD="python3 train.py \
    --data_path data \
    --model ${MODEL} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --val_split ${VAL_SPLIT} \
    --log_file output/${CONFIG}/train_log.csv \
    --save_model output/${CONFIG}/model.pth \
    --loss ${LOSS} \
    --focal_alpha ${FOCAL_ALPHA} \
    --focal_gamma ${FOCAL_GAMMA}"

  # Conditionally add --sampling
  if [ -n "$SAMPLING" ]; then
    CMD="${CMD} --sampling ${SAMPLING}"
  fi

  # Conditionally add --gastronet
  if [ "$GASTRONET" = "true" ]; then
    CMD="${CMD} --use_gastronet"
  fi

  echo "Executing command: ${CMD}"
  # Run the command
  eval $CMD

done
#!/bin/bash

# Simple DeepRPI Training Script
# Usage: ./run.sh

# Create logs directory
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"


echo "Starting DeepRPI training..."
echo "Log file: $LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Start training in background
nohup python train.py \
    --batch_size 2 \
    --max_epochs 10 \
    --hidden_dim 256 \
    --dropout 0.1 \
    --model_seed 42 \
    --data_split_seed 42 \
    > "$LOG_FILE" 2>&1 &

# Get process ID
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"

echo ""
echo "=========================================="
echo "Training is running in the background!"
echo "=========================================="
echo "Process ID: $TRAIN_PID"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor: tail -f $LOG_FILE"
echo "To stop: kill $TRAIN_PID"
echo "=========================================="

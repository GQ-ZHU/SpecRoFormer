#!/bin/bash
# =============================================================================
# train.sh
#
# Example launch script for training the final SpecRoFormer model
#
# Usage:
#   bash train.sh
#
# Notes:
#   - Run this script from inside the train_test/ directory.
#   - Override any argument by editing the command below or by
#     passing extra flags on the command line.
# =============================================================================

python ./train.py \
    --is_augment_plus \
    --batch_size 512 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss
#!/bin/bash
# =============================================================================
# test.sh
#
# Example launch script for evaluating the trained SpecRoFormer model
#
# Usage:
#   bash test.sh
#
# Notes:
#   - Run this script from inside the train_test/ directory.
#   - The corresponding model checkpoint is expected under
#     ../data/model/<model_name>-<model_backbone>-<loss_type>-bs<batch_size>/
# =============================================================================

python ./test_GNPS.py \
    --is_augment \
    --batch_size 128 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss
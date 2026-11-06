#!/bin/bash
# =============================================================================
# hparam_search.sh
#
# Example launch script for running the Optuna hyperparameter search
# over the final SpecRoFormer configuration.
#
# Usage:
#   bash hparam_search.sh
#
# Notes:
#   - Run from the directory that contains hyperparameter.py.
#   - Each line launches one independent search study; they can also
#     be run in parallel by backgrounding them.
# =============================================================================

python hyperparameter.py \
    --loss SupConWithTanimotoLoss \
    --model_name SpecRoFormer \
    --model_backbone transformer \
    --progress \
    --augment_mode aug

python hyperparameter.py \
    --loss SupConWithTanimotoLoss \
    --model_name SpecRoFormer \
    --model_backbone transformer \
    --progress \
    --augment_mode aug_plus
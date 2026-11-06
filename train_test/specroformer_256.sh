#!/bin/bash

python ./train.py \
    --is_augment_plus \
    --batch_size 256 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss

python ./test_GNPS.py \
    --is_augment_plus \
    --batch_size 256 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss

python ./test_MassBank.py \
    --is_augment_plus \
    --batch_size 256 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss

python ./test_MassSpecGymData.py \
    --is_augment_plus \
    --batch_size 256 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss

python ./test_MoNA.py \
    --is_augment_plus \
    --batch_size 256 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss

python ./test_MTBLS1572.py \
    --is_augment_plus \
    --batch_size 256 \
    --methods SpecRoFormer \
    --loss_type SupConWithTanimotoLoss
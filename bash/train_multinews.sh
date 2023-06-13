#!/bin/bash

cd ../model

DATASET_NAME="multinews"
#DATASET_NAME="multixscience"
#DATASET_NAME="wcep_100"
#DATASET_NAME="arxiv"

PLM_MODEL_PATH="allenai/PRIMERA"
python hgsum.py  \
                --batch_size 2 \
                --gpus 4  \
                --mode train \
                --model_name HGSum_${DATASET_NAME} \
                --model_path ../result/HGSum_${DATASET_NAME}/  \
                --dataset_name ${DATASET_NAME} \
                --data_path ../data/ \
                --pretrained_primer ${PLM_MODEL_PATH} \
                --num_workers 8 \
                --beam_size 2 \
                --test_imediate \
                --num_train_data 32 \
                --with_sent_sep \
                --adafactor \
                --label_smoothing 0.1 \
                --test_batch_size 1 \
                --compute_rouge
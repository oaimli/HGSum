#!/bin/bash

cd ../model

DATA_NAME="multinews"
PLM_MODEL_PATH="allenai/PRIMERA-multinews"

#DATA_NAME="multixscience"
#PLM_MODEL_PATH="allenai/PRIMERA-multixscience"

#DATA_NAME="arxiv"
#PLM_MODEL_PATH="allenai/PRIMERA-arxiv"

#DATA_NAME="wcep_100"
#PLM_MODEL_PATH="allenai/PRIMERA-wcep"
python hgsum.py  \
                --batch_size 2 \
                --gpus 1  \
                --mode test \
                --model_name HGSum_${DATA_NAME} \
                --model_path ../result/HGSum_${DATA_NAME}/  \
                --dataset_name ${DATA_NAME} \
                --data_path ../data/ \
                --pretrained_primer ${PLM_MODEL_PATH} \
                --num_workers 8 \
                --beam_size 5 \
                --with_sent_sep \
                --num_test_data 8
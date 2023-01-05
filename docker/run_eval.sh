#!/bin/bash

docker run -d\
    --rm \
    --shm-size="8g"\
    --gpus all\
    -v $PWD:/workspace \
    -v $HOME/mpt_manipulation_data:/root/data \
    -v $HOME/global_planner_data:/root/data2d \
    mpt:torch-o3d-pyb-tg \
    python3 eval_2d.py \
        --dict_model_folder=/root/data2d/general_mpt/model39/ \
        --ar_model_folder=/root/data2d/general_mpt/stage2/model11/ \
        --val_data_folder=/root/data2d/maze4/val \
        --start=0 \
        --samples=500 \
        --num_paths=1 \
        --map_type=maze
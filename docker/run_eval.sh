#!/bin/bash
SAMPLES=250
for CID in {0..1}
do
    docker run -d\
        --rm \
        --shm-size="8g"\
        --gpus all\
        -v $PWD:/workspace \
        -v $HOME/mpt_manipulation_data:/root/data \
        -v $HOME/global_planner_data:/root/data2d \
        mpt:torch-o3d-pyb-tg \
        python3 eval_14d.py \
            --ar_model_folder=/root/data/general_mpt_bi_panda/stage2/model1/ \
            --val_data_folder=/root/data/bi_panda/val \
            --start=$((2001+CID*SAMPLES)) \
            --samples=$SAMPLES \
            --num_paths=1 \
            --planner_type=bitstar
done

# SAMPLES=250
# for CID in {0..1}
# do
#     docker run -d\
#         --rm \
#         --shm-size="8g"\
#         --gpus all\
#         -v $PWD:/workspace \
#         -v $HOME/mpt_manipulation_data:/root/data \
#         -v $HOME/global_planner_data:/root/data2d \
#         mpt:torch-o3d-pyb-tg \
#         python3 eval_6d.py \
#             --ar_model_folder=/root/data/general_mpt/stage2/model8/ \
#             --val_data_folder=/root/data/pandav3/val \
#             --start=$((2000+CID*SAMPLES)) \
#             --samples=$SAMPLES \
#             --num_paths=1 \
#             --planner_type=informedrrtstar
# done


# SAMPLES=250
# for CID in {0..1}
# do
#     docker run -d\
#         --rm \
#         --shm-size="8g"\
#         --gpus all\
#         -v $PWD:/workspace \
#         -v $HOME/mpt_manipulation_data:/root/data \
#         -v $HOME/global_planner_data:/root/data2d \
#         mpt:torch-o3d-pyb-tg \
#         python3 eval_14d.py \
#             --ar_model_folder=/root/data/general_mpt_bi_panda/stage2/model1/ \
#             --val_data_folder=/root/data/bi_panda_shelf/val \
#             --start=0 \
#             --samples=1 \
#             --paths_start=$((CID*SAMPLES))\
#             --num_paths=$SAMPLES \
#             --planner_type=informedrrtstar
# done
#!/bin/bash

SAMPLES=100
for CID in {0..19}
do
	docker run -d\
	    --rm \
	    --shm-size="8g"\
        --gpus all\
		-v $PWD:/workspace \
		-v $HOME/mpt_manipulation_data:/root/data \
		-v $HOME/global_planner_data:/root/data2d \
		mpt:torch-o3d-pyb-tg \
	    python3 dual_arms/collect_data.py \
			--start=$((1+CID*SAMPLES)) \
			--samples=$SAMPLES \
			--num_paths=10 \
            --log_dir=/root/data/bi_panda/train
done

# do
# 	docker run -d\
# 	    --rm \
# 	    --shm-size="8g"\
#         --gpus all\
# 		-v $PWD:/workspace \
# 		-v $HOME/mpt_manipulation_data:/root/data \
# 		-v $HOME/global_planner_data:/root/data2d \
# 		mpt:torch-o3d-pyb-tg \
# 	    python3 dual_arms/dual_arm_utils.py \
# 			--start=$((1+CID*SAMPLES)) \
# 			--samples=$SAMPLES \
#           --log_dir=/root/data/bi_panda/train
# done
#!/bin/bash

SAMPLES=350
for CID in {0..4}
do
	docker run -d\
	    --rm \
	    --shm-size="8g"\
        --gpus all\
		-v $PWD:/workspace \
		-v $HOME/mpt_manipulation_data:/root/data \
		-v $HOME/global_planner_data:/root/data2d \
		mpt:torch-o3d-pyb-tg \
	    python3 save_quant_index.py \
			--start_env=$((750+CID*SAMPLES)) \
			--samples=$SAMPLES \
			--model_dir=/root/data2d/general_mpt/model30 \
            --data_dir=/root/data2d/maze4/ \
            --save_dir=/root/data2d/general_mpt/model30/quant_key/maze4/
done
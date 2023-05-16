#!/bin/bash

SAMPLES=5
for CID in {0..19}
do
	docker run -d\
	    --rm \
	    --shm-size="8g"\
        --gpus all\
		-v $PWD:/workspace \
		-v $HOME/mpt_manipulation_data:/root/data \
		-v $HOME/global_planner_data:/root/data2d \
		mpt:focal-ompl-mod-torch-tg \
	    python3 panda_constraint_shelf.py \
			--start=$((CID*SAMPLES)) \
			--samples=$SAMPLES \
            --fileDir=/root/data/panda_constraint/val \
            --numPaths=1
done
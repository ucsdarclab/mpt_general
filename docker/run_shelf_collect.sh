#!/bin/bash
# A script to run ompl-docker
SAMPLES=25
for CID in {0..19..1}
do
	docker run -d \
	    --rm \
	    --name=data_$1_$CID \
	    --shm-size="2g"\
        -v $PWD:/workspace \
        -v $HOME/mpt_manipulation_data:/root/data \
        -v $HOME/global_planner_data:/root/data2d \
        mpt:torch-o3d-pyb-tg \
	    python3 panda_shelf_env.py \
			--start=$((CID*SAMPLES)) \
			--samples=$SAMPLES \
			--fileDir=/root/data/panda_shelf/val\
			--numPaths=1
done
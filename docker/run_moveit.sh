#!/bin/bash
docker run --rm -it \
		-p $1:8800 \
		--gpus all \
		--shm-size="16g"\
		-e DISPLAY=$DISPLAY \
		-e QT_X11_NO_MITSHM=1 \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		-v $XAUTH:/root/.Xauthority \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v $PWD:/workspace \
		-v $HOME/mpt_manipulation_data:/root/data \
		-v $HOME/global_planner_data:/root/data2d \
		-v $HOME/mpt_ros_workspace:/root/ws \
		mpt:moveit \
		bash
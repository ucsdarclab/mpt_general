FROM mpt:torch-o3d-pyb-tg 

# Get rid of older nvidia keys:
RUN rm /etc/apt/sources.list.d/cuda.list && \
	rm /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-key del 7fa2af80 && \
	apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
	apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

	
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO melodic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-core=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*


# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-robot=1.4.1-0* \
    ros-melodic-moveit\
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install rospkg catkin_pkg

# Useful files for PC
RUN apt-get update && apt-get install -y \
	ros-melodic-pcl-ros \
	&& rm -rf /var/lib/apt/lists/*

	
# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
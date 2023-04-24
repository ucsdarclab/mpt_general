FROM ompl:focal-1.6-devel AS BUILDER

# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 AS BASE
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04 AS BASE

COPY --from=BUILDER /usr/local/include/ompl-1.6 /usr/include/ompl-1.6
COPY --from=BUILDER /usr/local/lib/libompl* /usr/local/lib/
COPY --from=BUILDER /usr/lib/libtriangle* /usr/lib/
COPY --from=BUILDER /usr/local/bin/ompl_benchmark_statistics.py /usr/bin/ompl_benchmark_statistics.py
COPY --from=BUILDER /usr/lib/python3/dist-packages/ompl /usr/lib/python3/dist-packages/ompl

ENV DEBIAN_FRONTEND=noninteractive

# Files required for OMPL
RUN apt-get update && apt-get install -y \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-numpy-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-python-dev \
    libboost-test-dev \
    libflann-dev \
    libode-dev \
    libeigen3-dev \
	python3-pip\
	&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
        pypy3 \
        wget && \
    # Install spot
    wget -O /etc/apt/trusted.gpg.d/lrde.gpg https://www.lrde.epita.fr/repo/debian.gpg && \
    echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libspot-dev && \
    pip3 install pygccxml pyplusplus
    
RUN python3 -m pip install -U pip

# RUN pip3 install torch==1.9.1+cu111 \
# 	torchvision==0.10.1+cu111 \
# 	torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install torch==1.12.0+cu113 \
    torchvision==0.13.0+cu113 \
    torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install jupyter \
				einops \
				scikit-image \
                webdataset \
                tensorboard \
                pybullet \
				 -e git+https://github.com/benelot/pybullet-gym.git@55eaa0defca7f4ae382963885a334c952133829d#egg=pybulletgym \
                tqdm\
                toolz \
                seaborn

                
RUN apt-get update && apt-get install -y \
	libgl1 \
	libgomp1 \
	libusb-1.0-0 \
	&& rm -rf  /var/lib/apt/lists/*

RUN pip3 install open3d

RUN pip3 install --no-index \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
    
RUN pip3 install torch-geometric 

WORKDIR /workspace
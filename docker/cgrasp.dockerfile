FROM mpt:focal-ompl-mod-torch-tg

# Install requirements for running contact-graspnet

RUN apt-get update && apt-get install -y \
    libqt5gui5

RUN pip3 install \
    tensorflow==2.11.0 \
    opencv-python-headless \
    trimesh \
    pyrender \
    mayavi \
    PyQt5
    

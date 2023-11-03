# VQ-MPT
This repo contains our code for replicating all our experiments. We have provided a new package called [VQ-MPT](https://github.com/jacobjj/vqmpt), an independent code base to generate the sampling distributions using VQ-MPT.

For more details on the paper, visit our [website](https://sites.google.com/eng.ucsd.edu/vq-mpt/home)
## Model Training

To train stage 1, you can run `train_stage1_panda.py <args>`. For more info about the arguments, run `train_stage1_panda.py -h`

To train stage 2, you can run `python3 train_stage2.py  <args>`. For more info about the arguments, run `train_stag2.py -h`

## Training Timings

We trained all our models on a system with AMD Ryzen Threadripper 1950X CPU with an RTX 3090 GPU. To train Stage 1 of the model, it took 1 hr for the 2D and 7D models and 2.7 hrs for the 14D model. Training Stage 2 took 2 hrs, 3.5 hrs, and 3.6 hrs for the 2D, 7D, and 14D planners respectively.

Model weights will be uploaded in the future.

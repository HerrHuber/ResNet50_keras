Residual Neural network with keras2.3
================================================================
A ResNet50 implementation using keras version 2.3

installation
------------
python3 setup.py install --user

execute tensorflow docker container
-----------------------------------
CPU:
docker run -it --rm  -u $(id -u):$(id -g) -v /home/user/Projects/ML/L_layer_model_tf:/tmp -w /tmp tensorflow/tensorflow:2.3 /bin/bash
GPU:
docker run --gpus all -it --rm  -u $(id -u):$(id -g) -v /home/user/Projects/ML/L_layer_model_tf:/tmp -w /tmp tensorflow/tensorflow:2.3-gpu-py3 /bin/bash

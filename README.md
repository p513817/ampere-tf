# Install NVIDIA-Tensorflow with Docker Container

# Reference
* [Accelerating TensorFlow on NVIDIA A100 GPUs](https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/)
* [无需源码编译 | 基于RTX3090配置tensorflow1.15环境](https://blog.csdn.net/qq_39543404/article/details/112171851)
* [RTX3080+Ubuntu18.04+cuda11.1+cudnn8.0.4+TensorFlow1.15.4+PyTorch1.7.0环境配置](https://blog.csdn.net/weixin_47658790/article/details/115419933)

# Requirement
* NVIDIA Driver 
* Docker
* nvidia-container

# Docker Image
> nvcr.io/nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04
* Run docker container
    ```bash
    $ docker run -it --name nv-tf --gpus all \
    -v `pwd`:/workspace -w /workspace \
    nvcr.io/nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04 bash
    
    # Output
    Unable to find image 'nvcr.io/nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04' locally
    11.2.0-cudnn8-devel-ubuntu18.04: Pulling from nvidia/cuda
    09db6f815738: Pull complete 
    d79696845ef2: Pull complete 
    9cace1db9258: Pull complete 
    5093a1370488: Pull complete 
    affabbc9735b: Pull complete 
    839e92906efc: Pull complete 
    36d15b49ae4c: Pull complete 
    be6750df422d: Pull complete 
    02a4c72adbe9: Pull complete 
    cc3c4b345c51: Pull complete 
    Digest: sha256:ba15e2c49bc81211a26f42e9f21374cb2e9e56a5d7b6ce710cf0291ed880327b
    Status: Downloaded newer image for nvcr.io/nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04
    ```
* Check CUDA and cuDNN
    ```bash
    nvcc -V
    
    # Output
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2021 NVIDIA Corporation
    Built on Sun_Feb_14_21:12:58_PST_2021
    Cuda compilation tools, release 11.2, V11.2.152
    Build cuda_11.2.r11.2/compiler.29618528_0
    
    ```

# Install NVIDIA Tensorflow
* Workflow
    ```bash
    apt update
    apt install -y python3-dev python3-pip git
    pip3 install --upgrade pip setuptools requests

    pip3 install nvidia-pyindex

    pip3 install nvidia-tensorflow[horovod]

    # Output
    Successfully installed absl-py-1.2.0 astor-0.8.1 cached-property-1.5.2 cloudpickle-2.1.0 dataclasses-0.8 gast-0.2.2 google-pasta-0.2.0 grpcio-1.47.0 h5py-3.1.0 importlib-metadata-4.8.3 keras-applications-1.0.8 keras-preprocessing-1.1.2 markdown-3.3.7 numpy-1.18.5 nvidia-cublas-11.2.1.74 nvidia-cuda-cupti-11.1.69 nvidia-cuda-nvcc-11.1.74 nvidia-cuda-nvrtc-11.1.105 nvidia-cuda-runtime-11.1.74 nvidia-cudnn-8.0.4.30 nvidia-cufft-10.3.0.74 nvidia-curand-10.2.2.74 nvidia-cusolver-11.0.0.74 nvidia-cusparse-11.2.0.275 nvidia-dali-cuda110-0.27.0 nvidia-dali-nvtf-plugin-0.27.0+nv20.11 nvidia-horovod-0.20.2+nv20.11 nvidia-nccl-2.8.2 nvidia-tensorboard-1.15.0+nv20.11 nvidia-tensorflow-1.15.4+nv20.11 nvidia-tensorrt-7.2.1.6 opt-einsum-3.3.0 protobuf-3.19.4 psutil-5.9.1 pyyaml-6.0 tensorboard-1.15.0 tensorflow-estimator-1.15.1 termcolor-1.1.0 typing-extensions-4.1.1 webencodings-0.5.1 werkzeug-2.0.3 wrapt-1.14.1 zipp-3.6.0
    ```

* Check 
    * Tensorflow Version
        ```bash
        $ python3 -c 'import tensorflow as tf; print(tf.__version__)'

        # Output
        2022-07-27 03:45:17.524269: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
        WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.
        1.15.4
        ```
    * Check GPU could capture in Tensorflow
        ```bash
        $ python3 -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))"

        # Output
        2022-07-27 03:45:36.465661: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
        WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.
        2022-07-27 03:45:37.229005: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
        2022-07-27 03:45:37.285247: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1082] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
        2022-07-27 03:45:37.285355: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1665] Found device 0 with properties: 
        name: NVIDIA GeForce RTX 3080 Ti major: 8 minor: 6 memoryClockRate(GHz): 1.665
        pciBusID: 0000:01:00.0
        2022-07-27 03:45:37.285372: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
        2022-07-27 03:45:37.287393: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
        2022-07-27 03:45:37.288186: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
        2022-07-27 03:45:37.288461: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
        2022-07-27 03:45:37.290611: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.11
        2022-07-27 03:45:37.291171: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
        2022-07-27 03:45:37.291294: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
        2022-07-27 03:45:37.291398: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1082] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
        2022-07-27 03:45:37.291544: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1082] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
        2022-07-27 03:45:37.291616: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1793] Adding visible gpu devices: 0
        Num GPUs Available:  1
        ```
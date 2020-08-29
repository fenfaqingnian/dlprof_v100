# DLprof for Tensorflow 1.15
## Executive summary - 
This repository contained scripts to kick-start DLprof for Tensorflow 1.15 using Nvidia NGC docker container 


## Requirements -
This repository use docker container from  NGC repo (go to https://ngc.nvidia.com/ ) with Ubuntu 18.04 as OS with 1 NVIDIA GPU. 

You will need at least the following minimum setup:
- Supported Hardware: NVIDIA GPU with Volta Architecture or later (Volta, Turing )
- Any supported OS for nvidia docker
- [NVIDIA-Docker 2](https://github.com/NVIDIA/nvidia-docker)
- NVIDIA Driver ver. 450.xx 
- install NVdashboard 


### NVIDIA Docker used
- tensorflow 20.07-tf1-py3 NGC container 
For a full list of supported systems and requirements on the NVIDIA docker, consult [this page](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/#framework-matrix-2020).

## Quick Start Guide for Interactive Experiments with 1CPU vs. 1GPU vs. MultiGPU

#### Step 0 -clone the git repo https://github.com/Zenodia/Profiler_DLprof_TF1.git

and cd into the repo directory 

#### Step 1.1 -  pull & run the NGC docker image for Tensorflow 1.15 repo 
```bash 
0a_run_docker_enabled_dl_prof.sh 1 2345
```
or 
```
sudo docker run --runtime=nvidia --cap-add=SYS_ADMIN  -it --rm --gpus '"device=<available_gpu_device>"' --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  -p 2345:2345  -p 6006:6006  -p 6007:6007  -p 6008:6008  -p 6009:6009  -v $(pwd):/workspace nvcr.io/nvidia/tensorflow:20.07-tf1-py3
```

#### Step 1.2 - install NVdashboard
```bash 
0b_nv_dashboard.sh .sh 1 2345
```
or 

```
pip install jupyterlab-nvdashboard &&
jupyter labextension install jupyterlab-nvdashboard
```

#### Step 1.3 - check the availbility of nsight  
```bash
bash 0b_nsys_check.sh
```

should see similar to the following 

Sampling Environment Check
Linux Kernel Paranoid Level = -1: OK
Linux Distribution = Ubuntu
Linux Kernel Version = 4.15.0-112: OK
Linux perf_event_open syscall available: OK
Sampling trigger event available: OK
Intel(c) Last Branch Record support: Not Available
Sampling Environment: OK


#### Step 1.4 - Launch the Jupyter Notebook
```bash
bash 0c_lanch_jupyter.sh.sh 2345
```
To run jupyter notebook in the host browser , remember to use the same port number you specify in docker run on step 1


#### Step 2 - call out a preferred browser and use jupyter as UI to interact with the running docker
call out firefox ( or other default browser )
type in in the browser url: `https://localhost:<port_number>`
If you are using a remote server, change the url accordingly: `http://you.server.ip.address:<port_number>`

after selected the tf_keras folder, then call out a terminal 
![alt text](<./pics/callout_terminal.JPG>) 


#### Step 3.1 - run through the 3 DLprof bash scripts to get profiling 
establish baseline by running profiling on original python scripts tf_keras_v0.py 


``` 2a_run_dlprof_with_nvtx_v0.sh ```

![alt text](<./pics/baseline.jpg>)



modify  tf_keras_v1.py python scripts according to the recommendation from previous DLprof dashboard 

``` 2b_run_dlprof_with_nvtx_v1.sh ```

![alt text](<./pics/improved_based_on_recommendations.jpg>)



building on top of the improvement, modify tf_keras_v2.py 's model architecture + data pipeline in order to increase the GPU utilisation and optimize TensorCore ( TC ) usage


``` 2c_run_dlprof_with_nvtx_v2.sh ```
 
![alt text](<./pics/improve_basedon_gpu_utils.jpg>)

one can also use NVdashboard plug in to jupyter lab and monitor the GPU utilization during training in real-time
![alt text](<./pics/gpu_utils_tf_keras_v2.JPG>)

#### Step 3.2 - tensorboard plug in for visualization
run through the notebook 
```3_visualized_DLprof_with_nvtx.ipynb``` 


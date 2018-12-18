# Object Detection Using TensorFlow
This repository contains the works which I have done using TensorFlow Object Detection API. Before running the code, you need to first install tensorflow and all its dependecies.
## System Requirements
```
* Ubuntu 16.04 LTS or later  
* Intel Pentium Quad Core Processor  
* Dependencies Installed  
```
Note: Above requirements is according to my system.  
Below installation procedure is for CPU which has no GPU.  
For installation of TensorFlow on GPU, refer [TensorFlow GPU Installation](https://www.tensorflow.org/install/)  
## Installation:
``pip`` is a python package manager used for installing all the modules, if ``pip`` is not installed, then run the following command before insalling dependencies:
 * Installing ```pip```
 ```bash
For python 2.7:
	sudo apt-get -y install python-pip python-dev build-essential  
	sudo pip install --upgrade pip  
For python 3:  
	sudo apt-get -y install python3-pip  
	pip3 -version (For checking the version of pip3)
```
 * Installing Dependencies:  
   From the ```terminal``` run the following command:  
``` bash
    bash dependecies.sh
```
Note: ```dependencies.sh``` file contains all the dependencies/packages in order to run the program.

* In python interpreter, run the following commands to check whether TensorFlow is installed sucessfully:
```python
import tensorflow as tf
print(tf.__version__) # print the version of tensorflow installed.
```
For detailed information on installation of TensorFlow, refer the below link:  
[TensorFlow Installation](https://www.tensorflow.org/install/)  
[TensorFlow Object Detection API Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
## OpenCV Installation:
* I am using ``OpenCV`` for implementing the object detection in real-time, you need to install ``OpenCV`` also.  
* Refer the below link for installation  
[OpenCV Installation](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)

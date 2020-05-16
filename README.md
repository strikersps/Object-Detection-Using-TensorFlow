# Object Detection Using TensorFlow
This repository contains implementation of object detection in [live video-stream](https://github.com/strikersps/Object-Detection-Using-TensorFlow/blob/master/Custom_Object_Detection/pen_detector/pen_detection/pen_detection_opencv.py
) and also in an [image](https://github.com/strikersps/Object-Detection-Using-TensorFlow/blob/master/Custom_Object_Detection/pen_detector/pen_detection/pen_detection.py
) using TensorFlow Object Detection API. Before running the code, you need to first install tensorflow and all of its dependencies.
## System Requirements
```
* Ubuntu 18.04 LTS Bionic Beaver or Later  
* Intel Core i3 Processor  
* Dependencies Installed  
```
**Note:** Above requirements is according to my system.  
Below installation procedure is for CPU not for GPU powered system.  
For installation of TensorFlow on GPU, refer [TensorFlow GPU Installation](https://www.tensorflow.org/install/)  
## Installation:
``pip`` is a python package manager used for installing all the modules, if ``pip`` is not installed, then run the following command before insalling dependencies:
 #### ```pip``` Installation:
 ```bash
For python 2.7:
	sudo apt-get -y install python-pip python-dev build-essential  
	sudo pip install --upgrade pip  
For python 3:  
	sudo apt-get -y install python3-pip  
	pip3 -version (For checking the version of pip3)
```
 #### Dependencies Installation:  
   From the ```terminal``` run the following command:
``` bash
    bash dependecies.sh
```
Note: ```dependencies.sh``` file contains all the dependencies/packages in order to run the program.

* In python interpreter, run the following commands to check whether TensorFlow is installed successfully:
Open the ```terminal``` and write ```python3``` or ```python``` depending upon your ```python``` version and run the following code:  
```python
import tensorflow as tf
print(tf.__version__) # print the version of tensorflow installed.
```
If you get the following output, then ```TensorFlow``` is installed successfully:  
```python
   '1.12.0' (Version of tensorflow installed on your system)
```
For detailed information on installation of ```TensorFlow```, refer the below link:  
[TensorFlow Installation](https://www.tensorflow.org/install/)  
[TensorFlow Object Detection API Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
#### OpenCV Installation:
* I used ``OpenCV`` for extracting frames from the camera in real-time.  
* Refer the below link for installation [OpenCV Installation](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)

# Custom Object Detection: Pen Detection

#### Introduction:
The repository contains an object detection model, specifically pen detection model which is able to detect pen in an image and videostream,
with a confidence/score of 90 to 95%.  

I have trained the **"ssd_mobilenet_v1_coco_2017_11_17"** model on my own set of pen images, the whole training procedure is explained in the link: [Training of Pen Detector](https://github.com/strikersps/InternshipProjects/blob/master/Object_Detection_Using_TensorFlow/Custom_Object_Detection/pen_detector/pen_dataset/README.md)  

#### Output:

1. Output when program takes images as input:  
```bash
   python3 penDetection.py
```

**Confidence Score: 99%**   
![alt-text](https://github.com/strikersps/InternshipProjects/blob/master/Object_Detection_Using_TensorFlow/Custom_Object_Detection/pen_detector/pen_detection/output/1.png)  
**Confidence Score: 99%**  
![alt-text](https://github.com/strikersps/InternshipProjects/blob/master/Object_Detection_Using_TensorFlow/Custom_Object_Detection/pen_detector/pen_detection/output/2.png)  
**Confidence Score: 96%**  
![alt-text](https://github.com/strikersps/InternshipProjects/blob/master/Object_Detection_Using_TensorFlow/Custom_Object_Detection/pen_detector/pen_detection/output/3.png)  

2. Output when program detects pen in live videostream:

``` bash
	python3 penDetection_opencv.py
```


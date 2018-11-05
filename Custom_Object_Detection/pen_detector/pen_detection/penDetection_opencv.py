import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2 as cv

from packaging import version

tensorflow_ver = tf.__version__
if version.parse(tensorflow_ver) < version.parse("1.4.0"):
    print("Upgrading Tensorflow...")
    os.system('pip3 install --upgrade tensorflow')

print('OpenCV Version: %s\nTensorFlow Version: %s\nNumPy Version: %s' % (cv.__version__,tf.__version__,np.__version__))

# cam = cv.VideoCapture(0) # Capturing the video feed from my WebCam, which is assigned an index value of 0

if cv.VideoCapture(1).isOpened():
    cam = cv.VideoCapture(1)
else:
	cam = cv.VideoCapture(0)

# scaling_factor = 0.5 # Image size scaling factor

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Object detection imports
from pen_detection.utils import label_map_util

from pen_detection.utils import visualization_utils as vis_util

# Model Preparation
MODEL_NAME = 'pen_detection_model_v1.0'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training','pen_label_map.pbtxt')
NUM_CLASSES = 1

# Load a frozen graph of Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while (cam.isOpened()): #isOpen() is simply checking whether the camera is open or not,if not it return false, otherwise true
            #  Defined two variables ret,frame, ret->boolean,checking whether any value is returned or not from
   	        # image_np variable stores each frame which is returned from the func, if no frame is returned,
            # error will not be generated rather it will store None
            ret,image_np=cam.read()
            if ret == True:
                # Definite input and output Tensors for detection_graph
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                # Actual detection.
                # Syntax of sess.run(fetches,feed_dict,options,run_metadata)
                (boxes, scores, classes, num) = sess.run(
            	   [detection_boxes, detection_scores, detection_classes, num_detections],feed_dict = {image_tensor:image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3)
                cv.imshow('Pen Detection Live', cv.resize(image_np,(800,800)))
                # This statemenet runs one time for every frame, and it says that if the user enters 'q', then 
                # simply break the loop.
                if((cv.waitKey(25) & 0xFF) == ord('q')):
                    cam.release() # releases the camera
                    cv.destroyAllWindows() # closes all open windows
                    break
            else:
                print('Stream not available!')
                break
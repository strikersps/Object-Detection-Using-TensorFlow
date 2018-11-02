import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from packaging import version

tensorflow_version = tf.__version__
numpy_version = np.__version__

if version.parse("1.4.0") > version.parse(tensorflow_version):
    print("Updating tensorflow to 1.4.0 or higher...")
    os.system('pip3 install --upgrade tensorflow')

print('Tensorflow Version: %s\nNumpy Version: %s\n' % (tensorflow_version, numpy_version))

# This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Object detection imports

from pen_detection.utils import label_map_util
from pen_detection.utils import visualization_utils as vis_util  # used for visualization

# Model Preparation
MODEL_NAME = 'pen_detection_model_v1.0'  # custom model trained to only detect pens.
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'pen_label_map.pbtxt')  # Add object_detection path

NUM_CLASSES = 1  # Trained the API for only one class i.e Pen

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection Part
# For the sake of simplicity you can use less images, add those images in the test_images directory 
# If you want to test the code with your images, just add path to the images to the PATH_TO_TEST_IMAGES_DIR.

PATH_TO_TEST_IMAGES_DIR = 'pen_test_images'  # change this directory according to your needs.
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'pen{}.jpg'.format(i)) for i in range(1, 14)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Print the object details to the console instead of visualizing them with the code above
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            boxes = np.squeeze(boxes)
            threshold = 0.45  # set a minimum score threshold of 45%
            obj_above_thresh = sum(n > threshold for n in scores)
            print("detected %s objects in %s above a %s score" % (obj_above_thresh, image_path, threshold))
            for c in range(0, len(classes)):
                if scores[c] >= threshold:
                    class_name = category_index[classes[c]]['name']
                    print(" object %s is a %s - score: %s, location: %s" % (c, class_name, scores[c], boxes[c]))
            # Below is used for visualizing with Matplot
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                               np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32),
                                                               np.squeeze(scores),
                                                               category_index,
                                                               use_normalized_coordinates=True,
                                                               line_thickness=8)  # Passing parameters end here
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()
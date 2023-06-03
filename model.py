import os
import sys
import time

import numpy as np
import cv2

# Suppressing TF messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


from utils import BBox


class FrozenGraph():

    def __init__(self, checkpoint_path):
        self.inference_list = []
        self.count = 0

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True    

    def run(self, image):
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # The array-based representation of the image
        # will be used later in order to prepare the
        # result image with bboxes and labels on it.

        # Expand dimensions since the model expects images
        # to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image
        # where a particular object was detected.
        bboxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        start_time = time.time()
        bboxes, scores, classes, num_detections = self.sess.run(
            [bboxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded}
        )
        self.inference_list.append(time.time() - start_time)
        self.count = self.count + 1
        average_inference = sum(self.inference_list) / self.count
        # print('Average inference time: {}'.format(average_inference))

        # Postprocessing.
        bboxes = np.squeeze(bboxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        bboxes_list = []
        for class_idx, score, bbox in zip(classes, scores, bboxes):
            # NOTE: ymin, xmin, ymax, xmax = bbox
            top = int(bbox[0] * image.shape[0])
            bottom = int(bbox[2] * image.shape[0])
            left = int(bbox[1] * image.shape[1])
            right = int(bbox[3] * image.shape[1])
            height = bottom - top + 1
            width = right - left + 1
            bboxes_list.append(BBox(class_idx=class_idx, score=score,
                                    top=top, left=left, height=height, width=width))

        return num_detections, bboxes_list

        
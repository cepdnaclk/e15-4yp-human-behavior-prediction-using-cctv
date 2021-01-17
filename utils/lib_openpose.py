# -- Libraries
if True: # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

import sys, os, time, argparse, logging
import cv2

# openpose packages
sys.path.append(ROOT + "src/githubs/tf-pose-estimation")
#from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common


# -- Settings
MAX_FRACTION_OF_GPU_TO_USE = 0.4
IS_DRAW_FPS = True

def _set_config():
    ''' Set the max GPU memory to use '''
    # For tf 1.13.1, The following setting is needed
    import tensorflow as tf
    from tensorflow import keras
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=MAX_FRACTION_OF_GPU_TO_USE
    return config

# -- Main class
class SkeletonDetector(object):
    # This class is mainly copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, model="cmu", image_size=[432, 368]):
        ''' Arguments:
            model {str}: "cmu" or "mobilenet_thin".
            image_size {str}: resize input images before they are processed. 
                Recommends : 432x368, 336x288, 304x240, 656x368, 
        '''
        # -- Check input
        #assert(model in ["mobilenet_thin", "cmu"])
        #self._w, self._h = _get_input_img_size_from_string(image_size)
        self._w = image_size[0]
        self._h = image_size[1]
        
        # -- Set up openpose model
        self._model = model
        self._resize_out_ratio = 4.0 # Resize heatmaps before they are post-processed. If image_size is small, this should be large.
        self._config = _set_config()
        self._tf_pose_estimator = TfPoseEstimator(self._model, target_size=(self._w, self._h),tf_config=self._config)
        self._prev_t = time.time()
        self._cnt_image = 0


    def detect(self, image):
        ''' Detect human skeleton from image.
        Arguments:
            image: RGB image with arbitrary size. It will be resized to (self._w, self._h).
        Returns:
            humans {list of class Human}: 
                `class Human` is defined in 
                "src/githubs/tf-pose-estimation/tf_pose/estimator.py"
                
                The variable `humans` is returned by the function
                `TfPoseEstimator.inference` which is defined in
                `src/githubs/tf-pose-estimation/tf_pose/estimator.py`.

                I've written a function `self.humans_to_skels_list` to 
                extract the skeleton from this `class Human`. 
        '''

        self._cnt_image += 1
        if self._cnt_image == 1:
            self._image_h = image.shape[0]
            self._image_w = image.shape[1]
            self._scale_h = 1.0 * self._image_h / self._image_w
        t = time.time()

        # Do inference
        humans = self._tf_pose_estimator.inference(image, resize_to_default=(self._w > 0 and self._h > 0), upsample_size=self._resize_out_ratio)

        return humans

    def humans_to_skels_list(self, humans, scale_h = None): 
        ''' Get skeleton data of (x, y * scale_h) from humans.
        Arguments:
            humans {a class returned by self.detect}
            scale_h {float}: scale each skeleton's y coordinate (height) value.
                Default: (image_height / image_widht).
        Returns:
            skeletons {list of list}: a list of skeleton.
                Each skeleton is also a list with a length of 36 (18 joints * 2 coord values).
            scale_h {float}: The resultant height(y coordinate) range.
                The x coordinate is between [0, 1].
                The y coordinate is between [0, scale_h]
        '''
        if scale_h is None:
            scale_h = self._scale_h
        skeletons = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(18*2)
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[2*idx] = body_part.x
                skeleton[2*idx+1] = body_part.y * scale_h
            skeletons.append(skeleton)
        return skeletons, scale_h

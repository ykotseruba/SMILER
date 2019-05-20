import numpy as np
import os
import sys
import time
import cv2

caffe_root = '/opt/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

EPSILON = 1e-8
IMAGE_DIM = 224


def prepare_image(img):
    im = np.array(img, dtype=np.float32)
    im = cv2.resize(im, (IMAGE_DIM, IMAGE_DIM), interpolation = cv2.INTER_LINEAR)
    im = im[..., ::-1]
    im -= np.array((103.939, 116.779, 123.68))
    im = np.transpose(im, (2, 0, 1))
    return im

class DHS():
    def __init__(self):
        caffe.set_mode_gpu()
        # choose which GPU you want to use
        caffe.set_device(0)
        caffe.SGDSolver.display = 0
        # load net
        self.net = caffe.Net('SO_RCL_deploy.prototxt', 'SO_RCL_models_iter_10000.caffemodel', caffe.TEST)

    def compute_saliency(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        im = prepare_image(img)

        # shape for input (data blob is N x C x H x W), set data
        self.net.blobs['img'].reshape(1, *im.shape)
        self.net.blobs['img'].data[...] = im
        # run net and take argmax for prediction
        res = self.net.forward()
        salmap = np.transpose(res['RCL1_sm'][:, 0, :, :], (1, 2, 0));
        salmap = cv2.resize(salmap, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_LINEAR) *255 #multiply by 255 because smiler_tools/image_processing converts to uint8 which may result in precision loss

        return salmap

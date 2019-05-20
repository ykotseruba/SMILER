import numpy as np
import os
import sys
import time
import cv2

#caffe_root = '/opt/caffe_dss/'
import sys
#sys.path.insert(0, caffe_root + 'python')

import caffe

EPSILON = 1e-8

def prepare_image(img):
    im = np.array(img, dtype=np.float32)
    im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    im = im[..., ::-1]
    im -= np.array((103.939, 116.779, 123.68))
    im = im.transpose((2,0,1))
    return im

class DSS():
    def __init__(self):
        caffe.set_mode_gpu()
        # choose which GPU you want to use
        caffe.set_device(0)
        caffe.SGDSolver.display = 0
        # load net
        self.net = caffe.Net('deploy.prototxt', 'dss_model_released.caffemodel', caffe.TEST)

    def compute_saliency(self, image_path):
        img = cv2.imread(image_path)
        im = prepare_image(img)

        # shape for input (data blob is N x C x H x W), set data
        self.net.blobs['data'].reshape(1, *im.shape)
        self.net.blobs['data'].data[...] = im
        # run net and take argmax for prediction
        self.net.forward()
        out1 = self.net.blobs['sigmoid-dsn1'].data[0][0,:,:]
        out2 = self.net.blobs['sigmoid-dsn2'].data[0][0,:,:]
        out3 = self.net.blobs['sigmoid-dsn3'].data[0][0,:,:]
        out4 = self.net.blobs['sigmoid-dsn4'].data[0][0,:,:]
        out5 = self.net.blobs['sigmoid-dsn5'].data[0][0,:,:]
        out6 = self.net.blobs['sigmoid-dsn6'].data[0][0,:,:]
        fuse = self.net.blobs['sigmoid-fuse'].data[0][0,:,:]
        res = (out3 + out4 + out5 + fuse) / 4
        salmap = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)
        cv2.resize(salmap, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_LINEAR)*float(255)
        return salmap

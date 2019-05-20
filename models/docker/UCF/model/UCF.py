# Rewrite of test_saliency_dataset.m for AMULET in Python
# Modified by Yulia Kotseruba (yulia_k@cse.yorku.ca)
# Jul 2018

 # Original Code Author: Pingping Zhang
 # Email: jssxzhpp@gmail.com
 # Date: 8/8/2017
 # The code is based on the following paper in ICCV2017:
 # Title: Amulet: Aggregating Multi-level Convolutional Features for Salient Object Detection
 # Authors: Pingping Zhang, Dong Wang, Huchuan Lu*, Hongyu Wang and Xiang Ruan
#############################################################################################

import numpy as np
import os
import sys
import time
import cv2

caffe_root = 'caffe-sal/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

EPSILON = 1e-8
IMAGE_DIM = 448


def prepare_image(img):
	im = np.array(img, dtype=np.float32)
	im = cv2.resize(im, (IMAGE_DIM, IMAGE_DIM), interpolation = cv2.INTER_LINEAR)
	im = im[..., ::-1]
	im -= np.array((104.00698793,116.66876762,122.67891434))
	im = np.transpose(im, (2, 0, 1))
	return im

class UCF():
	def __init__(self):
		caffe.set_mode_gpu()
		# choose which GPU you want to use
		caffe.set_device(0)
		caffe.SGDSolver.display = 0
		# load net
		self.net = caffe.Net('deploy.prototxt', 'iiau_redfcn_saliency_iter_200000.caffemodel', caffe.TEST)

	def compute_saliency(self, image_path):
		img = cv2.imread(image_path, cv2.IMREAD_COLOR)
		im = prepare_image(img)

		# shape for input (data blob is N x C x H x W), set data
		self.net.blobs['data'].reshape(1, *im.shape)
		self.net.blobs['data'].data[...] = im
		# run net and take argmax for prediction
		res = self.net.forward()
		salmap = np.transpose(res['loss'][:, 1, :, :], (1, 2, 0))
		salmap = cv2.resize(salmap, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_LINEAR)*255 #multiply by 255 because smiler_tools/image_processing converts to uint8 which may result in precision loss
		
		return salmap
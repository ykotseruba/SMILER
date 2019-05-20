#!/usr/bin/env python

import os
import sys

from smiler_tools.runner import run_model

from UCF import UCF


def compute_saliency(image_path):
    return ucf.compute_saliency(image_path)

if __name__ == "__main__":
    ucf = UCF()
    run_model(compute_saliency)

#!/usr/bin/env python

import os
import sys

from smiler_tools.runner import run_model

from DSS import DSS


def compute_saliency(image_path):
    return dss.compute_saliency(image_path)

if __name__ == "__main__":
    dss = DSS()
    run_model(compute_saliency)
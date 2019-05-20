#!/usr/bin/env python

import os
import sys

from smiler_tools.runner import run_model

from DHS import DHS


def compute_saliency(image_path):
    return dhs.compute_saliency(image_path)

if __name__ == "__main__":
    dhs = DHS()
    run_model(compute_saliency)
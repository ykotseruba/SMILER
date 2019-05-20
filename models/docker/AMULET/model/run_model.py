#!/usr/bin/env python

import os
import sys

from smiler_tools.runner import run_model

from AMULET import AMULET


def compute_saliency(image_path):
    return amulet.compute_saliency(image_path)

if __name__ == "__main__":
    amulet = AMULET()
    run_model(compute_saliency)

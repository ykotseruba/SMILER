#!/usr/bin/env python

import os
import sys
import json

from smiler_tools.runner import run_model

from AMULET import AMULET



if __name__ == "__main__":
	options = json.loads(os.environ['SMILER_PARAMETER_MAP'])
	test_type_string = options.get('test_type', 'fusion')

	amulet = AMULET(test_type=test_type_string)

	def compute_saliency(image_path):
		return amulet.compute_saliency(image_path)

	run_model(compute_saliency)

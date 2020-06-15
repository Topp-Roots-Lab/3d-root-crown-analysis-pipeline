#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import re
import sys
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count, shared_memory
from pprint import pprint
from time import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from tqdm import tqdm

from __init__ import __version__
from raw_utils.raw_utils.core.convert.convert import find_float_range, scale
from raw_utils.raw_utils.core.metadata import determine_bit_depth, read_dat


def parse_options():
	parser = argparse.ArgumentParser(description='Generate CSV of descriptive statistics about each slice for a given volume as grayscale images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
	parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
	parser.add_argument("-t", "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
	parser.add_argument("-f", '--force', action="store_true", help="Force file creation. Overwrite any existing files.")
	parser.add_argument("path", metavar='PATH', type=str, nargs='+', help='Input directory to process')
	args = parser.parse_args()

	# Configure logging, stderr and file logs
	logging_level = logging.INFO
	if args.verbose:
		logging_level = logging.DEBUG

	lfp = f"{dt.today().strftime('%Y-%m-%d')}_{os.path.splitext(os.path.basename(__file__))[0]}.log"

	logFormatter = logging.Formatter("%(asctime)s - [%(levelname)-4.8s] - %(filename)s %(lineno)d - %(message)s")
	rootLogger = logging.getLogger()
	rootLogger.setLevel(logging.DEBUG)

	fileHandler = logging.FileHandler(lfp)
	fileHandler.setFormatter(logFormatter)
	fileHandler.setLevel(logging.DEBUG) # always show debug statements in log file
	rootLogger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(logFormatter)
	consoleHandler.setLevel(logging_level)
	rootLogger.addHandler(consoleHandler)

	# Make sure user does not request more CPUs can available
	if args.threads > cpu_count():
		args.threads = cpu_count()

	args.path = list(set(args.path)) # remove duplicates

	logging.info(f'Running {__file__} {__version__}')

	return args

def describe(args, image, output_path):
	index_pattern = r".*(?P<index>\d{4}).*"
	match = re.match(index_pattern, os.path.basename(image))
	if match is None:
		return None
	index = int(match.group('index'))
	# logging.info(image)
	img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
	df = img.ravel()
	mean = np.mean(df)
	median = int(np.median(df))
	median_count = np.count_nonzero(df == median)
	mode, mode_count = stats.mode(df)
	mode = mode[0]
	mode_count = mode_count[0]
	minimum = np.amin(df)
	minimum_count = np.count_nonzero(df == minimum)
	maximum = np.amax(df)
	maximum_count = np.count_nonzero(df == maximum)
	triangle_threshold_value, tri_thresholded_image = cv2.threshold(img, 127, 255, cv2.THRESH_TRIANGLE)
	ofp = os.path.join(output_path, f"tri_{os.path.basename(image)}")
	if (not os.path.exists(ofp) or sys.getsizeof(tri_thresholded_image) == os.path.getsize(ofp)):
		cv2.imwrite(ofp, tri_thresholded_image, [cv2.IMWRITE_PNG_BILEVEL, 1])
	else:
		logging.debug(f"File exists. Skipping '{ofp}'")
	triangle_threshold_value = int(triangle_threshold_value)
	otsu_threshold_value, otsu_thresholded_image = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
	ofp = os.path.join(output_path, f"otsu_{os.path.basename(image)}")
	if (not os.path.exists(ofp) or sys.getsizeof(otsu_thresholded_image) == os.path.getsize(ofp)):
		cv2.imwrite(ofp, otsu_thresholded_image, [cv2.IMWRITE_PNG_BILEVEL, 1])
	else:
		logging.debug(f"File exists. Skipping '{ofp}'")
	otsu_threshold_value = int(otsu_threshold_value)
	distance_from_median_to_mode_ceil = math.ceil(abs(median - mode))

	# Replace lower values with median to redo triangle method
	img[img < median] = median
	df = img.ravel()
	alt_mean = np.mean(df)
	alt_median = int(np.median(df))
	alt_median_count = np.count_nonzero(df == alt_median)
	alt_mode, alt_mode_count = stats.mode(df)
	alt_mode = alt_mode[0]
	alt_mode_count = alt_mode_count[0]
	alt_minimum = np.amin(df)
	alt_minimum_count = np.count_nonzero(df == alt_minimum)
	alt_maximum = np.amax(df)
	alt_maximum_count = np.count_nonzero(df == alt_maximum)
	alt_triangle_threshold_value, alt_tri_thresholded_image = cv2.threshold(img, 127, 255, cv2.THRESH_TRIANGLE)
	ofp = os.path.join(output_path, f"alt_tri_{os.path.basename(image)}")
	if (not os.path.exists(ofp) or sys.getsizeof(alt_tri_thresholded_image) == os.path.getsize(ofp)):
		cv2.imwrite(ofp, alt_tri_thresholded_image, [cv2.IMWRITE_PNG_BILEVEL, 1])
	else:
		logging.debug(f"File exists. Skipping '{ofp}'")
	alt_triangle_threshold_value = int(alt_triangle_threshold_value)

	return {
		"index": index,
		"filepath": image,
		"mean": mean,
		"median": median,
		"median_count": median_count,
		"mode": mode, 
		"mode_count": mode_count,
		"minimum": minimum,
		"minimum_count": minimum_count,
		"maximum": maximum,
		"maximum_count": maximum_count,
		"distance_from_median_to_mode_ceil": distance_from_median_to_mode_ceil,
		"triangle_threshold_value": triangle_threshold_value,
		"alt_triangle_threshold_value": alt_triangle_threshold_value,
		"otsu_threshold_value": otsu_threshold_value,
		"alt_mean": alt_mean,
		"alt_median": alt_median,
		"alt_median_count": alt_median_count,
		"alt_mode": alt_mode,
		"alt_mode": alt_mode,
		"alt_mode_count": alt_mode_count,
		"alt_minimum": alt_minimum,
		"alt_minimum_count": alt_minimum_count,
		"alt_maximum": alt_maximum,
		"alt_maximum_count": alt_maximum_count,
	}

if __name__ == "__main__":
	args = parse_options()
	start_time = time()

	# Gather all folders containing grayscale images
	# Case 1: Provided path is a single volume that contains images
	# Case 2: Provided path contains sub-directories that contain images
	# Case 3: Provided path 
	images = []
	for fp in args.path:
		# If a directory, extract the images from it as a list
		if os.path.isdir(fp):
			folder_path = os.path.realpath(fp)
			images.extend([ os.path.join(folder_path, filepath) for filepath in os.listdir(fp) if filepath.endswith('png') ])
		# Otherwise, if a PNG or TIFF (and not symlink)
		elif os.path.isfile(fp) and not os.path.islink(fp) and (fp.endswith('.png')):
			images.append(os.path.realpath(fp))


	for image in list(set([ os.path.dirname(image) for image in images ])):
		threshold_image_folder = os.path.join(image, f"{image}_thresholded_images")
		logging.info(f"Make '{threshold_image_folder}'")
		if not os.path.exists(threshold_image_folder):
			os.mkdir(threshold_image_folder)

	results = []
	def update(result):
		results.append(result)
		pbar.update()
	# Dedicate N CPUs for processing
	description = f"Processing {fp}"
	pbar = tqdm(images, desc=description)
	with Pool(args.threads) as p:
		# For each slice in the volume...
		for image in images:
			output_path = f"{os.path.dirname(image)}_thresholded_images"
			p.apply_async(describe, args=(args, image, output_path), callback=update)
		p.close()
		p.join()
	pbar.close()
	pbar = None

	df = pd.DataFrame(results)
	df.set_index(['index'])
	csv_fp = os.path.join(fp, f"{fp}.csv")
	print(csv_fp)
	df.to_csv(csv_fp, index=False)

	logging.debug(f'Total execution time: {time() - start_time} seconds')

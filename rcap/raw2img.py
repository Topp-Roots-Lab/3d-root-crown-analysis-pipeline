#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
'''
Created on Jul 27, 2018

@author: Ni Jiang, Tim Parker
'''

import argparse
import logging
import os
import re
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count
from time import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from __init__ import __version__
from raw_utils.raw_utils.core.convert.convert import find_float_range, scale
from raw_utils.raw_utils.core.metadata import determine_bit_depth, read_dat
from utils import configure_logging


def parse_options():
	parser = argparse.ArgumentParser(description='Convert .raw 3d volume file to typical image format slices',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
	parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
	parser.add_argument("-t", "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
	parser.add_argument("-f", '--force', action="store_true", help="Force file creation. Overwrite any existing files.")
	parser.add_argument("-n", '--dry-run', dest='dryrun', action="store_true", help="Perform a trial run. Do not create image files, but logs will be updated.")
	parser.add_argument("--format", default='png', help="Set image filetype. Availble options: ['png', 'tif']")
	parser.add_argument("path", metavar='PATH', type=str, nargs=1, help='Input directory to process')
	args = parser.parse_args()

	# Make sure user does not request more CPUs can available
	if args.threads > cpu_count():
		args.threads = cpu_count()

	# Change format to always be lowercase
	args.format = args.format.lower()
	args.path = list(set(args.path)) # remove any duplicates

	args.module_name = f"{os.path.splitext(os.path.basename(__file__))[0]}"
	configure_logging(args)
	if args.dryrun:
		logging.info(f"DRY-RUN MODE ENABLED")

	return args

def slice_to_img(args, slice, x, y, bitdepth, image_bitdepth, old_min, old_max, new_min, new_max, ofp):
	"""Convert byte data of a slice into an image

	Args:
		args (Namespace): arguments object
		slice (numpy.ndarray): slice data
		x (int): width of the slice as an image
		y (int): height of the slice as an image
		ofp (str): intended output path of the image to be saved

	"""
	slice = slice.reshape([y,x])

	if bitdepth != image_bitdepth:
		slice = scale(slice, old_min, old_max, new_min, new_max)
		slice = np.floor(slice)

	if not args.dryrun:
		Image.fromarray(slice.astype(image_bitdepth)).save(ofp)

def extract_slices(args, fp):
	"""Extract each slice of a volume, one by one and save it as an image

	Args:
		args (Namespace): arguments object

	"""
	def update(*args):
		pbar.update()

	# Set an images directory for the volume
	imgs_dir = os.path.splitext(fp)[0]
	logging.debug(f"Slices directory: '{imgs_dir}'")
	dat_fp = f"{os.path.splitext(fp)[0]}.dat"
	logging.debug(f"DAT filepath: '{dat_fp}'")

	# Create images directory if does not exist
	try:
		if not os.path.exists(imgs_dir):
			os.makedirs(imgs_dir)
		# else:
			# logging.warning(f"Output directory for slices already exists '{imgs_dir}'.")
	except:
		raise
	# Images directory is created and ready
	else:
		# Get dimensions of the volume
		metadata = read_dat(dat_fp)
		x, y, z = metadata['xdim'], metadata['ydim'], metadata['zdim']
		xth, yth, zth = metadata['x_thickness'], metadata['y_thickness'], metadata['z_thickness']
		logging.debug(f"Volume dimensions:  <{x}, {y}, {z}> for '{fp}'")
		logging.debug(f"Slice thicknesses:  <{xth}, {yth}, {zth}> for '{fp}'")

		bitdepth = determine_bit_depth(fp, (x,y,z))
		logging.debug(f"Detected bit depth '{bitdepth}' for '{fp}'")

		# Pad the index for the slice in its filename based on the
		# number of digits for the total count of slices
		digits = len(str(z))
		num_format = '{:0'+str(digits)+'d}'

		# Set slice dimensions
		img_size = x * y
		offset = img_size * np.dtype(bitdepth).itemsize

		# Determine scaling parameters per volume for output images
		# Equate the image format to numpy dtype
		if args.format == 'tif':
			image_bitdepth = 'uint16'
		elif args.format == 'png':
			image_bitdepth = 'uint8'
		else:
			image_bitdepth = 'uint8'

		# Construct transformation function
		# If input bitdepth is an integer, get the max and min with iinfo
		if np.issubdtype(np.dtype(bitdepth), np.integer):
			old_min = np.iinfo(np.dtype(bitdepth)).min
			old_max = np.iinfo(np.dtype(bitdepth)).max
		# Otherwise, assume float32 input
		else:
			old_min, old_max = find_float_range(fp, dtype=bitdepth, buffer_size=offset)
		# If output image bit depth is an integer, get the max and min with iinfo
		if np.issubdtype(np.dtype(image_bitdepth), np.integer):
			new_min = np.iinfo(np.dtype(image_bitdepth)).min
			new_max = np.iinfo(np.dtype(image_bitdepth)).max
		# Otherwise, assume float32 output
		else:
			new_min = np.finfo(np.dtype(image_bitdepth)).min
			new_max = np.finfo(np.dtype(image_bitdepth)).max

		logging.debug(f"{bitdepth} ({old_min}, {old_max}) -> {image_bitdepth} ({new_min}, {new_max})")

		# Extract data from volume, slice-by-slice
		slices = []

		description = f"Extracting slices from {os.path.basename(fp)} ({bitdepth})"
		pbar = tqdm(total = z, desc=description)
		with open(fp, 'rb') as ifp:
			# Dedicate N CPUs for processing
			with Pool(args.threads) as p:
				# For each slice in the volume...
				for i in range(0, z):
					# Read slice data, and set job data for each process
					ifp.seek(i*offset)
					chunk = np.fromfile(ifp, dtype=bitdepth, count = img_size, sep="")
					ofp = os.path.join(imgs_dir, f"{os.path.splitext(os.path.basename(fp))[0]}_{num_format.format(i)}.{args.format}")
					# Check if the image already exists
					if os.path.exists(ofp) and not args.force:
						pbar.update()
						continue
					p.apply_async(slice_to_img, args=(args, chunk, x, y, bitdepth, image_bitdepth, old_min, old_max, new_min, new_max, ofp), callback=update)
				p.close()
				p.join()
		pbar.close()
		pbar = None

if __name__ == "__main__":
	args = parse_options()
	start_time = time()

	# Collect all volumes and validate their metadata
	try:
		# Gather all files
		args.files = []
		for p in args.path:
			for root, dirs, files in os.walk(p):
				for filename in files:
					args.files.append(os.path.join(root, filename))

		# Append any loose, explicitly defined paths to .RAW files
		args.files.extend([ f for f in args.path if f.endswith('.raw') ])

		# Get all RAW files
		args.files = [ f for f in args.files if f.endswith('.raw') ]
		logging.debug(f"All files: {args.files}")
		args.files = list(set(args.files)) # remove duplicates
		logging.info(f"Found {len(args.files)} volume(s).")
		logging.debug(f"Unique files: {args.files}")

		# Validate that a DAT file exists for each volume
		for fp in args.files:
			dat_fp = f"{os.path.splitext(fp)[0]}.dat" # .DAT filepath
			logging.debug(f"Validating DAT file: '{dat_fp}'")
			# Try to extract the dimensions to make sure that the file exists
			read_dat(dat_fp)
	except Exception as err:
		logging.error(err)
	else:
		# For each provided volume...
		pbar = tqdm(total = len(args.files), desc=f"Overall progress")
		for fp in args.files:
			logging.debug(f"Processing '{fp}'")
			# Extract slices for all volumes in provided folder
			extract_slices(args, fp)
			pbar.update()
		pbar.close()

	logging.debug(f'Total execution time: {time() - start_time} seconds')

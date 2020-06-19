#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import shutil
import subprocess
import threading
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from time import time

from tqdm import tqdm

from __init__ import __version__
from utils import configure_logging


def parse_options():
	parser = argparse.ArgumentParser(description='Root Crowns Segmentation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
	parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
	parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
	parser.add_argument("-f", "--force", action="store_true", help="Force file creation. Overwrite any existing files.")
	parser.add_argument("-n", "--dry-run", dest='dryrun', action="store_true", help="*Not yet implemented.* Perform a trial run. Do not create image files, but logs will be updated.")
	parser.add_argument("--progress", action="store_true", help="Enables multiple progress bar, one for each volume during processing.")
	parser.add_argument('--soil', action='store_true', help="Extract any soil during segmentation.")
	parser.add_argument('-s', "--sampling", help="resolution parameter", default=2)
	parser.add_argument("path", metavar='PATH', type=str, nargs=1, help='Input directory to process')
	args = parser.parse_args()

	# Make sure user does not request more CPUs can available
	if args.threads > cpu_count():
		args.threads = cpu_count()

	args.module_name = f"{os.path.splitext(os.path.basename(__file__))[0]}"
	configure_logging(args, ifp=os.path.basename(os.path.realpath(args.path[0])))
	if args.dryrun:
		logging.info(f"DRY-RUN MODE ENABLED")

	# Recode soil input to match the input of rootCrownSegmentation binary
	args.soil = 1 if args.soil else 0

	# Disable progress bars if verbose mode enabled
	if args.verbose:
		args.progress = False

	return args

# Async function to call rootCrownSegmentation binary
def run(cmd, args, lock, position):
	# Set up values for progress bar
	volume_name = os.path.basename(os.path.normpath(cmd[2]))
	text = f"Segmenting '{volume_name}'"
	slice_count = round(len([ img for img in os.listdir(fp) if img.endswith('png')  ]) / args.sampling)

	# Adjust command so that spaces are escaped
	# Requires that the input to Popen be a string instead of list
	adjusted_cmd = []
	for argument in cmd:
		if not argument.isnumeric():
			adjusted_cmd.append(f'"{argument}"')
		else:
			adjusted_cmd.append(argument)
	logging.debug(f"Run command: '{' '.join(adjusted_cmd)}'")

	if args.progress:
		with lock:
			progress_bar = tqdm(total = slice_count, desc=text, position=position, leave=False, unit="image")

	# Cast argument list to string for Popen to escape
	adjusted_cmd = ' '.join(adjusted_cmd)
	# Start processing
	with subprocess.Popen(adjusted_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:
		# Parse stdout from subprocess
		complete = False
		for line in iter(p.stdout.readline, b''):
			if complete:
				break
			if line != '':
				logging.debug(line.strip())
				if volume_name in line and line.startswith('Write'):
					if args.progress:
						with lock:
							progress_bar.update()
				if "Exiting" in line or "Abort" in line:
					complete = True

		# Report any run-time errors
		if p.returncode is not None and p.returncode > 0:
			logging.error(f"Error encountered. 'rootCrownSegmentation' returned {p.returncode}")

		# Clean up progress bar for volume
		if args.progress:
			with lock:
				progress_bar.close()
				progress_bar = None

if __name__ == "__main__":
	args = parse_options()
	start_time = time()

	# Clean up input folders
	args.path = [ os.path.realpath(fp) for fp in args.path ]
	args.path = list(set(args.path)) # remove duplicates

	# Count the number of sub-folders (i.e., volumes as PNG slices) per input folder
	input_paths = []
	for fp in args.path:
		# Check if the input folder exists
		if not os.path.exists(fp):
			raise FileNotFoundError(fp)
		for root, dirs, files in os.walk(fp):
			input_paths += [ os.path.join(fp, f) for f in dirs ]

	args.path = list(set(input_paths)) # remove duplicates

	logging.info(f"Found {len(args.path)} volume(s).")
	logging.debug(f"Processing {len(args.path)} volume(s): {args.path}")

	# Create threshold and model folders
	for fp in set([ os.path.dirname(path) for path in args.path ]):
		thresholded_folder = f"{fp}_thresholded_images"
		model_folder = f"{fp}_3d_models"

		if not os.path.exists(thresholded_folder):
			os.makedirs(thresholded_folder)
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

	# For each provided input folder, build a command for each volume
	cmd_list = []
	for fp in args.path:
		thresholded_folder = f"{os.path.dirname(fp)}_thresholded_images"
		model_folder = f"{os.path.dirname(fp)}_3d_models"

		# Create paths to the output files
		volume_name = os.path.basename(fp)
		ofp         = os.path.join(thresholded_folder, os.path.basename(fp))    # output folder for segmented images
		out_fp      = os.path.join(model_folder, f"{volume_name}.out")          # root system .OUT file
		obj_fp      = os.path.join(model_folder, f"{volume_name}.obj")          # root system .OBJ file
		soil_out_fp = os.path.join(model_folder, f"{volume_name}_soil.out")     # dirt .OUT file
		soil_obj_fp = os.path.join(model_folder, f"{volume_name}_soil.obj")     # dirt .OBJ file

		# Create the sub-directory for the set of thresholded images per volume
		if not os.path.exists(ofp):
			os.makedirs(ofp)

		binary_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'bin', 'rootCrownSegmentation')
		cmd = [binary_filepath, str(args.soil), f'{fp}/', str(args.sampling), f'{ofp}/', f'{out_fp}', f'{obj_fp}']
		if args.soil == 1:
			cmd += [soil_out_fp, soil_obj_fp]
		cmd_list.append(cmd)

	# Process data
	# Dedicate N CPUs for processing
	with ThreadPool(args.threads) as p:
		lock = threading.Lock()
		# Create overall progress bar
		if args.progress:
			progress_text = "Overall progress"
		else:
			progress_text = f"Processing {os.path.dirname(args.path[0])}"
		if not args.verbose:
			pbar = tqdm(total = len(cmd_list), position = 0, desc=progress_text, leave=True, unit="volume")
		def pbar_update(*response):
			if not args.verbose:
				pbar.update()
		def subprocess_error_callback(*response):
			logging.error(args)

		# For each slice in the volume...
		for i, cmd in enumerate(cmd_list, start = 1):
			# Run command as separate process
			p.apply_async(run, args=(cmd, args, lock, i), callback=pbar_update, error_callback=subprocess_error_callback)
		p.close()
		p.join()
		# Close progress bar
		pbar.close()

	logging.info(f'Total execution time: {time() - start_time} seconds')

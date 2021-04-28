#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
import logging
import os
import shutil
import subprocess
import threading
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pprint import pformat, pprint

import pandas as pd
from tqdm import tqdm


# Async function to call rootCrownSegmentation binary
def run(cmd, args, lock, position):
	# Set up values for progress bar
	volume_name = os.path.basename(os.path.normpath(cmd[2]))
	text = f"Segmenting '{volume_name}'"
	fp = os.path.normpath(cmd[2])
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

def csv(*args, **kwargs):
	"""
	TODO
	"""
	logging.info(kwargs)
	csv_fp = kwargs["csv"] # relative or absolute paths in a csv
	path = kwargs["path"] # search path for data
	logging.info(f"Processing files in '{path}'")

	try:
		# Load file list and bounds
		df = pd.read_csv(csv_fp)
		print(df)
		logging.info(f"{df.columns=}")

		# Check that each file exists
		fpaths = df["Predicted Grayscale Image Folder"]
		pprint(fpaths)
		df["path_exists"] = fpaths.apply(lambda x: os.path.exists(x) and os.path.isdir(x))

		dff = df[["UID", "Predicted Grayscale Image Folder", "Lower Bound Threshold Value (uint8)", "Upper Bound Threshold Value (uint8)", "path_exists"]]
		validated_df = dff[dff["path_exists"]]
		pprint(validated_df)

		# Count the number of existing scans
		logging.info(f"Found {len(validated_df)} volume(s).")
		logging.debug(validated_df)

		validated_scans = list(set(validated_df["Predicted Grayscale Image Folder"]))
		logging.info(path)

		# # Create threshold and model folders
		# for fp in set([ os.path.dirname(path) for path in validated_scans ]):
		# 	thresholded_folder = f"{fp}_thresholded_images"
		# 	model_folder = f"{fp}_3d_models"

		# 	if not os.path.exists(thresholded_folder):
		# 		logging.debug(f"Creating '{thresholded_folder=}'")
		# 		if not kwargs["dryrun"]:
		# 			os.makedirs(thresholded_folder)
		# 	if not os.path.exists(model_folder):
		# 		logging.debug(f"Creating '{model_folder=}'")
		# 		if not kwargs["dryrun"]:
		# 			os.makedirs(model_folder)
		
		# # For each validated volume, build a command to process it
		# for fp in args.path:
		# 	thresholded_folder = f"{os.path.dirname(fp)}_thresholded_images"
		# 	model_folder = f"{os.path.dirname(fp)}_3d_models"

		# 	# Create paths to the output files
		# 	volume_name = os.path.basename(fp)
		# 	ofp         = os.path.join(thresholded_folder, os.path.basename(fp))    # output folder for segmented images
		# 	out_fp      = os.path.join(model_folder, f"{volume_name}.out")          # root system .OUT file
		# 	obj_fp      = os.path.join(model_folder, f"{volume_name}.obj")          # root system .OBJ file
		# 	soil_out_fp = os.path.join(model_folder, f"{volume_name}_soil.out")     # dirt .OUT file
		# 	soil_obj_fp = os.path.join(model_folder, f"{volume_name}_soil.obj")     # dirt .OBJ file

		# 	# Create the sub-directory for the set of thresholded images per volume
		# 	if not os.path.exists(ofp):
		# 		os.makedirs(ofp)

		# 	binary_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib', 'rootCrownSegmentation')
		# 	cmd = [binary_filepath, str(args.soil), f'{fp}/', str(args.sampling), f'{ofp}/', f'{out_fp}', f'{obj_fp}']
		# 	if args.soil == 1:
		# 		cmd += [soil_out_fp, soil_obj_fp]
		# 	cmd_list.append(cmd)
	except Exception as err:
		raise err
	return 0

def main(args):
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

		binary_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib', 'rootCrownSegmentation')
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
			logging.error(response)

		# For each slice in the volume...
		for i, cmd in enumerate(cmd_list, start = 1):
			# Run command as separate process
			p.apply_async(run, args=(cmd, args, lock, i), callback=pbar_update, error_callback=subprocess_error_callback)
		p.close()
		p.join()
		if not args.verbose:
			# Close progress bar
			pbar.close()

	return 0

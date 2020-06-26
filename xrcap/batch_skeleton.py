#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
import logging
import os
import re
import shutil
import subprocess
import threading
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from pprint import pformat

from tqdm import tqdm


def process(ifp, ofp, scale, cmd, lock):
	cmd = [cmd, ifp, ofp, str(scale)]
	logging.debug(f"Run command: '{' '.join(cmd)}'")

	# Start processing
	with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:

		# volume_load_pbar = None
		# load_data_pattern = r"Reading\s+(?P<nvoxels>\d+).*" + ifp + ".*"
		# Parse stdout from subprocess
		complete = False
		for line in iter(p.stdout.readline, b''):
			if complete:
				break
			if line.strip() != '':
				logging.debug(f"{os.path.basename(ifp)} - {line.strip()}")
				# Check if a new point cloud dataset is being loaded into memory
				# load_data_match = re.match(load_data_pattern, line)
				# if load_data_match is not None and "nvoxels" in load_data_match.groupdict():
					# nvoxels = int(load_data_match.group("nvoxels"))
					# with lock:
					# 	if volume_load_pbar is None:
					# 		volume_load_pbar = tqdm(total=nvoxels, desc=f"Loading '{ifp}'")
				
				# # Check progress on volume loading
				# if volume_load_pbar is not None and ifp in line and "Read line" in line:
				# 	volume_load_pbar.update()
				if "Exiting" in line or "Abort" in line:
					complete = True

		if p.returncode is not None and p.returncode > 0:
			logging.error(f"Error encountered. 'Skeleton' returned {p.returncode}")

		# with lock:
		# 	if volume_load_pbar is not None:
		# 		volume_load_pbar.close()


def wrl2ctm(meshlabserver, ifp):
	ofp = "".join([os.path.splitext(ifp)[0], ".ctm"])
	cmd = [meshlabserver, '-i', ifp, '-o', ofp]
	logging.debug(' '.join(cmd))

	p = subprocess.run(cmd, check=True, capture_output=True, text=True)
	logging.debug(p.stdout)

	if p.returncode is not None and p.returncode > 0:
		logging.error(f"Error encountered. 'Meshlab' returned {p.returncode}")


def main(args):

	# Clean up input folders
	args.path = [ os.path.realpath(fp) for fp in args.path ]
	args.path = list(set(args.path)) # remove duplicates

	# Count the number of sub-folders (i.e., volumes as PNG slices) per input folder
	files = []
	for fp in args.path:
		# Check if the input folder exists
		if not os.path.exists(fp):
			raise FileNotFoundError(fp)
		# Append all found .out files to files list
		files += [os.path.join(fp, out) for out in os.listdir(fp) if out.endswith(".out")]
	args.path = list(set(files))

	logging.info(f"Found {len(args.path)} volume(s).")
	logging.debug(f"Processing {len(args.path)} volume(s): {pformat(args.path)}")

	# Remove previous features.tsv files
	for fp in list(set([ os.path.dirname(p) for p in args.path ])):
		features = ['Name','SurfArea','Volume','Convex_Volume','Solidity','MedR','MaxR','Bushiness','Depth','HorEqDiameter','TotalLength','SRL','Length_Distr','W_D_ratio','Number_bif_cl','Av_size_bif_cl','Edge_num','Av_Edge_length','number_tips','volume','surface_area','av_radius']
		fname = os.path.join(fp, "features.tsv")
		with open(fname, 'w') as ofp:
			ofp.write('\t'.join(features))

	# Create overall progress bar
	pbar = tqdm(total = len(args.path), desc=f"Overall progress")
	def pbar_update(*args):
		pbar.update()

	def subprocess_error_callback(*args):
		logging.error(args)

	# Process data
	# Dedicate N CPUs for processing
	lock = threading.Lock()
	with ThreadPool(args.threads) as p:
		# For each slice in the volume...
		binary_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib', 'Skeleton')
		for fp in args.path:
			# Read object files, and then spawn a child process per volume
			ofp = os.path.join(os.path.dirname(fp), "features.tsv")
			p.apply_async(process, args=(fp, ofp, args.scale, binary_filepath, lock), callback=pbar_update, error_callback=subprocess_error_callback)
		p.close()
		p.join()
	pbar.close()
	pbar = None

	# Convert produced WRL to CTM
	# Check if Meshlabserver is available
	meshlabserver = shutil.which("meshlabserver")
	if meshlabserver is not None:
		wrl_files = [os.path.join(os.path.dirname(args.path[0]),f) for f in os.listdir(os.path.dirname(args.path[0])) if f.endswith('.wrl')]
		pbar = tqdm(total = len(wrl_files), desc="Converting WRL to CTM with Meshlab")
		def conversion_callback(*args):
			if pbar is not None:
				pbar.update()

		def conversion_error_callback(*args):
			logging.error(args)

		with Pool(args.threads) as p:
			for fp in wrl_files:
				p.apply_async(wrl2ctm, args=(meshlabserver, fp), callback=conversion_callback, error_callback=conversion_error_callback)
			p.close()
			p.join()
		
		pbar.close()
		pbar = None
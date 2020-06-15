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
from pprint import pformat
from time import time
import re

from tqdm import tqdm

from __init__ import __version__
from utils import configure_logging


def parse_options():
	parser = argparse.ArgumentParser(description='Root Crowns Feature Extraction',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")
	parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
	parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
	parser.add_argument("-f", '--force', action="store_true", help="Force file creation. Overwrite any existing files.")
	parser.add_argument("-n", '--dry-run', dest='dryrun', action="store_true", help="Perform a trial run. Do not create image files, but logs will be updated.")
	parser.add_argument('-s', "--scale", help="The scale parameter using for skeleton.", default=2.25)
	parser.add_argument("path", metavar='PATH', type=str, nargs=1, help='Input directory to process')
	args = parser.parse_args()

	# Make sure user does not request more CPUs can available
	if args.threads > cpu_count():
		args.threads = cpu_count()

	args.module_name = f"{os.path.splitext(os.path.basename(__file__))[0]}"
	configure_logging(args)
	if args.dryrun:
		logging.info(f"DRY-RUN MODE ENABLED")

	return args

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


if __name__ == "__main__":
	args = parse_options()
	start_time = time()

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
		binary_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'bin', 'Skeleton')
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
		pbar = tqdm(total = len(wrl_files), desc="Converting all WRL to CTM with Meshlab")
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

	logging.debug(f'Total execution time: {time() - start_time} seconds')

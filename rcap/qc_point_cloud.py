#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import re
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from __init__ import __version__


def parse_options():
  parser = argparse.ArgumentParser(description='Create a downsampled version of point cloud data (.obj) based on a random selection of which points are kept.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
  parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
  parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
  parser.add_argument("-f", "--force", action="store_true", help="Force file creation. Overwrite any existing files.")
  parser.add_argument("-p", "--probability", type=float, default=0.05, help="Probability that a point will be kept ")
  parser.add_argument("-s", "--seed", type=int, help="Set seed for random sampling.")
  parser.add_argument("path", metavar='PATH', type=str, nargs='+', help='Input directory to process. Must contain folder with thresholded images.')
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
  # rootLogger.addHandler(fileHandler)

  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  consoleHandler.setLevel(logging_level)
  rootLogger.addHandler(consoleHandler)

  # Make sure user does not request more CPUs can available
  if args.threads > cpu_count():
    args.threads = cpu_count()

  # Make sure probability is between 0 and 1
  if args.probability < 0 or args.probability > 1.0:
    raise ValueError(f"User-defined probability is invalid: '{args.probability}'. It must be between 0 and 1.0.")

  logging.debug(f'Running {__file__} {__version__}\n')

  return args

def probabilistic_downsampling(args, fp, op, probability):
  kept_point_count = 0
  with open(fp, 'r') as ifp, open(op, 'w') as ofp:
    ofp.write(f"# {op}\n")
    ofp.write(f"# created at {dt.today().isoformat()}\n")
    ofp.write(f"# generated by {__file__} {__version__}\n")
    ofp.write(f"# data source: {fp}\n")
    ofp.write(f"# random sampling probability: {str(probability)}\n")
    pt = ifp.readline()
    while pt:
      l = random.random()
      seed = l
      if (l < probability):
        ofp.write(pt)
        kept_point_count += 1
      pt = ifp.readline()
  
  logging.debug(f"Downsampled PCD volume: '{op}' ({kept_point_count} preserved points)")

if __name__ == "__main__":
  args = parse_options()
  try:
    # Gather all files
    args.files = []
    for p in args.path:
      for root, dirs, files in os.walk(p):
        for filename in files:
          args.files.append(os.path.join(root, filename))

  except Exception as err:
    logging.error(err)
    raise
  else:
    # Callback method for updating overall progress bar during multiprocessing
    def update(*args):
      """Update progress bar for an async process call"""
      pbar.update()

    # Filter out all files but OBJ
    args.files = [ f for f in args.files if f.endswith('.obj') and re.match(r".*qc\-[\d\.]+.obj", f) is None ]
    args.path = list(set(args.files)) # remove duplicates

    logging.info(f"Found {len(args.path)} volume(s).")
    logging.debug(f"Unique files: {args.path}")

    if len(args.path) == 1:
      logging.warning(f"No volumes supplied.")
    else:
      pbar = tqdm(total = len(args.path), desc=f"Downsampling point cloud data (OBJ)")
      # Dedicate N CPUs for processing
      with Pool(args.threads) as p:
        # For each slice in the volume...
        for fp in args.path:
          # Downsample point cloud data based on a probability to keep each point
          op = f"{os.path.splitext(fp)[0]}.qc-{str(args.probability)}.obj" # output filepath
          p.apply_async(probabilistic_downsampling, args=(args, fp, op, args.probability), callback=update)
        p.close()
        p.join()
      pbar.close()
      pbar = None
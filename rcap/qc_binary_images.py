#!/usr/bin/python3.8
# -*- coding: utf-8 -*-


import argparse
import logging
import os
from datetime import datetime as dt
from pprint import pformat

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from __init__ import __version__


def parse_options():
  parser = argparse.ArgumentParser(description='Check tresholded images for pure white slices. Creates CSV of volumes that have more than a given percentage of white pixels.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
  parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
  parser.add_argument("-c", "--cutoff", type=float, default=0.8, help="The minimum percentage of white pixels for a given slice for it to be flagged as invalid.")
  parser.add_argument("-l", "--list", action="store_true", help="Output TXT files that lists bad binary images produced by segmentation")
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
  rootLogger.addHandler(fileHandler)

  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  consoleHandler.setLevel(logging_level)
  rootLogger.addHandler(consoleHandler)

  logging.debug(f'Running {__file__} {__version__}')

  return args


def find_white_slices(fp, cutoff):
  files = sorted([ os.path.join(fp, f) for f in os.listdir(fp) if f.endswith('png') ])
  logging.debug(pformat(files))
  logging.debug(os.listdir(fp))
  logging.debug(f"Thresholded images: {pformat(files)}")

  if len(files) == 0:
    raise Exception(f"Directory does not contain images: '{fp}'")

  flagged_images = []
  flagged_indexes = []
  for f in tqdm(files, total=len(files), desc=f"Processing '{fp}'"):
    i = int(os.path.splitext(f)[0].split('_')[-1])
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    white_ratio = (np.sum(img == 255) / img.size)
    if white_ratio > cutoff:
      flagged_images.append(f)
      flagged_indexes.append(i)
  flagged_images = sorted(flagged_images)
  if len(flagged_indexes) > 0:
    return min(flagged_indexes), max(flagged_indexes), flagged_images
  return None, None, None

if __name__ == "__main__":
  args = parse_options()
  # Collect all volumes and validate their metadata
  try:
    # Gather all files
    args.path = [ os.path.realpath(p) for p in args.path if os.path.isdir(p) ]
    args.path = list(set(args.path)) # remove duplicates
    for fp in args.path:
      if not os.path.isdir(fp):
        raise NotADirectoryError(fp)
      
  except NotADirectoryError as err:
    logging.error(f"Folder containing thresholded images not found for volume: '{str(err).replace('_thresholded_images', '')}'")
  except Exception as err:
    logging.error(err)
    raise
  else:
    if len(args.path) > 0:
      # For each provided directory...
      failed_volumes = []
      passed_volumes = []
      # Get the sub-directories that contain binary images
      binary_image_folders = []
      for parent_path in args.path:
        binary_image_folders.extend( [ os.path.join(parent_path, vp) for vp in os.listdir(parent_path) ] )
      binary_image_folders = [ vp for vp in binary_image_folders if os.path.isdir(vp) ]
      for fp in tqdm(binary_image_folders, desc=f"Overall progress"):
        logging.debug(f"Processing '{fp}'")
        # Extract slices for all volumes in provided folder
        start, end, flagged_slices = find_white_slices(fp, args.cutoff)
        if start is not None and end is not None and flagged_slices is not None:
          failed_volumes.append((fp, start, end))
          flagged_slice_op = f"{os.path.basename(fp)}.flagged_slices.txt"
          with open(flagged_slice_op, 'w') as ofp:
            ofp.write('\n'.join(flagged_slices))
        else:
          passed_volumes.append(fp)

      if set(passed_volumes) == set(binary_image_folders):
        logging.info(f"All volumes pass!")
      else:
        logging.debug(failed_volumes)
        df = pd.DataFrame.from_records(failed_volumes, columns = [ 'path', 'start', 'end' ])
        df['volume_name'] = df['path'].apply(os.path.basename)
        ofp = f"{dt.today().strftime('%Y-%m-%d_%H-%M-%S')}_{os.path.splitext(os.path.basename(__file__))[0]}.flagged_volumes.csv"
        df.to_csv(ofp, index=False)
        logging.debug(pformat(df))
        logging.info(f"Detected possible incorrect segmentation. Check volumes listed in '{ofp}' for details.")
    else:
      logging.info(f"No volumes supplied.")

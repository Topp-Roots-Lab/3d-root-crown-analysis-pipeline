#!/usr/bin/python3
# -*- coding: utf-8 -*-


import argparse
import logging
import os
from datetime import datetime as dt

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from __init__ import __version__


def parse_options():
  parser = argparse.ArgumentParser(description='Check tresholded images for pure white slices. Creates CSV of volumes that have more than 90% white pixels.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
  parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
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

  logging.debug(f'Running {__file__} {__version__}')

  return args


def find_white_slices(fp, cutoff = 0.9):
  files = sorted([ os.path.join(fp, f) for f in os.listdir(fp) if f.endswith('png') ])
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
    return min(flagged_indexes), max(flagged_indexes)
  return None, None

if __name__ == "__main__":
  args = parse_options()
  # Collect all volumes and validate their metadata
  try:
    # Gather all files
    args.path = [ f'{p}_thresholded_images' for p in args.path if os.path.isdir(p) ]
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
      flagged_files = []
      volume_paths = []
      for parent_path in args.path:
        volume_paths.extend( [ os.path.join(parent_path, vp) for vp in os.listdir(parent_path) ] )
      volume_paths = [ vp for vp in volume_paths if os.path.isdir(vp) ]
      for fp in tqdm(volume_paths, desc=f"Overall progress"):
        logging.debug(f"Processing '{fp}'")
        # Extract slices for all volumes in provided folder
        start, end = find_white_slices(fp)
        if start is not None and end is not None:
          flagged_files.append((fp, start, end))

      if len(flagged_files) > 0:
        logging.debug(flagged_files)
        df = pd.DataFrame.from_records(flagged_files, columns = [ 'path', 'start', 'end' ])
        df['volume_name'] = df['path'].apply(os.path.basename)
        ofp = f"{dt.today().strftime('%Y-%m-%d_%H-%M-%S')}_{os.path.splitext(os.path.basename(__file__))[0]}.flagged_volumes.csv"
        df.to_csv(ofp, index=False)
        logging.info(f"Created '{ofp}'")
      else:
        logging.info(f"All volumes look good!")

    else:
      logging.info(f"No volumes supplied.")
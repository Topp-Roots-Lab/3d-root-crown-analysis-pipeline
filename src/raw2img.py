#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Jul 27, 2018

@author: Ni Jiang, Tim Parker
'''

import argparse
import logging
import os
from datetime import datetime as dt
from multiprocessing import Pool, Value, cpu_count

import numpy as np
from PIL import Image
from tqdm import tqdm


def options():
    parser = argparse.ArgumentParser(description='Convert .raw 3d volume file to typical image format slices',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version='%(prog)s 1.1.0')
    parser.add_argument('-i', "--input_folder", action="store_true", help="Deprecated. Data folder.") # left in for backwards compatibility
    parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
    parser.add_argument('--force', action="store_true", help="Force file creation. Overwrite any existing files.")
    parser.add_argument('-f', "--format", default='png', help="Set image filetype. Availble options: ['png', 'tif']")
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

    return args

def get_volume_dimensions(args, fp):
    """Get the x, y, z dimensions of a volume.

    Args:
        args (Namespace): arguments object
        fp (str): .DAT filepath

    Returns:
        (int, int, int): x, y, z dimensions of volume as a tuple

    """
    with open(fp, 'r') as ifp:
        line = ifp.readlines()[1]
        dims = [int(s) for s in line.split() if s.isdigit()]
        if not dims or len(dims) != 3:
            raise Exception(f"Unable to extract dimensions from DAT file: '{fp}'. Found dimensions: '{dims}'.")
        return dims

def slice_to_img(args, slice, x, y, ofp):
    """Convert byte data of a slice into an image

    Args:
        args (Namespace): arguments object
        slice (numpy.ndarray): slice data
        x (int): width of the slice as an image
        y (int): height of the slice as an image
        ofp (str): intended output path of the image to be saved

    """
    slice = slice.reshape([y,x])
    if args.format == 'tif':
        datatype = 'uint16'
    elif args.format == 'png':
        slice = np.floor(slice * float((2 ** 8) - 1) / float((2 ** 16) - 1))
        datatype = 'uint8'
    else:
        datatype = 'uint8'

    Image.fromarray(slice.astype(datatype)).save(ofp)

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
        x, y, z = get_volume_dimensions(args, dat_fp)
        logging.debug(f"Volume dimensions:  <{x}, {y}, {z}>")

        # Pad the index for the slice in its filename based on the
        # number of digits for the total count of slices
        digits = len(str(z))
        num_format = '{:0'+str(digits)+'d}'

        # Set slice dimensions
        img_size = x * y
        offset = img_size * np.dtype('uint16').itemsize

        # Extract data from volume, slice-by-slice
        slices = []

        pbar = tqdm(total = z, desc=f"Extracting slices from {os.path.basename(fp)}")
        with open(fp, 'rb') as f_data:
            # Dedicate N CPUs for processing
            with Pool(args.threads) as p:
                # For each slice in the volume...
                for i in range(0, z):
                    # Read slice data, and set job data for each process
                    f_data.seek(i*offset)
                    chunk = np.fromfile(f_data, dtype='uint16', count = img_size, sep="")
                    ofp = os.path.join(imgs_dir, f"{os.path.splitext(os.path.basename(fp))[0]}_{num_format.format(i)}.{args.format}")
                    # Check if the image already exists
                    if os.path.exists(ofp) and not args.force:
                        continue
                    # Process each slice of the volume across N processes
                    p.apply_async(slice_to_img, args=(args, chunk, x, y, ofp), callback=update)
                p.close()
                p.join()
        pbar.close()
        pbar = None

if __name__ == "__main__":
    args = options()

    # Change format to always be lowercase
    args.format = args.format.lower()
    args.path = list(set(args.path)) # remove any duplicates

    # Collect all volumes and validate their metadata
    try:
        # Gather all files
        args.files = []
        for p in args.path:
            for root, dirs, files in os.walk(p):
                for filename in files:
                    args.files.append(os.path.join(root, filename))

        # Get all RAW files
        args.files = [ f for f in args.files if os.path.splitext(f)[1] == '.raw' ]
        logging.debug(f"All files: {args.files}")
        args.files = list(set(args.files)) # remove duplicates
        logging.info(f"Found {len(args.files)} volume(s).")
        logging.debug(f"Unique files: {args.files}")

        # Validate that a DAT file exists for each volume
        for fp in args.files:
            dat_fp = f"{os.path.splitext(fp)[0]}.dat" # .DAT filepath
            logging.debug(f"Validating DAT file: '{dat_fp}'")
            # Try to extract the dimensions to make sure that the file exists
            get_volume_dimensions(args, dat_fp)
    except Exception as err:
        logging.error(err)
    else:
        pass
        # For each provided directory...
        for fp in tqdm(args.files, desc=f"Overall progress"):
            logging.debug(f"Processing '{fp}'")
            # Extract slices for all volumes in provided folder
            extract_slices(args, fp)

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
from functools import reduce
from multiprocessing import Pool, cpu_count
from time import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from __init__ import __version__

prod = lambda x,y: x * y

def options():
    parser = argparse.ArgumentParser(description='Convert .raw 3d volume file to typical image format slices',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
    parser.add_argument("-i", "--input_folder", action="store_true", help="Deprecated. Data folder.") # left in for backwards compatibility
    parser.add_argument("-t", "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
    parser.add_argument("-f", '--force', action="store_true", help="Force file creation. Overwrite any existing files.")
    parser.add_argument("--format", default='png', help="Set image filetype. Availble options: ['png', 'tif']")
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

    # Change format to always be lowercase
    args.format = args.format.lower()
    args.path = list(set(args.path)) # remove any duplicates

    logging.debug(f'Running {__file__} {__version__}')

    return args

def determine_bit_depth(fp, dims, resolutions):
    """Determine the bit depth of a .RAW based on its dimensions and slick thickness (i.e., resolution)

    Args:
        fp (str): file path to .RAW
        dims (x, y, z): dimensions of .RAW extracted
        resolutions (xth, yth, zth): thickness of each slice for each dimension

    Returns:
        str: numpy dtype encoding of bit depth
    """
    file_size = os.stat(fp).st_size
    minimum_size = reduce(prod, dims) # get product of dimensions
    logging.debug(f"Minimum calculated size of '{fp}' is {minimum_size} bytes")
    if file_size == minimum_size:
        return 'uint8'
    elif file_size == minimum_size * 2:
        return 'uint16'
    elif file_size == minimum_size * 4:
        return 'float32'
    else:
        if file_size < minimum_size:
            logging.warning(f"Detected possible data corruption. File is smaller than expected '{fp}'. Expected at <{file_size * 2}> bytes but found <{file_size}> bytes. Defaulting to unsigned 16-bit.")
            return 'uint16'
        else:
            logging.warning(f"Unable to determine bit-depth of volume '{fp}'. Expected at <{file_size * 2}> bytes but found <{file_size}> bytes. Defaulting to unsigned 16-bit.")
            return 'uint16'

def get_volume_dimensions(args, fp):
    """Get the x, y, z dimensions of a volume.

    Args:
        args (Namespace): arguments object
        fp (str): .DAT filepath

    Returns:
        (int, int, int): x, y, z dimensions of volume as a tuple

    """
    with open(fp, 'r') as ifp:
        for line in ifp.readlines():
            # logging.debug(line.strip())
            pattern_old = r'\s+<Resolution X="(?P<x>\d+)"\s+Y="(?P<y>\d+)"\s+Z="(?P<z>\d+)"'
            pattern = r'Resolution\:\s+(?P<x>\d+)\s+(?P<y>\d+)\s+(?P<z>\d+)'

            # See if the DAT file is the newer version
            match = re.match(pattern, line, flags=re.IGNORECASE)
            # Otherwise, check the old version (XML)
            if match is None:
                match = re.match(pattern_old, line, flags=re.IGNORECASE)
                if match is not None:
                    logging.debug(f"XML format detected for '{fp}'")
                    break
            else:
                logging.debug(f"Text/plain format detected for '{fp}'")
                break

        if match is not None:
            logging.debug(f"Match: {match}")
            dims = [ match.group('x'), match.group('y'), match.group('z') ]
            dims = [ int(d) for d in dims ]

            # Found the wrong number of dimensions
            if not dims or len(dims) != 3:
                raise Exception(f"Unable to extract dimensions from DAT file: '{fp}'. Found dimensions: '{dims}'.")
            return dims
        else:
            raise Exception(f"Unable to extract dimensions from DAT file: '{fp}'.")

def get_volume_slice_thickness(args, fp):
    """Get the x, y, z dimensions of a volume.

    Args:
        args (Namespace): arguments object
        fp (str): .DAT filepath

    Returns:
        (int, int, int): x, y, z real-world thickness in mm

    """
    with open(fp, 'r') as ifp:
        for line in ifp.readlines():
            # logging.debug(line.strip())
            pattern = r'\w+\:\s+(?P<xth>\d+\.\d+)\s+(?P<yth>\d+\.\d+)\s+(?P<zth>\d+\.\d+)'
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if match is None:
                continue
            else:
                logging.debug(f"Match: {match}")
                df = match.groupdict()
                dims = [ match.group('xth'), match.group('yth'), match.group('zth') ]
                dims = [ float(s) for s in dims ]
                if not dims or len(dims) != 3:
                    raise Exception(f"Unable to extract slice thickness from DAT file: '{fp}'. Found slice thickness: '{dims}'.")
                return dims
        return (None, None, None) # workaround for the old XML format

def slice_to_img(args, slice, x, y, bitdepth, image_bitdepth, target_factor, input_factor, ofp):
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
        slice = np.floor(slice / float((2 ** input_factor) - 1) * float((2 ** target_factor) - 1))

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
        x, y, z = get_volume_dimensions(args, dat_fp)
        xth, yth, zth = get_volume_slice_thickness(args, f"{os.path.splitext(fp)[0]}.dat")
        logging.debug(f"Volume dimensions:  <{x}, {y}, {z}> for '{fp}'")
        logging.debug(f"Slice thicknesses:  <{xth}, {yth}, {zth}> for '{fp}'")

        bitdepth = determine_bit_depth(fp, (x,y,z), (xth, yth, zth))
        logging.debug(f"Detected bit depth: '{bitdepth}' for '{fp}'")

        # Pad the index for the slice in its filename based on the
        # number of digits for the total count of slices
        digits = len(str(z))
        num_format = '{:0'+str(digits)+'d}'

        # Set slice dimensions
        img_size = x * y
        logging.debug(f"Reading input as '{bitdepth}' (itemsize: {np.dtype(bitdepth).itemsize})")
        offset = img_size * np.dtype(bitdepth).itemsize

        # Determine scaling parameters per volume for output images
        # Equate the image format to numpy dtype
        if args.format == 'tif':
            image_bitdepth = 'uint16'
        elif args.format == 'png':
            image_bitdepth = 'uint8'
        else:
            image_bitdepth = 'uint8'

        target_factor = 8 * np.dtype(image_bitdepth).itemsize

        # When .RAW bit depth is *not* the same as the output image bit depth,
        # the data needs to be remapped from original bit depth to desired
        # image bit dpeth
        input_factor = 8 * np.dtype(bitdepth).itemsize

        # Extract data from volume, slice-by-slice
        slices = []

        description = f"Extracting slices from {os.path.basename(fp)} ({bitdepth})"
        pbar = tqdm(total = z, desc=description)
        with open(fp, 'rb') as f_data:
            # Dedicate N CPUs for processing
            with Pool(args.threads) as p:
                # For each slice in the volume...
                for i in range(0, z):
                    # Read slice data, and set job data for each process
                    f_data.seek(i*offset)
                    chunk = np.fromfile(f_data, dtype=bitdepth, count = img_size, sep="")
                    ofp = os.path.join(imgs_dir, f"{os.path.splitext(os.path.basename(fp))[0]}_{num_format.format(i)}.{args.format}")
                    # Check if the image already exists
                    if os.path.exists(ofp) and not args.force:
                        pbar.update()
                        continue
                    p.apply_async(slice_to_img, args=(args, chunk, x, y, bitdepth, image_bitdepth, target_factor, input_factor, ofp), callback=update)
                p.close()
                p.join()
        pbar.close()
        pbar = None

if __name__ == "__main__":
    args = options()
    start_time = time()

    # Collect all volumes and validate their metadata
    try:
        # Gather all files
        args.files = []
        for p in args.path:
            for root, dirs, files in os.walk(p):
                for filename in files:
                    args.files.append(os.path.join(root, filename))

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
            get_volume_dimensions(args, dat_fp)
    except Exception as err:
        logging.error(err)
    else:
        # For each provided directory...
        pbar = tqdm(total = len(args.files), desc=f"Overall progress")
        for fp in args.files:
            logging.debug(f"Processing '{fp}'")
            # Extract slices for all volumes in provided folder
            extract_slices(args, fp)
            pbar.update()
        pbar.close()

    logging.debug(f'Total execution time: {time() - start_time} seconds')

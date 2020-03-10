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
from multiprocessing import cpu_count, Pool, Value

import numpy as np
from PIL import Image
from tqdm import tqdm

pbar = None

def options():
    parser = argparse.ArgumentParser(description='Convert .raw 3d volume file to typical image format slices',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version='%(prog)s 1.1.0')
    parser.add_argument('-i', "--input_folder", help="Deprecated. Directory of .raw files")
    parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
    parser.add_argument('--force', action="store_true", help="Force file creation and overwrite existing files. You will be.warninged about which files are replaced.")
    parser.add_argument('-f', "--format", default='png', help="Set image filetype. Availble options: ['png', 'tif']")
    parser.add_argument("path", metavar='PATH', type=str, nargs='+', help='List of directories to process')
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
    print(f"Logging level: {logging_level} (INFO = {logging.INFO}, DEBUG = {logging.DEBUG})")
    consoleHandler.setLevel(logging_level)
    rootLogger.addHandler(consoleHandler)

    return args

def get_volume_dimensions(args, fp):
    with open(os.path.join(args.cwd, fp), 'r') as ifp:
        line = ifp.readlines()[1]
        dims = [int(s) for s in line.split() if s.isdigit()]
        if not dims or len(dims) != 3:
            raise Exception(f"Unable to extract dimensions from DAT file: '{fp}'. Found dimensions: '{dims}'.")
        return dims

def slice_to_img(df):
    """Convert byte data of a slice into an image"""
    global pbar
    args = df['args']
    slice = df['data']
    x, y, z = df['dims']
    ofp = df['ofp']

    slice = slice.reshape([y,x])
    if args.format == 'tif':
        datatype = 'uint16'
    elif args.format == 'png':
        slice = np.floor(slice * float((2 ** 8) - 1) / float((2 ** 16) - 1))
        datatype = 'uint8'
    else:
        datatype = 'uint8'

    Image.fromarray(slice.astype(datatype)).save(ofp)
    pbar.update(1)

def extract_slices(args):
    global pbar
    
    try:
        # Gather all files
        args.files = []
        for root, dirs, files in os.walk(args.cwd):
            for filename in files:
                args.files.append(os.path.join(root, filename))

        # Get all RAW files
        args.files = [ f for f in args.files if os.path.splitext(f)[1] == '.raw' ]
        logging.info(f"Found '{len(args.files)}' volume(s).")
        
        # Validate that a DAT file exists for each volume
        logging.debug("Validating DAT for each vol  ume")
        for fp in args.files:
            dat_fp = f"{os.path.splitext(fp)[0]}.dat" # .DAT filepath
            logging.debug(f"Checking DAT: '{dat_fp}'")
            # Try to extract the dimensions to make sure that the file exists
            get_volume_dimensions(args, dat_fp)
    except Exception as err:
        logging.error(err)

    # Otherwise, we know that a volume and its metadata exists
    else:
        # For each volume...
        for fp in tqdm(args.files, desc=f"Converting volumes to '{args.format.lower()}' slices"):
            logging.debug(f"Processing '{fp}'")
            
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

                # If a progress bar is not defined, create one
                if pbar is None:
                    pbar = tqdm(total = z, desc=f"Extracting slices from {os.path.basename(fp)}")
                with open(fp, 'rb') as f_data:
                    # Dedicate N CPUs for processing
                    with Pool(args.threads) as p:
                        # For each slice in the volume...
                        for i in range(0, z):
                            # Read slice data, and set job data for each process
                            f_data.seek(i*offset)
                            chunk = np.fromfile(f_data, dtype='uint16', count = img_size, sep="")
                            slices.append({
                                'args': args,
                                'data': chunk,
                                'dims': (x, y, z),
                                'ofp': os.path.join(imgs_dir, f"{os.path.splitext(os.path.basename(fp))[0]}_{num_format.format(i)}.{args.format}")
                            })
                        # Process each slice of the volume across N processes
                        p.map(slice_to_img, slices)
            pbar.close()
            pbar = None

if __name__ == "__main__":
    args = options()
    # NOTE(tparker): For now, I've kept the 'input_folder' argument in for
    # backwards compatibility. In the future, I would like to remove it. If
    # the user defines the input_folder, then replace any paths provided
    # as the positional argument 'directories'.
    if args.input_folder:
        args.path = args.input_folder
    args.format = args.format.lower()
    for d in args.path:
        args.cwd = os.path.realpath(d)
        logging.info(f"Processing '{args.cwd}")
        extract_slices(args)

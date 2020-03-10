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
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
from tqdm import tqdm

pbar = None

def options():
    parser = argparse.ArgumentParser(description='Convert .raw 3d volume file to typical image format slices',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version='%(prog)s 1.1.0')
    parser.add_argument('-i', "--input_folder", help="Deprecated. Directory of .raw files")
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
    global pbar

    logging.debug(df)
    logging.debug(df['dims'])
    args = df['args']
    slice = df['data']
    x, y, z = df['dims']
    i = df['index']
    # Pad the index for the slice in its filename based on the
    # number of digits for the total count of slices
    digits = len(str(z))
    num_format = '{:0'+str(digits)+'d}'

    slice = np.floor(slice * float((2 ** 8) - 1) / float((2 ** 16) - 1))
    slice = slice.reshape([y,x])
    imgname = f"{args.imgs_dir}/{args.filename}_{num_format.format(i)}.png"
    Image.fromarray(slice.astype('uint8')).save(os.path.join(args.imgs_dir, imgname))
    

def extract_slices(args):
    global pbar

    folder = args.cwd
    parent_path = os.path.dirname(folder)

    list_dirs = os.walk(folder)
    try:
        for root, dirs, files in list_dirs:
            for filename in files:
                basename, extension = os.path.splitext(filename)
                if extension == '.raw':
                    dat_filename = basename + '.dat'
                    x, y, z = get_volume_dimensions(args, dat_filename)
                    logging.debug(f"Volume dimensions:  <{x}, {y}, {z}>")

                    imgs_dir = os.path.join(folder, basename)
                    if not os.path.exists(imgs_dir):
                        os.makedirs(imgs_dir)
                    else:
                        logging.warning(f"Output directory for slices already exists '{imgs_dir}'.")

                    # Pad the index for the slice in its filename based on the
                    # number of digits for the total count of slices
                    digits = len(str(z))
                    num_format = '{:0'+str(digits)+'d}'

                    img_size = x*y
                    offset = img_size * np.dtype('uint16').itemsize

                    slices = []
                    if pbar is None:
                        pbar = tqdm(total = z, desc=f"Extracting slices from {basename}") # progress bar
                    with open(os.path.join(folder, filename), 'rb') as f_data:
                        # TODO(tparker): this is where a process should probably be spawn per slice
                        for i in range(0, z):
                            f_data.seek(i*offset)
                            slice = np.fromfile(f_data, dtype = 'uint16', count = img_size, sep = "")
                            slice = slice.reshape([y, x])
                            imgname = basename + '_' + num_format.format(i)
                            if args.format == 'png':
                                slice1 = slice*255.0/65535.0
                                slice1 = np.floor(slice1)
                                Image.fromarray(slice1.astype('uint8')).save(os.path.join(imgs_dir, imgname+'.png'))
                            if args.format == 'tif':
                                Image.fromarray(slice.astype('uint16')).save(os.path.join(imgs_dir, imgname+'.tif'))
                            pbar.update(1)
                        pbar.close()
    except Exception as err:
        logging.error(err)

if __name__ == "__main__":
    args = options()
    # NOTE(tparker): For now, I've kept the 'input_folder' argument in for
    # backwards compatibility. In the future, I would like to remove it. If
    # the user defines the input_folder, then replace any paths provided
    # as the positional argument 'directories'.
    if args.input_folder:
        args.path = args.input_folder

    for d in args.path:
        args.cwd = os.path.realpath(d)
        logging.info(f"Processing '{args.cwd}")
        extract_slices(args)

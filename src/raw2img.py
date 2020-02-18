#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Jul 27, 2018

@author: njiang
'''

import argparse
import logging
import os
from datetime import datetime as dt

import numpy as np
from PIL import Image
from tqdm import tqdm


def options():
    parser = argparse.ArgumentParser(description='Convert .raw 3d volume file to typical image format slices',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version='%(prog)s 1.0.0')
    parser.add_argument('-i', "--input_folder", help="directory of .raw files", required=True)
    parser.add_argument('-f', "--format", help="image format", default='png')    
    args = parser.parse_args()

    # Configure logging, stderr and file logs
    logging_level = logging.INFO
    if args.verbose:
        logging_level = logging.DEBUG

    lfp = f"{dt.today().strftime('%Y-%m-%d')}_{os.path.splitext(os.path.basename(__file__))[0]}.log"

    logFormatter = logging.Formatter("%(asctime)s - [%(levelname)-4.8s] - %(filename)s %(lineno)d - %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging_level)

    fileHandler = logging.FileHandler(lfp)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return args

def extract_slices(args):
    folder = args.input_folder
    parent_path = os.path.dirname(folder)

    list_dirs = os.walk(folder)
    for root, dirs, files in list_dirs:
        for filename in files:
            basename, extension = os.path.splitext(filename)
            if extension == '.raw':
                dat_filename = basename + '.dat'
                imgs_dir = os.path.join(folder, basename)
                if not os.path.exists(imgs_dir):
                    os.makedirs(imgs_dir)
                with open(os.path.join(folder, dat_filename), 'r') as f_info:
                    line = f_info.readlines()[1]
                dims = [int(s) for s in line.split() if s.isdigit()]
                x, y, z = dims
                logging.debug(f"Shape:  <{x}, {y}, {z}>")
                digits = len(str(z))
                num_format = '{:0'+str(digits)+'d}'
            
                img_size = x*y
                offset = img_size * np.dtype('uint16').itemsize
            
                with open(os.path.join(folder, filename), 'rb') as f_data:
                    pbar = tqdm(total = z, desc=f"Extracting slices from {basename}") # progress bar
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

if __name__ == "__main__":
    args = options()
    extract_slices(args)

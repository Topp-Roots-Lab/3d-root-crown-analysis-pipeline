#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import shutil
import subprocess
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def options():
    VERSION = "1.2.0"
    parser = argparse.ArgumentParser(description='Root Crowns Segmentation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {VERSION}')
    parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
    parser.add_argument("path", metavar='PATH', type=str, nargs='+', help='Input directory to process')

    parser.add_argument('--soil', action='store_true', help="Extract any soil during segmentation.")
    parser.add_argument('-i', "--input_folder", action="store_true", help="Deprecated. Directory of original image slices") # left in for backwards compatibility
    parser.add_argument('-s', "--sampling", help="resolution parameter", default=2)
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

    # Recode soil input to match the input of rootCrownSegmentation binary
    if args.soil:
        args.soil = 1
    else:
        args.soil = 0

    logging.debug(f'Running {__file__} {VERSION}')

    return args

# Async function to call rootCrownSegmentation binary
def run(cmd):
    logging.debug(f"Run command: '{cmd}'")
    proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

if __name__ == "__main__":
    args = options()

    # Clean up input folders
    args.path = [ os.path.realpath(fp) for fp in args.path ]
    args.path = list(set(args.path)) # remove duplicates

    # Count the number of sub-folders (i.e., volumes as PNG slices) per input folder
    input_paths = []
    for fp in args.path:
        # Check if the input folder exists
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        for root, dirs, files in os.walk(fp):
            input_paths += [ os.path.join(fp, f) for f in dirs ]

    args.path = list(set(input_paths)) # remove duplicates

    logging.info(f"Found {len(args.path)} volume(s).")
    logging.debug(f"Processing {len(args.path)} volume(s): {args.path}")

    # Create threshold and model folders
    for fp in set([ os.path.dirname(path) for path in args.path ]):
        thresholded_folder = f"{fp}_thresholded_images"
        model_folder = f"{fp}_3d_models"

        if not os.path.exists(thresholded_folder):
            os.makedirs(thresholded_folder)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

    # For each provided input folder, build a command for each volume
    cmd_list = []
    for fp in args.path:
        thresholded_folder = f"{os.path.dirname(fp)}_thresholded_images"
        model_folder = f"{os.path.dirname(fp)}_3d_models"

        # Create paths to the output files
        volume_name = os.path.basename(fp)
        ofp         = os.path.join(thresholded_folder, os.path.basename(fp))    # output folder for segmented images
        out_fp      = os.path.join(model_folder, f"{volume_name}.out")          # root system .OUT file
        obj_fp      = os.path.join(model_folder, f"{volume_name}.obj")          # root system .OBJ file
        soil_out_fp = os.path.join(model_folder, f"{volume_name}_soil.out")     # dirt .OUT file
        soil_obj_fp = os.path.join(model_folder, f"{volume_name}_soil.obj")     # dirt .OBJ file

        # Create the sub-directory for the set of thresholded images per volume
        if not os.path.exists(ofp):
            os.makedirs(ofp)

        cmd = ['rootCrownSegmentation', str(args.soil), f'"{fp}/"', str(args.sampling), f'"{ofp}/"', f'"{out_fp}"', f'"{obj_fp}"']
        if args.soil == 1:
            cmd += [soil_out_fp, soil_obj_fp]
        cmd_list.append(' '.join(cmd))

    def update(*args):
        pbar.update()

    # Process data
    pbar = tqdm(total = len(cmd_list), desc=f"Segmenting volumes")
    # Dedicate N CPUs for processing
    with Pool(args.threads) as p:
        # For each slice in the volume...
        for cmd in cmd_list:
            # Run command as separate process
            p.apply_async(run, args=(cmd,), callback=update)
        p.close()
        p.join()
    pbar.close()
    pbar = None

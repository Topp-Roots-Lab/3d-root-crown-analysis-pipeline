#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import shutil
import subprocess
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count
from pprint import pformat

from tqdm import tqdm

from __init__ import __version__


def options():
    parser = argparse.ArgumentParser(description='Root Crowns Feature Extraction',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")
    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
    parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
    parser.add_argument("path", metavar='PATH', type=str, nargs='+', help='Input directory to process.')
    parser.add_argument('-i', "--input_folder", action="store_true", help="Deprecated. Directory of .out files.") # left in for backwards compatibility
    parser.add_argument('-s', "--scale", help="The scale parameter using for skeleton.", default=2.25)
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

    logging.debug(f'Running {__file__} {__version__}')

    return args

def create_features_files(fp):
    output_name = os.path.join(fp, "features.tsv")

    return output_name

def process(ifp, ofp, scale, cmd):
    p = subprocess.run([cmd, ifp, ofp, str(scale)], capture_output=True, text=True, check=True)
    logging.debug(p.stdout)
    if p.returncode > 0:
        logging.error(f"Error encountered. 'Skeleton' returned {p.returncode}")

if __name__ == "__main__":
    args = options()

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

    def update(*args):
        """Update progress bar for an async process call"""
        pbar.update()

    # Remove previous features.tsv files
    for fp in list(set([ os.path.dirname(p) for p in args.path ])):
        features = ['Name','SurfArea','Volume','Convex_Volume','Solidity','MedR','MaxR','Bushiness','Depth','HorEqDiameter','TotalLength','SRL','Length_Distr','W_D_ratio','Number_bif_cl','Av_size_bif_cl','Edge_num','Av_Edge_length','number_tips','volume','surface_area','av_radius']
        fname = os.path.join(fp, "features.tsv")
        with open(fname, 'w') as ofp:
            ofp.write('\t'.join(features))

    # Process data
    pbar = tqdm(total = len(args.path), desc=f"Generating meshes")
    # Dedicate N CPUs for processing
    with Pool(args.threads) as p:
        # For each slice in the volume...
        binary_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'bin', 'Skeleton')
        for fp in args.path:
            # Read object files, and then spawn a child process per volume
            ofp = os.path.join(os.path.dirname(fp), "features.tsv")
            p.apply_async(process, args=(fp, ofp, args.scale, binary_filepath), callback=update)
        p.close()
        p.join()
    pbar.close()
    pbar = None

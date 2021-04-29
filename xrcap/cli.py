"""Console script for xrcap."""
import argparse
import logging
import os
import sys
import time
from importlib.metadata import version
from multiprocessing import cpu_count

from xrcap import (batch_segmentation, batch_skeleton, rootCrownImageAnalysis3D, collate, log,
                   qualitycontrol)

__version__ = version('xrcap')
GIT_COMMIT = '55d21bb'

def main():
    """Console script for xrcap."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    parser.parse_args()
    return 0

def segment():
    description = "Segmentation"
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__} (commit {GIT_COMMIT})')
    parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
    parser.add_argument("-f", "--force", action="store_true", help="Force file creation. Overwrite any existing files.")
    parser.add_argument("-n", "--dry-run", dest='dryrun', action="store_true", help="*Not yet implemented.* Perform a trial run. Do not create image files, but logs will be updated.")
    parser.add_argument("--progress", action="store_true", help="Enables multiple progress bar, one for each volume during processing.")
    parser.add_argument('--soil', action='store_true', help="Extract any soil during segmentation.")
    parser.add_argument('-s', "--sampling", type=int, help="resolution parameter", default=2)
    parser.add_argument("--csv", action="store", type=str, help="Input is a CSV containing list of file paths and lower- and upper-bounds for segmentation")
    parser.add_argument("path", metavar='PATH', type=str, nargs=1, help='Input directory to process')
    args = parser.parse_args()

    # Make sure user does not request more CPUs can available
    if args.threads > cpu_count():
        args.threads = cpu_count()

    args.module_name = 'batch_segmentation'
    log.configure(args)

    if args.dryrun:
        logging.info(f"DRY-RUN MODE ENABLED")

    # Disable progress bars if verbose mode enabled
    if args.verbose:
        args.progress = False

    start_time = time.perf_counter()
    if args.csv:
        returncode = batch_segmentation.csv(**vars(args))
    else:
        returncode = batch_segmentation.main(args)
    logging.info(f'Total execution time: {time.perf_counter() - start_time} seconds')
    return returncode

def skeleton():
    description = "Skeletonization, Meshing, and Feature extraction"
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")
    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__} (commit {GIT_COMMIT})')
    parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
    parser.add_argument("-f", '--force', action="store_true", help="Force file creation. Overwrite any existing files.")
    parser.add_argument("-n", '--dry-run', dest='dryrun', action="store_true", help="Perform a trial run. Do not create image files, but logs will be updated.")
    parser.add_argument("--progress", action="store_true", help="(Not yet implemented) Enables multiple progress bar, one for each volume during processing.")
    parser.add_argument('-s', "--scale", help="The scale parameter using for skeleton.", default=2.25)
    parser.add_argument("path", metavar='PATH', type=str, nargs=1, help='Input directory to process')
    args = parser.parse_args()

    # Make sure user does not request more CPUs can available
    if args.threads > cpu_count():
        args.threads = cpu_count()

    args.module_name = 'batch_skeleton'
    log.configure(args)

    if args.dryrun:
        logging.info(f"DRY-RUN MODE ENABLED")

    # Disable progress bars if verbose mode enabled
    if args.verbose:
        args.progress = False
    # NOTE(tparker): Remove this line once progress on an individual volume is implemented
    args.progress = False

    start_time = time.perf_counter()
    returncode = batch_skeleton.main(args)
    logging.info(f'Total execution time: {time.perf_counter() - start_time} seconds')
    return returncode

def image_analysis():
    parser = argparse.ArgumentParser(description='Root Crown Image Analysis', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__} (commit {GIT_COMMIT})')
    parser.add_argument("-f", '--force', action="store_true", help="(Not yet implemented) Force file creation. Overwrite any existing files.")
    parser.add_argument('-s', "--sampling", default=2, help="resolution parameter")
    parser.add_argument('-t', "--thickness", type=float, help="slice thickness in mm")
    parser.add_argument("--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
    parser.add_argument("--kde", action="store_true", help="Enable calculation for biomass_vhist and convexhull_vhist")
    parser.add_argument("path", metavar='input_folder', type=str, nargs=1, help='Input directory to process')
    args = parser.parse_args()

    # Make sure user does not request more CPUs can available
    if args.threads > cpu_count():
        args.threads = cpu_count()

    args.path = list(set(args.path)) # remove any duplicates

    args.module_name = 'rootCrownImageAnalysis3D'
    log.configure(args)

    start_time = time.perf_counter()
    returncode = rootCrownImageAnalysis3D.main(args)
    logging.info(f'Total execution time: {time.perf_counter() - start_time} seconds')
    return returncode

def qc_binary_images():
    description='Check tresholded images for pure white slices. Creates CSV of volumes that have more than a given percentage of white pixels.'
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__} (commit {GIT_COMMIT})')
    parser.add_argument("-c", "--cutoff", type=float, default=0.8, help="The minimum percentage of white pixels for a given slice for it to be flagged as invalid.")
    parser.add_argument("-l", "--list", action="store_true", help="Output TXT files that lists bad binary images produced by segmentation")
    parser.add_argument("path", metavar='PATH', type=str, nargs='+', help='Input directory to process. Must contain folder with thresholded images.')
    args = parser.parse_args()

    args.module_name = 'qc_binary_images'
    log.configure(args)

    start_time = time.perf_counter()
    returncode = qualitycontrol.binary_images(args)
    logging.info(f'Total execution time: {time.perf_counter() - start_time} seconds')
    return returncode

def qc_point_clouds():
    description='Create a downsampled version of point cloud data (.obj) based on a random selection of which points are kept.'
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__} (commit {GIT_COMMIT})')
    parser.add_argument('-t', "--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
    parser.add_argument("-f", "--force", action="store_true", help="Force file creation. Overwrite any existing files.")
    parser.add_argument("-p", "--probability", type=float, default=0.03, help="Probability that a point will be kept ")
    parser.add_argument("path", metavar='PATH', type=str, nargs='+', help='Input directory to process. Must contain folder with thresholded images.')
    args = parser.parse_args()

    args.module_name = 'qc_point_clouds'
    log.configure(args)

    # Make sure user does not request more CPUs can available
    if args.threads > cpu_count():
        args.threads = cpu_count()

    # Make sure probability is between 0 and 1
    if not (0 <= args.probability<= 1.0):
        raise ValueError(f"User-defined probability is invalid: '{args.probability}'. It must be between 0 and 1.0.")

    start_time = time.perf_counter()
    returncode = qualitycontrol.point_clouds(args)
    logging.info(f'Total execution time: {time.perf_counter() - start_time} seconds')
    return returncode

def collate_output():
    description='Combine many results files into one.'
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__} (commit {GIT_COMMIT})')
    parser.add_argument("-f", "--force", action="store_true", help="Force file creation. Overwrite any existing files.")
    parser.add_argument("--traits", action="store_true", help="Combine any traits.csv into one file.")
    parser.add_argument("--features", action="store_true", help="Combine any features.tsv into one file.")
    parser.add_argument("-o", dest="ofp", help="Specify output filepath.")
    parser.add_argument("path", metavar='PATH', type=str, nargs='+', help='Input directory to process. Must contain folder with thresholded images.')
    args = parser.parse_args()

    args.module_name = 'collate'
    log.configure(args)

    # If not action is specified, nothing can be done
    if not args.traits and not args.features:
        logging.error("No action was specified. Please enable either '--traits' or '--features'.")
        return 1

    # Get the extensionless filepath for output file
    if args.ofp is not None:
        args.ofp = os.path.splitext(args.ofp)[0]

    start_time = time.perf_counter()
    if args.traits:
        collate.process(args, type='traits')
    if args.features:
        collate.process(args, type='features')
    logging.info(f'Total execution time: {time.perf_counter() - start_time} seconds')

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

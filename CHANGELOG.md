# Changelog

All notable changes to this project will be documented in this file.

## v1.6.1 - 2020-08-31

### Changed

- Simplified installation and updating to use Makefile

### Fixed

- Location of system copies of log files to `/var/log/xrcap` as intended

## v1.6.0 - 2020-07-30

### Added

- Redundant logging to `/var/log/3drcap` for all modules
- Logging for `rootCrownSegmentation` (handled by `batch_segmentation` module)
- Separate module for centralizing logging configuration
- Functionality for automatically compressing VRML files to CTM using `meshlabserver` with xvfb wrapper
- The version of `rootCrownSegmentation` and calculated scale included in generated OBJ files

### Changed

- Converted `rootCrownImageAnalysis3D.py` from Python version 2 to 3.8
- Replaced multiprocessing Pool with ThreadedPool to allow for simultaneously existing progress bars (`batch_segmentation`)
- Logs are named to the nearest second (datetime) and include the module in the filename
- Updated log handling to match `Skeleton` v2.1.0
- Removed `.CSV` output file for slices flagged for incorrect segmentation (see note 1)
- Reduced default probability that a point is included in downsampled OBJ for the `qc_point_cloud` module
- Refactored `rootCrownSegmentation.cpp` to use more descriptive variable names and explicit comments
- Implemented additional checks during segmentations to prevent slices of air from being segmented as root system (see note 2)
- Traits calculated using kernel density estimation, i.e., biomass_vhist and convexhull_vhist, instead call compiled MATLAB code (use `--kde` flag) (see note 3)
- Refactored reading point data from thresholded images in `rootCrownImageAnalysis3D` to reduce the number of times copies of data while minimizing the amount of allocated memory per volume (`np.append()` changed to assign data to sub-range of `numpy.array`)

### Fixed

- Typos in comments and documentation
- `qc_point_cloud` now will correctly process a data folder with exactly one volume (previously it would state that no volumes were present)
- Removed a dummy point for the collection of all points and all points in the convex hull of the volume
- Removed dummy values from initialized images for density_T calculations
- Replaced initialization of projection images with `np.zeros()`. Values left behind in main memory caused ghost images or garbage values to be retained between volumes in batch processing

### Notes

1. This was removed because it provides less information than the individual TXT files that list the exact problematic slices.

2. For shallow root crowns, a narrow histogram of grayscale intensity values caused air to be segmented as root system. As a workaround, when a slice is determined to have a narrow range of values, the slice is assumed to be pure air and therefore removed. The values selected were determined by trail and error, so for future data, the value may need to be tweaked.

3. The MATLAB code is now an external module that must be installed independently but is a dependency for this this tool. Its installation file is stored on the Danforth Center's cluster in the Topp lab's data repository. See <https://github.com/Topp-Roots-Lab/New3DTraitsForRPF/tree/standalone-kde-traits>.

## v1.5.0 - 2020-04-23

### Added

- Script to downsample point cloud data for quality control version of point cloud data
- Cutoff value to input options for binary image segmentation script
- Detect bit depth of .RAW volumes during `raw2img` processing

### Changed

- Renamed quality control script for binary images
- Restructure files into bin, src, and package

## v1.4.1 - 2020-04-14

### Fixed

- `check_tresholded_images.py` to only take folder as input

## v1.4.0 - 2020-04-14

### Added

- Rudimentary script, `check_tresholded_images.py`, to check for bad segmentations

### Changed

- Version is pulled from `__init__.py`
- Changed file structure so name conformed to package name restrictions

## v1.2.0 - 2020-03-15

### Added

- Logging for `batch_skeleton` and `batch_segmentation`

### Changed

- Parallelized segmentation and skeletonization at the process level (not internally)

## v1.1.0 - 2020-03-11

### Added

- docstrings to `raw2img`
- `--threads`, `-t` option to specify how many threads to use while processing for `raw2img`
- Log file (logging level defaults to debug)
- Progress bar
- Changelog

### Changed

- `raw2img`: slice extraction parallelized
- `raw2img`: allow user to include more than one directory as input

# Changelog

All notable changes to this project will be documented in this file.

## v1.6.0 - 2020-06-18

### Added

- Redundant logging to `/var/log/3drcap` for all modules
- Additional logging for `rootCrownSegmentation` (handled by `batch_segmentation` module)
- Utility module for centralizing logging configuration
- Functionality for compressing VRML files to CTM using `meshlabserver`
- The version of `rootCrownSegmentation` included in generated OBJ files

### Changed

- Replaced multiprocessing Pool with ThreadedPool to allow for simultaneously existing progress bars (`batch_segmentation`)
- Logs are named to the nearest second and include the input folder as part of the file name
- Updated version of `Skeleton` to v2.1.0
- Removed `.CSV` output file for slices flagged for incorrect segmentation (see note 1)
- Reduce default probability that a point is included in downsampled OBJ for the `qc_point_cloud` module
- Refactored `rootCrownSegmentation.cpp` to use more descriptive variable names and explicit comments

### Fixed

- Typos in comments and documentation
- `qc_point_cloud` now will correctly process a data folder with exactly one volume (previously it would state that no volumes were present)

### Notes
1. This was removed because it provides less information than the individual TXT
files that list the exact problematic slices.

## v1.5.0 - 2020-04-23

### Added

- Script to downsample point cloud data for quality control version of point cloud data
- Added cutoff value to input options for binary image segmentation script
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
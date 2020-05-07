# Changelog

All notable changes to this project will be documented in this file.

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
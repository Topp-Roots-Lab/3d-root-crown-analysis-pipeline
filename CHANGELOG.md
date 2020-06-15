# Changelog

All notable changes to this project will be documented in this file.

## v1.6.0 - 2020-06-15

### Added

- Redundant logging to `/var/log/3drcap` for all modules
- Additional logging for `rootCrownSegmentation` (handled by `batch_segmentation` module)
- Utility module for centralizing logging configuration
- Functionality for compressing VRML files to CTM using `meshlabserver`
- Version of `rootCrownSegmentation` included in generated OBJ files

### Changed

- Mirgrated `rootCrownImageAnalysis3D` from Python 2 to 3
- Removed `-i` flag support in favor of a single positional argument, except for `rootCrownImageAnalysis3D`
- Replaced multiprocessing Pool with ThreadedPool to allow for simultaneously existing progress bars (`batch_segmentation`)
- Logs are named to the nearest minute
- Added alternative segmentation method when errors detected due to narrow histogram - (*under development*)
- Updated version of `Skeleton` to v2.1.0-rc
- Slice thickness is no longer required as input for the `rootCrownImageAnalysis3D` (see note 1)
- Removed `.CSV` output file for slices flagged for incorrect segmentation (see note 2)

### Fixed

- Typos in comments and documentation

### Notes
1. The slice thickness value, which is normally specified with the `-t` flag on `rootCrownImageAnalysis3D`, is now
extracted from the `.DAT` associated with any specified volume. If the file cannot be located, then user input is 
required. The `-t` flag will remain availble, and if it is supplied, it will take precedence over any values pulled
from a `.DAT` file.
2. This was removed because it provides less information than the individual TXT
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
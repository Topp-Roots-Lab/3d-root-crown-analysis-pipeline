# 3-D Root Crown Analysis Pipeline

Stable version: 1.0.0

Author: Ni Jiang

Compatibility: Python 3

This set of scripts is used to analyze 3-D volumes of maize root crowns.

## Individual Modules

### Volume to Image Conversion (`raw2img.py`)

Converts the a `.raw` volume into an grayscale image stack.
The volume is sliced along the Z axis, from top to bottom.

### Segmentation (`batch_segmentation.py`)

Converts the grayscale image stack into binary image stack. You have the option
to remove soil if the `--soil` option is provided.

### GiA3D or Skeletonization/Mesh Generation (`batch_skeleton.py`)

Converts a point cloud representation of the root system into a 3-D mesh.
Currently, this module also produces the skeleton of the root system as well.

### Analysis (`rootCrownImageAnalysis3D.py`)

Analyzes the root system based on the binary image stack and calculates traits
for the root system.

## Root Crown Analysis Pipeline Flowchart

This is an overview of the execution sequence for analyzing root crown x-ray
scans.

<p align="center">
  <img alt="Root Crown Analysis Pipeline Flowchart" src="docs/img/root-crown-pipeline-flowchart.png">
</p>

## Additional Notes
The underlying executables written in C++ can be altered in addition to the
Python scripts. Skeletonization requires a specific environment to be compiled,
and has been curated in another repository. Further development needs to be done
in order to allow for Windows-based execution and/or excluding skeleton
generation.

### features.tsv

This file specifies the traits that are calculated during mesh generation.
In addition, the current version of the software expects execution relative
to environment on `Ludo`. Currently, this file is used as input for an
executable (`Skeleton`).

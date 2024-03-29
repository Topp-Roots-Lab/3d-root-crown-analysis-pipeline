# Usage Guide

An example run of the entire pipeline is as follows:

```bash
raw2img -t 8 /path/to/volume;
batch-segmentation --progress -t 8 /path/to/volume;
batch-skeleton -t 8 /path/to/volume_3d_models;
rootCrownImageAnalysis3D /path/to/volume_thresholded_images
```

By convention, the sampling, denotated by `-s` flag, is set as 2 as the default. You will need to extract the slice thickness value from the `.DAT` file associated with the volume. The slice thickness is the real-world thickness, in millimeters, of each slice. They should be the same for each dimension. By convention, we round the the nearest thousandth (e.g., 0.1042 -\> 0.104).

### Root Crown Analysis Pipeline Flowchart

This is an overview of the execution sequence for analyzing root crown x-ray scans.

<p align="center">
  <img alt="Root Crown Analysis Pipeline Flowchart" src="docs/img/root-crown-pipeline-flowchart.png">
</p>

### Description

Below is a description of each individual module.

## Volume to Image Conversion (`raw2img.py`)

Converts the a `.raw` volume into an grayscale image stack. The volume is sliced along the Z axis, from top to bottom.

## Segmentation (`batch_segmentation.py`)

Converts the grayscale image stack into binary image stack. You have the option to remove soil if the `--soil` option is provided.

## Gia3D or Skeletonization/Mesh Generation (`batch_skeleton.py`)

Converts a point cloud representation of the root system into a 3-D mesh. Currently, this module also produces the skeleton of the root system as well.

## Analysis (`rootCrownImageAnalysis3D.py`)

Analyzes the root system based on the binary image stack and calculates traits for the root system.

### Workflow

The pipeline is installed as a system-wide tool, so you will not need to navigate to it. The following commands can be called from any location so long as you are logged in via SSH or have a terminal open on the machine. Only one step will require user input: rootCrownImageAnalysis3D, but keep in mind that additional flag/options are available to tweak how the data is processed (e.g., sampling).

1. Convert 3-D Volume to Grayscale 2-D Slices

---

**Example**

```bash
raw2img -t 10 /path/to/myVolume/
```

This folder should contain the .raw and .dat files.

This script creates sub-folders for each volume with extract grayscale images for each.

Note on threads: The -t option allows you to dedicate a maximum number of CPUs dedicated to processing your data. This applies to every step but the final one, rootCrownImageAnalysis3D. For Viper, generally you do not get a performance increase beyond using 30 threads. For Ludo, we recommend using no more than 8 threads.

2. Segmentation: Create Binary 2-D Slices & Generate Point Cloud

---

Be aware that this is one of two scripts that may require input at the end of the command (when you call the script, you can type -h for “help” at the end to get more description of what the input means and what options are available). By default the sampling is set to 2; this will downsample the data by half.

**Example**

```bash
batch_segmentation -t 10 /path/to/myVolume/
```

This folder should contain the .raw and .dat files. This script creates two sibling folders: /path/to/myVolume_3d_models/ and /path/to/myVolume_thresholded_images. Respectively, the first will contain point cloud data and the second will contain binary images.

#### Quality Control

Segmentation sometimes results in invalid point cloud data and binary images (stored in /path/to/myVolume_thresholded_images/). This may manifest as larger than expected point cloud data (.obj & .out) and binary images that primarily consist of white pixels.

If any point cloud data is several times larger than the median, check its respective binary images for completely white slices. If you find a large number of white slices, delete the slices that cover the same range in the grayscale images (output of raw2img), delete the generated thresholded_images folder, and then re-run this step of the pipeline.

If done correctly, batch_segmentation should run faster, if only marginally, than the last attempt and the file sizes should be more loosely uniformly distributed.

3. Generate 3-D Mesh and Skeleton

---

Run the batch_skeleton script on the desired slices folder (at this point, the scripts are sequentially run on the output of previous scripts and file path will need to include path of metadata created in the previous script)

**Example**

```bash
batch_skeleton -t 10 /path/to/myVolume_3d_models/
```

This folder should contain point cloud data: .obj and .out. This script will transform the point cloud data into meshes for quality control: .wrl files.

#### Quality Control

To further verify that segmentation was performed correctly, view the generated meshes in a 3-D model viewer such as Meshlab. Generally, make sure that the mesh generated looks like a real-world root system. Look out for solid planes that slice through the mesh and doubled root tips. The latter may indicate a mishap during reconstruction with the NSI software and can be reconstructed again to salvage the sample. Check with your supervisor for specifics on quality control for the meshes.

4. Measure Traits from Binary 2-D Slices

---

This script requires additional input from the user; keep in mind that -s does not mean the same thing as in the previous script and -t requires the resolution of the scans being analyzed to follow in millimeters. For resolution in μm, move the decimal three spaces to the left. The resolution should be a part of the volume’s filename, it is also stored as the SliceThickness value in its .dat file. By convention, we round this to the nearest thousandth.

**Example**

```bash
rootCrownImageAnalysis3D -s 2 -t 0.104 -i /path/to/myVolume_thresholded_images/
```

This folder should contain the binary images generated by the segmentation step. This script generates a .CSV of the measured traits for each volume.

**Done!** See below for additional information on running multiple projects in succession and available options for each script.
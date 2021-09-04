# Implementation

This document provides more technical descriptions of data ingest, internal data structures, and how each trait is calculated.

## Data and how it's represented

The input for this pipeline typically starts with a `.raw` X-ray scan. They should be **unsigned 16-bit byte sequences**. By convension, the object of interest (root crown) is oriented such that the stalk is near "top" of the volume and root tips towards the "bottom" of the volume. It's highly recommended to follow this convention. Keep in mind that if you choose to invert your input data, you will need to account for that when interpreting the phenotypes.

{INSERT EXAMPLE OF SCAN WITH SIDE PROJECTION}

We'll assume that you have already segmented your data and have a `.out` and binary image sequence of your data. 

## Measuring and calculating traits

Each trait is described in detail below. The associated code base that implements the trait is identified.

### SurfaceArea

Count of exposed faces of root system. Implemented by [Gia3D].

### Volume

Count of voxels that represent a root system. Implemented by [Gia3D].

### ConvexVolume

Total volume computed by qhull library (v2012.1). Implemented by [Gia3D].

### Solidity

The ratio of the volume to the convex hull volume. Implemented by [Gia3D].

VolumeConvexVolume

### MedR

The 50th percentile of the number of connected components for all slices. The number of connected components represent the number of roots that intersect a given slice.

The slices are ordered in ascending order from least number of roots to greatest, and the slice that falls on the 50th percentile is reported as the MedR. If the depth is an even number, it reports the average of the 50th percentile and that of the next lower number of roots for a given slice.

For each slice in root system
Gather the voxels for slice and convert them into a number of connected components (i.e., join neighboring voxels into a single unit, 26-neighbor method)
Store number of components in an unordered list
Once the number of connected components for each slice is calculated, sort in ascending order.


Depending on the depth (i.e., total number of slices), there are two cases.
There are an odd number of slices (odd depth), return 50th percentile count of connected components for all slices.
There are an even number of slices (even depth), return the 50th percentile count of connected components for all slices averaged with the next lower count of connected components for a slice. Implemented by [Gia3D].

### MaxR

The 84th percentile of the number of connected components for all slices. The number of connected components represent the number of roots that intersect a given slice.

The slices are ordered in ascending order from least number of roots to greatest, and the slice that falls on the 84th percentile is reported as the MaxR.

For each slice in root system
Gather the voxels for slice and convert them into a number of connected components (i.e., join neighboring voxels into a single unit, 26-neighbor method)
Store number of components in an unordered list
Once the number of connected components for each slice is calculated, sort in ascending order.

Return roughly the 84th percentile of the number of connected components for all slices by on depth. Implemented by [Gia3D].

### Bushiness

The ratio of MaxR to MedR. Implemented by [Gia3D].

### Depth

The difference in the number of voxels between the first slice and the deepest slice.
The slices are 0-indexed, so 1 is added. Implemented by [Gia3D].
slicen-slice0+1 

### HorEqDiameter

Largest value for widthS=2volumeCH,S, given the volume of the convex hull for any slice along the vertical axis, ignoring edge cases where volume cannot be defined. Convex hull is calculated by qhull library (v2012.1).

This is either referred to as the maximum horizontal distance for all slices or the equivalent diameter of the convex hull.

For each slice in root system
Gather voxels for slice and calculate width
Case A: 0 voxels -> skip to next slice
Case B: 1 voxel
Set widthS = 0 and continue to step 2
Case C: 2 voxels
Set widthS = euclidean distance between two voxels and continue to step 2
Case D: 3 or more voxels
Measure volume of convex hull of the slice (volumeCH,S) (qhull v2012.1) and continue to step 2
widthS=2volumeCH,S
If widthS is larger than the current known maximum width, set it to the widthS.
Once each slice has been checked for a greater width, return the largest as HorEqDiameter.

Implemented by [Gia3D].

### TotalLength

Count of voxels that represent a skeleton of the root system. Implemented by [Gia3D].

### SRL

Ratio of total root length to volume. Implemented by [Gia3D].

### Length_Distr

The ratio of root length in the upper 1⁄3 of the volume to the root length in the lower 2⁄3 of the volume.

Note, the root length for each is the number of voxels present in the skeleton for their respective portion relative to the depth. I labeled this feature as calculated using both the skeleton and PCD because the depth is calculated using the PCD, and the length is calculated using the voxels that make up the skeleton. Implemented by [Gia3D].

### W_D_ratio

The ratio of the HorEqDiameter to the depth. Implemented by [Gia3D].

### NumberBifCl

The number of connected components made up of neighboring branches in the skeleton.

For each voxel in the skeleton
If the voxel has more than two neighboring voxels, create a branch and add it to a list of known branches.

Note, this is stored in a hash_set/unordered_set using the linear index of the voxel “borrowed” from the skeleton’s voxset representation


Once every branch has been created, initialize a queue with the first branch found. We want to collect the branching nodes into connected components.


For each known branch, create a connected component for neighboring branches.

The number of connected components made up of neighboring branches is the number of bifurcation clusters.

Implemented by [Gia3D].

### AvgSizeBifCl

The total number of voxels for all connected components of neighboring branches divided by the number of bifurcation clusters. I.e., the average number of voxels for each bifurcation cluster (connected component with more than two neighboring voxels/branches). Implemented by [Gia3D].

### EdgeNum

Number of edges, “sequence of neighboring voxels”, excluding sequences that end with a tip. Each edge ends at its starting voxel or a bifurcation cluster. Sequences that end with a tip are excluded.

For each voxel in the skeleton
Count the number of voxels that are members of bifurcation clusters and not, separated into two collections: cluster “branch” & non-cluster “non-branch” voxels.
Branch = more than two neighboring voxels
Non-branch = 2 or fewer neighboring voxels
Collect all non-branching voxels belonging to the same branch
For each non-branching voxel, find its neighbors (should be 2 or fewer total). This is effectively looking for a “straight shot” down the root.
If no neighbors are found, expand search to branch voxels. When a voxel from a neighboring branch is found, add the neighboring voxel to the edge, end edge, and continue to step 1ai for the next non-branching voxel.
If no neighbors were found and no neighboring branches were found, it’s a root tip and the edge is completed. This voxel is flagged as a root tip and the “start” of an edge. Continue to step 1ai for the next non-branching voxel.
[NEED TO CONFIRM] Sort indices of edges by branching point (MedialCurve.cpp, Lineno. 591-625)
Iterate over all voxels in the skeleton and count the number of voxels with exactly one neighbor; this is the number of root tips.
For each edge that does not end with a root tip (i.e., only the length of root that connect by bifurcation clusters)
Count the number of edges
Sum the euclidean distance between each voxel that makes up each edge.

In other words, hop from voxel to voxel along an edge/root, adding up the distance of each, i=0v(xi+1-xi)2+(yi+1-yi)2+(zi+1-zi)2where v is the number of voxels that make up the edge, ordered.

Return number of edges as Edge_num, and return average length of edges by dividing the sum of the length of all edges by the number of edges.

Implemented by [Gia3D].

### AvgEdgeLength

Sum of the length of all edges calculated from the skeleton divided by the number of edges. The length of each edge is the euclidean distance traversed by walking along each edge, voxel by voxel. Implemented by [Gia3D].

### Number_tips

Number of root tips. Implemented by [Gia3D].

### volume

The sum of squares of estimated iterations needed to erode the shape of the PCD until it forms a one-voxel wide curve, multiplied by pi. I.e., r2for each voxel where the radius is the number of iterations to erode down to a skeleton. Implemented by [Gia3D].

### Surface_area

The sum of estimated iterations needed to erode the shape of the PCD until it forms a one-voxel wide curve, multiplied by 2. Implemented by [Gia3D].

### av_radius

Create voxset from PCD
Copy PCD voxset to “skel” object and then “create skeleton” using 2.25 as scale. This scale is “complexity,” not the resolution/dimensions of the voxels.
Create a hash_map to store erosion distances (distR)
Apply thinning (palagyi filter) via the applyWdist function provided by Patrick Min’s binvox and thinvox libraries. 

“Erosion distance is computed during the thinning algorithm, it is estimated by the number of the iterations which it takes to erode the shape til one-voxel wide curve.”

The code boils down to surface_area divided by the number of voxels in the skeleton. However, the surface_area appears to be calculated from the distance transformation map that is calculated by external libraries thinvox and binvox (https://www.patrickmin.com/thinvox/). So I don’t know the exact details of how the values are calculated, but if the above quote taken from a comment in the source code is correct, it is the estimated number of iterations needed to create a one-voxel wide curve.

Going off of this, the average radius may be the average number of iterations needed to create the skeleton from the PCD. Assuming that each iteration would remove 0 to 1 voxels were iteration, this would be a proxy for the radius of a curve/root in voxel units.

Implemented by [Gia3D].

### Elongation

PCA on 3D point cloud, taking the ratio between PC2 variance and PC1 variance; measures how elongated the root is. Implemented by [rootCrownImageAnalysis3D].

### Flatness

PCA on 3D point cloud, taking the ratio between PC3 variance and PC2 variance; measures how flat the root is. Implemented by [rootCrownImageAnalysis3D].

### Football

PCA on (x, y) of 3D point cloud, taking the ratio between PC2 variance and PC1 variance. Implemented by [rootCrownImageAnalysis3D].

### Biomass VHist

3D root point cloud vertical density distribution at n<sup>th</sup> cm below the top. Implemented by [New3DTraitsForRPF].

### Convex Hull VHist

Compute the convext hull for the root at each image slice first, then the trait is 3D convex hull point cloud vertical density distribution at n<sup>th</sup> cm below the top. Implemented by [New3DTraitsForRPF].

### Solidty VHist

The solidity at each slice is computed, then spline interpolated to the nt​h​ cm (1-20) below the top. Implemented by [New3DTraitsForRPF].

### DensityS

The frequency of voxels with different 6 overlap ratios from side view. S6 represents the largest overlap ratio. Higher numbers in greater overlap ratio means a denser root. Implemented by [rootCrownImageAnalysis3D].

### DensityT

The frequency of pixels with different 6 overlap ratios from top view. S6 represents the largest overlap ratio. Implemented by [rootCrownImageAnalysis3D].

### FractalDimensionS

Fractal dimension is estimated from the projected side-view image using the box-counting method. It is a measure of how complicated a root shape is using self-similarity. Implemented by [rootCrownImageAnalysis3D].

### FractalDimensionT

Fractal dimension estimated from the projected top-view image using the box-counting method. It is a measure of how complicated a root shape is using self-similarity. Implemented by [rootCrownImageAnalysis3D].

### N/CH/S Mean

Mean estimated from the distribution of biomass/volume (N), convex hull (CH), or solidity (S) along the z-axis. Implemented by [rootCrownImageAnalysis3D].

### N/CH/S Std

Standard deviation estimated from the distribution of biomass/volume (N), convex hull (CH), or solidity (S) along the z-axis. Implemented by [rootCrownImageAnalysis3D].

### N/CH/S Skewness

Skewness, or inequality, estimated from the distribution of biomass/volume (N), convex hull (CH), or solidity (S) along the z-axis. Negative value indicates that a large number of the values are lower than the mean (left-tailed); positive value indicates that a larger number of the values are higher than the mean (right-tailed). Implemented by [rootCrownImageAnalysis3D].

### N/CH/S Kurtosis

Kurtosis, or peakiness, estimated from the distribution of biomass/volume (N), convex hull (CH), or solidity (S) along the z-axis. High value indicates that the peak of the distribution around the mean is sharp and long-tailed; low value indicates that the peak around the mean is round and short-tailed. Implemented by [rootCrownImageAnalysis3D].

### N/CH/S Energy

Energy, or uniformity, estimated from the distribution of biomass/volume (N), convex hull (CH), or solidity (S) along the z-axis. A high value indicates that the distribution has a small number of different levels. Implemented by [rootCrownImageAnalysis3D].

### N/CH/S Entropy

Entropy, the inverse of energy, estimated from the distribution of biomass/volume (N), convex hull (CH), or solidity (S) along the z-axis. A high value indicates that the distribution has a higher number of different levels. Implemented by [rootCrownImageAnalysis3D].

### N/CH/S Smoothness

Smoothness estimated from the distribution of biomass/volume (N), convex hull (CH), or solidity (S) along the z-axis. Defined as <img src="https://render.githubusercontent.com/render/math?math=1-%5Cfrac%7B1%7D%7B1%2B(stddev)%5E2%7D">. Implemented by [rootCrownImageAnalysis3D].


[Gia3D]: <https://github.com/Topp-Roots-Lab/Gia3D>
[New3DTraitsForRPF]: <https://github.com/Topp-Roots-Lab/New3DTraitsForRPF/tree/standalone-kde-traits>
[rawtools]: <https://github.com/Topp-Roots-Lab/python-rawtools>
[rootCrownImageAnalysis3D]: <https://github.com/Topp-Roots-Lab/3d-root-crown-analysis-pipeline/blob/master/xrcap/rootCrownImageAnalysis3D.py>
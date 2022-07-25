#!/usr/bin/python3.8
"""
Created on Sep 20, 2018

@author: Ni Jiang
"""
import logging
import math
import os
import re
import threading
from importlib.metadata import version
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from sys import float_info

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.random import default_rng
from PIL import Image
from rawtools import dat
from scipy import interpolate
from skimage.morphology import convex_hull_image
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

__version__ = version("xrcap")


def image2Points(img, sliceID=0):
    """Convert image to points

    Args:
        img (numpy.uint8): binary image
        sliceID = relative index of image within 3-D volume

    Returns:
        tuple of 3-D position of points and number of points: (positions, point count)

    Example image (slice #26):
    img = [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]]
    img = np.array(img)

    An image like this would return the following structure for its indicies array and point count:

        (array([[ 0.,  2., 26.],
                [ 1.,  1., 26.],
                [ 1.,  2., 26.],
                [ 1.,  3., 26.],
                [ 2.,  0., 26.],
                [ 2.,  1., 26.],
                [ 2.,  4., 26.],
                [ 3.,  1., 26.],
                [ 3.,  2., 26.],
                [ 3.,  3., 26.],
                [ 4.,  2., 26.]]), 11)
    """
    indices = np.nonzero(img)
    num = len(indices[0])
    if not num == 0:
        pts = np.zeros((num, 3))
        # Set first column to row indices (X positions)
        pts[:, 0] = indices[0]
        # Set second column to column indices (Y positions)
        pts[:, 1] = indices[1]
        # Set third column to slice index (Z positions)
        pts[:, 2] = sliceID
        return pts, num
    else:
        return [], 0


def calDensity(img, rangeN):
    """"""
    img_scale = img * 255.0 / rangeN
    nZeroVal = img_scale[np.nonzero(img_scale)]
    hist, bin_edges = np.histogram(
        nZeroVal, bins=[0, 1, 5, 10, 20, 30, 255], density=True
    )
    return hist


def calFractalDim(img):
    """"""

    def boxcount(img, k):
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
            np.arange(0, img.shape[1], k),
            axis=1,
        )
        return np.count_nonzero(S)

    p = min(img.shape)
    n = 2 ** np.floor(np.log(p) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 0, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(img, size))
    # Fit
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def calStatTexture(hist):
    """"""
    bins = np.linspace(
        min(np.nonzero(hist)[0]), max(np.nonzero(hist)[0]), 33, dtype=int
    )
    logging.debug(f"{bins=}")
    hist_scale = [sum(hist[bins[x] : bins[x + 1]]) for x in range(0, 32, 1)]
    logging.debug(f"{hist_scale=}")
    probs = np.array(hist_scale, dtype=float) / sum(hist_scale)
    logging.debug(f"{probs=}")

    # NOTE(tparker): Since calculating entropy value requires division by each probability,
    # there's a chance you may divide by zero
    # There is precedent for this and is reported in
    # https://github.com/Topp-Roots-Lab/3d-root-crown-analysis-pipeline/issues/27
    # As a workaround, set probability of zero to some extremely small
    probs[probs == 0] = float_info.min
    logging.debug(f"{probs=}")

    b = list(range(1, 33, 1))
    mean = sum(probs * b)
    std = math.sqrt(sum((b - mean) ** 2 * probs))
    skewness = sum((b - mean) ** 3 * probs) / (std**3)
    kurtosis = sum((b - mean) ** 4 * probs) / (std**4)
    energy = sum(probs**2)
    entropy = -sum(probs * np.log2(probs))
    SM = 1 - 1 / (1 + std * std)
    return [mean, std, skewness, kurtosis, energy, entropy, SM]


def __find_filepath(name, path):
    """Find the realpath for the name of file

    Args:
        name (str): basename of desired file
        path (str): the starting directory to search for file

    Returns:
        str: realpath to file
    """
    for root, dirs, files in os.walk(os.path.realpath(path)):
        if name in files:
            logging.debug(os.path.join(root, name))
            return os.path.join(root, name)
    raise FileNotFoundError(name)


def validate_dat_metadata(args):
    """"""
    # For each provided threshold images path...
    dat_filepaths = []
    for fp in args.path:
        for root, dirs, files in os.walk(os.path.realpath(fp)):
            for subfolder in dirs:
                # Find folder that contains RAW and DAT files
                parent_folder = (
                    os.path.dirname(fp)
                    if not fp.endswith("/")
                    else os.path.dirname(os.path.dirname(fp))
                )
                basename = os.path.basename(root).split("_thresholded_images")[0]
                expected_data_directory = os.path.join(
                    parent_folder, basename
                )  # /path/to/data_thresholded_images -> /path/to/data
                logging.debug(f"{expected_data_directory=}")
                dat_filepath = __find_filepath(
                    f"{subfolder}.dat", expected_data_directory
                )
        dat_filepaths.append(dat_filepath)
    logging.debug(f"Located DAT files: {dat_filepaths}")
    # Validate DAT files
    for dat_fp in dat_filepaths:
        metadata = dat.read(dat_fp)


def plotTrait(values, uuid, trait, latent, axis=(1, 0)):
    PROBABILITY_KEPT = 0.01
    print(f"{values=}")

    subset_count = math.floor(PROBABILITY_KEPT * len(values))
    print(f"Selecting '{subset_count}' data points from '{len(values)}' total samples.")
    rng = default_rng()
    vals = rng.choice(values, size=subset_count, replace=False)

    xs = vals[:, axis[0]]
    ys = vals[:, axis[1]]
    fig = go.Figure(
        data=go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=1)),
        layout=go.Layout(
            title=go.layout.Title(text=f"{trait.capitalize()} for {uuid}")
        ),
    )
    fig.update_layout(xaxis_title=latent[0], yaxis_title=latent[1])
    fig.write_image(f"/home/tparker/Desktop/{uuid}.{trait}.png")


def process(args, fp, subfolder, thickness, scale, depth, pos, pbar_position):
    for s_root, s_dirs, s_files in os.walk(os.path.join(fp, subfolder)):
        # Get initial conditions and sizes from first image found
        img = cv.imread(os.path.join(fp, subfolder, s_files[0]), cv.IMREAD_GRAYSCALE)
        img_files = [f for f in s_files if f.endswith(".png")]
        maximum_number_of_points = len(img_files) * img.shape[0] * img.shape[1]
        chunksize = (
            (maximum_number_of_points * 20) // 100 // 3
        )  # Get 20% of the max points in a volume, and then a third of that for the column dimension for "all_pts"
        logging.debug(f"{maximum_number_of_points=}")
        logging.debug(f"{chunksize=}")
        c_all_pts = 0  # count of found points for volume
        c_all_pts_ch = 0  # count of found points in convex hulls

        # Sort any binary images found
        s_files.sort(key=lambda x: (-x.count("/"), x), reverse=False)
        z = 1
        all_pts = np.empty((chunksize, 3))  # all points for each slice in 3-D space
        all_pts_ch = np.empty(
            (chunksize, 3)
        )  # all points of convex hull of each slice in 3-D sapce
        num_hist = []
        num_ch_hist = []
        solidity = []

        bw_S1 = np.zeros((img.shape[1], 1))  # side projection (binary - side A)
        bw_S2 = np.zeros((img.shape[0], 1))  # side projection (binary - side B)
        im_S1 = np.zeros(
            (img.shape[1], 1), dtype=np.uint16
        )  # side projection (grayscale - side A)
        im_S2 = np.zeros(
            (img.shape[0], 1), dtype=np.uint16
        )  # side projection (grayscale - side B)
        # NOTE(tparker): Have to cast to uint8 after migration to Python3.8. Default dtype is float64 in Py3.
        bw_T = (img / 255).astype("uint8")  # top-down projection (binary)
        im_T = np.zeros(
            img.shape, dtype=np.uint16
        )  # top-down projection (grayscale - additive)
        # for img_name in s_files:
        # I.e., For each binary image...
        for img_name in tqdm(
            s_files,
            desc=f"Processing '{subfolder}'",
            position=pbar_position,
            leave=False,
        ):
            if os.path.splitext(img_name)[1] == ".png":
                # Read in binary image as grayscale and then conver to true binary with thresholding
                img = cv.imread(
                    os.path.join(fp, subfolder, img_name), cv.IMREAD_GRAYSCALE
                )
                retval, img = cv.threshold(img, 0, 1, cv.THRESH_BINARY)
                # Count the number of white pixels and convert them to an array of 3-D points
                pts, num = image2Points(img, z)

                # When at least one pixel is found...
                if num > 0:
                    # Allocate more memory if the number of points in the current slice
                    # would extend beyond the current allocated space
                    if len(all_pts) <= num + c_all_pts:
                        all_pts.resize(
                            (max(len(all_pts) * 2, num), 3)
                        )  # double allocated space
                    # Assign new points to positions in container for all points
                    all_pts[c_all_pts : c_all_pts + num] = pts
                    c_all_pts += num

                    chull = convex_hull_image(img)
                    pts_ch, num_ch = image2Points(chull, z)
                    # Allocate more memory if the number of points in the current slice
                    # would extend beyond the current allocated space
                    if len(all_pts_ch) <= num_ch + c_all_pts_ch:
                        all_pts_ch.resize(
                            max(len(all_pts_ch) * 2, num_ch), 3
                        )  # double allocated space
                    # Assign new points to positions in container for all points
                    all_pts_ch[c_all_pts_ch : c_all_pts_ch + num_ch] = pts_ch
                    c_all_pts_ch += num_ch

                    num_hist.append(num)
                    num_ch_hist.append(num_ch)
                    solidity.append(float(num) / num_ch)
                else:
                    num_hist.append(0)
                    num_ch_hist.append(0)
                    solidity.append(0.0)

                bw_S1 = np.append(bw_S1, np.amax(img, axis=0)[:, None], axis=1)
                bw_S2 = np.append(bw_S2, np.amax(img, axis=1)[:, None], axis=1)
                im_S1 = np.append(
                    im_S1, np.sum(img, axis=0, dtype=np.uint16)[:, None], axis=1
                )
                im_S2 = np.append(
                    im_S2, np.sum(img, axis=1, dtype=np.uint16)[:, None], axis=1
                )
                bw_T = cv.bitwise_or(bw_T, img)
                im_T += img

                z += 1

        # Resize all_pts to the minimum space required
        all_pts.resize((c_all_pts, 3))
        all_pts_ch.resize((c_all_pts_ch, 3))

        # Calculating the biomass and convex hull for a volume is computationally expensive (time)
        # Therefore, only perform the calculations if enabled
        if args.kde:
            import asyncio

            async def process_kde_with_matlab(cmd):

                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await proc.communicate()
                logging.debug(f"[{cmd!r} exited with {proc.returncode}]")
                if stdout:
                    vhist_pattern = r".*(?P<vhist_type>((biomass)|(convexHull))_vhist)(?P<vhist_number>\d+)\s+(?P<vhist_value>[\d\.]+)"
                    output = stdout.decode().split("\n")
                    biomass_vhist = []
                    convexhull_vhist = []
                    for line in output:
                        m = re.match(vhist_pattern, line)
                        if m is not None:
                            if "biomass" in m.group("vhist_type"):
                                biomass_vhist.append(m.group("vhist_value"))
                            elif "convex" in m.group("vhist_type"):
                                convexhull_vhist.append(m.group("vhist_value"))
                    logging.debug(biomass_vhist)
                    logging.debug(convexhull_vhist)
                    return biomass_vhist, convexhull_vhist
                if stderr:
                    logging.error(f"[stderr]\n{stderr.decode()}")

            biomass_hist, convexhull_hist = asyncio.run(
                process_kde_with_matlab(
                    f"kde-traits '{os.path.join(fp, subfolder)}' {thickness} {args.sampling} {args.depth}"
                )
            )

        if len(solidity) < depth:
            solidity = np.append(
                solidity, np.zeros(int(depth - len(solidity)))
            )  # pad with zeros for missing depth values

        # Generate an interpolation function (1-D) that maps from [1, N] to the actual solidity values
        # Use the slice index based on percentage of the volume
        # Currently, the solidity is the cummulative measurements by 5% increments of the volume (assumed vertical)
        solidity_hist = interpolate.interp1d(
            np.arange(1, len(solidity) + 1), solidity, kind="cubic"
        )(pos)

        pca = PCA(n_components=3)
        pca_data = pca.fit(all_pts)
        latent = pca_data.explained_variance_
        elong = math.sqrt(latent[1] / latent[0])

        pca_data = pca.transform(all_pts)
        plotTrait(
            pca_data,
            subfolder,
            trait="elongation",
            axis=(1, 0),
            latent=(latent[1], latent[0]),
        )

        flat = math.sqrt(latent[2] / latent[1])
        plotTrait(
            pca_data,
            subfolder,
            trait="flatness",
            axis=(2, 1),
            latent=(latent[2], latent[1]),
        )

        pca = PCA(n_components=2)
        latent = pca.fit(all_pts[:, [0, 1]]).explained_variance_
        pca_data = pca.transform(all_pts[:, [0, 1]])
        football = math.sqrt(latent[1] / latent[0])
        plotTrait(
            pca_data,
            subfolder,
            trait="football",
            axis=(1, 0),
            latent=(latent[1], latent[0]),
        )

        bw_S1 = np.delete(bw_S1, 0, 1)
        bw_S2 = np.delete(bw_S2, 0, 1)
        im_S1 = np.delete(im_S1, 0, 1)
        im_S2 = np.delete(im_S2, 0, 1)
        width_S1 = np.amax(np.nonzero(bw_S1)[0]) - np.amin(np.nonzero(bw_S1)[0]) + 1
        width_S2 = np.amax(np.nonzero(bw_S2)[0]) - np.amin(np.nonzero(bw_S2)[0]) + 1
        depth_S = np.amax(np.nonzero(bw_S1)[1]) - np.amin(np.nonzero(bw_S1)[1]) + 1

        densityS1 = calDensity(im_S1, width_S2)
        densityS2 = calDensity(im_S2, width_S1)
        densityT = calDensity(im_T, depth_S)
        FD_S1 = calFractalDim(bw_S1)
        FD_S2 = calFractalDim(bw_S2)
        FD_T = calFractalDim(bw_T)

        num_hist_texture = calStatTexture(num_hist)
        num_ch_hist_texture = calStatTexture(num_ch_hist)
        solidity_hist_texture = calStatTexture(solidity)

        traits = {}
        traits["FileName"] = subfolder
        traits["Pipeline Version"] = __version__
        traits["Scale"] = scale
        traits["Elongation"] = elong
        traits["Flatness"] = flat
        traits["Football"] = football

        kde_steps = math.floor(args.depth / 10) + 1
        if args.kde:
            for i in range(1, kde_steps):
                traits[f"Biomass_vhist{i}"] = biomass_hist[i - 1]
            for i in range(1, kde_steps):
                traits[f"Convexhull_vhist{i}"] = convexhull_hist[i - 1]

        solidity_hist = np.squeeze(solidity_hist)
        for i in range(1, kde_steps):
            traits[f"Solidity_vhist{i}"] = solidity_hist[i - 1]

        densityS = (densityS1 + densityS2) / 2
        for i in range(1, 7):
            traits[f"Density_S{i}"] = densityS[i - 1]

        for i in range(1, 7):
            traits[f"Density_T{i}"] = densityT[i - 1]

        FractalDimension_S, FractalDimension_T = [(FD_S1 + FD_S2) / 2, FD_T]
        traits["FractalDimension_S"] = FractalDimension_S
        traits["FractalDimension_T"] = FractalDimension_T
        for i, trait_name in enumerate(
            [
                "N_Mean",
                "N_Std",
                "N_Skewness",
                "N_Kurtosis",
                "N_Energy",
                "N_Entropy",
                "N_Smoothness",
            ]
        ):
            traits[trait_name] = num_hist_texture[i]
        for i, trait_name in enumerate(
            [
                "CH_Mean",
                "CH_Std",
                "CH_Skewness",
                "CH_Kurtosis",
                "CH_Energy",
                "CH_Entropy",
                "CH_Smoothness",
            ]
        ):
            traits[trait_name] = num_ch_hist_texture[i]

        for i, trait_name in enumerate(
            [
                "S_Mean",
                "S_Std",
                "S_Skewness",
                "S_Kurtosis",
                "S_Energy",
                "S_Entropy",
                "S_Smoothness",
            ]
        ):
            traits[trait_name] = solidity_hist_texture[i]

    return traits


def main(args):
    """Perform image analysis on binary images"""
    # Disable debug statements from matplotlib.font_manager
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # When not slice thickness is provided, try to extract it from .DAT
    if not args.thickness:
        logging.debug(
            f"Slice thickness was not provided. Extracting information from .DAT files."
        )
        validate_dat_metadata(args)

    for fp in args.path:
        list_dirs = os.walk(fp)
        results = []
        # For each subdirectory in the binary images folder... (i.e., for each volume...)
        volumes = [
            subfolder
            for subfolder in [dirs for root, dirs, files in os.walk(fp)]
            if subfolder != []
        ][0]
        pbar = tqdm(total=len(volumes), desc="Overall progress", position=0)

        def async_callback(*response):
            results.append(response[0])  # response is returned as a tuple
            pbar.update()

        def async_error_callback(*err):
            logging.error(err)

        slicethicknessCubicFlag = False
        flaggedVolumes = []
        with ThreadPool(args.threads) as p:
            for root, dirs, files in list_dirs:
                for pbar_position, subfolder in enumerate(dirs, start=1):
                    logging.debug(f"Processing {subfolder}")

                    # When not slice thickness is provided, try to extract it from .DAT
                    if args.thickness is None:
                        # Find folder that contains RAW and DAT files
                        parent_folder = root
                        basename = os.path.basename(root).split("_thresholded_images")[
                            0
                        ]
                        while "thresholded_images" in parent_folder:
                            parent_folder = os.path.dirname(parent_folder)
                        expected_data_directory = os.path.join(
                            parent_folder, basename
                        )  # /path/to/data_thresholded_images -> /path/to/data
                        logging.debug(f"{expected_data_directory=}")
                        dat_filepath = __find_filepath(
                            f"{subfolder}.dat", expected_data_directory
                        )
                        metadata = dat.read(dat_filepath)
                        logging.debug(f"Loaded metadata from DAT: {metadata}")
                        if not (
                            metadata["x_thickness"]
                            == metadata["y_thickness"]
                            == metadata["z_thickness"]
                        ):
                            slicethicknessCubicFlag = True
                            flaggedVolumes.append(subfolder)
                            logging.debug(
                                f"Slice thickness for '{subfolder}' are not the same. {metadata['x_thickness']=}, {metadata['y_thickness']=}, {metadata['z_thickness']=}"
                            )
                        thickness = round(float(metadata["z_thickness"]), 3)
                    else:
                        thickness = args.thickness
                    logging.debug(args)
                    logging.debug(f"{thickness=}")
                    # Account for downsampling during preprocessing
                    # If half the images were used, double the thickness per 'slice'
                    scale = float(args.sampling) * float(thickness)
                    logging.debug(f"Scale set to '{scale}'")
                    ## Changed (round)(200/scale) because in Python2 round will produce a float - (ex. 952.0) and now makes integer 952
                    # Calculate the number of expected images
                    kde_depth = args.depth
                    kde_depth_cm = math.floor(kde_depth / 10)
                    depth = int((round)(kde_depth / scale))
                    logging.debug(f"{depth=}")
                    # Create a list of evenly spaced numbers based on the depth
                    # NOTE(tparker): Added extra forward slash to preserve integer division for migration from Python 2.7
                    pos = np.linspace(depth // kde_depth_cm, depth, kde_depth_cm)[
                        :, None
                    ]
                    logging.debug(pos)

                    p.apply_async(
                        process,
                        args=(
                            args,
                            fp,
                            subfolder,
                            thickness,
                            scale,
                            depth,
                            pos,
                            pbar_position,
                        ),
                        callback=async_callback,
                        error_callback=async_error_callback,
                    )

                p.close()
                p.join()

        # If the output file does not exist, initialize it with a header
        results = pd.DataFrame(results, columns=list(results[0].keys()))
        out_filename = os.path.join(fp, "traits.csv")
        # If no file exists yet, inform user where it will be generated
        if not os.path.exists(out_filename):
            logging.debug(f"Create output file: {out_filename}")
            results.to_csv(out_filename, index=False)
        # Otherwise, either overwrite file or skip
        else:
            if args.force:
                logging.warning(f"File already exists '{out_filename}'. Overwriting.")
                results.to_csv(out_filename, index=False)
            else:
                ans = input(
                    f"Output file already exists. Do you wish to overwrite it? [y/N]: "
                )
                if ans.lower() == "y":
                    results.to_csv(out_filename, index=False)
                else:
                    pd.set_option("display.max_columns", None)
                    pd.set_option("display.max_rows", None)
                    logging.debug(results, result=False)
                    logging.info(
                        f"Results not saved to file. Unformatted results saved to log."
                    )

    if slicethicknessCubicFlag:
        logging.warning(
            f"The slicethickness value for at least one volume was found to be not exactly equal. Check the log for details. The following volumes were flagged: {flaggedVolumes}"
        )
    return 0

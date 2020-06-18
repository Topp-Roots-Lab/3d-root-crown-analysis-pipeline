#!/usr/bin/python3.8
'''
Created on Sep 20, 2018

@author: Ni Jiang
'''
import argparse
import logging
import math
import os
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count
from time import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import interpolate
from skimage.morphology import convex_hull_image
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from __init__ import __version__
from raw_utils.raw_utils.core.metadata import read_dat
from utils import configure_logging


def options():
	parser = argparse.ArgumentParser(description='Root Crown Image Analysis', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
	parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
	parser.add_argument("-f", '--force', action="store_true", help="(Not yet implemented) Force file creation. Overwrite any existing files.")
	parser.add_argument('-s', "--sampling", default=2, help="resolution parameter")
	parser.add_argument('-t', "--thickness", type=float, help="slice thickness in mm")
	parser.add_argument("--threads", type=int, default=cpu_count(), help=f"Maximum number of threads dedicated to processing.")
	parser.add_argument("--biomass", action="store_true", help="Enable calculation for biomass")
	parser.add_argument("--convexhull", action="store_true", help="Enable calculation for convex hull")
	parser.add_argument("path", metavar='input_folder', type=str, nargs=1, help='Input directory to process')
	args = parser.parse_args()

	# Make sure user does not request more CPUs can available
	if args.threads > cpu_count():
		args.threads = cpu_count()

	args.path = list(set(args.path)) # remove any duplicates

	args.module_name = f"{os.path.splitext(os.path.basename(__file__))[0]}"
	configure_logging(args, ifp=os.path.basename(os.path.realpath(args.path[0])))

	return args

def image2Points(img, sliceID = 0):
	""""""
	indices = np.nonzero(img)
	num = len(indices[0])
	if not num == 0:
		pts = np.zeros((num, 3))
		# Set first column to row indices
		pts[:, 0] = indices[0]
		# Set second column to column indices
		pts[:, 1] = indices[1]
		# Set third column to slice index
		pts[:, 2] = sliceID
		return pts, num
	else:
		return [], 0

def calDensity(img, rangeN):
	""""""
	img_scale = img*255.0/rangeN
	nZeroVal =  img_scale[np.nonzero(img_scale)]
	hist, bin_edges = np.histogram(nZeroVal, bins = [0, 1, 5, 10, 20, 30, 255], density = True)
	return hist

def calFractalDim(img):
	""""""
	def boxcount(img, k):
		S = np.add.reduceat(
			np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
							   np.arange(0, img.shape[1], k), axis=1)
		return np.count_nonzero(S)

	p = min(img.shape)
	n = 2**np.floor(np.log(p)/np.log(2))
	n = int(np.log(n)/np.log(2))
	sizes = 2**np.arange(n, 0, -1)

	# Actual box counting with decreasing size
	counts = []
	for size in sizes:
		counts.append(boxcount(img, size))
	# Fit
	coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
	return -coeffs[0]

def calStatTexture(hist):
	""""""
	bins = np.linspace(min(np.nonzero(hist)[0]), max(np.nonzero(hist)[0]), 33, dtype = int)
	hist_scale = [sum(hist[bins[x]:bins[x+1]]) for x in range(0, 32, 1)]
	probs = np.array(hist_scale, dtype = float)/sum(hist_scale)
	b = list(range(1, 33, 1))
	mean = sum(probs*b)
	std = math.sqrt(sum((b-mean)**2*probs))
	skewness = sum((b-mean)**3*probs)/(std**3)
	kurtosis = sum((b-mean)**4*probs)/(std**4)
	energy = sum(probs**2)
	entropy = -sum(probs*np.log2(probs))
	SM = 1 - 1/(1+std*std)
	return [mean, std, skewness, kurtosis, energy, entropy, SM]

def __find_filepath(name, path):
	""""""
	for root, dirs, files in os.walk(path):
		if name in files:
			logging.debug(os.path.join(root, name))
			return os.path.join(root, name)
	raise FileNotFoundError(name)

def validate_dat_metadata(args):
	""""""
	# For each provided threshold images path...
	dat_filepaths = []
	for fp in args.path:
		# Find the corresponding RAW/DAT folder
		for root, dirs, files in os.walk(fp):
			for volume_directory_basename in dirs:
				dat_filepath = __find_filepath(f"{volume_directory_basename}.dat", os.path.dirname(fp))
				logging.debug(f"Expected DAT filepath: '{dat_filepath}'")
				dat_filepaths.append(dat_filepath)
	logging.debug(f"Located DAT files: {dat_filepaths}")
	# Validate DAT files
	for dat_fp in dat_filepaths:
		metadata = read_dat(dat_fp)

if __name__ == "__main__":
	args = options()
	start_time = time()

	# Disable debug statements from matplotlib.font_manager
	logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

	# When not slice thickness is provided, try to extract it from .DAT
	if not args.thickness:
		validate_dat_metadata(args)

	for fp in args.path:

		original_folder = fp
		list_dirs = os.walk(original_folder)
		field = []

		field.extend(['FileName'])
		field.extend(['Pipeline_version'])
		# Mao's Traits
		field.extend(['Elongation', 'Flatness', 'Football'])

		if args.biomass:
			field.extend(['Biomass_vhist{}'.format(i) for i in range(1, 21)])
		if args.convexhull:
			field.extend(['Convexhull_vhist{}'.format(i) for i in range(1, 21)])
		field.extend(['Solidity_vhist{}'.format(i) for i in range(1, 21)])
		field.extend(['Density_S{}'.format(i) for i in range(1, 7)])
		field.extend(['Density_T{}'.format(i) for i in range(1, 7)])
		field.extend(['FractalDimension_S', 'FractalDimension_T'])
		field.extend(['N_Mean', 'N_Std', 'N_Skewness', 'N_Kurtosis', 'N_Energy', 'N_Entropy', 'N_Smoothness'])
		field.extend(['CH_Mean', 'CH_Std', 'CH_Skewness', 'CH_Kurtosis', 'CH_Energy', 'CH_Entropy', 'CH_Smoothness'])
		field.extend(['S_Mean', 'S_Std', 'S_Skewness', 'S_Kurtosis', 'S_Energy', 'S_Entropy', 'S_Smoothness'])

		# If the output file does not exist, initialize it with a header
		out_filename = os.path.join(original_folder, 'traits.csv')
		if not os.path.exists(out_filename):
			logging.info(f"Create output file: {out_filename}")
			out_file = open(out_filename, "a+")
			np.savetxt(out_file, np.array(field).reshape(1,  np.array(field).shape[0]), fmt='%s',  delimiter=',')
		else:
			out_file = open(out_filename, "a+")


		# For each subdirectory in the binary images folder...
		for root, dirs, files in list_dirs:
			for subfolder in dirs:
				volume_start_time = time()
				logging.debug(f"Processing {subfolder}")

				# When not slice thickness is provided, try to extract it from .DAT
				if args.thickness is None:
					# Find folder that contains RAW and DAT files
					expected_data_directory = os.path.join(os.path.dirname(root), os.path.basename(root).replace('_thresholded_images', '')) # /path/to/data_thresholded_images -> /path/to/data
					logging.debug(f"{expected_data_directory=}")
					dat = read_dat(__find_filepath(f"{subfolder}.dat", expected_data_directory))
					logging.debug(f"Loaded metadata from DAT: {dat}")
					if not (dat["x_thickness"] == dat["y_thickness"] == dat["z_thickness"]):
						logging.warning(f"Slice thickness for '{subfolder}' are not the same. {dat['x_thickness']=}, {dat['y_thickness']=}, {dat['z_thickness']=}")
					args.thickness = round(float(dat["z_thickness"]),3)
				logging.debug(f"Slice thickness set to '{args.thickness}'")

				# Account for downsampling during preprocessing
				# If half the images were used, double the thickness per 'slice'
				scale = float(args.sampling)*float(args.thickness)
				logging.debug(f"Scale set to '{scale}'")
				##Changed (round)(200/scale) because in Python2 round will produce a float - (ex. 952.0) and now makes integer 952
				# Calculate the number of expected images
				depth = int((round)(200/scale))
				# Create a list of evenly spaced numbers based on the depth
				pos = np.linspace(depth/20, depth, 20)[:, None]

				traits = []
				for s_root, s_dirs, s_files in os.walk(os.path.join(original_folder, subfolder)):
					# Sort any binary images found
					s_files.sort(key=lambda x: (-x.count('/'), x), reverse = False)
					z = 1
					all_pts = np.empty((1, 3))
					all_pts_ch = np.empty((1, 3))
					num_hist = []
					num_ch_hist = []
					solidity = []

					# Get initial conditions and sizes from first image found
					img = cv.imread(os.path.join(original_folder, subfolder, s_files[0]), cv.IMREAD_GRAYSCALE)
					bw_S1 = np.empty((img.shape[1], 1))
					bw_S2 = np.empty((img.shape[0], 1))
					im_S1 = np.empty((img.shape[1], 1), dtype = np.uint16)
					im_S2 = np.empty((img.shape[0], 1), dtype = np.uint16)
					# NOTE(tparker): Have to cast to uint8 after migration to Python3.8. Default dtype is float64.
					bw_T = (img/255).astype('uint8')
					im_T = np.empty(img.shape, dtype = np.uint16)
					# for img_name in s_files:
					for img_name in tqdm(s_files, desc=f"Processing '{subfolder}'"):
						if os.path.splitext(img_name)[1] == '.png':
							img = cv.imread(os.path.join(original_folder, subfolder, img_name), cv.IMREAD_GRAYSCALE)
							retval, img = cv.threshold(img, 0, 1, cv.THRESH_BINARY)
							pts, num = image2Points(img, z)
							if num > 0:
								chull = convex_hull_image(img)
								pts_ch, num_ch = image2Points(chull, z)
								all_pts = np.append(all_pts, pts, axis = 0)
								all_pts_ch = np.append(all_pts_ch, pts_ch, axis = 0)
								num_hist.append(num)
								num_ch_hist.append(num_ch)
								solidity.append(float(num)/num_ch)
							else:
								num_hist.append(0)
								num_ch_hist.append(0)
								solidity.append(.0)

							#print z

							bw_S1 = np.append(bw_S1, np.amax(img, axis = 0)[:, None], axis = 1)
							bw_S2 = np.append(bw_S2, np.amax(img, axis = 1)[:, None], axis = 1)
							im_S1 = np.append(im_S1, np.sum(img, axis = 0, dtype = np.uint16)[:, None], axis = 1)
							im_S2 = np.append(im_S2, np.sum(img, axis = 1, dtype = np.uint16)[:, None], axis = 1)
							bw_T = cv.bitwise_or(bw_T, img)
							im_T += img
							z += 1


					if args.biomass or args.convexhull:
						kde = KernelDensity(kernel = 'gaussian', bandwidth = 20).fit(all_pts[:, 2][:, None])

					if args.biomass:
						biomass_hist = np.exp(kde.score_samples(pos))

					if args.convexhull:
						convexhull_hist = np.exp(kde.score_samples(pos))

					if len(solidity) < depth:
						solidity = np.append(solidity, np.zeros(int(depth-len(solidity)))) # pad with zeros for missing depth values

					# Generate an interpolation function (1-D) that maps from [1, N] to the actual solidity values
					# Use the slice index based on percentage of the volume
					# Currently, the solidity is the cummulative measurements by 5% increments of the volume (assumed vertical)
					solidity_hist = interpolate.interp1d(np.arange(1, len(solidity)+1), solidity, kind = 'cubic')(pos)

					pca = PCA(n_components = 3)
					latent = pca.fit(all_pts).explained_variance_
					elong=math.sqrt(latent[1]/latent[0])
					flat=math.sqrt(latent[2]/latent[1])

					pca = PCA(n_components = 2)
					latent = pca.fit(all_pts[:, [0, 1]]).explained_variance_
					football=math.sqrt(latent[1]/latent[0])

					bw_S1 = np.delete(bw_S1, 0, 1)
					bw_S2 = np.delete(bw_S2, 0, 1)
					im_S1 = np.delete(im_S1, 0, 1)
					im_S2 = np.delete(im_S2, 0, 1)
					width_S1 = np.amax(np.nonzero(bw_S1)[0]) - np.amin(np.nonzero(bw_S1)[0]) + 1
					width_S2 = np.amax(np.nonzero(bw_S2)[0]) - np.amin(np.nonzero(bw_S2)[0]) + 1
					depth_S = np.amax(np.nonzero(bw_S1)[1]) - np.amin(np.nonzero(bw_S1)[1]) + 1

					densityS1 = calDensity(im_S1, width_S2)
					densityS2 = calDensity(im_S2, width_S1)
					densityT =calDensity(im_T, depth_S)
					FD_S1 = calFractalDim(bw_S1)
					FD_S2 = calFractalDim(bw_S2)
					FD_T = calFractalDim(bw_T)

					num_hist_texture = calStatTexture(num_hist)
					num_ch_hist_texture = calStatTexture(num_ch_hist)
					solidity_hist_texture = calStatTexture(solidity)

					traits.extend([subfolder])
					traits.extend([__version__])
					traits.extend([elong, flat, football])
					if args.biomass:
						traits.extend(biomass_hist)
					if args.convexhull:
						traits.extend(convexhull_hist)
					traits.extend(np.squeeze(solidity_hist))
					traits.extend((densityS1 + densityS2)/2)
					traits.extend(densityT)
					traits.extend([(FD_S1 + FD_S2)/2, FD_T])
					traits.extend(num_hist_texture)
					traits.extend(num_ch_hist_texture)
					traits.extend(solidity_hist_texture)

					np.savetxt(out_file, np.array(traits).reshape(1, np.array(traits).shape[0]), fmt='%s', delimiter=',')
					logging.debug(f"Done! Processing '{subfolder}' complete. Individual execution time: {time() - volume_start_time} seconds.")

		logging.info(f"Done! Total execution time: {time() - start_time} seconds.")
		out_file.close()

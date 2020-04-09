#!/usr/bin/python2
'''
Created on Sep 20, 2018

@author: njiang
'''

import os, argparse
import numpy as np
import cv2 as cv
from skimage.morphology import convex_hull_image
from scipy import interpolate
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import math


def image2Points(img, sliceID = 0):
    indices = np.nonzero(img)
    num = len(indices[0])
    if not num == 0:
        pts = np.zeros((num, 3))
        pts[:, 0] = indices[0]
        pts[:, 1] = indices[1]
        pts[:, 2] = sliceID
        return pts, num
    else:
        return [], 0

def calDensity(img, rangeN):
    img_scale = img*255.0/rangeN
    nZeroVal =  img_scale[np.nonzero(img_scale)]
    hist, bin_edges = np.histogram(nZeroVal, bins = [0, 1, 5, 10, 20, 30, 255], density = True)
    return hist

def calFractalDim(img):

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

    bins = np.linspace(min(np.nonzero(hist)[0]), max(np.nonzero(hist)[0]), 33, dtype = int)
    hist_scale = [sum(hist[bins[x]:bins[x+1]]) for x in range(0, 32, 1)]
    probs = np.array(hist_scale, dtype = float)/sum(hist_scale)
    b = range(1, 33, 1)
    mean = sum(probs*b)
    std = math.sqrt(sum((b-mean)**2*probs))
    skewness = sum((b-mean)**3*probs)/(std**3)
    kurtosis = sum((b-mean)**4*probs)/(std**4)
    energy = sum(probs**2)
    entropy = -sum(probs*np.log2(probs))
    SM = 1 - 1/(1+std*std)
    return [mean, std, skewness, kurtosis, energy, entropy, SM]

def options():

    parser = argparse.ArgumentParser(description='Root Crown Image Analysis',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', "--input_folder", help="directory of image slices", required=True)
    parser.add_argument('-s', "--sampling", help="resolution parameter", required=True)
    parser.add_argument('-t', "--thickness", help="slice thickness in mm", required=True)

    args = parser.parse_args()

    return args

args = options()
original_folder = args.input_folder
scale = float(args.sampling)*float(args.thickness)
##Changed (round)(200/scale) because in Python2 round will produce a float - (ex. 952.0) and now makes integer 952
depth = int((round)(200/scale))
#pos = range(depth/20, depth, depth/20)
pos = np.linspace(depth/20, depth, 20)[:, None]

list_dirs = os.walk(original_folder)
out_filename = os.path.join(original_folder, 'traits.csv')
out_file = open(out_filename, "a+")
field = []

field.extend(['FileName'])
field.extend(['Elongation', 'Flatness', 'Football'])
#field.extend(['Biomass_vhist{}'.format(i) for i in range(1, 21)])
#field.extend(['Convexhull_vhist{}'.format(i) for i in range(1, 21)])
field.extend(['Solidity_vhist{}'.format(i) for i in range(1, 21)])
field.extend(['Density_S{}'.format(i) for i in range(1, 7)])
field.extend(['Density_T{}'.format(i) for i in range(1, 7)])
field.extend(['FractalDimension_S', 'FractalDimension_T'])
field.extend(['N_Mean', 'N_Std', 'N_Skewness', 'N_Kurtosis', 'N_Energy', 'N_Entropy', 'N_Smoothness'])
field.extend(['CH_Mean', 'CH_Std', 'CH_Skewness', 'CH_Kurtosis', 'CH_Energy', 'CH_Entropy', 'CH_Smoothness'])
field.extend(['S_Mean', 'S_Std', 'S_Skewness', 'S_Kurtosis', 'S_Energy', 'S_Entropy', 'S_Smoothness'])

np.savetxt(out_file, np.array(field).reshape(1,  np.array(field).shape[0]), fmt='%s',  delimiter=',')

for root, dirs, files in list_dirs:
    for subfolder in dirs:
        print subfolder+'......'
        traits = []
        for s_root, s_dirs, s_files in os.walk(os.path.join(original_folder, subfolder)):
            s_files.sort(key=lambda x: (-x.count('/'), x), reverse = False)
            z = 1
            all_pts = np.empty((1, 3))
            all_pts_ch = np.empty((1, 3))
            num_hist = []
            num_ch_hist = []
            solidity = []
            img = cv.imread(os.path.join(original_folder, subfolder, s_files[0]), cv.IMREAD_GRAYSCALE)
            bw_S1 = np.empty((img.shape[1], 1))
            bw_S2 = np.empty((img.shape[0], 1))
            im_S1 = np.empty((img.shape[1], 1), dtype = np.uint16)
            im_S2 = np.empty((img.shape[0], 1), dtype = np.uint16)
            bw_T = img/255
            im_T = np.empty((img.shape[0], img.shape[1]), dtype = np.uint16)
            for img_name in s_files:
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


            #kde = KernelDensity(kernel = 'gaussian', bandwidth = 20).fit(all_pts[:, 2][:, None])
            #biomass_hist = np.exp(kde.score_samples(pos))



            #kde = KernelDensity(kernel = 'gaussian', bandwidth = 20).fit(all_pts_ch[:, 2][:, None])
            #convexhull_hist = np.exp(kde.score_samples(pos))


            if len(solidity) < depth:
                solidity = np.append(solidity, np.zeros(int(depth-len(solidity))))
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
            traits.extend([elong, flat, football])
            #traits.extend(biomass_hist)
            #traits.extend(convexhull_hist)
            traits.extend(np.squeeze(solidity_hist))
            traits.extend((densityS1 + densityS2)/2)
            traits.extend(densityT)
            traits.extend([(FD_S1 + FD_S2)/2, FD_T])
            traits.extend(num_hist_texture)
            traits.extend(num_ch_hist_texture)
            traits.extend(solidity_hist_texture)

            np.savetxt(out_file, np.array(traits).reshape(1,  np.array(traits).shape[0]), fmt='%s',  delimiter=',')
            print 'DONE!'

out_file.close()





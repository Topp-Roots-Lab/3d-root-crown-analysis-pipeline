'''
Created on Jul 27, 2018

@author: njiang
'''

import os, argparse
import numpy as np
from PIL import Image

def options():
    
    parser = argparse.ArgumentParser(description='Convert .raw 3d volume file to typical image format slices',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', "--input_folder", help="directory of .raw files", required=True)
    parser.add_argument('-f', "--format", help="image format", default='png')    
    
    args = parser.parse_args()

    return args

args = options()

folder = args.input_folder
parent_path = os.path.dirname(folder)

list_dirs = os.walk(folder)
for root, dirs, files in list_dirs:
    for filename in files:
        basename, extension = os.path.splitext(filename)
        if extension == '.raw':
            dat_filename = basename + '.dat'
            imgs_dir = os.path.join(folder, basename)
            if not os.path.exists(imgs_dir):
                os.makedirs(imgs_dir)
            f_info = open(os.path.join(folder, dat_filename), 'r')
            line = f_info.readline()
            line = f_info.readline()
            dims = [int(s) for s in line.split() if s.isdigit()]
            x = dims[0]
            y = dims[1]
            z = dims[2]
            digits = len(str(z))
            num_format = '{:0'+str(digits)+'d}'
            
            img_size = x*y
            offset = 2*img_size
            
            f_data = open(os.path.join(folder, filename), 'rb')
            
            for i in range(0, z):
                f_data.seek(i*offset)
                slice = np.fromfile(f_data, dtype = 'uint16', count = img_size, sep = "")
                slice = slice.reshape([y, x])
                imgname = basename + '_' + num_format.format(i)
                if args.format == 'png':
                    slice1 = slice*255.0/65535.0
                    slice1 = np.floor(slice1)
                    Image.fromarray(slice1.astype('uint8')).save(os.path.join(imgs_dir, imgname+'.png'))
                if args.format == 'tif':
                    Image.fromarray(slice.astype('uint16')).save(os.path.join(imgs_dir, imgname+'.tif'))
                        

'''
raw_filename = args.input_folder
basename = os.path.basename(raw_filename)[:-4]
dat_filename = raw_filename[:-4] + '.dat'


imgs_dir = os.path.join(parent_path, basename)
if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)

f_info = open(dat_filename, 'r')
line = f_info.readline()
line = f_info.readline()
dims = [int(s) for s in line.split() if s.isdigit()]
x = dims[0]
y = dims[1]
z = dims[2]
digits = len(str(z))
num_format = '{:0'+str(digits)+'d}'

img_size = x*y
offset = 2*img_size

f_data = open(raw_filename, 'rb')

for i in range(0, z):
    f_data.seek(i*offset)
    slice = np.fromfile(f_data, dtype = 'uint16', count = img_size, sep = "")
    slice = slice.reshape([y, x])
    imgname = basename + '_' + num_format.format(i)
    if args.format == 'png':
        slice1 = slice*255.0/65535.0
        slice1 = np.floor(slice1)
        Image.fromarray(slice1.astype('uint8')).save(os.path.join(imgs_dir, imgname+'.png'))
    if args.format == 'tif':
        Image.fromarray(slice.astype('uint16')).save(os.path.join(imgs_dir, imgname+'.tif'))
'''
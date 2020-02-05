#!/usr/bin/env python3
import os, argparse
import shutil

def options():
    
    parser = argparse.ArgumentParser(description='Root Crowns Segmentation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--soil', action='store_true')
    parser.add_argument('-i', "--input_folder", help="directory of original image slices", required=True)
    parser.add_argument('-s', "--sampling", help="resolution parameter", default=2)	
    
    args = parser.parse_args()

    return args

args = options()
original_folder = args.input_folder

sampling = args.sampling

parent_path = os.path.dirname(original_folder)
folder_name = os.path.abspath(original_folder)
thresholded_folder = os.path.join(parent_path, folder_name+"_thresholded_images")
model_folder = os.path.join(parent_path, folder_name+"_3d_models")
if not os.path.exists(thresholded_folder):
    os.makedirs(thresholded_folder)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
    
list_dirs = os.walk(original_folder)
for root, dirs, files in list_dirs:
    for subfolder in dirs:
        input_path = "\"" + os.path.join(original_folder, subfolder) + "/\""
        output_path = os.path.join(thresholded_folder, subfolder) 
        model_file = "\"" + os.path.join(model_folder, subfolder+".out") + "\""
        model_file2 = "\"" + os.path.join(model_folder, subfolder+".obj") + "\""
        model_file3 = "\"" + os.path.join(model_folder, subfolder+"_soil.out") + "\""
        model_file4 = "\"" + os.path.join(model_folder, subfolder+"_soil.obj") + "\""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = "\"" + output_path + "/\""
        if args.soil:
            command = "./rootCrownSegmentation " + "1 " + input_path + " " + str(sampling) + " " + output_path + " " + model_file + " " + model_file2+ " " + model_file3 + " " + model_file4
        else:
            command = "./rootCrownSegmentation " + "0 " + input_path + " " + str(sampling) + " " + output_path + " " + model_file + " " + model_file2
        
        print(command)
        print("\n")
        os.system(command)

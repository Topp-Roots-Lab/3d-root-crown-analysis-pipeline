#!/usr/bin/env python2
import commands
import os
import shutil
import argparse

def options():
    
    parser = argparse.ArgumentParser(description='Root Crowns Feature Extraction',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', "--input_folder", help="directory of .out files", required=True)
    parser.add_argument('-s', "--scale", help="the scale parameter using for skeleton", default=2.25)


    args = parser.parse_args()

    return args

args = options()
input_folder = args.input_folder
scale = args.scale
output_name = os.path.join(input_folder, "features.tsv")
shutil.copy2("/media/topplab/EA6E84586E842005/scripts/rootCrownAnalysis3D/features.tsv", output_name)

for fname in [out for out in os.listdir(input_folder) if out.endswith(".out")]:
	#input_name = "\"" + input_folder + fname + "\"";
	input_name = "\"" + os.path.join(input_folder, fname) + "\""
	command = "/media/topplab/EA6E84586E842005/scripts/rootCrownAnalysis3D/Skeleton " + input_name + " " + output_name + " " + str(scale);
	print command+"\n"
	os.system(command)

#!/usr/bin/env python3
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
features = ['Name','SurfArea','Volume','Convex_Volume','Solidity','MedR','MaxR','Bushiness','Depth','HorEqDiameter','TotalLength','SRL','Length_Distr','W_D_ratio','Number_bif_cl','Av_size_bif_cl','Edge_num','Av_Edge_length','number_tips','volume','surface_area','av_radius']
with open(output_name, 'w') as ofp:
    ofp.write('\t'.join(features))

for fname in [out for out in os.listdir(input_folder) if out.endswith(".out")]:
    input_name = "\"" + os.path.join(input_folder, fname) + "\""
    command = "Skeleton " + input_name + " " + output_name + " " + str(scale)
    print(command+"\n")
    os.system(command)
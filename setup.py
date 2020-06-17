import os

import setuptools

with open ('VERSION', 'r') as ifp:
	__version__ = ifp.readline()
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='rcap',  
     version=__version__,
     scripts=['raw2img',
              'batch_segmentation',
              'batch_skeleton',
              'rootCrownImageAnalysis3D',
              'Skeleton',
              'rootCrownSegmentation'] ,
     author="Timothy Parker",
     author_email="tparker@danforthcenter.org",
     description="Root crown 3-D x-ray image analysis pipeline",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/Topp-Roots-Lab/3d-root-crown-analysis-pipeline",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )

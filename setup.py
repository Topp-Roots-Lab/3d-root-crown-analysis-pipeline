#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

requirements = [ 'rawtools', 'tqdm', 'pandas', 'numpy', 'opencv-python', 'scikit-learn', 'scikit-image', 'scipy', 'xvfbwrapper' ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Tim Parker",
    author_email='Tim.ParkerD@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    description="X-ray root crown image analysis pipeline",
    entry_points={
        'console_scripts': [
            'batch-segmentation=xrcap.cli:segment',
            'batch-skeleton=xrcap.cli:skeleton',
            'qc-point-clouds=xrcap.cli:qc_point_clouds',
            'qc-binary-images=xrcap.cli:qc_binary_images',
            'xrcap-collate-results=xrcap.cli:collate_output',
            'rootCrownImageAnalysis3D=xrcap.cli:image_analysis'
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='xrcap',
    name='xrcap',
    packages=find_packages(include=['xrcap', 'xrcap.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Topp-Roots-Lab/xrcap',
    version='1.6.0',
    zip_safe=False,
)

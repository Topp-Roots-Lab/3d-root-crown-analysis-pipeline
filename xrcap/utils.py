"""Utility functions for pipeline management"""
import logging
import subprocess
from shutil import which
import re
import os


def fetch_gia3d_version():
    """Fetch version of Gia3D"""
    try:
        gia3d_binary = which("Skeleton")
        process = subprocess.Popen(
            [gia3d_binary, "-V"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        stdout, stderr = process.communicate()
        logging.debug(f"gia3d binary: {(stdout, stderr)}")
    except Exception as e:
        raise e
    else:
        pattern = r"^.*(?P<version>\d+\.\d+\.\d+)$"
        m = re.match(pattern, stdout)
        version = "unknown"
        if m is not None and "version" in m.groupdict():
            version = m.group("version")
        return version


def fetch_segmentation_version():
    """Fetch version of segmentation"""
    try:
        expected_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "lib", "rootCrownSegmentation"
        )
        process = subprocess.Popen(
            [expected_path, "-V"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        stdout, stderr = process.communicate()
        logging.debug(f"rootCrownSegmentation binary: {(stdout, stderr)}")
    except Exception as e:
        raise e
    else:
        pattern = r"^.*(?P<version>\d+\.\d+\.\d+)$"
        m = re.match(pattern, stdout)
        version = "unknown"
        if m is not None and "version" in m.groupdict():
            version = m.group("version")
        return version

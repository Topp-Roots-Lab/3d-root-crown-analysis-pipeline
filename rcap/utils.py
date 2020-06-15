#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
import argparse
import logging
import os
from datetime import datetime as dt

from __init__ import __version__


def configure_logging(args):
	# Configure logging, stderr and file logs
	logging_level = logging.INFO
	if args.verbose:
		logging_level = logging.DEBUG

	logFormatter = logging.Formatter("%(asctime)s - [%(levelname)-4.8s] - %(filename)s %(lineno)d - %(message)s")
	rootLogger = logging.getLogger()
	rootLogger.setLevel(logging.DEBUG)

	# Set project-level logging
	logfile_basename = f"{dt.today().strftime('%Y-%m-%d_%H-%M-%S')}_{args.module_name}.log"
	lfp = os.path.join(os.path.realpath(args.path[0]), logfile_basename)
	fileHandler = logging.FileHandler(lfp)
	fileHandler.setFormatter(logFormatter)
	fileHandler.setLevel(logging.DEBUG) # always show debug statements in log file
	rootLogger.addHandler(fileHandler)

	slfp = os.path.join('/', 'var', 'log', '3drcap', args.module_name, logfile_basename)
	syslogFileHandler = logging.FileHandler(slfp)
	syslogFileHandler.setFormatter(logFormatter)
	syslogFileHandler.setLevel(logging.DEBUG) # always show debug statements in log file
	rootLogger.addHandler(syslogFileHandler)

	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(logFormatter)
	consoleHandler.setLevel(logging_level)
	rootLogger.addHandler(consoleHandler)

	logging.debug(f'Running {args.module_name} {__version__}')
	logging.debug(f"Command: {args}")

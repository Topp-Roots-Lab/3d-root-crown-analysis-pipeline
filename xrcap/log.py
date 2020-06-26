# -*- coding: utf-8 -*-
"""Logging module"""
import logging
import os
from datetime import datetime as dt
from importlib.metadata import version

__version__ = version('rawtools')


def configure(args):
	"""Set up log files and associated handlers"""
	# Configure logging, stderr and file logs
	logging_level = logging.INFO
	if args.verbose:
		logging_level = logging.DEBUG

	logFormatter = logging.Formatter("%(asctime)s - [%(levelname)-4.8s] - %(filename)s %(lineno)d - %(message)s")
	rootLogger = logging.getLogger()
	rootLogger.setLevel(logging.DEBUG)

	# Set project-level logging
	if args.module_name is not None:
		logfile_basename = f"{dt.today().strftime('%Y-%m-%d_%H-%M-%S')}_{args.module_name}.log"
	lfp = os.path.join(os.path.realpath(os.path.dirname(args.path[0])), logfile_basename) # base log file path
	fileHandler = logging.FileHandler(lfp)
	fileHandler.setFormatter(logFormatter)
	fileHandler.setLevel(logging.DEBUG) # always show debug statements in log file
	rootLogger.addHandler(fileHandler)

	sdfp = os.path.join('/', 'var', 'log', 'rawtools', args.module_name) # system directory file path
	if not os.path.exists(sdfp):
		os.makedirs(sdfp)
	slfp = os.path.join(sdfp, logfile_basename) # system log file path
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

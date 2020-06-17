import os
with open (os.path.join(os.path.dirname(os.path.dirname(__file__)),'VERSION'), 'r') as ifp:
	__version__ = ifp.readline()
import argparse
import logging
import os
from datetime import datetime as dt

__version__ = '0.1.0'

def parse_options():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('-f', '--force', action='store_true', help='Force file creation. Overwrite any existing files.')
    parser.add_argument('path', metavar='PATH', type=str, nargs=2, help='Input directory to process')
    args = parser.parse_args()

    # Configure logging, stderr and file logs
    logging_level = logging.INFO
    if args.verbose:
        logging_level = logging.DEBUG

    lfp = f'{dt.today().strftime("%Y-%m-%d")}_{os.path.splitext(os.path.basename(__file__))[0]}.log'

    logFormatter = logging.Formatter('%(asctime)s - [%(levelname)-4.8s] - %(filename)s %(lineno)d - %(message)s')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging_level)
    rootLogger.addHandler(consoleHandler)

    args.path = list(set(args.path)) # remove any duplicates

    logging.debug(f'Running {__file__} {__version__}')

    return args

def process(args):
    afp, bfp = args.path
    logging.debug(f"{afp=}")
    logging.debug(f"{bfp=}")
    a_name = os.path.splitext(os.path.basename(afp))[0]
    b_name = os.path.splitext(os.path.basename(bfp))[0]
    dirname = os.path.dirname(os.path.realpath(afp))
    ext = os.path.splitext(afp)[1]
    ofp = os.path.join(dirname, f"{a_name}__{b_name}{ext}")
    with open(afp, 'r') as A, open(bfp, 'r') as B, open(ofp, 'w') as O:
        a_line = A.readline()
        b_line = B.readline()
        while a_line and b_line:
            if a_line != b_line:
                O.write(a_line)
                O.write(b_line)
            a_line = A.readline()
            b_line = B.readline()

if __name__ == '__main__':
    args = parse_options()
    # Collect all 
    try:
        # Gather all files
        args.path = [ os.path.realpath(p) for p in args.path ]
        args.path = list(set(args.path)) # remove duplicates

    except Exception as err:
        logging.error(err)
        raise

    else:
        #Process
        logging.debug(args.path)
        process(args)

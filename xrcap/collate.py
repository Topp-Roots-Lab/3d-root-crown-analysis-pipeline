"""Data collation for output file for meshing and image analysis"""
import logging
import os
import pandas as pd

def main(path:list[str], *args, **kwargs):
    """Main for collating all features & traits measured
    """

    verbose = 'verbose' in kwargs
    # if type == 'traits':
    #     extension = '.csv'
    #     sep = ','
    #     delimiter = ','
    #     index = False
    # elif type == 'features':
    #     extension = '.tsv'
    #     sep = "\\t"
    #     delimiter = '\t'
    #     index = False
    # else:
    #     return # not supported type
    # # Find all '{type}.csv'
    data_files = []
    for fp in args.path:
        for root, dirs, files in os.walk(fp):
            for file in files:
                if f'{type}{extension}' in file:
                    data_files.append(os.path.join(root, file))
    logging.debug(f"Found {type} files: {data_files}")
    logging.info(f"Processing {len(data_files)} file(s).")
    if not data_files:
        logging.warning(f"No input files were found for {args.path}")
        return 1
    
    dfs = []
    for fp in data_files:
        df = pd.read_csv(fp, sep=sep, engine='python')
        df[f'original_{type}_filepath'] = fp
        dfs.append(df)
    
    collated_results = pd.concat(dfs)

    # Rearrange columsn so that input file is the second column
    cols = collated_results.columns.to_list()
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    collated_results = collated_results[cols]

    if args.ofp is not None:
        ofp = args.ofp
        if os.path.isdir(ofp) or not os.path.exists(ofp) or args.force:
            if os.path.exists(ofp) and args.force:
                logging.warning(f"Output file already exists '{ofp}'. Overwriting.")
            if os.path.isfile(ofp) and ofp.endswith(extension):
                ofp += extension
            if os.path.isdir(ofp):
                ofp = os.path.join(ofp, f'{type}_collated_results{extension}')

            # Partition out duplicate entries based on volume or file name
            # Create boolean mask of duplicates
            duplicates = collated_results[cols[0]].duplicated(keep=False)
            # if there are any duplicates found
            if duplicates.any():
                if os.path.isdir(ofp):
                    dup_ofp = os.path.join(ofp, f"{type}_collated_results_duplicates{extension}")
                else:
                    dup_ofp = f"{os.path.splitext(ofp)[0]}_duplicates{extension}"
                collated_results[duplicates].to_csv(dup_ofp, index=index, sep=delimiter)
                logging.warning(f"Duplicates detected. Check '{dup_ofp}' for list of all duplicate entries.")
            
            collated_results.drop_duplicates(inplace=True)
            collated_results.to_csv(ofp, index=index, sep=delimiter)
            logging.info(f"Results saved to '{ofp}'")
        elif os.path.exists(ofp):
            logging.info(f"Output file already exists '{ofp}'. Skipping. Use '--force' option to overwrite.")
    else:
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(collated_results.to_string())
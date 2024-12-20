import glob
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tfsenc_parser import parse_arguments
from tfsenc_phase_shuffle import phase_randomize_1d
from tfsenc_read_datum import read_datum
from tfsenc_utils import encoding_regression, load_header, setup_environ
from utils import main_timer, write_config


def process_sig_electrodes(args, datum):
    """Run encoding on select significant elctrodes specified by a file
    """
    # Read in the significant electrodes
    sig_elec_file = os.path.join(
        os.path.join(os.getcwd(), 'code', args.sig_elec_file))
    sig_elec_list = pd.read_csv(sig_elec_file)
    e_list = []
    # Loop over each electrode
    for subject, elec_name in sig_elec_list.itertuples(index=False):
        args.electrode_name = elec_name
        args.subject = subject
        if isinstance(subject, int):
            #breakpoint()
            CONV_DIR = os.path.join(os.getcwd(), 'data','electrode_data')
            #CONV_DIR = '/projects/HASSON/247/data/podcast'
            #BRAIN_DIR_STR = 'preprocessed_all'
            subject_id = glob.glob(
                os.path.join(CONV_DIR, 'NY' + str(subject) + '*'))[0]
            subject_id = os.path.basename(subject_id)

        # Read subject's header
        labels = load_header(CONV_DIR, subject_id)
        if not labels:
            print('Header Missing')
        electrode_num = labels.index(elec_name)
        # Read electrode data
        brain_dir = os.path.join(CONV_DIR, subject_id)
        electrode_file = os.path.join(
            brain_dir, ''.join([
                subject_id, '_electrode_preprocess_file_',
                str(electrode_num + 1), '.mat'
            ]))
        try:
            elec_signal = loadmat(electrode_file)['p1st']
            elec_signal = elec_signal.reshape(-1, 1)
        except FileNotFoundError:
            print(f'Missing: {electrode_file}')
            continue
        # Perform encoding/regression
        if not args.avg_electrodes:
            if args.phase_shuffle: 
                elec_signal = phase_randomize_1d(elec_signal)
            encoding_regression(args, datum, elec_signal,
                                str(subject) + '_' + elec_name)
        else:
            e_list.append(elec_signal)
    if args.avg_electrodes:
        elec_signal = np.mean(e_list, axis=0)
        encoding_regression(args, datum, elec_signal,args.sig_elec_file[:-4])

    return

@main_timer
def main():
    args = parse_arguments()
    # Setup paths to data
    args = setup_environ(args)
    # Saving configuration to output directory
    write_config(vars(args))
    # Locate and read datum
    datum = read_datum(args)

    if args.sig_elec_file:
        #breakpoint()
        process_sig_electrodes(args, datum)

    return


if __name__ == "__main__":
    main()

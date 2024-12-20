import csv
import os
from functools import partial
from multiprocessing import Pool

import mat73
import numpy as np
from numba import jit, prange
from scipy import stats
from sklearn.model_selection import KFold
from utils import load_pickle
import pandas as pd

def encColCorr(CA, CB, args):
    """[summary]

    Args:
        CA ([type]): [description]
        CB ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = np.shape(CA)[0] - 2
    #breakpoint() 
    # get info
    '''
    odir= '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/word_bootstrap_test/'
    en = str(args.subject) + '_' + args.electrode_name
    layer = args.output_parent_dir.split('hs')[-1]
    stuff = args.pkl_identifier.split('-')
    if len(stuff) == 5:
        wv = '-'.join(stuff[1:3])
    elif len(stuff) == 4:
        wv = stuff[1]
    ofn = 'word_bootstrap_layer-' + str(layer) + '_electrode-' + str(en) + '_' + wv + '.npy' 
    # compute word bootstrap
    samps = np.load('1000_samps_of_1697_words.npy') 
    bootstrap_corrs = []
    for s in samps:
        #breakpoint()
        tCA = CA[s, :]
        tCB = CB[s, :]
        tCA -= np.mean(tCA, axis=0)
        tCB -= np.mean(tCB, axis=0)

        tr = np.sum(tCA * tCB, 0) / np.sqrt(np.sum(tCA * tCA, 0) * np.sum(tCB * tCB, 0))
        bootstrap_corrs.append(tr)
    np.save(odir + ofn, bootstrap_corrs) 
    '''
    CA -= np.mean(CA, axis=0)
    CB -= np.mean(CB, axis=0)

    r = np.sum(CA * CB, 0) / np.sqrt(np.sum(CA * CA, 0) * np.sum(CB * CB, 0))

    t = r / np.sqrt((1 - np.square(r)) / df)
    p = stats.t.sf(t, df)

    r = r.squeeze()
    if r.size > 1:
        r = r.tolist()
    else:
        r = float(r)

    #np.save(odir + 'TRUE_' + ofn, r)
    return r, p, t


def cv_lm_003(args, X, Y, kfolds):
    print(args.layer_idx)
    """Cross-validated predictions from a regression model using sequential
        block partitions with nuisance regressors included in the training
        folds

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        kfolds ([type]): [description]

    Returns:
        [type]: [description]
    """
    skf = KFold(n_splits=kfolds, shuffle=False)

    # Data size
    nSamps = X.shape[0]
    try:
        nChans = Y.shape[1]
    except E:
        nChans = 1

    # Extract only test folds
    folds = [t[1] for t in skf.split(np.arange(nSamps))]

    YHAT = np.zeros((nSamps, nChans))
    # Go through each fold, and split
    for i in range(kfolds):
        # Shift the number of folds for this iteration
        # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
        #                       ^ dev fold
        #                         ^ test fold
        #                 | - | <- train folds
        folds_ixs = np.roll(range(kfolds), i)
        test_fold = folds_ixs[-1]
        train_folds = folds_ixs[:-1]
        # print(f'\nFold {i}. Training on {train_folds}, '
        #       f'test on {test_fold}.')

        test_index = folds[test_fold]
        # print(test_index)
        train_index = np.concatenate([folds[j] for j in train_folds])

        # Extract each set out of the big matricies
        Xtra, Xtes = X[train_index], X[test_index]
        Ytra, Ytes = Y[train_index], Y[test_index]

        # run new pca here
        # pca including train folds and mid
        #breakpoint()
        if args.pca_flag:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=args.reduce_to, svd_solver='auto', random_state = 42)
            # load mid
            #mid = pd.DataFrame(load_pickle(args.PICKLE_DIR + '/777_full-mid-PAPER2-hs{}_gpt2-xl_cnxt_1024_embeddings.pkl'.format(args.layer_idx)))
            #print(args.layer_idx)
            #from tfsenc_read_datum import drop_nan_embeddings, remove_punctuation
            #mid = drop_nan_embeddings(mid)
            #mid = remove_punctuation(mid)
            #mid = mid[~mid['glove50_embeddings'].isna()]
            #mid_emb = np.array(list(mid.embeddings))
            # concat with train set
            #cat = np.vstack([mid_emb, Xtra]) 
            # fit pca
            #pca.fit(cat.tolist())
            #pca.fit(mid_emb.tolist())
            #breakpoint()
            pca.fit(Xtra.tolist())
            # transform train
            Xtra = pca.transform(Xtra)
            # transform test
            Xtes = pca.transform(Xtes)
            #print(Xtra.shape, Xtes.shape)

        # Mean-center
        Xtra -= np.mean(Xtra, axis=0)
        Xtes -= np.mean(Xtes, axis=0)
        Ytra -= np.mean(Ytra, axis=0)
        Ytes -= np.mean(Ytes, axis=0)

        # Fit model
        B = np.linalg.pinv(Xtra) @ Ytra
        #breakpoint()
        # Predict
        foldYhat = Xtes @ B
        #breakpoint()
        # Add to data matrices
        YHAT[test_index, :] = foldYhat.reshape(-1, nChans)

        # per fold result
        '''
        fold_rp, fold_p, _ = encColCorr(Ytes, foldYhat, args) # per fold
        trial_str = append_jobid_to_string(args, 'comp')
        name = args.sig_elec_file[:-4]
        name = str(args.subject) + '_' + args.electrode_name
        filename = os.path.join(args.full_output_dir, name + trial_str + '_fold' + str(i) + '.csv')
        #breakpoint()
        with open(filename, 'w') as csvfile:
            #breakpoint()
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows([fold_rp])
            csvwriter.writerows([fold_p])
        '''
    #breakpoint()
    return YHAT


@jit(nopython=True)
def fit_model(X, y):
    """Calculate weight vector using normal form of regression.

    Returns:
        [type]: (X'X)^-1 * (X'y)
    """
    beta = np.linalg.solve(X.T.dot(X), X.T.dot(y))
    return beta


@jit(nopython=True)
def build_Y(onsets, brain_signal, lags, window_size):
    """[summary]

    Args:
        onsets ([type]): [description]
        brain_signal ([type]): [description]
        lags ([type]): [description]
        window_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    #breakpoint()
    half_window = round((window_size / 1000) * 512 / 2)
    t = len(brain_signal)

    Y1 = np.zeros((len(onsets), len(lags), 2 * half_window + 1))

    for lag in prange(len(lags)):
        lag_amount = int(lags[lag] / 1000 * 512)
        #breakpoint()
        index_onsets = np.minimum(
            t - half_window - 1,
            np.maximum(half_window + 1,
                       np.round_(onsets, 0, onsets) + lag_amount))

        # subtracting 1 from starts to account for 0-indexing
        starts = index_onsets - half_window - 1
        stops = index_onsets + half_window

        # vec = brain_signal[np.array([np.arange(*item) for item in zip(starts, stops)])]

        for i, (start, stop) in enumerate(zip(starts, stops)):
            #if i == 939: breakpoint()
            Y1[i, lag, :] = brain_signal[np.int(start):np.int(stop)].reshape(-1)
    return Y1


def build_XY(args, datum, brain_signal):
    """[summary]

    Args:
        args ([type]): [description]
        datum ([type]): [description]
        brain_signal ([type]): [description]

    Returns:
        [type]: [description]
    """
    X = np.stack(datum.embeddings).astype('float64')

    onsets = datum.onset.values

    lags = np.array(args.lags)
    brain_signal = brain_signal.reshape(-1, 1)
    #breakpoint()
    Y = build_Y(onsets, brain_signal, lags, args.window_size)

    return X, Y


def encode_lags_numba(args, X, Y):
    """[summary]
    Args:
        X ([type]): [description]
        Y ([type]): [description]
    Returns:
        [type]: [description]
    """
    if args.shuffle:
        np.random.shuffle(Y)

    Y = np.mean(Y, axis=-1)
    PY_hat = cv_lm_003(args, X, Y, 10)
    rp, p, _ = encColCorr(Y, PY_hat, args)
    return rp, p


def encoding_mp(_, args, prod_X, prod_Y):
    perm_rc, perm_p = encode_lags_numba(args, prod_X, prod_Y)
    return perm_rc, perm_p


def run_save_permutation_pr(args, prod_X, prod_Y, filename):
    """[summary]
    Args:
        args ([type]): [description]
        prod_X ([type]): [description]
        prod_Y ([type]): [description]
        filename ([type]): [description]
    """
    if prod_X.shape[0]:
        perm_rc = encode_lags_numba(args, prod_X, prod_Y)
    else:
        perm_rc = None

    return perm_rc


def run_save_permutation(args, prod_X, prod_Y, filename):
    """[summary]

    Args:
        args ([type]): [description]
        prod_X ([type]): [description]
        prod_Y ([type]): [description]
        filename ([type]): [description]
    """
    if prod_X.shape[0]:
        # with Pool(16) as pool:
        #     perm_prod = pool.map(
        #         partial(encoding_mp, args=args, prod_X=prod_X, prod_Y=prod_Y),
        #         range(args.npermutations))

        perm_prod = []
        p_val = []
        for i in range(args.npermutations):
            perm_r, perm_p = encoding_mp(i, args, prod_X, prod_Y)
            perm_prod.append(perm_r)
            p_val.append(perm_p) 
        # comment out if doing per fold
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(perm_prod)
            csvwriter.writerows(p_val)

def load_header(conversation_dir, subject_id):
    """[summary]

    Args:
        conversation_dir ([type]): [description]
        subject_id (string): Subject ID

    Returns:
        list: labels
    """
    #misc_dir = os.path.join(conversation_dir, subject_id, 'misc')
    #header_file = os.path.join(misc_dir, subject_id + '_header.mat')
    header_file = os.path.join(conversation_dir, subject_id,subject_id + '_header.mat')
    if not os.path.exists(header_file):
        return
    header = mat73.loadmat(header_file)
    labels = header.header.label

    return labels


def create_output_directory(args):
    # output_prefix_add = '-'.join(args.emb_file.split('_')[:-1])

    # folder_name = folder_name + '-pca_' + str(args.reduce_to) + 'd'
    # full_output_dir = os.path.join(args.output_dir, folder_name)

    folder_name = '-'.join([args.output_prefix, str(args.sid)])
    folder_name = folder_name.strip('-')
    full_output_dir = os.path.join(os.getcwd(), 'results',
                                   args.output_parent_dir, folder_name)

    os.makedirs(full_output_dir, exist_ok=True)

    return full_output_dir


def encoding_regression_pr(args, datum, elec_signal, name):
    """[summary]
    Args:
        args (Namespace): Command-line inputs and other configuration
        sid (str): Subject ID
        datum (DataFrame): ['word', 'onset', 'offset', 'speaker', 'accuracy']
        elec_signal (numpy.ndarray): of shape (num_samples, 1)
        name (str): electrode name
    """
    # Build design matrices
    X, Y = build_XY(args, datum, elec_signal)

    # Split into production and comprehension
    prod_X = X[datum.speaker == 'Speaker1', :]
    comp_X = X[datum.speaker != 'Speaker1', :]

    prod_Y = Y[datum.speaker == 'Speaker1', :]
    comp_Y = Y[datum.speaker != 'Speaker1', :]

    # Run permutation and save results
    prod_corr = run_save_permutation_pr(args, prod_X, prod_Y, None)
    comp_corr = run_save_permutation_pr(args, comp_X, comp_Y, None)

    return (prod_corr, comp_corr)


def encoding_regression(args, datum, elec_signal, name):

    output_dir = args.full_output_dir

    # Build design matrices
    #breakpoint()
    X, Y = build_XY(args, datum, elec_signal)

    # Split into production and comprehension
    prod_X = X[datum.speaker == 'Speaker1', :]
    comp_X = X[datum.speaker != 'Speaker1', :]

    prod_Y = Y[datum.speaker == 'Speaker1', :]
    comp_Y = Y[datum.speaker != 'Speaker1', :]

    print(f'{args.sid} {name} Prod: {len(prod_X)} Comp: {len(comp_X)}')

    # Run permutation and save results
    trial_str = append_jobid_to_string(args, 'prod')
    filename = os.path.join(output_dir, name + trial_str + '.csv')
    #breakpoint()
    run_save_permutation(args, prod_X, prod_Y, filename)

    trial_str = append_jobid_to_string(args, 'comp')
    filename = os.path.join(output_dir, name + trial_str + '.csv')
        #breakpoint()
    run_save_permutation(args, comp_X, comp_Y, filename)

    return


def setup_environ(args):
    """Update args with project specific directories and other flags
    """
    PICKLE_DIR = os.path.join(os.getcwd(), 'data', 'pickles')
    path_dict = dict(PICKLE_DIR=PICKLE_DIR)

    if args.emb_type == 'glove50':
        stra = 'cnxt_' + str(args.context_length)
        args.emb_file = '_'.join(
            [str(args.sid), args.emb_type, stra, 'embeddings.pkl'])

        # args.emb_type = args.align_with
        # args.context_length = args.align_target_context_length

        stra = 'cnxt_' + str(args.align_target_context_length)
        args.load_emb_file = '_'.join([
            str(args.sid), args.pkl_identifier, args.align_with, stra,
            'embeddings.pkl'
        ])
    elif args.emb_type == 'bert':
        args.emb_file = 'layer00.pkl' # dummy (not used)

    elif args.emb_type == 'llama':
        args.emb_file = ''
    else:
        stra = 'cnxt_' + str(args.context_length)
        # 661 _full_ gpt2-xl cnxt_1024 embeddings
        args.emb_file = '_'.join([
            str(args.sid), args.pkl_identifier, args.emb_type, stra,
            'embeddings.pkl'
        ])
        # below: all in pkl_identifier but SID and ebeddings.pkl
        #breakpoint()
        #args.emb_file = '_'.join([str(args.sid), args.pkl_identifier, 'embeddings.pkl'])

        args.load_emb_file = args.emb_file

    #args.signal_file = '_'.join([str(args.sid), 'full_signal.pkl'])
    #args.electrode_file = '_'.join([str(args.sid), 'electrode_names.pkl'])
    args.stitch_file = '_'.join([str(args.sid), 'full_stitch_index.pkl'])

    #args.output_dir = os.path.join(os.getcwd(), 'results')
    args.full_output_dir = create_output_directory(args)

    vars(args).update(path_dict)
    return args


def append_jobid_to_string(args, speech_str):
    """Adds job id to the output eletrode.csv file.

    Args:
        args (Namespace): Contains all commandline agruments
        speech_str (string): Production (prod)/Comprehension (comp)

    Returns:
        string: concatenated string
    """
    speech_str = '_' + speech_str

    if args.job_id:
        trial_str = '_'.join([speech_str, f'{args.job_id:02d}'])
    else:
        trial_str = speech_str

    return trial_str

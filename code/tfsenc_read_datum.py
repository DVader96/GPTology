import os
import string

import numpy as np
import pandas as pd
from utils import load_pickle

import pickle
from lcs import lcs

def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding
    """
    df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())
    df = df[~df['is_nan']]

    return df


def return_stitch_index(args):
    """[summary]

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    stitch_file = os.path.join(args.PICKLE_DIR, args.stitch_file)
    stitch_index = load_pickle(stitch_file)
    return stitch_index


def adjust_onset_offset(args, df):
    """[summary]

    Args:
        args ([type]): [description]
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    stitch_index = return_stitch_index(args)
    assert len(stitch_index) == df.conversation_id.nunique()

    stitch_index = [0] + stitch_index[:-1]

    df['adjusted_onset'], df['onset'] = df['onset'], np.nan
    df['adjusted_offset'], df['offset'] = df['offset'], np.nan

    for idx, conv in enumerate(df.conversation_id.unique()):
        shift = stitch_index[idx]
        df.loc[df.conversation_id == conv,
               'onset'] = df.loc[df.conversation_id == conv,
                                 'adjusted_onset'] - shift
        df.loc[df.conversation_id == conv,
               'offset'] = df.loc[df.conversation_id == conv,
                                  'adjusted_offset'] - shift

    return df


def add_signal_length(args, df):
    """[summary]

    Args:
        args ([type]): [description]
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    stitch_index = return_stitch_index(args)

    signal_lengths = np.diff(stitch_index).tolist()
    signal_lengths.insert(0, stitch_index[0])

    df['conv_signal_length'] = np.nan

    for idx, conv in enumerate(df.conversation_id.unique()):
        df.loc[df.conversation_id == conv,
               'conv_signal_length'] = signal_lengths[idx]

    return df


def read_datum(args):
    """Read and process the datum based on input arguments

    Args:
        args (namespace): commandline arguments

    Raises:
        Exception: args.word_value should be one of ['top', 'bottom', 'all']

    Returns:
        DataFrame: processed datum
    """
    if args.emb_type == 'bert':
        base_fn =os.path.join(args.PICKLE_DIR, 'base_df.pkl')
        with open(base_fn, 'rb') as fh:
            base_df = pickle.load(fh)

        layer_fn = os.path.join(args.PICKLE_DIR, 'layer_{}.pkl'.format(str(args.layer_idx).zfill(2)))
        with open(layer_fn, 'rb') as fh:
            layer_df = pd.DataFrame(pickle.load(fh))
        df = pd.concat([base_df, layer_df], axis=1)
        #breakpoint() 
        reg_df = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-paper-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl'))

        t, f =  lcs(list(reg_df.word), list(df.word))
        df = df.iloc[f].reset_index(drop=True)
        #breakpoint()
        #df[np.isnan(df.onset)] = reg_df[np.isnan(df.onset)]
        #df[np.isnan(df.offset)] = reg_df[np.isnan(df.offset)]
        df.onset = reg_df.onset
        df.offset = reg_df.offset
        #df = pd.concat([df.reset_index(), reg_df.glove50_embeddings], axis=1)
        #breakpoint()
        df = pd.concat([df.reset_index(drop=True), reg_df.glove50_embeddings], axis=1)
        #df.word = df.word_without_punctuation
        #df = df.drop(columns='word_without_punctuation')
        df.token = df.word_without_punctuation
        #breakpoint()
        # correct
        #tdf = load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-REVPAPER-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl')
        #tdf = pd.DataFrame(tdf)
        #breakpoint()
        rt = []
        wg = []
        #breakpoint()
        topk=1
        for i in range(len(reg_df)):
            if reg_df['token_id'][i] in reg_df['top10'][i][:topk]:
                rt.append(i)
            else:
                wg.append(i)
        #breakpoint()
        df = df.iloc[rt].reset_index(drop=True)
        #wg_df = df.iloc[wg].reset_index(drop=True)

    elif args.emb_type == 'llama':
        from preprocess_llama_embeddings import concat_llama_embs
        df = concat_llama_embs(args.layer_idx)

        reg_df = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-paper-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl'))

        t, f =  lcs(list(reg_df.word), list(df.word))
        df = df.iloc[f].reset_index(drop=True)
        reg_df = reg_df.iloc[t].reset_index(drop=True)

        #breakpoint()
        cols = list(reg_df.columns)
        cols.remove('word')
        cols.remove('embeddings')
        #breakpoint()
        df = pd.concat([df, reg_df[cols]], axis=1)
        
        # correct
        rt = []
        wg = []
        topk=1
        for i in range(len(df)):
            if df['token_id'][i] in df['top10'][i][:topk]:
                rt.append(i)
            else:
                wg.append(i)
        df = df.iloc[rt].reset_index(drop=True)
        #breakpoint()
        #wg_df = df.iloc[wg].reset_index(drop=True)
    elif args.emb_type == 'symbolic':
        #phon_df = pd.read_csv('phonological_embeddings.csv')
        #phon_df = pd.read_pickle('phonological_embeddings.pkl') # sum
        #phon_df = pd.read_pickle('zscore_phonological_embeddings.pkl') # zscore sum
        #phon_df = pd.read_pickle('phonological_embeddings_pca_basic_padding.pkl') 
        ##phon_df = pd.read_pickle('phonological_embeddings_pca_complex_padding.pkl') # included stress
        #breakpoint()

        #phon_df = pd.read_pickle('no_stress_phonological_embeddings_pca_complex_padding.pkl') # phonology
        #phon_df = pd.read_pickle('semantic_embeddings.pkl') # wordnet (bad)
        #phon_df = pd.read_pickle('spacy_semantic_embeddings.pkl')# correct semantic
        #phon_df = pd.read_pickle('spacy_syntax2_embeddings.pkl') # bad
        #phon_df = pd.read_pickle('spacy_syntax_current_word_embeddings.pkl') # syntax just current word tag, pos
        #phon_df = pd.read_pickle('spacy_syntax_all_features_word_embeddings.pkl') # all tag, pos
        #phon_df = pd.read_pickle('morphological_embeddings_simple_padding_pca10.pkl')
        #phon_df = pd.read_pickle('')

        # pca 50 test
        #breakpoint()
        #phon_df = pd.read_pickle('no_stress_phonological_embeddings_pca_none_complex_padding.pkl') # phonology pca 50 test
        #phon_df = pd.read_pickle('no_stress_phonological_embeddings_pca_none_complex_padding_train_pca.pkl') # same as above
        #phon_df = pd.read_pickle('no_stress_phonological_embeddings_pca_none_basic_padding_train_pca.pkl') # same as above
        #phon_df = pd.read_pickle('morphological_embeddings_simple_padding_pca_none.pkl')
        #phon_df = pd.read_pickle('spacy_syntax_all_features_word_embeddings_pca_none.pkl')
        #phon_df = pd.read_pickle('spacy_semantic_embeddings_pca_none.pkl')
        # fixed no stress phonology
        phon_df = pd.read_pickle('no_stress_phonological_embeddings_pca_none_basic_padding_fixed.pkl')
        #phon_df = pd.read_pickle('no_stress_phonological_embeddings_pca_none_complex_padding_fixed.pkl')


        reg_df = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-paper-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl'))
        t, f =  lcs(list(reg_df.word), list(phon_df.word))
        df = phon_df.iloc[f].reset_index(drop=True)
        reg_df = reg_df.iloc[t].reset_index(drop=True)
        cols = list(reg_df.columns)
        cols.remove('word')
        cols.remove('embeddings')

        df = pd.concat([df, reg_df[cols]], axis=1)
        rt = []
        wg = []
        topk=1
        for i in range(len(df)):
            if df['token_id'][i] in df['top10'][i][:topk]:
                rt.append(i)
            else:
                wg.append(i)
        df = df.iloc[rt].reset_index(drop=True)
    else:
        # this is normal operation
        file_name = os.path.join(args.PICKLE_DIR, args.load_emb_file)
        datum = load_pickle(file_name)
        df = pd.DataFrame.from_dict(datum)
        # alignment with our "all" datum 
        #file_name = os.path.join(args.PICKLE_DIR, args.load_emb_file)
        #datum = load_pickle(file_name)
        #df = pd.DataFrame.from_dict(datum)

        #breakpoint()
        #reg_df = pd.DataFrame(load_pickle(os.path.join(args.PICKLE_DIR, '777_full-paper-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl')))
        #t, f =  lcs(list(reg_df.word), list(df.word))
        #df = df.iloc[f].reset_index(drop=True)
        #reg_df = reg_df.iloc[t].reset_index(drop=True)
        #cols = list(reg_df.columns)
        ##cols.remove('word')
        #cols.remove('embeddings')
        # merge so that get embeddings from your df
        #df = pd.concat([df['embeddings'], reg_df[cols]],axis=1)
        #df = pd.concat([df, reg_df[cols]], axis=1)

    df = add_signal_length(args, df)

    if args.project_id == 'tfs' and not all(
        [item in df.columns
         for item in ['adjusted_onset', 'adjusted_offset']]):
        df = adjust_onset_offset(args, df)
    
    if args.part_of_model == 'attention':
        df['embeddings'] = df['attn_embeddings']
    elif args.part_of_model == 'feedforward':
        df['embeddings'] = df['ffw_embeddings']
    df = drop_nan_embeddings(df)
    df = remove_punctuation(df)

    if args.conversation_id:
        df = df[df.conversation_id == args.conversation_id]

    # use columns where token is root
    if args.project_id == 'tfs':
        if 'gpt2-xl' in [args.align_with, args.emb_type]:
            df = df[df['gpt2-xl_token_is_root']]
        elif 'bert' in [args.align_with, args.emb_type]:
            df = df[df['bert_token_is_root']]
        else:
            pass
    
    # Filter out words with nan glove embeddings
    df = df[~df['glove50_embeddings'].isna()]

    # option to use glove embeddings in downstream analyses
    if args.emb_type == 'glove50':
        df['embeddings'] = df['glove50_embeddings']
   
    return df



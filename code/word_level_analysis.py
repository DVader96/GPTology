#--------------------------------------------------#
# word level analysis functionality                #
#--------------------------------------------------#
from utils import load_pickle, save_pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from brain_color_prepro import get_n_layers
from scipy.stats import kendalltau as kt
import torch
from copy import deepcopy

# metrics
# for use in the analyses
def l1_dist(a,b):
    return np.sum(np.abs(a - b), axis = -1)

def l2_dist(a,b):
    return np.sum((a-b)**2, axis = -1)

def cos_dist(a,b):
    #breakpoint()
    from scipy import spatial
    dist = []
    for i in range(a.shape[0]):
        dist.append(spatial.distance.cosine(a[i,:],b[i,:]))
    #breakpoint()
    return dist

# rank analyses
# can use these to divide pickles up for encoding

# takes words, ranks by how much they change as you move from the beginning to the end
# metric 1 --> average distance across x layers. distance function: 
def word_change_rank(num_words, num_layers, jump_size, dist):
    #return 0
    # delta_v shape: (num_words,)
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    delta_v = np.zeros(num_words)
    layers = np.arange(1, num_layers+1, jump_size)
    print(layers)
    layer_avg = []
    for i,layer in enumerate(layers):
        pkl_name = pdir + '777_full-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
        df = pd.DataFrame(load_pickle(pkl_name))
        # if first, just save the current embeddings
        if i == 0:
            prev_emb = np.array(list(map(list, df.embeddings)))
        else:
            emb = np.array(list(map(list, df.embeddings)))
            diff =  dist(emb,prev_emb)
            delta_v +=diff
            layer_avg.append(np.mean(diff))
            prev_emb = emb 
    #breakpoint()
    delta_v /= len(layers)
    #y = np.sort(delta_v)
    #z = [y[i+1] - y[i] for i in range(0, len(y)-1)]
    # output df index order based on the changes. smallest change --> largest
    return np.argsort(delta_v), np.sort(delta_v), layer_avg


# divide words based on the amount of context needed to predict them
def amt_context_rank():
    return 0

# post processing of df
def split_dfs(num_words, num_layers,word_rank, split, jump_size):
    df1s = math.ceil(num_words*split)
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    breakpoint()
    for layer in range(1,num_layers+1):
        pkl_name = pdir + '777_full-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
        df = pd.DataFrame(load_pickle(pkl_name))
        df1 = df.loc[word_rank,:][:df1s].reset_index(drop=True)
        df2 = df.loc[word_rank,:][-df1s:].reset_index(drop=True)
        save_pickle(df1, pdir + '777_full-split1-jump' + str(jump_size) + '-l1-50-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')
        save_pickle(df2,  pdir + '777_full-split2-jump' + str(jump_size) + '-l1-50-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')

# assumes df has top10 column
# applies to one df only
def ci_split_uptodate_func(topk, tdf):
    rt = []
    wg = []
    for i in range(len(tdf)):
        if tdf['token_id'][i] in tdf['top10'][i][:topk]:
            rt.append(i)
        else:
            wg.append(i)

    print('frac correct', len(rt)/len(tdf))
    rt_df = tdf.iloc[rt].reset_index(drop=True)
    wg_df = tdf.iloc[wg].reset_index(drop=True)
    print('right', len(rt_df))
    print('wrong', len(wg_df))
    return rt_df, wg_df

def test_ci_split(topk, model_n):
    breakpoint()
    # revpaper
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    #fn1 = '777_full-paper-gpt2-xl_contextual2-interpolation_hs'
    #ID = 'REVPAPER-contextual2-interpolation' 
    #fn1 = '777_full-paper-gpt2-xl_1k-contextual2-interpolation_hs'
    #ID = '1k-contextual2-interpolation'
    #fn1 = '777_full-paper-gpt2-xl_1k-contextual3-interpolation_hs'
    #ID = '1k-contextual3-interpolation'
    
    #fn1 = '777_full-REV2-no-resid-gpt2-xlhs' # prefix for input
    #ID = 'REV2-no-resid' # for saving
    # NOTE: test.
    #fdir = '/scratch/gpfs/eham/podcast-pickling/results/763/'
    #fn_s =  '_gpt2-xl_cnxt_1024_embeddings.pkl' # suffix for input
    
    # for shifts (ex. T-3 --> T is shift--2)
    #fn1 = '777_full-REVPAPER-SHIFT-1-gpt2-xlhs'
    #ID = 'REV2-shift-1'
    #fn_s = '_gpt2-xl_cnxt_1024_embeddings.pkl' # suffix for input
  
    # bert
    #fn1 = '777_'
    #ID = ''
    #fn_s = 

    # diff context lengths
    #fn1 = '777_full-REVPAPER-gpt2-xlhs'
    #ID = 'REV2-CNXT-100'
    #fn_s = '_gpt2-xl_cnxt_100_embeddings.pkl' # suffix for input

    # test_hidden_states-git-gpt2-xl
    #pdir = fdir + 'test_hidden_states-git-gpt2-xl/'
    #fn1 = '777_full-gpt2-xlhs'
    #ID = 'GIT_SPACE'
    #pdir1 = fdir + 'test_hidden_states-git-model-accuracy-gpt2-xl/'
    #tdf1 = load_pickle(pdir1 + fn1 + '1' + fn_s)
    # these give same top10 values.
    #pdir2 = fdir + 'test_hidden_states-git-model-accuracygpt2-xl/'
    #tdf2 = load_pickle(pdir2 + fn1 + '48' + fn_s)
    #breakpoint()
    
    # test_hidden_states-update-glove-gpt2-xl
    #pdir = fdir + 'test_hidden_states-update-glove-gpt2-xl/'
    #fn1 = '777_full-update-glove-gpt2-xlhs'
    #ID = 'GLOVE'
    # test_hidden_states-aug2021-gpt2-xl
    #pdir = fdir + 'test_hidden_states-aug2021-gpt2-xl/'
    #pdir1 = fdir + 'test_hidden_states-git-model-accuracygpt2-xl/'
    #pdir1 = fdir + 'test_hidden_states-git-model-accuracy-gpt2-xl/'
    #fn1 = '777_full-gpt2-xlhs'
    #ID = 'AUG_NOSPACE'
    
    # THIS IS WHAT YOU USED IN PAPER
    # paper_hidden_states-gpt2-xl
    #pdir = fdir + 'paper_hidden_states-gpt2-xl/'
    #fn1 = '777_full-paper-gpt2-xlhs'
    #ID = 'PAPER2'
    
    print(ID)
    #tdf = load_pickle(pdir1 + fn1 + '1' + fn_s) # needed for diff folder for accuracy
    print(pdir + fn1 + '1' + fn_s)
    #tdf = load_pickle(pdir + fn1 + '1' + fn_s)
    #tdf = load_pickle('/scratch/gpfs/eham/podcast-pickling/results/763/paper_hidden_states-gpt2-xl/777_full-paper-gpt2-xlhs1'+fn_s)
    tdf = load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-REVPAPER-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl')
    tdf = pd.DataFrame(tdf)
    #breakpoint()
    rt = []
    wg = []
    #breakpoint()
    for i in range(len(tdf)):
        if tdf['token_id'][i] in tdf['top10'][i][:topk]:
            rt.append(i)
        else:
            wg.append(i)

    breakpoint()
    print('frac correct', len(rt)/len(tdf))
    num_layers = get_n_layers(model_n) 
    #print(model_n, ' num layers: ', num_layers)
    #num_layers = 1035
    for layer in range(1,num_layers+1):
        pkl_name = pdir + fn1 + str(layer) + fn_s
        print(pkl_name)
        df = pd.DataFrame(load_pickle(pkl_name))
        rt_df = df.iloc[rt].reset_index(drop=True)
        wg_df = df.iloc[wg].reset_index(drop=True)
        print('right', len(rt_df))
        print('wrong', len(wg_df))
        odir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/' 
        #breakpoint()
        save_pickle(rt_df, odir + '777_full-top' + str(topk) + '-correct-' + ID + '-hs' + str(layer) + fn_s)
        #save_pickle(wg_df, odir + '777_full-top' + str(topk) + '-incorrect-' + ID + '-hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')
    
# saves set of middle words (not in top1 correct or top 5 incorrect (for pca)
def get_middle_split(model_n):
    fdir = '/scratch/gpfs/eham/podcast-pickling/results/763/'
    #pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    # THIS IS WHAT YOU USED IN PAPER
    # paper_hidden_states-gpt2-xl
    pdir = fdir + 'paper_hidden_states-gpt2-xl/'
    fn1 = '777_full-paper-gpt2-xlhs'
    ID = 'PAPER2'
    fn_s = '_gpt2-xl_cnxt_1024_embeddings.pkl' # suffix for input

    print(ID)
    #tdf = load_pickle(pdir1 + fn1 + '1' + fn_s) # needed for diff folder for accuracy
    print(pdir + fn1 + '1' + fn_s)
    #tdf = load_pickle(pdir + fn1 + '1' + fn_s)
    tdf = load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-REVPAPER-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl')
    tdf = pd.DataFrame(tdf)
    #breakpoint()
    rt = []
    wg = []
    mid = []
    for i in range(len(tdf)):
        if tdf['token_id'][i] in tdf['top10'][i][:1]:
            rt.append(i)
        elif tdf['token_id'][i] not in tdf['top10'][i][:5]:
            wg.append(i)
        else:
            mid.append(i)
    #breakpoint()
    print('frac correct', len(rt)/len(tdf))
    num_layers = get_n_layers(model_n) 
    #print(model_n, ' num layers: ', num_layers)
    #num_layers = 1035
    for layer in range(1,num_layers+1):
        pkl_name = pdir + fn1 + str(layer) + fn_s
        print(pkl_name)
        df = pd.DataFrame(load_pickle(pkl_name))
        mid_df = df.iloc[mid].reset_index(drop=True)
        print('mid', len(mid_df))
        odir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/' 
        save_pickle(mid_df, odir + '777_full-mid-' + ID + '-hs' + str(layer) + fn_s)
       


# topk can be up to 10 in current set up (top1, top5, top10). 
# means, was the word in the topk predictions?
def correct_incorrect_split(topk, model_n, pkl):
    # use my pickles
    if pkl == 'mine':
        pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
        tdf = load_pickle(pdir + '777_full-paper-' + model_n + 'hs1_gpt2-xl_cnxt_1024_embeddings.pkl') # only compute correct/incorrect for layer 1 (whatever init is) 

        # use zaid's pickles
    elif pkl == 'zaid':
        pdir = '/scratch/gpfs/eham/247-encoding-brain/data/podcast/777/pickles/'
        tdf = load_pickle(pdir + '777_full_gpt2-xl_cnxt_1024_layer_01_embeddings.pkl') 

    # load df with correct/incorrect
    #tdf = load_pickle('/scratch/gpfs/eham/podcast-pickling/results/763/test_hidden_states-git-model-accuracy-gpt2-xl/777_full-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl') # gpt2-xl
    #pdir = '/scratch/gpfs/eham/podcast-pickling/results/763/paper_hidden_states-' + model_n + '/'# new (all files have accuracy)
       # glove   777_full-update-glove-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl
    #tdf = load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-update-glove-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl')
    tdf = pd.DataFrame(tdf)
    #len(np.where(z.word_without_punctuation == z.top1_pred.str.strip())[0])
    #breakpoint()
    rt = []
    wg = []
    breakpoint()
    if pkl == 'mine':
        for i in range(len(tdf)):
            if tdf['token_id'][i] in tdf['top10'][i][:topk]:
                rt.append(i)
            else:
                wg.append(i)
    elif pkl == 'zaid':
        for i in range(len(tdf)):
            if tdf.word_without_punctuation[i] == tdf.top1_pred[i].strip(): 
                rt.append(i)
            else:
                wg.append(i)

    print('frac correct', len(rt)/len(tdf))
    #breakpoint() 
    # determine correct/incorrect words. 
    # partition based on this.
    #pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/' # gpt2-xl
    #odir = pdir # gpt2-xl
    #odir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'

    #breakpoint()
    # glove
    num_layers = get_n_layers(model_n) 
    print(model_n, ' num layers: ', num_layers)
    for layer in range(1,num_layers+1):
        #pkl_name = pdir + '777_full-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl' # gpt2-xl
        
        # use my pickle
        if pkl == 'mine':
            pkl_name = pdir + '777_full-paper-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl' # gpt2-xl
      
        elif pkl == 'zaid':
            # use zaid's pickle
            if layer < 10:
                adj_layer = '0' + str(layer)
            else: 
                adj_layer = str(layer)

            pkl_name = pdir + '777_full_gpt2-xl_cnxt_1024_layer_' + adj_layer + '_embeddings.pkl' # gpt2-xl
        
        # glove
        #pkl_name = pdir + '777_full-update-glove-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
        df = pd.DataFrame(load_pickle(pkl_name))
        #df1 = df.loc[word_rankI,:][:df1s].reset_index(drop=True)
        #df2 = df.loc[word_rank,:][-df1s:].reset_index(drop=True)
        rt_df = df.iloc[rt].reset_index(drop=True)
        wg_df = df.iloc[wg].reset_index(drop=True)
        
        # use my pickle
        if pkl == 'mine':
            save_pickle(rt_df, pdir + '777_full-correct-top-' + str(topk) + '-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')
            save_pickle(wg_df,  pdir + '777_full-incorrect-top-' + str(topk) + '-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')
        elif pkl == 'zaid':
            # use zaid's pickle
            save_pickle(rt_df, pdir + '777_full-correct-top-' + str(topk) + '_gpt2-xl_cnxt_1024_layer_' + adj_layer + '_embeddings.pkl')
            save_pickle(wg_df,  pdir + '777_full-incorrect-top-' + str(topk) + '_gpt2-xl_cnxt_1024_layer_' + adj_layer + '_embeddings.pkl')


        # glove
        #save_pickle(rt_df, pdir + '777_full-update-glove-correct-top-' + str(topk) + '-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')
        #save_pickle(wg_df,  pdir + '777_full-update-glove-incorrect-top-' + str(topk) + '-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')

def shift_glove():
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    #breakpoint()
    layer = 1 # glove embeddings same for all layers
    #pkl_name = pdir + '777_full-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
    #pkl_name = pdir + '777_full-update-glove-gpt2-xlhs'  + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
    #pkl_name = pdir + '777_full-update-glove-correct-top-1-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
    pkl_name = pdir + '777_full-update-glove-incorrect-top-1-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
    df = pd.DataFrame(load_pickle(pkl_name))
    #breakpoint()
    wds = df['glove50_embeddings']
    none_count = 0
    for i, val in enumerate(wds):
        try: 
            print(len(val))
        except TypeError:
            print('none', df.word[i])
            none_count +=1
    #breakpoint()
    
    ndf = deepcopy(df)
    # align next embedding with word (last is a throw away)
    twds = list(wds[1:])
    twds.append(wds[0])
    ndf['glove50_embeddings'] = twds
    
    pdf = deepcopy(df)
    # align previous embedding with word (first is a throw away)
    twds = []
    twds.append(wds[len(wds)-1])
    twds.extend(wds[:len(wds)-1])
    pdf['glove50_embeddings'] = twds
    #df1 = df.loc[word_rankI,:][:df1s].reset_index(drop=True)
    #df2 = df.loc[word_rank,:][-df1s:].reset_index(drop=True)
    #rt_df = df.iloc[rt].reset_index(drop=True)
    #wg_df = df.iloc[wg].reset_index(drop=True)
    save_pickle(pdf, pdir + '777_full-prev-glove-incorrect-top-1-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')
    save_pickle(ndf,  pdir + '777_full-next-glove-incorrect-top-1-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')

# takes standard word aligned pickle as input and shifts embeddings relative to words
# result is nw --> next word, pw --> previous word, p2w --> 2 words back (word before previous)
def shift_pkls(model_n):
    #pdir = '/scratch/gpfs/eham/podcast-pickling/results/763/paper_hidden_states-' + model_n + '/'# new (all files have accuracy)
    #pdir = '/scratch/gpfs/eham/podcast-pickling/results/763/paper_hidden_states-' + model_n + '/'# new (all files have accuracy)
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    num_layers = get_n_layers(model_n)
    identifier = 'paper'
    #identifier = 'correct-top1'
    #identifier = 'incorrect-top1'
    print(model_n, ' num layers: ', num_layers)
    for layer in range(1,num_layers+1):
        pkl_name = pdir + '777_full-' + identifier + '-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl' # gpt2-xl
        df = pd.DataFrame(load_pickle(pkl_name))
        wds = df['embeddings']
        none_count = 0
        for i, val in enumerate(wds):
            try: 
                print(len(val))
            except TypeError:
                print('none', df.word[i])
                none_count +=1
        odf = deepcopy(df)
        # align next embedding with current word
        twds = list(wds[1:])
        twds.append(wds[0])
        odf['embeddings'] = twds 
        save_pickle(odf, pdir + '777_full-nw-' + identifier + '-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')
        # align previous embedding with current word
        twds = []
        twds.append(wds[len(wds)-1])
        twds.extend(wds[:len(wds)-1])
        odf['embeddings'] = twds
        save_pickle(odf, pdir + '777_full-pw-' + identifier + '-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')
        #breakpoint()
        # align embedding before previous with current word
        twds = []
        twds.extend(wds[len(wds)-2:]) # extend because adding 2 items not 1 as above
        twds.extend(wds[:len(wds)-2])
        odf['embeddings'] = twds
        save_pickle(odf, pdir + '777_full-p2w-' + identifier + '-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')

# generate a single embedding-shifted output. if amt < 0 --> shift backward (previous word). > 0 --> next word
def shift_pkl_amt(model_n, amt):
    if amt > 0:
        out_id = 'n' + str(amt)
    elif amt < 0:
        out_id = 'p' + str(-amt)
    else: 
        print('use input')
        return

    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    num_layers = get_n_layers(model_n)
    #identifier = 'paper'
    #identifier = 'correct-top-1'
    identifier = 'incorrect-top-1'
    print(model_n, ' num layers: ', num_layers)
    for layer in range(1,num_layers+1):
        pkl_name = pdir + '777_full-' + identifier + '-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl' # gpt2-xl
        df = pd.DataFrame(load_pickle(pkl_name))
        wds = df['embeddings']
        none_count = 0
        '''
        for i, val in enumerate(wds):
            try: 
                print(len(val))
            except TypeError:
                print('none', df.word[i])
                none_count +=1
        '''
        odf = deepcopy(df)
     
        twds = []
        twds.extend(wds[amt:]) # extend because adding 2 items not 1 as above
        twds.extend(wds[:amt])
        odf['embeddings'] = twds
        print(pdir + '777_full-' + out_id + 'w-' + identifier + '-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')
        #breakpoint()
        save_pickle(odf, pdir + '777_full-' + out_id + 'w-' + identifier + '-' + model_n + 'hs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl')


# use the saved lm head and dictionary to determine how well each layer predicts different words around the current one
def layer_decoding_test(topk, num_layers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #breakpoint()
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    # load the df
    num_correct = np.zeros(num_layers) 
    lm_head = torch.tensor(np.load('gpt2-xl-lm-head.npy'), dtype=torch.float32).to(device)
    for layer in range(1,num_layers+1):
        print(layer)
        pkl_name = pdir + '777_full-gpt2-xlhs' + str(layer) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
        df = pd.DataFrame(load_pickle(pkl_name))
        # for each word, see what matches, current, next or prev.  
        num_words = len(df)
        print(layer)
        # prev word: 1 to num_words, df['token_id'][i-1]
        # current word: 0 to num_words, df['token_id'][i]
        # next word: 0 to num_words - 1, df['token_id'][i+1]
        for i in range(2, num_words-0):
            #last_hs = model(ex, labels=label).hidden_states[-1][-2] # shape is (1,words, emb) 
            #word_dict = load_pickle('vocab.pkl')
            #breakpoint()
            
            last_hs = torch.tensor(np.array(df['embeddings'][i]), dtype=torch.float32).to(device)
            last_hs_pred = torch.matmul(lm_head, last_hs)
            #del last_hs
            # sorted order is largest --> smallest (top 1 to top 10)
            topx = torch.topk(last_hs_pred, topk, sorted=True)[1].detach().cpu().numpy()
            if df['token_id'][i-2] in set(topx):
                num_correct[layer-1] +=1
    num_correct /= num_words
    breakpoint()
    #del last_hs_pred
    #topk.append(top10)
    #del top10
 

def plot_splits(y,y_id):
    #z = [y[i+1] - y[i] for i in range(0, len(y)-1)]
    plt.figure()
    #y = (y - np.mean(y))/np.std(y)
    #z = (z-np.mean(z))/np.std(z)
    plt.plot(np.arange(len(y)), y, label='diff')
    #plt.plot(np.arange(len(z)), z, label='diff deriv')
    plt.legend()
    plt.savefig('word_level_test_' + str(y_id) + '.png')

def compare_ranks(num_words, dist, num_layers):
    #reg, regv = word_change_rank(num_words, num_layers, jump, l1_dist)
    ranks = []
    for i in range(1, num_layers):
        r, rv = word_change_rank(num_words, num_layers, i, dist)
        ranks.append(r)
    ks = []
    for j in range(len(ranks) - 1):
        ks.append(kt(ranks[j], ranks[j+1]))
    breakpoint()
    print('x')

if __name__ == '__main__':
    print('main')
    # only need to use one pickle (doesn't matter which layer)
    #breakpoint()
    #pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    #pkl_name = pdir + '777_full-gpt2-xlhs9_gpt2-xl_cnxt_1024_embeddings.pkl'
    #df = pd.DataFrame(load_pickle(pkl_name))
    #num_words = len(list(df.word))
    #model_name = 'gpt2-xl'
    #num_layers = get_n_layers(model_name)
    #compare_ranks(num_words, l1_dist, num_layers)
    #jump_size = 5
    #breakpoint()
    #rank5, rv5 = word_change_rank(num_words,num_layers, jump_size, l1_dist)
    #plot_splits(rv5, jump_size)
    #jump_size = 1
    #rank1, rv1 = word_change_rank(num_words,num_layers, jump_size, l1_dist)
    #plot_splits(rv1, jump_size)
    #jump_size = 10
    #rank10, rv10 = word_change_rank(num_words,num_layers, jump_size, l1_dist)
    #breakpoint()
    #plot_splits(rv10, jump_size)
    #jump_size = 47
    #rank48, rv48 = word_change_rank(num_words,num_layers, jump_size, l1_dist)
    #plot_splits(rv48, jump_size)
    #jump_size = 47
    #jump_size = 1
    #rank1, rankvals1, layer_avg = word_change_rank(num_words, num_layers, jump_size, l1_dist)
    #plot_splits(rankvals1, 'vals')
    #plot_splits(layer_avg, 'word_avg_per_layer')
    #split_dfs(num_words, num_layers,rank1, 0.5, jump_size)
    #layer_decoding_test(10, num_layers)
    model_n = 'gpt2-xl'
    #correct_incorrect_split(1, model_n, 'zaid')
    #test_ci_split(1, model_n, 'mine')
    #test_ci_split(5, model_n)  # paper correct incorrect split
    #test_ci_split(5, model_n)  # revision correct incorrect split
    get_middle_split(model_n)

    #shift_pkls(model_n)
    #shift_pkl_amt(model_n, 1) # test 1 # nw
    #shift_pkl_amt(model_n, 2) # what we want to run #n2w
    #shift_pkl_amt(model_n, -2) # test 2
    #shift_glove()



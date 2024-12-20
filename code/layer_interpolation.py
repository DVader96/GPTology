#----------------------------------------------------------------#
# Takes an input layer embedding and an output layer embedding   #
# constructs set number of intermediate layers via linear        #
# interpolation. these can then be used as a baseline            #
#----------------------------------------------------------------#
from utils import load_pickle, save_pickle
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from copy import deepcopy
import os
from sklearn.linear_model import LinearRegression

# interpolate between static embedding of previous word and current word. 
def get_static_interpolation(model):
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    #df = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-REVPAPER-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl'))[1:]
    df = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-REVPAPER-gpt2-xlhs1_gpt2-xl_cnxt_1024_embeddings.pkl'))
    assert(model != 'glove')
    if model == 'glove':
        print('glove embeddings')
        col = 'glove50_embeddings'
    elif model == 'gpt2':
        print('gpt2 embeddings')
        col = 'gpt2_static_embeddings'
    sl = np.array(list(df[col][:-1]))
    el = np.array(list(df[col][1:]))
    #breakpoint()
    linfit = interp1d([start, end], [sl,el], axis = 0) 
    layers = [linfit(i) for i in range(start, end+1)]
    for i,layer in enumerate(layers):
        df2 = deepcopy(df[1:])
        #breakpoint()
        df2['embeddings'] = list(layer)
        save_pickle(df2.to_dict('records'), os.path.join(pdir, '777_full-' + str(model) + '_static3-interpolation_hs' + str(i+1) + '_gpt2-xl_cnxt_1024_embeddings.pkl'))



    
# NOTE: our pickles are aligned so that embedding for word x is the prediction of the model on words x-n:x-1. 
# for all layers, it's the output of the layer given x-n:x-1. so the first layer represents previous word. last layer represents current word.
# in the static embedding, the embedding in our pickle for word x is the glove embedding of x. 
# so to compare, static has to start at 1, where we get 0-->1. whereas contextual could start at 0. but we start at 1 for fair comparison. 
def get_interpolation(start, end, model):
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    layers = []
    # don't get last value for better comparison with static
    # NOTE: in static cannot interpolate between last word and next. in contextual you can since you are using the last
    # layer from the last word as the next word
    sl = np.array(list(pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-'+model+'hs'+str(start)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))['embeddings'][1:]))
    df = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-'+model+'hs'+str(end)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))[1:]
    #breakpoint()
    el = np.array(list(df['embeddings'])) # note that you don't do [1:] here because using df, which you already indexed into. 
    linfit = interp1d([start, end], [sl,el], axis = 0) 
    layers = [linfit(i) for i in range(start, end+1)] 
    for i,layer in enumerate(layers):
        df2 = deepcopy(df)
        #breakpoint()
        df2['embeddings'] = list(layer)
        save_pickle(df2.to_dict('records'), os.path.join(pdir, '777_full-paper-' + str(model) + '_1k-contextual2-interpolation_hs' + str(i+1) + '_gpt2-xl_cnxt_1024_embeddings.pkl'))

# get more interpolated values between set of layers (defined by 
def get_more_interpolation(start, end, model, f):
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/' # store on della scratch
    #pdir = '/tigress/eham/247-encoding-updated/data/podcast2/777/pickles/'  # store on tigress
    layers = []
    
    # don't get last value for better comparison with static
    # NOTE: in static cannot interpolate between last word and next. in contextual you can since you are using the last
    # layer from the last word as the next word
    sl = np.array(list(pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-'+model+'hs'+str(start)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))['embeddings'][1:]))
    df = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-'+model+'hs'+str(end)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))[1:]
    breakpoint()
    el = np.array(list(df['embeddings'])) # note that you don't do [1:] here because using df, which you already indexed into. 
    # f = 1 corresponds to get_interpolation
    linfit = interp1d([start, f*end-(f-1)], [sl,el], axis = 0) # number of total layers in end is f*end- (f-1)
    layers = [linfit(i) for i in range(start, f*end- f+2)] # f*end - (f-1) + 1 --> f*end - f + 1 + 1 = f*end - f + 2
    # NOTE: layers[1] of original becomes layers[f] of new one. 
    breakpoint()
    for i,layer in enumerate(layers):
        df2 = deepcopy(df)
        #breakpoint()
        df2['embeddings'] = list(layer)
        save_pickle(df2.to_dict('records'), os.path.join(pdir, '777_full-paper-' + str(model) + '_1k-contextual3-interpolation_hs' + str(i+1) + '_gpt2-xl_cnxt_1024_embeddings.pkl'))


def mixing_interpolation(num_layers, model):
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    df = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-'+model+'hs'+str(1)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))[1:]
    for i in range(num_layers):
        if i ==0:
            df2 = deepcopy(df)
            save_pickle(df2.to_dict('records'), os.path.join(pdir, '777_full-paper-' + str(model) + '_mixing-interpolation_hs' + str(i+1) + '_gpt2-xl_cnxt_1024_embeddings.pkl'))
            continue
        
        start = i
        end = i + 1
        print(start, '-->', end)
        sl = np.array(list(map(np.array, pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-'+model+'hs'+str(start)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))['embeddings'][1:])))
        el = np.array(list(map(np.array, pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-'+model+'hs'+str(end)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))['embeddings'][1:])))
        
        reg = LinearRegression()
        reg.fit(sl, el)
        print(reg.score(sl, el))
        layer = reg.predict(sl)
        df2 = deepcopy(df)
        df2['embeddings'] = list(layer)
        save_pickle(df2.to_dict('records'), os.path.join(pdir, '777_full-paper-' + str(model) + '_mixing-interpolation_hs' + str(i+1) + '_gpt2-xl_cnxt_1024_embeddings.pkl'))


if __name__ == '__main__':
    start = 1
    end = 48
    #layers = get_interpolation(start, end, 'gpt2-xl')
    layers = get_more_interpolation(start, end,'gpt2-xl', 22)
    breakpoint()
    #layers = get_static_interpolation('gpt2')
    #mixing_interpolation(48, 'gpt2-xl')

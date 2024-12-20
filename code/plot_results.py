#-----------------------------------------------------#
# plot results                                        #
#-----------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import glob
import pandas as pd

from brain_color_prepro import get_e_list
from paper_plots import quantify_nans,paper_enc, encoding_lag_layer
# ba: brain area
def get_params(elec_list, num_layers, ba):
    if ba == 'nyu_ifg':
        color = 'black'
    elif ba == 'pton1_mSTG': 
        color = 'green'
    elif ba == 'pton1_aSTG':
        color = 'red'
    elif ba == 'pton1_TP':
        color = 'blue'
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    #breakpoint()
    num_lags = len(lags)
    layer_list = list(np.arange(1, num_layers +1))
    num_electrodes = len(elec_list)
    params = {'color':color,'lshift': lshift, 'lags': lags, 'num_lags': num_lags, 'layer_list': layer_list, 'num_layers': num_layers, 'num_electrodes': num_electrodes, 'ba':ba}
    
    return params

def preprocess_encoding_results(fpath, e_list, ps):
    # parameters
    assert(ps['num_layers'] == len(ps['layer_list']))
    out = np.zeros((ps['num_layers'], ps['num_electrodes'], ps['num_lags']))
    for i, l in enumerate(ps['layer_list']):
        for j, e in enumerate(e_list):
            file_list = glob.glob(os.path.join(fpath+str(l), '777',str(e)+'_comp.csv'))
            if len(file_list) != 1:
                breakpoint()
            #breakpoint()
            assert(len(file_list) == 1)    
            for file in file_list:
                ve = file.split('/')[-1]
                es = len(ve.split('_'))
                ce = '_'.join(ve.split('_')[:es-1])
                if ce not in e_list:
                    breakpoint()
                assert(ce in e_list)
                e_sig = pd.read_csv(file)
                # NOTE: old (mine)
                #sig_len = len(e_sig.loc[0])
                sig_len = len(e_sig.columns)
                e_f = pd.read_csv(file, names = range(sig_len))
                e_Rs = list(e_f.loc[0])
                out[i,j,:] = e_Rs
    
    return out

def paper_prepro(elec_list, slag, elag, omit_e, fpath, verbose, ps):
    half_lags = ps['num_lags']//2
    start = math.floor(slag/ps['lshift']) +half_lags # reverse of below 161//2 = 80. 
    end = math.floor(elag/ps['lshift']) + half_lags
    if verbose == True:print(fpath)
    
    all_sig_R = preprocess_encoding_results(fpath, elec_list, ps)
    print('num nans: ', quantify_nans(all_sig_R))
    layer_count = list(np.zeros(ps['num_layers']))
    max_lag_per_layer = []
    med_lag_per_layer = []
    
    for i in range(ps['num_layers']):
        
        # top lags 
        max_elags = []
        med_elags = []
        for j, e in enumerate(elec_list):
            assert(e not in omit_e)
            if verbose==True:print(e)
            big_e_array = all_sig_R[i,j]
            # iterate through layers. 
            # get the lags sorted by correlation
            top_lags = np.argsort(big_e_array, axis=-1)
            # remove lags out of the range
            top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
            # verify no nan entries
            top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[top_lags[p]])]
            # median
            top_lags_med = np.median(top_lags)
            # convert to ms
            assert(ps['lshift']*(top_lags_med-half_lags) == ps['lags'][int(top_lags_med)])
            med_elags.append(ps['lshift']*(top_lags_med-half_lags))
            # max
            top_lags_max = top_lags[-1]
            #breakpoint()
            max_elags.append(ps['lags'][top_lags_max])

        # average the top lags for all electrodes for a given layer
        med_lag_per_layer.append(np.mean(med_elags))
        max_lag_per_layer.append(np.mean(max_elags))
        
    assert(len(med_lag_per_layer) == len(max_lag_per_layer))
    assert(len(med_lag_per_layer) == ps['num_layers'])
    encoding_array = np.mean(all_sig_R, axis = 1)

    return med_lag_per_layer, max_lag_per_layer, encoding_array 

def sep_paper_plots(brain_area, num_layers, encoding_name):
    
    elec_list = get_e_list(brain_area+ '_e_list.txt', '\t')
    print(elec_list)
    print(encoding_name)

    params = get_params(elec_list, num_layers, brain_area)
    fpath = os.path.join(os.getcwd(),'results/',encoding_name + '-hs')
    omit_e = [] 
    opath = os.path.join(os.getcwd(), 'results/figures/')
    el_med,el_max, enc_bea = paper_prepro(elec_list, -500, 500, omit_e, fpath, False, params)
    
    # plot normalized encoding
    fig, ax3 = plt.subplots(figsize=[55,50])
    paper_enc(ax3, enc_bea, -500, 750, True, params)
    fig.savefig(opath + encoding_name + '-enc-norm-' + brain_area+ '.png') 
    plt.close()

    # plot lag layer
    fig, ax0 = plt.subplots(figsize=[55,50])
    #filter size for smoothing 1: no filtering
    fs=1
    encoding_lag_layer(ax0, enc_bea, fs, params)
    fig.savefig(opath + encoding_name + '-encoding-laglayer_' + brain_area+  '_'+str(fs)+'.png') 
    plt.close()
    
    # Plot encoding 
    fig, ax2 = plt.subplots(figsize=[55,50])
    paper_enc(ax2, enc_bea, -500, 1500, False, params)
    fig.savefig(opath + encoding_name+ '-enc-zoom' + brain_area+ '.png') 
    plt.close()
 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding_name',type=str, default = '')
    parser.add_argument('--num_layers',type=int, default = 48)
    args = parser.parse_args()

    rois = ['pton1_TP', 'nyu_ifg', 'pton1_mSTG', 'pton1_aSTG']
    encoding_name = args.encoding_name
    num_layers =args.num_layers 
    for roi in rois: 
        sep_paper_plots(roi, num_layers,encoding_name)



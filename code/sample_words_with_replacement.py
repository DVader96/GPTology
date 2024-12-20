#----------------------------------------#
# for word boostrapping                  #
# create set of sampnles before encoding #
# so you can use same ones for all layers#
# and electrodes                         #
#----------------------------------------#

import numpy as np
from random import choices
from scipy.stats import pearsonr
from brain_color_prepro import get_e_list
import csv

import matplotlib.pyplot as plt
import matplotlib.colors as clr

def sample_words_with_replacement():
    num_words = 1697
    N = 1000
    # create N samples from 0 to 1696
    samps = []
    #breakpoint()
    for i in range(N):
        z = choices(np.arange(num_words), k=num_words)
        samps.append(z)
    #breakpoint()
    np.save(str(N) + '_samps_of_' + str(num_words) + '_words.npy',samps)

# average electrodes to get N sets of bootstrapped "encoding plots" per roi
def word_bootstrap_prepro(wv, roi, layer):
    N = 1000
    num_lags = 161
    elec_list = get_e_list(roi + '_e_list.txt', '\t')
    fdir = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/word_bootstrap_test/'
    bea = np.zeros((len(elec_list), N, num_lags))
    #breakpoint()
    for i, e in enumerate(elec_list):
        f = fdir + 'word_bootstrap_layer-' + str(layer) + '_electrode-' + e + '_' + wv + '.npy'
        #breakpoint()
        fv = np.load(f)
        bea[i,:,:] = fv
    #breakpoint()
    return bea

# compute lag that maximizes correlation for each bootstrapped "encoding plot"
# correlate this with layers (1, 2, ..., 48) to get N lag layer correlations
# construct confidence interval 
def CIs_for_word_bootstrap_lag_layer(wv, roi):
    N = 1000 
    num_layers = 48
    num_lags = 161
    max_lag_per_layer = np.zeros((num_layers, N))
    #breakpoint()
    for layer in range(1, num_layers+1):
        #breakpoint()
        bea = word_bootstrap_prepro(wv, roi, layer)
        be_avg = np.mean(bea, axis = 0) # output: 1000x161
        max_lag_per_layer[layer-1] = np.argmax(be_avg, axis=1)

    rs = []
    nan_count = 0
    for i in range(N):
        r = pearsonr(np.arange(1, 49), max_lag_per_layer[:,i])[0]
        if np.isnan(r): 
            nan_count +=1
            continue
        rs.append(r)
    
    print(roi, nan_count) 
    #breakpoint()
    ci = [np.percentile(rs, 2.5), np.percentile(rs, 97.5)] #95%
    return ci

# generate CIs for each lag, layer combination in the encoding plot
def CIs_for_word_bootstrap_encoding(wv,roi):
    num_layers = 48
    num_lags = 161
    cis = np.zeros((num_layers, num_lags, 2))
    for layer in range(1, num_layers+1):
        bea = word_bootstrap_prepro(wv, roi, layer)
        be_avg = np.mean(bea, axis = 0)
        #breakpoint() 
        for l in range(be_avg.shape[1]):
            ci = [np.percentile(be_avg[:,l], 2.5), np.percentile(be_avg[:,l], 97.5)] #95%   
            #breakpoint()
            cis[layer-1, l] = ci
    #breakpoint() 
    return cis

# plots CIs for each layer, shifted along axis
def plot_encoding_CIs(roi, wv):
    num_layers = 48
    lags = np.arange(-2000, 2025, 25)
    hues = list(np.arange(0, 240 + 240/(num_layers - 1), 240/(num_layers-1))/360) # red to blue
    c_a = list(map(lambda x: clr.hsv_to_rgb([x, 1.0, 1.0]), hues))
    
    f = 'encoding_confidence_intervals_'+roi +'_' + wv + '_07072022.npy'
    cis = np.load(f)
    #breakpoint()
    plt.figure()
    for layer in range(cis.shape[0]):
        c = c_a[layer]
        plt.fill_between(lags, cis[layer,:,0], cis[layer,:,1],color=c, alpha=0.2)
        #breakpoint()
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.savefig('plot_CIs_' + roi + '_' + wv + '.png')
    #ax.tick_params(axis='both', labelsize=128)
    #ax.set_xlim([xl, xr])

if __name__ == '__main__':
    rois = ['nyu_ifg', 'pton1_mSTG', 'pton1_aSTG', 'pton1_TP']
    word_values = ['correct','top5-incorrect', 'incorrect']#['correct']#, 'top5-incorrect']

    # plot CIs for encoding
    #for roi in rois:
    #    for wv in word_values:
    #        plot_encoding_CIs(roi, wv)
    #breakpoint()
    #sample_words_with_replacement()
    #tp = word_bootstrap_prepro('correct', 'nyu_ifg',1)
    #tlag_cis = CIs_for_word_bootstrap_encoding('correct', 'nyu_ifg')
    #tll_cis = CIs_for_word_bootstrap_lag_layer('correct', 'nyu_ifg')
   
   

    csv_o = open('laglayer_CIs2.csv', 'w')
    writer = csv.writer(csv_o)
    header = ['roi', 'correct1', 'correct2', 'top5-incorrect1', 'top5-incorrect2', 'incorrect1', 'incorrect2']
    writer.writerow(header)
    
    for roi in rois:
        ll_cis = []
        for wv in word_values:
            ll_cis.extend(CIs_for_word_bootstrap_lag_layer(wv, roi))
            #np.save('laglayer_confidence_interval_' + roi + '_' + wv + '_07072022.npy', ll_ci)
            #lag_cis = CIs_for_word_bootstrap_encoding(wv, roi)
            #breakpoint() 
            #np.save('encoding_confidence_intervals_' + roi + '_' + wv + '_07072022.npy', lag_cis)
        row = [roi]
        row.extend(ll_cis)
        writer.writerow(row)
    
    csv_o.close()

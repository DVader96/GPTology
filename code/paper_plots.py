import glob
import os
from copy import deepcopy

import matplotlib.colors as clr
from statsmodels.stats.multitest import fdrcorrection as fdr
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from random import sample, choices
seed = 42
rng = default_rng(seed)

import math
from brain_color_prepro import get_e_list, get_n_layers, save_e2l
import matplotlib.image as img
import pandas as pd
from scipy.stats import pearsonr
from utils import load_pickle
import random
import csv

def concat_csvs():
    breakpoint()
    efs = glob.glob('REVPAPER-[pn]*o-enc*.csv')
    combined_csv = pd.concat([pd.read_csv(f) for f in efs], axis=0)
    combined_csv.to_csv('REVPAPER-allrois-maxout-truortho-enc_export.csv', index=False)
    
    llfs = glob.glob('REVPAPER-[pn]*o-laglayer*.csv')
    combined_csv = pd.concat([pd.read_csv(f) for f in llfs], axis=0)
    combined_csv.to_csv('REVPAPER-allrois-maxout-truortho-laglayer_export.csv', index=False)


def quantify_nans(a):
    ts = 1
    for s in a.shape:
        ts*=s
    num_nan = np.count_nonzero(np.isnan(a))
    return num_nan/ts


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

def preprocess_247_encoding_results(fp, e_list, ps):
    assert(ps['num_layers'] == len(ps['layer_list']))
    out = np.zeros((ps['num_layers'], ps['num_electrodes'], ps['num_lags']))
    for i, l in enumerate(ps['layer_list']):
        for j, e in enumerate(e_list):
            fpath = fp + 'kw-tfs-full-798-gpt2-xl-lag10k-25-quarter-' + str(l).zfill(2) + '/kw-200ms-all-798/' + str(e) + '_comp.csv'
            #print(fpath)
            file_list = glob.glob(fpath)
            #print(file_list)
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
                #breakpoint()
                e_sig = pd.read_csv(file)
                fsize = len(e_sig.columns)
                out[i,j,:] = e_sig.columns[int((fsize - ps['num_lags'])/2):int((fsize - ps['num_lags'])/2) + ps['num_lags']]
    
    return out


def preprocess_encoding_fold_results(fpath, e_list, ps):
    #breakpoint()
    # parameters
    assert(ps['num_layers'] == len(ps['layer_list']))
    out = np.zeros((ps['num_layers'], ps['num_electrodes'], ps['num_lags']))
    for i, l in enumerate(ps['layer_list']):
        for j, e in enumerate(e_list):
            #breakpoint()
            file_list = glob.glob(os.path.join(fpath+str(l), '777',str(e)+'_comp_fold{}.csv'.format(ps['fold'])))
            #print(file_list)
            if len(file_list) != 1:
                breakpoint()
            #breakpoint()
            assert(len(file_list) == 1)    
            for file in file_list:
                ve = file.split('/')[-1]
                es = len(ve.split('_'))
                ce = '_'.join(ve.split('_')[:es-1])
                #if ce not in e_list:
                #    breakpoint()
                #assert(ce in e_list)
                #breakpoint()
                e_sig = pd.read_csv(file)
                #breakpoint()
                # NOTE: old (mine)
                #sig_len = len(e_sig.loc[0])
                sig_len = len(e_sig.columns)
                e_f = pd.read_csv(file, names = range(sig_len))
                e_Rs = list(e_f.loc[0])
                out[i,j,:] = e_Rs
    
    return out



def fast_preprocess_encoding_results(fpath, e_list, ps):
    #breakpoint()
    # parameters
    out = np.zeros((ps['num_layers'], ps['num_electrodes'], ps['num_lags']))
    for i, l in enumerate(ps['layer_list']):
        for j, e in enumerate(e_list):
            file_list = glob.glob(os.path.join(fpath+str(l), '777',str(e)+'_comp.csv'))
            if len(file_list) != 1:
                breakpoint()
            assert(len(file_list) == 1)    
            for file in file_list:
                ve = file.split('/')[-1]
                es = len(ve.split('_'))
                ce = '_'.join(ve.split('_')[:es-1])
                if ce not in e_list:
                    breakpoint()
                assert(ce in e_list)
                #e_sig = pd.read_csv(file)
                #breakpoint()
                # NOTE: old (mine)
                #sig_len = len(e_sig.loc[0])
                #sig_len = len(e_sig.columns)
                sig_len = 161
                e_f = pd.read_csv(file, names = range(sig_len))
                e_Rs = list(e_f.loc[0])
                out[i,j,:] = e_Rs
    
    return out


# ba: brain area

def get_params(elec_list, model, ba, wv):
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    #breakpoint()
    num_lags = len(lags)
    num_layers = get_n_layers(model)
    layer_list = list(np.arange(1, num_layers +1))
    num_electrodes = len(elec_list)
    params = {'lshift': lshift, 'lags': lags, 'num_lags': num_lags, 'layer_list': layer_list, 'num_layers': num_layers, 'num_electrodes': num_electrodes, 'wv':wv, 'ba':ba}
    
    return params

#def get_params_with_layer_list()

#modular_lag_layer_nocorr(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x):
def paper_plots(slag, elag, brain_area):
    core_roi = brain_area
    #breakpoint()
    elec_list = get_e_list(core_roi + '_e_list.txt', '\t')
    print(elec_list)

    params = get_params(elec_list, ) 
    from matplotlib import gridspec
    fig = plt.figure(constrained_layout = True, figsize=[50,15])
    gs = gridspec.GridSpec(1, 3, figure = fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    omit_e = [] 
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    el_med,el_max, enc_bea = naper_prepro(elec_list, -500, 500, omit_e, fpath, False, params)
    
    paper_lag_layer(ax1, el_med, el_max)
    paper_enc(ax2, enc_bea, -2000, 2000, False, params)
    paper_enc(ax3, enc_bea, -500, 500, True, params)
    plt.close()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/paper-plots_' + core_roi + '_' + str(slag)+ '_to_' + str(elag) + 'shuffle=false_SIG.png') 

#modular_lag_layer_nocorr(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x):
def sep_paper_plots(slag, elag, brain_area, model, word_value, ID, bea_only=False):
    
    elec_list = get_e_list(brain_area+ '_e_list.txt', '\t')
    print(elec_list)
    print(word_value)

    params = get_params(elec_list, model, brain_area, word_value)
    print(ID)
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + word_value + 'gpt2-xl-hs'
    #dir = os.path.join(os.getcwd(), '../results/podcast2-gpt2-xl-pca50d-full-REV2-train-pca50-symbolic-top1-correct-gpt2-xl-hs1/')
    omit_e = [] 
    opath = '/scratch/gpfs/eham/247-encoding-updated/results/figures/' 
    #opath = os.path.join(os.getcwd(), '../results/figures/')
    el_med,el_max, enc_bea = paper_prepro(elec_list, -500, 500, omit_e, fpath, False, params)
    
    if bea_only: 
        return enc_bea 
    
    #upt = 'mean' #u plot type (max or mean)
    upt = 'max'
    inverted_u_plot(enc_bea, brain_area, word_value, upt)

    # plot normalized encoding
    fig, ax3 = plt.subplots(figsize=[55,50])
    paper_enc(ax3, enc_bea, -500, 750, True, params)
    fig.savefig(opath + ID + '-enc-norm-' + brain_area+ '_' + word_value + 'cw_shuffle=false_SIG.png') 
    plt.close()

    # plot lag layer
    fig, ax0 = plt.subplots(figsize=[55,50])
    fs=1
    encoding_lag_layer(ax0, enc_bea, fs, params)
    fig.savefig(opath + ID + '-encoding-laglayer_' + brain_area+ '_' + word_value + '_'+str(fs)+'.png') 
    plt.close()
    
    # Plot encoding 
    fig, ax2 = plt.subplots(figsize=[55,50])
    paper_enc(ax2, enc_bea, -500, 1500, False, params)
    fig.savefig(opath + ID + '-enc-zoom' + brain_area+ '_' + word_value + 'cw_shuffle=false_SIG.png') 
    plt.close()
    
def sep_electrode_plots(slag, elag, brain_area, layers):
    core_roi = brain_area
    elec_list = get_e_list(core_roi + '_e_list.txt', '\t')
    print(elec_list)

    params = get_params(elec_list) 
       
    omit_e = [] 
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    opath = '/scratch/gpfs/eham/247-encoding-updated/results/figures/' 
    all_sig_R = preprocess_encoding_results(fpath, elec_list, params)
    if params['num_layers'] > 1:
        hues = list(np.arange(0, 240 + 240/(params['num_layers'] - 1), 240/(params['num_layers']-1))/360) # red to blue
    else:
        hues = [0]
    c_a = list(map(lambda x: clr.hsv_to_rgb([x, 1.0, 1.0]), hues))
    for layer in layers:
        color = c_a[layer-1] 
        #sep_electrode_enc_test(all_sig_R[layer-1], -1000, 1000, params, layer, color)
        #if layer == 24: color = [0, 100/255, 0]
        sep_electrode_enc_final(all_sig_R[layer-1], -1000, 1000, params, layer, color)
   
def plot_regression(model='gpt2-xl'):
    #elec_list = ['ifg_for_stg'] 
    #elec_list = ['mSTG_for_ifg']
    elec_list = ['aSTG_for_ifg']
    #elec_list = ['aSTG_for_mSTG']
    #elec_list = ['mSTG_for_aSTG']
    params = get_params(elec_list, model) 
    omit_e = []
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-regr-ifg-mSTG-correct-top1-gpt2-xl-hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pred-ifg-mSTG-correct-top1-gpt2-xl-hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pred-mSTG-ifg-correct-top1-gpt2-xl-hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-regr-mSTG-ifg-correct-top1-gpt2-xl-hs'
    ##fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-regr-ifg-mSTG-correct-top1-gpt2-xl-hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-regr-ifg-aSTG-correct-top1-gpt2-xl-hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-regr-aSTG-mSTG-correct-top1-gpt2-xl-hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-regr-mSTG-aSTG-correct-top1-gpt2-xl-hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-regr-mSTG-ifg-correct-top1-gpt2-xl-hs'
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-regr-aSTG-ifg-correct-top1-gpt2-xl-hs'
    el_med,el_max, enc_bea = paper_prepro(elec_list, -500, 500, omit_e, fpath, False, params)
    fig, ax2 = plt.subplots(figsize=[55,50])
    paper_enc(ax2, enc_bea, -2000, 2000, False, params)
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/new_' + elec_list[0] + '_regression_plot.png') 
    #fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/' + elec_list[0] + '_prediction_plot.png') 
    plt.close()


def paper_llR(el_med, el_max):
    med_val = pearsonr(np.arange(len(el_med)), el_med)
    med_lag_corr = med_val[0]
    med_sig = med_val[1]
    max_val = pearsonr(np.arange(len(el_max)), el_max) 
    max_lag_corr = max_val[0] 
    max_sig = max_val[1]
    return np.round(med_lag_corr,2), np.round(med_sig, 2), np.round(max_lag_corr, 2), np.round(max_sig, 2)

def encoding_lag_layer(ax, encoding_array, filter_size, ps):
    nl = ps['num_layers']
    maxes = [ps['lags'][np.argmax(encoding_array[j,:])] for j in range(nl)] 
    max_avg = maxes 
    # test moving window average
    if filter_size > 1:
        from scipy.ndimage import uniform_filter1d as uf
        max_avg = uf(max_avg, mode='nearest', size=filter_size)
    print ('max avg: ', max_avg)
    ax.plot(np.arange(1,len(max_avg)+1), max_avg, 'ok', markersize=30,color = ps['color'], linewidth=10,label='lag with\ntop R')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=42)
    max_lag_corr, max_sig = pearsonr(max_avg, np.arange(len(max_avg)))
    ax.set_title('R = ' + str(np.round(max_lag_corr,3)) + ' Sig = ' + str(np.round(max_sig,10)), fontsize=128)
    ax.tick_params(axis='both', labelsize=128)
    
    x = np.arange(1, len(max_avg)+1)
    m, b = np.polyfit(x, max_avg, 1)
    ax.plot(x, m*x + b,'--k', linewidth=10, color=ps['color'])
    ax.axis(ymin=-200,ymax=500)

def symbolic_encoding_lag_layer(ax, encoding_array, filter_size, ps):
    nl = ps['num_layers']
    #breakpoint()
    #print(encoding_array.shape)
    #maxes = [ps['lags'][np.argmax(encoding_array[j])] for j in range(nl)] 
    maxes = [ps['lags'][np.argmax(encoding_array[j,:])] for j in range(nl)] 
    #maxes3 = list(ps['lags'][np.argmax(encoding_array, axis = -1)])
    max_avg = maxes 
    # test moving window average
    if filter_size > 1:
        from scipy.ndimage import uniform_filter1d as uf
        max_avg = uf(max_avg, mode='nearest', size=filter_size)
    print ('max avg: ', max_avg)
    #ax.plot(np.arange(1,len(med_avg)+1), med_avg, '-o', markersize=2,color = 'blue', label='med lag') 
    #ax.plot(np.arange(1,len(max_avg)+1), max_avg, 'ok', markersize=30,color = ps['color'], linewidth=10,label='lag with\ntop R')
    ax.scatter(np.arange(1,len(max_avg)+1), max_avg,s=2000, color = ['teal','orange','magenta','darkviolet'], label='lag with\ntop R')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=42)
    #breakpoint()
    max_lag_corr, max_sig = pearsonr(max_avg, np.arange(len(max_avg)))
    ax.set_title('R = ' + str(np.round(max_lag_corr,3)) + ' Sig = ' + str(np.round(max_sig,10)), fontsize=128)
    #print('med R: ', str(med_lag_corr), ' max R: ', str(max_lag_corr))
    ax.tick_params(axis='both', labelsize=64)
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['phonology','morphology','syntax','semantics'])
    
    x = np.arange(1, len(max_avg)+1)
    m, b = np.polyfit(x, max_avg, 1)
    #print(ps['fold'], m, b)
    #M.append(m)
    #B.append(b)
    #ax.plot(x,y,'ok',markersize=5) #axs[ri]
    ax.plot(x, m*x + b,'--k', linewidth=10, color=ps['color'])
    ax.axis(ymin=-200,ymax=500)



def paper_lag_layer(ax, el_med, el_max, filter_size):
    ts = ' Top Lag Per Layer \n(MedR, MaxR) = ('
    med_lag_corr, med_sig, max_lag_corr, max_sig = paper_llR(el_med, el_max)
    #med_avg = el_med
    max_avg = el_max
    # test moving window average
    if filter_size > 1:
        from scipy.ndimage import uniform_filter1d as uf
        max_avg = uf(max_avg, mode='nearest', size=filter_size)
    print ('max avg: ', max_avg)
    #ax.plot(np.arange(1,len(med_avg)+1), med_avg, '-o', markersize=2,color = 'blue', label='med lag') 
    ax.plot(np.arange(1,len(max_avg)+1), max_avg, '-o', markersize=2,color = 'blue', linewidth=10,label='lag with\ntop R')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=42)
    ax.set_title('R = ' + str(max_lag_corr) + ' Sig = ' + str( max_sig), fontsize=128)
    print('med R: ', str(med_lag_corr), ' max R: ', str(max_lag_corr))
    ax.tick_params(axis='both', labelsize=128)

def single_paper_enc(ax,encoding_array, xl,xr,norm, ps, label, c, lw,ls):
    ts = ' Avg Electrode Encoding Per Layer'
    if norm == True:
        ts = 'Norm\n' + ts
    maxes = []
    ax.tick_params(axis='both', labelsize=128)
    maxes.append(ps['lags'][np.argmax(encoding_array[0,:])])
    if norm == True:
        ax.plot(ps['lags'], encoding_array[0,:]/np.max(encoding_array[0,:]), color=c, label=label,zorder=-1, linewidth=lw,linestyle=ls) #**

    else:
        ax.plot(ps['lags'], encoding_array[0,:], color=c, label=label,zorder=-1,linewidth=lw,linestyle=ls) #**
    
    ax.legend(fontsize=32)
    if norm == False:
        ax.set_ylim([-0.0825, 0.5])
        ax.set_xlim([-2000, 2000])
    else: 
        ymin = 0.8
        ax.set_ylim([ymin, 1.08])
        ax.set_xlim([xl, xr])
        ax.set_yticks((ymin, 1.0))
        ax.set_yticklabels((str(ymin), '1.0'))

def paper_enc(ax,encoding_array, xl,xr,norm, ps):
    max_lags = list(ps['lags'][np.argmax(encoding_array, axis = -1)])
    ld = {}
    h = []
    h0 = 1.01
    alpha = 0.005
    for l in max_lags:
        if l in ld:
            h.append(h0 + alpha*ld[l])
            ld[l] +=1
        else:
            ld[l] = 1
            h.append(h0)
    num_layers = ps['num_layers']
    if num_layers > 1:
        hues = list(np.arange(0, 240 + 240/(num_layers - 1), 240/(num_layers-1))/360) # red to blue
    else:
        hues = [0]
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import cm
    c_a = list(map(lambda x: clr.hsv_to_rgb([x, 1.0, 1.0]), hues))
    cmap1 = LinearSegmentedColormap.from_list("mycmap", c_a)
    ts = ' Avg Electrode Encoding Per Layer'
    if norm == True:
        ts = 'Norm\n' + ts
    maxes = []
    for j in range(num_layers):
        maxes.append(ps['lags'][np.argmax(encoding_array[j,:])])
        c = c_a[j]
        if norm == True:
            ax.plot(ps['lags'], encoding_array[j,:]/np.max(encoding_array[j,:]), color=c, label='layer' + str(j+1),zorder=-1) #**
            ax.scatter([max_lags[j]], h[j], color=c, s=500, zorder=-1)

        else:
            ax.plot(ps['lags'], encoding_array[j,:], color=c, label='layer' + str(j+1),zorder=-1) #**
    if norm == False:
        ax.set_ylim([-0.02, 0.325])
        ax.set_xlim([-2000, 2000])
    else: 
        ymin = 0.8
        ax.set_ylim([ymin, 1.15])
        ax.set_xlim([xl, xr])
        ax.set_yticks((ymin, 1.0))
        ax.set_yticklabels((str(ymin), '1.0'))
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap1))
    cbar.set_ticks([0,0.25, 0.5, 0.75,1])
    cbar.set_ticklabels([1, int(num_layers/4), int(num_layers/2),int(3*num_layers/4), num_layers])
    cbar.ax.tick_params(labelsize=128)
    ax.tick_params(axis='both', labelsize=128)

def inverted_u_plot(encoding_array, ba, wv, pt):
    #pt = 'max'
    # bar plot
    #means = np.mean(encoding_array, axis = -1) # get mean correlation for each layer
    #errs = np.std(encoding_array, axis = -1)/np.sqrt(encoding_array.shape[0]) # standard error
    #errs = np.std(encoding_array, axis = -1)**2 # variance
    x = np.arange(1, encoding_array.shape[0] + 1)
    fig, ax = plt.subplots()
    
    if pt == 'mean':
        means = np.mean(encoding_array, axis = -1) # get mean correlation for each layer
        #errs = np.std(encoding_array, axis = -1)/np.sqrt(encoding_array.shape[0]) # standard error
        errs = np.std(encoding_array, axis = -1)**2 # variance
        ax.bar(x, means, yerr=errs, align = 'center', alpha=0.5, ecolor='black', capsize=0)
    elif pt == 'max':
        maxes = np.max(encoding_array, axis = -1) # get mean correlation for each layer
        ax.bar(x, maxes, align = 'center', alpha=0.5, capsize=0)

    ax.set_ylabel('R')
    ax.set_title('Inverted U Curve')
    ax.set_xlim([0.5, 48.5])
    ax.set_xticks([1, 24, 48])
    ax.set_xticklabels([1, 24, 48])

    plt.tight_layout()
    plt.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/inverted_u_bar_' + ba + '_' + wv[:-1] + '_' + pt + '.png')

    '''
    # line plot
    fig = plt.figure()
    mv = np.max(encoding_array, axis = -1) # get maximum correlation per layer
    plt.plot(range(1, encoding_array.shape[0] + 1), mv, label = 'max R')
    plt.xlabel('Layer')
    plt.ylabel('R')
    plt.legend()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/inverted_u_' + ba + '_' + wv + '.png')
    '''
def sep_electrode_enc_test(encoding_array, xl, xr, ps, layer, color):
    print('layer: ', layer)
    avg = np.mean(encoding_array, axis = 0)
    e_list = list(np.arange(encoding_array.shape[0]))
    #e_list = [14, 15, 20]
    for e in e_list:
        #breakpoint()
        fig, ax = plt.subplots(figsize=[55,50])
        ax.plot(ps['lags'], encoding_array[e], color=color, label='layer' + str(layer),linewidth=10,zorder=-1) #**
        ax.plot(ps['lags'], avg, color=color, label='layer' + str(layer) + ' avg', linestyle='-.',linewidth=10,zorder=-1) #**
        ax.set_xlim([xl, xr])
        ax.set_ylim([-0.02, 0.5])
        ax.tick_params(axis='both', labelsize=128)
        ax.vlines([0], -0.02, 0.4, colors = ['grey'], linestyles=['dashed'], linewidth=10)
        max_lag = ps['lags'][np.argmax(encoding_array[e])]
        ax.set_title(str(max_lag) + 'ms', fontsize=128)
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/max_label_layer_' + str(layer) + '_single_e' + str(e) + '_paper_test.png') 
        plt.close()

def sep_electrode_enc_final(encoding_array, xl, xr, ps, layer, color):
    print('layer: ', layer)
    avg = np.mean(encoding_array, axis = 0)
    if layer == 1:
        e_list = [4, 14, 15]
    elif layer == 24:
        e_list = [9, 12, 17]
    elif layer == 48:
        e_list = [16, 17, 20]
    fig, ax = plt.subplots(figsize=[55,50])
    alpha = 0.375
    for e in e_list:
        ax.plot(ps['lags'], encoding_array[e], alpha = alpha, color=color, label='layer' + str(layer),linewidth=14,zorder=1) #**
        #alpha-= 0.25
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-0.02, 0.5])

    ax.vlines([0], -0.02, 0.5, colors = ['grey'], linestyles=['dashed'], linewidth=10, alpha=0.5, zorder = -1)
    ax.plot(ps['lags'], avg, color=color, label='layer' + str(layer) + ' avg',linewidth=20, alpha = 1, zorder = 0) #**
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/vl_test_final_layer_' + str(layer) + '_single_e' + str(e) + '_paper_test.png') 
    plt.close()


def paper_prepro(elec_list, slag, elag, omit_e, fpath, verbose, ps):
    half_lags = ps['num_lags']//2
    start = math.floor(slag/ps['lshift']) +half_lags # reverse of below 161//2 = 80. 
    end = math.floor(elag/ps['lshift']) + half_lags
    if verbose == True:print(fpath)
    
    all_sig_R = preprocess_encoding_results(fpath, elec_list, ps)
    #all_sig_R = preprocess_247_encoding_results(fpath, elec_list, ps)
    #all_sig_R = preprocess_encoding_fold_results(fpath, elec_list, ps) # for folds
    #breakpoint()
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
            #breakpoint()
            '''
            thresholded_lags = [top_lags[p] for p in range(len(top_lags)) if big_e_array[top_lags_nonan[p]] > 0] # in general
            if len(thresholded_lags) != len(top_lags):
                print('change', len(thresholded_lags))
            if len(thresholded_lags) == 0:
                breakpoint()
            '''
            #assert(len(top_lags_nonan) == len(top_lags)) # verify no nan values bc not doing significance test. 
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
    # get elecrode values
    #import csv
    #csv_o = open(ps['ba'] + '_' + ps['wv'] + '_e_vals.csv', 'w')
    #writer = csv.writer(csv_o)
    #header = ['electrode', 'roi']
    #header.extend(list(range(1, 49)))
    #writer.writerow(header)
    #breakpoint()
    #for j,e in enumerate(elec_list):
    #    data = [str(e), ps['ba']]
    #    print(str(e) + '\n')
    #    #breakpoint()
    #    ear_ray = all_sig_R[:,j, :]
    #    mv = ps['lags'][np.argmax(ear_ray, axis=1)]
    #    data.extend(mv)
    #    writer.writerow(data)
    #    #for val in mv: print(val)
    #csv_o.close()
    #breakpoint() 
    encoding_array = np.mean(all_sig_R, axis = 1)
    '''
    max_lags = np.argmax(encoding_array, -1)
    es = all_sig_R[list(range(ps['num_layers'])),:,max_lags.tolist()]
    csv_o = open(ps['ba'] + '_' + ps['wv'] + 'bar_vals.csv', 'w')
    writer = csv.writer(csv_o)
    header = ['layer', 'electrodes']
    writer.writerow(header)
    for i in range(ps['num_layers']):
        data = [str(i+1)]
        data.extend(es[i])
        writer.writerow(data)
    csv_o.close()

    '''
    return med_lag_per_layer, max_lag_per_layer, encoding_array 
def compare_pickles(p1,p2):
    p1 = pd.DataFrame(load_pickle(p1))
    p2 = pd.DataFrame(load_pickle(p2))
    from lcs import lcs
    p1f, p2f = lcs(p1.columns, p2.columns)
    cols = p1.columns[p1f]
    out = False
    #for col in cols:
    #breakpoint()
    for col in ['word', 'onset', 'embeddings', 'speaker']:
        #breakpoint(
        print(col)
        if col == 'glove50_embeddings':
            continue
        #if col != 'glove50_embeddings':
        #    continue
        #breakpoint()
        #if isinstance(p1[col][0], list):
        if isinstance(p1[col][0], (list, tuple, np.ndarray)):
            #breakpoint()
            out = out or not np.allclose(np.vstack(p1[col]), np.vstack(p2[col]), atol = 1e-8)
            #for i in range(len(p1)):
            #    breakpoint()
            #    #out = out or np.allclose(p1[col],p2[col])
        else:
            out = out or np.any(p1[col] != p2[col])
        print(out) 
    if out:
        return 'No Match' 
    
    return 'Match'

def monte_carlo_permutation(roi_1, roi_2, word_value):
    print(roi_1, ' ', roi_2, word_value)
    #roi_list = ['nyu_ifg', 'pton1_mSTG']
    roi_list = [roi_1, roi_2]
    #word_value = 'correct'
    #word_value = 'top5-incorrect'
    N = 1000
    model = 'gpt2-xl'
    ID ='PAPER2'
    slope_difs = []
    e_union = []
    num_es = []

    for roi in roi_list:
        elec_list = get_e_list(roi + '_e_list.txt', '\t')
        fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + word_value + '-gpt2-xl-hs'
        ps = get_params(elec_list, model,roi, word_value)
        num_es.append(ps['num_electrodes'])
        breakpoint()
        all_sig_R = preprocess_encoding_results(fpath, elec_list, ps)
        e_union.append(all_sig_R)
    
    # compute true difference 
    true_1 = np.mean(e_union[0], axis=1)
    true_1ll = np.argmax(true_1, axis=1)
    t1_scaled = ps['lags'][true_1ll]
    true_2 = np.mean(e_union[1], axis=1)
    true_2ll = np.argmax(true_2, axis=1)
    t2_scaled = ps['lags'][true_2ll]
    tm1, tb1 = np.polyfit(np.arange(1,49), t1_scaled, 1)
    tm2, tb2 = np.polyfit(np.arange(1,49), t2_scaled, 1)
    true = tm2 - tm1
    #print(roi_list[0], ': ', tm1, roi_list[1], ': ', tm2)

    for i in range(1, N+1):
        #print(i, '/', N)
        # pool both sets of electrodes. 
        ea = np.hstack(e_union) # ea.shape = (layers, electrodes, lags)
        # randomly sample electrodes WITHOUT replacement
        # NOTE: each layer gets same electrodes (just like we normally would do)
        #breakpoint()
        z = sample(set(np.arange(np.sum(num_es))), num_es[0])
        # assemble remaining electrodes
        first_roi = ea[:, z, :]
        z2 = [i for i in range(np.sum(num_es)) if i not in z]
        second_roi = ea[:,z2 , :] 
        # average electrodes in each array
        first_avg = np.mean(first_roi, axis=1)
        second_avg = np.mean(second_roi, axis=1)
        # compute lag layers
        first_ll = np.argmax(first_avg, axis = 1)
        fll_scaled = ps['lags'][first_ll]
        second_ll = np.argmax(second_avg, axis = 1)
        sll_scaled = ps['lags'][second_ll]
        # compute slopes
        m1, b1 = np.polyfit(np.arange(1,49), fll_scaled, 1)
        m2, b2 = np.polyfit(np.arange(1,49), sll_scaled, 1)
        # slope difference
        #dif = np.abs(m2 - m1)
        dif = m2 - m1
        slope_difs.append(dif)

    # find where true difference fits in distribution in terms of percentile. 
    #breakpoint()
    from scipy.stats import percentileofscore
    # weak: percentile of 80% means 80% of values are 
    # less than or equal to the provided score
    pcnt = percentileofscore(slope_difs, true, kind = 'weak') 
    print('Percentile: ', pcnt)

def bootstrapping_within_roi_test(roi, word_value):
    print(roi, word_value)
    N = 10000
    model = 'gpt2-xl'
    ID ='PAPER2'
    #slope_difs = []
    #e_union = []
    num_es = []
    # get roi electrodes
    elec_list = get_e_list(roi + '_e_list.txt', '\t')
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + word_value + '-gpt2-xl-hs'
    ps = get_params(elec_list, model,roi, word_value)
    num_es.append(ps['num_electrodes'])
    all_sig_R = preprocess_encoding_results(fpath, elec_list, ps)
    # get actual R
    t_avg = np.mean(all_sig_R, axis=1)
    t_ll = np.argmax(t_avg, axis = 1)
    t_scaled = ps['lags'][t_ll]
    tpr = pearsonr(np.arange(1,49), t_scaled)[0]
    print('True R: ', tpr)

    prs = []
    i = 1
    while i < N+1:
        #breakpoint()
        # sample electrodes with replacement. 
        z = choices(np.arange(all_sig_R.shape[1]), k= all_sig_R.shape[1])
        rs = all_sig_R[:,z,:]
        # get lag layer
        first_avg = np.mean(rs, axis=1)
        first_ll = np.argmax(first_avg, axis = 1)
        fll_scaled = ps['lags'][first_ll]
        # this will give undefined pearsonr. 
        if len(set(fll_scaled)) == 1:
            #breakpoint()
            continue # redo this loop
        pr = pearsonr(np.arange(1,49), fll_scaled)[0]
        prs.append(pr)
        i+=1

    #breakpoint()
    # get confidence interval
    #print('[', np.percentile(prs, .005),',', np.percentile(prs, 99.995), ']')
    #print('[', np.percentile(prs, .5),',', np.percentile(prs, 99.5), ']') #99%
    #breakpoint()
    print('[', np.percentile(prs, 2.5),',', np.percentile(prs, 97.5), ']') #95%
    print('[', np.percentile(prs, 5),',', np.percentile(prs, 95), ']') #90%

def permute_laglayer(roi, word_value):
    print(roi, word_value)
    N = 100000
    model = 'gpt2-xl'
    ID ='PAPER2'
    #slope_difs = []
    #e_union = []
    num_es = []
    # get roi electrodes
    elec_list = get_e_list(roi + '_e_list.txt', '\t')
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + word_value + '-gpt2-xl-hs'
    ps = get_params(elec_list, model,roi, word_value)
    num_es.append(ps['num_electrodes'])
    all_sig_R = preprocess_encoding_results(fpath, elec_list, ps)
    # get actual R
    t_avg = np.mean(all_sig_R, axis=1)
    t_ll = np.argmax(t_avg, axis = 1)
    t_scaled = ps['lags'][t_ll]
    idxs = list(np.arange(1,49))
    tpr = pearsonr(idxs, t_scaled)[0]
    print('True R: ', tpr)
    # permute
    rs = []
    rng = np.random.default_rng()
    
    for i in range(N):
        #breakpoint()
        # shuffle indices
        idxs = list(np.arange(1,49))
        rng.shuffle(idxs)
        # recompute correlation
        r = pearsonr(idxs, t_scaled)[0]
        rs.append(r)

    breakpoint()
    from scipy.stats import percentileofscore
    # weak: percentile of 80% means 80% of values are 
    # less than or equal to the provided score
    pcnt = percentileofscore(rs, tpr, kind = 'weak')
    print('Percentile: ', pcnt/100)
    print('1-Percentile: ', 1-(pcnt/100))
    mtpr = np.maximum(tpr, -tpr)
    count = len([val for val in rs if val >= mtpr])
    print('P-Value', (count+1)/(N+1))

def csv_lag_layer(ID, word_values, roi_list):
    # NOTE: CSV lag layer
    #ID = 'truortho'
    #ID = 'REVPAPER'
    #ID = 'REVPAPER-CTXT-INTP2'
    #word_values = ['correct-' + ID + '-', 'top5-incorrect-' + ID + '-']
    #word_values = ['all-']
    #word_values = ['correct-' + ID + '-', 'top5-incorrect-' + ID + '-', 'all-' + ID + '-']
    #word_values = ['all-contextual-interpolation-']
    #word_values = ['all-', 'top1-correct-', 'top5-incorrect-']
    #roi_list = ['nyu_ifg', 'pton1_mSTG', 'pton1_aSTG', 'pton1_TP']
    print(ID)
    print(word_values)
    print(roi_list)

    import csv
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    #breakpoint()
    #csv_o = open('PAPER2-' + ID + '-laglayer_export.csv', 'w')
    csv_o = open(ID + '-laglayer_export.csv', 'w')
    #csv_o = open('test10.csv', 'w')
    writer = csv.writer(csv_o)
    header = ['roi', 'word value']
    header.extend(list(range(1, 49)))
    writer.writerow(header)
    for roi in roi_list:
        print(roi)
        for wv in word_values:
            print(wv)
            # only get bea from sep_paper_plots, no plotting
            bea = sep_paper_plots(-500, 500, roi, 'gpt2-xl', wv,ID, True)
            #breakpoint()
            mv = lags[np.argmax(bea, axis=1)]
            data = [str(roi),str(wv)]
            data.extend(mv.tolist())
            writer.writerow(data)
            #for val in mv: print(val)
    csv_o.close()

def csv_encoding(ID, word_values, roi_list):
    # NOTE: CSV encoding
    # make sure to edit sep_paper_plots to return after generating array
    #ID = 'truortho'
    #ID = 'REVPAPER'
    #ID = 'REVPAPER-CTXT-INTP2'
    #word_values = ['correct-' + ID + '-', 'top5-incorrect-' + ID + '-', 'all-' + ID + '-']
    #word_values = ['all-contextual-interpolation-']
    #word_values = ['all-contextual-interpolation-']
    #word_values = ['all-']
    #word_values = ['all-', 'top1-correct-', 'top5-incorrect-']
    #roi_list = ['nyu_ifg', 'pton1_mSTG', 'pton1_aSTG', 'pton1_TP']
    print(ID)
    print(word_values)
    print(roi_list)
    num_layers = 48
    hues = list(np.arange(0, 240 + 240/(num_layers - 1), 240/(num_layers-1))/360) # red to blue
    c_a = list(map(lambda x: clr.hsv_to_rgb([x, 1.0, 1.0]), hues))
    
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    #breakpoint()
    #csv_o = open('PAPER2-' + ID + '-enc_export.csv', 'w') # output file
    csv_o = open(ID + '-enc_export.csv', 'w') # output file
    writer = csv.writer(csv_o)
    header = ['roi', 'word value','layer', 'R', 'G', 'B']
    header.extend(lags.tolist())
    writer.writerow(header)
    for roi in roi_list:
        print(roi)
        for wv in word_values:
            print(wv)
            # only get bea from sep_paper_plots, no plotting
            bea = sep_paper_plots(-500, 500, roi, 'gpt2-xl', wv,ID, True )
            for layer in range(bea.shape[0]):
                #breakpoint()
                color = c_a[layer].tolist()
                data = [str(roi),str(wv), str(layer+1), color[0], color[1], color[2]]
                data.extend(bea[layer,:].tolist())
                writer.writerow(data)
    csv_o.close()

# generates csv where you first compute max lag for each electrode, then average these.
def csv_max_avg_lag_layer(ID, word_values, roi_list):
    print(ID)
    print(word_values)
    print(roi_list)
    model = 'gpt2-xl'
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    
    u_csv = open(ID + '-max-avg-laglayer_export.csv', 'w') # output file
    stde_csv = open(ID + '-max-avg-laglayer_stde_export.csv', 'w') # output file
    u_writer = csv.writer(u_csv)
    stde_writer = csv.writer(stde_csv)
    header = ['roi', 'word value']
    header.extend(list(range(1, 49)))
    u_writer.writerow(header)
    stde_writer.writerow(header)
    #breakpoint()
    for roi in roi_list:
        print(roi)
        elec_list = get_e_list(roi+ '_e_list.txt', '\t')
        print(elec_list)
        for wv in word_values:
            print(wv)
            ps = get_params(elec_list, model, roi, wv)
            fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + wv + 'gpt2-xl-hs'
            
            lel = preprocess_encoding_results(fpath, elec_list, ps) # layer x electrode x lag = lel
            mv = lags[np.argmax(lel, axis=-1)] # layers x electrodes 
            #breakpoint()
            # avg
            umv = np.mean(mv, axis = -1)
            data = [str(roi),str(wv)]
            data.extend(umv.tolist())
            u_writer.writerow(data)

            stderr = np.std(mv, axis=-1)/mv.shape[-1]
            stde_data = [str(roi), str(wv)]
            stde_data.extend(stderr.tolist())

            stde_writer.writerow(stde_data)


    u_csv.close()
    stde_csv.close()

def csv_ariel_latency_regression_lag_layer(ID, word_values, roi_list):
    print(ID)
    print(word_values)
    print(roi_list)
    model = 'gpt2-xl'
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    
    csv_o = open(ID + '-ariel-regress-laglayer_export.csv', 'w') # output file
    writer = csv.writer(csv_o)
    header = ['layer', 'roi', 'word value', 'patient', 'electrode', 'lag']
    writer.writerow(header)
    for roi in roi_list:
        print(roi)
        elec_list = get_e_list(roi+ '_e_list.txt', '\t')
        print(elec_list)
        for wv in word_values:
            print(wv)
            ps = get_params(elec_list, model, roi, wv)
            fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + wv + 'gpt2-xl-hs'
            
            lel = preprocess_encoding_results(fpath, elec_list, ps) # layer x electrode x lag = lel
            mv = lags[np.argmax(lel, axis=-1)]
            for i, e in enumerate(elec_list):
                #breakpoint()
                vals = e.split('_')
                if len(vals) == 3:
                    breakpoint()
                    elec_val = '_'.join(vals[1:])
                else:
                    elec_val = vals[-1]
                patient = vals[0]
                for layer in range(mv.shape[0]):
                    data = [str(layer+1), str(roi),str(wv), patient, elec_val, mv[layer][i]]
                    writer.writerow(data)
    csv_o.close()




def plot_single_electrode_laglayer(ID, word_values, roi_list):
    print(ID)
    print(word_values)
    print(roi_list)
    #breakpoint()
    #elec_list = get_e_list(core_roi + '_nonsig_e_list.txt', '\t')
    model = 'gpt2-xl'
 
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    for roi in roi_list:
        print(roi)
        for wv in word_values:
            print(wv)
            elec_list = get_e_list(roi+ '_e_list.txt', '\t')
            print(elec_list)
            ps = get_params(elec_list, model, roi, wv)
            fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + wv + 'gpt2-xl-hs'
            
            lel = preprocess_encoding_results(fpath, elec_list, ps) # layer x electrode x lag = lel
            mv = lags[np.argmax(lel, axis=-1)]
            
            # avg
            #umv = np.mean(mv, axis = -1)
            #stdmv = np.std(mv, axis = -1)

            #fig, ax = plt.subplots(figsize=[55,50])
            #ax.plot(np.arange(1,mv.shape[0]+1), umv, '-o', markersize=2,color = 'blue', linewidth=10,label='lag with\ntop R')
            #ax.fill_between(np.arange(1, mv.shape[0] + 1), umv-stdmv, umv+stdmv, color = 'blue', alpha=0.5)
            #ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=42)
            #max_lag_corr, max_sig = pearsonr(umv, np.arange(mv.shape[0]))
            #ax.set_title('Avg Single E Max R, R= ' + str(max_lag_corr), fontsize=128)
            #ax.tick_params(axis='both', labelsize=128)
            #fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/' + 
            #        ID + '-avg-single-electrode-peak-laglayer_' + roi + '_' + wv[:-1]+ '.png') 
            #plt.close()
            # single e
            for e in range(mv.shape[-1]): 
                # plot
                fig, ax = plt.subplots(figsize=[55,50])
                ax.plot(np.arange(1,len(mv[:,e])+1), mv[:,e], '-o', markersize=2,color = 'blue', linewidth=10,label='lag with\ntop R')
                ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=42)
                max_lag_corr, max_sig = pearsonr(mv[:,e], np.arange(mv.shape[0]))
                ax.set_title(elec_list[e] + ' R = ' + str(max_lag_corr), fontsize=128)
                #print('med R: ', str(med_lag_corr), ' max R: ', str(max_lag_corr))
                ax.tick_params(axis='both', labelsize=128)
                fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/' + 
                        ID + '-single-electrode-encoding-laglayer_' + roi + '_' + wv+ elec_list[e] + '.png') 
                plt.close()

def plot_single_electrode_encoding(ID, word_values, roi_list):
    print(ID)
    print(word_values)
    print(roi_list)
    #breakpoint()
    #elec_list = get_e_list(core_roi + '_nonsig_e_list.txt', '\t')
    model = 'gpt2-xl'
    num_layers = 48
    hues = list(np.arange(0, 240 + 240/(num_layers - 1), 240/(num_layers-1))/360) # red to blue
    c_a = list(map(lambda x: clr.hsv_to_rgb([x, 1.0, 1.0]), hues))

    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    for roi in roi_list:
        print(roi)
        for wv in word_values:
            print(wv)
            elec_list = get_e_list(roi+ '_e_list.txt', '\t')
            print(elec_list)
            ps = get_params(elec_list, model, roi, wv)
            fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + wv + 'gpt2-xl-hs'
            
            lel = preprocess_encoding_results(fpath, elec_list, ps) # layer x electrode x lag = lel
            for e in range(lel.shape[1]): 
                fig, ax = plt.subplots(figsize=[55,50])
                for layer in range(lel.shape[0]):
                    #breakpoint()
                    color = c_a[layer].tolist()
                    ax.plot(ps['lags'], lel[layer, e,:]/np.max(lel[layer,e,:]), color = color, label=layer)
                
                
                ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=42)
                ax.set_title(elec_list[e] + ' Encoding', fontsize=128)
                ax.tick_params(axis='both', labelsize=128)
                #print('med R: ', str(med_lag_corr), ' max R: ', str(max_lag_corr))
                fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/' + 
                        ID + '-single-electrode-layer-encoding-_' + roi + '_' + wv+ elec_list[e] + '.png') 
                plt.close()
 
# outputs bar plot cvs --> layer x electrode (separate file per roi, condition)
def bar_csv(ID, word_values, roi_list):
    print(ID)
    print(word_values)
    print(roi_list)
    model = 'gpt2-xl'
    num_layers = 48

    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    for roi in roi_list:
        print(roi)
        for wv in word_values:
            print(wv)
            elec_list = get_e_list(roi+ '_e_list.txt', '\t')
            print(elec_list)
            ps = get_params(elec_list, model, roi, wv)
            fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + wv + 'gpt2-xl-hs'
            
            lel = preprocess_encoding_results(fpath, elec_list, ps) # layer x electrode x lag = lel
            #breakpoint()
            # average over electrodes
            encoding_array = np.mean(lel, axis = 1)
            # compute max lag for electrode averaged encoding
            max_lags = np.argmax(encoding_array, -1)
            # get electrodes that average together to make max correlation (corresponding to lag above)
            es = lel[list(range(ps['num_layers'])),:,max_lags.tolist()]
            csv_o = open(ID + '_' + ps['ba'] + '_' + ps['wv'] + 'bar_vals.csv', 'w')
            writer = csv.writer(csv_o)
            header = ['layer', 'electrodes']
            writer.writerow(header)
            for i in range(ps['num_layers']):
                data = [str(i+1)]
                data.extend(es[i])
                writer.writerow(data)
            csv_o.close()


    return 0

def many_interpolation_test(word_values, roi_list, num_iter, num_layers):
    print(word_values)
    print(roi_list)
    model = 'gpt2-xl' 
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    
    ps = {'lshift': lshift, 'lags': lags, 'num_lags': len(lags), 'num_layers': 48}
    for roi in roi_list:
        print(roi)
        elec_list = get_e_list(roi+ '_e_list.txt', '\t')
        ps['ba'] = roi
        ps['num_electrodes'] = len(elec_list)
        for wv in word_values:
            print(wv)
            ps['wv'] = wv 
            tps = deepcopy(ps)
            tps['layer_list'] = list(np.arange(1, 49))
            # ge ps['layer_list']= layer_list
            #tfpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-REVPAPER-1k-contextual2-interpolation-' + wv + '-gpt2-xl-hs'
            if wv == 'top1-correct':
                tfpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-PAPER2-correct-gpt2-xl-hs'
            else:
                tfpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-PAPER2-' + wv + '-gpt2-xl-hs'
            tlel = preprocess_encoding_results(tfpath, elec_list, tps) # layer x electrode x lag = lel
            #breakpoint()
            tavg_lel = np.mean(tlel, axis=1)
            tmv = lags[np.argmax(tavg_lel, axis=-1)] 
            assert(tmv.shape[0] == tps['num_layers'])
            # fit best fit line. 
            tm, tb = np.polyfit(np.arange(1,49), tmv, 1)
            # t true slope
            print('True slope: ', tm)

            slopes = []
            for i in range(num_iter):
                if i % 100 ==0: print(i)
                layer_list = [1]
                # randomly sample 46 values from (1, num_layers)
                values = set(np.arange(2, num_layers)) # always include 1st and last layer, so don't include when sample
                assert 1 not in values
                assert num_layers not in values
                z = random.sample(values, k = ps['num_layers'] - 2)
                # sort values
                z = np.sort(z)
                layer_list.extend(z)
                layer_list.append(num_layers)
                assert len(layer_list) == ps['num_layers']
                ps['layer_list']= layer_list
                fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-REVPAPER-1k-contextual2-interpolation-' + wv + '-gpt2-xl-hs'
                breakpoint()
                lel = preprocess_encoding_results(fpath, elec_list, ps) # layer x electrode x lag = lel
                
                #breakpoint()
                avg_lel = np.mean(lel, axis=1)
                mv = lags[np.argmax(avg_lel, axis=-1)] 
                assert(mv.shape[0] == ps['num_layers'])
                # fit best fit line. 
                m, b = np.polyfit(np.arange(1,49), mv, 1)
                # add to array
                slopes.append(m)

            slopes = np.sort(slopes)
            from scipy.stats import percentileofscore
            # weak: percentile of 80% means 80% of values are 
            # less than or equal to the provided score
            pcnt = percentileofscore(slopes, tm, kind = 'weak') 
            print('Percentile: ', pcnt)


def faster_many_interpolation_test(word_values, roi_list, num_iter, num_layers):
    
    import csv
    csv_o = open('Correlation_histogram_data.csv', 'w')
    writer = csv.writer(csv_o)
    header = ['roi', 'word value', 'Interpolated R', 'Contextual R']
    iter_header = []
    for i in range(num_iter): iter_header.append('iter' + str(i+1))
    header.extend(iter_header)
    breakpoint()
    writer.writerow(header)
 
    
    
    print(word_values)
    print(roi_list)
    model = 'gpt2-xl' 
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    
    ps = {'lshift': lshift, 'lags': lags, 'num_lags': len(lags)}
    for roi in roi_list:
        print(roi)
        elec_list = get_e_list(roi+ '_e_list.txt', '\t')
        ps['ba'] = roi
        ps['num_electrodes'] = len(elec_list)
        for wv in word_values:
            
            print(wv)
            ps['wv'] = wv 
            tps = deepcopy(ps)
            tps['num_layers'] = 48
            tps['layer_list'] = list(np.arange(1, 49))
            # ge ps['layer_list']= layer_list
            #tfpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-REVPAPER-1k-contextual2-interpolation-' + wv + '-gpt2-xl-hs'
            if wv == 'top1-correct':
                tfpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-PAPER2-correct-gpt2-xl-hs'
            else:
                tfpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-PAPER2-' + wv + '-gpt2-xl-hs'

            #elif wv == 'top5-incorrect':
            #    #tfpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-PAPER2-incorrect-gpt2-xl-hs'
            tlel = preprocess_encoding_results(tfpath, elec_list, tps) # layer x electrode x lag = lel
            #breakpoint()
            tavg_lel = np.mean(tlel, axis=1)
            tmv = lags[np.argmax(tavg_lel, axis=-1)] 
            assert(tmv.shape[0] == tps['num_layers'])
            # fit best fit line. 
            #tm, tb = np.polyfit(np.arange(1,49), tmv, 1)
            tm = pearsonr(np.arange(1,49), tmv)[0]
            # t true slope
            print('True slope: ', tm)

            
            slopes = []
           
            ps['num_layers'] = num_layers
            ps['layer_list'] = list(np.arange(1, num_layers+1))
            
            fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-REVPAPER-1k-contextual3-interpolation-' + wv + '-gpt2-xl-hs'
            lel = fast_preprocess_encoding_results(fpath, elec_list, ps) # layer x electrode x lag = lel
            avg_lel = np.mean(lel, axis=1)
            mv = lags[np.argmax(avg_lel, axis=-1)] 
            #breakpoint()
            #print(mv[::22]) # sanity check
            #tim, tib = np.polyfit(np.arange(1,49),mv[::22], 1)
            tim = pearsonr(np.arange(1, 49), mv[::22])[0]
            #breakpoint()
            print('OG Interpolate Slope Should be: ', tim)
            csv_data = [roi, wv, tim, tm]
            # DEBUGGING
            #ps['num_layers'] = 48
            #ps['layer_list']= list(np.arange(1, 1036, 22))
            #ps['layer_list'] = list(np.arange(1, 49))
            #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-TESTREVPAPER-1k-contextual3-interpolation-top1-correct-gpt2-xl-hs'
            #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-REVPAPER-CTXT-INTP2-' + wv + '-gpt2-xl-hs'
            #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-TESTREVPAPER-ws200-1k-contextual3-interpolation-top1-correct-gpt2-xl-hs'
            #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-TESTREVPAPER-ws200-1k-contextual3-interpolation-top5-incorrect-gpt2-xl-hs'
            #            #print(mv)
            #print(mv[::22])
            # test CNTXT 
            #og_path = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-REVPAPER-CTXT-INTP2-' + wv + '-gpt2-xl-hs'
            #og_path = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-TESTREVPAPER-contextual2-interpolation-top1-correct-gpt2-xl-hs'
            #og_path = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-TESTREVPAPER-ws200-contextual2-interpolation-top1-correct-gpt2-xl-hs'
            #og_path = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-REVPAPER-CTXT-INTP2-' + wv + '-gpt2-xl-hs'
            #ps['layer_list'] = list(np.arange(1, 49))
            #og = fast_preprocess_encoding_results(og_path, elec_list, ps) # layer x electrode x lag = lel
            #breakpoint()
            #ogu = np.mean(og, axis=1)
            #og_mv = lags[np.argmax(ogu, axis=-1)]

            for i in range(num_iter):
                #if i % 100 ==0: print(i)
                layer_list = [0] # layer 1 indexed at 0
                # randomly sample 46 values from (1, num_layers)
                values = set(np.arange(1, num_layers-1)) # always include 1st (0) and last layer (1000-1 = 999), so don't include when sample
                assert 0 not in values
                assert num_layers-1 not in values # num_layers-1 is index of last layer
                z = random.sample(values, k = 46)# 48 layers minus 1st and last (z is indices)
                # sort values
                z = np.sort(z)
                assert(len(set(z)) == len(z)) # make sure no duplicates
                layer_list.extend(z)
                layer_list.append(num_layers-1) # layer 1000
                #if i % 50 == 0: print(layer_list)
                #breakpoint()
                #breakpoint()
                #iter_lel = lel[layer_list,:,:] # old
                #assert not np.may_share_memory(iter_lel, lel)
                #assert iter_lel.shape[0] == 48
                #breakpoint()
                iter_mv = mv[layer_list]
                assert(iter_mv.shape[0] == 48)
                #breakpoint()
                #avg_lel = np.mean(iter_lel, axis=1)
                #mv = lags[np.argmax(avg_lel, axis=-1)] 
                # fit best fit line. 
                #m, b = np.polyfit(np.arange(1,49), iter_mv, 1)
                #breakpoint()
                m = pearsonr(np.arange(1, 49), iter_mv)[0]
                # add to array
                #print(m)
                slopes.append(m)

            slopes = np.sort(slopes)
            #breakpoint()
            csv_data.extend(slopes)
            writer.writerow(csv_data)
            #breakpoint()
            '''
            plt.figure()
            plt.hist(slopes, bins=100)
            plt.axvline(tm, linestyle='--', color = 'k', label = 'Contextual Embeddings')
            plt.axvline(tim, linestyle='--', color = 'b', label='Interpolated Embeddings')
            plt.legend()
            plt.savefig('Correlations_histogram_' + roi + '_' + wv + '.png')
            # weak: percentile of 80% means 80% of values are 
            # less than or equal to the provided score
            '''
            from scipy.stats import percentileofscore
            pcnt = percentileofscore(slopes, tm, kind = 'strict') 
            print('Percentile: ', pcnt)


    csv_o.close()


if __name__ == '__main__':
    #word_values = ['top1-correct', 'top5-incorrect']#, 'all']
    #word_values = ['top5-incorrect']#, 'all']
    #word_values = ['all']
    #roi_list = ['pton1_TP', 'nyu_ifg', 'pton1_mSTG', 'pton1_aSTG']
    #roi_list = ['pton1_mSTG']
    #roi_list = ['nyu_ifg', 'pton1_aSTG', 'pton1_TP','pton1_mSTG']
    #roi_list = ['pton1_aSTG']
    #num_iter = 10000
    #num_layers = 1035
    #faster_many_interpolation_test(word_values, roi_list, num_iter, num_layers)

    #breakpoint()
    # different window sizes
    #rois = ['pton1_TP', 'nyu_ifg', 'pton1_mSTG', 'pton1_aSTG']
    #word_values = ['correct-','all-', 'top5-incorrect-']
    #word_values = ['all-']
    #window_sizes = ['100', '50', '300']
    #window_sizes = ['300']
    #for ws in window_sizes:
    #    ID = 'REVPAPER-WS' + ws
    #    csv_encoding(ID, word_values, rois)
    #    csv_lag_layer(ID, word_values, rois)
    #breakpoint()
    # contextual interpolation
    #ID = 'REVPAPER-CTXT-INTP2'
    #word_values = ['all-', 'top1-correct-', 'top5-incorrect-']
    
    # normal stuff
    #ID = 'PAPER2'
    #rois = ['pton1_TP', 'nyu_ifg', 'pton1_mSTG', 'pton1_aSTG']
    word_values = ['all-']#, 'correct-', 'top5-incorrect-']
    
    # truortho (project out)
    rois = ['pton1_TP', 'nyu_ifg', 'pton1_mSTG', 'pton1_aSTG']
    #rois = ['nyu_ifg', 'pton1_mSTG', 'pton1_aSTG']
    #word_values = ['all-', 'top1-correct-', 'top5-incorrect-']
    #word_values = ['all-', 'correct-', 'top5-incorrect-']
    #word_values = ['correct-']#, 'top5-incorrect-']
    for roi in rois: 
        #ID = 'REVPAPER-' + roi + '-maxout-truortho'
        #ID = 'REVPAPER-' + roi + '-maxout-lin-reg'
        #ID = 'REVPAPER-' + roi + '-maxout-half-out'
        #ID = 'REVPAPER-' + roi + '-maxout-truortho-decorr'
        #ID = 'REVPAPER-' + roi + '-maxout-regress-out-take2'
        #ID = 'REVPAPER-static3-interpolation'
        #ID = 'REVPAPER-mixing-interpolation'
        #ID = 'REV2-ffw'
        ID = 'REV2-attn'
        #breakpoint()
        #wv = 'all-'
        #wv = 'correct-'
        for wv in word_values:
            sep_paper_plots(-500, 500, roi, 'gpt2-xl', wv, ID, False)
        #bar_csv(ID, word_values, [roi])
        
        # saving
        #csv_encoding(ID, word_values, [roi])
        #csv_lag_layer(ID, word_values, [roi])
    breakpoint() 
    '''
    #csv_individual_electrode_lag_layer(ID, word_values, rois) #individual electrode lag layer
    #csv_max_avg_lag_layer(ID, word_values, rois) # avg of max lags per electrode encoding
    #csv_ariel_latency_regression_lag_layer(ID, word_values, rois)

    #plot_single_electrode_laglayer(ID, word_values, rois) 
    #plot_single_electrode_encoding(ID, word_values, rois) 
    #breakpoint()
    csv_encoding(ID, word_values, rois)
    csv_lag_layer(ID, word_values, rois)

    #breakpoint()
    #word_value = 'correct-minusmax-'
    #word_value = 'top5-incorrect-minusmax-'
    
    #wvs = ['correct-orthomax-']#, 'top5-incorrect-minusmax-']
    #wvs = ['top5-incorrect-minusmax-']
    #wvs = ['correct-truortho-']#, 'top5-incorrect-minusmax-']
    #wvs = ['correct-enc2resid-']#, 'top5-incorrect-enc2resid-']
    #wvs = ['correct-enc2resid-']#, 'top5-incorrect-enc2resid-']
    #wvs = ['all-static-interpolation-', 'all-contextual-interpolation-']
    #wvs = ['all-static-interpolation-']#, 'all-contextual-interpolation-']
    #wvs = ['all-static2-interpolation-']#, 'all-contextual-interpolation-']
    #podcast2-gpt2-xl-pca50d-full-' + ID + '-' + word_value + 'gpt2-xl-hs'
    for roi in rois:
        print(roi)
        for wv in word_values:
            print(wv)
            sep_paper_plots(-500, 500, roi, 'gpt2-xl', wv, ID, False)
    
    breakpoint()
    rois = ['nyu_ifg', 'pton1_mSTG', 'pton1_TP', 'pton1_aSTG']
    #rois = ['pton1_TP']
    wvs = ['correct', 'top5-incorrect', 'incorrect']
    for roi in rois:
        for wv in wvs:
            permute_laglayer(roi, wv)
    breakpoint()
    
    # Bootstrapping Test
    rois = ['nyu_ifg', 'pton1_mSTG', 'pton1_TP', 'pton1_aSTG']
    wvs = ['correct', 'top5-incorrect']
    for roi in rois:
        for wv in wvs:
            bootstrapping_within_roi_test(roi, wv)

    breakpoint()
    '''
    # Permutation Test
    wvs = ['correct', 'top5-incorrect']
    rois = ['pton1_mSTG', 'pton1_TP', 'pton1_aSTG']
    #breakpoint()
    import itertools
    roi_sets = list(itertools.combinations(rois, 2))
    for i in range(5):
        for roi_set in roi_sets:
            for wv in wvs:
                monte_carlo_permutation(roi_set[0], roi_set[1], wv)
                monte_carlo_permutation(roi_set[1], roi_set[0], wv)

    breakpoint()
    #sep_paper_plots(-500, 500, 'ifg', 'gpt2-xl', 'all', False)
    #sep_electrode_plots(-500, 500, 'mSTG', [1, 24, 48])
    #plot_regression()
    #pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    #for i in range(1, 49):
    #    p1 = pdir + '777_full-correct-GIT-hs'+ str(i) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
    #    p2 = pdir + '777_full-correct-PAPER-hs' + str(i) + '_gpt2-xl_cnxt_1024_embeddings.pkl'
    #    print(compare_pickles(p1, p2))
    pdir = '/scratch/gpfs/eham/247-encoding-updated/'
    #rdp = pdir + 'RD_PAPER_correct.pkl'
    #rdg = pdir + 'RD_GIT_correct.pkl'
    #pdp = pdir + 'PCA_PAPER_correct.pkl'
    #pdg = pdir + 'PCA_GIT_correct.pkl'
    #print(compare_pickles(pdp, pdg))
    #roi_list = ['TP', 'ifg', 'mSTG', 'aSTG']
    #roi_list = ['222-ifg']
    #roi_list = ['nyu-ifg'] #, 'nyu-TP', 'nyu-rSTG', 'nyu-mSTG']
    #roi_list = ['222_ba45', '222_ba44']
    #word_values = ['all', 'allnw', 'alln2w']
    #word_values = ['correct', 'correctnw', 'correctn2w']
    #word_values = ['correct-', 'all-']
    #word_values = ['correct-', 'all-', 'incorrect-']
    #word_values = ['incorrectnw', 'incorrectn2w']
    #word_values = ['incorrect-top5-']
    #word_values = ['correct-top5-']
    #word_values = ['allnw', 'alln2w']
    #word_values = ['incorrectn2w']
    #word_values = ['correct', 'incorrect']
    #roi_list = ['pton1_ba45']
    #roi_list = ['pton1_ba44']
    #roi_list = ['pton1_ifg','pton1_ba44', 'pton1_ba45', 'pton1_mSTG', 'pton1_aSTG', 'pton1_TP']
    #roi_list = ['pton1_ifg', 'pton1_mSTG', 'pton1_aSTG', 'pton1_TP']
    #roi_list = ['pton1_aSTG', 'pton1_TP', 'pton1_mSTG']
    #roi_list = ['pton1_ba44', 'pton1_ba45']
    #roi_list = ['ifg']#, '222-ifg']
    #roi_list = ['pton1_ba45']
    #roi_list = ['pton1_ba44']
    #roi_list = ['nyu_ifg', 'nyu_ba44', 'nyu_ba45', 'nyu_rSTG', 'nyu_cSTG', 'nyu_mSTG']
    #roi_list = ['all160_sig']
    # all ifgs
    #roi_list = ['pton1_ifg', 'pton1_ba45', 'pton1_ba44', 'nyu_ba45', 'nyu_ifg', 'nyu_ba44']
    # all STGs
    #roi_list = ['pton1_mSTG', 'pton1_aSTG', 'pton1_TP', 'nyu_mSTG', 'nyu_cSTG', 'nyu_rSTG']
    #roi_list = ['pton1_ba44', 'pton1_ba45']
    #roi_list = ['pton1_ifg']
    # FINAL LISTS
    roi_list = ['nyu_ifg']
    #roi_list = ['pton1_mSTG', 'pton1_aSTG', 'pton1_TP', 'nyu_ifg']
    #roi_list = ['pton1_mSTG', 'pton1_aSTG', 'pton1_TP']
    #roi_list = ['pton1_mSTG']
    #roi_list = ['pton1_aSTG']
    #roi_list = ['pton1_TP']
    #word_values = ['all-']
    #word_values = ['incorrect-']
    #word_values = ['incorrect-',  'correct-', 'all-']
    #word_values = ['incorrect-',  'correct-']
    #word_values = ['top5-incorrect-',  'top5-correct-', 'all-']
    #word_values = ['top5-incorrect-',  'top5-correct-']
    #word_values = ['correct-']
    word_pfx = 'correct-phase-shuffled-'
    N = 1000
    num_layers = 3
    output = np.zeros((num_layers, N))
    model = 'gpt2-xl'
    ID ='PAPER2'
    elec_list = get_e_list(roi_list[0] + '_e_list.txt', '\t')
    for i in range(1, N+1):
        print(i, '/', N)
        word_value = word_pfx + str(i)
        fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + word_value + '-gpt2-xl-hs'
        ps = get_params(elec_list, model,roi_list[0], word_value)
        ps['layer_list'] = [5, 25, 45]
        ps['num_layers'] = num_layers
        all_sig_R = preprocess_encoding_results(fpath, elec_list, ps)
        # TODO: get/save max
        e_avg = np.mean(all_sig_R, axis=1) # average over electrodes
        layer_maxes = np.max(e_avg, axis=-1)
        output[:,i-1] = layer_maxes
    # TODO: get distribution 
    breakpoint()
    for j in range(num_layers):
        top1 = np.percentile(output[j,:], 99)
        print(ps['layer_list'][j], ': ', top1)

    breakpoint()
    # NOTE: CSV lag layer
    # make sure to edit sep_paper_plots to return after generating array
    #import csv
    #lshift = 25 # ms between time points
    #lags = np.arange(-2000, 2001, lshift)
    #breakpoint()
    #csv_o = open('PAPER2-top5-lag_layer_export.csv', 'w')
    #csv_o = open('test10.csv', 'w')
    #writer = csv.writer(csv_o)
    #header = ['roi', 'word value']
    #header.extend(list(range(1, 49)))
    #writer.writerow(header)
    #for roi in roi_list:
    #    print(roi)
    #    for wv in word_values:
    #        print(wv)
    #        bea = sep_paper_plots(-500, 500, roi, 'gpt2-xl', wv, True)
    #        breakpoint()
    #        mv = lags[np.argmax(bea, axis=1)]
    #        data = [str(roi),str(wv)]
    #        data.extend(mv.tolist())
    #        writer.writerow(data)
    #        #for val in mv: print(val)
    #csv_o.close()
    # NOTE: CSV encoding
    # make sure to edit sep_paper_plots to return after generating array
    #import csv
    #num_layers = 48
    #hues = list(np.arange(0, 240 + 240/(num_layers - 1), 240/(num_layers-1))/360) # red to blue
    #c_a = list(map(lambda x: clr.hsv_to_rgb([x, 1.0, 1.0]), hues))
    
    #lshift = 25 # ms between time points
    #lags = np.arange(-2000, 2001, lshift)
    #breakpoint()
    #csv_o = open('PAPER2-top1-enc_export.csv', 'w')
    #csv_o = open('PAPER2-top5-enc_export.csv', 'w')
    #writer = csv.writer(csv_o)
    #header = ['roi', 'word value','layer', 'R', 'G', 'B']
    #header.extend(lags.tolist())
    #writer.writerow(header)
    #for roi in roi_list:
    #    print(roi)
    #    for wv in word_values:
    #        print(wv)
    #        bea = sep_paper_plots(-500, 500, roi, 'gpt2-xl', wv, True)
    #        for layer in range(bea.shape[0]):
    #            #breakpoint()
    #            color = c_a[layer].tolist()
    #            data = [str(roi),str(wv), str(layer+1), color[0], color[1], color[2]]
    #            data.extend(bea[layer,:].tolist())
    #            writer.writerow(data)
    #csv_o.close()

    for roi in roi_list:
        print(roi)
        for wv in word_values:
            print(wv)
            bea = sep_paper_plots(-500, 500, roi, 'gpt2-xl', wv, True)


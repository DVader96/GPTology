import glob
import os
from copy import deepcopy

import matplotlib.colors as clr
from statsmodels.stats.multitest import fdrcorrection as fdr
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
seed = 42
rng = default_rng(seed)

import math
from brain_color_prepro import get_e_list, get_n_layers, save_e2l
import matplotlib.image as img
import pandas as pd
from scipy.stats import pearsonr


# TODO: this file is work in progress
def sigmoid(x):
    return 1/(1+math.exp(-x))

def quantify_nans(a):
    ts = 1
    for s in a.shape:
        ts*=s
    num_nan = np.count_nonzero(np.isnan(a))
    return num_nan/ts

def extract_correlations(directory_list):
    #breakpoint()
    all_corrs = []
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, '*.csv'))
        if len(file_list) > 1:
            breakpoint()
        for file in file_list:
            #if file[-16:] == '742_G64_comp.csv':
            #    continue
            with open(file, 'r') as csv_file:
                ha = list(map(float, csv_file.readline().strip().split(',')))
            all_corrs.append(ha)

    hat = np.stack(all_corrs)
    mean_corr = np.mean(hat, axis=0)
    return mean_corr

def extract_avg_sig_correlations(directory_list, e_list):
    all_corrs = []
    count = 0
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, '*.csv'))
        if len(file_list) > 1:
            breakpoint()
        for file in file_list:
            ve = file.split('/')[-1]
            es = len(ve.split('_'))
            ce = '_'.join(ve.split('_')[:es-1])
            
            if file[-16:] == '742_G64_comp.csv':
                continue
            elif ce not in e_list:
                continue
            count +=1
            
            #breakpoint()
            #with open(file, 'r') as csv_file:
            #    ha = list(map(float, csv_file.readline().strip().split(',')))
            e_sig = pd.read_csv(file)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(file, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            e_Ps = list(e_f.loc[1])
            e_Qs = fdr(e_Ps, alpha=0.01, method='i')[1] # use bh method
            for i in range(len(e_Rs)):
                if e_Qs[i] > 0.01:
                    #breakpoint()
                    e_Rs[i] = np.nan
            #breakpoint()
            all_corrs.append(e_Rs)
            # add electrode to dict with index
    if count == 0:
        breakpoint()
    #breakpoint()
    hat = np.stack(all_corrs)
    mean_corr = np.nanmean(hat, axis=0)

    return mean_corr 


def extract_single_correlation(directory_list, elec):
    count = 0
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, elec + '_comp.csv'))
        if len(file_list) > 1:
            breakpoint()
        for file in file_list:
            if file[-16:] == '742_G64_comp.csv':
                continue
            count +=1
            
            #breakpoint()
            # ha is a sanity check
            #with open(file, 'r') as csv_file:
            #    ha = list(map(float, csv_file.readline().strip().split(',')))
            e_sig = pd.read_csv(file)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(file, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            #all_corrs.append(ha)
            # add electrode to dict with index
    if count == 0:
        breakpoint()
    #hat = np.stack(all_corrs)
    return e_Rs


def extract_single_sig_correlation(directory_list, elec):
    count = 0
    #if elec == '717_LGA4':
    #    breakpoint()
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, elec + '_comp.csv'))
        if len(file_list) > 1:
            breakpoint()
        for file in file_list:
            if file[-16:] == '742_G64_comp.csv':
                continue
            count +=1
            
            #breakpoint()
            #with open(file, 'r') as csv_file:
            #    ha = list(map(float, csv_file.readline().strip().split(',')))
            e_sig = pd.read_csv(file)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(file, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            e_Ps = list(e_f.loc[1])
            e_Qs = fdr(e_Ps, alpha=0.01, method='i')[1] # use bh method
            #breakpoint()
            for i in range(len(e_Rs)):
                if e_Qs[i] > 0.01:
                    #breakpoint()
                    e_Rs[i] = np.nan
            #breakpoint()
            #all_corrs.append(ha)
            # add electrode to dict with index
    if count == 0:
        breakpoint()
    #hat = np.stack(all_corrs)
    return e_Rs#, test_eRs

def extract_electrode_set(fpath, e_list, sig):
    r_list = []
    p_list = []
    for e in e_list:
        count = 0
        file_list = glob.glob(os.path.join(fpath, '777',str(e)+'_comp.csv'))
        if len(file_list) != 1:
            breakpoint()
        assert(len(file_list) == 1)    
        for file in file_list:
            ve = file.split('/')[-1]
            es = len(ve.split('_'))
            ce = '_'.join(ve.split('_')[:es-1])
            if ce not in e_list:
                continue
            count +=1
            
            e_sig = pd.read_csv(file)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(file, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            e_Ps = list(e_f.loc[1])
        r_list.append(e_Rs)
        p_list.append(e_Ps)
    if sig == True:
        p_list = np.array(p_list)
        r_list = np.array(r_list)
        #breakpoint()
        og_shape = p_list.shape
        ps = p_list.flatten('F')
        #test_ps = np.reshape(ps, og_shape, 'F')
        e_Qs = fdr(ps, alpha=0.01, method='i')[1] # use bh method
        e_Qs = np.reshape(e_Qs, og_shape, 'F')
        #rs = r_list.flatten('F')
        #breakpoint()
        r_list[np.nonzero(e_Qs > 0.01)] = np.nan
    else:
        r_list = np.array(r_list)
    if count == 0:
        breakpoint()
    breakpoint()
    #hat = np.stack(all_corrs)
    #mean_corr = np.nanmean(hat, axis=0)
    
    return np.nanmean(r_list, axis = 0)

def preprocess_encoding_results(fpath, e_list, ps):
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
                e_sig = pd.read_csv(file)
                sig_len = len(e_sig.loc[0])
                e_f = pd.read_csv(file, names = range(sig_len))
                e_Rs = list(e_f.loc[0])
                out[i,j,:] = e_Rs
    
    return out


# construct a dictionary for every electrode
# hold in each entry an array (layer x lag) of Q values
# also construct a dictionary with R values
def extract_single_bigbigfdr_correlation(fpath, e_list, layer_list, num_lags, sig):
    num_layers = len(layer_list)
    #eR_dict = {}
    #eP_dict = {}
    r_list = []
    p_list = []
    #eQ_dict = {}
    #breakpoint()
    #layer_list = [1]
    for e in e_list:
        print(e)
        ra = np.zeros((num_layers, num_lags))
        pa = np.zeros((num_layers, num_lags))
        for i, l in enumerate(layer_list):
            count = 0
            #if l == 34:
            #    breakpoint()

            #file_list = glob.glob(os.path.join(fpath, '777',str(e)+'_comp.csv'))
            file_list = glob.glob(os.path.join(fpath+str(l), '777',str(e)+'_comp.csv'))

            #breakpoint()
            if len(file_list) != 1:
                breakpoint()
            assert(len(file_list) == 1)    
            for file in file_list:
                ve = file.split('/')[-1]
                #breakpoint()
                es = len(ve.split('_'))
                ce = '_'.join(ve.split('_')[:es-1])
                #breakpoint()
                if ce not in e_list:
                    #breakpoint()
                    continue
                count +=1
            
                e_sig = pd.read_csv(file)
                sig_len = len(e_sig.loc[0])
                e_f = pd.read_csv(file, names = range(sig_len))
                e_Rs = list(e_f.loc[0])
                e_Ps = list(e_f.loc[1])
                #breakpoint()
                ra[i,:] = e_Rs
                pa[i,:] = e_Ps
                #p_list.extend(e_Ps)
        #eR_dict[e] = ra
        r_list.append(ra)
        #eP_dict[e] = pa
        p_list.append(pa)
    #breakpoint()
    p_list = np.array(p_list)
    r_list = np.array(r_list)

    if sig == True:
        #breakpoint()
        og_shape = p_list.shape
        ps = p_list.flatten('F')
        #test_ps = np.reshape(ps, og_shape, 'F')
        e_Qs = fdr(ps, alpha=0.01, method='i')[1] # use bh method
        e_Qs = np.reshape(e_Qs, og_shape, 'F')
        #rs = r_list.flatten('F')
        #breakpoint()
        r_list[np.nonzero(e_Qs > 0.01)] = np.nan
    if count == 0:
        breakpoint()
    #breakpoint()
    #hat = np.stack(all_corrs)
    #mean_corr = np.nanmean(hat, axis=0)

    return r_list

def get_brain_im(elec):
    patient = elec.split('_')[0]
    elecn = elec[len(patient) + 1:] # skip patient and underscore
    file_n = '/scratch/gpfs/eham/247-encoding-updated/data/images/' + patient + '_left_both' + patient + '_' + elecn + '_single.png'
    try: 
        im = img.imread(file_n)    
    except:
        print('no file for ' + elec)

    return im

def all_rest_true_plots(w_type, topn, emb, true, rest, full):
    fig, ax = plt.subplots()
    lags = np.arange(-2000, 2001, 25)
    ax.plot(lags, emb, 'k', label='contextual') #**
    ax.plot(lags, true, 'r', label='true')
    ax.plot(lags, rest, 'orange', label='not true')
    ax.plot(lags, full, 'b', label = 'all')
    ax.legend()
    ax.set(xlabel='lag (s)', ylabel='correlation', title=w_type + ' top' + str(topn))
    ax.grid()

    fig.savefig("comparison_new_" + w_type + "_top" + str(topn) + "_no_norm_pca.png")
    #fig.savefig("comparison_old_p_weight_test.png")
    #plt.show()
    

def get_signals(topn, w_type):
    emb_dir = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-emb/*')
    emb = extract_correlations(emb_dir)

    true_dir = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-' + w_type + '-true-top' + str(topn) + '/*')
    true = extract_correlations(true_dir)
    
    rest_dir = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-' + w_type + '-rest-top' + str(topn) + '/*')
    rest = extract_correlations(rest_dir)
     
    full_dir = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-' + w_type + '-all-top' + str(topn) + '/*')
    full = extract_correlations(full_dir)
  
    return emb, true, rest, full
# 161 lags
def plot_electrodes(num_layers, in_type, in2, elec_list, omit_e,num_lags, ph):
    avg_list = []
    init_grey = 0
    #breakpoint()
    #lags = np.arange(-2000, 2001, 25)
    lags = np.arange(-5000, 5001, 25)
    num_lags = len(lags)
    for i, e in enumerate(elec_list):
        if e in omit_e:
            continue
        #print(e)
        fig, ax = plt.subplots()
        big_e_array = np.zeros((num_layers, num_lags))
        # note layers go from 1 to num_layers. 0 is input
        for layer in range(1,num_layers+1):
            if ph:
                ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '*phase-shuffle*/*')
                #breakpoint()
            else:
                ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '*5000/*')
            #breakpoint()
            big_e_array[layer-1] = extract_single_correlation(ldir, e)
        
        #init_grey/=(1+1e-2) 
        #breakpoint() 
        
        e_layer = np.argmax(big_e_array, axis=0)
        avg_list.append(e_layer)
        #print(i, init_grey)
        ax.plot(lags, e_layer, color = str(init_grey), label=e) 
    
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax.set(xlabel='lag (s)', ylabel='layer', title= in2 + ' top layer per lag for ' + e)
        ax.grid()
        if ph:
            fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Top_Layer_Per_Lag_' + in2 + '_' + e + 'phase_shuffle.png')
        else:
            fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Top_Layer_Per_Lag_' + in2 + '_' + e + '_5000.png')
        plt.close()
    
    # get baseline
    #ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-*' + in2[:-3] + '*phase_shuffle*/*')
    #breakpoint()
    #shuf = extract_correlations(ldir)
    #breakpoint()
    fig, ax = plt.subplots()
    breakpoint() # I am not sure this stde is correct
    layer_mean = np.mean(avg_list, axis = 0)
    layer_stde = np.std(avg_list, axis=0)/(np.sqrt(len(layer_mean)))
    ax.plot(lags, layer_mean, '-o', color = 'orange', label='avg')
    #ax.plot(lags, shuf, color = 'k', label='phase shuffled top lag 0 layer')
    ax.fill_between(lags, layer_mean - layer_stde, layer_mean + layer_stde,color='orange', alpha=0.2)
    ax.set_ylim([1, num_layers])
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set(xlabel='lag (s)', ylabel='layer', title= in2 + ' top layer per lag for ' + e)
    ax.grid()
    if ph:
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test_Top_Layer_Per_Lag_Avg' + in2 + '_' + e + 'phase_shuffle.png')
    else:
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test_Top_Layer_Per_Lag_Avg' + in2 + '_' + e + '_5000.png')
    plt.close()

def plot_sig_lags_for_layers(num_layers, in_type, in2, elec_list, omit_e,lags, ph, cutoff_R=0.15):
    avg_list = []
    avg_R = []
    #no_scale_list = []
    init_grey = 0
    num_lags = len(lags)
    for i, e in enumerate(elec_list):
        if e in omit_e:
            continue
        print(e)
        big_e_array = np.zeros((num_layers, num_lags))
        for layer in range(1, num_layers+1):
            ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '/*')
                       
            big_e_array[layer-1] = extract_single_sig_correlation(ldir, e)
        
        #no_scale_top_lag = []
        e_layer = []
        e_layer_corr = []
        for i in range(big_e_array.shape[0]):
            try:
                #weights = big_e_array[:,i]/np.nansum(big_e_array[:,i])
                #e_layer.append(np.nansum(weights*np.arange(1, num_layers+1)))
                # get lag that maximizes correlation for a given layer
                #breakpoint()
                max_R = np.nanmax(big_e_array[i,:])
                # ignore 
                if max_R < cutoff_R:
                    e_layer.append(np.nan)
                    e_layer_corr.append(np.nan)
                    continue
                #no_scale_top_lag.append(np.nanargmax(big_e_array[i,:]))
                e_layer.append(25*(np.nanargmax(big_e_array[i,:])-len(big_e_array[i,:])//2))
                e_layer_corr.append(max_R)
            except ValueError:
                e_layer.append(np.nan) # append nan if all in column are nan
                e_layer_corr.append(np.nan)
        #no_scale_list.append(no_scale_top_lag)
        avg_list.append(e_layer)
        avg_R.append(e_layer_corr)
    breakpoint() 
    fig, ax = plt.subplots(figsize=[20,10])
    ela = np.array(avg_list)
    #no_scale_stack = np.array(no_scale_list)
    #no_scale_avg = np.mean(no_scale_stack, axis =0)
    #for i in range(no_scale_avg.shape[0]):
    #print(i+1, no_scale_avg[i])
    layer_mean = np.nanmean(ela, axis=0)
    layer_std = np.nanstd(ela, axis=0)
    sqrt_size = np.sqrt(np.count_nonzero(~np.isnan(ela), axis=0))
    breakpoint() # check this
    layer_stde = layer_std/sqrt_size
    
    ela_R = np.array(avg_R)
    layer_corr_mean = np.nanmean(ela_R, axis=0)
    layer_corr_std = np.nanstd(ela_R, axis=0)
    R_sqrt_size = np.sqrt(np.count_nonzero(~np.isnan(ela_R), axis=0))
    breakpoint() # check this
    layer_corr_stde = layer_corr_std/R_sqrt_size

    ax.plot(np.arange(1, num_layers+1), layer_mean, '-o', markersize=2,color = 'orange', label='layer avg')
    ax.fill_between(np.arange(1,num_layers+1), layer_mean - layer_stde, layer_mean + layer_stde,color='orange', alpha=0.2)
    ax.set_ylim([-100, 500])
    #ax.secondary_yaxis('right', 
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' top lag per layer')
    ax.grid()
    ax2 = ax.twinx()
    ax2.set_ylabel('R', color='blue')
    ax2.plot(np.arange(1,num_layers+1), layer_corr_mean, markersize=2, color='blue', label='R avg')
    ax2.fill_between(np.arange(1,num_layers+1), layer_corr_mean - layer_corr_stde, layer_corr_mean + layer_corr_stde,color='blue', alpha=0.2)
    ax2.legend()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test_Top_Lag_Per_Layer' + in2 + '_' + e + '_' + str(cutoff_R) + '_SIG.png')
    plt.close()
 
def plot_heatmap_sig_lags_for_layers(num_layers, in_type, in2, elec_list, omit_e,lags, ph):
    #avg_list = []
    e_vals = []
    #avg_R = []
    init_grey = 0
    num_lags = len(lags)
    for i, e in enumerate(elec_list):
        if e in omit_e:
            continue
        print(e)
        big_e_array = np.zeros((num_layers, num_lags))
        for layer in range(1, num_layers+1):
            ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '/*')
            
            big_e_array[layer-1] = extract_single_sig_correlation(ldir, e)
        e_vals.append(big_e_array)    
        #e_layer_corr = []
        #for i in range(big_e_array.shape[0]):
        #    try:
        #        #weights = big_e_array[:,i]/np.nansum(big_e_array[:,i])
        #        #e_layer.append(np.nansum(weights*np.arange(1, num_layers+1)))
        #        # get lag that maximizes correlation for a given layer
        #        #breakpoint()
        #        e_layer.append(25*(np.nanargmax(big_e_array[i,:])-len(big_e_array[i,:])//2))
        #        e_layer_corr.append(np.nanmax(big_e_array[i,:]))
        #    except ValueError:
        #        e_layer.append(np.nan) # append nan if all in column are nan
        #        e_layer_corr.append(np.nan)
        #avg_list.append(e_layer)
        #avg_R.append(e_layer_corr)

    breakpoint() 
    fig, ax = plt.subplots(figsize=[20,10])
    #ela = np.array(avg_list)
    ela = np.nanmean(e_vals, axis = 0)
    ela = np.transpose(ela)
    #layer_mean = np.nanmean(ela, axis=0)
    #layer_std = np.nanstd(ela, axis=0)
    #sqrt_size = np.sqrt(np.count_nonzero(~np.isnan(ela), axis=0))
    #layer_stde = layer_std/sqrt_size
    
    #ela_R = np.array(avg_R)
    #layer_corr_mean = np.nanmean(ela_R, axis=0)
    #layer_corr_std = np.nanstd(ela_R, axis=0)
    #R_sqrt_size = np.sqrt(np.count_nonzero(~np.isnan(ela_R), axis=0))
    #layer_corr_stde = layer_corr_std/R_sqrt_size
    im = ax.imshow(ela, origin='lower', cmap='Greys_r')
    #ax.set_yticks(np.arange(76, 101, 4)) # -100 to 500
    #breakpoint()
    ax.set_yticks(np.arange(40, 121, 4))# -1000 to 1000 (-1000/25 + 80 --> 40)
    #ax.set_xticks(np.arange(1, 49, 1))
    #ax.set_yticklabels(np.arange(-100, 501, 100)) #-100 to 500
    ax.set_yticklabels(np.arange(-1000, 1001, 100))
    #ax.set_xticklabels(np.arange(1, 49, 1))
    #ax.plot(np.arange(1, num_layers+1), layer_mean, '-o', markersize=2,color = 'orange', label='layer avg')
    #ax.fill_between(np.arange(1,num_layers+1), layer_mean - layer_stde, layer_mean + layer_stde,color='orange', alpha=0.2)
    #ax.set_ylim([75, 101]) # -100 to 500
    ax.set_ylim([39, 121])
    #ax.secondary_yaxis('right', 
    #ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set(xlabel='layer', ylabel='lag (ms)', title= in2)
    ax.grid()
    #ax2 = ax.twinx()
    #ax2.set_ylabel('R', color='blue')
    #ax2.plot(np.arange(1,num_layers+1), layer_corr_mean, markersize=2, color='blue', label='R avg')
    #ax2.fill_between(np.arange(1,num_layers+1), layer_corr_mean - layer_corr_stde, layer_corr_mean + layer_corr_stde,color='blue', alpha=0.2)
    #ax2.legend()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/heatmap_bw_test_Lag_Per_Layer' + in2 + '_' + e + '_SIG.png')
    plt.close()
 
def plot_sig_electrodes(num_layers, in_type, in2, elec_list, omit_e,lags, ph):
    avg_list = []
    #test_list = []
    init_grey = 0
    #breakpoint()
    #lags = np.arange(-2000, 2001, 25)
    #lags = np.arange(-5000, 5001, 25)
    num_lags = len(lags)
    for i, e in enumerate(elec_list):
        if e in omit_e:
            #print('b
            continue
        #print(e)
        #fig, ax = plt.subplots()
        big_e_array = np.zeros((num_layers, num_lags))
        #test = np.zeros((num_layers, num_lags))
        for layer in range(1, num_layers+1):
            if ph:
                #breakpoint()
                ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '-phase-shuffle*/*')
            else:
                ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '/*')
            # R values per layer per lag for a given electrode
            
            #breakpoint()
            big_e_array[layer-1] = extract_single_sig_correlation(ldir, e)
        #init_grey/=(1+1e-2) 
        #breakpoint() 
        # top R values for each lag, recorded as the layer for which they occured
        #e_layer = np.argmax(big_e_array, axis=0)
        # need to handle each column separatetly for all nan cases
        e_layer = []
        for i in range(big_e_array.shape[1]):
            #breakpoint()
            try:
                # big_e_array is the R value for each electrode
                # analysis 1: take max
                #e_layer.append(np.nanargmax(big_e_array[:,i]))
                # analysis 2: take weighted average of layers, with R as weights. 
                #breakpoint()
                # normalize the weights first
                #weights = (big_e_array[:,i] - np.mean(big_e_array[:,i]))/np.std(big_e_array[:,i])
                #try:
                
                weights = big_e_array[:,i]/np.nansum(big_e_array[:,i])
                #except RuntimeWarning:
                
                #breakpoint()     
                e_layer.append(np.nansum(weights*np.arange(1, num_layers+1)))
                #print(np.nansum(big_e_array[:,i]*np.arange(1, num_layers+1)), np.nanargmax(big_e_array[:,i]))
            except ValueError:
                #breakpoint()
                e_layer.append(np.nan) # append nan if all in column are nan
        # list of top R values for each lag for each electrode
        avg_list.append(e_layer)
        #test_list.append(np.argmax(test, axis=0))
        #print(i, init_grey)
        #ax.plot(lags, e_layer, color = str(init_grey), label=e) 
    
        #ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        #ax.set(xlabel='lag (s)', ylabel='layer', title= in2 + ' top layer per lag for ' + e)
        #ax.grid()
        #if ph:
        #    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Top_Layer_Per_Lag_' + in2 + '_' + e + 'phase_shuffle.png')
        #else:
        #    fig.savefig('/scratch/pfs/eham/247-encoding-updated/results/figures/Top_Layer_Per_Lag_' + in2 + '_' + e + '_5000.png')
        #plt.close()
    
    # get baseline
    #ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-*' + in2[:-3] + '*phase_shuffle*/*')
    #breakpoint()
    #shuf = extract_correlations(ldir)
    #breakpoint()
    fig, ax = plt.subplots(figsize=[20,10])
    #layer_mean = np.mean(avg_list, axis = 0)
    #layer_stde = np.std(avg_list, axis=0)/(np.sqrt(len(layer_mean)))
    #test_ela = np.array(test_list)
    #layer_sum = np.sum(test_ela, axis = 0)
    #non_zero = np.count_nonzero(test_ela, axis = 0)
    #layer_u_test = layer_sum/non_zero
    
    #breakpoint()
    ela = np.array(avg_list)
    layer_mean = np.nanmean(ela, axis=0)
    layer_std = np.nanstd(ela, axis=0)
    sqrt_size = np.sqrt(np.count_nonzero(~np.isnan(ela), axis=0))
    breakpoint() # I am not sure this stde is correct
    layer_stde = layer_std/sqrt_size
    ax.plot(lags, layer_mean, '-o', markersize=2,color = 'orange', label='avg')
    #ax.plot(lags, shuf, color = 'k', label='phase shuffled top lag 0 layer')
    ax.fill_between(lags, layer_mean - layer_stde, layer_mean + layer_stde,color='orange', alpha=0.2)
    ax.set_ylim([1, num_layers+1])
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set(xlabel='lag (s)', ylabel='layer', title= in2 + ' top layer per lag for ' + e)
    ax.grid()
    if ph:
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test_Top_Layer_Per_Lag_Avg' + in2 + '_' + e + '_new_wavg_SIG_phase_shuffle.png')
    else:
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test_Top_Layer_Per_Lag_Avg' + in2 + '_' + e + '_new_wavg_SIG.png')
    plt.close()

# creates single encoding plot with all layers for a given model (greyscale and red plot)
def layered_encoding_plot(num_layers, in_type,elec, e_list, max_only):
    #breakpoint()
    if elec == '742_G64':
        print('bad electrode, stop')
        return
    fig, ax = plt.subplots(figsize=[20, 10])
    lags = np.arange(-2000, 2001, 25)
    init_grey = 1 
    lag_avg = [] 
    linestyle = (0, (1, 10**5))
    # adding 0 would add input. i at num_layers is contextual embedding (at output of model)
    for i in range(1,num_layers+1):
        #breakpoint()
        ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-' + in_type[:-3] + '-hs' + str(i) + '/*')
        if elec == 'all':
            layer = extract_correlations(ldir)
        elif elec == 'all_sig':
            layer = extract_avg_sig_correlations(ldir, e_list)
        else: 
            layer = extract_single_correlation(ldir, elec)
        init_grey -= 1/(math.exp(i*0.001)*(num_layers+1))
        #init_grey /= 1.25
        if max_only == True:
            max_idx = np.argmax(layer)
            print(i, max_idx)
            layer = [layer[i] if i == max_idx else 0 for i in range(len(layer))]
        if i == 0:
            ax.plot(lags, layer, '-o', color='b',linestyle=linestyle, label='layer' + str(i)) #**
        elif i == num_layers:
            ax.plot(lags, layer, '-o', color='r', linestyle=linestyle, label='layer' + str(i)) #**
        else:
            ax.plot(lags, layer, '-o',color=str(init_grey), linestyle=linestyle,label='layer' + str(i)) #**
        
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set(xlabel='lag (s)', ylabel='correlation', title= in_type + ' Encoding Over Layers ' + elec)
    ax.set_xlim([-300, 300])
    ax.grid()
    fig.savefig("/scratch/gpfs/eham/247-encoding-updated/results/figures/comparison_new_" + in_type +'_' + elec +  str(num_layers) + "where_shift.png")
    plt.close()
#cutoff is R if plot_type is 'max'. it's % if med
# 4 plots: median, encoding, brain, heatmap
def layered_encoding_plus_sig_lags_plot(elec_list, omit_e,lags, num_layers, in2, plot_type,cutoff_P, cutoff_R, slag, elag):
    num_lags = len(lags)
    start = math.floor(slag/25) + num_lags//2 # reverse of below 161//2 = 80. 
    end = math.floor(elag/25) + num_lags//2
    for i, e in enumerate(elec_list):
        if e in omit_e:
            #print('bad electrode found')
            continue
 
        print(e)
        big_e_array = np.zeros((num_layers, num_lags))
        for layer in range(1, num_layers+1):
            ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '/*')
                        
            big_e_array[layer-1] = extract_single_sig_correlation(ldir, e)
        
        e_layer = []
        e_layer_R = []
        if plot_type == 'all_p':
            e_layer_min = []
            e_layer_med = []
            e_layer_max = []
            e_layer_max_lag = []
        #topk = math.ceil(cutoff_P*(num_lags)) 
        #topkminR = []
        topk_size = []
        for k in range(big_e_array.shape[0]):
            try:
                if plot_type == 'max':
                    #weights = big_e_array[:,i]/np.nansum(big_e_array[:,i])
                    #e_layer.append(np.nansum(weights*np.arange(1, num_layers+1)))
                    # get lag that maximizes correlation for a given layer
                    #breakpoint()
                    #no_scale_top_lag.append(np.nanargmax(big_e_array[i,:]))
                    max_R = np.nanmax(big_e_array[k,:])

                    # ignore 
                    if max_R < cutoff_R:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        continue

                    e_layer.append(25*(np.nanargmax(big_e_array[k,:])-len(big_e_array[k,:])//2))
                    e_layer_R.append(max_R)
                elif plot_type == 'med_p':
                    #breakpoint()
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0.15] # in general
                    #if len(top_lags_nonan) < 32:
                    #    breakpoint()
                    topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                    topk_lags = thresholded_lags[-topk:]
                    #breakpoint()
                    topk_size.append(topk)
                    #if big_e_array.shape[1] != 161:
                    #    print(big_e_array.shape[1])
                    #if cutoff_P*big_e_array.shape[1] > len(top_lags_nonan):
                    #    print(topk)
                    # Iif topk_lags is empty (all nan) then just put in nan.
                    if len(topk_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        continue
                    
                    #breakpoint()
                    top_lags_med = np.sort(topk_lags)[len(topk_lags)//2] # we don't want true median which can give fractional values)
                    #minR = big_e_array[top_lags_nonan[0]] # minimum correlation of top lags 
                    #topkminR.append(minR)
                    # top_avg_R = np.mean(big_e_array[top_lags_nonan])
                    # top_std_R = np.std(big_e_array[top_lags_nonan])
                    # top_stde = top_stde_R/np.sqrt(len(top_lags_nonan)) 
                    #breakpoint()
                    e_layer.append(25*(top_lags_med-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                    e_layer_R.append(big_e_array[k, top_lags_med])
                elif plot_type == 'med_r':
                    #breakpoint()
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > cutoff_R]
                    topk_size.append(len(thresholded_lags))
                    if len(thresholded_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        continue
                    
                    #breakpoint()
                    top_lags_med = np.sort(thresholded_lags)[len(thresholded_lags)//2] # we don't want true median which can give fractional values)
                    #print(big_e_array[k,top_lags_med])
                    #minR = big_e_array[top_lags_nonan[0]] # minimum correlation of top lags 
                    #topkminR.append(minR)
                    # top_avg_R = np.mean(big_e_array[top_lags_nonan])
                    # top_std_R = np.std(big_e_array[top_lags_nonan])
                    # top_stde = top_stde_R/np.sqrt(len(top_lags_nonan)) 
                    e_layer.append(25*(top_lags_med-len(big_e_array[k,:])//2))
                    e_layer_R.append(big_e_array[k, top_lags_med])
                elif plot_type == 'med_pr':
                    # here e_layer is the p threshold (percent) while e_layer_R is the R threshold. 
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    
                    #top %
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0.15] # in general
                    topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                    topk_lags = thresholded_lags[-topk:]
                    #topk_size.append(topk)
                    if len(topk_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        continue
                    
                    top_lags_med = np.sort(topk_lags)[len(topk_lags)//2] # we don't want true median which can give fractional values)
                    e_layer.append(25*(top_lags_med-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                    
                    # R threshold 
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > cutoff_R]
                    #topk_size.append(len(thresholded_lags))
                    if len(thresholded_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        continue
                    
                    top_lags_med = np.sort(thresholded_lags)[len(thresholded_lags)//2] # we don't want true median which can give fractional values)
                    e_layer_R.append(25*(top_lags_med-len(big_e_array[k,:])//2))
                elif plot_type == 'min_p':
                    #breakpoint()
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    
                    #top %
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0.15] # in general
                    topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                    topk_lags = np.sort(thresholded_lags[-topk:])
                    #topk_size.append(topk)
                    if len(topk_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        continue
                    
                    top_lags_min = topk_lags[0] 
                    e_layer.append(25*(top_lags_min-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                    e_layer_R.append(big_e_array[k, top_lags_min])

                elif plot_type == 'all_p':
                    # min
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    
                    #top %
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0.1] # in general
                    #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0.0] # in general
                    topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                    topk_lags = np.sort(thresholded_lags[-topk:])
                    topk_size.append(topk)
                    #topk_size.append(topk)
                    if len(topk_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        e_layer_min.append(np.nan)
                        e_layer_med.append(np.nan)
                        e_layer_max.append(np.nan)
                        continue
                    
                    #breakpoint()
                    top_lags_min = topk_lags[0] 
                    e_layer_min.append(25*(top_lags_min-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                    #e_layer_R.append(big_e_array[k, top_lags_min])
                    # median
                    top_lags_med = topk_lags[len(topk_lags)//2] # we don't want true median which can give fractional values) already sorted here
                    e_layer_med.append(25*(top_lags_med-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                    e_layer_R.append(big_e_array[k, top_lags_med])
                    #max
                    #breakpoint()
                    topk_vals = [big_e_array[k, p] if p in topk_lags else 0 for p in range(len(big_e_array[k,:]))]
                    top_lags_max = np.nanargmax(topk_vals)
                    assert(top_lags_max in topk_lags)
                    
                    #print(top_lags_max, np.nanargmax(big_e_array[k,:])) 
                    #breakpoint()
                    e_layer_max.append(25*(top_lags_max-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                    max_lag = topk_lags[-1]
                    e_layer_max_lag.append(25*(max_lag-len(big_e_array[k,:])//2))
            except ValueError:
                e_layer.append(np.nan) # append nan if all in column are nan
                e_layer_R.append(np.nan)
                topk_size.append(np.nan)  
        #breakpoint() 
        # plot max lag over layers for this electrode
        fig, (ax, ax2, ax3, ax4) = plt.subplots(1,4,figsize=[50,20])
        if plot_type != 'all_p':
            ax.plot(np.arange(1, len(e_layer) + 1), e_layer, '-o', markersize=2,color = 'orange', label='top lag')
        else:
            ax.plot(np.arange(1, len(e_layer_min) + 1), e_layer_min, '-o', markersize=2,color = 'orange', label='min lag')
            ax.plot(np.arange(1, len(e_layer_med) + 1), e_layer_med, '-o', markersize=2,color = 'blue', label='med lag')
            ax.plot(np.arange(1, len(e_layer_max) + 1), e_layer_max, '-o', markersize=2,color = 'red', label='max lag')
            ax.plot(np.arange(1, len(e_layer_max_lag) + 1), e_layer_max_lag, '-o', markersize=2, color='green', label='lag with max R')

        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax.set_ylim([slag, elag])
        # for topk_size --> have one value for each layer (number of lags taken). average this, get for the electrode
        #breakpoint()
        ax.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' top lag per layer for electrode ' + e + 'avg # above threshold ' + str(np.mean(topk_size)))
        #ax.set_ylim([-500, 1000])
        #if plot_type[0:3] != 'med':
        #    ax2p5 = ax.twinx()
        #    ax2p5.plot(np.arange(1, len(e_layer_R) + 1), e_layer_R, '-o', markersize=2, color = 'k', label='top lag R')
        #    ax2p5.set_ylabel('R', color='blue')
        #    ax2p5.legend()
        #    ax2p5.tick_params(axis='both', labelsize=16)
        #    ax2p5.xaxis.label.set_fontsize(20)
        #    ax2p5.yaxis.label.set_fontsize(20)
        #else:
        #    ax2p5 = ax.twinx()
        #
        #    #ax2p5.plot(np.arange(1, len(topk_size) + 1), topk_size, '-o', markersize=2, color = 'b', label='Number Significant')
        #
        #    ax2p5.plot(np.arange(1, len(e_layer_R) + 1), e_layer_R, '-o', markersize=2, color = 'k', label='Median Correlation')
        #    ax2p5.set_ylabel('R', color='blue')
        #    ax2p5.legend()
        #    ax2p5.tick_params(axis='both', labelsize=16)
        #    ax2p5.xaxis.label.set_fontsize(20)
        #    ax2p5.yaxis.label.set_fontsize(20)
        ax.tick_params(axis='both', labelsize=16)
        init_grey = 1
        # plot encoding for all layers for this electrode 
        for j in range(1, len(e_layer) + 1):
            if j == 0:
                clr = 'b'
                ax2.plot(lags, big_e_array[j-1], color=clr, label='layer' + str(j)) #**
            elif j == num_layers:
                clr = 'r'
                ax2.plot(lags, big_e_array[j-1], color=clr, label='layer' + str(j)) #**
            else:
                init_grey -= 1/(math.exp((j-1)*0.001)*(num_layers+1))
                clr = str(init_grey)
                ax2.plot(lags, big_e_array[j-1], color=clr,label='layer' + str(j)) #**
            try:
                ax2.scatter([lags[np.nanargmax(big_e_array[j-1])]], [np.nanmax(big_e_array[j-1])], color=clr) 
            except ValueError:
                continue
        ax2.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax2.set(xlabel='lag (s)', ylabel='correlation', title= in_type + ' Encoding Over Layers ' + e)
        ax2.set_ylim([0, 0.4])
        ax2.set_xlim([slag, elag])
        ax2.tick_params(axis='both', labelsize=16)
        #ax2.set_xlim([-300, 300])
        #breakpoint()
        brain_im = get_brain_im(e)
        ax3.imshow(brain_im)
        
        #breakpoint() 
        ela = np.transpose(big_e_array) 
        ela[np.where(ela < cutoff_R)] = np.nan
        im = ax4.imshow(ela, origin='lower', cmap='Greys_r')
        ax4.set_yticks(np.arange(40, 121, 4))# -1000 to 1000 (-1000/25 + 80 --> 40)
        ax4.set_yticklabels(np.arange(-1000, 1001, 100))
        #ax4.set_ylim([39, 121])
        ax4.set_ylim([start, end])
        #ax4.set_xlim([0, 48])
        cbar = ax4.figure.colorbar(im, ax=ax4)
        ax4.set(xlabel='layer', ylabel='lag (ms)', title= in2)
        ax4.tick_params(axis='both', labelsize=16)
        ax4.grid()
        labels_for_font = [ax.xaxis.label, ax.yaxis.label, ax2.xaxis.label,ax2.yaxis.label, ax4.xaxis.label, ax4.yaxis.label]
        for item in labels_for_font:
            item.set_fontsize(20)
       
        #ax.grid()
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/'+ plot_type + '_omit_encoding_plus_lagplayer_' + in2 + '_' + e + '_' + str(slag)+ '_to_' + str(elag) + '_SIG.png')
        plt.close()
        #return
# average of median plots generated by layered_encoding_plus_sig_lags_plot
def avg_lag_layer_plot_updated(elec_list, omit_e, lags,lshift,num_layers,in2,plot_type, cutoff_P, cutoff_R, slag, elag, brain_area, bbfdr, num_back = 0):
    if plot_type == 'all_p': 
        all_e_layer_min = []
        all_e_layer_med = []
        all_e_layer_max = []
        all_e_layer_max_lag = []
        all25 = []
        all75 = []
    else:
        all_e_layer = []
        all_e_layer_R = []
    num_lags = len(lags)
    half_lags = num_lags//2
    #start = math.floor(slag/lshift) + num_lags # reverse of below 161//2 = 80. 
    #end = math.floor(elag/lshift) + num_lags
    #breakpoint()
    start = math.floor(slag/lshift) +half_lags # reverse of below 161//2 = 80. 
    end = math.floor(elag/lshift) + half_lags

    #all_e_big_e_array = np.zeros((num_layers-1, num_lags)) # partial
    #layer_count = list(np.zeros(num_layers-1)) #partial
    if bbfdr == True:
        fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-pval-' + in2
        #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-ws200-non-overlapping-pval-' + in2
        #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] = '-pca50d-full-git-ws200-pval-' + in2 
        #breakpoint()
        all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, list(np.arange(1,num_layers+1)), len(lags), True)
    
    all_e_big_e_array = np.zeros((num_layers-num_back, num_lags))
    layer_count = list(np.zeros(num_layers-num_back))
    for i, e in enumerate(elec_list):
        if e in omit_e:
            print('bad electrode found')
            continue
 
        print(e)
        if not bbfdr:
            big_e_array = np.zeros((num_layers-num_back, num_lags))
            #big_e_array = np.zeros((num_layers-1, num_lags)) # partial
            # add ones to both start and end if starting from 1. remove first if starting from 0 (1 to 49 or 0 to 49 not inclusive of 49. 
            for layer in range(1+num_back, num_layers+1):
                #for layer in range(2, num_layers + 1): # partial
                #breakpoint()
                if num_back == 0:
                    ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-pval-' + in2  + str(layer) + '/*')
                    #ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-linear_interp_of_aug2021-ws200-pval-' + in2  + str(layer) + '/*')
                else:
                    ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-prev_regressed_out_of_hs' + str(layer) + '/*') # partial        
                #big_e_array[layer-2] = extract_single_sig_correlation(ldir, e) # partial
                # -1 for no first layer
                big_e_array[layer-(num_back)-1] = extract_single_sig_correlation(ldir, e)
        else:
            big_e_array = all_sig_R[i]
        #breakpoint()
        #all_e_big_e_array = np.nansum([big_e_array, all_e_big_e_array], axis = 0)
        #temp = deepcopy(big_e_array)
        #temp[big_e_array < 0.15] = 0
        #all_e_big_e_array = np.nansum([temp, all_e_big_e_array], axis = 0)
        e_layer = []
        e_layer_R = []
        if plot_type == 'all_p':
            e_layer_min = []
            e_layer_med = []
            e_layer_max = []
            e_layer_max_lag = []
            el25 = []
            el75 = []
        #topk = math.ceil(cutoff_P*(num_lags))[len(topk_lags)//2]
        #topkminR = []
        topk_size = []
        #breakpoint()
        for k in range(big_e_array.shape[0]):
            try:
                if plot_type == 'max':
                    #weights = big_e_array[:,i]/np.nansum(big_e_array[:,i])
                    #e_layer.append(np.nansum(weights*np.arange(1, num_layers+1)))
                    # get lag that maximizes correlation for a given layer
                    #breakpoint()
                    #no_scale_top_lag.append(np.nanargmax(big_e_array[i,:]))
                    max_R = np.nanmax(big_e_array[k,:])
                    # ignore 
                    if max_R < cutoff_R:
                        e_layer.append(np.nan)
                        ##e_layer_R.append(np.nan)
                        continue

                    e_layer.append(lshift*(np.nanargmax(big_e_array[k,:])-half_lags))
                    e_layer_R.append(max_R)
                elif plot_type == 'med_p':
                    #breakpoint()
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                   
                    #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0.15] # in general
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0.0] # in general
                    topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                    topk_lags = thresholded_lags[-topk:]
                    #breakpoint()
                    topk_size.append(topk)
                    #if big_e_array.shape[1] != 161:
                    #    print(big_e_array.shape[1])
                    #if cutoff_P*big_e_array.shape[1] > len(top_lags_nonan):
                    #    print(topk)
                    # Iif topk_lags is empty (all nan) then just put in nan.
                    if len(topk_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        continue
                    
                    top_lags_med =np.sort(topk_lags)[len(topk_lags)//2]
                    #minR = big_e_array[top_lags_nonan[0]] # minimum correlation of top lags 
                    #topkminR.append(minR)
                    # top_avg_R = np.mean(big_e_array[top_lags_nonan])
                    # top_std_R = np.std(big_e_array[top_lags_nonan])
                    # top_stde = top_stde_R/np.sqrt(len(top_lags_nonan)) 
                    e_layer.append(lshift*(top_lags_med-half_lags))
                    e_layer_R.append(big_e_array[k, top_lags_med])
                elif plot_type == 'med_r':
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > cutoff_R]
                    topk_size.append(len(thresholded_lags))
                    if len(thresholded_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        continue
                    
                    #breakpoint()
                    top_lags_med = np.sort(thresholded_lags)[len(thresholded_lags)//2] # we don't want true median which can give fractional values)
                    #print(big_e_array[k,top_lags_med])
                    #minR = big_e_array[top_lags_nonan[0]] # minimum correlation of top lags 
                    #topkminR.append(minR)
                    # top_avg_R = np.mean(big_e_array[top_lags_nonan])
                    # top_std_R = np.std(big_e_array[top_lags_nonan])
                    # top_stde = top_stde_R/np.sqrt(len(top_lags_nonan)) 
                    e_layer.append(lshift*(top_lags_med-half_lags))
                    #e_layer_R.append(big_e_array[k, top_lags_med])
                elif plot_type == 'min':
                    breakpoint()
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    
                    #top %
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0.15] # in general
                    topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                    topk_lags = np.sort(thresholded_lags[-topk:])
                    #topk_size.append(topk)
                    if len(topk_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        continue
                    
                    top_lags_med = topk_lags[0] 
                    e_layer.append(lshift*(top_lags_min-half_lags)) # no start/end here because want actual length
                    #e_layer_R.append(big_e_array[k, top_lags_min])
                elif plot_type == 'all_p':
                    #breakpoint()
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    #breakpoint()
                    #top %
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > cutoff_R] # in general
                    #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0] 
                    #breakpoint()
                    topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                    topk_lags = np.sort(thresholded_lags[-topk:])
                    #print(len(topk_lags), cutoff_P*big_e_array.shape[1])
                    topk_size.append(topk)
                    #topk_size.append(topk)
                    if len(topk_lags) == 0:
                        e_layer.append(np.nan)
                        e_layer_R.append(np.nan)
                        e_layer_max.append(np.nan)
                        e_layer_min.append(np.nan)
                        e_layer_med.append(np.nan)
                        e_layer_max_lag.append(np.nan)
                        el25.append(np.nan)
                        el75.append(np.nan)
                        continue
                    
                    all_e_big_e_array[k,:] = np.nansum([big_e_array[k,:], all_e_big_e_array[k,:]], axis = 0)
                    layer_count[k] +=1
                    #breakpoint()
                    top_lags_min = topk_lags[0]
                    e_layer_min.append(lshift*(top_lags_min-half_lags)) # no start/end here because want actual length
                    #e_layer_R.append(big_e_array[k, top_lags_min])
                    # median
                    top_lags_med = np.nanmedian(topk_lags) #topk_lags[len(topk_lags)//2] # we don't want true median which can give fractional values) already sorted here
                    e_layer_med.append(lshift*(top_lags_med-half_lags)) # no start/end here because want actual length
                    #e_layer_R.append(big_e_array[k, top_lags_med])
                    #max
                    #breakpoint()
                    topk_vals = [big_e_array[k, p] if p in topk_lags else 0 for p in range(len(big_e_array[k,:]))]
                    top_lags_max = np.nanargmax(topk_vals)
                    assert(top_lags_max in topk_lags)
                    #print(top_lags_max, np.nanargmax(big_e_array[k,:])) 
                    #breakpoint()
                    e_layer_max.append(lshift*(top_lags_max-half_lags)) # no start/end here because want actual length
                    max_lag = topk_lags[-1]
                    #breakpoint()
                    e_layer_max_lag.append(lshift*(max_lag-half_lags))
                    
                    #el25.append(25*(topk_lags[len(topk_lags)//4] - len(big_e_array[k,:])//2))
                    #el75.append(25*(topk_lags[3*(len(topk_lags)//4)] - len(big_e_array[k,:])//2))
                    el25.append(lshift*(np.percentile(topk_lags, 25) - half_lags))
                    el75.append(lshift*(np.percentile(topk_lags, 75) - half_lags))

            except ValueError:
                e_layer.append(np.nan) # append nan if all in column are nan
                #e_layer_R.append(np.nan)
                topk_size.append(np.nan)  
        #breakpoint()
        if plot_type != 'all_p':
            all_e_layer.append(e_layer)
            all_e_layer_R.append(e_layer_R)
        else:
            all_e_layer_min.append(e_layer_min)
            all_e_layer_med.append(e_layer_med)
            all_e_layer_max.append(e_layer_max)
            all_e_layer_max_lag.append(e_layer_max_lag)
            all25.append(el25)
            all75.append(el75)
    # plot max lag over layers for this electrode
    
    fig, (ax, ax4,ax2, ax3)= plt.subplots(1, 4, figsize=[30,10])
    if plot_type != 'all_p':
        e_layer_avg = np.nanmean(all_e_layer, axis=0)
        #breakpoint()
        #e_R_avg = np.nanmean(all_e_layer_R, axis=0)
        assert(len(e_layer_avg) == len(all_e_layer[0]))
    
    
        ax.plot(np.arange(len(e_layer_avg)), e_layer_avg, '-o', markersize=2,color = 'orange', label='top lag')
    else:
        #breakpoint()
        min_avg = np.nanmean(all_e_layer_min, axis=0)
        med_avg = np.nanmean(all_e_layer_med, axis=0)
        max_avg = np.nanmean(all_e_layer_max, axis=0)
        #breakpoint()
        max_lag_avg = np.nanmean(all_e_layer_max_lag, axis=0)
        all25_avg = np.nanmean(all25, axis = 0)
        all75_avg = np.nanmean(all75, axis=0)
        #breakpoint()
        #e_R_avg = np.nanmean(all_e_layer_R, axis=0)
        assert(len(min_avg) == len(med_avg) == len(max_avg) == len(all_e_layer_med[0]))
        #ax.plot(np.arange(1,len(min_avg)+1), min_avg, '-o', markersize=2,color = 'orange', label='min lag')
        #ax.plot(np.arange(1,len(all25_avg)+1), all25_avg, '-o', markersize=2, color = 'cyan', label='25 percentile lag')
        ax.plot(np.arange(1,len(med_avg)+1), med_avg, '-o', markersize=2,color = 'blue', label='med lag')
        #ax.plot(np.arange(1,len(all75_avg)+1), all75_avg, '-o', markersize=2, color = 'purple', label='75 percentile lag')
        ax.plot(np.arange(1,len(max_avg)+1), max_avg, '-o', markersize=2,color = 'red', label='lag with top R')
        #ax.plot(np.arange(1,len(max_lag_avg)+1), max_lag_avg, '-o', markersize=2, color='green', label='max lag')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' Top Lag Per Layer Avg of Electrodes')
    ax.set_ylim([-500, 500])
    ax.tick_params(axis='both', labelsize=16)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.title.set_fontsize(16)

    # add 1s because not starting at 0
    #ax4.plot(np.arange(1,len(min_avg)+1), min_avg-np.mean(min_avg), '-o', markersize=2,color = 'orange', label='min lag')
    #ax4.plot(np.arange(1,len(all25_avg)+1), all25_avg-np.mean(all25_avg), '-o', markersize=2, color = 'cyan', label='25 percentile lag')
    ax4.plot(np.arange(1,len(med_avg)+1), med_avg-np.mean(med_avg), '-o', markersize=2,color = 'blue', label='med lag')
    #ax4.plot(np.arange(1,len(all75_avg)+1), all75_avg-np.mean(all75_avg), '-o', markersize=2, color = 'purple', label='75 percentile lag')
    ax4.plot(np.arange(1,len(max_avg)+1), max_avg-np.mean(max_avg), '-o', markersize=2,color = 'red', label='lag with top R')
    #ax4.plot(np.arange(1,len(max_lag_avg)+1), max_lag_avg-np.mean(max_lag_avg), '-o', markersize=2, color='green', label='max lag')
    ax4.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax4.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' Top Lag Layer Minus Mean')
    ax4.set_ylim([-200, 200])
    ax4.tick_params(axis='both', labelsize=16)
    ax4.xaxis.label.set_fontsize(16)
    ax4.yaxis.label.set_fontsize(16)
    ax4.title.set_fontsize(16)


    #ax2 = ax.twinx()
    #ax2.plot(np.arange(1, len(e_R_avg) + 1),e_R_avg, '-o', markersize=2, color = 'b', label='Correlation')
    #ax2.set_ylabel('R', color='blue')
    #a4x2.legend()
    #ax2.tick_params(axis='both', labelsize=16)
    #ax2.xaxis.label.set_fontsize(20)
    #ax2.yaxis.label.set_fontsize(20)
    #breakpoint()
    min_lag_corr = pearsonr(np.arange(len(min_avg)), min_avg)[0]
    #breakpoint()
    all25_lag_corr = pearsonr(np.arange(len(all25_avg)), all25_avg)[0]
    med_lag_corr = pearsonr(np.arange(len(med_avg)), med_avg)[0]
    all75_lag_corr = pearsonr(np.arange(len(all75_avg)), all75_avg)[0]
    #breakpoint()
    max_lag_corr = pearsonr(np.arange(len(max_lag_avg)), max_lag_avg)[0]
    corr_a = [min_lag_corr, all25_lag_corr, med_lag_corr, all75_lag_corr, max_lag_corr]
    
    #max_R_corr = pearsonr(np.arange(1, len(max_lag_avg) + 1), max_lag_avg)
    
    ax2.scatter(np.arange(len(corr_a)), corr_a, s = 100, c= ['orange', 'cyan', 'blue', 'purple', 'green'], zorder=1)
    ax2.plot(np.arange(len(corr_a)),corr_a, '-o', markersize=2,color = 'k', label='min lag', zorder = -1)
    #ax2.scatter([1],[min_lag_corr], color='orange')
    #ax2.scatter([2], [all25_lag_corr], color='cyan')
    #ax2.scatter([3], [med_lag_corr], color='blue')
    #ax2.scatter([4], [all75_lag_corr], color='purple')
    #ax2.scatter([5], [max_lag_corr], color='green')
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel('R')
    ax2.set_title(in2 +  'Correlations for Lines in Plot to Left')
    ax2.set_xlabel('Order')
    ax2.tick_params(axis='both', labelsize=16)
    ax2.xaxis.label.set_fontsize(16)
    ax2.yaxis.label.set_fontsize(16)
    ax2.title.set_fontsize(16)

    #breakpoint()
    # plot encoding
    #breakpoint()
    for k in range(num_layers-num_back):
        #for k in range(num_layers-1): # partial
        if layer_count[k] == 0:
            print('FAIL')
        all_e_big_e_array[k,:] /= layer_count[k]
    init_grey = 1
    # plot encoding for all layers for this electrode. 
    for j in range(num_layers - num_back):
        #for j in range(1, num_layers): #partial
        # input or first layer
        if j == 0:
            clr = 'b'
            ax3.plot(lags, all_e_big_e_array[j], color=clr, label='layer' + str(j+1),zorder=-1) #**
        elif j == num_layers-1:
            #elif j == num_layers - 1: #partial
            clr = 'r'
            ax3.plot(lags, all_e_big_e_array[j], color=clr, label='layer' + str(j+1), zorder=-1) #**
        else:
            init_grey -= 1/(math.exp((j)*0.001)*(num_layers+1))
            clr = str(init_grey)
            ax3.plot(lags, all_e_big_e_array[j], color=clr,label='layer' + str(j+1),zorder=-1) #**
        try:
            ax3.scatter([lags[np.nanargmax(all_e_big_e_array[j])]], [np.nanmax(all_e_big_e_array[j])], color=clr, zorder=-1)
        except ValueError:
            continue
    #breakpoint()
    ax3.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax3.set(xlabel='lag (s)', ylabel='R', title= in_type + ' Avg Electrode Encoding Over Layers')
    ax3.set_ylim([0, 0.4])
    #breakpoint()
    ax3.set_xlim([slag, elag])
    ax3.tick_params(axis='both', labelsize=16)
    #breakpoint()
    # construct array of key lags per layer. and their values
    min_scatter = min_avg/lshift + half_lags 
    min_scatter = list(map(math.floor, min_scatter))
    min_inds = lshift*(np.array(min_scatter)- half_lags)
    ax3.scatter(min_inds, list(all_e_big_e_array[np.arange(0, num_layers-num_back), min_scatter]), color='orange',s=20, zorder=1)
    #ax3.scatter(min_inds, list(all_e_big_e_array[np.arange(0, 47), min_scatter]), color='orange',s=20, zorder=1) # partial
    
    a25_scatter = all25_avg/lshift + half_lags
    a25_scatter = list(map(math.floor, a25_scatter))
    a25_inds = lshift*(np.array(a25_scatter)- half_lags)
    ax3.scatter(a25_inds, all_e_big_e_array[np.arange(0, num_layers-num_back), a25_scatter], color='cyan',s=20, zorder=1)
    #ax3.scatter(a25_inds, all_e_big_e_array[np.arange(0, 47), a25_scatter], color='cyan',s=20, zorder=1) # partial
    
    med_scatter = med_avg/lshift + half_lags
    med_scatter = list(map(math.floor, med_scatter))
    med_inds = lshift*(np.array(med_scatter)-half_lags)
    ax3.scatter(med_inds, all_e_big_e_array[np.arange(0, num_layers-num_back), med_scatter], color='blue',s=20, zorder=1)
    #ax3.scatter(med_inds, all_e_big_e_array[np.arange(0, 47), med_scatter], color='blue',s=20, zorder=1) # partial
    a75_scatter = all75_avg/lshift + half_lags
    a75_scatter = list(map(math.floor, a75_scatter))
    a75_inds = lshift*(np.array(a75_scatter)-half_lags)
    ax3.scatter(a75_inds, all_e_big_e_array[np.arange(0, num_layers-num_back), a75_scatter], color='purple',s=20, zorder=1)
    #ax3.scatter(a75_inds, all_e_big_e_array[np.arange(0, 47), a75_scatter], color='purple',s=20, zorder=1) #partial
    max_scatter = max_lag_avg/lshift + half_lags
    max_scatter = list(map(math.floor, max_scatter))
    max_inds = lshift*(np.array(max_scatter)-half_lags)
    ax3.scatter(max_inds, all_e_big_e_array[np.arange(0, num_layers-num_back), max_scatter], color='green',s=20, zorder=1)
    #ax3.scatter(max_inds, all_e_big_e_array[np.arange(0, 47), max_scatter], color='green',s=20, zorder=1) # partial
    ax3.xaxis.label.set_fontsize(16)
    ax3.yaxis.label.set_fontsize(16)
    ax3.title.set_fontsize(16)

    
    #ax3.set_xlim([-200,200])
    #if plot_type != 'med':
    #    ax2p5 = ax.twinx()
    #    ax2p5.plot(np.arange(1, len(e_layer_R) + 1), e_layer_R, '-o', markersize=2, color = 'b', label='top lag R')
    #    ax2p5.set_ylabel('R', color='blue')
    #    ax2p5.legend()
    #    ax2p5.tick_params(axis='both', labelsize=16)
    #    ax2p5.xaxis.label.set_fontsize(20)
    #    ax2p5.yaxis.label.set_fontsize(20)
    #else:
    #    ax2p5 = ax.twinx()
    #    ax2p5.plot(np.arange(1, len(topk_size) + 1), topk_size, '-o', markersize=2, color = 'b', label='Number Significant')
    #    ax2p5.set_ylabel('Count', color='blue')
    #    ax2p5.legend()
    #    ax2p5.tick_params(axis='both', labelsize=16)
    #    ax2p5.xaxis.label.set_fontsize(20)
    #    ax2p5.yaxis.label.set_fontsize(20)
    #fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/more_lag4_' + brain_area + '_top100_corr_threshold_ws200_partial_' + str(num_back) + '_back_avg_'+ plot_type + '_omit_encoding_plus_lagplayer_' + in2 + '_' + e + '_' + str(slag)+ '_to_' + str(elag) + '_SIG.png')
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/git_bbfdr_update_' + brain_area + '_top' + str(cutoff_P) + 'p_corr_threshold' + str(cutoff_R) + '_ws200_'+ plot_type + '_'+  in2 + '_' + str(slag)+ '_to_' + str(elag) + '_SIG.png')

    plt.close()
    return

def get_laglayer_R(el_med, el_max, weight_avg):
    #breakpoint()
    if  weight_avg:
        med_avg = el_med
        max_avg = el_max
    else:
        med_avg = el_med
        max_avg = el_max

        #med_avg = np.nanmean(el_med, axis=0)
        #max_avg = np.nanmean(el_max, axis=0)
    med_avg_nn = [med_avg[p] for p in range(len(med_avg)) if not np.isnan(med_avg[p])]
    if len(med_avg_nn) >= 2:
        med_lag_corr = pearsonr(np.arange(len(med_avg_nn)), med_avg_nn)[0]
    else:
        print('<2 vals')
        med_lag_corr = 0

    max_avg_nn = [max_avg[p] for p in range(len(max_avg)) if not np.isnan(max_avg[p])]
    if len(max_avg_nn) >= 2:
        max_lag_corr = pearsonr(np.arange(len(max_avg_nn)), max_avg_nn)[0]
    else:
        print('<2 vals')
        max_lag_corr = 0

    return np.round(med_lag_corr,2), np.round(max_lag_corr, 2)

def single_lag_layer(ax, el_med, el_max, norm, cutoff_R, in2, title, weight_avg, e_list):
    ts = ' Top Lag Per Layer \n(MedR, MaxR) = ('
    #breakpoint()
    # NOTE: where average
    if weight_avg:
        el_med = np.array(el_med)
        el_max = np.array(el_max)
        num_e = []
        for e in e_list:
            num_e.append(float(np.load(e + '_nume.npy')))
        med_avg = np.matmul(num_e, el_med)
        max_avg = np.matmul(num_e, el_max)
        #breakpoint()
        med_lag_corr, max_lag_corr = get_laglayer_R(med_avg, max_avg, weight_avg)
        #med_avg = np.nansum(el_med, axis = 0)
        #max_avg = np.nansum(el_max, axis = 0)
    
    else:
        med_lag_corr, max_lag_corr = get_laglayer_R(el_med, el_max, False)
        #med_avg = np.nanmean(el_med, axis=0)
        #max_avg = np.nanmean(el_max, axis=0)
        med_avg = el_med
        max_avg = el_max
        print ('max avg: ', max_avg)
        #avg_out = pd.DataFrame(max_avg)
        #avg_out.to_csv('max_avg.csv')
    if norm == True:
        med_avg = (med_avg-np.nanmean(med_avg))/np.nanstd(med_avg)
        max_avg = (max_avg-np.nanmean(max_avg))/np.nanstd(max_avg)
        ts = 'Z-Score' + ts 
    #breakpoint()
    ax.plot(np.arange(1,len(med_avg)+1), med_avg, '-o', markersize=2,color = 'blue', label='med lag for R > ' + str(cutoff_R))
    ax.plot(np.arange(1,len(max_avg)+1), max_avg, '-o', markersize=2,color = 'red', label='lag with top R')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=24)
    print('med R: ', str(med_lag_corr), ' max R: ', str(max_lag_corr))
    #ax.set(xlabel='layer', ylabel='lag (ms)', title= in2[:-3] + ts + str(med_lag_corr) + ', ' + str(max_lag_corr) + ')\n' + title)
    #ax.set_ylim([-250, 250])
    ax.tick_params(axis='both', labelsize=64)
    #ax.xaxis.label.set_fontsize(64)
    #ax.yaxis.label.set_fontsize(64)
    #ax.title.set_fontsize(64)

def basic_encoding_plot(file, title):
    data = extract_correlations([file])
    
    lags = np.arange(-2000, 2001, 25)
    plt.figure()
    plt.plot(lags, data, label='encoding',zorder=-1) #**
    plt.savefig(str(title) + '.png')
    plt.close()

# average set of electrodes with or withtou significance
def basic_encoding_electrode_set_plot(fpath, roi,title):
    e_list = get_e_list(roi + '_e_list.txt', '\t')
    data = extract_electrode_set(fpath, e_list, sig = False)
    lags = np.arange(-2000, 2001, 25)
    plt.figure()
    plt.plot(lags, data, label='encoding',zorder=-1) #**
    plt.savefig(str(title) + '.png')
    plt.close()

def single_lag_layer_enc(ax,num_layers, big_e_array, lags, xl,xr,norm):
    #max_lags = list(max_lags[0])
    #breakpoint()
    # handle all nan cases
    max_lags = np.zeros(big_e_array.shape[0]) + np.nan
    z = np.nanmax(big_e_array, axis = -1)
    max_lags[~np.isnan(z)] = lags[np.nanargmax(big_e_array[~np.isnan(z)], axis = -1)]
    max_lags = list(max_lags)
    #z = np.nanargmax(big_e_array, axis = -1)
    #max_lags = list(lags[np.nanargmax(big_e_array, axis = -1)])
    ld = {}
    h = []
    h0 = 1.01
    alpha = 0.001
    for l in max_lags:
        if l in ld:
            h.append(h0 + alpha*ld[l])
            ld[l] +=1
        else:
            ld[l] = 1
            h.append(h0)

    #breakpoint()
    #print(max_lags)
    hues = list(np.arange(0, 240 + 240/(num_layers - 1), 240/(num_layers-1))/360) # red to blue
    # plot encoding for all layers for this electrode. 
    ts = ' Avg Electrode Encoding Per Layer'
    if norm == True:
        ts = 'Norm\n' + ts
    
    print('hues:')
    c_out = []
    for j in range(num_layers):
        # input or first layer
        c = list(clr.hsv_to_rgb([hues[j], 1.0, 1.0]))
        c_out.append(c)
        #print(c) 
        if norm == True:
            ax.plot(lags, big_e_array[j]/np.nanmax(big_e_array[j]), color=c, label='layer' + str(j+1),zorder=-1) #**
            #breakpoint()
            #ax.scatter([lags[np.nanargmax(big_e_array[j])]], 1.01, color=c, zorder=-1)
            #assert(lags[np.nanargmax(big_e_array[j])] == max_lags[j])
            ax.scatter([max_lags[j]], h[j], color=c, zorder=-1)
            #ax.scatter([ags[np.nanargmax(big_e_array[j])]], h[j], color=c, zorder=-1)
        else:
            ax.plot(lags, big_e_array[j], color=c, label='layer' + str(j+1),zorder=-1) #**
            #ax.scatter([lags[np.nanargmax(big_e_array[j])]], [np.nanmax(big_e_array[j])], color=c, zorder=-1)
            #ax.scatter([lags[np.nanargmax(big_e_array[j])]], [h[j]], color=c, zorder=-1)
    if not norm:
        ax.set_ylim([0, 0.3])
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=10)
    #ax.set(xlabel='lag (ms)', ylabel='R', title=ts)
    #ax.vlines([-500, 500], 0, 0.4, colors = ['grey'], linestyles=['dashed'])
    ax.set_xlim([xl, xr])
    ax.tick_params(axis='both', labelsize=64)
    #ax.xaxis.label.set_fontsize(64)
    #ax.yaxis.label.set_fontsize(64)
    #ax.set_title(in_type[:-3] +ts, loc='right')
    #ax.title.set_fontsize(64)
    #c_out = pd.DataFrame(c_out)
    #c_out.to_csv('color_out.csv')
# option to run in sep_roi_stats
# given signals, compute derivative and subtract. 
# to compute derivative --> subtract adjacent point values and divide by 1, so just subtract. do for each set of adjacent points 
def derivative_plot(in1, in2, lags, lshift, num_layers, in_type, cutoff_P, cutoff_R, omit_e):
    weight_avg = False
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    omit_e = []
    rois = [in1, in2]
    roi_sigs = []
    for p, r in enumerate(rois):
        #print(r)
        elec_list = get_e_list(r + '_e_list.txt', '\t')
        _,_,_, enc_bea = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, False)
        enc_bea = np.mean(np.array_split(enc_bea, 12, axis = 0), axis = 0)
        roi_sigs.append(enc_bea)
    ds = []
    for mat in roi_sigs:
        # compute derivative for each signal, subtract. 
        #breakpoint()
        shift = np.roll(mat, shift=1,axis = 1)
        breakpoint()
        new_mat = (shift - mat)[:,1:-1]
        ds.append(new_mat)
    fig, ((ax1, ax2, diff), (ax3, ax4,ax5))= plt.subplots(2, 3, figsize=[40,30])
    
    
    hues = list(np.arange(0, 240 + 240/(mat.shape[0] - 1), 240/(mat.shape[0]-1))/360) # red to blue
    for j in range(ds[0].shape[0]):
        # input or first layer
        c = list(clr.hsv_to_rgb([hues[j], 1.0, 1.0]))
        ax1.plot(lags[1:-1], ds[0][j], color=c, label='layer' + str(j+1),zorder=-1) #**
        ax2.plot(lags[1:-1], ds[1][j], color=c, label='layer' + str(j+1),zorder=-1) #**
        diff.plot(lags[1:-1], ds[0][j] - ds[1][j], color=c, label='layer' + str(j+1),zorder=-1) #**
        
        breakpoint() 
        ax3.plot(lags[1:-1], roi_sigs[0][j][1:-1], color = c, label='layer' + str(j+1), zorder=-1)
        ax4.plot(lags[1:-1], roi_sigs[1][j][1:-1], color = c, label='layer' + str(j+1), zorder=-1)
    plot_settings(ax1, in1)
    plot_settings(ax2, in2)
    plot_settings(diff, in1 + '-' + in2)
    plot_settings(ax3, in2)
    plot_settings(ax4, in2)

    fig.savefig('deriv_test12.png')
    plt.close()

def plot_settings(ax, ts):
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=10)
    ax.set(xlabel='lag (ms)', ylabel='R', title=ts)
    ax.tick_params(axis='both', labelsize=32)
    ax.xaxis.label.set_fontsize(64)
    ax.yaxis.label.set_fontsize(64)
    #ax.set_title(in_type[:-3] +ts, loc='right')
    ax.title.set_fontsize(64)



    

def sep_roi_stats(lags, lshift, num_layers, in_type, cutoff_P, cutoff_R, omit_e, num_samps, sampling_amt):
    #roi_imR = np.zeros((len(rois), num_layers))
    #roi_imT = np.zeros((len(rois), num_layers))
    weight_avg = False
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    omit_e = []
    #rois = ['mSTG', 'aSTG', 'ifg', 'TP']
    rois = ['ifg','mSTG','aSTG','depth','TP','MFG','AG','preCG','MTG']
    roi_avg = []
    ev_by_roi = {} # dictionary of e --> value by roi
    for p, r in enumerate(rois):
        #print(r)
        elec_list = get_e_list(r + '_e_list.txt', '\t')
        #breakpoint()
        if num_samps > 0:
            print('sampling')
            avg = 0
            for i in range(num_samps):
                #inds = np.random.randint(len(elec_list), size = sampling_amt)
                #breakpoint()
                inds = rng.choice(len(elec_list), size = sampling_amt, replace=False)
                fe_list = list(np.array(elec_list)[inds])
                #print(i, r, elec_list)
                breakpoint()
                el_med,el_max, el_maxR, enc_bea = modular_prepro(lags,-500, 500, lshift,  fe_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, False)
                breakpoint()
                from matplotlib.gridspec import GridSpec
                fig = plt.figure(constrained_layout = True, figsize=[55, 15])
                gs = GridSpec(1, 3, figure = fig)
                ax = fig.add_subplot(gs[0,0])
                #ax2 = fig.add_subplot(gs[0,1])
                #ax3 = fig.add_subplot(gs[0,2])
                ax4 = fig.add_subplot(gs[0,1])
                #ax5 = fig.add_subplot(gs[1,1])
                ax6 = fig.add_subplot(gs[0,2])

                single_lag_layer(ax, el_med, el_max, False, cutoff_R, in_type, '-500 to 500', weight_avg, elec_list)
   
                single_lag_layer_enc(ax4,num_layers, enc_bea, lags, -2000,2000, False)
                single_lag_layer_enc(ax6,num_layers, enc_bea, lags, -500,500, True)
                ax6.set_ylim([0.9, 1.08]) 
                    
                fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/confounder_' + str(r) + '_' + str(i)+ '.png') 
                plt.close() 

                max_lag_per_layer = np.nanargmax(enc_bea, axis = -1)
                std =  np.nanstd(max_lag_per_layer, ddof=1) # ddof=1 yields unbiased estimator
                #print(r,std)
                avg += std
            #print('avg std: ', avg/num_samps)
            roiv = avg/num_samps
            roi_avg.append(roiv)
            for e in elec_list:
                ev_by_roi[e] = roiv
        else:
            #print('bad')
            print('no sampling')
            print(r, elec_list)
            #breakpoint()
            el_med,el_max, el_maxR, layer_lag = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, False)
            #z = [i for i in range(len(elec_list)) if elec_list[i] in e_list]
            max_lag_per_layer = np.nanargmax(layer_lag, axis = -1)
            #roi_max = np.nanmean(np.array(el_max)[z], axis = 0)
            #roi_maxR = np.nanmean(np.array(el_maxR)[z], axis = 0)
            #roi_imT[p,:] = roi_max 
            #roi_imR[p,:] = roi_maxR
            #breakpoint()
            #print(r,'std first then avg', np.nanmean(std))
            roiv = np.nanstd(max_lag_per_layer, ddof=1)
            print(r, roiv) # ddof=1 yields unbiased estimator
            for e in elec_list:
                ev_by_roi[e] = roiv*lshift # no need to shift for standard deviation. just convert from lags to ms
     
    #return roi_imT, roi_imR
    #breakpoint()
    if num_samps > 0:
        for i in range(len(rois)):
            print(rois[i], 'avg std: ', roi_avg[i])
    return ev_by_roi 

def elec_brain_plot(lags, lshift, num_layers, in_type, cutoff_P, cutoff_R, omit_e):
    weight_avg = False
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    omit_e = ['742_G64']
    elec_list = get_e_list('all_e_list.txt', '\t')
    el_med,el_max, el_maxR, layer_lag = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, False)
    e2v= {}
    for i, e in enumerate(elec_list): 
        breakpoint()
        e2v[e] = np.nanvar(np.array(el_max)[i,:], ddof=1)
        print(e2v[e])
    in_f = os.path.join(os.getcwd(), 'brain_map_input.txt')
    save_e2l(e2v, in_f, 'variance', 'gpt2-xl', omit_e)

def sep_roi_heatmap(rois, elec_list, el_max, el_maxR, num_layers):
    roi_imR = np.zeros((len(rois), num_layers))
    roi_imT = np.zeros((len(rois), num_layers))
    for p, r in enumerate(rois):
        e_list = get_e_list(r + '_e_list.txt', '\t')
        z = [i for i in range(len(elec_list)) if elec_list[i] in e_list]
        roi_max = np.nanmean(np.array(el_max)[z], axis = 0)
        roi_maxR = np.nanmean(np.array(el_maxR)[z], axis = 0)
        roi_imT[p,:] = roi_max 
        roi_imR[p,:] = roi_maxR

    return roi_imT, roi_imR

def degree_of_frontal_heatmap(rois, elec_list, el_max, el_maxR,num_layers):
    # get electrode positions for given regions
    # sort them, index into el_max, el_maxR to get the values
    # subject\telectrode\tX\tY\tZ
    e2idx = {} # maps electrode to index regardless of region
    e2roi = {} # maps electrode to roi it is in
    for p, r in enumerate(rois):
        e_list = get_e_list(r + '_e_list.txt', '\t')
        z = [i for i in range(len(elec_list)) if elec_list[i] in e_list]
        e2idx.update(zip(e_list, z))    
        e2roi.update(zip(e_list,np.repeat(r, len(e_list))))

    e2coord = {}
    with open('e_coords.txt', 'r') as coords:
        for line in coords:
            line_items = line.split('\t')
            elec = '_'.join(line_items[:2]) 
            if elec in e2idx:
                e2coord[elec] = float(line_items[3])
    
    #breakpoint()
    e2coord_sorted = dict(sorted(e2coord.items(), key = lambda x: x[1]))
    roi_imR = np.zeros((len(e2coord_sorted), num_layers))
    roi_imT = np.zeros((len(e2coord_sorted), num_layers))
    roi_order = []
    for i, e in enumerate(e2coord_sorted):
        roi_imT[i,:] = el_max[e2idx[e]]
        roi_imR[i,:] = el_maxR[e2idx[e]]
        roi_order.append(e2roi[e])

    return roi_imT, roi_imR, roi_order

def spatiotemporal_heatmaps(lags, lshift, elec_list, num_layers, in_type,cutoff_P, cutoff_R, omit_e):
    #fig, (ax1, ax2)= plt.subplots(1, 2, figsize=[20,10])
    #plt.subplots_adjust(hspace=0.001, wspace=0.5)
    weight_avg = False 
    omit_e = []
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    #el_med,el_max, el_maxR, layer_lag = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg)
    rois = ['mSTG', 'aSTG', 'ifg', 'TP']
    
    #roi_imT, roi_imR = sep_roi_heatmap(rois, elec_list, el_max, el_maxR, num_layers) # each roi separate, average electrodes
    #roi_imT, roi_imR, roi_order = degree_of_frontal_heatmap(rois, elec_list, el_max, el_maxR, num_layers) 
    #breakpoint()
    #sep_roi_stats(rois, elec_list, layer_lag)
    #breakpoint()
    #plot_spatiotemporal_heatmaps(ax1, np.transpose(roi_imT), rois, 'Lag with Max R')
    #plot_spatiotemporal_heatmaps(ax2, np.transpose(roi_imR), rois, 'Max R')


   
    #fig.savefig('all_e_heatmap_test.png')
    #plt.close()

def plot_spatiotemporal_heatmaps(ax, bea,rois, tl):
    breakpoint()
    im = ax.imshow(bea, origin='lower', cmap='Greys_r')
    #ax.set_yticks(np.arange(0, 4))# -1000 to 1000 (-1000/25 + 80 --> 40)
    #ax.set_yticklabels(rois)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set(ylabel='layer', xlabel='rois', title = tl) 
    #ax.grid()

def paper_llR(el_med, el_max):
    med_lag_corr = pearsonr(np.arange(len(el_med)), el_med)[0]
    max_lag_corr = pearsonr(np.arange(len(el_max)), el_max)[0]

    return np.round(med_lag_corr,2), np.round(max_lag_corr, 2)


def paper_lag_layer(ax, el_med, el_max):
    ts = ' Top Lag Per Layer \n(MedR, MaxR) = ('
    med_lag_corr, max_lag_corr = paper_llR(el_med, el_max)
    med_avg = el_med
    max_avg = el_max
    print ('max avg: ', max_avg)
    ax.plot(np.arange(1,len(med_avg)+1), med_avg, '-o', markersize=2,color = 'blue', label='med lag') 
    ax.plot(np.arange(1,len(max_avg)+1), max_avg, '-o', markersize=2,color = 'red', label='lag with top R')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=24)
    print('med R: ', str(med_lag_corr), ' max R: ', str(max_lag_corr))
    ax.tick_params(axis='both', labelsize=64)

def paper_enc(ax,encoding_array, xl,xr,norm, ps):
    max_lags = list(ps['lags'][np.argmax(encoding_array, axis = -1)])
    ld = {}
    h = []
    h0 = 1.01
    alpha = 0.001
    for l in max_lags:
        if l in ld:
            h.append(h0 + alpha*ld[l])
            ld[l] +=1
        else:
            ld[l] = 1
            h.append(h0)

    hues = list(np.arange(0, 240 + 240/(ps['num_layers'] - 1), 240/(ps['num_layers']-1))/360) # red to blue
    # red = [255, 0, 0]
    # blue [0, 0, 255]
    #breakpoint()
    #cvals = list(np.arange(0, 255, ps['num_layers']))
    c_a = np.zeros((3, ps['num_layers']))
    #c_a[2,:,:] = cvals
    #c_a[0,:,:] = np.flip(c_vals)
    ts = ' Avg Electrode Encoding Per Layer'
    if norm == True:
        ts = 'Norm\n' + ts
    
    for j in range(ps['num_layers']):
        c = list(clr.hsv_to_rgb([hues[j], 1.0, 1.0]))
        #c = hues[j]
        if norm == True:
            ax.plot(ps['lags'], encoding_array[j]/np.max(encoding_array[j]), color=c, label='layer' + str(j+1),zorder=-1) #**
            ax.scatter([max_lags[j]], h[j], color=c, zorder=-1)
        else:
            ax.plot(ps['lags'], encoding_array[j], color=c, label='layer' + str(j+1),zorder=-1) #**
    if not norm:
        ax.set_ylim([0, 0.3])
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=10)
    ax.set_xlim([xl, xr])
    ax.tick_params(axis='both', labelsize=64)

def paper_prepro(elec_list, slag, elag, omit_e, fpath, verbose, ps):
    breakpoint()

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
            #breakpoint()
            # iterate through layers. 
            # get the lags sorted by correlation
            top_lags = np.argsort(big_e_array, axis=-1)
            # remove lags out of the range
            top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
            # verify no nan entries
            top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[top_lags[p]])]
            assert(len(top_lags_nonan) == len(top_lags)) # verify no nan values bc not doing significance test. 
            # median
            top_lags_med = np.median(top_lags)
            # convert to ms
            assert(ps['lshift']*(top_lags_med-half_lags) == ps['lags'][int(top_lags_med)])
            med_elags.append(ps['lshift']*(top_lags_med-half_lags))
 
            # max
            # TODO: check this
            # get the correlations for the lags in the range provided, replace rest with -inf so the indices
            # are still valid. 
            #topk_vals = [big_e_array[p] if p in top_lags else -np.inf for p in range(len(big_e_array))]
            # get the index with the maximum correlation. note that it is not the max lag in the range, 
            # but the lag that maximizes the correlation within the range.
            top_lags_max = top_lags[-1]
            #top_lags_max = np.nanargmax(topk_vals)
            #assert(alt_max == top_lags_max)
            # convert to ms
            #assert(lshift*(top_lags_max-half_lags) == lags[top_lags_max])
            #max_elags.append(lshift*(top_lags_max-half_lags))
            max_elags.append(ps['lags'][top_lags_max])

        # average the top lags for all electrodes for a given layer
        #breakpoint()
        med_lag_per_layer.append(np.mean(med_elags))
        max_lag_per_layer.append(np.mean(max_elags))
        
    assert(len(med_lag_per_layer) == len(max_lag_per_layer))
    assert(len(med_lag_per_layer) == ps['num_layers'])
    encoding_array = np.mean(all_sig_R, axis = 1)
    
    return med_lag_per_layer, max_lag_per_layer, encoding_array 

def modular_prepro(lags,slag, elag, lshift, elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg,verbose):
    all_e_layer_med = []
    all_e_layer_max = []
    all_e_layer_maxR = []
    num_lags = len(lags)
    half_lags = num_lags//2
    start = math.floor(slag/lshift) +half_lags # reverse of below 161//2 = 80. 
    end = math.floor(elag/lshift) + half_lags

    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in2 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split1-jump1-l1-50-git-ws200-pval-gpt2-xl-hs' 
    if verbose == True:print(fpath)
    #breakpoint()
    
    #all_sig_R_p = extract_single_bigbigfdr_correlation(fpath, elec_list, list(np.arange(1,num_layers+1)), len(lags), False)
    all_sig_R = preprocess_encoding_results(fpath, elec_list)
    all_sig_R = np.transpose(all_sig_R, (1,0,2))
    breakpoint()

    print('num nans: ', quantify_nans(all_sig_R))
    #breakpoint()
    all_e_big_e_array = np.zeros((num_layers, num_lags))
    layer_count = list(np.zeros(num_layers))
    for i, e in enumerate(elec_list):
        if e in omit_e:
            print('bad electrode found')
            continue
        #breakpoint() 
        if verbose==True:print(e)
        #   breakpoint()
        big_e_array = all_sig_R[i]
        e_layer_med = []
        e_layer_max = []
        e_layer_maxR = []
        topk_size = []
        # iterate through layers. 
        for k in range(big_e_array.shape[0]):
            try:
                top_lags = np.argsort(big_e_array[k,:], axis=-1)
                top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                #top %
                #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > cutoff_R] # in general
                thresholded_lags = top_lags_nonan
                #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0] # in general
                #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0] 
                #breakpoint()
                topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                #breakpoint()
                topk_lags = np.sort(thresholded_lags[-topk:])
                #print(len(topk_lags), cutoff_P*big_e_array.shape[1])
                topk_size.append(topk)
                assert(len(top_lags_nonan) == len(top_lags)) # verify no nan values bc not doing significance test. 
                assert(len(thresholded_lags) == len(top_lags))
                assert(topk == len(top_lags))
                #topk_size.append(topk)
                if len(topk_lags) == 0:
                    e_layer_max.append(np.nan)
                    e_layer_med.append(np.nan)
                    e_layer_maxR.append(np.nan)
                    break
                    continue
                #breakpoint()
                #breakpoint() 
                # NOTE: where you add for big array
                if weight_avg:
                    #breakpoint()
                    num_e = float(np.load(e + '_nume.npy'))
                    all_e_big_e_array[k,:] = np.nansum([num_e*big_e_array[k,:], all_e_big_e_array[k,:]], axis = 0)
                else:
                    all_e_big_e_array[k,:] = np.nansum([big_e_array[k,:], all_e_big_e_array[k,:]], axis = 0)
                    layer_count[k] +=1
                #breakpoint()
                # median
                top_lags_med = np.nanmedian(topk_lags) #topk_lags[len(topk_lags)//2] # we don't want true median which can give fractional values) already sorted here
                e_layer_med.append(lshift*(top_lags_med-half_lags)) # no start/end here because want actual length
                #e_layer_R.append(big_e_array[k, top_lags_med])
                #max
                #breakpoint()
                topk_vals = [big_e_array[k, p] if p in topk_lags else -np.inf for p in range(len(big_e_array[k,:]))]
                top_lags_max = np.nanargmax(topk_vals)
                tl_max_val = np.max(topk_vals)
                if top_lags_max not in topk_lags:
                    breakpoint()
                
                assert(top_lags_max in topk_lags)
                e_layer_max.append(lshift*(top_lags_max-half_lags)) # no start/end here because want actual length
                e_layer_maxR.append(tl_max_val)
            except ValueError:
                e_layer_max.append(np.nan)
                e_layer_med.append(np.nan)
                e_layer_maxR.append(np.nan)
                topk_size.append(np.nan)  
        all_e_layer_med.append(e_layer_med)
        all_e_layer_max.append(e_layer_max)
        all_e_layer_maxR.append(e_layer_maxR) 
    #breakpoint()
    if not weight_avg:
        for k in range(num_layers):
            if layer_count[k] == 0:
                #breakpoint()
                print(k, 'FAIL')
            all_e_big_e_array[k,:] /= layer_count[k]
    #breakpoint()
    return all_e_layer_med, all_e_layer_max, all_e_layer_maxR, all_e_big_e_array

def avg_e_first_laglayer_prepro(ax,big_e_array, slag, elag, cutoff_P, cutoff_R,lshift):
    num_lags = len(lags)
    half_lags = num_lags//2
    start = math.floor(slag/lshift) +half_lags # reverse of below 161//2 = 80. 
    end = math.floor(elag/lshift) + half_lags
    max_lags = []
    med_lags = []
    for k in range(big_e_array.shape[0]):
        #if k == 42:
        #    breakpoint()
        top_lags = np.argsort(big_e_array[k,:], axis=-1)
        top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
        
        top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
        thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > cutoff_R] # in general
        topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
        topk_lags = np.sort(thresholded_lags[-topk:])
        if len(topk_lags) == 0:
            med_lags.append(np.nan)
            max_lags.append(np.nan)
            continue
        top_lags_med = np.nanmedian(topk_lags) #topk_lags[len(topk_lags)//2] # we don't want true median which can give fractional values) already sorted here
        med_lags.append(lshift*(top_lags_med-half_lags)) # no start/end here because want actual length

        topk_vals = [big_e_array[k, p] if p in topk_lags else 0 for p in range(len(big_e_array[k,:]))]
        
        top_lags_max = np.nanargmax(topk_vals)
        #if top_lags_max not in topk_lags:
        #    print(k)
        #    #breakpoint()
        
        assert(top_lags_max in topk_lags)
        max_lags.append(lshift*(top_lags_max-half_lags)) # no start/end here because want actual length

    return np.expand_dims(med_lags, 0), np.expand_dims(max_lags, 0)

def plot_mat(mat):
    fig = plt.figure()
    for i in range(mat.shape[0]):
        plt.plot(range(mat.shape[1]), mat[i, :])
    plt.title('Test Enc Plot')
    plt.savefig('test5_enc.png')
    plt.close()

#modular_lag_layer_nocorr(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x):
def modular_lag_layer_nocorr(elec_list, omit_e, lags, lshift, num_layers, in_type, cutoff_P, cutoff_R,slag, elag, brain_area):
    brain_dict = {'ifg':'ifg', 'all':'all','mSTG':'mSTG','aSTG':'aSTG','depth':'depth','TP':'TP','MFG':'MFG','AG':'AG','preCG':'preCG','MTG':'MTG'}

    # below replicates previous avg_lag_layer_nocorr
    '''
    fig, (ax, ax2, ax3, ax4)= plt.subplots(1, 4, figsize=[40,10])
    plt.subplots_adjust(hspace=0.001, wspace=0.5)
     fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in_type
    el_med,el_max, _, bea = modular_prepro(lags,slag,elag, lshift, elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath)

    single_lag_layer(ax, el_med, el_max, False, cutoff_R, in_type)
    single_lag_layer(ax2, el_med, el_max, True, cutoff_R, in_type)
    single_lag_layer_enc(ax3,num_layers, bea, lags, -2000,2000, False)
    single_lag_layer_enc(ax4,num_layers, bea, lags, -250,250, True)
    ''' 
    
    #fig, ((ax, ax2, ax3), (ax4, ax6))= plt.subplots(3, 2, figsize=[50,20])
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(constrained_layout = True, figsize=[35,35])
    #gs = GridSpec(1, 3, figure = fig)
    gs = GridSpec(2, 2, figure = fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    #ax = fig.add_subplot(gs[0,0])
    #ax2 = fig.add_subplot(gs[0,1])
    #ax3 = fig.add_subplot(gs[0,2])
    #ax4 = fig.add_subplot(gs[0,1])
    #ax5 = fig.add_subplot(gs[1,1])
    #ax6 = fig.add_subplot(gs[0,2])
    #weight_avg = True
    #plt.subplots_adjust(hspace=0.1, wspace=0.5)
    core_roi = 'mSTG'
    out_roi = 'TP'
    patient_list = ['742']
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in_type
    #omit_e = ['742_G64']
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-nov-avg-' + core_roi + '-' + out_roi + '-enc-correct-top1-gpt2-xl-hs'
    elec_list = get_e_list(core_roi + '_e_list.txt', '\t')
    weight_avg = False
    el_med,el_max, _, enc_bea = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    single_lag_layer_enc(ax1,num_layers, enc_bea, lags, -2000,2000, False)
    single_lag_layer_enc(ax2,num_layers, enc_bea, lags, -500,500, True)
    ax1.set_title(core_roi + ' no norm')
    ax2.set_title(core_roi + ' norm')
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-incorrect-top1-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-incorrect-top5-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-avg-*ifg-enc-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-avg-*ifg-enc-correct-top1-gpt2-xl-hs' 
    
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-avg-individual-' + core_roi + '-' + out_roi + '-enc-correct-top1-gpt2-xl-hs'
    
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-nov-avg-' + core_roi + '-' + out_roi + '-enc-correct-top1-gpt2-xl-hs'
    #fpath= '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-avg-742_mSTG-TP-enc-correct-top1-gpt2-xl-hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-nov-avg-TP-mSTG-enc-correct-top1-gpt2-xl-hs'
    #elec_list = [core_roi + '_enc']
    
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-742-reg-avg-' + core_roi + '-' + out_roi + '-enc-correct-top1-gpt2-xl-hs'
    elec_list = []
    for p in patient_list:
        elec_list.append(p + '_' + core_roi + '_enc')
    #elec_list = ['aSTG_for_TP']
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-avg-individual-TP-mSTG-enc-correct-top1-gpt2-xl-hs'
    #elec_list = ['798_TP_enc','742_TP_G54','742_TP_G55', '742_TP_G56', '742_TP_G63']
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-group5-group0-gpt2-xl'
    #elec_list = ['bobbi_up2date2'] 
    #elec_list = ['TP_enc']
    #elec_list = ['742_TP_enc']
    #weight_avg = True
    #elec_list = ['742_TP_enc', '798_TP_enc']
    #elec_list = ['717_aSTG_enc', '741_aSTG_enc', '742_aSTG_enc']
    #elec_list = ['717_ifg_enc', '72_ifg_enc', '798_ifg_enc']
    #elec_list = ['cSTG_enc']
    #elec_list = ['mSTG_for_TP']
    #elec_list = ['717_mSTG_enc']
    elec_list = ['742_mSTG_enc']
    #elec_list = ['717_mSTG_enc', '741_mSTG_enc', '742_mSTG_enc', '743_mSTG_enc', '763_mSTG_enc']
    #elec_list = ['742_TP_enc']
    #weight_avg = True
    omit_e = []
    el_med,el_max, _, enc_bea2 = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    #breakpoint()
    breakpoint()
    #enc_bea2 /= 2
    plot_mat(enc_bea2)
    #print(np.nanargmax(enc_bea, axis=0))
    # uri version
    #med_lags, max_lags = avg_e_first_laglayer_prepro(ax,enc_bea, slag, elag, cutoff_P, cutoff_R,lshift)
    #single_lag_layer(ax5, med_lags, max_lags, False, cutoff_R, in_type, '-500 to 500 Avg E Girst')

    # regular -500 to 500
    #single_lag_layer(ax, el_med, el_max, False, cutoff_R, in_type, '-500 to 500', weight_avg, elec_list)
   
    #el_med,el_max, _, bea = modular_prepro(lags,-500, 0, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath)
    #single_lag_layer(ax2, el_med, el_max, False, cutoff_R, in_type, '-500 to 0')
   
    #el_med,el_max, _,bea = modular_prepro(lags,0,500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath)
    #single_lag_layer(ax3, el_med, el_max, False, cutoff_R, in_type, '0 to 500')
    #breakpoint()
    single_lag_layer_enc(ax3,num_layers, enc_bea2, lags, -2000,2000, False)
    #single_lag_layer_enc(ax5,num_layers, enc_bea, lags, -500,500, max_lags,False)
    single_lag_layer_enc(ax4,num_layers, enc_bea2, lags, -500,500, True)
    #ax4.set_ylim([0.9, 1.08]) 
    ax3.set_title(core_roi + '-' + out_roi + ' no norm')
    ax4.set_title(core_roi + '-' + out_roi + ' norm')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.title.set_fontsize(64)
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/new-test-742-no-omit-weight-avg-after-correct-top1_' + core_roi + '-' + out_roi + '_' + brain_dict[brain_area] + '_top' + str(cutoff_P) + 'p_corr_threshold' + str(cutoff_R) + '_ws200_'+  in2 + '_' + str(slag)+ '_to_' + str(elag) + 'shuffle=False_SIG.png') 
    
    #fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/_' + brain_dict[brain_area] + '_top' + str(cutoff_P) + 'p_corr_threshold' + str(cutoff_R) + '_ws200_'+  in2 + '_' + str(slag)+ '_to_' + str(elag) + 'shuffle=False_SIG.png') 
 
    plt.close()
    return

#modular_lag_layer_nocorr(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x):
def modular_encoding_lag_layer_nov_update(elec_list, omit_e, lags, lshift, num_layers, in_type, cutoff_P, cutoff_R,slag, elag, brain_area):
    brain_dict = {'ifg':'ifg', 'all':'all','mSTG':'mSTG','aSTG':'aSTG','depth':'depth','TP':'TP','MFG':'MFG','AG':'AG','preCG':'preCG','MTG':'MTG'}
    
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(constrained_layout = True, figsize=[70,35])
    gs = GridSpec(2, 4, figure = fig)
    # ll# corresponds to encoding plot ax#
    ll1 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
   
    ll2 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    
    ll3 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    
    ll4 = fig.add_subplot(gs[1, 2])
    ax4 = fig.add_subplot(gs[1, 3])

    
    omit_e = [] 
    weight_avg = False
    core_roi = 'ifg'
    out_roi = 'TP'
    patient_list = ['798', '742']
   
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    elec_list = get_e_list(core_roi + '_e_list.txt', '\t')
    el_med,el_max, _, enc_bea = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    single_lag_layer_enc(ax1,num_layers, enc_bea, lags, -2000,2000, False)
    single_lag_layer(ll1, el_med, el_max, False, cutoff_R, in_type, '-500 to 500', weight_avg, elec_list)
    
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-nov-avg-' + core_roi + '-' + out_roi + '-enc-correct-top1-gpt2-xl-hs'
    elec_list = []
    for p in patient_list:
        elec_list.append(p + '_' + core_roi + '_enc')
    el_med,el_max, _, enc_bea = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    single_lag_layer_enc(ax2,num_layers, enc_bea, lags, -2000,2000, False)
    single_lag_layer(ll2, el_med, el_max, False, cutoff_R, in_type, '-500 to 500', weight_avg, elec_list)
    
    #ax1.set_title(core_roi)
    #ax2.set_title(core_roi+ '-' + out_roi)
   
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    elec_list = get_e_list(out_roi + '_e_list.txt', '\t')
    el_med,el_max, _, enc_bea2 = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    single_lag_layer_enc(ax3,num_layers, enc_bea2, lags, -2000,2000, False)
    single_lag_layer(ll3, el_med, el_max, False, cutoff_R, in_type, '-500 to 500', weight_avg, elec_list)

    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-nov-avg-' + out_roi + '-' + core_roi + '-enc-correct-top1-gpt2-xl-hs'
    elec_list = []
    for p in patient_list:
        elec_list.append(p + '_' + out_roi + '_enc')
    el_med,el_max, _, enc_bea2 = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    single_lag_layer_enc(ax4,num_layers, enc_bea2, lags, -2000,2000, False)
    single_lag_layer(ll4, el_med, el_max, False, cutoff_R, in_type, '-500 to 500', weight_avg, elec_list)
    
    #ax3.set_title(out_roi)
    #ax4.set_title(out_roi + '-' + core_roi)
    
    #for ax in [ax1, ax2, ax3, ax4]:
    #    ax.title.set_fontsize(64)
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/nov-update-lag-layer-no-omit-no-weight-avg-after-correct-top1_' + core_roi + '-and-' + out_roi + '_top' + str(cutoff_P) + 'p_corr_threshold' + str(cutoff_R) + '_ws200_'+  in2 + '_' + str(slag)+ '_to_' + str(elag) + 'shuffle=False_SIG.png') 
 

#modular_lag_layer_nocorr(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x):
def og_modular_encoding_lag_layer_nov_update(omit_e, lags, lshift, num_layers, in_type, cutoff_p, cutoff_r,slag, elag, brain_area):
    print('start')
    brain_dict = {'ifg':'ifg', 'all':'all','mstg':'mstg','astg':'astg','depth':'depth','tp':'tp','mfg':'mfg','ag':'ag','precg':'precg','mtg':'mtg'}
    
    from matplotlib.gridspec import gridspec
    fig = plt.figure(constrained_layout = true, figsize=[50,15])
    gs = gridspec(1, 3, figure = fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    omit_e = [] 
    weight_avg = false
    core_roi = brain_area
   
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-nov-new-no-weight-avg-ifg_nonsig-enc-correct-top1-gpt2-xl-hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-udpated/results/podcast2/podcast2-gpt2-xl-pca50d-full-avg-742_' + core_roi + '-enc-correct-top1-gpt2-xl-hs'
    elec_list = get_e_list(core_roi + '_e_list.txt', '\t')
    #breakpoint()
    print(elec_list)
    #el_med,el_max, _, enc_bea = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_p, cutoff_r, omit_e, fpath, weight_avg, true)
    el_med,el_max, enc_bea = paper_prepro(elec_list, -500, 500, omit_e, fpath, false)
    #el_med = np.array(list(map(np.array, el_med)))
    #el_max = np.array(list(map(np.array, el_max)))
    #single_lag_layer(ax1, el_med, el_max, false, cutoff_r, in_type, '-500 to 500', weight_avg, elec_list)
    
    paper_lag_layer(ax1, el_med, el_max)
    #single_lag_layer_enc(ax2,num_layers, enc_bea, lags, -2000,2000, false)
    #single_lag_layer_enc(ax3,num_layers, enc_bea, lags, -500,500, true)
    paper_enc(ax2, enc_bea, -2000, 2000, false)
    paper_enc(ax3, enc_bea, -500, 500, true)
    plt.close()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/paper-code-og-lag-layer-no-omit-no-weight-avg-after-correct-top1_' + core_roi + '_top' + str(cutoff_p) + 'p_corr_threshold' + str(cutoff_r) + '_ws200_'+  in2 + '_' + str(slag)+ '_to_' + str(elag) + 'shuffle=false_notsig.png') 

def get_params(elec_list):
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    num_lags = len(lags)
    layer_list = list(np.arange(1, 49))
    num_layers = len(layer_list)
    num_electrodes = len(elec_list)
    params = {'lshift': lshift, 'lags': lags, 'num_lags': num_lags, 'layer_list': layer_list, 'num_layers': num_layers, 'num_electrodes': num_electrodes}
    return params

#modular_lag_layer_nocorr(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x):
def paper_plots(omit_e, lags, lshift, num_layers, in_type, cutoff_p, cutoff_r,slag, elag, brain_area):
    core_roi = brain_area
    elec_list = get_e_list(core_roi + '_e_list.txt', '\t')
    print(elec_list)

    params = get_params(elec_list) 
    from matplotlib import gridspec
    fig = plt.figure(constrained_layout = True, figsize=[50,15])
    gs = gridspec.GridSpec(1, 3, figure = fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    omit_e = [] 
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    el_med,el_max, enc_bea = paper_prepro(elec_list, -500, 500, omit_e, fpath, False, params)
    
    paper_lag_layer(ax1, el_med, el_max)
    paper_enc(ax2, enc_bea, -2000, 2000, False, params)
    paper_enc(ax3, enc_bea, -500, 500, True, params)
    plt.close()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/paper-code-og-lag-layer-no-omit-no-weight-avg-after-correct-top1_' + core_roi + '_top' + str(cutoff_p) + 'p_corr_threshold' + str(cutoff_r) + '_ws200_'+  in2 + '_' + str(slag)+ '_to_' + str(elag) + 'shuffle=false_notsig.png') 


def paper_plot():
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(constrained_layout = True, figsize=[50,15])
    gs = GridSpec(1, 3, figure = fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
#modular_lag_layer_nocorr(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x):
def modular_encoding_nov_update(elec_list, omit_e, lags, lshift, num_layers, in_type, cutoff_P, cutoff_R,slag, elag, brain_area):
    brain_dict = {'ifg':'ifg', 'all':'all','mSTG':'mSTG','aSTG':'aSTG','depth':'depth','TP':'TP','MFG':'MFG','AG':'AG','preCG':'preCG','MTG':'MTG'}
    
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(constrained_layout = True, figsize=[35,35])
    gs = GridSpec(2, 2, figure = fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    
    omit_e = [] 
    weight_avg = False
    core_roi = 'aSTG'
    out_roi = 'ifg'
    patient_list = ['742']
   
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    elec_list = get_e_list(core_roi + '_e_list.txt', '\t')
    el_med,el_max, _, enc_bea = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    single_lag_layer_enc(ax1,num_layers, enc_bea, lags, -2000,2000, False)
    
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-nov-avg-' + core_roi + '-' + out_roi + '-enc-correct-top1-gpt2-xl-hs'
    elec_list = []
    for p in patient_list:
        elec_list.append(p + '_' + core_roi + '_enc')
    el_med,el_max, _, enc_bea = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    single_lag_layer_enc(ax2,num_layers, enc_bea, lags, -2000,2000, False)
    
    ax1.set_title(core_roi)
    ax2.set_title(core_roi+ '-' + out_roi)
   
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    elec_list = get_e_list(out_roi + '_e_list.txt', '\t')
    el_med,el_max, _, enc_bea2 = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    single_lag_layer_enc(ax3,num_layers, enc_bea2, lags, -2000,2000, False)
    

    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-nov-avg-' + out_roi + '-' + core_roi + '-enc-correct-top1-gpt2-xl-hs'
    elec_list = []
    for p in patient_list:
        elec_list.append(p + '_' + out_roi + '_enc')
    el_med,el_max, _, enc_bea2 = modular_prepro(lags,-500, 500, lshift,  elec_list,num_layers, cutoff_P, cutoff_R, omit_e, fpath, weight_avg, True)
    single_lag_layer_enc(ax4,num_layers, enc_bea2, lags, -2000,2000, False)

    ax3.set_title(out_roi)
    ax4.set_title(out_roi + '-' + core_roi)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.title.set_fontsize(64)
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/nov-update-no-omit-no-weight-avg-after-correct-top1_' + core_roi + '-and-' + out_roi + '_top' + str(cutoff_P) + 'p_corr_threshold' + str(cutoff_R) + '_ws200_'+  in2 + '_' + str(slag)+ '_to_' + str(elag) + 'shuffle=False_SIG.png') 
 
def plot_glove_encoding(elec_list,lags):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = [50,20])
    # all
    all_enc = []
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-update-glove-gpt2-xlhs'
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, [1], len(lags), False)
    all_enc.append(np.mean(all_sig_R, axis = 0).squeeze())
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-prev-glove-gpt2-xlhs' 
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, [1],len(lags), False)
    all_enc.append(np.mean(all_sig_R, axis = 0).squeeze())
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-next-glove-gpt2-xlhs'
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, [1], len(lags), False)
    all_enc.append(np.mean(all_sig_R, axis = 0).squeeze())
    # correct
    correct_enc = []
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-update-glove-correct-top-1-gpt2-xlhs'
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, [1], len(lags), False)
    correct_enc.append(np.mean(all_sig_R, axis = 0).squeeze())
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-prev-glove-correct-top-1-gpt2-xlhs'
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, [1], len(lags), False)
    correct_enc.append(np.mean(all_sig_R, axis = 0).squeeze())
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-next-glove-correct-top-1-gpt2-xlhs'
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, [1], len(lags), False)
    correct_enc.append(np.mean(all_sig_R, axis = 0).squeeze())
    # incorrect
    incorrect_enc = []
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-update-glove-incorrect-top-1-gpt2-xlhs'
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, [1], len(lags), False)
    incorrect_enc.append(np.mean(all_sig_R, axis = 0).squeeze())
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-prev-glove-incorrect-top-1-gpt2-xlhs'
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, [1], len(lags), False)
    incorrect_enc.append(np.mean(all_sig_R, axis = 0).squeeze())
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-next-glove-incorrect-top-1-gpt2-xlhs'
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, [1], len(lags), False)
    incorrect_enc.append(np.mean(all_sig_R, axis = 0).squeeze())
    order = ['reg', 'prev', 'next']
    colors = ['blue', 'red', 'orange']
    breakpoint()
    for i,val in enumerate(all_enc):
        ax1.plot(lags, val, color=colors[i], label=order[i],zorder=-1) #**
    for i,val in enumerate(correct_enc):
        ax2.plot(lags, val, color=colors[i], label=order[i],zorder=-1) #**
    for i,val in enumerate(incorrect_enc):
        ax3.plot(lags, val, color=colors[i], label=order[i],zorder=-1) #**



    ax1.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=10)
    ax1.set(xlabel='lag (ms)', ylabel='R')
    ax1.vlines([-500, 500], 0, 0.4, colors = ['grey'], linestyles=['dashed'])
    #ax1.set_xlim([xl, xr])
    ax1.tick_params(axis='both', labelsize=32)
    ax1.xaxis.label.set_fontsize(64)
    ax1.yaxis.label.set_fontsize(64)
    ax1.set_title('all')
    ax1.set_xlim([-1000, 1000])
    ax1.title.set_fontsize(64)

    ax2.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=10)
    ax2.set(xlabel='lag (ms)', ylabel='R')
    ax2.vlines([-500, 500], 0, 0.4, colors = ['grey'], linestyles=['dashed'])
    #ax.set_xlim([xl, xr])
    ax2.tick_params(axis='both', labelsize=32)
    ax2.xaxis.label.set_fontsize(64)
    ax2.yaxis.label.set_fontsize(64)
    ax2.set_title('correct')
    ax2.set_xlim([-1000, 1000])
    ax2.title.set_fontsize(64)
    
    ax3.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=10)
    ax3.set(xlabel='lag (ms)', ylabel='R')
    ax3.vlines([-500, 500], 0, 0.4, colors = ['grey'], linestyles=['dashed'])
    #ax.set_xlim([xl, xr])
    ax3.tick_params(axis='both', labelsize=32)
    ax3.xaxis.label.set_fontsize(64)
    ax3.yaxis.label.set_fontsize(64)
    ax3.set_title('incorrect')
    ax3.set_xlim([-1000, 1000])
    ax3.title.set_fontsize(64)
    plt.savefig('glove_plots.png')

def lag_layer_nocorr(elec_list, omit_e, lags,lshift,num_layers,in2, cutoff_P, cutoff_R, slag, elag, brain_area):
    #all_e_layer_med = []
    #all_e_layer_max = []
    num_lags = len(lags)
    half_lags = num_lags//2
    start = math.floor(slag/lshift) +half_lags # reverse of below 161//2 = 80. 
    end = math.floor(elag/lshift) + half_lags

    #all_e_big_e_array = np.zeros((num_layers-1, num_lags)) # partial
    #layer_count = list(np.zeros(num_layers-1)) #partial
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-pval-' + in2
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-ws200-non-overlapping-pval-' + in2
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in2 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split2-jump1-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split1-jump1-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split1-jump47-cos-25-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split2-jump1-cos-25-git-ws200-pval-gpt2-xl-hs' 
    #breakpoint() 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-ws200-shuffle-pval-' + in2
    #breakpoint()
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-gpt2-xl-new_shuff_hs'
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-correct-top1-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-ncorrect-top5-git-ws200-pval-gpt2-xl-hs' 
   
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in_type
    #breakpoint()
    print(fpath)
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, list(np.arange(1,num_layers+1)), len(lags), False)
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in2 
    #all_sig_Rt = extract_single_bigbigfdr_correlation(fpath, elec_list, list(np.arange(1,num_layers+1)), len(lags))
    #breakpoint()
    #all_e_big_e_array = np.zeros((num_layers, num_lags))
    layer_count = list(np.zeros(num_layers))
    for i, e in enumerate(elec_list):
        if e in omit_e:
            print('bad electrode found')
            continue
        fig, (ax, ax4, ax3, ax3n)= plt.subplots(1, 4, figsize=[40,10])
        plt.subplots_adjust(hspace=0.001, wspace=0.5)

        print(e)
        big_e_array = all_sig_R[i]
        e_layer_med = []
        e_layer_max = []
        topk_size = []
        # iterate through layers. 
        for k in range(big_e_array.shape[0]):
            try:
                top_lags = np.argsort(big_e_array[k,:], axis=-1)
                top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                #top %
                thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > cutoff_R] # in general
                #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0] # in general

                #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0] 
                #breakpoint()
                topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                topk_lags = np.sort(thresholded_lags[-topk:])
                #print(len(topk_lags), cutoff_P*big_e_array.shape[1])
                topk_size.append(topk)
                #topk_size.append(topk)
                if len(topk_lags) == 0:
                    e_layer_max.append(np.nan)
                    e_layer_med.append(np.nan)
                    continue
                
                #all_e_big_e_array[k,:] = np.nansum([big_e_array[k,:], all_e_big_e_array[k,:]], axis = 0)
                layer_count[k] +=1
                #breakpoint()
                # median
                top_lags_med = np.nanmedian(topk_lags) #topk_lags[len(topk_lags)//2] # we don't want true median which can give fractional values) already sorted here
                e_layer_med.append(lshift*(top_lags_med-half_lags)) # no start/end here because want actual length
                #e_layer_R.append(big_e_array[k, top_lags_med])
                #max
                #breakpoint()
                topk_vals = [big_e_array[k, p] if p in topk_lags else 0 for p in range(len(big_e_array[k,:]))]
                top_lags_max = np.nanargmax(topk_vals)
                assert(top_lags_max in topk_lags)
                e_layer_max.append(lshift*(top_lags_max-half_lags)) # no start/end here because want actual length

            except ValueError:
                e_layer_max.append(np.nan)
                e_layer_med.append(np.nan)
                topk_size.append(np.nan)  
        #all_e_layer_med.append(e_layer_med)
        #all_e_layer_max.append(e_layer_max)
        med_avg_nn = [e_layer_med[p] for p in range(len(e_layer_med)) if not np.isnan(e_layer_med[p])]
        if len(med_avg_nn) >= 2:
            med_lag_corr = pearsonr(np.arange(len(med_avg_nn)), med_avg_nn)[0]
        else:
            print('<2 vals')
            med_lag_corr = 0

        max_avg_nn = [e_layer_max[p] for p in range(len(e_layer_max)) if not np.isnan(e_layer_max[p])]
        if len(max_avg_nn) >= 2:
            max_lag_corr = pearsonr(np.arange(len(max_avg_nn)), max_avg_nn)[0]
        else:
            print('<2 vals')
            max_lag_corr = 0


        ax.plot(np.arange(1,len(e_layer_med)+1), e_layer_med, '-o', markersize=2,color = 'blue', label='med lag for R > ' + str(cutoff_R))
        ax.plot(np.arange(1,len(e_layer_max)+1), e_layer_max, '-o', markersize=2,color = 'red', label='lag with top R')
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax.set(xlabel='layer', ylabel='lag (ms)', title= in2[:-3] + ' Top Lag Per Layer (Avg\'d over Electrodes), (MedR, MaxR) = (' + str(med_lag_corr) + ', ' + str(max_lag_corr) + ')')
        #ax.set_ylim([-250, 250])
        ax.tick_params(axis='both', labelsize=16)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)
        ax.title.set_fontsize(16)

        # add 1s because not starting at 0
        #breakpoint()
        med_zcore = (e_layer_med-np.nanmean(e_layer_med))/np.nanstd(e_layer_med)
        max_zscore = (e_layer_max-np.nanmean(e_layer_max))/np.nanstd(e_layer_max)
        ax4.plot(np.arange(1,len(e_layer_med)+1), (e_layer_med-np.nanmean(e_layer_med))/np.nanstd(e_layer_med), '-o', markersize=2,color = 'blue', label='med lag for R > ' + str(cutoff_R))
        ax4.plot(np.arange(1,len(e_layer_max)+1), (e_layer_max-np.nanmean(e_layer_max))/np.nanstd(e_layer_max), '-o', markersize=2,color = 'red', label='lag with top R')
        ax4.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax4.set(xlabel='layer', ylabel='lag (ms)', title= in2[:-3] + ' Top Lag Per Layer Z-Score')
        #ax4.set_ylim([-2,2])
        ax4.tick_params(axis='both', labelsize=16)
        ax4.xaxis.label.set_fontsize(16)
        ax4.yaxis.label.set_fontsize(16)
        ax4.title.set_fontsize(16)

        # layer_count is number of non nan entries per layer
        # in previous plots you do nanmean to account for nans in average.
        # all_e_big_e_array is the sum for a given layer over all electrodes. we divide by the count here to average. to get the average correlation per layer 
        # across all electrodes
        #for k in range(num_layers):
        #    #for k in range(num_layers-1): # partial
        #    if layer_count[k] == 0:
        #        print(k, 'FAIL')
        #    all_e_big_e_array[k,:] /= layer_count[k]
        #init_grey = 1
        #breakpoint()
        hues = list(np.arange(0, 240 + 240/(num_layers - 1), 240/(num_layers-1))/360) # red to blue
        # plot encoding for all layers for this electrode. 
        for j in range(num_layers):
      
            # input or first layer
            c = list(clr.hsv_to_rgb([hues[j], 1.0, 1.0]))
            #if j == 0:
            ax3.plot(lags, big_e_array[j], color=c, label='layer' + str(j+1),zorder=-1) #**
            ax3n.plot(lags, big_e_array[j]/np.nanmax(big_e_array[j]), color=c, label='layer' + str(j+1),zorder=-1) #**

            #elif j == num_layers-1:
            #    #elif j == num_layers - 1: #partial
            #    #clr = 'r'
            #    ax3.plot(lags, all_e_big_e_array[j], color=clr, label='layer' + str(j+1), zorder=-1) #**
            #else:
            #    #init_grey -= 1/(math.exp((j)*0.001)*(num_layers+1))
            #    #clr = str(init_grey)
            #    ax3.plot(lags, all_e_big_e_array[j], color=clr,label='layer' + str(j+1),zorder=-1) #**
            #try:
            #    #ax3.scatter([lags[np.nanargmax(all_e_big_e_array[j])]], [np.nanmax(all_e_big_e_array[j])], color=c, zorder=-1)
            #except ValueError:
            #    continue
        #breakpoint()
        ax3.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize='x-small')
        ax3.set(xlabel='lag (ms)', ylabel='R', title= in_type[:-3] + ' Avg Electrode Encoding Per Layer (Left 2 Plots Come From Points Between Dashed Lines)')
        #ax3.set_ylim([0, 0.4])
        ax3.vlines([-500, 500], 0, 0.4, colors = ['grey'], linestyles=['dashed'])
        #breakpoint()
        ax3.set_xlim([-2000, 2000])
        ax3.tick_params(axis='both', labelsize=16)
        #breakpoint()
        ax3.xaxis.label.set_fontsize(16)
        ax3.yaxis.label.set_fontsize(16)
        ax3.title.set_fontsize(16)

        ax3n.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize='x-small')
        ax3n.set(xlabel='lag (ms)', ylabel='R', title= in_type[:-3] + ' Avg Electrode Encoding Per Layer (Left 2 Plots Come From Points Between Dashed Lines)')
        #ax3.set_ylim([0, 0.4])
        ax3n.vlines([-500, 500], 0, 0.4, colors = ['grey'], linestyles=['dashed'])
        #breakpoint()
        ax3n.set_xlim([-2000, 2000])
        ax3n.tick_params(axis='both', labelsize=16)
        #breakpoint()
        ax3n.xaxis.label.set_fontsize(16)
        ax3n.yaxis.label.set_fontsize(16)
        ax3n.title.set_fontsize(16)

        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/nolimit_reg-e' + str(e) + '_git_bbfdr_update_' + brain_area + '_top' + str(cutoff_P) + 'p_corr_threshold' + str(cutoff_R) + '_ws200_'+  in2 + '_' + str(slag)+ '_to_' + str(elag) + 'shuffle=False_SIG.png') 
        plt.close()
        #break
    return



def avg_lag_layer_nocorr(elec_list, omit_e, lags,lshift,num_layers,in2, cutoff_P, cutoff_R, slag, elag, brain_area, num_back = 0):
    all_e_layer_med = []
    all_e_layer_max = []
    num_lags = len(lags)
    half_lags = num_lags//2
    start = math.floor(slag/lshift) +half_lags # reverse of below 161//2 = 80. 
    end = math.floor(elag/lshift) + half_lags

    #all_e_big_e_array = np.zeros((num_layers-1, num_lags)) # partial
    #layer_count = list(np.zeros(num_layers-1)) #partial
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-pval-' + in2
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-ws200-non-overlapping-pval-' + in2
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split2-jump1-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split2-jump1-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split1-jump47-cos-25-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split2-jump1-cos-25-git-ws200-pval-gpt2-xl-hs' 
    #breakpoint() 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-ws200-shuffle-pval-' + in2
    #breakpoint()
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-gpt2-xl-new_shuff_hs'
    
    # NOTE: these are vetted
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-incorrect-top1-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-incorrect-top5-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in2 
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split1-jump1-l1-50-git-ws200-pval-gpt2-xl-hs' 
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-split2-jump1-l1-50-git-ws200-pval-gpt2-xl-hs' 
    #glove_path = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-glove50-pca50d-full-incorrect-top1-git-ws200-pval-gpt2-xl-hs'
    #breakpoint()
    print(fpath)
    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, list(np.arange(1,num_layers+1)), len(lags), True)
    #breakpoint()
    #all_glove_sig_R = extract_single_bigbigfdr_correlation(glove_path, elec_list, list(np.arange(1,num_layers+1)), len(lags), False) # don't do sig here

    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in2 
    #all_sig_Rt = extract_single_bigbigfdr_correlation(fpath, elec_list, list(np.arange(1,num_layers+1)), len(lags))
    #breakpoint()
    all_e_big_e_array = np.zeros((num_layers-num_back, num_lags))
    layer_count = list(np.zeros(num_layers-num_back))
    for i, e in enumerate(elec_list):
        if e in omit_e:
            print('bad electrode found')
            continue
 
        print(e)
        big_e_array = all_sig_R[i]
        e_layer_med = []
        e_layer_max = []
        topk_size = []
        # iterate through layers. 
        for k in range(big_e_array.shape[0]):
            try:
                top_lags = np.argsort(big_e_array[k,:], axis=-1)
                top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                #top %
                thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > cutoff_R] # in general
                #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0] # in general

                #thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > 0] 
                #breakpoint()
                topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                topk_lags = np.sort(thresholded_lags[-topk:])
                #print(len(topk_lags), cutoff_P*big_e_array.shape[1])
                topk_size.append(topk)
                #topk_size.append(topk)
                if len(topk_lags) == 0:
                    e_layer_max.append(np.nan)
                    e_layer_med.append(np.nan)
                    continue
                
                all_e_big_e_array[k,:] = np.nansum([big_e_array[k,:], all_e_big_e_array[k,:]], axis = 0)
                layer_count[k] +=1
                #breakpoint()
                # median
                top_lags_med = np.nanmedian(topk_lags) #topk_lags[len(topk_lags)//2] # we don't want true median which can give fractional values) already sorted here
                e_layer_med.append(lshift*(top_lags_med-half_lags)) # no start/end here because want actual length
                #e_layer_R.append(big_e_array[k, top_lags_med])
                #max
                #breakpoint()
                topk_vals = [big_e_array[k, p] if p in topk_lags else 0 for p in range(len(big_e_array[k,:]))]
                top_lags_max = np.nanargmax(topk_vals)
                #if top_lags_max not in topk_lags:
                #    breakpoint()
                assert(top_lags_max in topk_lags)
                e_layer_max.append(lshift*(top_lags_max-half_lags)) # no start/end here because want actual length

            except ValueError:
                e_layer_max.append(np.nan)
                e_layer_med.append(np.nan)
                topk_size.append(np.nan)  
        all_e_layer_med.append(e_layer_med)
        all_e_layer_max.append(e_layer_max)
    # plot max lag over layers for this electrode
    fig, (ax, ax4, ax3, ax3n)= plt.subplots(1, 4, figsize=[40,10])
    plt.subplots_adjust(hspace=0.001, wspace=0.5)
    #plt.tight_layout()
    med_avg = np.nanmean(all_e_layer_med, axis=0)
    max_avg = np.nanmean(all_e_layer_max, axis=0)
    med_avg_nn = [med_avg[p] for p in range(len(med_avg)) if not np.isnan(med_avg[p])]
    if len(med_avg_nn) >= 2:
        med_lag_corr = pearsonr(np.arange(len(med_avg_nn)), med_avg_nn)[0]
    else:
        print('<2 vals')
        med_lag_corr = 0

    max_avg_nn = [max_avg[p] for p in range(len(max_avg)) if not np.isnan(max_avg[p])]
    if len(max_avg_nn) >= 2:
        max_lag_corr = pearsonr(np.arange(len(max_avg_nn)), max_avg_nn)[0]
    else:
        print('<2 vals')
        max_lag_corr = 0


    #med_lag_corr = np.round(pearsonr(np.arange(len(med_avg)), med_avg)[0], 3)
    #max_lag_corr = np.round(pearsonr(np.arange(len(max_avg)), max_avg)[0],3)
 
    ax.plot(np.arange(1,len(med_avg)+1), med_avg, '-o', markersize=2,color = 'blue', label='med lag for R > ' + str(cutoff_R))
    ax.plot(np.arange(1,len(max_avg)+1), max_avg, '-o', markersize=2,color = 'red', label='lag with top R')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set(xlabel='layer', ylabel='lag (ms)', title= in2[:-3] + ' Top Lag Per Layer (Avg\'d over Electrodes), (MedR, MaxR) = (' + str(med_lag_corr) + ', ' + str(max_lag_corr) + ')')
    #ax.set_ylim([-250, 250])
    ax.tick_params(axis='both', labelsize=16)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.title.set_fontsize(16)

    # add 1s because not starting at 0
    ax4.plot(np.arange(1,len(med_avg)+1), (med_avg-np.mean(med_avg))/np.std(med_avg), '-o', markersize=2,color = 'blue', label='med lag for R > ' + str(cutoff_R))
    ax4.plot(np.arange(1,len(max_avg)+1), (max_avg-np.mean(max_avg))/np.std(max_avg), '-o', markersize=2,color = 'red', label='lag with top R')
    ax4.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax4.set(xlabel='layer', ylabel='lag (ms)', title= in2[:-3] + ' Top Lag Per Layer Z-Score')
    #ax4.set_ylim([-2,2])
    ax4.tick_params(axis='both', labelsize=16)
    ax4.xaxis.label.set_fontsize(16)
    ax4.yaxis.label.set_fontsize(16)
    ax4.title.set_fontsize(16)

    # layer_count is number of non nan entries per layer
    # in previous plots you do nanmean to account for nans in average.
    # all_e_big_e_array is the sum for a given layer over all electrodes. we divide by the count here to average. to get the average correlation per layer 
    # across all electrodes
    for k in range(num_layers-num_back):
        #for k in range(num_layers-1): # partial
        if layer_count[k] == 0:
            print(k, 'FAIL')
        all_e_big_e_array[k,:] /= layer_count[k]
    #init_grey = 1
    #breakpoint()
    hues = list(np.arange(0, 240 + 240/(num_layers - 1), 240/(num_layers-1))/360) # red to blue
    # plot encoding for all layers for this electrode. 
    for j in range(num_layers - num_back):
        #for j in range(1, num_layers): #partial
        # input or first layer
        c = list(clr.hsv_to_rgb([hues[j], 1.0, 1.0]))
        #if j == 0:
        ax3.plot(lags, all_e_big_e_array[j], color=c, label='layer' + str(j+1),zorder=-1) #**
        ax3n.plot(lags, all_e_big_e_array[j]/np.max(all_e_big_e_array[j]), color=c, label='layer' + str(j+1),zorder=-1) #**

        #elif j == num_layers-1:
        #    #elif j == num_layers - 1: #partial
        #    #clr = 'r'
        #    ax3.plot(lags, all_e_big_e_array[j], color=clr, label='layer' + str(j+1), zorder=-1) #**
        #else:
        #    #init_grey -= 1/(math.exp((j)*0.001)*(num_layers+1))
        #    #clr = str(init_grey)
        #    ax3.plot(lags, all_e_big_e_array[j], color=clr,label='layer' + str(j+1),zorder=-1) #**
        #try:
        #    #ax3.scatter([lags[np.nanargmax(all_e_big_e_array[j])]], [np.nanmax(all_e_big_e_array[j])], color=c, zorder=-1)
        #except ValueError:
        #    continue
    #breakpoint()
    ax3.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize='x-small')
    ax3.set(xlabel='lag (ms)', ylabel='R', title= in_type[:-3] + ' Avg Electrode Encoding Per Layer (Left 2 Plots Come From Points Between Dashed Lines)')
    #ax3.set_ylim([0, 0.4])
    ax3.vlines([-500, 500], 0, 0.4, colors = ['grey'], linestyles=['dashed'])
    #breakpoint()
    ax3.set_xlim([-2000, 2000])
    ax3.tick_params(axis='both', labelsize=16)
    #breakpoint()
    ax3.xaxis.label.set_fontsize(16)
    ax3.yaxis.label.set_fontsize(16)
    ax3.title.set_fontsize(16)

    ax3n.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize='x-small')
    ax3n.set(xlabel='lag (ms)', ylabel='R', title= in_type[:-3] + ' Avg Electrode Encoding Per Layer (Left 2 Plots Come From Points Between Dashed Lines)')
    #ax3.set_ylim([0, 0.4])
    ax3n.vlines([-500, 500], 0, 0.4, colors = ['grey'], linestyles=['dashed'])
    #breakpoint()
    ax3n.set_ylim([0, 1.1])
    ax3n.set_xlim([-250, 250])
    ax3n.tick_params(axis='both', labelsize=16)
    #breakpoint()
    ax3n.xaxis.label.set_fontsize(16)
    ax3n.yaxis.label.set_fontsize(16)
    ax3n.title.set_fontsize(16)

    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/split2-jump1-l1-50_bbfdr_update_' + brain_area + '_top' + str(cutoff_P) + 'p_corr_threshold' + str(cutoff_R) + '_ws200_'+  in2 + '_' + str(slag)+ '_to_' + str(elag) + 'shuffle=False_SIG.png') 
    plt.close()
    return


def multiverse_plot(elec_list, omit_e, lags,lshift,num_layers,in2, cutoff_P, cutoff_Rs, slag, elag, brain_area):
    num_lags = len(lags)
    half_lags = num_lags//2
    start = math.floor(slag/lshift) +half_lags # reverse of below 161//2 = 80. 
    end = math.floor(elag/lshift) + half_lags

    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-ws200-shuffle-pval-' + in2
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-pval-' + in2
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in2 

    all_sig_R = extract_single_bigbigfdr_correlation(fpath, elec_list, list(np.arange(1,num_layers+1)), len(lags), True)
    
    init_grey = 1
    fig, ((medc, medl,medr), (maxc, maxl,maxr))= plt.subplots(2, 3, figsize=[40,25])
    
    #maxRl = maxl.twinx()
    #maxRl.set_ylabel('R')
    #medRl = medl.twinx()
    #medRl.set_ylabel('R')
    
    #colors = ['orange', 'cyan', 'blue', 'purple', 'green', 'red']
    maxt_cors = []
    medt_cors = []
    for q,t in enumerate(cutoff_Rs):
        print(t)
        #all_e_layer_min = []
        all_e_layer_med = []
        all_e_layer_max = []
        #all_medR = []
        #all_maxR = []
        #all_e_layer_max_lag = []
        #all25 = []
        #all75 = []
        all_e_big_e_array = np.zeros((num_layers, num_lags))
        layer_count = list(np.zeros(num_layers))


        for i, e in enumerate(elec_list):
            if e in omit_e:
                print('bad electrode found')
                continue
     
            #print(e)
            big_e_array = all_sig_R[i]

            #e_layer_min = []
            e_layer_med = []
            e_layer_max = []
            #medR = []
            #maxR = []
            #e_layer_max_lag = []
            #el25 = []
            #el75 = []
            topk_size = []
            for k in range(big_e_array.shape[0]):
                try:
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > t] # in general
                    
                    topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                    topk_lags = np.sort(thresholded_lags[-topk:])
                    topk_size.append(topk)
                    if len(topk_lags) == 0:
                        #e_layer.append(np.nan)
                        #e_layer_R.append(np.nan)
                        e_layer_max.append(np.nan)
                        #e_layer_min.append(np.nan)
                        e_layer_med.append(np.nan)
                        #e_layer_max_lag.append(np.nan)
                        #el25.append(np.nan)
                        #el75.append(np.nan)
                        #medR.append(np.nan)
                        #maxR.append(np.nan)

                        continue
                            
                    all_e_big_e_array[k,:] = np.nansum([big_e_array[k,:], all_e_big_e_array[k,:]], axis = 0)
                    layer_count[k] +=1
                    #top_lags_min = topk_lags[0]
                    #e_layer_min.append(25*(top_lags_min-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                            
                    # median
                    top_lags_med = np.nanmedian(topk_lags) #topk_lags[len(topk_lags)//2] # we don't want true median which can give fractional values) already sorted here
                    #medR.append(big_e_array[k,math.floor(top_lags_med)])
                    e_layer_med.append(25*(top_lags_med-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                    
                    #max
                    topk_vals = [big_e_array[k, p] if p in topk_lags else 0 for p in range(len(big_e_array[k,:]))]
                    top_lags_max = np.nanargmax(topk_vals)
                    #maxR.append(big_e_array[k,math.floor(top_lags_max)])
                    #breakpoint()
                    assert(top_lags_max in topk_lags)
                    e_layer_max.append(lshift*(top_lags_max-half_lags)) # no start/end here because want actual length
                except ValueError:
                    e_layer.append(np.nan) # append nan if all in column are nan
                    #e_layer_R.append(np.nan)
                    topk_size.append(np.nan)  
                    e_layer_max.append(np.nan)
                    e_layer_med.append(np.nan)
                    #medR.append(np.nan)
                    #maxR.append(np.nan)
            #all_e_layer_min.append(e_layer_min)
            all_e_layer_med.append(e_layer_med)
            all_e_layer_max.append(e_layer_max)
            #all_medR.append(medR)
            #all_maxR.append(maxR)
            #all_e_layer_max_lag.append(e_layer_max_lag)
            #all25.append(el25)
            #all75.append(el75)
        
        init_grey -= 1/(math.exp((q)*0.001)*(len(cutoff_Rs) + 1))
        
        med_avg = np.nanmean(all_e_layer_med, axis=0)
        max_avg = np.nanmean(all_e_layer_max, axis=0)

        medl.plot(np.arange(1,len(med_avg)+1), (med_avg-np.mean(med_avg))/np.std(med_avg), '-o', markersize=2,color = str(init_grey), label='med lag t=' + str(t))
        maxl.plot(np.arange(1,len(max_avg)+1), (max_avg-np.mean(max_avg))/np.std(max_avg), '-o', markersize=2,color = str(init_grey), label='max lag t=' + str(t))
        #breakpoint()
        #med_Ravg = np.nanmean(all_medR, axis=0)
        #max_Ravg = np.nanmean(all_maxR, axis=0)
        
        #medRl.plot(np.arange(1,len(med_Ravg)+1), med_Ravg, '-o', markersize=2,color = [init_grey,0,0], label='med R t=' + str(t))
        #maxRl.plot(np.arange(1,len(max_Ravg)+1), max_Ravg,'-o', markersize=2,color = [init_grey, 0, 0], label='max R t=' + str(t))

        medc.plot(np.arange(1,len(med_avg)+1),med_avg, '-o', markersize=2,color = str(init_grey), label='med lag t=' + str(t))
        maxc.plot(np.arange(1,len(max_avg)+1),max_avg, '-o', markersize=2,color = str(init_grey), label='max lag t=' + str(t))
        
        
        med_avg_nn = [med_avg[p] for p in range(len(med_avg)) if not np.isnan(med_avg[p])]
        if len(med_avg_nn) >= 2:
            med_avg_corr = pearsonr(np.arange(len(med_avg_nn)), med_avg_nn)[0]
        else:
            print('<2 vals')
            med_avg_corr = 0
        medt_cors.append(med_avg_corr)

        max_avg_nn = [max_avg[p] for p in range(len(max_avg)) if not np.isnan(max_avg[p])]
        if len(max_avg_nn) >= 2:
            max_avg_corr = pearsonr(np.arange(len(max_avg_nn)), max_avg_nn)[0]
        else:
            print('<2 vals')
            max_avg_corr = 0
        maxt_cors.append(max_avg_corr)        

        medr.scatter(t,med_avg_corr , s = 100, c= [str(init_grey)], zorder=1)
        
        maxr.scatter(t,max_avg_corr, s = 100, c= [str(init_grey)], zorder=1)

    # med left (top)
    medl.legend(bbox_to_anchor=(1,1), loc='upper left')
    medl.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' Med Z-Score')
    medl.set_ylim([-5, 5])
    medl.tick_params(axis='both', labelsize=16)
    medl.xaxis.label.set_fontsize(16)
    medl.yaxis.label.set_fontsize(16)
    medl.title.set_fontsize(16)

    #medRl.legend(bbox_to_anchor=(1,1), loc='lower left')
    
    #max left (bottom)
    maxl.legend(bbox_to_anchor=(1,1), loc='upper left')
    maxl.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' Max Z-Score')
    maxl.set_ylim([-5, 5])
    maxl.tick_params(axis='both', labelsize=16)
    maxl.xaxis.label.set_fontsize(16)
    maxl.yaxis.label.set_fontsize(16)
    maxl.title.set_fontsize(16)
    
    #maxRl.legend(bbox_to_anchor=(1,1), loc='lower left')

    medc.legend(bbox_to_anchor=(1,1), loc='upper left')
    medc.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' Med')
    medc.set_ylim([-250, 250])
    medc.tick_params(axis='both', labelsize=16)
    medc.xaxis.label.set_fontsize(16)
    medc.yaxis.label.set_fontsize(16)
    medc.title.set_fontsize(16)

    maxc.legend(bbox_to_anchor=(1,1), loc='upper left')
    maxc.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' Max')
    maxc.set_ylim([-250, 250])
    maxc.tick_params(axis='both', labelsize=16)
    maxc.xaxis.label.set_fontsize(16)
    maxc.yaxis.label.set_fontsize(16)
    maxc.title.set_fontsize(16)


    medr.plot(cutoff_Rs,medt_cors, '-o', markersize=2,color = 'k', label='med lag', zorder = -1)
    medr.set_ylim([0, 1.0])
    medr.set_ylabel('R')
    medr.set_title(in2 +  'Correlations for Lines in Plot to Left')
    medr.set_xlabel('Threshold')
    medr.tick_params(axis='both', labelsize=16)
    medr.xaxis.label.set_fontsize(16)
    medr.yaxis.label.set_fontsize(16)
    medr.title.set_fontsize(16)
    medr.legend(bbox_to_anchor=(1,1), loc='upper left')
    
    maxr.plot(cutoff_Rs,maxt_cors, '-o', markersize=2,color = 'k', label='max lag', zorder = -1)
    maxr.set_ylim([0, 1.0])
    maxr.set_ylabel('R')
    maxr.set_title(in2 +  'Correlations for Lines in Plot to Left')
    maxr.set_xlabel('Threshold')
    maxr.tick_params(axis='both', labelsize=16)
    maxr.xaxis.label.set_fontsize(16)
    maxr.yaxis.label.set_fontsize(16)
    maxr.title.set_fontsize(16)
    maxr.legend(bbox_to_anchor=(1,1), loc='upper left')
    sdir = '/scratch/gpfs/eham/247-encoding-updated/results/figures/'
    fig.savefig(sdir + 'multiverse_git_' + brain_area + '_top' + str(cutoff_P) + '_ws200_'+ in2 + '_' + str(slag)+ '_to_' + str(elag) + '_SIG.png')

def multiverse_roi_plot(omit_e, lags,lshift,num_layers,in2, cutoff_P, slag, elag, cutoff_R = 0.15):
    rois = ['ifg', 'all','mSTG','aSTG','depth','TP','MFG','AG','preCG','MTG']
    plot_rois = ['ifg', 'all','mSTG','aSTG','depth','TP','MFG','AG','preCG','MTG']
    colors = ['orange', 'cyan', 'blue', 'purple', 'green', 'red', 'black', 'magenta', 'grey', 'pink']
    num_lags = len(lags)
    half_lags = num_lags//2
    start = math.floor(slag/lshift) +half_lags # reverse of below 161//2 = 80. 
    end = math.floor(elag/lshift) + half_lags

    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-ws200-shuffle-pval-' + in2
    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-pval-' + in2
    fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-' + in_type[:-3] + '-pca50d-full-git-ws200-pval-' + in2 
    
    init_grey = 1
    fig, ((medc, medl,medr), (maxc, maxl,maxr))= plt.subplots(2, 3, figsize=[40,25])
    plt.subplots_adjust(hspace=0.3, wspace=0.5)
    #maxRl = maxl.twinx()
    #maxRl.set_ylabel('R')
    #medRl = medl.twinx()
    #medRl.set_ylabel('R')
    
    maxt_cors = []
    medt_cors = []
    for q, r in enumerate(rois):
        print(r)
        e_list = get_e_list(r + '_e_list.txt', '\t')
        all_sig_R = extract_single_bigbigfdr_correlation(fpath, e_list, list(np.arange(1,num_layers+1)), len(lags), True)
        #breakpoint()
        #all_e_layer_min = []
        all_e_layer_med = []
        all_e_layer_max = []
        #all_medR = []
        #all_maxR = []
        #all_e_layer_max_lag = []
        #all25 = []
        #all75 = []
        all_e_big_e_array = np.zeros((num_layers, num_lags))
        layer_count = list(np.zeros(num_layers))


        for i, e in enumerate(e_list):
            if e in omit_e:
                print('bad electrode found')
                continue
     
            #print(e)
            big_e_array = all_sig_R[i]

            #e_layer_min = []
            e_layer_med = []
            e_layer_max = []
            #medR = []
            #maxR = []
            #e_layer_max_lag = []
            #el25 = []
            #el75 = []
            topk_size = []
            for k in range(big_e_array.shape[0]):
                try:
                    top_lags = np.argsort(big_e_array[k,:], axis=-1)
                    top_lags = [top_lags[p] for p in range(len(top_lags)) if top_lags[p] >= start and top_lags[p] <= end]
                    
                    top_lags_nonan = [top_lags[p] for p in range(len(top_lags)) if not np.isnan(big_e_array[k, top_lags[p]])]
                    
                    thresholded_lags = [top_lags_nonan[p] for p in range(len(top_lags_nonan)) if big_e_array[k, top_lags_nonan[p]] > cutoff_R] # in general
                    
                    topk = math.floor(np.minimum(cutoff_P*big_e_array.shape[1], len(thresholded_lags))) # if > 20% of lags are nan, use all of nonan. else use 20%. 
                    topk_lags = np.sort(thresholded_lags[-topk:])
                    topk_size.append(topk)
                    if len(topk_lags) == 0:
                        #e_layer.append(np.nan)
                        #e_layer_R.append(np.nan)
                        e_layer_max.append(np.nan)
                        #e_layer_min.append(np.nan)
                        e_layer_med.append(np.nan)
                        #e_layer_max_lag.append(np.nan)
                        #el25.append(np.nan)
                        #el75.append(np.nan)
                        #medR.append(np.nan)
                        #maxR.append(np.nan)

                        continue
                            
                    all_e_big_e_array[k,:] = np.nansum([big_e_array[k,:], all_e_big_e_array[k,:]], axis = 0)
                    layer_count[k] +=1
                    #top_lags_min = topk_lags[0]
                    #e_layer_min.append(25*(top_lags_min-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                            
                    # median
                    top_lags_med = np.nanmedian(topk_lags) #topk_lags[len(topk_lags)//2] # we don't want true median which can give fractional values) already sorted here
                    #medR.append(big_e_array[k,math.floor(top_lags_med)])
                    e_layer_med.append(25*(top_lags_med-len(big_e_array[k,:])//2)) # no start/end here because want actual length
                    
                    #max
                    topk_vals = [big_e_array[k, p] if p in topk_lags else 0 for p in range(len(big_e_array[k,:]))]
                    top_lags_max = np.nanargmax(topk_vals)
                    #maxR.append(big_e_array[k,math.floor(top_lags_max)])
                    #breakpoint()
                    assert(top_lags_max in topk_lags)
                    e_layer_max.append(lshift*(top_lags_max-half_lags)) # no start/end here because want actual length
                except ValueError:
                    e_layer.append(np.nan) # append nan if all in column are nan
                    #e_layer_R.append(np.nan)
                    topk_size.append(np.nan)  

                    e_layer_max.append(np.nan)
                    e_layer_med.append(np.nan)
                    #medR.append(np.nan)
                    #maxR.append(np.nan)
            #all_e_layer_min.append(e_layer_min)
            all_e_layer_med.append(e_layer_med)
            all_e_layer_max.append(e_layer_max)
            #all_medR.append(medR)
            #all_maxR.append(maxR)
            #all_e_layer_max_lag.append(e_layer_max_lag)
            #all25.append(el25)
            #all75.append(el75)
        
        #init_grey -= 1/(math.exp((q)*0.001)*(len(rois) + 1))
        
        med_avg = np.nanmean(all_e_layer_med, axis=0)
        max_avg = np.nanmean(all_e_layer_max, axis=0)
        if r == 'aTG':
            out_label = 'TP'
        elif r == 'pTG':
            out_label = 'MTG'
        elif r == 'occipital':
            out_label = 'AG'
        else:
            out_label= r
        medl.plot(np.arange(1,len(med_avg)+1), (med_avg-np.mean(med_avg))/np.std(med_avg), '-o', markersize=2,color =colors[q], label=out_label)
        maxl.plot(np.arange(1,len(max_avg)+1), (max_avg-np.mean(max_avg))/np.std(max_avg), '-o', markersize=2,color = colors[q], label=out_label)
        #breakpoint()
        #med_Ravg = np.nanmean(all_medR, axis=0)
        #max_Ravg = np.nanmean(all_maxR, axis=0)
        
        #medRl.plot(np.arange(1,len(med_Ravg)+1), med_Ravg, '-o', markersize=2,color = [init_grey,0,0], label='med R t=' + str(t))
        #maxRl.plot(np.arange(1,len(max_Ravg)+1), max_Ravg,'-o', markersize=2,color = [init_grey, 0, 0], label='max R t=' + str(t))

        medc.plot(np.arange(1,len(med_avg)+1),med_avg, '-o', markersize=2,color = colors[q], label=out_label)
        maxc.plot(np.arange(1,len(max_avg)+1),max_avg, '-o', markersize=2,color = colors[q], label= out_label)
        #breakpoint()
        
        med_avg_nn = [med_avg[p] for p in range(len(med_avg)) if not np.isnan(med_avg[p])]
        if len(med_avg_nn) >= 2:
            med_avg_corr = pearsonr(np.arange(len(med_avg_nn)), med_avg_nn)[0]
        else:
            print('<2 vals')
            med_avg_corr = 0
        medt_cors.append(med_avg_corr)

        max_avg_nn = [max_avg[p] for p in range(len(max_avg)) if not np.isnan(max_avg[p])]
        if len(max_avg_nn) >= 2:
            max_avg_corr = pearsonr(np.arange(len(max_avg_nn)), max_avg_nn)[0]
        else:
            print('<2 vals')
            max_avg_corr = 0
        maxt_cors.append(max_avg_corr)        

        medr.scatter(q,med_avg_corr , s = 100, c= [colors[q]], zorder=1)
        
        maxr.scatter(q,max_avg_corr, s = 100, c= [colors[q]], zorder=1)
        print(r, ' done')

    
    # med left (top)
    medl.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=20)
    medl.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' ROI Med Z-Score')
    medl.set_ylim([-5, 5])
    medl.tick_params(axis='both', labelsize=20)
    medl.xaxis.label.set_fontsize(24)
    medl.yaxis.label.set_fontsize(24)
    medl.title.set_fontsize(24)

    #medRl.legend(bbox_to_anchor=(1,1), loc='lower left')
    
    #max left (bottom)
    maxl.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=20)
    maxl.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' ROI Max Z-Score')
    maxl.set_ylim([-5, 5])
    maxl.tick_params(axis='both', labelsize=20)
    maxl.xaxis.label.set_fontsize(24)
    maxl.yaxis.label.set_fontsize(24)
    maxl.title.set_fontsize(24)
    
    #maxRl.legend(bbox_to_anchor=(1,1), loc='lower left')

    medc.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=20)
    medc.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' ROI Med')
    medc.set_ylim([-250, 250])
    medc.tick_params(axis='both', labelsize=20)
    medc.xaxis.label.set_fontsize(24)
    medc.yaxis.label.set_fontsize(24)
    medc.title.set_fontsize(24)

    maxc.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=20)
    maxc.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' ROI Max')
    maxc.set_ylim([-250, 250])
    maxc.tick_params(axis='both', labelsize=20)
    maxc.xaxis.label.set_fontsize(24)
    maxc.yaxis.label.set_fontsize(24)
    maxc.title.set_fontsize(24)

    medr.plot(plot_rois,medt_cors, '-o', markersize=2,color = 'k', zorder = -1)
    medr.set_ylim([0, 1.0])
    medr.set_ylabel('R')
    medr.set_title(in2 +  'Correlations for Lines in Plot to Left')
    medr.set_xlabel('Threshold')
    medr.tick_params(axis='both', labelsize=20)
    medr.xaxis.label.set_fontsize(24)
    medr.yaxis.label.set_fontsize(24)
    medr.title.set_fontsize(24)
    medr.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=20)
    
    maxr.plot(plot_rois,maxt_cors, '-o', markersize=2,color = 'k', zorder = -1)
    maxr.set_ylim([0, 1.0])
    maxr.set_ylabel('R')
    maxr.set_title(in2 +  'Correlations for Lines in Plot to Left')
    maxr.set_xlabel('Threshold')
    maxr.tick_params(axis='both', labelsize=20)
    maxr.xaxis.label.set_fontsize(24)
    maxr.yaxis.label.set_fontsize(24)
    maxr.title.set_fontsize(24)
    maxr.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=20)
    sdir = '/scratch/gpfs/eham/247-encoding-updated/results/figures/'
    fig.savefig(sdir + 'multiverse_update_all_roi_top' + str(cutoff_P) + '_ws200_'+ in2 + '_' + str(slag)+ '_to_' + str(elag) + '_SIG.png')

               #
# divide words into two sets based on how much they change throughout the model. 
# run avg analyses on these sets separately
#def changing_word_analysis():

def lag_layer_top_median(elec_list,omit_e, lags, num_layers, in2, cutoff_P):
    num_lags = len(lags)
    for i, e in enumerate(elec_list):
        if e in omit_e:
            continue
        print(e)
        big_e_array = np.zeros((num_layers, num_lags))
        for layer in range(1, num_layers+1):
            ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-pval-' + in2  + str(layer) + '/*')
                        
            big_e_array[layer-1] = extract_single_sig_correlation(ldir, e)
        
        topk = math.ceil(cutoff_P*(num_lags)) 
        e_layer = []
        #e_layer_R = []
        for j in range(big_e_array.shape[0]):
            try:
                # get top cuttof_P percent
                #breakpoint()
                top_lags = np.argsort(big_e_array[j,:], axis=-1)
                top_lags_nonan = [top_lags[k] for k in range(len(top_lags)) if not np.isnan(big_e_array[j, top_lags[k]])]
                topk_lags = top_lags_nonan[-topk:]
                # if topk_lags is empty (all nan) then just put in nan.
                if len(topk_lags) == 0:
                    e_layer.append(np.nan)
                    continue
                top_lags_med = np.nanmedian(topk_lags)
                e_layer.append(25*(top_lags_med-len(big_e_array[j,:])//2))
                #e_layer_R.append(max_R)
                 
            except ValueError:
                e_layer.append(np.nan) # append nan if all in column are nan
                #e_layer_R.append(np.nan)
        
        
 
        #breakpoint() 
        # plot max lag over layers for this electrode
        fig, (ax, ax2, ax3, ax4) = plt.subplots(1,4,figsize=[50,20])
    
        ax.plot(np.arange(1, len(e_layer) + 1), e_layer, '-o', markersize=2,color = 'orange', label='top lag')
            
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        ax.set(xlabel='layer', ylabel='lag (s)', title= in2 + ' top lag per layer for electrode ' + e)
        ax.set_ylim([-500, 1000])
        ax.tick_params(axis='both', labelsize=16)
        fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/median_lagplayer_' + in2 + '_' + e + '_SIG.png')
        plt.close()
# for each electrode, for each layer, gets the 
#def avg_corr_analysis():
def plot_layers(num_layers, in_type, elec, corr_type):
    if elec == '742_G64':
        print('bad electrode, stop')
        return
    fig, (ax, ax2, ax3) = plt.subplots(1,3, figsize=[30, 10])
    lags = np.arange(-2000, 2001, 25)
    init_grey = 1 
    #max_cors = []
    #zero_cors = []
    #lag_300_cors = []
    #lag_n300_cors = []
    m0 = []
    #m250 = []
    m500 = []
    #p250 = []
    p500 = []
    m1000 = []
    p1000 = []
    lag_avg = [] 
    #print(corr_type)
    for i in range(1,num_layers+1):
        #breakpoint()
        ldir = glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-'+in_type[:-3] +  '-hs' + str(i) + '/*')
        #breakpoint()
        if corr_type == 'all':
            layer = extract_correlations(ldir)
        elif corr_type == 'all_sig':
            layer = extract_avg_sig_correlations(ldir)
        elif corr_type == 'single_sig':
            if i == 1: print('single sig')
            layer = extract_single_sig_correlation(ldir, elec)
        else: 
            layer = extract_single_correlation(ldir, elec)
        #max_cors.append(np.max(layer))
        #zero_cors.append(layer[len(layer)//2])
        #lag_300_cors.append(layer[len(layer)//2 + 12]) # 12*25 = 300. so 12 right from 0 is 300ms
        #lag_n300_cors.append(layer[len(layer)//2 + -12]) # 12*25 = 300. so 12 right from 0 is 300ms
        l0 = len(layer)//2
        m1000.append(layer[l0 - 40])
        m500.append(layer[l0 - 20])
        #m250.append(layer[l0 - 10])
        m0.append(layer[l0])
        #p250.append(layer[l0 + 10])
        p500.append(layer[l0 + 20])
        p1000.append(layer[l0 + 40])
        #lag_avg.append(np.mean(layer))
        #rgb = np.random.rand(3,)
        #init_grey /= 1.25

        if i == 0:
            ax.plot(lags, layer, color='b', label='layer' + str(i)) #**
        elif i == num_layers:
            ax.plot(lags, layer, color='r', label='layer' + str(i)) #**
        else:
            init_grey -= 1/(math.exp((i-1)*0.001)*(num_layers+1))
            ax.plot(lags, layer, color=str(init_grey), label='layer' + str(i)) #**
    
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    ax.set_ylim([0, 0.4])
    ax.set(xlabel='lag (s)', ylabel='correlation', title= in_type + ' Encoding Over Layers ' + elec)
    ax.grid()
    #fig.savefig("/scratch/gpfs/eham/247-encoding-updated/results/figures/comparison_new_" + in_type +'_' + elec +  str(num_layers) + "where_shift.png")
    #fig.savefig("comparison_old_p_weight_test.png")
    #plt.show()
    #plt.close() 
    #fig2, ax2 = plt.subplots() #figure()
    #plt.plot(range(len(max_cors)), max_cors, '-o', color='r', label='max')
    #breakpoint()
    c = 1.2
    init_grey = 1/c
    
    ax2.plot(range(len(m0)), m1000, '-o',color= (0, init_grey,0), label='-1000ms')
    init_grey /= c
    ax2.plot(range(len(m0)), m500, '-o',color= (0, init_grey,0), label='-500ms')
    #init_grey /= c
    #ax2.plot(range(len(m0)), m250, '-o',color= (0, init_grey, 0), label = '-250ms')
    init_grey /=c
    ax2.plot(range(len(m0)), m0, '-o', color=(0, init_grey, 0), label='0ms')
    init_grey /=c 
    #ax2.plot(range(len(m0)), p250, '-o',color= (0, init_grey, 0), label='250ms')
    #init_grey /=c 
    ax2.plot(range(len(m0)), p500, '-o',color= (0, init_grey, 0), label = '500ms')
    init_grey /= c
    ax2.plot(range(len(m0)), p1000, '-o',color= (0, init_grey,0), label='1000ms')
    #ax2.plot(range(len(m0)), lag_avg, '-o', color = (1, 0, 0), label='avg over lags') 
    #plt.title('Corr vs depth')
    #plt.xlabel('Layer')
    #plt.ylabel('R')
    ax2.set(xlabel='Layer', ylabel='R', title='Corr vs Depth ' + elec)
    ax2.set_ylim([0, 0.4])
    ax2.legend()
    #plt.legend()
    #fig2.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/Corr_vs_Depth' +in_type +'_' +  elec + '.png')
    # brain plot
    #breakpoint()
    if elec != 'all':
        brain_im = get_brain_im(elec)
        ax3.imshow(brain_im)


    fig.tight_layout()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/All_Combined_range' + in_type + '_' + elec + '_new.png')
    plt.close()
    #return m1000, m500, m250, m0, p250, p500, p1000


if __name__ == '__main__':
    #plot_layers(11, 'key')
    # averagei
    in_type = 'gpt2-xl-hs'
    #lags = np.arange(-2000, 2001, 25)
    lshift = 25 
    lags = np.arange(-2000, 2001, lshift)
    num_lags = len(lags)
    in2 = 'gpt2-xl-hs'
    num_layers = get_n_layers(in2[:-3])
    brain_as = ['ifg', 'all','mSTG','aSTG','depth','TP','MFG','AG','preCG','MTG']
    #brain_x = brain_as[1]
    #print(brain_x)
    #e_list = get_e_list(brain_x + '_e_list.txt', '\t')
    #e_list = get_e_list('brain_map_input.txt', '\t')
    #e_list = get_e_list('old_test.txt', ',')
    #print(len(e_list))
    #print(e_list)
    
    omit_e = []
    with open('omit_e_list.txt','r') as f:
        for line in f:
            omit_e.append(line.strip('\n').strip(' '))
    #omit_e = [] # test all electrodes
    #breakpoint()
    #plot_layers(num_layers, 'gpt2-xl-hs', 'all')
    ph = False 
    #plot_electrodes(num_layers, in_type, in2, e_list,omit_e, num_lags, ph)
    #plot_sig_electrodes(num_layers, in_type, in2, e_list,omit_e, lags, ph)
    #plot_sig_lags_for_layers(num_layers,in_type,in2,e_list,omit_e,lags,ph, 0.15)
    #plot_heatmap_sig_lags_for_layers(num_layers,in_type,in2,e_list,omit_e,lags,ph)
    #plot_layers(48, in_type, 'all')
    #layered_encoding_plot(num_layers, in2, 'all_sig', e_list, omit_e,True)
    #layered_encoding_plus_sig_lags_plot(e_list, omit_e,lags, num_layers, in2, 'max',0.15)
    #layered_encoding_plus_sig_lags_plot(e_list/, omit_e,lags, num_layers, in2, 'all_p',0.2, 0.17, -500, 500)
    #for brain_a in brain_as:
    #    #print(brain_a)
    #    #avg_lag_layer_plot_updated(e_list, omit_e,lags, num_layers,in2, 'all_p', 0.2, 0.17, -500, 500, brain_a)
    #    #print(brain_a, ' done')
    # regular
    num_back = 0
    bbfdr = True
    #avg_lag_layer_plot_updated(e_list, omit_e,lags,lshift, num_layers,in2, 'all_p', 1.0, 0.0, -500, 500, brain_x,bbfdr, 0)
    cutoff_Rs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    #multiverse_plot(e_list, omit_e, lags,lshift,num_layers,in2, 1.0, cutoff_Rs, -500, 500, brain_x)
    #multiverse_roi_plot(omit_e, lags,lshift,num_layers,in2, 1.0, -500, 500, cutoff_R = 0.15)
    #avg_lag_layer_nocorr(e_list, omit_e, lags,lshift,num_layers,in2, 1.0, 0.15, -500, 500, brain_x, num_back = 0)
    
    #spatiotemporal_heatmaps(lags, lshift, e_list, num_layers, in2,1.0, .15, omit_e)
    #elec_brain_plot(lags, lshift, num_layers, in2, 1.0, 0.15, omit_e)
    '''
    sampling = 0
    sampling_amt = 5
    e2v = sep_roi_stats(lags, lshift, num_layers, in2, 1.0, 0.15, omit_e, sampling, sampling_amt)
    in_f = os.path.join(os.getcwd(), 'brain_map_input.txt')
    omit_e = ['742_G64']
    #breakpoint()
    save_e2l(e2v, in_f, 'std_per_region', 'gpt2-xl', omit_e)
    '''
    #derivative_plot('mSTG', 'TP', lags, lshift, num_layers, in2, 1.0, 0.15, omit_e)
    #omit_e = []
    #modular_lag_layer_nocorr(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x)

    #modular_encoding_nov_update(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x)
    #modular_encoding_lag_layer_nov_update(e_list, omit_e, lags, lshift, num_layers, in2, 1.0, 0.15, -500, 500, brain_x)
    #paper_plots(omit_e, lags, lshift, num_layers, in2, 1.0, 0.0, -500, 500, 'current_all')
    print('omit:\n',omit_e)
    omit_e = []
    paper_plots(omit_e, lags, lshift, num_layers, in2, 1.0, 0.0, -500, 500, 'ifg')
    #elec_list = []
    #for val in e_list:
    #    if val not in omit_e:
    #        elec_list.append(val)
    #        print(val)
    #breakpoint()
    #breakpoint()
    #plot_glove_encoding(elec_list,lags)
    #lag_layer_nocorr(e_list, omit_e, lags,lshift,num_layers,in2, 1.0, 0.0, -2000, 2000, brain_x)

    #fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-'+ in_type[:-3] + '-pca50d-full-aug2021-pval-' + in2
    #extract_single_bigbigfdr_correlation(fpath, e_list, list(np.arange(1,num_layers+1)), len(lags))
    #num_back = 2 # regress out of layer before previous
    #num_layers = num_layers - num_back
    #avg_lag_layer_plot_updated(e_list, omit_e,lags, num_layers,in2, 'all_p', 1.0, 0.15, -500, 500, brain_x, num_back)


     #lag_layer_top_median(e_list, omit_e,lags, num_layers, in2, 0.2)
    #for elec in e_list:
    #    plot_layers(48, in2, str(elec), 'single_sig')
    '''
    # individual electrodes
    e_list = get_e_list('brain_map_input.txt')
    e_lags = [[],[],[],[],[],[],[]]
    labels = ['-1000ms', '-500ms', '-250ms', '0ms', '250ms', '500ms', '1000ms']
    in_type = 'gpt2-xl-hs'
    for elec in e_list: 
        print(elec)
        #get_brain_im(elec)
        m1000, m500, m250, m0, p250, p500, p1000 = plot_layers(48, in_type, str(elec))
        e_lags[0].append(m1000)
        e_lags[1].append(m500)
        e_lags[2].append(m250)
        e_lags[3].append(m0)
        e_lags[4].append(p250)
        e_lags[5].append(p500)
        e_lags[6].append(p1000)
    
    fig, ax2 = plt.subplots()
    c = 1.3
    init_grey = 1/c
    #plot_clr = [.9, .1, 0]
    for i, lag in enumerate(e_lags):
        avg = np.mean(lag, axis = 0)
        #ax2.plot(range(len(avg)), avg, '-o',color= (0, init_grey,0), label=labels[i])
        if i > len(e_lags)//2:
            ax2.plot(range(len(avg)), avg, '-o',color= (init_grey, 0, 0), label=labels[i])
            init_grey *= c
        elif i == len(e_lags)//2:
            ax2.plot(range(len(avg)), avg, '-o',color= (0, 0, 0), label=labels[i])
        elif i < len(e_lags)//2:
            ax2.plot(range(len(avg)), avg, '-o',color= (0, init_grey, 0), label=labels[i])
            init_grey /= c
 
        #ax2.plot(range(len(avg)), avg, '-o',color= (plot_clr[0], plot_clr[1], 0), label=labels[i])
        #plot_clr[0] /= 1.2
        #plot_clr[1] *= 1.2
        #init_grey /= c
    
    ax2.set(xlabel='Layer', ylabel='R', title='Corr vs Depth ' + elec)
    ax2.legend()
    fig.savefig('/scratch/gpfs/eham/247-encoding-updated/results/figures/test2_LayerPlot_avg_electrodes_' + in_type + '.png')
    plt.close()
    #max_l2_grad = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-max-l2-grad/*')
    #l2_grad = extract_correlations(max_l2_grad)
    '''
    #paper_test = extract_correlations(glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-gpt2-xl-lm_out_bobbi2_test/*'))
    #norm_test = extract_correlations(glob.glob('/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-gpt2-xl-norm_test/*'))
    #fig, ax = plt.subplots()
    #lags = np.arange(-2000, 2001, 25)
    #ax.plot(lags,norm_test, 'r', label='contextual')
    ##ax.plot(lags,paper_test, 'r', label='contextual')
    #ax.legend()
    #ax.set(xlabel='lag (s)', ylabel='correlation', title='Here it is, max: ' + str(np.max(norm_test)))
    #ax.grid()

    #fig.savefig("/scratch/gpfs/eham/247-encoding-updated/results/figures/norm_test_gpt2-xl.png")
#

    #topn_list = [0, 3, 5]
    #w_list = ['reg','pmint', 'pw']
    #for topn in topn_list:
    #    for wtype in w_list:
    #        emb, true, rest, full = get_signals(topn, wtype)
    #        all_rest_true_plots(wtype, topn, emb, true, rest, full)
    ''' 
    reg_true = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-reg-true0-update-no-norm/*')
    r_true = extract_correlations(reg_true)

    reg_all = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-reg-all0-update-no-norm/*')
    r_all = extract_correlations(reg_all)

    reg_rest = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-reg-rest0-update-no-norm/*')
    r_rest = extract_correlations(reg_rest)
    '''
    #emb = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-emb-emb0-update-no-norm/*')
    #emb_v = extract_correlations(emb)

    #all_rest_true_plots('reg-real-no-norm', 0, emb_v, r_true, r_rest, r_all)

    #true_abs = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-abs-abs0-update-no-norm/*') 
    #true_abs_v = extract_correlations(true_abs)

    #sum_abs = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-abs_sum-abs_sum0-update-no-norm/*')
    #sum_abs_v = extract_correlations(sum_abs)

    #rest_abs = true_abs_v

    #all_rest_true_plots('abs', 0, emb_v, true_abs_v, rest_abs, sum_abs_v)
    #sgd_emb = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-train-emb/*'))
    #sgd_rest = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-sgd-rest/*'))
    #sgd_true = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-sgd-true/*'))
    #sgd_all = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-sgd-all/*'))
    #adam_rest = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-adam-rest/*'))
    #adam_true = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-adam-true/*'))
    #adam_all = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-adam-all/*'))
    #eval_rest = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-eval-rest/*'))
    #eval_true = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-eval-true/*'))
    #eval_all = extract_correlations(glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-eval-all/*'))


    #all_rest_true_plots('sgd', 0, sgd_emb, sgd_true, sgd_rest, sgd_all)
    #all_rest_true_plots('eval', 0, emb_v, eval_true, eval_rest, eval_all)
    
'''
top1_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-top1-pca-new/*')

#python_dir_list = glob.glob(os.path.join(os.getcwd(), 'test-NY*'))
top1_mean_corr = extract_correlations(top1_dir_list)

#matlab_dir_list = glob.glob(os.path.join(os.getcwd(), 'NY*'))
#m_mean_corr = extract_correlations(matlab_dir_list)

w_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-weight-pca-new/*')
w_mean_corr = extract_correlations(w_dir_list)

top1w_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-top1_weight-pca-new/*')
top1w_mean_corr = extract_correlations(top1w_dir_list)

dLdC_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-dLdC/*')
dLdC_mean_corr = extract_correlations(dLdC_dir_list)

wpw_dir_list =  glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-wpw/*')
wpw_mean_corr = extract_correlations(wpw_dir_list)

concat_dLdC_true = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-concat-dLdC-true/*')
concat_dLdC_t_mean_corr = extract_correlations(concat_dLdC_true)

one_over_pmt = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-one_over_pmt/*')
one_over_pmt_mean_corr = extract_correlations(one_over_pmt)



#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pw/*')
#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-old-pw-test/*')
#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pw-nopca/*')
#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-reg-no-norm/*')
# no norm
#p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pw-no-norm-pca/*')
# re norm
p_weight_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-pw-pca2/*')
p_mean_corr = extract_correlations(p_weight_dir_list)

#no_w_avg = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-avg/*')
#no_w_avg = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-avg-nopca/*')
# no norm
#no_w_avg = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-avg-no-norm-pca/*')
# re norm
no_w_avg = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-avg-pca/*') 
no_w_avg_mean_corr = extract_correlations(no_w_avg)

#pmint_w = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pmint/*')
#pmint_w = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pmint-nopca/*')
# no norm
#pmint_w = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-pmint-no-norm-pca/*')
# re norm
pmint_w = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-pmint-pca2/*') 
pmint_w_mean_corr = extract_correlations(pmint_w)

#true_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-true/*')
#true_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-true-nopca/*')
# no norm
#true_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-true-no-norm-pca/*')
# re norm
true_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-true-pca/*') 
true_mean_corr = extract_correlations(true_dir_list)

#reg_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-emb/*')
#reg_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-emb-nopca/*')
# no norm
#reg_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-new-emb-no-norm-pca/*')
# re norm
reg_dir_list = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-emb-pca/*') 
reg_mean_corr = extract_correlations(reg_dir_list)

# verify if you take no norm pw, normalize it, you get an effect
#pw_norm_ver = glob.glob('/scratch/gpfs/eham/podcast-encoding/results/no-shuffle-verify-norm-change-pw-pca/*')
#pw_norm_ver_corr = extract_correlations(pw_norm_ver)

fig, ax = plt.subplots()
lags = np.arange(-2000, 2001, 25)
ax.plot(lags, true_mean_corr, 'r', label='true grad')
#ax.plot(lags, pw_norm_ver_corr, 'b', label='norm pw')
#ax.plot(lags, top1_mean_corr, 'b', label='top1 grad') #**
ax.plot(lags, reg_mean_corr, 'k', label='contextual') #**
#ax.plot(lags, w_mean_corr, 'g', label='true weight') #**
#ax.plot(lags, top1w_mean_corr, 'orange', label = 'top1 weight')
ax.plot(lags, p_mean_corr, 'orange', label='p weighted') #**
#ax.plot(lags, dLdC_mean_corr, 'purple', label='dLdC')  #**
#ax.plot(lags, wpw_mean_corr, 'magenta', label = 'wpw') #**
#ax.plot(lags, m_mean_corr, 'r', label='matlab')
#ax.plot(lags, concat_dLdC_t_mean_corr, 'plum', label = 'concatdLdCtrue') #**
#ax.plot(lags, one_over_pmt_mean_corr, 'chartreuse', label = 'grad weight 1/(p-t)') #**
ax.plot(lags, no_w_avg_mean_corr, 'burlywood', label = 'uniform avg')
ax.plot(lags, pmint_w_mean_corr, 'lightcoral', label = 'p-t weighted')
ax.legend()
ax.set(xlabel='lag (s)', ylabel='correlation', title='Here it is')
ax.grid()

fig.savefig("comparison_new_no_norm_pca2.png")
#fig.savefig("comparison_old_p_weight_test.png")
plt.show()
Aii'''

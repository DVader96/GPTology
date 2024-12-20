#------------------------------------------------#
# Prepares text file input for matlab brain plot #
# Author: Eric Ham                               #
#------------------------------------------------#

import os
import pandas as pd
import numpy as np
#import matplotlib.colors as clr
from statsmodels.stats.multitest import fdrcorrection as fdr
import argparse
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clr
# get number of layers from the model type
def get_n_layers(mt):
    if mt == 'gpt2-xl':
        nl = 48 # 48 blocks (0 to 47). then input is first.
    elif mt == 'gpt2':
        nl = 12 # 12 blocks ...
    elif mt == 'gpt2-large':
        nl = 36 # 36 blocks 
    elif mt == 'bert':
        nl = 24
    elif mt == 'llama':
        nl = 33
    return nl

# get all electrodes to look through
def get_e_list(in_f, split_token):
    e_list = []
    sep = '_'
    for line in open(in_f, 'r'):
        items = line.strip('\n').split(split_token) # '\t' or ',' depending on input
        if items[0] == '742' and items[1] == 'G64':
            continue
        e_list.append(sep.join(items[:2]))    
    return e_list

# get the top layer for each electrode
def get_e2l(e_list, num_layers, model_type, lag, threshold):
    path_s = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-' + str(model_type) + '-hs'
    # go through all electrodes
    print(lag)
    e2l  = {}
    for e in e_list:
        print(e)
        l_corrs = []
        p_vals = []
        for l in range(num_layers):
            e_dir = path_s + str(l) + '/777/' + e + '_comp.csv'
            #e_sig = pd.read_csv(e_dir + e + '_comp.csv')
            #with open(e_dir, 'r') as e_f:
            #    e_sig = list(map(float, e_f.readline().strip().split(',')))
            #if e == '717_LGA10':
            #    breakpoint()
            #breakpoint()
            e_sig = pd.read_csv(e_dir)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(e_dir, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            e_Ps = list(e_f.loc[1])
            #breakpoint()
            e_Qs = fdr(e_Ps, alpha=0.01, method='i')[1] # use bh method
            l_corrs.append(e_Rs[len(e_Rs)//2 + lag])
            p_vals.append(e_Qs[len(e_Ps)//2 + lag])
        # if p value low enought, then significant.
        max_l = np.argmax(l_corrs)
        #breakpoint()
        if p_vals[max_l]  <= threshold:
            e2l[e] = max_l
        else:
            #breakpoint()
            e2l[e] = -1 # this means not significant correlation

    return e2l

def get_e2lag_max(e_list, num_layers,model_type, threshold):
    path_s = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-' + str(model_type) + '-hs'
    #top_lag_per_e = []
    e2lag = {}
    for e in e_list:
        #big_e_array = np.zeros((num_layers, num_lags)) 
        top_lag_per_layer = []
        top_lag_val_per_layer = []
        p_vals = []
        for l in range(num_layers):
            e_dir = path_s + str(l) + '/777/' + e + '_comp.csv'
            e_sig = pd.read_csv(e_dir)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(e_dir, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            e_Ps = list(e_f.loc[1])
            #breakpoint()
            e_Qs = fdr(e_Ps, alpha=0.01, method='i')[1] # use bh method
            top_lag_per_layer.append(25*(np.argmax(e_Rs)-80))
            top_lag_val_per_layer.append(e_Rs[np.argmax(e_Rs)])
            #l_corrs.append(e_Rs[len(e_Rs)//2 + lag])
            #p_vals.append(e_Qs[len(e_Ps)//2 + lag])
            p_vals.append(e_Qs[np.argmax(e_Rs)])
        max_lag = top_lag_per_layer[np.argmax(top_lag_val_per_layer)]
        #breakpoint()
        if p_vals[np.argmax(top_lag_val_per_layer)]  <= threshold:
            e2lag[e] = max_lag
        else:
            #breakpoint()
            e2lag[e] = -1 # this means not significant correlation

 
    return e2lag

def get_e2lag_med(e_list, num_layers,model_type, threshold):
    path_s = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-pval-' + str(model_type) + '-hs'
    #top_lag_per_e = []
    e2lag = {}
    for e in e_list:
        all_lags = []
        #p_vals = []
        for l in range(num_layers):
            e_dir = path_s + str(l) + '/777/' + e + '_comp.csv'
            e_sig = pd.read_csv(e_dir)
            sig_len = len(e_sig.loc[0])
            e_f = pd.read_csv(e_dir, names = range(sig_len))
            e_Rs = list(e_f.loc[0])
            e_Ps = list(e_f.loc[1])
            e_Qs = fdr(e_Ps, alpha=0.01, method='i')[1] # use bh method
            lags = []
            #breakpoint()
            for p in range(len(e_Rs)):
                # add if above threshold and significant
                if e_Rs[p] > 0.15 and e_Qs[p] <= threshold:
                    lags.append(p)
                else:
                    continue
            #breakpoint()
            # accumulate all layers into one vector
            if len(lags) != 0 and 25*(np.max(lags)-80)>1500:
                print(e, 25*(np.max(lags)-80), e_Rs[np.max(lags)])
            lags = 25*(np.array(lags) - 80) 
            all_lags.extend(lags)
        # handle if no R over threshold or significant
        #breakpoint()
        if len(all_lags) == 0:
            e2lag[e] = -1
        else:
            #out_lag = np.median(all_lags)
            #out_lag = np.min(all_lags)
            out_lag = np.max(all_lags)
            #print(e
            #out_lag = np.percentile(all_lags, 25)
            #out_lag = np.percentile(all_lags, 75)
            e2lag[e] = out_lag
 
    return e2lag



# get layer --> color dictionary
def get_dicts(num_gs, layers):
    #num_layers = len(layers)
    nl_pg = int(num_layers/num_layers) #make >1 for grouping layers
    layers = list(range(num_layers))
    #possible_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #g2c = {} # group to color
    #l2g = {} # layer to group
    #init_grey = 1
    l2c = {}
    #breakpoint()
    hues = list(np.arange(0, 240 + 240/(num_gs - 1), 240/(num_gs-1))/360) # red to blue
    breakpoint()
    for i in range(num_gs):
        #init_grey -= 1/(math.exp(i*0.0001)*(num_gs + 1))
        #g2c[i] = clr.to_rgb(possible_colors[i])
        #l2g.update(dict(zip(layers[i*nl_pg:(i+1)*nl_pg], np.ones(nl_pg)*i)))    
        #l2c.update(dict(zip(layers[i*nl_pg:(i+1)*nl_pg], np.repeat(np.array([clr.to_rgb(possible_colors[i])]),nl_pg ,axis=0)))) 
        #breakpoint()
        l2c.update(dict(zip(layers[i*nl_pg:(i+1)*nl_pg], np.repeat(np.array([clr.hsv_to_rgb([hues[i], 1.0, 1.0])]),nl_pg ,axis=0)))) 
    #return l2g, g2c
    # set white as color for not significant electrodes
    #breakpoint()
    # plot and save colorbar
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    #cmap = mpl.cm.cool
    #breakpoint()
    cmap = clr.LinearSegmentedColormap.from_list('custom',[list(l2c[i]) for i in range(len(l2c))])
    norm = mpl.colors.Normalize(vmin=0, vmax=num_layers-1)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,orientation='horizontal')
    cb1.set_label('Layers')
    fig.savefig('cbar_' + str(num_layers) + '.png')
    l2c[-1] = clr.to_rgb('w')
    return l2c

# get electrode --> color dictionary
def get_e2c(e2l, l2c):
    e2c = {}
    for e, l in e2l.items():
        e2c[e] = l2c[l]
    #breakpoint()
    return e2c
 
def save_e2c(e2c, in_f, lag, model_type, omit_e):
    out_f = open('brain_map_text_out_lag' + str(lag) + '_' + model_type + '_Qsig_e2c_hsv.txt', 'w')
    sep = '\t'
    for line in open(in_f, 'r'):
        items = line.strip('\n').split('\t')
        print(sep.join(items[1:]))
        #breakpoint()
        if '_'.join(items[:2]) in omit_e:
            out_f.write(sep.join(items[1:]) + '\t0.0 0.0 0.0\n')
        else:
            #breakpoint()
            out_f.write(sep.join(items[1:]) + '\t' + sep.join(list(map(str,e2c['_'.join(items[:2])]))) + '\n')
    
    out_f.close()

def save_e2l(e2l, in_f, lag, model_type, omit_e):
    out_f = open('brain_map_text_out_lag' + str(lag) + '_' + model_type + '_Qsig_e2l.txt', 'w')
    sep = '\t'
    count = 1
    #omit_out = open('brain_map_text_out_lag' + str(lag) + '_' + model_type + '_Qsig_e2l_to_omit.txt','w')
    #omit_out.write('[')
    #omit = 0
    for line in open(in_f, 'r'):
        items = line.strip('\n').split('\t')
        print(sep.join(items[1:]))
        if '_'.join(items[:2]) == '717_LOF4':
            breakpoint()
        
        if '_'.join(items[:2]) in omit_e or '_'.join(items[:2]) not in e2l or e2l['_'.join(items[:2])] == -1: 
            #out_f.write(sep.join(items[1:]) + '\t0.0\n')
            #if omit == 0:
            #    omit_out.write(str(count))
            #    omit = 1
            #else:
            #    omit_out.write(';' + str(count))
            continue
        else:
            #breakpoint()
            print(e2l['_'.join(items[:2])])
            out_f.write(sep.join(items[1:]) + '\t' + str(e2l['_'.join(items[:2])]) + '\n')
        count +=1
    #omit_out.write(']')
    #omit_out.close()
    out_f.close()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lag', type=int, default=0)
    args = parser.parse_args()
    return args

# color electrodes according to the layer that maximizes their correlation
def top_layer_analysis(args):
    in_f = os.path.join(os.getcwd(), 'brain_map_input.txt')
    e_list = get_e_list(in_f, '\t')
    # get top layer for each electrode
    model_type = 'gpt2-xl'
    num_layers = get_n_layers(model_type)
    lag = args.lag
    adjusted_lag = int(lag/25) #this adjusts for binning  
    threshold = 0.01
    e2l = get_e2l(e_list, num_layers, model_type, adjusted_lag, threshold)
    # divide layers into groups and assign colors 
    #num_gs = 4 # up to 7 (see line below) num groups
    #save_e2l = 

    num_gs = num_layers
    #layer2group, group2color = get_dicts(num_gs, num_layers)    
    layer2color = get_dicts(num_gs, np.arange(1, num_layers + 1))
    # assign color to electrode
    e2c = get_e2c(e2l, layer2color)    
    # save output    
    
    #breakpoint()
    save_e2c(e2c, in_f, lag, model_type)

# color electrodes according to the lag that maximizes their correlation
def top_lag_analysis():
    in_f = os.path.join(os.getcwd(), 'brain_map_input.txt')
    e_list = get_e_list(in_f, '\t')
    new_e_list = []
    omit_e = []
    with open('omit_e_list.txt','r') as f:
        for line in f:
            omit_e.append(line.strip('\n').strip(' '))
   
    #breakpoint()
    new_e_list = [val for val in e_list if val not in omit_e]
    
    # get top layer for each electrode
    #num_lags = 161
    model_type = 'gpt2-xl'
    num_layers = get_n_layers(model_type)
    #lag = args.lag
    #adjusted_lag = int(lag/25) #this adjusts for binning  
    threshold = 0.01
    #e2l = get_e2l(e_list, num_layers, model_type, adjusted_lag, threshold)
    #e2lag = get_e2lag_max(e_list, num_layers,model_type, threshold)
    e2lag = get_e2lag_med(e_list, num_layers,model_type, threshold)

    # divide layers into groups and assign colors 
    #num_gs = 4 # up to 7 (see line below) num groups
    #save_e2l = 

    #num_gs = num_lags
    #layer2group, group2color = get_dicts(num_gs, num_layers)    
    #lag2color = get_dicts(num_lags, np.arange(-2000, 2001, 25))
    # assign color to electrode
    #breakpoint()
    #e2c = get_e2c(e2lag, lag2color)    
    # save output    
    #save_e2c(e2c, in_f, 'all', model_type, omit_e)

    #breakpoint()
    save_e2l(e2lag, in_f,'all_max', model_type, omit_e)

def plot_region(roi):
    in_f = os.path.join(os.getcwd(), 'brain_map_input.txt')
    e_list = get_e_list(roi + '_e_list.txt', '\t')
    
    out_f = open('brain_map_roi=' + roi + '.txt', 'w')
    sep = '\t'
    color = ['1', '0', '0']
    #breakpoint()
    for line in open(in_f, 'r'):
        items = line.strip('\n').split('\t')

        if '_'.join(items[:2]) not in e_list:
            continue
        print(sep.join(items[1:]))
        #if '_'.join(items[:2]) in omit_e:
        #    out_f.write(sep.join(items[1:]) + '\t0.0 0.0 0.0\n')
        #else:
        #breakpoint()
        #breakpoint()
        out_f.write(sep.join(items[1:]) + '\t' + sep.join(color) + '\n')
    
    out_f.close()


if __name__ == '__main__':
    args = parse_args()
    #top_layer_analysis(args) 
    #top_lag_analysis()
    plot_region('pton1_TP')
    plot_region('pton1_mSTG')
    plot_region('pton1_aSTG')
    plot_region('nyu_rSTG')
    plot_region('nyu_cSTG')
    plot_region('nyu_mSTG')


#------------------------------------------------------#
# file for post processing embeddings after generation #
# post processing is before encoding                   #
# Author: Eric Ham                                     #
#------------------------------------------------------#

from utils import load_pickle, save_pickle
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import os
import math 
from scipy.stats import pearsonr

from brain_color_prepro import get_n_layers, get_e_list
from paper_plots import get_params, preprocess_encoding_results

import matplotlib.pyplot as plt

def get_max_layer(ID, wv, rois):
    print(ID)
    print(wv)
    print(rois)
    model = 'gpt2-xl'
    lshift = 25 # ms between time points
    lags = np.arange(-2000, 2001, lshift)
    max_layers = []
    for roi in rois:
        print(roi)
        elec_list = get_e_list(roi+ '_e_list.txt', '\t')
        print(elec_list)
        ps = get_params(elec_list, model, roi, wv)
        fpath = '/scratch/gpfs/eham/247-encoding-updated/results/podcast2/podcast2-gpt2-xl-pca50d-full-' + ID + '-' + wv + '-gpt2-xl-hs'
        lel = preprocess_encoding_results(fpath, elec_list, ps) # layer x electrode x lag = lel
        aoe = np.max(lel, axis=1) # avg over electrodes
        max_l = np.argmax(np.max(aoe, axis=-1)) + 1 # goes from 0 to 47, but our layers are 1 to 48 in names
        max_layers.append(max_l)
    return max_layers

# project y from x: project x onto y, then subtract that (component of x in direction of y) from y
def project_out(x, y):
    #if np.all(np.isclose(x,y)):
    #    breakpoint()
    # scalar projection
    #sp = np.dot(x, y/np.linalg.norm(y))
    #proj = sp*y/np.linalg.norm(y)
    #scale = np.dot(x,y)/(np.linalg.norm(y)**2) # doesn't give 1 if x == y because dot and norm give diff # decimals
    scale = np.dot(x,y)/np.dot(y,y) # norm(y)**2 is same as dot(y,y)
    out = x - y*scale 
    return out

def test_projection(la, mla):
    # test 1: no change
    breakpoint()
    r1 = project_out(la[i], mla[i])
    print(np.dot(r1, mla[i]))
    print(pearsonr(r1, mla[i]))

    # test 2: centered 
    la -= np.expand_dims(np.mean(la, axis=1), axis=1)
    mla -= np.expand_dims(np.mean(mla, axis=1), axis=1)
    r2 = project_out(la[i], mla[i])
    print(np.dot(r2, mla[i]))
    print(pearsonr(r2, mla[i]))
    
    # test 3: standardized
    la/=np.expand_dims(np.std(la, axis=1), axis=1)
    mla/=np.expand_dims(np.std(mla, axis=1), axis=1)
    r3 = project_out(la[i], mla[i])
    print(np.dot(r3, mla[i]))
    print(pearsonr(r3, mla[i]))
    print('mean, std')
    print(np.mean(r3), np.std(r3))
    r4 = r3/np.std(r3)
    print(np.dot(r4, mla[i]))
    print(pearsonr(r4, mla[i]))

# output is pickle per layer where max for that roi is projected from all layers.
def project_out_max(maxl, wv, out_ID, roi, centered):
    print(maxl)
    num_layers = get_n_layers('gpt2-xl')
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    sfx = '_gpt2-xl_cnxt_1024_embeddings.pkl'
    if wv == 'all':
        max_layer =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(maxl)+sfx))
    else:
        max_layer =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(maxl)+sfx))
    breakpoint()
    mla= np.array(list(map(np.array, max_layer['embeddings'])))
    
    if centered == True:
        mla -= np.expand_dims(np.mean(mla, axis=1), axis=1)
    not_close = []
    for l in range(1, num_layers+1):
        if wv == 'all':
            layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(l)+ sfx))
        else:
            layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(l)+ sfx))
        
        la = np.array(list(map(np.array, layer['embeddings'])))
        breakpoint()
        if centered == True:
            la -= np.expand_dims(np.mean(la, axis=1), axis=1)
        #print(l, la.shape)
        # fully orthogonal (each result is orthogonal to the max)
        # project embedding for each word in max from each word in layer
        # get diff size for correct incorrect all
        out_emb = np.zeros_like(la)
        #assert(out_emb.shape[0] == 4744 and out_emb.shape[1] == 1600)
        #breakpoint()
        #if l == maxl: breakpoint() 
        for i in range(la.shape[0]):
            #if l == maxl: breakpoint()
            #test_projection(la, mla)
            result = project_out(la[i], mla[i]) # NOTE
            #proj = np.dot(la[i], mla[i])*mla[i]/(np.linalg.norm(mla[i])**2)
            #result = la[i] - proj
            #assert(np.isclose(np.dot(result, mla[i]),0))
            #print(maxl, ' ', l, ' ', str(np.dot(result, mla[i])))
            if not math.isclose(np.dot(result, mla[i]), 0, abs_tol=1e-7):
                breakpoint()
                not_close.append(np.dot(result, mla[i]))
            # verify decorrelated if centered
            if centered == True and l != maxl and not math.isclose(pearsonr(result, mla[i])[0], 0, abs_tol=1e-7):
                print('not uncorrelated')
                breakpoint()
            out_emb[i,:] = result

        #breakpoint()
        layer['embeddings'] = list(out_emb)
        #reg = LinearRegression()
        #reg.fit(out_emb, mla)
        #breakpoint()
        #print(reg.score(out_emb, mla))

        # save the pickle
        save_pickle(layer.to_dict('records'), os.path.join(pdir, '777_full-' + wv + '-'+ roi + '-maxout-' + out_ID + '-PAPER2-hs' + str(l) + sfx))
    #breakpoint()
    #print(np.max(not_close))

# use max layer to try to predict all other layers
# subtract this from other layers (remove information linearly predictable in layer from max layer)
def reg_out_max(maxl, wv, out_ID, roi):
    print(maxl)
    num_layers = get_n_layers('gpt2-xl')
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    sfx = '_gpt2-xl_cnxt_1024_embeddings.pkl'
    if wv == 'all':
        max_layer =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(maxl)+sfx))
    else:
        max_layer =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(maxl)+sfx))
    #breakpoint()
    mla= np.array(list(map(np.array, max_layer['embeddings'])))
     
    not_close = []
    for l in range(1, num_layers+1):
        if wv == 'all':
            layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(l)+ sfx))
        else:
            layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(l)+ sfx))
        la = np.array(list(map(np.array, layer['embeddings'])))
        reg = LinearRegression()
        reg.fit(mla, la)
        print(reg.score(mla, la))
        resid = la - reg.predict(mla)
        layer['embeddings'] = list(resid)
 
        save_pickle(layer.to_dict('records'), os.path.join(pdir, '777_full-' + wv + '-'+ roi + '-maxout-' + out_ID + '-PAPER2-hs' + str(l) + sfx))

# lin reg project out. project out but with 1 matrix. 
# in project out you remove vector corresponding to max from each embedding of hte layer. 
# here, you do an imperfect version of this. you learn a mapping from all embeddings in the layer to all embeddings 
# in the max layer (1 matrix). you then remove this
def halfway_out_max(maxl, wv, out_ID, roi):
    print(maxl)
    num_layers = get_n_layers('gpt2-xl')
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    sfx = '_gpt2-xl_cnxt_1024_embeddings.pkl'
    if wv == 'all':
        max_layer =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(maxl)+sfx))
    else:
        max_layer =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(maxl)+sfx))
    #breakpoint()
    mla= np.array(list(map(np.array, max_layer['embeddings'])))
     
    not_close = []
    for l in range(1, num_layers+1):
        if wv == 'all':
            layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(l)+ sfx))
        else:
            layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(l)+ sfx))
        la = np.array(list(map(np.array, layer['embeddings'])))
        reg = LinearRegression()
        reg.fit(la, mla)
        print(reg.score(la, mla))
        resid = la - reg.predict(la)
        layer['embeddings'] = list(resid)
        breakpoint()
        print('test')
        #save_pickle(layer.to_dict('records'), os.path.join(pdir, '777_full-' + wv + '-'+ roi + '-maxout-' + out_ID + '-PAPER2-hs' + str(l) + sfx))



def partial_out_max(maxl):
    num_layers = get_n_layers('gpt2-xl')
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    sfx = '_gpt2-xl_cnxt_1024_embeddings.pkl'
    # NOTE: CHANGE
    #wv = 'correct'
    #wv = 'top5-incorrect'
    wv = 'all'
    if wv == 'all':
        max_layer =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(maxl)+sfx))
    else:
        max_layer =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(maxl)+sfx))
    not_close = []
    for l in range(1, num_layers+1):
        # NOTE: CHANGE
        if wv == 'all':
            layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(l)+ sfx))
        else:
            layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(l)+ sfx))
        # regress max from layer
        #reg = LinearRegression()
        #breakpoint()
        la = np.array(list(map(np.array, layer['embeddings'])))
        mla= np.array(list(map(np.array, max_layer['embeddings'])))
        # option `: bad
        #reg.fit(mla, la)
        #print(reg.score(mla, la))
        #resid = la - reg.predict(mla)
        #layer['embeddings'] = list(resid)
        
        # semi-orthogonal (one matrix for all embeddings)
        #reg.fit(la, mla)
        #print(reg.score(la, mla))
        #result = la - reg.predict(la) 
        #layer['embeddings'] = list(result)

        # fully orthogonal (each result is orthogonal to the max)
        #breakpoint()
        out_emb = np.zeros_like(la)
        for i in range(la.shape[0]):
            #sp = np.dot(la[i], mla[i]/np.linalg.norm(mla[i]))# scalar projection
            #proj = sp*(mla[i]/np.linalg.norm(mla[i]))
            proj = np.dot(la[i], mla[i])*mla[i]/(np.linalg.norm(mla[i])**2)
            result = la[i] - proj
            #assert(np.isclose(np.dot(result, mla[i]),0))
            if not math.isclose(np.dot(result, mla[i]), 0, abs_tol=1e-7):
                breakpoint()
                not_close.append(np.dot(result, mla[i]))
            out_emb[i,:] = result

        layer['embeddings'] = list(out_emb)
        # save the pickle
        # NOTE: change
        ##save_pickle(layer.to_dict('records'), os.path.join(pdir, '777_full-' + wv + '-minusmax-PAPER2-hs' + str(l) + sfx))
        save_pickle(layer.to_dict('records'), os.path.join(pdir, '777_full-' + wv + '-truortho-PAPER2-hs' + str(l) + sfx))
    breakpoint()
    print(np.max(not_close))

def partial_out_prev(model, num_back):
    num_layers = get_n_layers(model)
    # NOTE: we cannot do for layer 1 (nothing to partial out but the input). start at layer 2.
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    #breakpoint()
    print(num_back)
    for i in range(num_back+1, num_layers):
        # get layers
        layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-'+model+'hs'+str(i)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))
        prev_layer = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-'+model+'hs'+str(i-num_back)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))
        print(i, i-num_back)

        # regress out the previous layer (get the residual from partial correlation function)
        reg = LinearRegression()
        
        la = np.array(list(map(np.array, layer['embeddings'])))
        pla= np.array(list(map(np.array, prev_layer['embeddings'])))
        reg.fit(pla, la)
        #print('fit done')
        print(reg.score(pla, la))
        continue
        # breakpoint()
        resid = la - reg.predict(pla)
        layer['embeddings'] = list(resid)
        # save the pickle
        save_pickle(layer.to_dict('records'), os.path.join(pdir, '777_full-' + str(model) + '-prev' + str(num_back) + '_layer_regressed_out_of_hs' + str(i) + '_gpt2-xl_cnxt_1024_embeddings.pkl'))

#averages sets of 8 layers. (6 sets) partials out in order
# 5 - 0, 5-1, 5-2, 5-3, 5-4
# 4 - 0, 4-1, 4-2, 4-3
# 3 -0, 3-1, 3-2
# 2 - 0, 2-1
# 1-0
# then saves results
# you can then run encoding. 
def avg_partial_out(model):
    num_layers = get_n_layers(model)
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    nlpg = 8 # num layers per group
    layer_avgs = []
    og = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + model + 'hs1_gpt2-xl_cnxt_1024_embeddings.pkl'))
    for i in range(int(num_layers/nlpg)):
        #print('!', i, '!')
        layer_sum = 0
        for j in range(1 + nlpg*i,1 + nlpg*(i + 1)):
            #print(j)
            layer_sum += np.array(list(map(np.array, pd.DataFrame.from_dict(load_pickle(pdir + '777_full-'+model+'hs'+str(j)+'_gpt2-xl_cnxt_1024_embeddings.pkl'))['embeddings'])))
        layer_avgs.append(layer_sum/nlpg)
    breakpoint()
    for i in range(1, int(num_layers/nlpg)):
        print('!', i, '!')
        for j in range(i):
            print(j)
            reg = LinearRegression()
            Y = layer_avgs[i]
            X = layer_avgs[j]
            reg.fit(X, Y)
            print(reg.score(X,Y))
            resid = Y - reg.predict(X)
            og['embeddings'] = list(resid)
            # save the pickle
            save_pickle(og.to_dict('records'), os.path.join(pdir, '777_full-' + str(model) + '-group-' +str(j) + '-outofgroup-' + str(i) + '_gpt2-xl_cnxt_1024_embeddings.pkl'))



# regress X from Y
def lin_reg_out(X, Y):
    reg = LinearRegression()
        
    X = np.array(list(map(np.array, X)))
    Y = np.array(list(map(np.array,Y))) 
    reg.fit(X, Y)
    print(reg.score(X, Y))
    pred = reg.predict(X)
    #resid = Y - reg.predict(X)
    return Y-pred, pred

# create matrix (n layers x n layers) where ij --> average (over words) of correlations
# ex. 12 --> (correlate layer 3 word 1 with layer 2 word 1, ... layer 1 word n with layer 2 word n.).mean()
# note: correlation is symmetric, so if matrix is A, Aij == Aji
def embedding_correlation_matrix(wv):
    print('pairwise over emb dims')
    num_layers = get_n_layers('gpt2-xl')
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    sfx = '_gpt2-xl_cnxt_1024_embeddings.pkl'
    corr_mat = np.zeros((num_layers, num_layers))
    for l1 in range(1, num_layers+1):
        if wv == 'all':
            layer1 = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(l1)+ sfx))
        else:
            layer1 = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(l1)+ sfx))
        
        la1 = np.array(list(map(np.array, layer1['embeddings'])))
        for l2 in range(l1, num_layers+1):
            if wv == 'all':
                layer2 = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(l2)+ sfx))
            else:
                layer2 = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(l2)+ sfx))
        
            la2 = np.array(list(map(np.array, layer2['embeddings'])))
            avg_corr = 0
            all_corrs = []
            for w in range(la1.shape[0]): 
                all_corrs.append(pearsonr(la1[w], la2[w])[0])
            corr = np.mean(all_corrs)
            print(l1, l2, corr)
            if l1 == l2: assert(np.isclose(corr,1.0))
            corr_mat[l1 - 1, l2 - 1] = corr
    
    breakpoint()
    l1_to_l48 = corr_mat[0, -1]
    zero_diag = corr_mat - np.diag(np.diag(corr_mat))
    corr_u = np.mean(zero_diag[np.nonzero(zero_diag)])
    corr_mat = corr_mat + np.transpose(corr_mat) - np.diag(np.ones(num_layers)) # get full matrix. diagonal is duplicated (all 1 because pearsonr(x, x)[0] = 1
    plt.figure()
    plt.imshow(corr_mat)
    plt.colorbar()
    plt.title('Average: ' + str(corr_u))
    plt.savefig('per_word_corr_mat_' + wv + '.png')

    return corr_mat

# same as above but fit linear model from embedding 1 to 2. then predict and compute correlation between true and predicted
def mixed_embedding_correlation_matrix(wv, rand=False):
    print('mixed')
    num_layers = get_n_layers('gpt2-xl')
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    sfx = '_gpt2-xl_cnxt_1024_embeddings.pkl'
    corr_mat = np.zeros((num_layers, num_layers))
    if rand == True: rand_mat = np.random.randn(num_layers, 4744, 1600)
    for l1 in range(1, num_layers+1):
        if wv == 'all':
            layer1 = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(l1)+ sfx))
        else:
            layer1 = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(l1)+ sfx))
        la1 = np.array(list(map(np.array, layer1['embeddings'])))
        if rand == True: la1 = rand_mat[l1-1]
        for l2 in range(l1, num_layers+1):
            if wv == 'all':
                layer2 = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(l2)+ sfx))
            else:
                layer2 = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(l2)+ sfx))
        
            la2 = np.array(list(map(np.array, layer2['embeddings'])))
            
            if rand == True: la2 = rand_mat[l2-1]
            reg = LinearRegression()
            reg.fit(la1, la2)
            out = reg.predict(la1)
            avg_corr = 0
            all_corrs = []
            for w in range(out.shape[0]): 
                all_corrs.append(pearsonr(out[w], la2[w])[0])
            corr = np.mean(all_corrs)
            print(l1, l2, corr)
            if l1 == l2: assert(np.isclose(corr,1.0))
            corr_mat[l1 - 1, l2 - 1] = corr
    
    breakpoint()
    l1_to_l48 = corr_mat[0, -1]
    zero_diag = corr_mat - np.diag(np.diag(corr_mat))
    #plt.figure()
    plt.imshow(corr_mat)
    corr_u = np.mean(zero_diag[np.nonzero(zero_diag)])
    corr_mat = corr_mat + np.transpose(corr_mat) - np.diag(np.ones(num_layers)) # get full matrix. diagonal is duplicated (all 1 because pearsonr(x, x)[0] = 1
    #plt.figure()
    #plt.imshow(corr_mat)
    #plt.colorbar()
    #plt.title('Average: ' + str(corr_u))
    #plt.savefig('per_word_corr_mat_' + wv + '.png')
    return corr_mat

# per roi (max layer differs) fit a regression model on all words on the max layer for the given wv
# then compute residuals for words in the subset defined by wv (c/i/a) for all layers
# save residuals
def regress_out_max_new(rois, max_layer_per_roi, wv, out_ID):
    num_layers = get_n_layers('gpt2-xl')
    pdir = '/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/'
    sfx = '_gpt2-xl_cnxt_1024_embeddings.pkl'
    print(wv)
    for i in range(len(rois)):
        maxl = max_layer_per_roi[i]
        roi = rois[i]
        print(roi, maxl)
        max_layer =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(maxl)+sfx))
        mla= np.array(list(map(np.array, max_layer['embeddings'])))

        for l1 in range(1, num_layers+1):
            layer1 = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-paper-gpt2-xlhs'+str(l1)+ sfx))
            la1 = np.array(list(map(np.array, layer1['embeddings'])))
                
            reg = LinearRegression()
            reg.fit(mla, la1)
            if l1 == maxl: breakpoint() 
            if wv == 'all':
                #breakpoint()
                resid = la1 - reg.predict(mla)
            else:
                max_layer_subset = pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(maxl)+sfx))
                mla_s = np.array(list(map(np.array, max_layer_subset['embeddings'])))
                layer_subset =  pd.DataFrame.from_dict(load_pickle(pdir + '777_full-' + wv + '-PAPER2-hs'+str(l1)+sfx))
                la_s = np.array(list(map(np.array, layer_subset['embeddings'])))
                resid = la_s - reg.predict(mla_s)
            
            # ensure embedding dims are ddecorrelated across words
            if l1 != maxl:
                for i in range(resid.shape[1]):
                    #assert(math.isclose(pearsonr(resid[:,i], mla[:,i])[0], 0, abs_tol=1e-7))
                    if wv == 'all':
                        if not math.isclose(pearsonr(resid[:,i], mla[:,i])[0], 0, abs_tol=1e-7): breakpoint()
                    else: 
                        if not math.isclose(pearsonr(resid[:,i], mla_s[:,i])[0], 0, abs_tol=1e-7): breakpoint()
            print(resid.shape)
            out_df = layer1.copy(deep=True)       
            out_df['embeddings'] = list(resid)
            #save_pickle(out_df.to_dict('records'), os.path.join(pdir, '777_full-' + wv + '-'+ roi + '-maxout-' + out_ID + '-PAPER2-hs' + str(l1) + sfx))

if __name__ == '__main__':
    #num_back = 20
    #partial_out_prev('gpt2-xl', num_back)    
    #avg_partial_out('gpt2-xl') 
    #partial_out_max(22)
    #wv = 'correct'
    #wv = 'top5-incorrect'
    #wv_for_max = 'all'
    #wv_for_max = 'correct'
    #wv_for_max = 'top5-incorrect'
    #ID = 'truortho'
    #wv = 'correct'
    #rois = ['nyu_ifg', 'pton1_mSTG', 'pton1_aSTG']
    #rois = ['pton1_TP'])
    #rois = ['nyu_ifg', 'pton1_mSTG', 'pton1_aSTG']
    #max_layer_per_roi = get_max_layer(ID, wv_for_max, rois)
    #breakpoint() 
    #max_layer_per_roi = get_max_layer(ID, 'correct', rois)
    #print(max_layer_per_roi)
    word_values = ['all', 'correct', 'top5-incorrect']
    #word_values = ['top5-incorrect']
    #word_values = ['all']
    # regress out
    # project out
    ID = 'PAPER2'
    #word_values = ['top5-incorrect']
    #word_values = ['correct', 'top5-incorrect']
    #word_values = ['all']#, 'correct', 'top5-incorrect']
    rois = ['pton1_TP', 'nyu_ifg', 'pton1_mSTG', 'pton1_aSTG']
    #rois = ['pton1_mSTG']#, 'pton1_aSTG']
    out_ID = 'regress-out-take2'
    for wv in word_values:
        print(wv)
        #max_layer_per_roi = get_max_layer(ID, wv, rois)
        
        print(rois)
        #print(max_layer_per_roi)

        #regress_out_max_new(rois, max_layer_per_roi, wv, out_ID)
    breakpoint()
    #for wv in word_values:
    #    print(wv)
    #    max_layer_per_roi = get_max_layer(ID, wv, rois)
    #    print(max_layer_per_roi)
    #breakpoint()
    for wv in word_values:
        print(wv)
        max_layer_per_roi = get_max_layer(ID, wv, rois)
        print(max_layer_per_roi)
        for i, roi in enumerate(rois):
            print(roi)
            project_out_max(max_layer_per_roi[i], wv, 'truortho-decorr', roi, True) # project out + decorrelate
            #project_out_max(max_layer_per_roi[i], wv, 'truortho', roi, False) # project out
            
            #reg_out_max(max_layer_per_roi[i], wv, 'lin-reg', roi) # project out
            #halfway_out_max(max_layer_per_roi[i], wv, 'half-out', roi) # project out
    


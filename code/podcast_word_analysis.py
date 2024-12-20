from utils import load_pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')  
nltk.download('universal_tagset')

from word_level_analysis import ci_split_uptodate_func
# analyze word list w/ nltk
# NOTE: Don't use this! the problem is the context is broken when you split into correct + incorrect so some homonyms may be misclassified!!!
def get_word_types(word_list):
    breakpoint()
    og_size = len(word_list)
    word_list = [word for word in word_list if word != "gonna" and word !="wanna"]
    word_types =nltk.pos_tag(word_tokenize(" ".join(word_list)), 'universal')
    counts = np.unique(np.array(word_types)[:,-1], return_counts=True) 
    assert(np.sum(counts[1]) == len(word_list))
    for i,cnt in enumerate(counts[1]):
        print(counts[0][i], ': ', round(cnt*100/og_size,1))

# NOTE: Don't use this! the problem is the context is broken when you split into correct + incorrect so some homonyms may be misclassified!!!
def get_ci_word_types():
   
    correct = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-correct-PAPER2-hs38_gpt2-xl_cnxt_1024_embeddings.pkl'))
    clist = list(correct.token2word)
    print('Correct: ')
    print(len(clist))
    get_word_types(clist)
       
    incorrect = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-top5-incorrect-PAPER2-hs38_gpt2-xl_cnxt_1024_embeddings.pkl'))
    ilist = list(incorrect.token2word)
    print('Incorrect: ')
    print(len(ilist))
    get_word_types(ilist)


# NOTE: USE THIS
def get_pos_info(df):
    og_size = len(df)
    print(og_size)
    pos_list = [df.POS[i] for i in range(len(df)) if df.word[i] != "gonna" and df.word[i] !="wanna"] # filter out 'n/a's 
    counts = np.unique(pos_list, return_counts=True)
    ids = list(counts[0])
    nums = list(counts[1])
    #breakpoint()
    assert(np.sum(counts[1]) == len(pos_list))
    if 'X' in ids:
        nums[np.nonzero('X' == np.array(ids))[0][0]] += og_size - len(pos_list)
    else: 
        ids.append('X')
        nums.append(og_size - len(pos_list))
    for i,cnt in enumerate(nums):
        print(ids[i], ': ', round(cnt*100/og_size,1))

# NOTE: USE THIS. Accounts for context before splitting into correct/incorrect, which disrupts context
def get_podcast_word_types():
    # load df
    allwords = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-paper-gpt2-xlhs38_gpt2-xl_cnxt_1024_embeddings.pkl'))
    
    # get parts of speech
    #breakpoint()
    alist = list(allwords.token2word)
    word_list = [word for word in alist if word != "gonna" and word !="wanna"]
    word_types =nltk.pos_tag(word_tokenize(" ".join(word_list)), 'universal')
    gonnas = list(np.where(allwords.word == 'gonna')[0])
    wannas = list(np.where(allwords.word == 'wanna')[0])
    
    # add 'n/a' for ommited words
    c = 0 # tracks position in word_types
    pos_list = []
    for i in range(len(allwords)):
        if i in gonnas or i in wannas:
            pos_list.append('n/a')
        else: 
            pos_list.append(word_types[c][1])
            c += 1
    
    # add parts of speech
    allwords['POS'] = pos_list
   
    #test_correct = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-correct-PAPER2-hs38_gpt2-xl_cnxt_1024_embeddings.pkl'))
    #test_incorrect = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-top5-incorrect-PAPER2-hs38_gpt2-xl_cnxt_1024_embeddings.pkl'))

    # split into correct and incorrect
    correct, _ = ci_split_uptodate_func(1, allwords) # get top 1 correct
    _, incorrect = ci_split_uptodate_func(5, allwords) # get top 5 incorrect
    
    print('top1 correct')
    get_pos_info(correct)
    print('top5 incorrect')
    get_pos_info(incorrect)

def generate_wordcloud():
    correct = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-correct-PAPER2-hs38_gpt2-xl_cnxt_1024_embeddings.pkl'))
    incorrect = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-top5-incorrect-PAPER2-hs38_gpt2-xl_cnxt_1024_embeddings.pkl'))

    clist = list(correct.token2word)
    ilist = list(incorrect.token2word)

    np.unique(clist, return_counts=True)
    np.unique(ilist, return_counts=True)

    c_input = " ".join(clist) + " "
    i_input = " ".join(ilist) + " "

    from wordcloud import WordCloud


    wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate(c_input)

    import matplotlib.pyplot as plt
    plt.figure(figsize = (20, 20), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('word-cloud-correct-test.png')



    incorrect_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate(i_input)

    plt.figure(figsize = (20, 20), facecolor = None)
    plt.imshow(incorrect_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.savefig('word-cloud-incorrect-test.png')

def predictability_freq_prepro():
    #Frequency/correctness plots.
    #breakpoint()
    all = pd.DataFrame(load_pickle('/scratch/gpfs/eham/247-encoding-updated/data/podcast2/777/pickles/777_full-paper-gpt2-xlhs38_gpt2-xl_cnxt_1024_embeddings.pkl'))                                            

    word_dict = {} # format: {'word': ['freq', 'top1n', 'top5n', 'not_in_top5'], 'word2': 
    for i in range(len(all)):
        word = all['token2word'][i]
        tid = all['token_id'][i]
        top10 = all['top10'][i]
        if word in word_dict:
            word_dict[word][0] +=1
            if tid in top10[:1]: word_dict[word][1] += 1
            if tid in top10[:5]: 
                word_dict[word][2] += 1
            else:
                word_dict[word][3] += 1
        else: 
            word_dict[word] = [0,0,0,0]
            word_dict[word][0] +=1
            if tid in top10[:1]: word_dict[word][1] += 1
            if tid in top10[:5]: 
                word_dict[word][2] += 1
            else:
                word_dict[word][3] += 1

    #Convert dict to data frame
    df = pd.DataFrame.from_dict(word_dict, orient='index',columns = ['freq', 'top1n', 'top5n', 'not_in_top5'])
    assert np.all(np.array(list(df['top5n'] + df['not_in_top5'] == df['freq'])))
    breakpoint()
    return df

def plot_freq_vs_predictability(df, topk):
    plt.figure()
    breakpoint()
    if topk == 1: y = list(df.top1n/df.freq)
    else: y = list(df.top5n/df.freq)
    plt.xlabel('Frequency')
    plt.ylabel('% times in top ' + str(topk) + ' / 100')
    plt.scatter(list(df.freq), y, s=10, alpha=0.2)
    plt.xscale('log')
    #plt.yscale('log')
    plt.savefig('freq_vs_pred_plot_top' + str(topk) + '.png')

if __name__ == '__main__':
    get_podcast_word_types()
    #get_ci_word_types() NOTE: don't use! see above!
    #df = predictability_freq_prepro()
    #plot_freq_vs_predictability(df, 1)



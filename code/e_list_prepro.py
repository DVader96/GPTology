#------------------------------------------------------#
# preprocessing for encoding with averages over regions#
# and for separating regions by patient                #
#------------------------------------------------------#

import glob
import numpy as np

def omit_es(r):
    omit_e = []
    with open('omit_e_list.txt','r') as f:
        for line in f:
            omit_e.append(line.strip('\n').strip(' '))
    breakpoint() 
    f.close()
    out = open(r + '_enc.csv', 'w')
    with open(r + '_enc_temp.csv', 'r') as f:
        for i,line in enumerate(f):
            if '_'.join(line.strip('\n').split(',')) not in omit_e: 
                out.write(line)
            else:
                print(line)
    out.close()

def get_n(rs):
    for r in rs:
        #breakpoint()
        f_list = glob.glob('/scratch/gpfs/eham/247-encoding-updated/code/*_' + r + '_enc.csv')
        total_cnt = 0
        f_cnts = []
        for f in f_list:
            cnt = 0
            with open(f, 'r') as rf:
                for line in rf:
                    if line[0] == '7': 
                        cnt +=1
                        total_cnt +=1
                print(f.split('/')[-1], ' ' + str(cnt))
            f_cnts.append(cnt)
        #breakpoint()
        print(total_cnt)
        for i, f in enumerate(f_list):
            np.save(f.split('/')[-1][:-4] + '_nume.npy', f_cnts[i]/total_cnt)
            print(f.split('/')[-1], ' ' + str(f_cnts[i]/total_cnt))
    return 0

if __name__ == '__main__':
   #omit_es('ifg')
   get_n(['ifg', 'mSTG', 'aSTG', 'TP'])

# preprocess ifg files from bobbi
# .txt format: subject\telectrode 
# NOTE: ^ no header
# .csv format: subject,electrode
# NOTE: ^ format is header
   
def prepro_ifg_all():
    all_ifg = open('all_ifg_electrodes.txt','r')
    all_sig = open('all_brain_significant_electrodes.txt','r')
    
    all_csv_out = open('222_all_enc.csv', 'w')
    all_csv_out.write('subject,electrode\n')
    all_txt_out = open('222_all_e_list.txt', 'w')    

    sig_dict = {}
    for line in all_sig:
        #breakpoint()
        items = line.split('_')
        patient = items[0].strip('NY')
        if len(items) == 6:
            print('1')
            en = '_'.join(items[-2:])
        elif len(items) == 5:
            en = items[-1]
        else: 
            print('error')
        if patient in sig_dict:
            sig_dict[patient].append(en)
        else:
            sig_dict[patient] = []
            sig_dict[patient].append(en)
        
        all_csv_out.write(patient + ',' + en)
        all_txt_out.write(patient + '\t' + en)
    
    all_csv_out.close()
    all_txt_out.close()
    all_sig.close()

    #breakpoint()
    ifg_txt_out = open('222_ifg_e_list.txt', 'w')
    ifg_csv_out = open('222_ifg_enc.csv', 'w')
    ifg_csv_out.write('subject,electrode\n')
    for line in all_ifg:
        items = line.split('_')
        patient = items[0].strip('NY')
        if len(items) == 6:
            print('1')
            en = '_'.join(items[-2:])
        elif len(items) == 5:
            en = items[-1]
        else: 
            print('error')
        if patient in sig_dict and en in sig_dict[patient]:
            #breakpoint()
            ifg_csv_out.write(patient + ',' + en)
            ifg_txt_out.write(patient + '\t' + en)

    all_ifg.close()
    ifg_txt_out.close()
    ifg_csv_out.close()

def prepro_stg():
    all_stg = open('Podcast_MNI_STG.txt', 'r')
    stg_txt_out = open('222_stg_e_list.txt', 'w')
    stg_csv_out = open('222_stg_enc.csv', 'w')
    stg_csv_out.write('subject,electrode\n')
    edict = {}
    for line in all_stg:
        items = line.split(' ')[0]
        patient = items[:3]
        #breakpoint()
        en = items[3:]
        if patient not in edict:
            edict[patient] = []
            edict[patient].append(en)
        elif patient in edict and en in edict[patient]:
            continue
        else:
            edict[patient].append(en)
        stg_csv_out.write(patient + ',' + en + '\n')
        stg_txt_out.write(patient + '\t' + en + '\n')
    stg_csv_out.close()
    stg_txt_out.close()
    all_stg.close()

def prepro_sep_ifg():
    ba45 = open('Podcast_MNI_BA45.txt','r')
    all_ifg = open('222-ifg_e_list.txt','r')
    
    ba45_out = open('222_ba45_e_list.txt','w')
    ba44_out = open('222_ba44_e_list.txt','w')
    
    ba45d = {}
    for line in ba45:
        items = line.split(' ')[0]
        patient = items[:3]
        en = items[3:]
        if patient not in ba45d:
            ba45d[patient] = []
            ba45d[patient].append(en)
        else:
            ba45d[patient].append(en)
    #breakpoint()
    for line in all_ifg:
        items = line.split('\t')
        patient = items[0]
        en = items[1].strip('\n')
        if patient in ba45d and en in ba45d[patient]:
            ba45_out.write(line)
        else:
            ba44_out.write(line)    
    
    ba45.close()
    all_ifg.close()
    ba45_out.close()
    ba44_out.close()

if __name__ == '__main__':
   #prepro_ifg_all() 
   prepro_stg()
   #prepro_sep_ifg()

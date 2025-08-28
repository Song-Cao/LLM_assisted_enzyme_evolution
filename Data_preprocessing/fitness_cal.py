import pandas as pd
import numpy as np
import math
import matplotlib
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from collections import defaultdict
from statistics import mean, pstdev


def nt_to_aa(nt_seq, nt2aa_table):
    nt_new = ''
    aa_new = ''
    start = len(nt_seq)
    for i in range(len(nt_seq)-2):
        if nt_seq[i: i+3] == 'atg':
            start = i
            break
    for i in range((len(nt_seq)-start)//3):
        codon = nt_seq[start+i*3: start+i*3+3]
        aa = nt2aa_table[codon.upper()]
        nt_new += codon
        aa_new += aa
        if aa == '*':
            break
    if aa_new[-1] == '*':
        return nt_new, aa_new
    return '', ''

def compute_mut(a,b, start_idx=1):
    muts = []
    a = a[start_idx-1: len(a)]
    b = b[start_idx-1: len(b)]
    for i in range(len(a)):
        if a[i] != b[i]:
            muts.append(a[i] + str(i+1) + b[i])
    return muts

def contain_proline_mut(muts):
    contain = False
    for mut in muts:
        if mut[-1] == 'P':
            contain = True
            break
    return contain

def assign_wt_seq(round):
    if round == 'r1':
        return 'MAGSGSPLAQQIKNTLTFIGQANAAGRMDEVRTLQQNLHPLWAEYFQQTEGSGGSPLAQQIQYGHVLIHQARAAGRMDEVRRLSENTLQLMKEYFQQSD*'
    elif round == 'r2':
        return 'MAGSGSPLAKQIKNTLTFIGQANAAGRMDEVRTLQQNLHPLWAEYFQQTEGSGGSPLAQQIQNGHVLIHQARAAGRMDEVRRLSEKTLQLMKEYFQQSD*'
    elif round == 'r3':
        return 'MAGSDSPLAEQIKNTLTFIGQANAAGRMDEVRTLQQNLHPLWAEYFRQTEGSGGSPLAQQIQNGHVLIHQARAAGRMDEVRRLSEKTLQLMKEYFQQSD*'


if __name__ == '__main__':
    
    ################################################### calculate fitness #########################################################

    # filter the count csv to merge sequences
    
    nt2aa_table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
        'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 
        'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W', 
    }
    
    round = 'r3'
    wt_seq = assign_wt_seq(round)
    input_path = f'Mid1_{round}_deep_dimsum_summary.csv'
    df = pd.read_csv(input_path)
    print(len(df))
    
    count_dict = defaultdict(lambda: [0,0,0,0])
    aa_dict = dict()
    for i in tqdm(range(len(df))):
        # check if the protein sequence contains proper start/stop codon
        nt_seq, aa_seq = nt_to_aa(df.loc[i, 'nt_seq'], nt2aa_table)
        if nt_seq == '':
            continue
        count_dict[aa_seq][0] += df.loc[i, 'a_e1_s0_bNA_count']
        count_dict[aa_seq][1] += df.loc[i, 'b_e1_s1_b1_count']
        count_dict[aa_seq][2] += df.loc[i, 'c_e1_s1_b2_count']
        count_dict[aa_seq][3] += df.loc[i, 'd_e1_s1_b3_count']

    aa_list, wt_list = [],[]
    count_list = [[] for i in range(4)]
    for aa_seq in count_dict:
        aa_list.append(aa_seq)
        wt_list.append(aa_seq==wt_seq)
        for i in range(4):
            count_list[i].append(count_dict[aa_seq][i])
    
    df = pd.DataFrame(data = {'seq': aa_list, 'wt': wt_list, 'count_input': count_list[0], \
                              'count_repl1': count_list[1], 'count_repl2': count_list[2], 'count_repl3': count_list[3]})
    
    
    # calculate hamming distances with WT
    df_wt = df[df['wt'] == True].reset_index(drop=True)
    muts, nham = [], []
    for i in tqdm(range(len(df))):
        seq = df.loc[i, 'seq']
        if len(seq) != len(wt_seq):
            muts.append('N/A')
            nham.append('N/A')
        else:
            mutations = compute_mut(wt_seq, seq, start_idx=5)
            muts.append((',').join(mutations))
            nham.append(len(mutations))
    df['mutations'] = muts
    df['nham'] = nham


    # calculate fitness scores
    
    fit_wt_s1 = math.log(df_wt.loc[0, 'count_repl1'] / df_wt.loc[0, 'count_input'])
    fit_wt_s2 = math.log(df_wt.loc[0, 'count_repl2'] / df_wt.loc[0, 'count_input'])
    fit_wt_s3 = math.log(df_wt.loc[0, 'count_repl3'] / df_wt.loc[0, 'count_input'])
    
    fitness_s1, fitness_s2, fitness_s3 = [],[],[]
    for i in range(len(df)):
        if df.loc[i, 'count_input'] < 5:
            fitness_s1.append(np.nan)
            fitness_s2.append(np.nan)
            fitness_s3.append(np.nan)
        else: 
            if df.loc[i, 'count_repl1'] < 1:
                fitness_s1.append(np.nan)
            else:
                fit_raw_s1 = math.log(df.loc[i, 'count_repl1'] / df.loc[i, 'count_input'])
                fitness_s1.append(fit_raw_s1-fit_wt_s1)
            
            if df.loc[i, 'count_repl2'] < 1:
                fitness_s2.append(np.nan)
            else:
                fit_raw_s2 = math.log(df.loc[i, 'count_repl2'] / df.loc[i, 'count_input'])
                fitness_s2.append(fit_raw_s2-fit_wt_s2)

            if df.loc[i, 'count_repl3'] < 1:
                fitness_s3.append(np.nan)
            else:
                fit_raw_s3 = math.log(df.loc[i, 'count_repl3'] / df.loc[i, 'count_input'])
                fitness_s3.append(fit_raw_s3-fit_wt_s3)
    df['fitness_repl1'] = fitness_s1
    df['fitness_repl2'] = fitness_s2
    df['fitness_repl3'] = fitness_s3

    avg_fitness, std_fitness = [], []
    for i in range(len(df)):
        if df.loc[i,'fitness_repl1']==df.loc[i,'fitness_repl1'] and df.loc[i,'fitness_repl2']==df.loc[i,'fitness_repl2'] and df.loc[i,'fitness_repl3']==df.loc[i,'fitness_repl3']:
            avg_fitness.append(mean([float(df.loc[i,'fitness_repl1']),float(df.loc[i,'fitness_repl2']),float(df.loc[i,'fitness_repl3'])]))
            std_fitness.append(pstdev([float(df.loc[i,'fitness_repl1']),float(df.loc[i,'fitness_repl2']),float(df.loc[i,'fitness_repl3'])]))
        else:
            avg_fitness.append(np.nan)
            std_fitness.append(np.nan)
    df['fitness_average'] = avg_fitness
    df['fitness_std'] = std_fitness

    df.to_csv(f'Mid1_{round}_merged.csv', index=False)
    

    # save fitness_s3 filtered csv file and plot correlation between fitness
    df_0 = pd.read_csv(f'Mid1_{round}_merged.csv')
    print(len(df_0))
    df = df_0[df_0['count_input'] >= 5].reset_index(drop=True)
    print(len(df))
    df_s1 = df[df['fitness_repl1'].notnull()].reset_index(drop=True)
    print(len(df_s1))
    df_s2 = df_s1[df_s1['fitness_repl2'].notnull()].reset_index(drop=True)
    print(len(df_s2))
    df_s3 = df_s2[df_s2['fitness_repl3'].notnull()].reset_index(drop=True)
    print(len(df_s3))
    df_s3.to_csv('Variant_data_3_rounds_Mid1_' + round + '.csv', index=False)
    

    
    # Plot count of single mutant along the seqeunce
    df = pd.read_csv(f'Mid1_{round}_merged.csv')
    pos2nummut_one = defaultdict(set)
    pos2nummut_all = defaultdict(set)
    for i in range(len(df)):
        if df.loc[i,'fitness_repl1']!=df.loc[i,'fitness_repl1'] and df.loc[i,'fitness_repl2']!=df.loc[i,'fitness_repl2'] and df.loc[i,'fitness_repl3']!=df.loc[i,'fitness_repl3']:
            continue
        if df.loc[i, 'nham'] == df.loc[i, 'nham']:
            if int(df.loc[i, 'nham']) == 1:
                pos2nummut_one[int(df.loc[i, 'mutations'][1:-1])].add(df.loc[i, 'mutations'])
    for key in pos2nummut_one:
        pos2nummut_one[key] = len(pos2nummut_one[key])
    x, y = [],[]
    for key in pos2nummut_one:
        x.append(key+4)
        y.append(pos2nummut_one[key])
    matplotlib.rc('font', size=10)
    plt.bar(x=x, height=y, width = 0.2)
    plt.title('Frequency of single mutants with at less one fitness score')
    plt.xlabel('Position')
    plt.xticks(np.arange(0, 100, step=5))
    plt.savefig(f'Frequency_single_mutants_with_at_less_one_fitness_score_{round}.png')
    plt.close()

    for i in range(len(df)):
        if df.loc[i, 'fitness_average'] == df.loc[i, 'fitness_average'] and df.loc[i, 'nham'] == df.loc[i, 'nham']:
            if int(df.loc[i, 'nham']) == 1:
                pos2nummut_all[int(df.loc[i, 'mutations'][1:-1])].add(df.loc[i, 'mutations'])
    for key in pos2nummut_all:
        pos2nummut_all[key] = len(pos2nummut_all[key])
    x, y = [],[]
    for key in pos2nummut_all:
        x.append(key+4)
        y.append(pos2nummut_all[key])
    matplotlib.rc('font', size=10)
    plt.bar(x = x, height=y, width = 0.2)
    plt.title('Frequency of single mutants with fitness score in all 3 sorts')
    plt.xlabel('Position')
    plt.xticks(np.arange(0, 100, step=5))
    plt.savefig(f'Frequency_single_mutants_with_fitness_score_in_all_3_sorts_{round}.png')
    plt.close()
    


    # Plot correlations between fitness from different sorts

    matplotlib.rc('font', size=10)
    plt.scatter(df_s2['fitness_repl1'], df_s2['fitness_repl2'], s = 5)
    plt.xlabel('Fitness from sort 1')
    plt.ylabel('Fitness from sort 2')
    corr = stats.spearmanr(list(df_s2['fitness_repl1']), list(df_s2['fitness_repl2']))[0]
    plt.title(f'Correlation between fitness in s1 and s2 for round {round}: {corr}')
    plt.savefig(f'correlation_bt_fitness_s1_and_fitness_s2_{round}.png')
    plt.close()
    
    plt.scatter(df_s3['fitness_repl2'], df_s3['fitness_repl3'], s = 5)
    plt.xlabel('Fitness from sort 2')
    plt.ylabel('Fitness from sort 3')
    corr = stats.spearmanr(list(df_s3['fitness_repl2']), list(df_s3['fitness_repl3']))[0]
    plt.title(f'Correlation between fitness in s2 and s3 for round {round}: {corr}')
    plt.savefig(f'correlation_bt_fitness_s2_and_fitness_s3_{round}.png')
    plt.close()

    plt.scatter(df_s3['fitness_repl1'], df_s3['fitness_repl3'], s = 5)
    plt.xlabel('Fitness from sort 1')
    plt.ylabel('Fitness from sort 3')
    corr = stats.spearmanr(list(df_s3['fitness_repl1']), list(df_s3['fitness_repl3']))[0]
    plt.title(f'Correlation between fitness in s1 and s3 for round {round}: {corr}')
    plt.savefig(f'correlation_bt_fitness_s1_and_fitness_s3_{round}.png')
    plt.close()


    # plot fitness distribution 

    plt.hist(df_s1['fitness_repl1'], bins = 30, alpha=0.5, label='Sort1 fitness')
    plt.hist(df_s2['fitness_repl2'], bins = 30, alpha=0.5, label='Sort2 fitness')
    plt.hist(df_s3['fitness_repl3'], bins = 30, alpha=0.5, label='Sort3 fitness')
    plt.xlabel('Fitness score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(f'hist_fitness_{round}.png')
    plt.close()


    # plot length distribution of all merged variants

    len_all = []
    for i in range(len(df_0)):
        len_all.append(len(df_0.loc[i, 'seq'])-1)
    plt.hist(len_all, bins=20)
    plt.xlabel('Sequence length')
    plt.ylabel('Frequency')
    plt.savefig(f'hist_seq_length_{round}.png')
    plt.close()


    # plot fitness of early stop variants vs. normal variants

    fitness_early_stop, fitness_normal_stop = [], []
    for i in range(len(df_s3)):
        if len(df_s3.loc[i, 'seq']) == len(wt_seq):
            fitness_normal_stop.append(df_s3.loc[i, 'fitness_average'])
        else:
            fitness_early_stop.append(df_s3.loc[i, 'fitness_average'])
    print(f'{len(fitness_early_stop)} early stop variants')
    print(f'{len(fitness_normal_stop)} normal stop variants')
    plt.hist(fitness_early_stop, bins=30, alpha=0.5, label='Early stop variants')
    plt.hist(fitness_normal_stop, bins=30, alpha=0.5, label='Normal stop variants')
    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(f'hist_fitness_stop_codon_{round}.png')
    plt.close()

    
    # plot fitness of proline variants vs. normal variants
    fitness_with_proline, fitness_without_proline = [], []
    for i in range(len(df_s3)):
        if df.loc[i, 'nham'] != df.loc[i, 'nham']:
            continue
        if int(df.loc[i, 'nham']) > 0:
            muts = df.loc[i, 'mutations'].split(',')
            if contain_proline_mut(muts):
                fitness_with_proline.append(df_s3.loc[i, 'fitness_average'])
            else:
                fitness_without_proline.append(df_s3.loc[i, 'fitness_average'])
        else:
            fitness_without_proline.append(df_s3.loc[i, 'fitness_average'])
    print(f'{len(fitness_with_proline)} variants with proline mutations')
    print(f'{len(fitness_without_proline)} variants without proline mutations')
    plt.hist(fitness_with_proline, bins=30, alpha=0.5, label='Variants with proline mutations')
    plt.hist(fitness_without_proline, bins=30, alpha=0.5, label='Variants without proline mutations')
    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(f'hist_fitness_proline_{round}.png')
    plt.close()

    
    ####################################### run stats on fitness trend and calculate fitness curve ######################################
    
    '''
     # get statistics for the general trend of fitness
    round = 'r2'
    df = pd.read_csv(f'Variant_data_3_rounds_Mid1_{round}.csv')
    HML,HLM,MHL,MLH,LMH,LHM = [],[],[],[],[],[]
    for i in range(len(df)):
        fit = [df.loc[i, 'fitness_repl1'], df.loc[i, 'fitness_repl2'], df.loc[i, 'fitness_repl3']]
        sortidx = list(np.argsort(fit))
        if sortidx == [2,1,0]:
            HML.append(i)
        if sortidx == [1,2,0]:
            HLM.append(i)
        if sortidx == [2,0,1]:
            MHL.append(i)
        if sortidx == [1,0,2]:
            MLH.append(i)
        if sortidx == [0,1,2]:
            LMH.append(i)
        if sortidx == [0,2,1]:
            LHM.append(i)

    # creating the bar plot
    categories = ['HML', 'HLM', 'MHL', 'MLH', 'LMH', 'LHM']
    freq = [len(HML), len(HLM), len(MHL), len(MLH), len(LMH), len(LHM)]
    
    fig = plt.figure(figsize = (10, 5))
    plt.bar(categories, freq)
    plt.xlabel("Groups")
    plt.ylabel("Frequency")
    plt.title("Distribution of fitness trend")
    plt.savefig(f'Distribution_of_fitness_trend_{round}.png')
    plt.close()



    df = df.sort_values(by=['fitness_repl3'], ascending=False).reset_index(drop = True)
    #plot & calculate fit curves
    x_values = np.array([0.,1.,2.], dtype=np.float64)
   
    def exp_obj(x,a,b):
        return np.exp(a*x) + b
    def sat_obj(x,a,b):
        return - np.exp(-a*x)+ b
    def lin_obj(x,a,b):
        return a*x+b
    def cal_err(obj_func, params, x_values, y_values):
        y_pred = obj_func(x_values, *params)
        err = np.sum((y_pred-y_values)**2)
        return err
            

    num_d = {}
    num_d['ori_exp'], num_d['ori_sat'], num_d['ori_lin'] = 0,0,0
    err_d = {}
    err_d['ori_exp'], err_d['ori_sat'], err_d['ori_lin'] = [],[],[]
    para_d = {}
    para_d['ori_exp'], para_d['ori_sat'], para_d['ori_lin'] = [],[],[]

    for i in range(len(df)):
    
        ori_y_values = [df.loc[i, 'fitness_repl1'], df.loc[i, 'fitness_repl2'], df.loc[i, 'fitness_repl3']]
        # calculate parameters and err of estimation for raw/normalized fitness score given specific function
        obj = [exp_obj, sat_obj, lin_obj]
        obj_name = ['exp', 'sat', 'lin']
        for j in range(len(obj)):
            try:
                ori_popt, ori_pcov = curve_fit(obj[j], x_values, np.array(ori_y_values, dtype=np.float64))
                # err_d['ori_'+obj_name[j]].append(np.sqrt(np.diag(ori_pcov)))
                err_d['ori_'+obj_name[j]].append(cal_err(obj[j], ori_popt, x_values, np.array(ori_y_values, dtype=np.float64)))
                num_d['ori_'+obj_name[j]] += 1
                para_d['ori_'+obj_name[j]].append(ori_popt)
            except OverflowError:
                err_d['ori_'+obj_name[j]].append(0)
                para_d['ori_'+obj_name[j]].append((0,0))


    for key in err_d:
        err_d[key] = sum(list(np.array(err_d[key])/num_d[key]))
        print(key + ': ' +str(num_d[key]) + ' fitted')
    err_sorted = sorted(err_d.items(), key=lambda x:x[1])
    print(err_sorted)
    winner = err_sorted[0][0]

    categories = ['Exp', 'Sat', 'Lin']
    freq = [math.log(err_d['ori_exp']), math.log(err_d['ori_sat']), math.log(err_d['ori_lin'])]
    fig = plt.figure(figsize = (5, 5))
    plt.bar(categories, freq)
    plt.xlabel("Curve type")
    plt.ylabel("Logged square error of est.")
    plt.title(f'Performance of different curves for {round}')
    plt.savefig(f'Perf_curves_{round}.png')
    plt.close()

    
    # plot curve fit for top and bottom 10 candidates based on sort 3 fitness
    for i in list(range(10))+list(range(len(df)-10, len(df))):
        y_values = [df.loc[i, 'fitness_repl1'], df.loc[i, 'fitness_repl2'], df.loc[i, 'fitness_repl3']]
        fig = plt.figure(figsize = (5, 5))
        plt.scatter(x_values, y_values)
        xline = np.arange(-1, 3, 0.2)
        yline = lin_obj(xline, *para_d[winner][i])
        plt.plot(xline, yline, '--', color='red')
        plt.xlabel("Sort")
        plt.ylabel("Fitness")
        plt.savefig(f'trend_plots_{round}/Fitness_vs_sort_{i}.png')
        plt.close()
        print(para_d[winner][i][0])



    # calculate correlation between fitness and slope
    print(winner)
    slope_list, bias_list = [],[]
    for i in range(len(para_d[winner])):
        slope = para_d[winner][i][0]
        slope_list.append(slope)
        bias = para_d[winner][i][1]
        bias_list.append(bias)

    for i in range(1,4):
        fitness_list = list(df[f'fitness s{i}'])

        corr_s = stats.spearmanr(fitness_list, slope_list)[0]
        print('slope-fitness corr: ' + str(corr_s))
        fig = plt.figure(figsize = (5, 5))
        plt.scatter(fitness_list, slope_list, s = 5)
        plt.xlabel(f'Fitness from sort {i}')
        plt.ylabel('Slope coefficient')
        plt.title(f'Correlation sort {i} round {round[1:]}: {corr_s}')
        plt.savefig(f'correlation_bt_fitness_sort_{i}_and_slope_{round}.png')
        plt.close()

        corr_b = stats.spearmanr(fitness_list, bias_list)[0]
        print('bias-fitness corr: ' + str(corr_b))
        fig = plt.figure(figsize = (5, 5))
        plt.scatter(fitness_list, bias_list, s = 5)
        plt.xlabel(f'Fitness from sort {i}')
        plt.ylabel('Bias coefficient')
        plt.savefig(f'correlation_bt_fitness_sort_{i}_and_bias_{round}.png')
        plt.close()
    '''

    '''
    # Save top 10 variants based on slope within the LMH group
    # and save top 10 variants based on sort 3 fitness
    df['slope coefficient'] = slope_list
    df_inc = df.loc[LMH]
    df_inc = df_inc.sort_values(by=['slope coefficient'], ascending=False).reset_index(drop = True)
    df_inc.loc[:9].to_csv(f'Top_10_slope_LMH_{round}.csv', index=False)
    df_top10_s3 = df.loc[:9]
    df_top10_s3.to_csv(f'Top_10_fitness_s3_{round}.csv', index=False)
    print(20-len(set(list(df_inc.loc[:9]['nt seq']) + list(df_top10_s3['nt seq']))))

    df = df.sort_values(by=[_], ascending=False).reset_index(drop = True)
    df_top10_all = df.loc[:9]
    df_top10_all.to_csv(f'Top_10_fitness_average_{round}.csv', index=False)
    print(20-len(set(list(df_top10_s3['nt seq']) + list(df_top10_all['nt seq']))))
    

    with open(f'Top_10_fitness_s3_{round}.fasta', 'w') as wf:
        for i in range(10):
            wf.write('>Variant ' + str(i) + '\n' + df_top10_s3.loc[i,'aa_seq'] + '\n')
    
    with open(f'Top_10_fitness_average_{round}.fasta', 'w') as wf:
        for i in range(10):
            wf.write('>Variant ' + str(i) + '\n' + df_top10_all.loc[i,'aa_seq'] + '\n')
    '''
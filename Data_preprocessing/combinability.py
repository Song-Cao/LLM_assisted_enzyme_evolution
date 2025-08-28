import pandas as pd
import math
from statistics import mean, pstdev, median
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict
import dill
import numpy as np


def addlabels(x,y, fontsize=7):
    for i in range(len(x)):
        if y[i] > 0:
            plt.text(i, y[i], x[i], ha = 'center', fontsize=fontsize)

if __name__ == '__main__':
    rd = 'r2'
    csv_file = f'Variant_data_3_rounds_Mid1_{rd}.csv'
    df = pd.read_csv(csv_file)

    # Record the fitness and std for point mutations
    F_obs = dict()
    std_obs = dict()
    for i in tqdm(range(len(df))):
        muts = df.loc[i, 'mutations']
        if muts != muts:
            continue
        muts = muts.split(',')
        if len(muts) == 1:
            F_obs[muts[0]] = float(df.loc[i, 'fitness average'])
            std_obs[muts[0]] = float(df.loc[i, 'fitness std'])


    # Check the combinability for each variant and calculate combinability score and 
    # mutability score for each residue position
    combinable_list = []
    F_exp_list = []
    std_exp_list = []
    Position2comb_score = defaultdict(lambda: 0)
    Position2mut_score = defaultdict(list)
    Position2comb_vars = defaultdict(list)

    for i in tqdm(range(len(df))):
        muts = df.loc[i, 'mutations']
        if muts != muts:
            combinable_list.append('N/A')
            F_exp_list.append('N/A')
            std_exp_list.append('N/A')
            continue
        muts = muts.split(',')

        if len(muts) == 1:
            combinable_list.append('N/A')
            F_exp_list.append('N/A')
            std_exp_list.append('N/A')
            Position2mut_score[int(muts[0][1:-1])].append(float(df.loc[i, 'fitness average']))
        
        else:
            combinable = True
            F_obs_variant = float(df.loc[i, 'fitness average'])
            std_obs_variant = float(df.loc[i, 'fitness std'])
            F_exp_variant = 0
            std_exp_variant = 0
            for mut in muts:
                if mut not in F_obs:
                    combinable = False
                    break
                F_exp_variant += F_obs[mut]
                std_exp_variant += std_obs[mut]**2
            
            # Combinable == False at this stage implies expected fitness is incalculable
            if combinable == False:
                combinable_list.append('N/A')
                F_exp_list.append('N/A')
                std_exp_list.append('N/A')
            
            # Expected fitness is calculable but let's check whether the variant is truly combinable
            else:
                std_exp_variant = math.sqrt(std_exp_variant)
                if F_obs_variant <= std_obs_variant:
                    combinable = False
                elif (F_obs_variant - F_exp_variant) <= -std_exp_variant:
                    combinable = False
                combinable_list.append(combinable)
                F_exp_list.append(F_exp_variant)
                std_exp_list.append(std_exp_variant)
                if combinable == True:
                    for mut in muts:
                        pos = int(mut[1:-1])
                        Position2comb_score[pos] += len(muts)
                        Position2comb_vars[pos].append(muts)
    
    for key in Position2mut_score:
        Position2mut_score[key] = median(Position2mut_score[key])
    assert len(combinable_list) == len(F_exp_list) == len(std_exp_list)

    # Update the dataframe and write combinability mutability scores into pickle/csv files
    df['combinability'] = combinable_list
    df['expected fitness'] = F_exp_list
    df['expected fitness std'] = std_exp_list
    df.to_csv('Variant_data_combinability_Mid1_{rd}.csv', index=False)

    print(Position2comb_score)
    print(Position2mut_score)
    print(Position2comb_vars)

    dill.dump(Position2comb_score, open(f"position_to_combinability_score_mapping_{rd}.pkl","wb"))
    dill.dump(Position2mut_score, open(f"position_to_mutability_score_mapping_{rd}.pkl","wb"))
    dill.dump(Position2comb_vars, open(f"position_to_combinable_variants_{rd}.pkl","wb"))
    
    positions = list(Position2comb_score.keys())
    positions.sort()
    comb_scores = []
    comb_vars = []
    for pos in positions:
        comb_scores.append(Position2comb_score[pos])
        comb_var = Position2comb_vars[pos]
        for i,muts in enumerate(comb_var):
            comb_var[i] = '(' + (',').join(muts) + ')'
        comb_var = (', ').join(comb_var)
        comb_vars.append(comb_var)
    df_comb = pd.DataFrame({'residue position': positions, 'combinability score': comb_scores, \
                            'combinable variants': comb_vars})
    df_comb.to_csv(f'Combinanility_info_{rd}.csv', index=False)


    # Plot the combinability and mutability information into bar plot
    positions = [i+1 for i in range(96)]
    comb_scores = []
    mut_scores = []
    print(Position2comb_score[1])
    print(Position2mut_score[1])
    for idx in positions:
        if idx in Position2comb_score:
            comb_scores.append(Position2comb_score[idx])
        else:
            comb_scores.append(0.0)
        if idx in Position2mut_score:
            mut_scores.append(Position2mut_score[idx])
        else:
            mut_scores.append(0.0)
        
    matplotlib.rc('font', size=20)
    plt.figure(figsize=(12, 8))
    plt.bar(x = positions, height = comb_scores, width = 0.8)
    plt.title(f'Combinability score at different residue positions for {rd}')
    plt.ylabel('Combinability score')
    plt.xlabel('Position')
    plt.xticks(np.arange(0, 101, 10))
    plt.grid(axis = 'x')
    # plt.yscale('log')
    # addlabels(positions,comb_scores)
    plt.savefig(f'Combinability_{rd}.png')
    plt.close()

    matplotlib.rc('font', size=20)
    plt.figure(figsize=(12, 8))
    matplotlib.rc('font', size=10)
    plt.bar(x = positions, height = mut_scores, width = 0.8)
    plt.title(f'Mutability score at different residue positions for {rd}')
    plt.xlabel('Position')
    plt.xticks(np.arange(0, 101, 10))
    plt.grid(axis = 'x')
    # addlabels(positions,mut_scores)
    plt.savefig(f'Mutability_{rd}.png')
    plt.close()

            



                                       

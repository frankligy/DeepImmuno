'''
This script is for generating all figures for deepImmuno paper
'''

import pandas as pd
import numpy as np
import scipy.stats as sc
import matplotlib as mpl
from matplotlib.colors import ListedColormap, rgb2hex
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

if __name__ == '__main__':
    # figure1
    fig,ax = plt.subplots()
    x = np.linspace(0,1,5000)
    y1 = sc.beta.pdf(x,10,1)
    y2 = sc.beta.pdf(x,20,1)
    ax.plot(x,y1,color='black',label='Prior')
    ax.plot(x,y2,color='red',label='Posterior')
    ax.yaxis.set_tick_params(length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('f(x)')
    ax.set_xlabel('potential(x)')
    ax.legend(frameon=False)
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure1/pos_beta.pdf')

    fig,ax = plt.subplots()
    x = np.linspace(0,1,5000)
    y1 = sc.beta.pdf(x,3,4)
    y2 = sc.beta.pdf(x,3,14)
    ax.plot(x,y1,color='black',label='Prior')
    ax.plot(x,y2,color='red',label='Posterior')
    ax.yaxis.set_tick_params(length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('f(x)')
    ax.set_xlabel('potential(x)')
    ax.legend(frameon=False)
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure1/neg_beta.pdf')

    fig,ax = plt.subplots()
    dummy = np.random.rand(3,6)
    ax.imshow(dummy,cmap='Set3')
    ax.set_xticks([0.5,1.5,2.5,3.5,4.5])
    ax.set_yticks([0.5,1.5])
    ax.tick_params(length=0,labelleft=False,labelbottom=False)
    ax.grid(alpha=1,color='black')
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure1/input_mat_pep.pdf')

    # figure2
    # figure2A,B see CNN_index.py

    # now let's plot figure2C,D

    fig,ax1 = plt.subplots()
    x1, x2, x3 = [1,6,11], [2,7,12], [3,8,13]
    y1, y2, y3 = [3,4,0], [1,3,0], [4,8,0]

    ax1.bar(x1,y1,color='#E36DF2',width=0.8,label='DeepHLApan')
    ax1.bar(x2,y2,color='#04BF7B',width=0.8,label='IEDB')
    ax1.bar(x3,y3,color='#F26D6D',width=0.8,label='DeepImmuno-CNN')

    ax1.set_xticks([2,7,12])
    ax1.set_ylabel('Number of immunogenic peptides')
    ax1.set_xticklabels(['Top20','Top50','Hard cutoff'])
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)
    tmp1x = [1,2,3,6,7,8]
    tmp1y = [3,1,4,4,3,8]
    for i in range(len(tmp1x)):
        ax1.text(tmp1x[i]-0.2,tmp1y[i]+0.1,s=tmp1y[i])

    ax2 = ax1.twinx()
    ax2.bar([11,12,13],[0.34,0.63,0.83],color=['#E36DF2','#04BF7B','#F26D6D'])
    ax2.set_ylim([0,1.05])
    ax2.set_ylabel('Sensitivity')
    tmp2x = [11,12,13]
    tmp2y = [0.34,0.63,0.83]
    for i in range(len(tmp2x)):
        ax2.text(tmp2x[i]-0.3,tmp2y[i]+0.01,s=tmp2y[i])
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure2/cell_test.pdf')

    # now Nathan suggests to split figure2C,2D
    fig = plt.figure(figsize=(6.4,4.8))
    ax1 = plt.axes([0.1,0.1,0.55,0.80])
    ax2 = plt.axes([0.75,0.1,0.20,0.80])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    x1, x2, x3 = [1,6], [2,7], [3,8]
    y1, y2, y3 = [3,4], [1,3], [4,8]

    ax1.bar(x1,y1,color='#E36DF2',width=0.8,label='DeepHLApan')
    ax1.bar(x2,y2,color='#04BF7B',width=0.8,label='IEDB')
    ax1.bar(x3,y3,color='#F26D6D',width=0.8,label='DeepImmuno-CNN')

    ax1.set_xticks([2,7])
    ax1.set_ylabel('Number of immunogenic peptides')
    ax1.set_xticklabels(['Top20','Top50'])
    ax1.legend(frameon=True)
    ax1.grid(alpha=0.3,axis='y')
    tmp1x = [1,2,3,6,7,8]
    tmp1y = [3,1,4,4,3,8]
    for i in range(len(tmp1x)):
        ax1.text(tmp1x[i]-0.2,tmp1y[i]+0.1,s=tmp1y[i])

    ax2.bar([1,2,3],[0.34,0.63,0.83],color=['#E36DF2','#04BF7B','#F26D6D'])
    ax2.set_ylim([0,1.05])
    ax2.set_ylabel('Sensitivity')
    ax2.grid(alpha=0.3,axis='y')
    ax2.set_xticks([2])
    ax2.set_xticklabels(['hard cutoff'])
    tmp2x = [1,2,3]
    tmp2y = [0.34,0.63,0.83]
    for i in range(len(tmp2x)):
        ax2.text(tmp2x[i]-0.3,tmp2y[i]+0.01,s=tmp2y[i])

    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure2/neoantigen.pdf')

    # figureD please refer to immuno3_7


    # Figure3

    fig,ax = plt.subplots()
    dummy = np.random.rand(5,9)
    ax.imshow(dummy,cmap='Set3')
    ax.set_xticks(np.linspace(0.5,7.5,8))
    ax.set_yticks(np.linspace(0.5,3.5,4))
    ax.tick_params(length=0,labelleft=False,labelbottom=False)
    ax.grid(alpha=1,color='black')
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure3/mat.pdf')

    fig,ax = plt.subplots()
    dummy = [[ i * 10 for i in [0.005,0.012,0.002,0.034,0.016,0.013,0.001,0.010,0.007]]]
    hm = ax.imshow(dummy,cmap='viridis')
    ax.set_xticks(np.linspace(0.5,7.5,8))
    ax.set_yticks([])
    ax.tick_params(length=0,labelleft=False,labelbottom=False)
    ax.grid(alpha=1,color='black')
    fig.colorbar(hm)
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure3/mat_below.pdf')

    # figure3B, figure3C are in CNN_index


    # figure4
    # PCA,t-sne all in pytorhc immuno6.py























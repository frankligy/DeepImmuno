

def plot_violin(ax,dataset,xlabel=None,ylabel=None):
    vp = ax.violinplot(dataset = dataset,showextrema=False)
    for part in vp['bodies']:
        part.set(facecolor='#D43F3A',edgecolor='black',alpha=1)

    tmp = [np.percentile(data,[25,50,75]) for data in dataset]
    def get_whisker(tmp,dataset):
        whisker = []
        for item,data in zip(tmp,dataset):
            data = np.array(data)
            q1 = item[0]
            median = item[1]
            q3 = item[2]
            iqr = q3-q1
            upper = q3 + 1.5*iqr
            upper = np.clip(upper,q3,data.max())
            lower = q1 - 1.5*iqr
            lower = np.clip(lower,data.min(),q1)
            whisker.append((upper,lower))
        return whisker
    whisker = get_whisker(tmp,dataset)
    x = np.arange(len(dataset)) + 1
    ax.scatter(x,[item[1] for item in tmp],marker='o',color='white',s=30,zorder=3)
    ax.vlines(x,[item[0] for item in tmp],[item[2] for item in tmp],color='black',linestyle='-',lw=5)
    ax.vlines(x,[item[0] for item in whisker], [item[1] for item in whisker],color='k',linestyle='-',lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabel,fontsize=6)
    ax.set_ylabel(ylabel)


import pickle

with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_pseudo/holding_ML_real_aa_pseudo.p', 'rb') as f:
    holding_ML = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_pseudo/holding_CNN_aa_pseudo.p', 'rb') as f:
    holding_CNN = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_pseudo/holding_reslike_aa_pseudo.p', 'rb') as f:
    holding_res = pickle.load(f)

import matplotlib.pyplot as plt
import itertools
ax = plt.subplot(3,3,1)
bp = ax.boxplot(positions=[1,2,3,4,5,6,7], x=[holding_ML['elasticnet']['validation']
           , holding_ML['KNN']['validation'], holding_ML['SVR']['validation'], holding_ML['randomforest']['validation']
           , holding_ML['adaboost']['validation'], holding_CNN['validation'], holding_res['validation']])
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
ax.set_title('validation RMSE',fontsize=8)
for flier in bp['fliers']:
    flier.set(markersize=1.5)


ax = plt.subplot(3,3,2)
bp=ax.boxplot(positions=[1,2,3,4,5,6,7], x=[holding_ML['elasticnet']['dengue']
           , holding_ML['KNN']['dengue'] , holding_ML['SVR']['dengue'] , holding_ML['randomforest']['dengue']
           , holding_ML['adaboost']['dengue'] , holding_CNN['dengue'] , holding_res['dengue']])
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
ax.set_title('dengue accuracy',fontsize=8)
for flier in bp['fliers']:
    flier.set(markersize=1.5)

def pick_cell_result1(cell):
    return [item[0] for item in cell]
def pick_cell_result2(cell):
    return [item[1] for item in cell]
def pick_cell_result3(cell):
    return [item[2] for item in cell]
ax = plt.subplot(3,3,3)
bp=ax.boxplot(positions=[1,2,3,4,5,6,7], x=[pick_cell_result1(holding_ML['elasticnet']['cell'])
           , pick_cell_result1(holding_ML['KNN']['cell']) , pick_cell_result1(holding_ML['SVR']['cell'])
           , pick_cell_result1(holding_ML['randomforest']['cell'])
           , pick_cell_result1(holding_ML['adaboost']['cell']) , pick_cell_result1(holding_CNN['cell'])
           , pick_cell_result1(holding_res['cell'])])
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
ax.set_title('neoantigen recall',fontsize=8)
for flier in bp['fliers']:
    flier.set(markersize=1.5)

ax = plt.subplot(3,3,4)
bp=ax.boxplot(positions=[1,2,3,4,5,6,7], x=[pick_cell_result2(holding_ML['elasticnet']['cell'])
           , pick_cell_result2(holding_ML['KNN']['cell']) , pick_cell_result2(holding_ML['SVR']['cell'])
           , pick_cell_result2(holding_ML['randomforest']['cell'])
           , pick_cell_result2(holding_ML['adaboost']['cell']) , pick_cell_result2(holding_CNN['cell'])
           , pick_cell_result2(holding_res['cell'])])
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
ax.set_title('neoantigen top20',fontsize=8)
for flier in bp['fliers']:
    flier.set(markersize=1.5)

ax = plt.subplot(3,3,5)
bp=ax.boxplot(positions=[1,2,3,4,5,6,7], x=[pick_cell_result3(holding_ML['elasticnet']['cell'])
           , pick_cell_result3(holding_ML['KNN']['cell']) , pick_cell_result3(holding_ML['SVR']['cell'])
           , pick_cell_result3(holding_ML['randomforest']['cell'])
           , pick_cell_result3(holding_ML['adaboost']['cell']) , pick_cell_result3(holding_CNN['cell'])
           , pick_cell_result3(holding_res['cell'])])
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
ax.set_title('neoantigen top50',fontsize=8)
for flier in bp['fliers']:
    flier.set(markersize=1.5)


def pick_covid_result1(covid):
    return [item[0] for item in covid]
def pick_covid_result2(covid):
    return [item[1] for item in covid]
def pick_covid_result3(covid):
    return [item[2] for item in covid]
def pick_covid_result4(covid):
    return [item[3] for item in covid]
ax = plt.subplot(3,3,6)
bp=ax.boxplot(positions=[1,2,3,4,5,6,7], x=[pick_covid_result1(holding_ML['elasticnet']['covid'])
           , pick_covid_result1(holding_ML['KNN']['covid']), pick_covid_result1(holding_ML['SVR']['covid'])
           , pick_covid_result1(holding_ML['randomforest']['covid'])
           , pick_covid_result1(holding_ML['adaboost']['covid']), pick_covid_result1(holding_CNN['covid'])
           , pick_covid_result1(holding_res['covid'])])
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
ax.set_title('covid convalescent recall',fontsize=8)
for flier in bp['fliers']:
    flier.set(markersize=1.5)

ax = plt.subplot(3,3,7)
bp=ax.boxplot(positions=[1,2,3,4,5,6,7], x=[pick_covid_result2(holding_ML['elasticnet']['covid'])
           , pick_covid_result2(holding_ML['KNN']['covid']), pick_covid_result2(holding_ML['SVR']['covid'])
           , pick_covid_result2(holding_ML['randomforest']['covid'])
           , pick_covid_result2(holding_ML['adaboost']['covid']), pick_covid_result2(holding_CNN['covid'])
           , pick_covid_result2(holding_res['covid'])])
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
ax.set_title('covid unexposed recall',fontsize=8)
for flier in bp['fliers']:
    flier.set(markersize=1.5)

ax = plt.subplot(3,3,8)
bp=ax.boxplot(positions=[1,2,3,4,5,6,7], x=[pick_covid_result3(holding_ML['elasticnet']['covid'])
           , pick_covid_result3(holding_ML['KNN']['covid']), pick_covid_result3(holding_ML['SVR']['covid'])
           , pick_covid_result3(holding_ML['randomforest']['covid'])
           , pick_covid_result3(holding_ML['adaboost']['covid']), pick_covid_result3(holding_CNN['covid'])
           , pick_covid_result3(holding_res['covid'])])
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
ax.set_title('covid convalescent precision',fontsize=8)
for flier in bp['fliers']:
    flier.set(markersize=1.5)

ax = plt.subplot(3,3,9)
bp=ax.boxplot(positions=[1,2,3,4,5,6,7], x=[pick_covid_result4(holding_ML['elasticnet']['covid'])
           , pick_covid_result4(holding_ML['KNN']['covid']), pick_covid_result4(holding_ML['SVR']['covid'])
           , pick_covid_result4(holding_ML['randomforest']['covid'])
           , pick_covid_result4(holding_ML['adaboost']['covid']), pick_covid_result4(holding_CNN['covid'])
           , pick_covid_result4(holding_res['covid'])])
ax.set_xticks([1,2,3,4,5,6,7])
ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
ax.set_title('covid unexposed precision',fontsize=8)
for flier in bp['fliers']:
    flier.set(markersize=1.5)




# try violin plot




import numpy as np
import matplotlib.pyplot as plt
import itertools
ax = plt.subplot(3,3,1)
plot_violin(ax,[holding_ML['elasticnet']['validation']
           , holding_ML['KNN']['validation'], holding_ML['SVR']['validation'], holding_ML['randomforest']['validation']
           , holding_ML['adaboost']['validation'], holding_CNN['validation'], holding_res['validation']],['eNet','KNN','SVR','RF','ada','CNN','res'])
ax.set_title('validation RMSE',fontsize=8)



ax = plt.subplot(3,3,2)
plot_violin(ax,[holding_ML['elasticnet']['dengue']
           , holding_ML['KNN']['dengue'] , holding_ML['SVR']['dengue'] , holding_ML['randomforest']['dengue']
           , holding_ML['adaboost']['dengue'] , holding_CNN['dengue'] , holding_res['dengue']],['eNet','KNN','SVR','RF','ada','CNN','res'])
ax.set_title('dengue accuracy',fontsize=8)


def pick_cell_result1(cell):
    return [item[0] for item in cell]
def pick_cell_result2(cell):
    return [item[1] for item in cell]
def pick_cell_result3(cell):
    return [item[2] for item in cell]
ax = plt.subplot(3,3,3)
plot_violin(ax,[pick_cell_result1(holding_ML['elasticnet']['cell'])
           , pick_cell_result1(holding_ML['KNN']['cell']) , pick_cell_result1(holding_ML['SVR']['cell'])
           , pick_cell_result1(holding_ML['randomforest']['cell'])
           , pick_cell_result1(holding_ML['adaboost']['cell']) , pick_cell_result1(holding_CNN['cell'])
           , pick_cell_result1(holding_res['cell'])],['eNet','KNN','SVR','RF','ada','CNN','res'])
ax.set_title('neoantigen recall',fontsize=8)


ax = plt.subplot(3,3,4)
plot_violin(ax,[pick_cell_result2(holding_ML['elasticnet']['cell'])
           , pick_cell_result2(holding_ML['KNN']['cell']) , pick_cell_result2(holding_ML['SVR']['cell'])
           , pick_cell_result2(holding_ML['randomforest']['cell'])
           , pick_cell_result2(holding_ML['adaboost']['cell']) , pick_cell_result2(holding_CNN['cell'])
           , pick_cell_result2(holding_res['cell'])],['eNet','KNN','SVR','RF','ada','CNN','res'])
ax.set_title('neoantigen top20',fontsize=8)


ax = plt.subplot(3,3,5)
plot_violin(ax,[pick_cell_result3(holding_ML['elasticnet']['cell'])
           , pick_cell_result3(holding_ML['KNN']['cell']) , pick_cell_result3(holding_ML['SVR']['cell'])
           , pick_cell_result3(holding_ML['randomforest']['cell'])
           , pick_cell_result3(holding_ML['adaboost']['cell']) , pick_cell_result3(holding_CNN['cell'])
           , pick_cell_result3(holding_res['cell'])],['eNet','KNN','SVR','RF','ada','CNN','res'])
ax.set_title('neoantigen top50',fontsize=8)



def pick_covid_result1(covid):
    return [item[0] for item in covid]
def pick_covid_result2(covid):
    return [item[1] for item in covid]
def pick_covid_result3(covid):
    return [item[2] for item in covid]
def pick_covid_result4(covid):
    return [item[3] for item in covid]
ax = plt.subplot(3,3,6)
plot_violin(ax,[pick_covid_result1(holding_ML['elasticnet']['covid'])
           , pick_covid_result1(holding_ML['KNN']['covid']), pick_covid_result1(holding_ML['SVR']['covid'])
           , pick_covid_result1(holding_ML['randomforest']['covid'])
           , pick_covid_result1(holding_ML['adaboost']['covid']), pick_covid_result1(holding_CNN['covid'])
           , pick_covid_result1(holding_res['covid'])],['eNet','KNN','SVR','RF','ada','CNN','res'])
ax.set_title('covid convalescent recall',fontsize=8)


ax = plt.subplot(3,3,7)
plot_violin(ax,[pick_covid_result2(holding_ML['elasticnet']['covid'])
           , pick_covid_result2(holding_ML['KNN']['covid']), pick_covid_result2(holding_ML['SVR']['covid'])
           , pick_covid_result2(holding_ML['randomforest']['covid'])
           , pick_covid_result2(holding_ML['adaboost']['covid']), pick_covid_result2(holding_CNN['covid'])
           , pick_covid_result2(holding_res['covid'])],['eNet','KNN','SVR','RF','ada','CNN','res'])
ax.set_title('covid unexposed recall',fontsize=8)


ax = plt.subplot(3,3,8)
plot_violin(ax,[pick_covid_result3(holding_ML['elasticnet']['covid'])
           , pick_covid_result3(holding_ML['KNN']['covid']), pick_covid_result3(holding_ML['SVR']['covid'])
           , pick_covid_result3(holding_ML['randomforest']['covid'])
           , pick_covid_result3(holding_ML['adaboost']['covid']), pick_covid_result3(holding_CNN['covid'])
           , pick_covid_result3(holding_res['covid'])],['eNet','KNN','SVR','RF','ada','CNN','res'])
ax.set_title('covid convalescent precision',fontsize=8)


ax = plt.subplot(3,3,9)
plot_violin(ax,[pick_covid_result4(holding_ML['elasticnet']['covid'])
           , pick_covid_result4(holding_ML['KNN']['covid']), pick_covid_result4(holding_ML['SVR']['covid'])
           , pick_covid_result4(holding_ML['randomforest']['covid'])
           , pick_covid_result4(holding_ML['adaboost']['covid']), pick_covid_result4(holding_CNN['covid'])
           , pick_covid_result4(holding_res['covid'])],['eNet','KNN','SVR','RF','ada','CNN','res'])
ax.set_title('covid unexposed precision',fontsize=8)


# per method plot


import matplotlib.pyplot as plt
import pickle
import numpy as np
import itertools

def pick_covid_result1(covid):
    return [item[0] for item in covid]
def pick_covid_result2(covid):
    return [item[1] for item in covid]
def pick_covid_result3(covid):
    return [item[2] for item in covid]
def pick_covid_result4(covid):
    return [item[3] for item in covid]

def pick_cell_result1(cell):
    return [item[0] for item in cell]
def pick_cell_result2(cell):
    return [item[1] for item in cell]
def pick_cell_result3(cell):
    return [item[2] for item in cell]

with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_paratope/holding_ML_real.p','rb') as f:
    ap_ML = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/benchmark/onehot_paratope/holding_ML_real_onehot.p','rb') as f:
    op_ML = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_pseudo/holding_ML_real_aa_pseudo.p','rb') as f:
    aps_ML = pickle.load(f)

with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_paratope/holding_CNN.p','rb') as f:
    ap_CNN = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/benchmark/onehot_paratope/holding_CNN_onehot.p','rb') as f:
    op_CNN = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_pseudo/holding_CNN_aa_pseudo.p','rb') as f:
    aps_CNN = pickle.load(f)

with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_paratope/holding_reslike.p','rb') as f:
    ap_res = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/benchmark/onehot_paratope/holding_reslike_onehot.p','rb') as f:
    op_res = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_pseudo/holding_reslike_aa_pseudo.p','rb') as f:
    aps_res = pickle.load(f)

ap = np.linspace(1,33,9).astype(np.int)
op = np.linspace(2,34,9).astype(np.int)
aps = np.linspace(3,35,9).astype(np.int)

def get_positions(inp):
    return list(itertools.chain.from_iterable(itertools.repeat(x,10) for x in inp))

def main_plot_ML(ax,model,s=3):

    #ax.set_title(model,fontsize=13)
    ph = ap_ML[model]
    ax.scatter(x=get_positions(ap[0:7]),y=ph['validation'] + ph['dengue'] + pick_cell_result1(ph['cell']) +
                                    pick_covid_result1(ph['covid']) + pick_covid_result2(ph['covid']) +
                                    pick_covid_result3(ph['covid']) + pick_covid_result4(ph['covid']),c='k',s=s)
    ph = op_ML[model]
    ax.scatter(x=get_positions(op[0:7]),y=ph['validation'] + ph['dengue'] + pick_cell_result1(ph['cell']) +
                                    pick_covid_result1(ph['covid']) + pick_covid_result2(ph['covid']) +
                                    pick_covid_result3(ph['covid']) + pick_covid_result4(ph['covid']),c='r',s=s)
    ph = aps_ML[model]
    ax.scatter(x=get_positions(aps[0:7]),y=ph['validation'] + ph['dengue'] + pick_cell_result1(ph['cell']) +
                                    pick_covid_result1(ph['covid']) + pick_covid_result2(ph['covid']) +
                                    pick_covid_result3(ph['covid']) + pick_covid_result4(ph['covid']),c='orange',s=s)
    ax1 = ax.twinx()
    ax1.set_ylim([0,12])
    ph = ap_ML[model]
    ax1.scatter(x=get_positions(ap[7:]),y=pick_cell_result2(ph['cell']) + pick_cell_result3(ph['cell']), c='k',s=s)
    ph = op_ML[model]
    ax1.scatter(x=get_positions(op[7:]),y=pick_cell_result2(ph['cell']) + pick_cell_result3(ph['cell']), c='r',s=s)
    ph = aps_ML[model]
    ax1.scatter(x=get_positions(aps[7:]),y=pick_cell_result2(ph['cell']) + pick_cell_result3(ph['cell']), c='orange',s=s)

    ax.set_xticks(np.linspace(2,34,9).astype(np.int))
    ax.set_xticklabels(['validation','dengue','neoantigen_R','con-R','un-R','con-P','un-P','top20','top50'],rotation=60,
                       fontsize=4)
    ax.tick_params(axis='y',labelsize=2)
    ax1.tick_params(axis='y',labelsize=2)

def main_plot_CNN(ax,s=3):

    #ax.set_title('CNN',fontsize=13)
    ph = ap_CNN
    ax.scatter(x=get_positions(ap[0:7]),y=ph['validation'] + ph['dengue'] + pick_cell_result1(ph['cell']) +
                                    pick_covid_result1(ph['covid']) + pick_covid_result2(ph['covid']) +
                                    pick_covid_result3(ph['covid']) + pick_covid_result4(ph['covid']),c='k',s=s)
    ph = op_CNN
    ax.scatter(x=get_positions(op[0:7]),y=ph['validation'] + ph['dengue'] + pick_cell_result1(ph['cell']) +
                                    pick_covid_result1(ph['covid']) + pick_covid_result2(ph['covid']) +
                                    pick_covid_result3(ph['covid']) + pick_covid_result4(ph['covid']),c='r',s=s)
    ph = aps_CNN
    ax.scatter(x=get_positions(aps[0:7]),y=ph['validation'] + ph['dengue'] + pick_cell_result1(ph['cell']) +
                                    pick_covid_result1(ph['covid']) + pick_covid_result2(ph['covid']) +
                                    pick_covid_result3(ph['covid']) + pick_covid_result4(ph['covid']),c='orange',s=s)
    ax1 = ax.twinx()
    ax1.set_ylim([0,12])
    ph = ap_CNN
    ax1.scatter(x=get_positions(ap[7:]),y=pick_cell_result2(ph['cell']) + pick_cell_result3(ph['cell']), c='k',s=s)
    ph = op_CNN
    ax1.scatter(x=get_positions(op[7:]),y=pick_cell_result2(ph['cell']) + pick_cell_result3(ph['cell']), c='r',s=s)
    ph = aps_CNN
    ax1.scatter(x=get_positions(aps[7:]),y=pick_cell_result2(ph['cell']) + pick_cell_result3(ph['cell']), c='orange',s=s)

    ax.set_xticks(np.linspace(2,34,9).astype(np.int))
    ax.set_xticklabels(['validation','dengue','neoantigen_R','con-R','un-R','con-P','un-P','top20','top50'],rotation=60,
                       fontsize=4)
    ax.tick_params(axis='y',labelsize=2)
    ax1.tick_params(axis='y',labelsize=2)

def main_plot_reslike(ax,s=3):

    #ax.set_title('ResLike',fontsize=13)
    ph = ap_res
    ax.scatter(x=get_positions(ap[0:7]),y=ph['validation'] + ph['dengue'] + pick_cell_result1(ph['cell']) +
                                    pick_covid_result1(ph['covid']) + pick_covid_result2(ph['covid']) +
                                    pick_covid_result3(ph['covid']) + pick_covid_result4(ph['covid']),c='k',s=s)
    ph = op_res
    ax.scatter(x=get_positions(op[0:7]),y=ph['validation'] + ph['dengue'] + pick_cell_result1(ph['cell']) +
                                    pick_covid_result1(ph['covid']) + pick_covid_result2(ph['covid']) +
                                    pick_covid_result3(ph['covid']) + pick_covid_result4(ph['covid']),c='r',s=s)
    ph = aps_res
    ax.scatter(x=get_positions(aps[0:7]),y=ph['validation'] + ph['dengue'] + pick_cell_result1(ph['cell']) +
                                    pick_covid_result1(ph['covid']) + pick_covid_result2(ph['covid']) +
                                    pick_covid_result3(ph['covid']) + pick_covid_result4(ph['covid']),c='orange',s=s)
    ax1 = ax.twinx()
    ax1.set_ylim([0,12])
    ph = ap_res
    ax1.scatter(x=get_positions(ap[7:]),y=pick_cell_result2(ph['cell']) + pick_cell_result3(ph['cell']), c='k',s=s)
    ph = op_res
    ax1.scatter(x=get_positions(op[7:]),y=pick_cell_result2(ph['cell']) + pick_cell_result3(ph['cell']), c='r',s=s)
    ph = aps_res
    ax1.scatter(x=get_positions(aps[7:]),y=pick_cell_result2(ph['cell']) + pick_cell_result3(ph['cell']), c='orange',s=s)

    ax.set_xticks(np.linspace(2,34,9).astype(np.int))
    ax.set_xticklabels(['validation','dengue','neoantigen_R','con-R','un-R','con-P','un-P','top20','top50'],rotation=60,
                       fontsize=4)
    ax.tick_params(axis='y',labelsize=2)
    ax1.tick_params(axis='y',labelsize=2)

fig,ax = plt.subplots(2,4,figsize=(9.6,7.2))
main_plot_ML(ax[0,0],'elasticnet')
main_plot_ML(ax[0,1],'KNN')
main_plot_ML(ax[0,2],'SVR')
main_plot_ML(ax[1,0],'randomforest')
main_plot_ML(ax[1,1],'adaboost')

main_plot_CNN(ax[1,2])
main_plot_reslike(ax[1,3])


h2 = [ax[0,3].plot([],[],color=i,marker='o',ls='',markersize=5)[0] for i in ['black','red','orange']]
ax[0,3].legend(handles=h2,labels=['AAindex + Paratopes', 'Onehot + Paratopes', 'AAindex + HLA_Pseudo34'],frameon=False,fontsize=5)
ax[0,3].axis('off')

plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/ablation_test.pdf')



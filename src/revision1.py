import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from itertools import repeat


# original training, epoch100, epoch150, we decide to choose epoch 150, we need to justify
# that we don't overfit the data, the validation accuracy is still decent
import pickle
with open('/Users/ligk2e/Desktop/github/DeepImmuno/files/benchmark/aaindex_paratope/holding_CNN.p','rb') as f:
    tmp1 = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/revision/holding_CNN_100.p','rb') as f:
    tmp2 = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/revision/holding_cnn_150.p','rb') as f:
    tmp3 = pickle.load(f)

total_df = []
for i,tmp in enumerate([tmp1,tmp2,tmp3]):
    if i == 0: fill = 'Epoch64'
    elif i == 1: fill = 'Epoch100'
    elif i == 2: fill = 'Epoch150'
    # validation set
    slot_validation = tmp['validation']
    id_col = list(repeat('{}'.format(fill),10))
    type_col = list(repeat('validation',10))
    df_validation = pd.DataFrame({'value':slot_validation,'ID':id_col,'type':type_col})
    # dengue
    slot_dengue = tmp['dengue']
    id_col = list(repeat('{}'.format(fill),10))
    type_col = list(repeat('dengue',10))
    df_dengue = pd.DataFrame({'value':slot_dengue,'ID':id_col,'type':type_col})
    # cell paper TESLA
    slot_recall, slot_top20, slot_top50 = zip(*tmp['cell'])
    id_col = list(repeat('{}'.format(fill),10*3))
    type_col = list(repeat('cell_recall',10)) + list(repeat('cell_top20',10)) + list(repeat('cell_top50',10))
    df_cell = pd.DataFrame({'value':list(slot_recall)+list(slot_top20)+list(slot_top50),
                            'ID':id_col,'type':type_col})
    # covid
    slot_recall_con,slot_recall_un,slot_pre_con,slot_pre_un = zip(*tmp['covid'])
    id_col = list(repeat('{}'.format(fill),10*4))
    type_col = list(repeat('covid_recall_con',10)) + list(repeat('covid_recall_un',10)) + \
               list(repeat('covid_precision_con',10)) + list(repeat('covid_precision_un',10))
    df_covid = pd.DataFrame({'value':list(slot_recall_con)+list(slot_recall_un)+list(slot_pre_con) + list(slot_pre_un),
                            'ID':id_col,'type':type_col})
    # concat
    df = pd.concat([df_validation,df_dengue,df_cell,df_covid])
    total_df.append(df)

total = pd.concat(total_df)
total_01 = total.loc[(total['type'] != 'cell_top20') & (total['type'] != 'cell_top50'),]
total_non_01 = total.loc[(total['type'] == 'cell_top20') | (total['type'] == 'cell_top50'),]

fig,ax = plt.subplots()
p = sns.boxplot(x='type',y='value',hue='ID',data=total_non_01,ax=ax)
p_legend = p.get_legend()
p_legend.set_frame_on(False)
plt.savefig('/Users/ligk2e/Desktop/tmp/revision/epochs_performance_non01.pdf',bbox_inches='tight')
plt.close()

fig,ax = plt.subplots()
p = sns.stripplot(x='type',y='value',hue='ID',data=total_01,size=5,dodge=True,ax=ax)
p_legend = p.get_legend()
p_legend.set_frame_on(False)
plt.savefig('/Users/ligk2e/Desktop/tmp/revision/epochs_performance_01.pdf',bbox_inches='tight')
plt.close()


# we want to remove some of the HLA, also label these HLA as unreliable
data = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
legit = []
non_legit = []
for hla,chunk in data.groupby('HLA'):
    absolute = np.count_nonzero(chunk['immunogenicity'].values == 'Negative')
    ratio = np.count_nonzero(chunk['immunogenicity'].values == 'Negative') / len(chunk['immunogenicity'])
    if 'Negative' in chunk['immunogenicity'].tolist():
            legit.append(hla)
    else:
        non_legit.append(hla)


# let's draw a plot for epoch50 and shuffle
from itertools import repeat
with open('/Users/ligk2e/Desktop/immuno3/revision/holding_cnn_150.p','rb') as f:
    tmp1 = pickle.load(f)
with open('/Users/ligk2e/Desktop/immuno3/revision/holding_shuffle.p','rb') as f:
    tmp2 = pickle.load(f)

total_df = []
for i,tmp in enumerate([tmp1,tmp2]):
    if i == 0: fill = 'Epoch150'
    elif i == 1: fill = 'shuffle'
    # validation set
    slot_validation = tmp['validation']
    id_col = list(repeat('{}'.format(fill),10))
    type_col = list(repeat('validation',10))
    df_validation = pd.DataFrame({'value':slot_validation,'ID':id_col,'type':type_col})
    # dengue
    slot_dengue = tmp['dengue']
    id_col = list(repeat('{}'.format(fill),10))
    type_col = list(repeat('dengue',10))
    df_dengue = pd.DataFrame({'value':slot_dengue,'ID':id_col,'type':type_col})
    # cell paper TESLA
    slot_recall, slot_top20, slot_top50 = zip(*tmp['cell'])
    id_col = list(repeat('{}'.format(fill),10*3))
    type_col = list(repeat('cell_recall',10)) + list(repeat('cell_top20',10)) + list(repeat('cell_top50',10))
    df_cell = pd.DataFrame({'value':list(slot_recall)+list(slot_top20)+list(slot_top50),
                            'ID':id_col,'type':type_col})
    # covid
    slot_recall_con,slot_recall_un,slot_pre_con,slot_pre_un = zip(*tmp['covid'])
    id_col = list(repeat('{}'.format(fill),10*4))
    type_col = list(repeat('covid_recall_con',10)) + list(repeat('covid_recall_un',10)) + \
               list(repeat('covid_precision_con',10)) + list(repeat('covid_precision_un',10))
    df_covid = pd.DataFrame({'value':list(slot_recall_con)+list(slot_recall_un)+list(slot_pre_con) + list(slot_pre_un),
                            'ID':id_col,'type':type_col})
    # concat
    df = pd.concat([df_validation,df_dengue,df_cell,df_covid])
    total_df.append(df)

total = pd.concat(total_df)
total_01 = total.loc[(total['type'] != 'cell_top20') & (total['type'] != 'cell_top50'),]
total_non_01 = total.loc[(total['type'] == 'cell_top20') | (total['type'] == 'cell_top50'),]

# we are going to draw three plots in one figure
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(12,6))
gs = mpl.gridspec.GridSpec(nrows=1,ncols=3,width_ratios=(0.1,0.7,0.2),wspace=0.3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

# validation RMSE
valid = total_01.loc[total_01['type'] == 'validation',:]
valid['value'] = - valid['value'].values
import seaborn as sns
p = sns.violinplot(x='type',y='value',hue='ID',data=valid,ax=ax1)
ax1.get_legend().remove()
# other 01 metrics
non_valid = total_01.loc[total_01['type'] != 'validation',:]
p = sns.violinplot(x='type',y='value',hue='ID',data=non_valid,ax=ax2)
ax2.get_legend().remove()
# non-01
p=sns.violinplot(x='type',y='value',hue='ID',data=total_non_01,ax=ax3)
ax3.get_legend().remove()
# save
plt.savefig('/Users/ligk2e/Desktop/tmp/revision/shuffling.pdf',bbox_inches='tight')




# let's test the degree of underfitting to epoch64,epoch100,epoch150
epoch150 = pd.read_csv('/Users/ligk2e/Desktop/tmp/revision/epoch150.txt',sep='\t')
epoch100 = pd.read_csv('/Users/ligk2e/Desktop/tmp/revision/epoch100.txt',sep='\t')
epoch64 = pd.read_csv('/Users/ligk2e/Desktop/tmp/revision/epoch64.txt',sep='\t')

def test(df):
    fp = []
    fn = []
    for i in range(df.shape[0]):
        label = df.iloc[i,:]['immunogenicity.1']
        pred = df.iloc[i,:]['immunogenicity']
        if pred > 0.5 and label == 'Negative':
            fp.append(i)
        elif pred < 0.5 and label != 'Negative':
            fn.append(i)
    return fp,fn

a,b = test(epoch64)

# let's draw the diagnostic plot
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# the figure to reviewer1
fig,ax = plt.subplots()
ax.plot([1,2,3],[0.17,0.20,0.12],marker='o',label='False Positive Rate')
ax.plot([1,2,3],[0.18,0.06,0.05],marker='o',label='False Negative Rate')
ax.set_ylim(0,0.3)
coords = [(1,0.17-0.02),(2,0.20+0.01),(3-0.03,0.12+0.01),(1,0.18+0.01),(2,0.06+0.01),(3-0.03,0.05+0.01)]
texts = [0.17,0.20,0.12,0.18,0.06,0.05]
for coord,text in zip(coords,texts):
    ax.text(x=coord[0],y=coord[1],s=text)
ax.legend(frameon=False)
ax.set_xticks([1,2,3])
ax.set_xticklabels(['epoch64','epoch100','epoch150'])
ax.set_ylabel('performances')
plt.savefig('/Users/ligk2e/Desktop/tmp/revision/training_performance.pdf',bbox_inches='tight')


# illustrate the GAN part
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)
color1 = np.random.choice(['r'],size=10)
color2 = np.random.choice(['b'],size=990)
color = np.concatenate([color1,color2])
np.random.shuffle(color)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x,y,z,c=color,s=6)
ax.legend(handles=[mlines.Line2D([],[],marker='o',linestyle='',color=i) for i in 'rb'],
          labels=['In training data',
                  'NOT in training data'],
          frameon=False)
plt.savefig('/Users/ligk2e/Desktop/tmp/revision/illustrate_gan.pdf',bbox_inches='tight')


# some random


# epoch 100- epoch1000, figures
fig, ax = plt.subplots()
ax.bar(np.arange(5), [0.69, 0.73, 0.76, 0.77, 0.82], color='orange', width=0.4)
ax.set_ylim([0, 1])
ax.plot(np.arange(5), [0.69, 0.73, 0.76, 0.77, 0.82], marker='o', linestyle='-', color='k')
y = [0.69, 0.73, 0.76, 0.77, 0.82]
for i in range(5):
    ax.text(i - 0.1, y[i] + 0.05, s=y[i])
ax.set_xticks(np.arange(5))
ax.set_xticklabels(['epoch100', 'epoch300', 'epoch500', 'epoch750','epoch1000'])
ax.set_ylabel('Proportion of immunogenic peptides')
ax.grid(True, alpha=0.3)
plt.savefig('/Users/ligk2e/Desktop/tmp/revision/analyzer.pdf',bbox_inches='tight')













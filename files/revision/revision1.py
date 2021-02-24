import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from itertools import repeat


# original training, epoch100, epoch150, we decide to choose epoch 150, we need to justify
# that we don't overfit the data, the validation accuracy is still decent

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
sns.boxplot(x='type',y='value',hue='ID',data=total_non_01,ax=ax)
plt.savefig('/Users/ligk2e/Desktop/immuno3/revision/epochs_performance_non01.pdf',bbox_inches='tight')
plt.close()
fig,ax = plt.subplots()
sns.stripplot(x='type',y='value',hue='ID',data=total_01,size=5,dodge=True,ax=ax)
plt.savefig('/Users/ligk2e/Desktop/immuno3/revision/epochs_performance_01.pdf',bbox_inches='tight')
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












'''
for immuno3 project
neoplasma dataset, rigorous data cleaning, 1001 instances for training
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import matplotlib as mpl
import matplotlib.pyplot as plt

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic

def construct_aaindex(ori,hla_dic,after_pca):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1,-1)   # [1,1]

        encode_pep = peptide_data_aaindex(peptide,after_pca)    # [10,12]

        encode_hla = hla_data_aaindex(hla_dic,hla_type,after_pca)   # [46,12]
        series.append((encode_pep, encode_hla, immuno))
    return series

def hla_data_aaindex(hla_dic,hla_type,after_pca):    # return numpy array [34,12,1]
    try:
        seq = hla_dic[hla_type]
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]
    encode = aaindex(seq,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode

def rescue_unknown_hla(hla, dic_inventory):
    type_ = hla[4]
    first2 = hla[6:8]
    last2 = hla[8:]
    big_category = dic_inventory[type_]
    #print(hla)
    if not big_category.get(first2) == None:
        small_category = big_category.get(first2)
        distance = [abs(int(last2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(first2) + str(optimal)
    else:
        small_category = list(big_category.keys())
        distance = [abs(int(first2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(optimal) + str(big_category[optimal][0])

def dict_inventory(inventory):
    dicA, dicB, dicC = {}, {}, {}
    dic = {'A': dicA, 'B': dicB, 'C': dicC}

    for hla in inventory:
        type_ = hla[4]  # A,B,C
        first2 = hla[6:8]  # 01
        last2 = hla[8:]  # 01
        try:
            dic[type_][first2].append(last2)
        except KeyError:
            dic[type_][first2] = []
            dic[type_][first2].append(last2)

    return dic

def peptide_data_aaindex(peptide,after_pca):   # return numpy array [10,12,1]
    length = len(peptide)
    if length == 10:
        encode = aaindex(peptide,after_pca)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = aaindex(peptide,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode

def aaindex(peptide,after_pca):

    amino = 'ARNDCQEGHILKMFPSTWYV-'
    matrix = np.transpose(after_pca)   # [12,21]
    encoded = np.empty([len(peptide), 12])  # (seq_len,12)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]

    return encoded

def draw_cv(trees,bucket):
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(trees)),[item[0] for item in bucket],'o-r')
    ax.fill_between(np.arange(len(trees)),[item[0]-item[1] for item in bucket],[item[0]+item[1] for item in bucket],color='r',alpha=0.2)
    ax.set_ylim(0,1)
    ax.set_xticks(np.arange(len(trees)))
    ax.set_xticklabels(trees)

def draw_ROC(y_true,y_pred):
    from sklearn.metrics import roc_curve,auc
    fpr,tpr,_ = roc_curve(y_true,y_pred,pos_label=1)
    area_mine = auc(fpr,tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % area_mine)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def draw_PR(y_true,y_pred):
    from sklearn.metrics import precision_recall_curve,auc
    precision,recall,_ = precision_recall_curve(y_true,y_pred,pos_label=1)
    area_PR = auc(recall,precision)
    baseline = np.sum(np.array(y_true) == 1) / len(y_true)

    plt.figure()
    lw = 2
    plt.plot(recall,precision, color='darkorange',
            lw=lw, label='PR curve (area = %0.2f)' % area_PR)
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve example')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/all_assay_data.csv')
    data = data.sample(frac=1).set_index(pd.Index(np.arange(data.shape[0])))
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    hla = pd.read_csv('immuno2/data/hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)
    dataset = construct_aaindex(data,hla_dic,after_pca)
    X = np.empty((len(dataset), 12 * 56))
    Y = data['immunogenicity'].values
    for i, (x, y, _) in enumerate(dataset):
        x = x.reshape(-1)
        y = y.reshape(-1)
        X[i, :] = np.concatenate([x, y])

    # random forest
    bucket = []
    trees = [30,100,200,500]
    leaf = [1,5,20,50]
    for item in leaf:
        clf = RandomForestClassifier(min_samples_leaf=item)  #trees = [30,100,200,500]
        cv_result = cross_validate(clf,X,Y,n_jobs=-1,verbose=5)
        bucket.append([cv_result['test_score'].mean(),cv_result['test_score'].std()])
    draw_cv(leaf,bucket)

    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X,Y)


    # cell data
    test = pd.read_csv('data/cell_paper/cell_paper_data_filter910.txt',sep='\t')
    dataset = construct_aaindex(test,hla_dic,after_pca)
    X = np.empty((len(dataset), 12 * 56))
    Y = data['immunogenicity'].values
    for i, (x, y, _) in enumerate(dataset):
        x = x.reshape(-1)
        y = y.reshape(-1)
        X[i, :] = np.concatenate([x, y])
    result = clf.predict_proba(X)
    test['result'] = result[:,1]

    test = test.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(test.shape[0])))
    place = test.index.values[test['immunogenicity'].values.astype(bool)]
    fig,ax = plt.subplots()
    ax.hist(place,bins=20,cumulative=True)

    # dengue data
    data = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/all_assay_data.csv')
    dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/dengue_test.csv')['peptide'].values
    cond = []
    for i in range(data.shape[0]):
        inner = True
        if data['peptide'].iloc[i] in dengue:
            inner = False
        cond.append(inner)
    data_minus = data.loc[cond]
    data_minus = data_minus.set_index(pd.Index(np.arange(data_minus.shape[0])))
    data_minus.to_csv('/Users/ligk2e/Desktop/immuno3/data/all_assay_remove_dengue.csv',index=None)
    dataset = construct_aaindex(data_minus,hla_dic,after_pca)
    X = np.empty((len(dataset), 12 * 56))
    Y = data_minus['immunogenicity'].values
    for i, (x, y, _) in enumerate(dataset):
        x = x.reshape(-1)
        y = y.reshape(-1)
        X[i, :] = np.concatenate([x, y])
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X, Y)

    test = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/dengue_test.csv')
    dataset = construct_aaindex(test,hla_dic,after_pca)
    X = np.empty((len(dataset), 12 * 56))
    Y = test['immunogenicity'].values
    for i, (x, y, _) in enumerate(dataset):
        x = x.reshape(-1)
        y = y.reshape(-1)
        X[i, :] = np.concatenate([x, y])
    result = clf.predict_proba(X)
    test['result'] = result[:,1]

    fig,ax = plt.subplots()
    ax.hist(test['result'],bins=1000,density=True,cumulative=True,histtype='step')
    ax.set_xlim(0,1)
    ax.set_xlabel('predictive immunogenic score')
    ax.set_ylabel('cumulative density')
    ax.set_title('Dengue virus 408 positive epitopes')

    # four neoantigen dataset
    strogen = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/strogen2016.csv')
    dataset = construct_aaindex(strogen,hla_dic,after_pca)
    X = np.empty((len(dataset), 12 * 56))
    Y = strogen['immunogenicity'].values
    for i, (x, y, _) in enumerate(dataset):
        x = x.reshape(-1)
        y = y.reshape(-1)
        X[i, :] = np.concatenate([x, y])
    result = clf.predict_proba(X)
    strogen['result'] = result[:,1]
    draw_ROC(Y,strogen['result'])
    draw_PR(Y,strogen['result'])

    two_groups = []
    for i in strogen.groupby(by='immunogenicity'):
        two_groups.append(i[1]['result'])





















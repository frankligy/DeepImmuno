from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO
from Bio.SubsMat import MatrixInfo
import json
import numpy as np
from collections import Counter
import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler,MinMaxScaler,RobustScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score,accuracy_score
import collections
import itertools
import matplotlib.pyplot as plt
from alternative_encoding import *


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


def blosum50_new(peptide):
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    dic = MatrixInfo.blosum50
    matrix = np.zeros([21, 21])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            try:
                matrix[i, j] = dic[(amino[i], amino[j])]
            except KeyError:
                try:
                    matrix[i, j] = dic[(amino[j], amino[i])]
                except:
                    matrix[i, j] = -1

    encoded = np.empty([len(peptide), 21])  # (seq_len,21)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]

    return encoded


def paratope_dic(hla):
    df = hla
    dic = {}
    for i in range(df.shape[0]):
        hla = df['hla'].iloc[i]
        paratope = df['paratope'].iloc[i]
        dic[hla] = paratope
    return dic


def peptide_data(peptide):   # return numpy array [10,21,1]
    length = len(peptide)
    if length == 10:
        encode = blosum50_new(peptide)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = blosum(peptide)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode


def hla_data(hla, dic_inventory, hla_type):    # return numpy array [46,21,1]
    dic = paratope_dic(hla)
    try:
        seq = dic[hla_type]
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type, dic_inventory)
        seq = dic[hla_type]
    encode = blosum(seq)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode


def construct(ori, hla, dic_inventory):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenecity'].iloc[i]).reshape(1,-1)   # [1,1]

        encode_pep = peptide_data(peptide)    # [10,21]

        encode_hla = hla_data(hla, dic_inventory, hla_type)   # [46,21]
        series.append((encode_pep, encode_hla, immuno))
    return series

def draw_ROC(y_true,y_pred):

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
    precision,recall,cutoff = precision_recall_curve(y_true,y_pred,pos_label=1)
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
    return cutoff

def draw_history(history):
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()


def add_X(array):
    me = np.mean(array)
    array = np.append(array, me)
    return array


def read_index(path):
    with open(path, 'r') as f:
        data = f.readlines()
        array = []
        for line in data:
            line = line.lstrip(' ').rstrip('\n')
            line = re.sub(' +', ' ', line)

            items = line.split(' ')
            items = [float(i) for i in items]
            array.extend(items)
        array = np.array(array)
        array = add_X(array)
        Index = collections.namedtuple('Index',
                                       ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                        'T', 'W', 'Y', 'V', 'X'])
        I = Index._make(array)
    return I, array  # namedtuple


def read_all_indices():
    table = np.empty([21, 566])
    for i in range(566):
        if len(str(i)) == 1:
            ii = '00' + str(i)
        elif len(str(i)) == 2:
            ii = '0' + str(i)
        else:
            ii = str(i)

        NA_list_str = ['472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '520', '523', '524']
        NA_list_int = [int(i) for i in NA_list_str]
        if ii in NA_list_str: continue

        path = '/Users/ligk2e/Desktop/NeoAntigenWorkflow/immunogenecity/AAindex1/index{0}.txt'.format(ii)

        _, array = read_index(path)

        table[:, i] = array
    table = np.delete(table, NA_list_int, 1)
    return table


def scaling(table):  # scale the features
    table_scaled = RobustScaler().fit_transform(table)
    return table_scaled


def wrapper_read_scaling():
    table = read_all_indices()
    table_scaled = scaling(table)
    return table_scaled


def pca_get_components(result):
    pca= PCA()
    pca.fit(result)
    result = pca.explained_variance_ratio_
    sum_ = 0
    for index,var in enumerate(result):
        sum_ += var
        if sum_ > 0.95:
            return index    # 25 components



def pca_apply_reduction(result):   # if 95%, 12 PCs, if 99%, 17 PCs, if 90%,9 PCs
    pca = PCA(n_components=9)  # or strictly speaking ,should be 26, since python is 0-index
    new = pca.fit_transform(result)
    return new


# from an excel to txt
def clean_series(series):  # give a pandas series

    if series.dtype == object:  # pandas will store str as object since string has variable length, you can use astype('|S')
        clean = []
        for item in series:
            item = item.lstrip(' ')  # remove leading whitespace
            item = item.rstrip(' ')  # remove trailing whitespace
            item = item.replace(' ', '')  # replace all whitespace in the middle
            clean.append(item)
    else:
        clean = series

    return pd.Series(clean)


def clean_data_frame(data):  # give a pandas dataFrame

    peptide_clean = clean_series(data['peptide'])
    hla_clean = clean_series(data['HLA'])
    immunogenecity_clean = clean_series(data['immunogenecity'])

    data_clean = pd.concat([peptide_clean, hla_clean, immunogenecity_clean], axis=1)
    data_clean.columns = ['peptide', 'HLA', 'immunogenecity']

    return data_clean


def convert_hla(hla):
    cond = True
    hla = hla.replace(':', '')
    if len(hla) < 9:
        cond = False  # HLA-A3
    elif len(hla) == 9:  # HLA-A3002
        f = hla[0:5]  # HLA-A
        e = hla[5:]  # 3002
        hla = f + '*' + e
    return hla, cond


def convert_hla_series(df):
    new = []
    col = []
    for i in df['HLA']:
        hla, cond = convert_hla(i)
        col.append(cond)
        if cond == True: new.append(hla)
    df = df.loc[pd.Series(col)]
    df = df.set_index(pd.Index(np.arange(df.shape[0])))
    df['HLA'] = new
    return df


def test_no_space(series):
    for i in series:
        if ' ' in i:
            print('damn')
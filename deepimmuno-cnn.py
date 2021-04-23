

'''
Program to run deepimmuno-cnn
'''


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import argparse
import os


def seperateCNN():
    input1 = keras.Input(shape=(10, 12, 1))
    input2 = keras.Input(shape=(46, 12, 1))

    x = layers.Conv2D(filters=16, kernel_size=(2, 12))(input1)  # 9
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
    x = layers.Flatten()(x)
    x = keras.Model(inputs=input1, outputs=x)

    y = layers.Conv2D(filters=16, kernel_size=(15, 12))(input2)     # 32
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)  # 16
    y = layers.Conv2D(filters=32,kernel_size=(9,1))(y)    # 8
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  # 4
    y = layers.Flatten()(y)
    y = keras.Model(inputs=input2,outputs=y)

    combined = layers.concatenate([x.output,y.output])
    z = layers.Dense(128,activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1,activation='sigmoid')(z)

    model = keras.Model(inputs=[input1,input2],outputs=z)
    return model

def pull_peptide_aaindex(dataset):
    result = np.empty([len(dataset),10,12,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][0]
    return result


def pull_hla_aaindex(dataset):
    result = np.empty([len(dataset),46,12,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][1]
    return result


def pull_label_aaindex(dataset):
    col = [item[2] for item in dataset]
    result = [0 if item == 'Negative' else 1 for item in col]
    result = np.expand_dims(np.array(result),axis=1)
    return result

def pull_label_aaindex(dataset):
    result = np.empty([len(dataset),1])
    for i in range(len(dataset)):
        result[i,:] = dataset[i][2]
    return result

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


def peptide_data_aaindex(peptide,after_pca):   # return numpy array [10,12,1]
    length = len(peptide)
    if length == 10:
        encode = aaindex(peptide,after_pca)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = aaindex(peptide,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode

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






def hla_data_aaindex(hla_dic,hla_type,after_pca,dic_inventory):    # return numpy array [46,12,1]
    try:
        seq = hla_dic[hla_type]
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]
    encode = aaindex(seq,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode

def construct_aaindex(ori,hla_dic,after_pca,dic_inventory):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1,-1)   # [1,1]

        encode_pep = peptide_data_aaindex(peptide,after_pca)    # [10,12]

        encode_hla = hla_data_aaindex(hla_dic,hla_type,after_pca,dic_inventory)   # [46,12]
        series.append((encode_pep, encode_hla, immuno))
    return series

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic



def computing_s(peptide,mhc):

    after_pca = np.loadtxt('./data/after_pca.txt')
    hla = pd.read_csv('./data/hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)
    cnn_model = seperateCNN()
    cnn_model.save_weights('./models/cnn_model_331_3_7/')
    peptide_score = [peptide]
    hla_score = [mhc]
    immuno_score = ['0']
    ori_score = pd.DataFrame({'peptide':peptide_score,'HLA':hla_score,'immunogenicity':immuno_score})
    dataset_score = construct_aaindex(ori_score,hla_dic,after_pca,dic_inventory)
    input1_score = pull_peptide_aaindex(dataset_score)
    input2_score = pull_hla_aaindex(dataset_score)
    label_score = pull_label_aaindex(dataset_score)
    scoring = cnn_model.predict(x=[input1_score,input2_score])
    return float(scoring)


def file_process(upload,download,result_file='deepimmuno-cnn-result.txt'):

    after_pca = np.loadtxt('./data/after_pca.txt')
    hla = pd.read_csv('./data/hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)
    cnn_model = seperateCNN()
    cnn_model.save_weights('./models/cnn_model_331_3_7/')

    ori_score = pd.read_csv(upload, sep=',', header=None)
    ori_score.columns = ['peptide', 'HLA']
    ori_score['immunogenicity'] = ['0'] * ori_score.shape[0]
    dataset_score = construct_aaindex(ori_score, hla_dic, after_pca, dic_inventory)

    input1_score = pull_peptide_aaindex(dataset_score)
    input2_score = pull_hla_aaindex(dataset_score)
    label_score = pull_label_aaindex(dataset_score)
    scoring = cnn_model.predict(x=[input1_score, input2_score])
    scoring = cnn_model.predict(x=[input1_score, input2_score])
    ori_score['immunogenicity'] = scoring
    ori_score.to_csv(os.path.join(download,result_file), sep='\t', index=None)

def main(args):
    mode = args.mode
    if mode == 'single':
        print("mode is single")
        epitope = args.epitope
        print("queried epitope is {}".format(epitope))
        hla= args.hla
        print("queried epitope is {}".format(hla))
        score = computing_s(epitope,hla)
        print(score)
    elif mode == 'multiple':
        print("mode is multiple")
        intFile = args.intdir
        print("input file is {}".format(intFile))
        outFolder = args.outdir
        print("output will be in {}".format(outFolder))
        outName = args.outname
        print("output name will be {}".format(outName))
        file_process(intFile,outFolder,outName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepImmuno-CNN command line')
    parser.add_argument('--mode',type=str,default='single',help='single mode or multiple mode')
    parser.add_argument('--epitope',type=str,default=None,help='if single mode, specifying your epitope')
    parser.add_argument('--hla',type=str,default=None,help='if single mode, specifying your HLA allele')
    parser.add_argument('--intdir',type=str,default=None,help='if multiple mode, specifying the path to your input file')
    parser.add_argument('--outdir',type=str,default=None,help='if multiple mode, specifying the path to your output folder')
    parser.add_argument('--outname',type=str,default='deepimmuno-cnn-result.txt',help='specify the name of the output txt file. Default: deepimmuno-cnn-result.txt')
    args = parser.parse_args()
    main(args)

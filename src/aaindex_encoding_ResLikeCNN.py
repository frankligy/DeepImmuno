import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score,accuracy_score
import matplotlib.pyplot as plt
import numpy as np




class ResBlock(layers.Layer):
    def __init__(self,in_channel,pool_size):
        super(ResBlock,self).__init__()
        intermediate_channel = in_channel
        out_channel = in_channel * 2
        self.conv1 = layers.Conv2D(filters=intermediate_channel,kernel_size=(1,1),strides=(1,1),padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters=intermediate_channel,kernel_size=(3,1),strides=(1,1),padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filters=out_channel,kernel_size=(1,1),strides=(1,1),padding='same')
        self.bn3 = layers.BatchNormalization()
        self.identity = layers.Conv2D(filters=out_channel,kernel_size=(1,1),strides=(1,1))
        self.maxpool = layers.MaxPool2D(pool_size=pool_size,strides=pool_size)



    def call(self,x):
        out = keras.activations.relu(self.bn1(self.conv1(x)))   # (8,1,16)
        out = keras.activations.relu(self.bn2(self.conv2(out)))  # (8,1,16)
        out = keras.activations.relu(self.bn3(self.conv3(out)))   # (8,1,32)
        identity_map = self.identity(x)   # change original input (8,1,16)  --> (8,1,32)
        out = out + identity_map    # (8,1,32)
        out = self.maxpool(out)    # (4,1,32)

        return out


class CNN_peptide_aaindex(layers.Layer):
    def __init__(self):
        super(CNN_peptide_aaindex,self).__init__()
        self.conv = layers.Conv2D(filters=16,kernel_size=(3,12),strides=(1,1))
        self.block1 = ResBlock(16,(2,1))
        self.block2 = ResBlock(32,(2,1))
        self.block3 = ResBlock(64,(2,1))

    def call(self,x):    # (10,21,1)
        out = self.conv(x)   # (8,1,16)
        out = self.block1(out)   # (4,1,32)
        out = self.block2(out)   # (2,1,64)
        out = self.block3(out)   # (1,1,128)
        return out


class CNN_MHC_aaindex(layers.Layer):
    def __init__(self):
        super(CNN_MHC_aaindex,self).__init__()
        self.conv = layers.Conv2D(filters=16,kernel_size=(15,12),strides=(1,1)) # (32,1,16)
        self.block1 = ResBlock(16, (2, 1))    # (16,1,32)
        self.block2 = ResBlock(32, (2, 1))    # (8,1,64)
        self.block3 = ResBlock(64, (2, 1))    # (4,1,128)
        self.conv_add = layers.Conv2D(filters=128,kernel_size=(4,1),strides=(1,1))
        self.bn = layers.BatchNormalization()


    def call(self, x):
        out = self.conv(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = keras.activations.relu(self.bn(self.conv_add(out)))   # (1,1,128)
        return out


class model_aaindex(keras.Model):
    def __init__(self):
        super(model_aaindex,self).__init__()
        self.br_pep = CNN_peptide_aaindex()
        self.br_mhc = CNN_MHC_aaindex()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128,activation='relu')
        self.fc2 = layers.Dense(1,activation='sigmoid')

    def call(self,input):
        x1,x2 = input[0],input[1]  # x1: (10,12,1)    x2: (46,12,1)
        out1 = self.flatten(self.br_pep(x1))
        out2 = self.flatten(self.br_mhc(x2))
        out = layers.concatenate([out1,out2])
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def model(self):
        x1 = keras.Input(shape=(10,12,1))
        x2 = keras.Input(shape=(46,12,1))
        return keras.Model(inputs=[x1,x2],outputs=self.call([x1,x2]))



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

# def aaindex(peptide,after_pca):
#     amino = 'ARNDCQEGHILKMFPSTWYV-'
#     encoded = np.empty([len(peptide),21])
#     onehot = np.identity(21)
#     for i in range(len(peptide)):
#         query = peptide[i]
#         if query == 'X': query = '-'
#         query = query.upper()
#         encoded[i,:] = onehot[:,amino.index(query)]
#     return encoded






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
    result = np.empty([len(dataset),1])
    for i in range(len(dataset)):
        result[i,:] = dataset[i][2]
    return result



def retain_910(ori):
    cond = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        if len(peptide) == 9 or len(peptide) == 10:
            cond.append(True)
        else:
            cond.append(False)
    data = ori.loc[cond]
    data = data.set_index(pd.Index(np.arange(data.shape[0])))
    return data

def wrapper_train():
    # train
    reslike_model = model_aaindex()
    reslike_model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=['accuracy'])
    callback_val = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=False)
    callback_train = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=False)
    history = reslike_model.fit(
        x=[train_input1, train_input2],  # feed a list into
        y=train_label,
        validation_data=([test_input1, test_input2], test_label),
        batch_size=128,
        epochs=200,
        class_weight={0: 0.5, 1: 0.5},  # I have 20% positive and 80% negative in my training data
        callbacks=[callback_val, callback_train])
    return reslike_model

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic

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






def hla_data_aaindex(hla_dic,hla_type,after_pca):    # return numpy array [34,12,1]
    try:
        seq = hla_dic[hla_type]
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]
    encode = aaindex(seq,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode

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

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic

if __name__ == '__main__':
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    ori = ori.sample(frac=1,replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    hla = pd.read_csv('immuno2/data/hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    ori['immunogenicity'],ori['potential'] = ori['potential'],ori['immunogenicity']

    dataset = construct_aaindex(ori,hla_dic,after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(input1.shape[0])))
    holding = {'validation':[],'dengue':[],'cell':[],'covid':[]}
    i = 1
    for fold in fold_indices:
        # split
        ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori['immunogenicity'], ori['potential'] = ori['potential'], ori['immunogenicity']
        dataset = construct_aaindex(ori, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        train_input1, train_input2, train_label = input1[fold[0]],input2[fold[0]],label[fold[0]]
        test_input1, test_input2, test_label = input1[fold[1]],input2[fold[1]],label[fold[1]]
        print('round {}, split finished'.format(i))
        # train
        reslike_model = wrapper_train()
        # predict in validation set
        result = reslike_model.predict([test_input1,test_input2])
        from sklearn.metrics import mean_squared_error
        loss = mean_squared_error(test_label,result,squared=False)
        holding['validation'].append(loss)
        print('round {}, finished validation'.format(i))
        # predict in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = reslike_model.predict([input1,input2])
        from sklearn.metrics import accuracy_score,recall_score,precision_score
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(label, hard)


        holding['dengue'].append(result)
        print('round {}, finished dengue'.format(i))
        # predict in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = reslike_model.predict([input1,input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(label, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50


        holding['cell'].append((result1, result2, result3))
        print('round {}, finished cell'.format(i))
        # predict in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = reslike_model.predict([input1,input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall

        holding['covid'].append((result1, result2, result3, result4))
        print('round {}, finished covid'.format(i))
        i += 1

    # onehot + paratope
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    ori = ori.sample(frac=1,replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    hla = pd.read_csv('immuno2/data/hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    ori['immunogenicity'],ori['potential'] = ori['potential'],ori['immunogenicity']

    dataset = construct_aaindex(ori,hla_dic,after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(input1.shape[0])))
    holding = {'validation':[],'dengue':[],'cell':[],'covid':[]}
    i = 1
    for fold in fold_indices:
        # split
        ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori['immunogenicity'], ori['potential'] = ori['potential'], ori['immunogenicity']
        dataset = construct_aaindex(ori, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        train_input1, train_input2, train_label = input1[fold[0]],input2[fold[0]],label[fold[0]]
        test_input1, test_input2, test_label = input1[fold[1]],input2[fold[1]],label[fold[1]]
        print('round {}, split finished'.format(i))
        # train
        reslike_model = wrapper_train()
        # predict in validation set
        result = reslike_model.predict([test_input1,test_input2])
        from sklearn.metrics import mean_squared_error
        loss = mean_squared_error(test_label,result,squared=False)
        holding['validation'].append(loss)
        print('round {}, finished validation'.format(i))
        # predict in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = reslike_model.predict([input1,input2])
        from sklearn.metrics import accuracy_score,recall_score,precision_score
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(label, hard)


        holding['dengue'].append(result)
        print('round {}, finished dengue'.format(i))
        # predict in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = reslike_model.predict([input1,input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(label, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50


        holding['cell'].append((result1, result2, result3))
        print('round {}, finished cell'.format(i))
        # predict in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = reslike_model.predict([input1,input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall

        holding['covid'].append((result1, result2, result3, result4))
        print('round {}, finished covid'.format(i))
        i += 1

# aaindex + pseudo
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    ori = ori.sample(frac=1,replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    hla = pd.read_csv('immuno2/data/pseudo34_clean.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    ori['immunogenicity'],ori['potential'] = ori['potential'],ori['immunogenicity']

    dataset = construct_aaindex(ori,hla_dic,after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(input1.shape[0])))
    holding = {'validation':[],'dengue':[],'cell':[],'covid':[]}
    i = 1
    for fold in fold_indices:
        # split
        ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori['immunogenicity'], ori['potential'] = ori['potential'], ori['immunogenicity']
        dataset = construct_aaindex(ori, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        train_input1, train_input2, train_label = input1[fold[0]],input2[fold[0]],label[fold[0]]
        test_input1, test_input2, test_label = input1[fold[1]],input2[fold[1]],label[fold[1]]
        print('round {}, split finished'.format(i))
        # train
        reslike_model = wrapper_train()
        # predict in validation set
        result = reslike_model.predict([test_input1,test_input2])
        from sklearn.metrics import mean_squared_error
        loss = mean_squared_error(test_label,result,squared=False)
        holding['validation'].append(loss)
        print('round {}, finished validation'.format(i))
        # predict in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = reslike_model.predict([input1,input2])
        from sklearn.metrics import accuracy_score,recall_score,precision_score
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(label, hard)


        holding['dengue'].append(result)
        print('round {}, finished dengue'.format(i))
        # predict in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = reslike_model.predict([input1,input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(label, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50


        holding['cell'].append((result1, result2, result3))
        print('round {}, finished cell'.format(i))
        # predict in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = reslike_model.predict([input1,input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall

        holding['covid'].append((result1, result2, result3, result4))
        print('round {}, finished covid'.format(i))
        i += 1




'''
this script is for revision,
1. HLA sequence representation
2. encoding strategy
3. number of PCs
'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc,precision_recall_curve,roc_curve,confusion_matrix


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

def seperateCNN():
    input1 = keras.Input(shape=(10, 17, 1))
    input2 = keras.Input(shape=(46, 17, 1))

    x = layers.Conv2D(filters=16, kernel_size=(2, 17))(input1)  # 9
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
    x = layers.Flatten()(x)
    x = keras.Model(inputs=input1, outputs=x)

    y = layers.Conv2D(filters=16, kernel_size=(3, 17))(input2)     # 32
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
    result = np.empty([len(dataset),10,17,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][0]
    return result


def pull_hla_aaindex(dataset):
    result = np.empty([len(dataset),46,17,1])
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
    encoded = np.empty([len(peptide), 17])  # (seq_len,12)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]

    return encoded

def aaindex(peptide,after_pca):
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    encoded = np.empty([len(peptide),21])
    onehot = np.identity(21)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i,:] = onehot[:,amino.index(query)]
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
        immuno = np.array(ori['immunogenicity-con'].iloc[i]).reshape(1,-1)   # [1,1]

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


def draw_history(history):
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()
    plt.show()

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

if __name__ == '__main__':
    # let's graph the comparison between ML and DL
    import pickle
    with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_pseudo/holding_ML_real_aa_pseudo.p','rb') as f:
        holding_ML = pickle.load(f)
    with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_pseudo/holding_CNN_aa_pseudo.p','rb') as f:
        holding_CNN = pickle.load(f)
    with open('/Users/ligk2e/Desktop/immuno3/benchmark/aaindex_pseudo/holding_reslike_aa_pseudo.p','rb') as f:
        holding_res = pickle.load(f)

    import itertools
    ax = plt.subplot(3,3,1)
    ax.scatter(list(itertools.repeat([1,2,3,4,5,6,7],10)), holding_ML['elasticnet']['validation']
               + holding_ML['KNN']['validation'] + holding_ML['SVR']['validation'] + holding_ML['randomforest']['validation']
               + holding_ML['adaboost']['validation'] + holding_CNN['validation'] + holding_res['validation'],color='black',
               s=5)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
    ax.set_title('validation RMSE',fontsize=8)

    ax = plt.subplot(3,3,2)
    ax.scatter(list(itertools.repeat([1,2,3,4,5,6,7],10)), holding_ML['elasticnet']['dengue']
               + holding_ML['KNN']['dengue'] + holding_ML['SVR']['dengue'] + holding_ML['randomforest']['dengue']
               + holding_ML['adaboost']['dengue'] + holding_CNN['dengue'] + holding_res['dengue'],color='black',
               s=5)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
    ax.set_title('dengue accuracy',fontsize=8)

    def pick_cell_result1(cell):
        return [item[0] for item in cell]
    def pick_cell_result2(cell):
        return [item[1] for item in cell]
    def pick_cell_result3(cell):
        return [item[2] for item in cell]
    ax = plt.subplot(3,3,3)
    ax.scatter(list(itertools.repeat([1,2,3,4,5,6,7],10)), pick_cell_result1(holding_ML['elasticnet']['cell'])
               + pick_cell_result1(holding_ML['KNN']['cell']) + pick_cell_result1(holding_ML['SVR']['cell'])
               + pick_cell_result1(holding_ML['randomforest']['cell'])
               + pick_cell_result1(holding_ML['adaboost']['cell']) + pick_cell_result1(holding_CNN['cell'])
               + pick_cell_result1(holding_res['cell']),color='black',s=5)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
    ax.set_title('neoantigen recall',fontsize=8)

    ax = plt.subplot(3,3,4)
    ax.scatter(list(itertools.repeat([1,2,3,4,5,6,7],10)), pick_cell_result2(holding_ML['elasticnet']['cell'])
               + pick_cell_result2(holding_ML['KNN']['cell']) + pick_cell_result2(holding_ML['SVR']['cell'])
               + pick_cell_result2(holding_ML['randomforest']['cell'])
               + pick_cell_result2(holding_ML['adaboost']['cell']) + pick_cell_result2(holding_CNN['cell'])
               + pick_cell_result2(holding_res['cell']),color='black',s=5)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
    ax.set_title('neoantigen top20',fontsize=8)

    ax = plt.subplot(3,3,5)
    ax.scatter(list(itertools.repeat([1,2,3,4,5,6,7],10)), pick_cell_result3(holding_ML['elasticnet']['cell'])
               + pick_cell_result3(holding_ML['KNN']['cell']) + pick_cell_result3(holding_ML['SVR']['cell'])
               + pick_cell_result3(holding_ML['randomforest']['cell'])
               + pick_cell_result3(holding_ML['adaboost']['cell']) + pick_cell_result3(holding_CNN['cell'])
               + pick_cell_result3(holding_res['cell']),color='black',s=5)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
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
    ax.scatter(list(itertools.repeat([1,2,3,4,5,6,7],10)), pick_covid_result1(holding_ML['elasticnet']['covid'])
               + pick_covid_result1(holding_ML['KNN']['covid']) + pick_covid_result1(holding_ML['SVR']['covid'])
               + pick_covid_result1(holding_ML['randomforest']['covid'])
               + pick_covid_result1(holding_ML['adaboost']['covid']) + pick_covid_result1(holding_CNN['covid'])
               + pick_covid_result1(holding_res['covid']),color='black',s=5)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
    ax.set_title('covid convalescent recall',fontsize=8)

    ax = plt.subplot(3,3,7)
    ax.scatter(list(itertools.repeat([1,2,3,4,5,6,7],10)), pick_covid_result2(holding_ML['elasticnet']['covid'])
               + pick_covid_result2(holding_ML['KNN']['covid']) + pick_covid_result2(holding_ML['SVR']['covid'])
               + pick_covid_result2(holding_ML['randomforest']['covid'])
               + pick_covid_result2(holding_ML['adaboost']['covid']) + pick_covid_result2(holding_CNN['covid'])
               + pick_covid_result2(holding_res['covid']),color='black',s=5)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
    ax.set_title('covid unexposed recall',fontsize=8)

    ax = plt.subplot(3,3,8)
    ax.scatter(list(itertools.repeat([1,2,3,4,5,6,7],10)), pick_covid_result3(holding_ML['elasticnet']['covid'])
               + pick_covid_result3(holding_ML['KNN']['covid']) + pick_covid_result3(holding_ML['SVR']['covid'])
               + pick_covid_result3(holding_ML['randomforest']['covid'])
               + pick_covid_result3(holding_ML['adaboost']['covid']) + pick_covid_result3(holding_CNN['covid'])
               + pick_covid_result3(holding_res['covid']),color='black',s=5)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
    ax.set_title('covid convalescent precision',fontsize=8)

    ax = plt.subplot(3,3,9)
    ax.scatter(list(itertools.repeat([1,2,3,4,5,6,7],10)), pick_covid_result4(holding_ML['elasticnet']['covid'])
               + pick_covid_result4(holding_ML['KNN']['covid']) + pick_covid_result4(holding_ML['SVR']['covid'])
               + pick_covid_result4(holding_ML['randomforest']['covid'])
               + pick_covid_result4(holding_ML['adaboost']['covid']) + pick_covid_result4(holding_CNN['covid'])
               + pick_covid_result4(holding_res['covid']),color='black',s=5)
    ax.set_xticks([1,2,3,4,5,6,7])
    ax.set_xticklabels(['eNet','KNN','SVR','RF','ada','CNN','res'],fontsize=6)
    ax.set_title('covid unexposed precision',fontsize=8)















    # what if using 34 pseudo-sequence
    after_pca = np.loadtxt('immuno2/data/after_pca17.txt')
    ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123.csv')
    ori = ori.sample(frac=1,replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    hla = pd.read_csv('immuno2/data/pseudo34_clean.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    dataset = construct_aaindex(ori,hla_dic,after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)

    cnn_model = seperateCNN()
    cnn_model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=['accuracy'])

    # let's do a simple train, validation split
    array = np.arange(len(dataset))
    train_index = np.random.choice(array,int(len(dataset)*0.9),replace=False)
    valid_index = [item for item in array if item not in train_index]

    input1_train = input1[train_index]
    input1_valid = input1[valid_index]
    input2_train = input2[train_index]
    input2_valid = input2[valid_index]
    label_train = label[train_index]
    label_valid = label[valid_index]

    callback_val = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,restore_best_weights=False)
    callback_train = keras.callbacks.EarlyStopping(monitor='loss',patience=2,restore_best_weights=False)
    history = cnn_model.fit(
        x=[input1_train,input2_train],   # feed a list into
        y=label_train,
        validation_data = ([input1_valid,input2_valid],label_valid),
        batch_size=128,
        epochs=200,
        class_weight = {0:0.5,1:0.5},   # I have 20% positive and 80% negative in my training data
        callbacks = [callback_val,callback_train])
    valid = ori.loc[valid_index]
    valid['cnn_regress'] = cnn_model.predict([input1_valid,input2_valid])
    valid = valid.sort_values(by='cnn_regress',ascending=False).set_index(pd.Index(np.arange(valid.shape[0])))
    y_true = [1 if not item == 'Negative' else 0 for item in valid['immunogenicity']]
    y_pred = valid['cnn_regress']
    from sklearn.metrics import auc,roc_curve
    precision,recall,_ = precision_recall_curve(y_true,y_pred)
    area = auc(recall,precision)

    # in dengue, neoantigen dataset
    ori_test = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
    testing_dataset = construct_aaindex(ori_test,hla_dic,after_pca)
    input1_test = pull_peptide_aaindex(testing_dataset)
    input2_test = pull_hla_aaindex(testing_dataset)
    label_test = pull_label_aaindex(testing_dataset)

    result = cnn_model.predict(x=[input1_test,input2_test])
    ori_test['result'] = result[:,0]
    ori_test = ori_test.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(ori_test.shape[0])))

    # testing-cell dataset
    ori_test =pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt',sep='\t')
    testing_dataset = construct_aaindex(ori_test,hla_dic,after_pca)
    input1_test = pull_peptide_aaindex(testing_dataset)
    input2_test = pull_hla_aaindex(testing_dataset)
    label_test = pull_label_aaindex(testing_dataset)

    result = cnn_model.predict(x=[input1_test,input2_test])
    ori_test['result'] = result[:,0]
    ori_test = ori_test.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(ori_test.shape[0])))

    # test in covid dataset
    ori_test = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt',sep='\t')
    ori_test = retain_910(ori_test)
    testing_dataset = construct_aaindex(ori_test,hla_dic,after_pca)
    input1_test = pull_peptide_aaindex(testing_dataset)
    input2_test = pull_hla_aaindex(testing_dataset)

    result = cnn_model.predict(x=[input1_test,input2_test])
    ori_test['result'] = result[:,0]
    ori_test = ori_test.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(ori_test.shape[0])))
    draw_PR(ori_test['immunogenicity-un'],ori_test['result'])
    draw_ROC(ori_test['immunogenicity-un'],ori_test['result'])

    # save models
    cnn_model.save_weights('immuno2/models/cnn_34pseudo_aaindex12_mymodel_2_6_26_15_7/')  # first conv1d for mhc is kernel-size=3











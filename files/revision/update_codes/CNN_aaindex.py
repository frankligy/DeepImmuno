'''
this script is going to use pseudo34 as well, but use ordinary CNN model
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

def wrapper_train():
    # train
    cnn_model = seperateCNN()
    cnn_model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=['accuracy'])
    #callback_val = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=False)
    #callback_train = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=False)
    history = cnn_model.fit(
        x=[train_input1, train_input2],  # feed a list into
        y=train_label,
        validation_data=([test_input1, test_input2], test_label),
        batch_size=128,
        epochs=150,
        class_weight={0: 0.5, 1: 0.5},)  # I have 20% positive and 80% negative in my training data
        #callbacks=[callback_val, callback_train])
    return cnn_model

if __name__ == '__main__':
    # aaindex + paratopes
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

    # for revision, we spotted some underfitting issue, and want to test if training more epoch would help
    from sklearn.model_selection import ShuffleSplit
    ss = ShuffleSplit(n_splits=1)
    fold = list(ss.split(input1.reshape(input1.shape[0],-1)))[0]
    train_input1, train_input2, train_label = input1[fold[0]], input2[fold[0]], label[fold[0]]
    test_input1, test_input2, test_label = input1[fold[1]], input2[fold[1]], label[fold[1]]
    cnn_model = seperateCNN()
    cnn_model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=['accuracy'])
    #callback_val = keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=False)
    #callback_train = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=False)
    history = cnn_model.fit(
        x=[train_input1, train_input2],  # feed a list into
        y=train_label,
        validation_data=([test_input1, test_input2], test_label),
        batch_size=128,
        epochs=100,
        class_weight={0: 0.5, 1: 0.5},)  # I have 20% positive and 80% negative in my training data
        #callbacks=[callback_val, callback_train]

    cnn_model.save_weights('/Users/ligk2e/Desktop/immuno3/revision/cnn_ptrain5_ptest50_e157_tl_0190_vl_0741/')
    cnn_model.save_weights('/Users/ligk2e/Desktop/immuno3/revision/cnn_no_callback_e500_tl_0069_vl_0888/')
    cnn_model.save_weights('/Users/ligk2e/Desktop/immuno3/revision/cnn_no_callback_e100_tl_0247_vl_0745/')

    '''
    Now the conclusion is: the issue is indeed due to underfit, we now want to know, if we let the training go a little bit 
    longer, would our four validation fail?
    
    based on the curve we drew, it seems like epoch 100 is a reasonable stopping point, but I also want to see if increase to 150
    what would happen?
    
    so I am going to borrow previous kfold vaildation code base and test it, see below
    
    1. first change the wrapper train function a bit, adjust to epoch 100, remove callback funcion
    2. after doing above, we can see epoch100 doesn't observe big compromise in testing case, but the consistency of high immugenic
    peptide in training set is still not that, high, so we want to test what if epoch increase to 150, because epoch 150 seems
    generate decent consitency in the training dataset
    '''

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(input1.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
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
        train_input1, train_input2, train_label = input1[fold[0]], input2[fold[0]], label[fold[0]]
        test_input1, test_input2, test_label = input1[fold[1]], input2[fold[1]], label[fold[1]]
        print('round {}, split finished'.format(i))
        # train
        cnn_model = wrapper_train()
        # predict in validation set
        result = cnn_model.predict([test_input1, test_input2])
        from sklearn.metrics import mean_squared_error

        loss = mean_squared_error(test_label, result, squared=False)
        holding['validation'].append(loss)
        print('round {}, finished validation'.format(i))
        # predict in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = cnn_model.predict([input1, input2])
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(label, hard)

        if result > 0.86:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_dengue_nice/')

        holding['dengue'].append(result)
        print('round {}, finished dengue'.format(i))
        # predict in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = cnn_model.predict([input1, input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(label, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50

        if result1 >= 0.8 and result2 >= 4 and result3 >= 8:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_neoantigen_nice/')

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
        prediction = cnn_model.predict([input1, input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        if result1 >= 0.76 and result2 >= 0.875:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_covid_nice/')
        holding['covid'].append((result1, result2, result3, result4))
        print('round {}, finished covid'.format(i))
        i += 1


    '''
    Then the reviewer2 said, what if you shuffle the label, we should expect a decrease of the performance
    Let's test it.  
    '''

    # did the same thing until getting the input1, input2, label
    # split up but only shuffle the label in training dataset

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(input1.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
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
        train_input1, train_input2, train_label = input1[fold[0]], input2[fold[0]], label[fold[0]]
        test_input1, test_input2, test_label = input1[fold[1]], input2[fold[1]], label[fold[1]]
        print('round {}, split finished'.format(i))
        # shuffle training labels
        np.random.shuffle(train_label)
        print('round {}, shuffle finished'.format(i))
        # train
        cnn_model = wrapper_train()
        # predict in validation set
        result = cnn_model.predict([test_input1, test_input2])
        from sklearn.metrics import mean_squared_error

        loss = mean_squared_error(test_label, result, squared=False)
        holding['validation'].append(loss)
        print('round {}, finished validation'.format(i))
        # predict in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = cnn_model.predict([input1, input2])
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(label, hard)

        if result > 0.86:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_dengue_nice/')

        holding['dengue'].append(result)
        print('round {}, finished dengue'.format(i))
        # predict in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = cnn_model.predict([input1, input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(label, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50

        if result1 >= 0.8 and result2 >= 4 and result3 >= 8:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_neoantigen_nice/')

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
        prediction = cnn_model.predict([input1, input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        if result1 >= 0.76 and result2 >= 0.875:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_covid_nice/')
        holding['covid'].append((result1, result2, result3, result4))
        print('round {}, finished covid'.format(i))
        i += 1

    # save the holding
    import pickle
    with open('/Users/ligk2e/Desktop/immuno3/revision/holding_shuffle.p','wb') as f:
        pickle.dump(holding,f)


    # the CNN benchmark from original manucript preparation
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
        cnn_model = wrapper_train()
        # predict in validation set
        result = cnn_model.predict([test_input1,test_input2])
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
        prediction = cnn_model.predict([input1,input2])
        from sklearn.metrics import accuracy_score,recall_score,precision_score
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(label, hard)

        if result > 0.86:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_dengue_nice/')

        holding['dengue'].append(result)
        print('round {}, finished dengue'.format(i))
        # predict in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = cnn_model.predict([input1,input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(label, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50

        if result1 >= 0.8 and result2 >= 4 and result3 >= 8:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_neoantigen_nice/')

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
        prediction = cnn_model.predict([input1,input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        if result1 >= 0.76 and result2 >= 0.875:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_covid_nice/')
        holding['covid'].append((result1, result2, result3, result4))
        print('round {}, finished covid'.format(i))
        i += 1

    # onehot + paratope
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    hla = pd.read_csv('immuno2/data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    ori['immunogenicity'], ori['potential'] = ori['potential'], ori['immunogenicity']

    dataset = construct_aaindex(ori, hla_dic, after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(input1.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
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
        train_input1, train_input2, train_label = input1[fold[0]], input2[fold[0]], label[fold[0]]
        test_input1, test_input2, test_label = input1[fold[1]], input2[fold[1]], label[fold[1]]
        print('round {}, split finished'.format(i))
        # train
        cnn_model = wrapper_train()
        # predict in validation set
        result = cnn_model.predict([test_input1, test_input2])
        from sklearn.metrics import mean_squared_error

        loss = mean_squared_error(test_label, result, squared=False)
        holding['validation'].append(loss)
        print('round {}, finished validation'.format(i))
        # predict in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = cnn_model.predict([input1, input2])
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(label, hard)

        if result > 0.86:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_dengue_nice_onehot/')

        holding['dengue'].append(result)
        print('round {}, finished dengue'.format(i))
        # predict in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = cnn_model.predict([input1, input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(label, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50

        if result1 >= 0.8 and result2 >= 4 and result3 >= 8:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_neoantigen_nice_onehot/')

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
        prediction = cnn_model.predict([input1, input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        if result1 >= 0.76 and result2 >= 0.875:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_covid_nice_onehot/')
        holding['covid'].append((result1, result2, result3, result4))
        print('round {}, finished covid'.format(i))
        i += 1




    # aaindex + pseudo34
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    hla = pd.read_csv('immuno2/data/pseudo34_clean.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    ori['immunogenicity'], ori['potential'] = ori['potential'], ori['immunogenicity']

    dataset = construct_aaindex(ori, hla_dic, after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(input1.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
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
        train_input1, train_input2, train_label = input1[fold[0]], input2[fold[0]], label[fold[0]]
        test_input1, test_input2, test_label = input1[fold[1]], input2[fold[1]], label[fold[1]]
        print('round {}, split finished'.format(i))
        # train
        cnn_model = wrapper_train()
        # predict in validation set
        result = cnn_model.predict([test_input1, test_input2])
        from sklearn.metrics import mean_squared_error

        loss = mean_squared_error(test_label, result, squared=False)
        holding['validation'].append(loss)
        print('round {}, finished validation'.format(i))
        # predict in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = cnn_model.predict([input1, input2])
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(label, hard)

        if result > 0.86:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_dengue_nice_psudo34/')

        holding['dengue'].append(result)
        print('round {}, finished dengue'.format(i))
        # predict in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label_aaindex(dataset)
        prediction = cnn_model.predict([input1, input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(label, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50

        if result1 >= 0.8 and result2 >= 4 and result3 >= 8:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_neoantigen_nice_pseudo34/')

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
        prediction = cnn_model.predict([input1, input2])
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent precision
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed precision
        if result1 >= 0.76 and result2 >= 0.875:
            cnn_model.save_weights('./immuno2/models/cnn/cnn_covid_nice_pseudo34/')
        holding['covid'].append((result1, result2, result3, result4))
        print('round {}, finished covid'.format(i))
        i += 1





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
    draw_history(history)
    valid = ori.loc[valid_index]
    valid['cnn_regress'] = cnn_model.predict([input1_valid,input2_valid])
    valid = valid.sort_values(by='cnn_regress',ascending=False).set_index(pd.Index(np.arange(valid.shape[0])))
    # import scipy.stats as sc
    # sc.pearsonr(valid['potential'].values,valid['cnn_regress'].values)
    # sc.spearmanr(valid['potential'].values,valid['cnn_regress'].values)
    # fig,ax = plt.subplots()
    # ax.scatter(valid['potential'].values,valid['cnn_regress'].values)
    y_true = [1 if not item == 'Negative' else 0 for item in valid['immunogenicity']]
    y_pred = valid['cnn_regress']
    draw_ROC(y_true,y_pred)
    draw_PR(y_true,y_pred)


    # draw 10-fold cross-validation result
    #bucket = []
    fpr,tpr,_ = roc_curve(y_true,y_pred)
    area = auc(fpr,tpr)
    bucket.append((fpr,tpr,_,area))
    import pickle
    with open('bucket_ROC.p','wb') as f:
        pickle.dump(bucket,f)
    with open('bucket_ROC.p','rb') as f:
        bucket = pickle.load(f)
    fig,ax = plt.subplots()
    for i in range(10):
        ax.plot(bucket[i][0],bucket[i][1],lw=0.5,label='CV(Fold={0}), AUC={1:.2f}'.format(i+1,bucket[i][3]))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right",fontsize=9)
    plt.show()
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure2/ROC_10fold.pdf')

    # same for PR curve
    #bucket = []
    precision,recall,_ = precision_recall_curve(y_true,y_pred)
    area = auc(recall,precision)
    bucket.append((precision,recall,_,area))
    with open('bucket_PR.p','wb') as f:
        pickle.dump(bucket,f)
    with open('bucket_PR.p','rb') as f:
        bucket = pickle.load(f)
    fig,ax = plt.subplots()
    for i in range(10):
        ax.plot(bucket[i][1],bucket[i][0],lw=0.5,label='CV(Fold={0}),AUC={1:.2f}'.format(i+1,bucket[i][3]))
    #baseline = np.sum(np.array(y_true) == 1) / len(y_true)  # 0.4735
    baseline = 0.4735
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    #ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR curve example')
    ax.legend(loc="lower left",fontsize=8)
    plt.show()
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure2/PR_10fold.pdf')


    # testing-virus dataset
    ori_test =pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
    testing_dataset = construct_aaindex(ori_test,hla_dic,after_pca)
    input1_test = pull_peptide_aaindex(testing_dataset)
    input2_test = pull_hla_aaindex(testing_dataset)
    label_test = pull_label_aaindex(testing_dataset)

    result = cnn_model.predict(x=[input1_test,input2_test])
    ori_test['result'] = result[:,0]
    ori_test = ori_test.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(ori_test.shape[0])))

    ori_test.to_csv('/Users/ligk2e/Desktop/cnn_regress_virus.csv',index=None)



    # testing-cell dataset
    ori_test =pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt',sep='\t')
    testing_dataset = construct_aaindex(ori_test,hla_dic,after_pca)
    input1_test = pull_peptide_aaindex(testing_dataset)
    input2_test = pull_hla_aaindex(testing_dataset)
    label_test = pull_label_aaindex(testing_dataset)

    result = cnn_model.predict(x=[input1_test,input2_test])
    ori_test['result'] = result[:,0]
    ori_test = ori_test.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(ori_test.shape[0])))

    ori_test.to_csv('/Users/ligk2e/Desktop/cnn_regress_cell.csv',index=None)

    # check GAN generated ones
    df_all = pd.read_csv('/Users/ligk2e/Desktop/df_all.csv')
    testing_dataset = construct_aaindex(df_all,hla_dic,after_pca)
    input1_test = pull_peptide_aaindex(testing_dataset)
    input2_test = pull_hla_aaindex(testing_dataset)
    label_test = pull_label_aaindex(testing_dataset)
    result = cnn_model.predict(x=[input1_test,input2_test])
    df_all['result'] = result[:,0]
    df_all = df_all.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(df_all.shape[0]))) # 659/1024

    # check GAN noise
    df_all = pd.read_csv('/Users/ligk2e/Desktop/df_noise.csv')
    testing_dataset = construct_aaindex(df_all, hla_dic, after_pca)
    input1_test = pull_peptide_aaindex(testing_dataset)
    input2_test = pull_hla_aaindex(testing_dataset)
    label_test = pull_label_aaindex(testing_dataset)
    result = cnn_model.predict(x=[input1_test, input2_test])
    df_all['result'] = result[:, 0]
    df_all = df_all.sort_values(by='result', ascending=False).set_index(
        pd.Index(np.arange(df_all.shape[0])))  # 414/1024

    # check GAN time-series data
    df_all = pd.read_csv('/Users/ligk2e/Desktop/df_all_epoch100.csv')
    testing_dataset = construct_aaindex(df_all,hla_dic,after_pca)
    input1_test = pull_peptide_aaindex(testing_dataset)
    input2_test = pull_hla_aaindex(testing_dataset)
    label_test = pull_label_aaindex(testing_dataset)
    result = cnn_model.predict(x=[input1_test,input2_test])
    df_all['result'] = result[:,0]
    df_all = df_all.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(df_all.shape[0]))) # 659/1024

    '''
    epoch0: 414, 0.4
    epoch20: 515, 0.50
    epoch40: 622, 0.61
    epoch60: 650, 0.63
    epoch80: 659, 0.64
    epoch100: 679, 0.66
    '''

    fig,ax = plt.subplots()
    ax.bar(np.arange(6),[414,515,622,650,659,679],color='orange',width=0.4)
    ax.set_ylim([0,800])
    ax.plot(np.arange(6),[414,515,622,650,659,679],marker='o',linestyle='-',color='k')
    y = [414,515,622,650,659,679]
    for i in range(6):
        ax.text(i-0.1,y[i]+15,s=y[i])
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(['noise','epoch20','epoch40','epoch60','epoch80','epoch100'])
    ax.set_ylabel('Amount of immunogenic peptides')
    ax.grid(True,alpha=0.3)
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure4/analyzer.pdf')


    # save some model
    cnn_model.save_weights('immuno2/models/cnn_model_305_4_8/')
    cnn_model.save_weights('immuno2/models/cnn_model_331_3_7/')

    # load the model
    cnn_model.load_weights('immuno2/models/cnn_model_331_3_7/')

    # completely focues on interpretability
    print(cnn_model.summary())
    '''
    __________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 46, 12, 1)]  0                                            
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 10, 12, 1)]  0                                            
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 1, 16)    2896        input_2[0][0]                    
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 9, 1, 16)     400         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 1, 16)    64          conv2d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 9, 1, 16)     64          conv2d[0][0]                     
__________________________________________________________________________________________________
tf_op_layer_Relu_2 (TensorFlowO [(None, 32, 1, 16)]  0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
tf_op_layer_Relu (TensorFlowOpL [(None, 9, 1, 16)]   0           batch_normalization[0][0]        
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 16, 1, 16)    0           tf_op_layer_Relu_2[0][0]         
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 8, 1, 32)     1056        tf_op_layer_Relu[0][0]           
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 8, 1, 32)     4640        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 8, 1, 32)     128         conv2d_1[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 8, 1, 32)     128         conv2d_3[0][0]                   
__________________________________________________________________________________________________
tf_op_layer_Relu_1 (TensorFlowO [(None, 8, 1, 32)]   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
tf_op_layer_Relu_3 (TensorFlowO [(None, 8, 1, 32)]   0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 4, 1, 32)     0           tf_op_layer_Relu_1[0][0]         
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 4, 1, 32)     0           tf_op_layer_Relu_3[0][0]         
__________________________________________________________________________________________________
flatten (Flatten)               (None, 128)          0           max_pooling2d[0][0]              
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 128)          0           max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 256)          0           flatten[0][0]                    
                                                                 flatten_1[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 128)          32896       concatenate[0][0]                
__________________________________________________________________________________________________
dropout (Dropout)               (None, 128)          0           dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            129         dropout[0][0]                    
==================================================================================================
Total params: 42,401
Trainable params: 42,209
Non-trainable params: 192
__________________________________________________________________________________________________
None   
    '''

    # view kernels/filters
    a = cnn_model.layers[3].get_weights()  # peptide first conv2d
    a[1].shape    # [2,12,1,16]
    for i in range(16):
        fig,ax = plt.subplots()
        ax.imshow(np.transpose(a[0][:,:,0,i]))
        ax.set_title('filter {}'.format(i))

    # view feature maps
    part_model = keras.Model(inputs=cnn_model.inputs[0],outputs=cnn_model.layers[3].output)
    print(part_model.summary())
    case = np.expand_dims(input1_test[0],axis=0)  # [1,10,12,1]
    intermediate = part_model.predict(case)  # [1,9,1,16]
    fig,ax = plt.subplots()
    ax.imshow(np.squeeze(intermediate[0],axis=1).T)
    ax.set_xticks(np.arange(9))
    ax.set_xticklabels(['p1','p2','p3','p4','p5','p6','p7','p8','p9'])
    ax.set_yticks(np.arange(16))
    ax.set_yticklabels(['filter{}'.format(i+1) for i in range(16)])
    mat = np.squeeze(intermediate[0],axis=1).T
    np.argsort(np.mean(mat,axis=0)) + 1

    ## do a hierarchical clustering on feature map
    df = pd.DataFrame(data=mat,index=['filter{}'.format(i+1) for i in range(16)],columns=['p1','p2','p3','p4','p5','p6','p7','p8','p9'])
    import seaborn as sns
    sns.clustermap(df,cmap='viridis')


    # occusion sensitivity
    ## we have input1,input2 variables, only use those positive instances
    new_dataset = []
    for item in dataset:
        if item[2] != 'Negative':
            new_dataset.append(item)
    input1 = pull_peptide_aaindex(new_dataset)
    input2 = pull_hla_aaindex(new_dataset)
    # original, no occlusion
    pred = cnn_model([input1,input2]).numpy().mean()  # pred_positive: 0.616 pred_negative: 0.39
    # occlude different position
    input1[:,9,:,:] = 0
    decrease = 0.616 - cnn_model([input1,input2]).numpy().mean()
    '''
    1 2 3 4 5 _ 6 7 8 9
    1 2 3 4 5 6 7 8 9 10
    
    in 9mer, P4-P6 are important, means in my schema, P4,P5,P7 are important, which is exactly the case
    less so significant in P7, which is P8 in this case.
    also signicant in P8, which is P9 in this case
    
    Then P2 and P9 in 9mer, which is P2 and P10 in this case
    
    occlude1: 0.005
    occlude2: 0.012
    occlude3: 0.002
    occlude4: 0.034   (yes)
    occlude5: 0.016   (yes)
    occlude6: 0.006
    occlude7: 0.013   (yes)
    occlude8: 0.001    (no)
    occlude9: 0.010    (yes)
    occlude10: 0.007   
    '''
    fig,ax = plt.subplots()
    hm = ax.imshow(np.array([[0.005,0.012,0.002,0.034,0.016,0.006,0.013,0.001,0.010,0.007]]))
    ax.yaxis.set_tick_params(length=0,labelleft=False)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
    fig.colorbar(hm)

    # how about retain all value, not just mean
    new_dataset = []
    for item in dataset:
        if item[2] != 'Negative':
            new_dataset.append(item)
    input1 = pull_peptide_aaindex(new_dataset)
    input2 = pull_hla_aaindex(new_dataset)
    # original, no occlusion
    pred = cnn_model([input1,input2]).numpy()
    # occlude different position
    input1[:,(3,4),:,:] = 0
    decrease_big = pred - cnn_model([input1,input2]).numpy()
    input1[:,(0,2),:,:] = 0
    decrease_small = pred - cnn_model([input1, input2]).numpy()
    from scipy.stats import mannwhitneyu
    mannwhitneyu(decrease_big,decrease_small,alternative='greater')
    fig,ax = plt.subplots()
    bp = ax.boxplot([decrease_big[:,0],decrease_small[:,0]],patch_artist=True)
    for flier in bp['fliers']:
        flier.set(markersize=1.5)
    for box in bp['boxes']:
        box.set(facecolor='#087E8B',alpha=0.6,linewidth=1)
    for median in bp['medians']:
        median.set(color='black',linewidth=1)
    ax.set_xticklabels(['Occlude P4&P5', 'Occlude P3&P1'])
    ax.set_ylabel('Performance Drop')
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure3/U_test.pdf')


    # let's boostrap 100 times, see if the rank holds
    n = 100
    position_rank = np.empty([n, 9])
    for m in range(n):
        ind = np.random.choice(np.arange(len(new_dataset)),2000)  # bootstrap 2000 positive samples
        sample = [new_dataset[i] for i in ind]  # get samples
        input1 = pull_peptide_aaindex(sample)
        input2 = pull_hla_aaindex(sample)
        pred_ori = cnn_model([input1,input2]).numpy().mean()  # original prediction
        array = []  # store all the importance, based on the decrease when eliminating each position
        for i in range(10):
            if i != 5:
                input1[:,i,:,:] = 0
                importance = pred_ori - cnn_model([input1,input2]).numpy().mean()
                array.append(importance)
                input1 = pull_peptide_aaindex(sample)
                input2 = pull_hla_aaindex(sample)
        ranking = np.argsort(array) + 1   # ascending order
        tmp = []
        for i in range(9):
            tmp.append(list(ranking).index(i+1))
        position_rank[m,:] = tmp



    max_value = np.max(position_rank,axis=0) + 1
    min_value = np.min(position_rank,axis=0) + 1
    median_value = np.median(position_rank,axis=0) + 1

    # draw the scatter plot of the rank, using size parameter
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap('tab10')
    delim = np.linspace(0,1,9)
    colors = [mpl.colors.rgb2hex(cmap(i)[:3]) for i in delim]
    from collections import Counter
    fig,ax = plt.subplots()
    for i in np.arange(9):
        y = list(Counter(position_rank[:,i]+1).keys())
        s = list(Counter(position_rank[:,i]+1).values())
        ax.scatter([i for n in range(len(y))],y, s=[m*4 for m in s],c=colors[i])
    ax.set_ylim(0.5,9.5)
    ax.set_yticks(np.arange(9)+1)
    ax.set_xticks(np.arange(9))
    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9'])
    ax.set_xlabel('Position')
    ax.set_ylabel('Ranking(ascending)')
    ax.grid(True,alpha=0.2)
    h1 = [ax.plot([],[],color='grey',marker='o',markersize=i,ls='')[0] for i in range(8,15,2)]
    leg1 = ax.legend(handles=h1,labels=[10,40,70,100],title='Frequency',loc='lower left',bbox_to_anchor=(1,0.6),frameon=False)
    h2 = [ax.plot([],[],color=i,marker='o',markersize=5,ls='')[0] for i in colors]
    leg2 = ax.legend(handles=h2,labels=['p1','p2','p3','p4','p5','p6','p7','p8','p9'],title='Position',loc='lower left',bbox_to_anchor=(1,0),frameon=False)
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure3/ranking.pdf',bbox_inches='tight')

    with open('position_rank.p','wb') as f:
        pickle.dump(position_rank,f)


    fig,ax = plt.subplots()
    ax.plot(np.arange(9),median_value,marker='o',color='k',linestyle='-')
    ax.fill_between(np.arange(9),min_value,max_value,alpha=0.5,color='orange')
    ax.set_ylim(0.5,9.5)
    ax.set_yticks(np.arange(9)+1)
    ax.set_xticks(np.arange(9))
    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9'])
    ax.set_xlabel('Position')
    ax.set_ylabel('Ranking(ascending)')
    ax.grid(True,alpha=0.2)












    # CAM and Grad-cam will require a convolusion layer, CAM will addtionally require this followed by dense layer

    # saliency map, guided backpropogation
    # reflect how output probability will change with respect to the small change of certain pixel
    input1 = tf.constant(input1)
    input2 = tf.constant(input2)
    with tf.GradientTape() as tape:
        tape.watch([input1,input2])
        prediction = cnn_model([input1,input2])
        gradient = tape.gradient(prediction,input1)

    a = np.squeeze(np.mean(gradient.numpy(),axis=0),axis=2)
    fig,ax = plt.subplots()
    ax.imshow(a.T)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12'])
    np.argsort(np.mean(a,axis=1))+1

    # saliency map get different answer, let's occlude (8,5) and (7,4), not significant

    # gradient ascent
    # maximize a certain feature map

    ### finally, let's generate motif, starts with HLA-A0201
    data = pd.read_csv('../pytorch/wassGAN/gan_a0201.csv')
    peptide_mat = np.empty((data.shape[0],10),dtype=object)
    for i in range(data.shape[0]):
        pep = data['peptide'].iloc[i]
        if len(pep) == 9:
            pep = list(pep[:5] + '-' + pep[5:])
        pep = list(pep)
        peptide_mat[i,:] = pep
    frequency_mat = np.empty([21,10])
    from collections import Counter
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    for j in range(peptide_mat.shape[1]):
        dic = Counter(peptide_mat[:,j])
        col = [dic[a] for a in amino]
        frequency_mat[:,j] = col
    frequency_mat_normalized = frequency_mat / np.sum(frequency_mat,axis=0)
    from scipy.special import softmax
    scale = np.array([0.005,0.012,0.002,0.034,0.016,0.006,0.013,0.001,0.010,0.007]) * 100
    frequencyc_mat_normalized_scale = frequency_mat_normalized * scale
    fig,ax = plt.subplots()
    hm = ax.imshow(frequencyc_mat_normalized_scale[:20,:],cmap='hot')
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10'])
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(list(amino[:20]))
    fig.colorbar(hm)
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/supplementary3/hla0201_motif.pdf')

    data = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/gan_2402.csv')
    peptide_mat = np.empty((data.shape[0],10),dtype=object)
    for i in range(data.shape[0]):
        pep = data['peptide'].iloc[i]
        if len(pep) == 9:
            pep = list(pep[:5] + '-' + pep[5:])
        pep = list(pep)
        peptide_mat[i,:] = pep
    frequency_mat = np.empty([21,10])
    from collections import Counter
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    for j in range(peptide_mat.shape[1]):
        dic = Counter(peptide_mat[:,j])
        col = [dic[a] for a in amino]
        frequency_mat[:,j] = col
    frequency_mat_normalized = frequency_mat / np.sum(frequency_mat,axis=0)
    from scipy.special import softmax
    scale = np.array([0.005,0.012,0.002,0.034,0.016,0.006,0.013,0.001,0.010,0.007]) * 100
    frequencyc_mat_normalized_scale = frequency_mat_normalized * scale
    fig,ax = plt.subplots()
    hm = ax.imshow(frequencyc_mat_normalized_scale[:20,:],cmap='hot')
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10'])
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(list(amino[:20]))
    fig.colorbar(hm)
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/supplementary3/hla2402_motif.pdf')

    data = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/gan_b0801.csv')
    peptide_mat = np.empty((data.shape[0],10),dtype=object)
    for i in range(data.shape[0]):
        pep = data['peptide'].iloc[i]
        if len(pep) == 9:
            pep = list(pep[:5] + '-' + pep[5:])
        pep = list(pep)
        peptide_mat[i,:] = pep
    frequency_mat = np.empty([21,10])
    from collections import Counter
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    for j in range(peptide_mat.shape[1]):
        dic = Counter(peptide_mat[:,j])
        col = [dic[a] for a in amino]
        frequency_mat[:,j] = col
    frequency_mat_normalized = frequency_mat / np.sum(frequency_mat,axis=0)
    from scipy.special import softmax
    scale = np.array([0.005,0.012,0.002,0.034,0.016,0.006,0.013,0.001,0.010,0.007]) * 100
    frequencyc_mat_normalized_scale = frequency_mat_normalized * scale
    fig,ax = plt.subplots()
    hm = ax.imshow(frequencyc_mat_normalized_scale[:20,:],cmap='hot')
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10'])
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(list(amino[:20]))
    fig.colorbar(hm)
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/supplementary3/hlab0801_motif.pdf')

    data = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/gan_b0702.csv')
    peptide_mat = np.empty((data.shape[0], 10), dtype=object)
    for i in range(data.shape[0]):
        pep = data['peptide'].iloc[i]
        if len(pep) == 9:
            pep = list(pep[:5] + '-' + pep[5:])
        pep = list(pep)
        peptide_mat[i, :] = pep
    frequency_mat = np.empty([21, 10])
    from collections import Counter

    amino = 'ARNDCQEGHILKMFPSTWYV-'
    for j in range(peptide_mat.shape[1]):
        dic = Counter(peptide_mat[:, j])
        col = [dic[a] for a in amino]
        frequency_mat[:, j] = col
    frequency_mat_normalized = frequency_mat / np.sum(frequency_mat, axis=0)
    from scipy.special import softmax

    scale = np.array([0.005, 0.012, 0.002, 0.034, 0.016, 0.006, 0.013, 0.001, 0.010, 0.007]) * 100
    frequencyc_mat_normalized_scale = frequency_mat_normalized * scale
    fig, ax = plt.subplots()
    hm = ax.imshow(frequencyc_mat_normalized_scale[:20, :], cmap='hot')
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(list(amino[:20]))
    fig.colorbar(hm)
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/supplementary3/hlab0702_motif.pdf')





















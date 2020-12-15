'''
This is a novel endeavor,
will frame this problem as a regression problem using beta-binomial bayesian approach
to predict the potential of a p-MHC complex to trigger T cell response
if worked, will attempt to integrate VDJ information to refine the results

to this end, might try all traditional regression algorithm along with ResNet with MSE loss function
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers,callbacks

def assign_prior(label):
    if label == 'Positive-High':
        return (32,1)   # beta(15,1)   mean=0.93, std = 0.05
    elif label == 'Positive' or label == 'Positive-Intermediate':
        return (30,1)   # beta(10,1)   mean = 0.90  std = 0.08
    elif label == 'Positive-Low':
        return (28,1)    # beta(5,1)   mean = 0.83, std = 0.14
    elif label == 'Negative':
        return (3,3)    # beta(3,4)   mean = 0.42, std = 0.17

def assign_posterior(data):
    data['test'] = [0 if item == -1 else int(item) for item in data['test']]
    data['respond'] = [0 if item == -1 else int(item) for item in data['respond']]
    posterior = []
    for i in range(data.shape[0]):
        print(i)
        label = data['immunogenicity'].iloc[i]
        test = data['test'].iloc[i]
        respond = data['respond'].iloc[i]
        success = int(respond)
        failure = int(test) - int(respond)
        (prior_alpha,prior_beta) = assign_prior(label)
        posterior_alpha, posterior_beta = prior_alpha + success, prior_beta + failure
        estimate = np.random.beta(posterior_alpha,posterior_beta,100).mean()
        posterior.append(estimate)
    data['potential'] = posterior
    return data

def data_remove_dengue(data):
    dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/dengue_test.csv')['peptide'].values
    cond = []
    for i in range(data.shape[0]):
        peptide = data['peptide'].iloc[i]
        if peptide in dengue:
            cond.append(False)
        else:
            cond.append(True)
    data_new = data.loc[cond]
    data_new = data_new.set_index(pd.Index(np.arange(data_new.shape[0])))
    return data_new

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
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
        immuno = np.array(ori['potential'].iloc[i]).reshape(1,-1)   # [1,1]

        encode_pep = peptide_data_aaindex(peptide,after_pca)    # [10,12]

        encode_hla = hla_data_aaindex(hla_dic,hla_type,after_pca)   # [46,12]
        series.append((encode_pep, encode_hla, immuno))
    return series

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

# new model
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
        #print(out.shape)
        out = self.block1(out)
        #print(out.shape)
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

def draw_history(history):
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    # plot accuracy during training
    # plt.subplot(212)
    # plt.title('Accuracy')
    # plt.plot(history.history['accuracy'], label='train')
    # plt.plot(history.history['val_accuracy'], label='validation')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    # tweak the prior probability assignment
    data = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    data = data_remove_dengue(data)
    data_regression = assign_posterior(data)
    data_regression.to_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv',
                           index=None)

    data = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/regression_data/all_assay_data.csv')
    data = data_remove_dengue(data)
    data_regression = assign_posterior(data)
    data_regression.to_csv('/Users/ligk2e/Desktop/immuno3/data/regression_data/all_assay_data_with_potential.csv',index=None)

    data_regression = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/regression_data/all_assay_data_with_potential.csv')
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    hla = pd.read_csv('immuno2/data/hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    dataset = construct_aaindex(data_regression,hla_dic,after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)

    resnet_aaindex = model_aaindex()
    resnet_aaindex.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(lr=0.0001),)

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
    history = resnet_aaindex.fit(
        x=[input1_train,input2_train],   # feed a list into
        y=label_train,
        validation_data = ([input1_valid,input2_valid],label_valid),
        batch_size=128,
        epochs=200,
        callbacks = [callback_val,callback_train])

    # validation
    a = resnet_aaindex.predict([input1_valid,input2_valid])
    valid = data_regression.iloc[valid_index]
    valid['result'] = a[:,0]
    frame_sort = valid.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(valid.shape[0])))
    frame_sort.to_csv('/Users/ligk2e/Desktop/valid_regression.csv',index=None)
    draw_ROC(frame['label'],frame['result'])
    draw_PR(frame['label'],frame['result'])
    fig,ax = plt.subplots()
    ax.bar(frame_sort.index.values,frame_sort['result'].values)

    # testing
    test = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/regression_data/dengue_test.csv')
    test_regression = assign_posterior(test)


    dataset = construct_aaindex(test_regression,hla_dic,after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)

    result = resnet_aaindex.predict([input1,input2])
    test_regression['result'] = result[:,0]
    test_regression_sort = test_regression.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(test_regression.shape[0])))
    test_regression_sort.to_csv('/Users/ligk2e/Desktop/dengue_test_regression.csv',index=None)

    # testing cell paper
    cell = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/regression_data/complete_data_filter910.txt',sep='\t')
    dataset = construct_aaindex(cell,hla_dic,after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)
    result = resnet_aaindex.predict([input1,input2])
    cell['result'] = result[:,0]
    cell_sort = cell.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(cell.shape[0])))










